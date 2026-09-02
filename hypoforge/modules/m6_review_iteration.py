"""
M6: Review & Iterative Refinement.

Three specialist reviewer agents (scientific_logic / evidence_consistency /
method_feasibility) score the top hypothesis + research plan on a 1–5 scale;
each reasons *before* it scores (rubric-anchored).  M6 additionally persists a
calibrated eight-dimension score with explicit caps; this display score is
intentionally independent from iterative routing.

Output: ``reviews`` appended; ``iteration_count`` incremented.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..observability import emit_event
from ..context import ContextPlanner, ContextRequest, emit_context_built
from ..evidence_audit import EvidenceAuditService, PremiseAuditResult
from ..evaluation.m6_scoring import (
    M6_SCORE_WEIGHTS,
    M6ScoreConditions,
    PlanQualityAssessment,
    SemanticScoreAssessment,
    aggregate_m6_scoring,
)
from ..graph_context import build_graph_context
from ..protocol import ModuleProtocol
from ..prompts.m6_prompts import (
    M6_EVIDENCE_VERDICT_SYSTEM,
    M6_EVIDENCE_VERDICT_TEMPLATE,
    M6_FORMAT_NOTE,
    M6_NOVELTY_REVIEW_SYSTEM,
    M6_PLAN_QUALITY_SYSTEM,
    M6_REASON_FIRST,
    M6_REVIEWER_PROMPTS,
    M6_USER_TEMPLATE,
)
from ..evaluation.rubric import review_rubric_line
from ..evaluation.scorer import _quality_gates
from ..registry import ModuleRegistry
from ..state import (
    EvidenceGap,
    EvidenceSufficiencyVerdict,
    ExperimentalValidationVerdict,
    FactualPremiseAudit,
    GraphCorrectionRequest,
    HypothesisPremise,
    PipelineState,
    ReviewResult,
    ResearchPlan,
    ReviewerDimension,
    ScoreDimensionDetail,
    make_gap_id,
)
from ..synthesis_contract import (
    synthesis_contract_for_state,
    synthesis_problem_payload,
)
from ..task_alignment import assess_task_alignment
from ..tools.qwen_client import QwenClient

logger = logging.getLogger(__name__)


def _is_invalid_fallback_hypothesis(hypothesis: Any) -> bool:
    """Identify M4's explicit last-resort output without judging its content."""

    hypothesis_id = str(getattr(hypothesis, "hypothesis_id", "") or "").strip()
    statement = str(getattr(hypothesis, "statement", "") or "").lower()
    mechanism = str(getattr(hypothesis, "mechanism", "") or "").lower()
    rationale = str(getattr(hypothesis, "ranking_rationale", "") or "").lower()
    return (
        hypothesis_id.upper().startswith("FH")
        or "{q}" in statement
        or "fast-mode fallback" in mechanism
        or "deterministic fallback hypothesis" in rationale
    )


def _has_missing_core_validation_target(
    verdict: Optional[ExperimentalValidationVerdict],
) -> bool:
    """Only a completely missing core M4 target warrants the severe cap."""

    if verdict is None:
        return False
    core_kinds = {"statement", "mechanism", "prediction", "falsification"}
    return any(
        item.target_kind in core_kinds and item.verdict == "missing"
        for item in verdict.items
    )


_CORE_OVERALL_DIMENSIONS = {
    "scientific_logic",
    "objective_evidence_consistency",
    "method_feasibility",
    "experimental_validation_coverage",
}


@dataclass(frozen=True)
class _AuditableClaim:
    """One factual claim that is allowed to create an M6 search gap."""

    source_claim_type: str
    source_claim_id: str
    claim: str
    required: bool = True
    supporting_evidence_ids: tuple[str, ...] = field(default_factory=tuple)


def _aggregate_overall_reviews(
    reviews: List[ReviewResult],
) -> tuple[float, List[str]]:
    """Aggregate scientific quality without averaging in bookkeeping gates."""

    core_scores = [
        review.score for review in reviews
        if review.dimension.value in _CORE_OVERALL_DIMENSIONS
    ]
    score = sum(core_scores) / len(core_scores) if core_scores else 3.0
    failed_gates = [
        review.dimension.value for review in reviews
        if review.hard_gate_passed is False
    ]
    evidence_review = next(
        (
            review for review in reviews
            if review.dimension.value == "objective_evidence_consistency"
        ),
        None,
    )
    if "task_alignment" in failed_gates:
        score = min(score, 1.9)
    elif "objective_evidence_consistency" in failed_gates:
        score = min(score, 2.9)
    elif failed_gates:
        score = min(score, 2.9)
    elif evidence_review is not None and evidence_review.score < 4.0:
        score = min(score, 3.9)
    return round(score, 1), failed_gates


@ModuleRegistry.register
class M6ReviewIteration(ModuleProtocol):
    module_name = "m6"
    module_version = "0.1.0"
    description = "Multi-reviewer assessment + iterative refinement decision"

    @staticmethod
    def _normalise_graph_corrections(
        review: ReviewResult,
        *,
        valid_evidence_ids: set[str],
        version: int,
    ) -> List[GraphCorrectionRequest]:
        if review.dimension.value != "objective_evidence_consistency":
            return []
        output: List[GraphCorrectionRequest] = []
        for request in review.graph_correction_requests:
            evidence_ids = list(dict.fromkeys(
                evidence_id for evidence_id in request.evidence_ids
                if evidence_id in valid_evidence_ids
            ))
            request_id = request.request_id.strip()
            if not request_id:
                fingerprint = "|".join([
                    request.operation,
                    request.source_node_id,
                    request.target_node_id,
                    str(request.current_relation or ""),
                    str(request.proposed_relation or ""),
                    str(version),
                ])
                request_id = "GCR_" + hashlib.sha256(
                    fingerprint.encode("utf-8")
                ).hexdigest()[:12]
            output.append(request.model_copy(update={
                "request_id": request_id,
                "evidence_ids": evidence_ids,
                "requested_by": review.dimension.value,
                "iteration": version,
                "status": "pending",
                "rejection_reason": "",
            }))
        return output

    @staticmethod
    def _objective_evidence_review(
        verdict: EvidenceSufficiencyVerdict,
        *,
        version: int,
    ) -> ReviewResult:
        """Derive the evidence score from the persisted factual-premise audit.

        This is the sole score source for objective evidence consistency when
        the structured M6 evidence audit is enabled.  Novel mechanisms,
        predictions and working assumptions are intentionally absent from the
        input contract and therefore cannot lower this dimension.
        """

        audits = list(verdict.premise_audits)
        audit_verdicts = {item.verdict for item in audits}
        if "contradicted" in audit_verdicts:
            score = 1.0
            passed = False
        elif audit_verdicts & {"unsupported", "invalid_citation"}:
            score = 2.0
            passed = False
        elif "partially_supported" in audit_verdicts:
            score = 3.5
            passed = True
        else:
            score = 5.0
            passed = True

        evidence_ids = list(dict.fromkeys(
            evidence_id
            for audit in audits
            for evidence_id in audit.evidence_ids
            if str(evidence_id).strip()
        ))
        reasoning = "; ".join(
            f"{audit.premise_id}: {audit.verdict} — {audit.rationale}"
            for audit in audits
        ) or "No required factual premise needs evidence auditing."
        suggestions = ""
        if not passed:
            suggestions = (
                "Revise, remove, or supplement the unsupported factual "
                "premises identified by the premise audit."
            )
        elif score < 5.0:
            suggestions = (
                "Keep partially supported premises qualified and preserve "
                "their canonical evidence IDs."
            )
        return ReviewResult(
            dimension=ReviewerDimension("objective_evidence_consistency"),
            attribution="hypothesis",
            reasoning=reasoning,
            score=score,
            comments=(
                "Derived exclusively from the structured factual-premise "
                "audit; conjectural mechanism text is out of scope."
            ),
            suggestions=suggestions,
            evidence_ids=evidence_ids,
            hard_gate_passed=passed,
            version=version,
        )

    @staticmethod
    def _reconcile_graph_corrections(
        verdict: EvidenceSufficiencyVerdict,
        corrections: List[GraphCorrectionRequest],
    ) -> List[GraphCorrectionRequest]:
        """Drop stale raw-review corrections after a sufficient fact audit."""

        return [] if verdict.sufficient else list(corrections)

    # ------------------------------------------------------------------
    # Configurable
    # ------------------------------------------------------------------

    def __init__(
        self,
        reviewers: Optional[List[str]] = None,
        mode: str = "llm",
        llm_config: Optional[Any] = None,
        m6_evidence_revisit: bool = False,
        gap_no_gain_limit: int = 3,
        semantic_alignment_timeout_seconds: float = 60.0,
        reviewer_timeout_seconds: float = 180.0,
        fast_mode: bool = False,
        **kwargs,
    ):
        self.fast_mode = bool(fast_mode)
        if semantic_alignment_timeout_seconds <= 0:
            raise ValueError(
                "semantic_alignment_timeout_seconds must be positive"
            )
        if reviewer_timeout_seconds <= 0:
            raise ValueError("reviewer_timeout_seconds must be positive")
        self.reviewer_dims = reviewers or [
            "scientific_logic",
            "objective_evidence_consistency",
            "method_feasibility",
            "overall",
        ]
        self.mode = mode
        self.llm_config = llm_config
        # Iteration-core switch: when on, one extra structured LLM call judges
        # evidence sufficiency after the reviewer round.  Off ⇒ zero extra calls.
        self.m6_evidence_revisit = m6_evidence_revisit
        # Convergence protection: after this many CONSECUTIVE rounds where
        # every gap gained zero new effective evidence (metrics["m3_gap_gain"]
        # all zero), the remaining open/pending gaps are marked
        # ``unimprovable`` so supplement search stops firing (gap-level
        # marker; the routing layer's search_round cap is the backstop).
        # Default mirrors config.gap_no_gain_limit; overridable via
        # module_overrides.m6.kwargs.gap_no_gain_limit.
        self.gap_no_gain_limit = max(1, int(gap_no_gain_limit))
        self.semantic_alignment_timeout_seconds = float(
            semantic_alignment_timeout_seconds
        )
        self.reviewer_timeout_seconds = float(reviewer_timeout_seconds)
        self.client = QwenClient.from_config(llm_config) if llm_config else None
        # Real configured runs receive the detailed semantic audits. Tests and
        # legacy callers that inject a fake client without an LLM config keep
        # the bounded deterministic fallback unless they opt in explicitly.
        self.detailed_scoring = llm_config is not None

    @staticmethod
    def _build_scoring_summary(
        dimensions: list[Any],
        conditions: M6ScoreConditions,
    ):
        """Build the persisted score through the pure calibration contract."""
        return aggregate_m6_scoring(dimensions, conditions)

    @staticmethod
    def _dimension_detail(
        dimension: str,
        score: float,
        *,
        source: str,
        confidence: float,
        strengths: Optional[List[str]] = None,
        weaknesses: Optional[List[str]] = None,
        deductions: Optional[List[str]] = None,
    ) -> ScoreDimensionDetail:
        weight = M6_SCORE_WEIGHTS[dimension]
        return ScoreDimensionDetail(
            dimension=dimension,
            score=max(1.0, min(5.0, round(float(score), 1))),
            weight=weight,
            weighted_contribution=round(float(score) * weight, 4),
            source=source,
            confidence=max(0.0, min(1.0, float(confidence))),
            strengths=list(strengths or []),
            weaknesses=list(weaknesses or []),
            deductions=list(deductions or []),
        )

    @staticmethod
    def _alignment_score(assessment: Any, semantic_passed: bool) -> float:
        """Map the deterministic 0–1 alignment score to the 1–5 rubric."""
        if not semantic_passed or not assessment.passed:
            return 1.0
        return round(1.0 + max(0.0, min(1.0, assessment.score)) * 4.0, 1)

    @staticmethod
    def _reproducibility_detail(plan: ResearchPlan) -> ScoreDimensionDetail:
        """Score explicit reproducibility safeguards, not field presence."""
        text = json.dumps(plan.model_dump(mode="json"), ensure_ascii=False).casefold()
        indicators = {
            "sample_size_basis": any(term in text for term in (
                "power analysis", "效应量", "sample size", "样本量依据",
            )),
            "randomisation": any(term in text for term in (
                "random", "随机", "randomization", "随机化",
            )),
            "blinding": any(term in text for term in (
                "blind", "盲法", "blinded", "双盲",
            )),
            "replicates": any(term in text for term in (
                "replicate", "重复", "biological n", "生物学重复",
            )),
            "batch_control": any(term in text for term in (
                "batch", "批次", "lot", "批间",
            )),
            "primary_endpoint": any(term in text for term in (
                "primary endpoint", "主要终点", "主要指标",
            )),
            "multiplicity": any(term in text for term in (
                "multiple comparison", "多重比较", "预注册", "pre-spec",
            )),
        }
        count = sum(indicators.values())
        score = round(1.0 + 4.0 * count / len(indicators), 1)
        missing = [name for name, present in indicators.items() if not present]
        return M6ReviewIteration._dimension_detail(
            "statistics_reproducibility",
            score,
            source="deterministic",
            confidence=0.8,
            strengths=[name for name, present in indicators.items() if present],
            weaknesses=missing,
            deductions=[f"Missing safeguard: {name}" for name in missing],
        )

    async def _review_novelty(
        self,
        state: PipelineState,
        hypothesis: Any,
        plan: ResearchPlan,
        graph_context: Any,
    ) -> SemanticScoreAssessment:
        if self.fast_mode or not self.detailed_scoring or self.client is None:
            return SemanticScoreAssessment(
                score=3.0,
                confidence=0.0,
                weaknesses=["Detailed novelty audit skipped in fast/legacy mode."],
                deductions=["No semantic novelty audit was executed."],
            )
        try:
            payload = await asyncio.wait_for(
                self.client.structured_chat(
                    system_prompt=M6_NOVELTY_REVIEW_SYSTEM,
                    user_prompt=M6_USER_TEMPLATE.format(
                        original_question=state.input_question,
                        problem_card_json=json.dumps(
                            synthesis_problem_payload(state.problem_card),
                            ensure_ascii=False,
                            indent=2,
                        ),
                        hypothesis_json=json.dumps(
                            hypothesis.model_dump(mode="json"),
                            ensure_ascii=False,
                            indent=2,
                        ),
                        plan_json=json.dumps(
                            plan.model_dump(mode="json"),
                            ensure_ascii=False,
                            indent=2,
                        ),
                        graph_context=graph_context.render(),
                        facts_count=len(state.evidence_graph.established_facts)
                        if state.evidence_graph else 0,
                        conflicts_count=len(state.evidence_graph.conflicts)
                        if state.evidence_graph else 0,
                        gaps_count=len(state.evidence_graph.knowledge_gaps)
                        if state.evidence_graph else 0,
                    ),
                    output_schema=SemanticScoreAssessment.model_json_schema(),
                    max_tokens=4096,
                    temperature=getattr(self.llm_config, "temperature", 0.1),
                    disable_thinking=True,
                ),
                timeout=self.reviewer_timeout_seconds,
            )
            return SemanticScoreAssessment.model_validate(payload)
        except Exception as exc:
            logger.warning("M6 novelty audit degraded: %s", exc)
            return SemanticScoreAssessment(
                score=3.0,
                confidence=0.0,
                weaknesses=["Novelty semantic audit failed."],
                deductions=[f"Novelty audit unavailable: {type(exc).__name__}"],
            )

    async def _review_plan_quality(
        self,
        state: PipelineState,
        hypothesis: Any,
        plan: ResearchPlan,
        graph_context: Any,
    ) -> PlanQualityAssessment:
        if self.fast_mode or not self.detailed_scoring or self.client is None:
            fallback = SemanticScoreAssessment(
                score=3.0,
                confidence=0.0,
                weaknesses=["Detailed plan-quality audit skipped in fast/legacy mode."],
                deductions=["No semantic plan-quality audit was executed."],
            )
            return PlanQualityAssessment(
                task_coverage=fallback,
                evidence_reliability=fallback,
                testability=fallback,
                experimental_rigor=fallback,
                statistics_reproducibility=fallback,
                technical_feasibility=fallback,
            )

        try:
            payload = await asyncio.wait_for(
                self.client.structured_chat(
                    system_prompt=M6_PLAN_QUALITY_SYSTEM,
                    user_prompt=M6_USER_TEMPLATE.format(
                        original_question=state.input_question,
                        problem_card_json=json.dumps(
                            synthesis_problem_payload(state.problem_card),
                            ensure_ascii=False,
                            indent=2,
                        ),
                        hypothesis_json=json.dumps(
                            hypothesis.model_dump(mode="json"),
                            ensure_ascii=False,
                            indent=2,
                        ),
                        plan_json=json.dumps(
                            plan.model_dump(mode="json"),
                            ensure_ascii=False,
                            indent=2,
                        ),
                        graph_context=graph_context.render(),
                        facts_count=len(state.evidence_graph.established_facts)
                        if state.evidence_graph else 0,
                        conflicts_count=len(state.evidence_graph.conflicts)
                        if state.evidence_graph else 0,
                        gaps_count=len(state.evidence_graph.knowledge_gaps)
                        if state.evidence_graph else 0,
                    ),
                    output_schema=PlanQualityAssessment.model_json_schema(),
                    max_tokens=8192,
                    temperature=getattr(self.llm_config, "temperature", 0.1),
                    disable_thinking=True,
                ),
                timeout=self.reviewer_timeout_seconds,
            )
            return PlanQualityAssessment.model_validate(payload)
        except Exception as exc:
            logger.warning("M6 plan-quality audit degraded: %s", exc)
            fallback = SemanticScoreAssessment(
                score=3.0,
                confidence=0.0,
                weaknesses=["Plan-quality semantic audit failed."],
                deductions=[f"Plan-quality audit unavailable: {type(exc).__name__}"],
            )
            return PlanQualityAssessment(
                task_coverage=fallback,
                evidence_reliability=fallback,
                testability=fallback,
                experimental_rigor=fallback,
                statistics_reproducibility=fallback,
                technical_feasibility=fallback,
            )

    async def _build_modern_scoring_dimensions(
        self,
        *,
        state: PipelineState,
        hypothesis: Any,
        plan: ResearchPlan,
        graph_context: Any,
        hypothesis_alignment: Any,
        plan_alignment: Any,
        semantic_alignment_passed: bool,
        deterministic_alignment_passed: bool,
        factual_verdict: Optional[EvidenceSufficiencyVerdict],
        experimental_validation_verdict: Optional[ExperimentalValidationVerdict],
        gates: Dict[str, Any],
        reviews: List[ReviewResult],
    ) -> tuple[list[ScoreDimensionDetail], M6ScoreConditions]:
        """Collect eight modern dimensions and structured cap conditions."""

        def latest_review(*names: str) -> Optional[ReviewResult]:
            for item in reversed(reviews):
                if item.dimension.value in names:
                    return item
            return None

        logic_review = latest_review("scientific_logic")
        logic_score = logic_review.score if logic_review else 3.0
        logic_source = "llm" if logic_review else "degraded"
        logic_confidence = 0.8 if logic_review else 0.0
        logic_row = self._dimension_detail(
            "scientific_logic",
            logic_score,
            source=logic_source,
            confidence=logic_confidence,
            strengths=[logic_review.comments] if logic_review and logic_review.comments else [],
            weaknesses=[logic_review.suggestions] if logic_review and logic_review.suggestions else [],
        )

        # Reuse the existing bounded plan-quality call for the dimensions that
        # require semantic judgement.  No extra LLM request is introduced.
        plan_quality = await self._review_plan_quality(
            state, hypothesis, plan, graph_context
        )

        structural_task_score = min(
            self._alignment_score(hypothesis_alignment, semantic_alignment_passed),
            self._alignment_score(plan_alignment, semantic_alignment_passed),
        )
        task_semantic = plan_quality.task_coverage
        task_score = min(structural_task_score, task_semantic.score)
        task_row = self._dimension_detail(
            "task_coverage",
            task_score,
            source="hybrid",
            confidence=max(
                task_semantic.confidence,
                1.0 if semantic_alignment_passed else 0.4,
            ),
            strengths=task_semantic.strengths,
            weaknesses=(
                list(task_semantic.weaknesses)
                +
                list(getattr(hypothesis_alignment, "missing_anchors", []) or [])
                + list(getattr(plan_alignment, "missing_anchors", []) or [])
            ),
            deductions=task_semantic.deductions,
        )

        novelty_semantic = await self._review_novelty(
            state, hypothesis, plan, graph_context
        )
        novelty_score = novelty_semantic.score
        novelty_source = "llm" if novelty_semantic.confidence else "degraded"
        novelty_confidence = novelty_semantic.confidence
        novelty_weaknesses = list(novelty_semantic.weaknesses)
        novelty_deductions = list(novelty_semantic.deductions)
        if self.detailed_scoring and not self.fast_mode and state.evidence_graph:
            try:
                from ..evaluation.metrics import NoveltyMetric

                knowledge_entries = [
                    entry
                    for result in state.literature_results
                    for entry in result.knowledge_entries
                ]
                independent_result = await asyncio.wait_for(
                    NoveltyMetric(llm_config=self.llm_config).compute(
                        hypothesis,
                        knowledge_entries,
                        evidence_graph=state.evidence_graph,
                    ),
                    timeout=self.reviewer_timeout_seconds,
                )
                independent_score, trace = (
                    independent_result
                    if isinstance(independent_result, tuple)
                    else (independent_result, {})
                )
                claim_rows = list((trace or {}).get("claims_novelty", []))
                usable = [
                    row for row in claim_rows
                    if row.get("assessment") != "insufficient_graph_coverage"
                ]
                coverage = len(usable) / len(claim_rows) if claim_rows else 0.0
                if coverage >= 0.6:
                    novelty_score = round(
                        0.6 * float(independent_score) * 5.0
                        + 0.4 * novelty_semantic.score,
                        1,
                    )
                    novelty_source = "hybrid"
                    novelty_confidence = round(
                        min(novelty_semantic.confidence, coverage), 2
                    )
                else:
                    novelty_score = min(novelty_semantic.score, 3.5)
                    novelty_source = "degraded"
                    novelty_confidence = min(novelty_semantic.confidence, 0.5)
                    novelty_weaknesses.append(
                        "Evidence-graph coverage was insufficient for an independent novelty estimate."
                    )
                    novelty_deductions.append(
                        f"Usable graph novelty claims: {coverage:.0%}."
                    )
            except Exception as exc:
                logger.warning("M6 independent novelty score degraded: %s", exc)
                novelty_score = min(novelty_semantic.score, 3.5)
                novelty_source = "degraded"
                novelty_confidence = min(novelty_semantic.confidence, 0.3)
                novelty_deductions.append(
                    f"Independent novelty metric unavailable: {type(exc).__name__}."
                )
        novelty_row = self._dimension_detail(
            "novelty",
            novelty_score,
            source=novelty_source,
            confidence=novelty_confidence,
            strengths=novelty_semantic.strengths,
            weaknesses=novelty_weaknesses,
            deductions=novelty_deductions,
        )

        audits = list(factual_verdict.premise_audits) if factual_verdict else []
        audit_verdicts = {item.verdict for item in audits}
        if "contradicted" in audit_verdicts:
            factual_score = 1.0
        elif audit_verdicts & {"unsupported", "invalid_citation"}:
            factual_score = 2.0
        elif "partially_supported" in audit_verdicts:
            factual_score = 3.5
        elif audits:
            factual_score = 5.0
        else:
            factual_score = 3.0
        evidence_coverage = float(gates.get(
            "evidence_coverage_hypothesis",
            gates.get("evidence_coverage", 0.0),
        ))
        source_quality = float(gates.get("source_quality", 0.0))
        deterministic_evidence_score = round(
            0.6 * factual_score
            + 0.25 * evidence_coverage * 5.0
            + 0.15 * source_quality * 5.0,
            1,
        )
        evidence_semantic = plan_quality.evidence_reliability
        # A structured factual-premise audit is authoritative.  The semantic
        # plan-quality reviewer may comment on source diversity, but it must
        # not lower evidence reliability merely because an explicitly novel
        # architecture or mechanism has no paper that already proves it.
        evidence_score = (
            deterministic_evidence_score
            if audits else evidence_semantic.score
        )
        evidence_row = self._dimension_detail(
            "evidence_reliability",
            evidence_score,
            source="hybrid" if audits else "degraded",
            confidence=max(
                evidence_semantic.confidence,
                0.9 if audits else 0.3,
            ),
            strengths=(
                [f"{len(audits)} factual premise(s) received a structured audit."]
                if audits else evidence_semantic.strengths
            ),
            weaknesses=(
                [item.rationale for item in audits if item.verdict != "supported"]
                if audits else (
                    list(evidence_semantic.weaknesses)
                    or ["No auditable factual premises were available."]
                )
            ),
            deductions=[] if audits else evidence_semantic.deductions,
        )

        predictions = list(getattr(hypothesis, "observable_predictions", []) or [])
        falsifications = list(getattr(hypothesis, "falsification_conditions", []) or [])
        if experimental_validation_verdict is not None:
            target_items = experimental_validation_verdict.items
            validation_ratio = (
                sum(
                    1.0 if item.verdict == "covered"
                    else 0.5 if item.verdict == "partial" else 0.0
                    for item in target_items
                ) / len(target_items)
                if target_items else 0.0
            )
            structural_testability_score = round(
                1.0 + 4.0 * validation_ratio
                if predictions and falsifications else 1.0,
                1,
            )
        else:
            structural_testability_score = 5.0 if predictions and falsifications else 2.0
            validation_ratio = 1.0 if predictions and falsifications else 0.0
        testability_semantic = plan_quality.testability
        testability_score = min(
            structural_testability_score,
            testability_semantic.score,
        )
        testability_row = self._dimension_detail(
            "testability",
            testability_score,
            source="hybrid" if experimental_validation_verdict else "deterministic",
            confidence=max(
                testability_semantic.confidence,
                0.9 if experimental_validation_verdict else 0.5,
            ),
            strengths=testability_semantic.strengths,
            weaknesses=(
                list(testability_semantic.weaknesses)
                + ([] if predictions and falsifications else [
                    "Missing observable predictions or falsification conditions."
                ])
            ),
            deductions=testability_semantic.deductions,
        )

        experimental_row = self._dimension_detail(
            "experimental_rigor",
            plan_quality.experimental_rigor.score,
            source="llm" if plan_quality.experimental_rigor.confidence else "degraded",
            confidence=plan_quality.experimental_rigor.confidence,
            strengths=plan_quality.experimental_rigor.strengths,
            weaknesses=plan_quality.experimental_rigor.weaknesses,
            deductions=plan_quality.experimental_rigor.deductions,
        )
        reproducibility_row = self._reproducibility_detail(plan).model_copy(update={
            "score": round(
                0.6 * plan_quality.statistics_reproducibility.score
                + 0.4 * self._reproducibility_detail(plan).score,
                1,
            ),
            "source": (
                "hybrid" if plan_quality.statistics_reproducibility.confidence
                else "deterministic"
            ),
            "confidence": max(
                plan_quality.statistics_reproducibility.confidence,
                0.5,
            ),
            "weaknesses": (
                list(plan_quality.statistics_reproducibility.weaknesses)
                + list(self._reproducibility_detail(plan).weaknesses)
            ),
            "deductions": (
                list(plan_quality.statistics_reproducibility.deductions)
                + list(self._reproducibility_detail(plan).deductions)
            ),
        })
        feasibility_row = self._dimension_detail(
            "technical_feasibility",
            plan_quality.technical_feasibility.score,
            source="llm" if plan_quality.technical_feasibility.confidence else "degraded",
            confidence=plan_quality.technical_feasibility.confidence,
            strengths=plan_quality.technical_feasibility.strengths,
            weaknesses=plan_quality.technical_feasibility.weaknesses,
            deductions=plan_quality.technical_feasibility.deductions,
        )

        conditions = M6ScoreConditions(
            invalid_hypothesis_output=_is_invalid_fallback_hypothesis(hypothesis),
            # Broken/missing trace pointers are an auditability defect, not by
            # themselves proof that the scientific answer is off-task.  The
            # severe display cap requires both structural and semantic failure.
            task_misaligned=(
                not deterministic_alignment_passed
                and not semantic_alignment_passed
            ),
            core_fact_contradicted="contradicted" in audit_verdicts,
            core_untestable=not predictions or not falsifications,
            experimental_validation_missing=_has_missing_core_validation_target(
                experimental_validation_verdict
            ),
            multiple_major_design_defects=sum(
                len(item.blocking_issues)
                for item in (
                    plan_quality.experimental_rigor,
                    plan_quality.statistics_reproducibility,
                    plan_quality.technical_feasibility,
                )
            ) >= 2,
        )
        return [
            task_row,
            novelty_row,
            logic_row,
            evidence_row,
            testability_row,
            experimental_row,
            reproducibility_row,
            feasibility_row,
        ], conditions

    async def _audit_pair_semantics(
        self,
        state: PipelineState,
        hypothesis: Any,
        plan: Any,
    ) -> tuple[bool, str]:
        """Audit the hypothesis-plan research object without sync thread joins."""

        assert self.client is not None
        contract = state.problem_card.task_contract if state.problem_card else None
        primary_objects = [
            {
                "entity_id": entity.entity_id,
                "name": entity.name,
                "aliases": list(entity.aliases),
            }
            for entity in (contract.entities if contract is not None else [])
            if entity.role == "primary_object"
        ]
        if not primary_objects:
            return True, "No required primary object needs semantic auditing."

        prompt = (
            "Judge whether BOTH the hypothesis and its research plan study the "
            "same type of primary object required by the original task. Judge "
            "meaning, not mere word overlap. Answer ONLY yes or no, followed "
            "by one short reason.\n\nPrimary task objects:\n"
            + json.dumps(primary_objects, ensure_ascii=False, indent=2)
            + "\n\nHypothesis:\n"
            + str(hypothesis.statement)
            + "\n"
            + str(hypothesis.mechanism)
            + "\n\nPlan study subjects:\n"
            + str(plan.study_subjects)
        )
        started_at = time.monotonic()
        emit_event(
            "tool_started",
            module="m6",
            tool="task_contract_auditor",
            status="running",
            message="M6 task-object semantic audit started",
            details={
                "timeout_seconds": self.semantic_alignment_timeout_seconds,
            },
        )
        try:
            response = await asyncio.wait_for(
                self.client.chat(
                    user_prompt=prompt,
                    max_tokens=512,
                    temperature=0.0,
                    disable_thinking=True,
                ),
                timeout=self.semantic_alignment_timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            emit_event(
                "tool_failed",
                module="m6",
                tool="task_contract_auditor",
                status="failed",
                message="M6 task-object semantic audit timed out",
                elapsed_seconds=time.monotonic() - started_at,
            )
            raise RuntimeError(
                "M6 semantic task-contract audit timed out after "
                f"{self.semantic_alignment_timeout_seconds:g} seconds"
            ) from exc
        answer = str(response or "").strip()
        consistent = answer.casefold().startswith("yes")
        emit_event(
            "tool_completed",
            module="m6",
            tool="task_contract_auditor",
            status="completed",
            message=(
                "M6 task-object semantic audit passed"
                if consistent else "M6 task-object semantic audit failed"
            ),
            elapsed_seconds=time.monotonic() - started_at,
            details={"consistent": consistent},
        )
        return consistent, answer

    # ------------------------------------------------------------------
    # ModuleProtocol implementation
    # ------------------------------------------------------------------

    async def __call__(
        self,
        state: PipelineState,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        version = state.iteration_count + 1

        if self.client is None:
            raise RuntimeError(
                "M6 requires an LLM client — pass llm_config / set OPENAI_API_KEY."
            )

        if not state.top_hypotheses:
            raise RuntimeError(
                "M6 requires at least one top hypothesis to review. "
                "Ensure M4 completed and produced candidate hypotheses."
            )
        if not state.research_plans:
            raise RuntimeError(
                "M6 requires at least one research plan to review. "
                "Ensure M5 completed and produced plans."
            )

        # Historical snapshots may still carry M1 retrieval trace IDs (R1...Rn).
        # Rebuild both sides from their literal output before review so a
        # resumed run is judged against the same whole-question Q0 contract as
        # a fresh run.  The canonicalizers never synthesize scientific text.
        from .m4_hypothesis_generation import M4HypothesisGeneration
        from .m5_research_plan import M5ResearchPlan

        normalised_hypotheses = M4HypothesisGeneration._canonicalize_task_traces(
            state, list(state.top_hypotheses)
        )
        normalisation_state = state.model_copy(update={
            "top_hypotheses": normalised_hypotheses,
        })
        normalised_plans = [
            M5ResearchPlan._canonicalize_task_trace(normalisation_state, item)
            for item in state.research_plans
        ]
        evaluation_state = normalisation_state.model_copy(update={
            "research_plans": normalised_plans,
        })

        hypothesis = normalised_hypotheses[0]
        plan = next(
            (
                p for p in normalised_plans
                if p.hypothesis_id == hypothesis.hypothesis_id
            ),
            None,
        )
        if plan is None:
            raise RuntimeError(
                f"M6 cannot find a research plan matching top hypothesis "
                f"{hypothesis.hypothesis_id!r}.  M6 must review a "
                f"hypothesis-plan pair that were designed together."
            )
        graph = state.evidence_graph
        graph_context = build_graph_context(state)
        synthesis_contract = synthesis_contract_for_state(state)
        valid_evidence_ids = set(graph_context.available_evidence_ids)
        # M4 may generate hypotheses scoped to a subset of atomic
        # requirements, but M6 approves the final hypothesis-plan pair for the
        # user's *whole* question.  Do not let a candidate hide an omitted M1
        # requirement simply by leaving its ID out of its own trace.  This also
        # keeps the live M6 hard gate consistent with the posthoc scorer.
        hypothesis_alignment = assess_task_alignment(
            state,
            hypothesis.model_dump_json(exclude={"task_trace"}),
            subject_text="\n".join([hypothesis.statement, hypothesis.mechanism]),
            trace=hypothesis.task_trace,
            semantic_client=None,
            contract_override=synthesis_contract,
        )
        plan_alignment = assess_task_alignment(
            state,
            plan.model_dump_json(exclude={"task_trace"}),
            subject_text=plan.study_subjects,
            trace=plan.task_trace,
            semantic_client=None,
            contract_override=synthesis_contract,
        )
        if self.fast_mode:
            # Fast mode skips the extra task-object semantic LLM call.  The
            # deterministic lexical alignment above is sufficient for a
            # quick review; the run always ends after this pass.
            semantic_alignment_passed = True
            semantic_alignment_rationale = (
                "fast mode: semantic pair audit skipped for speed"
            )
        else:
            semantic_alignment_passed, semantic_alignment_rationale = (
                await self._audit_pair_semantics(state, hypothesis, plan)
            )
        deterministic_alignment_passed = (
            hypothesis_alignment.passed
            and plan_alignment.passed
            and semantic_alignment_passed
        )
        deterministic_alignment_rationale = " ".join(filter(None, [
            f"Hypothesis: {hypothesis_alignment.rationale}",
            f"Plan: {plan_alignment.rationale}",
            f"Semantic pair audit: {semantic_alignment_rationale}",
        ]))

        failed_hyp = not hypothesis_alignment.passed or not semantic_alignment_passed
        failed_plan = not plan_alignment.passed
        if failed_hyp and failed_plan:
            alignment_attr = "both"
        elif failed_hyp:
            alignment_attr = "hypothesis"
        elif failed_plan:
            alignment_attr = "plan"
        else:
            alignment_attr = "both"

        # Task alignment is a deterministic, independent hard gate. It does
        # not consume an LLM call and cannot be overridden by a plausible but
        # off-topic model self-assessment.
        new_reviews: List[ReviewResult] = [ReviewResult(
            dimension=ReviewerDimension("task_alignment"),
            attribution=alignment_attr,
            reasoning=deterministic_alignment_rationale,
            score=5.0 if deterministic_alignment_passed else 1.0,
            comments="Compared the task contract with the hypothesis and study subject.",
            suggestions=(
                "" if deterministic_alignment_passed
                else "Restore the original research object, domain, and requested task before approval."
            ),
            hard_gate_passed=deterministic_alignment_passed,
            version=version,
        )]

        # Remaining specialist reviewers use LLMs; "overall" is computed.
        specialist_dims = [
            dimension for dimension in self.reviewer_dims
            if dimension not in {"overall", "task_alignment"}
        ]
        auditable_factual_claims = bool(
            self._build_evidence_audit_scope(hypothesis, plan)
        )
        factual_verdict: Optional[EvidenceSufficiencyVerdict] = None

        for dim in specialist_dims:
            context_purpose = (
                "m6_feasibility"
                if dim == "method_feasibility"
                else "m6_logic"
            )
            context_pack = ContextPlanner().plan(
                graph_context,
                ContextRequest(
                    purpose=context_purpose,
                    focus_evidence_ids=tuple(dict.fromkeys([
                        *hypothesis.supporting_evidence,
                        *plan.supporting_evidence_ids,
                    ])),
                ),
            )
            emit_context_built(
                "m6",
                f"reviewer:{dim}",
                context_pack,
            )
            # Anchor the score (rubric) and force reason-before-score, both
            # sourced from the single rubric definition.
            system_prompt = "\n\n".join(
                p for p in (
                    M6_REVIEWER_PROMPTS[dim],
                    review_rubric_line(dim),
                    M6_REASON_FIRST,
                    M6_FORMAT_NOTE,
                ) if p
            )
            started_at = time.monotonic()
            emit_event(
                "tool_started",
                module="m6",
                tool=f"reviewer:{dim}",
                status="running",
                message=f"Reviewer agent started: {dim}",
                details={"version": version},
            )
            try:
                payload = await asyncio.wait_for(
                    self.client.structured_chat(
                        system_prompt=system_prompt,
                        user_prompt=M6_USER_TEMPLATE.format(
                            original_question=state.input_question,
                            problem_card_json=json.dumps(
                                synthesis_problem_payload(state.problem_card),
                                ensure_ascii=False,
                                indent=2,
                            ),
                            hypothesis_json=json.dumps(hypothesis.model_dump(mode="json"), ensure_ascii=False, indent=2),
                            plan_json=json.dumps(plan.model_dump(mode="json"), ensure_ascii=False, indent=2),
                            graph_context=context_pack.rendered,
                            facts_count=len(graph.established_facts) if graph else 0,
                            conflicts_count=len(graph.conflicts) if graph else 0,
                            gaps_count=len(graph.knowledge_gaps) if graph else 0,
                        ),
                        output_schema=ReviewResult.model_json_schema(),
                        max_tokens=8192,
                        temperature=getattr(self.llm_config, "temperature", 0.1),
                        disable_thinking=True,
                    ),
                    timeout=self.reviewer_timeout_seconds,
                )
            except asyncio.TimeoutError as exc:
                raise RuntimeError(
                    f"M6 reviewer {dim!r} timed out after "
                    f"{self.reviewer_timeout_seconds:g} seconds"
                ) from exc
            payload = dict(payload)
            payload["dimension"] = dim
            payload["version"] = version
            if dim in {
                "scientific_logic", "objective_evidence_consistency",
                "testability", "novelty",
            }:
                payload["attribution"] = "hypothesis"
            elif dim in {"method_feasibility"}:
                payload["attribution"] = "plan"
            else:
                payload["attribution"] = "both"
            review = ReviewResult.model_validate(payload)
            cited_evidence = list(dict.fromkeys(
                identifier
                for identifier in review.evidence_ids
                if identifier in valid_evidence_ids
            ))
            graph_corrections = self._normalise_graph_corrections(
                review,
                valid_evidence_ids=valid_evidence_ids,
                version=version,
            )
            if dim == "objective_evidence_consistency" and factual_verdict is not None:
                review = self._objective_evidence_review(
                    factual_verdict,
                    version=version,
                ).model_copy(update={
                    "graph_correction_requests": graph_corrections,
                })
            else:
                updates: Dict[str, Any] = {"evidence_ids": cited_evidence}
                if dim == "objective_evidence_consistency":
                    if auditable_factual_claims:
                        calibrated_score = review.score
                        if not cited_evidence:
                            calibrated_score = min(calibrated_score, 2.0)
                        hard_gate_passed = calibrated_score >= 3.0
                    else:
                        # There is no factual-premise scope to audit.  Novel
                        # statements and M3 bridges are assessed by logic and
                        # experimental-validation reviewers, not this evidence
                        # gate; absence of paper IDs is therefore neutral.
                        calibrated_score = 5.0
                        hard_gate_passed = True
                    updates.update({
                        "score": calibrated_score,
                        "hard_gate_passed": hard_gate_passed,
                    })
                updates["graph_correction_requests"] = graph_corrections
                review = review.model_copy(update=updates)
            new_reviews.append(review)
            emit_event(
                "tool_completed",
                module="m6",
                tool=f"reviewer:{dim}",
                status="completed",
                message=f"Reviewer agent completed: {dim}, score {review.score:.1f}/5",
                elapsed_seconds=time.monotonic() - started_at,
                details={"version": version, "score": review.score},
            )

        # Preserve the established external-call order (specialist reviews
        # first, factual audit second), then reconcile the evidence dimension
        # before any quality gate or overall score is calculated.  The raw LLM
        # review may still propose graph corrections, but it is no longer an
        # independent score source.
        if self.m6_evidence_revisit and not self.fast_mode:
            factual_verdict = await self._judge_evidence_sufficiency(
                state, hypothesis, plan, version,
            )
            authoritative_evidence_review = self._objective_evidence_review(
                factual_verdict,
                version=version,
            )
            for index, review in enumerate(new_reviews):
                if review.dimension.value != "objective_evidence_consistency":
                    continue
                new_reviews[index] = authoritative_evidence_review.model_copy(
                    update={
                        "graph_correction_requests": self._reconcile_graph_corrections(
                            factual_verdict,
                            review.graph_correction_requests,
                        ),
                    }
                )
                break

        # --- Objective Quality Gates & Metrics ---
        gates: Dict[str, Any] = {}
        try:
            gates = _quality_gates(evaluation_state)
            
            # 1. evidence_coverage
            coverage = gates.get("evidence_coverage", 0.0)
            coverage_hyp = gates.get("evidence_coverage_hypothesis", 1.0)
            coverage_plan = gates.get("evidence_coverage_plan", 1.0)
            coverage_score = coverage * 5.0
            coverage_passed = coverage_score >= 4.5
            
            if not coverage_passed:
                failed_hyp = coverage_hyp < 0.9
                failed_plan = coverage_plan < 0.9
                if failed_hyp and failed_plan:
                    cov_attr = "both"
                elif failed_hyp:
                    cov_attr = "hypothesis"
                elif failed_plan:
                    cov_attr = "plan"
                else:
                    cov_attr = "both"
            else:
                cov_attr = "both"
                
            new_reviews.append(ReviewResult(
                dimension=ReviewerDimension("evidence_coverage_gate"),
                attribution=cov_attr,
                reasoning=f"Calculated evidence coverage is {coverage*100:.1f}%.",
                score=round(coverage_score, 1),
                comments="Objective code-level assessment of cited evidence coverage.",
                suggestions="" if coverage_passed else "Cite more valid evidence IDs from the evidence graph.",
                hard_gate_passed=coverage_passed,
                version=version,
            ))

            # 2. answer_completeness
            completeness = gates.get("answer_completeness", 0.0)
            completeness_score = completeness * 5.0
            completeness_passed = completeness_score >= 3.0
            new_reviews.append(ReviewResult(
                dimension=ReviewerDimension("answer_completeness_gate"),
                attribution="plan",
                reasoning=f"Calculated answer completeness is {completeness*100:.1f}%.",
                score=round(completeness_score, 1),
                comments="Objective code-level assessment of research plan structural completeness.",
                suggestions="" if completeness_passed else "Ensure the research plan contains required structured sections.",
                hard_gate_passed=completeness_passed,
                version=version,
            ))

            # 3. source_quality
            source = gates.get("source_quality", 0.0)
            source_score = source * 5.0
            new_reviews.append(ReviewResult(
                dimension=ReviewerDimension("source_quality_gate"),
                attribution="both",
                reasoning=f"Calculated source quality is {source*100:.1f}%.",
                score=round(source_score, 1),
                comments="Objective code-level assessment of source paper text availability.",
                suggestions="Try to search for papers with structured fulltext or OA PDFs.",
                hard_gate_passed=True,
                version=version,
            ))

        except Exception as e:
            logger.warning(f"Failed to compute objective quality gates/metrics: {e}")

        # M5 already audits every M4 validation target and persists the final
        # verdict.  Reuse that result here instead of repeating the same LLM
        # request inside M6.
        experimental_validation_verdict = state.experimental_validation_verdict
        if experimental_validation_verdict is not None:
            item_count = len(experimental_validation_verdict.items)
            weighted_coverage = sum(
                1.0 if item.verdict == "covered" else 0.5 if item.verdict == "partial" else 0.0
                for item in experimental_validation_verdict.items
            )
            validation_score = round(
                5.0 * weighted_coverage / item_count if item_count else 5.0,
                1,
            )
            new_reviews.append(ReviewResult(
                dimension=ReviewerDimension("experimental_validation_coverage"),
                attribution="plan",
                reasoning=experimental_validation_verdict.rationale,
                score=validation_score,
                comments="Independent target-by-target audit of M4 claims against the M5 plan.",
                suggestions="\n".join(
                    f"- {item.target_id}: {item.rationale}"
                    for item in experimental_validation_verdict.items
                    if item.verdict != "covered" and item.rationale
                ),
                hard_gate_passed=experimental_validation_verdict.sufficient,
                version=version,
            ))

        # Modern calibrated score.  This is deliberately a separate reporting
        # layer: the ReviewResult rows below carry ``hard_gate_passed=None`` so
        # their numeric values cannot accidentally become routing signals.
        scoring_dimensions, scoring_conditions = (
            await self._build_modern_scoring_dimensions(
                state=state,
                hypothesis=hypothesis,
                plan=plan,
                graph_context=graph_context,
                hypothesis_alignment=hypothesis_alignment,
                plan_alignment=plan_alignment,
                semantic_alignment_passed=semantic_alignment_passed,
                deterministic_alignment_passed=deterministic_alignment_passed,
                factual_verdict=factual_verdict,
                experimental_validation_verdict=experimental_validation_verdict,
                gates=gates,
                reviews=new_reviews,
            )
        )
        m6_scoring_summary = self._build_scoring_summary(
            scoring_dimensions,
            scoring_conditions,
        )
        legacy_failed_gates = [
            review.dimension.value
            for review in new_reviews
            if review.hard_gate_passed is False
        ]
        modern_attribution = {
            "task_coverage": "both",
            "novelty": "hypothesis",
            "scientific_logic": "hypothesis",
            "evidence_reliability": "hypothesis",
            "testability": "hypothesis",
            "experimental_rigor": "plan",
            "statistics_reproducibility": "plan",
            "technical_feasibility": "plan",
        }
        for detail in scoring_dimensions:
            new_reviews.append(ReviewResult(
                dimension=ReviewerDimension(detail.dimension),
                attribution=modern_attribution[detail.dimension],
                reasoning=(
                    f"Calibrated {detail.dimension} score from {detail.source} "
                    f"assessment; confidence {detail.confidence:.2f}."
                ),
                score=detail.score,
                comments="\n".join(detail.strengths),
                suggestions="\n".join(
                    [*detail.weaknesses, *detail.deductions]
                ),
                hard_gate_passed=None,
                version=version,
            ))

        # ``overall`` is a display row backed by the persisted summary.  It is
        # not a quality gate and therefore cannot request another iteration.
        if "overall" in self.reviewer_dims:
            overall_started_at = time.monotonic()
            emit_event(
                "tool_started",
                module="m6",
                tool="overall_score_aggregator",
                status="running",
                message="Aggregating overall score",
                details={"version": version},
            )
            new_reviews.append(ReviewResult(
                dimension=ReviewerDimension("overall"),
                reasoning=m6_scoring_summary.rationale,
                score=m6_scoring_summary.final_score,
                comments=(
                    "Weighted eight-dimension score for display and reporting; "
                    "numeric score is excluded from iteration routing."
                ),
                suggestions="\n".join(
                    cap.reason for cap in m6_scoring_summary.applied_caps
                ),
                evidence_ids=list(dict.fromkeys(
                    evidence_id
                    for review in new_reviews
                    for evidence_id in review.evidence_ids
                )),
                hard_gate_passed=not legacy_failed_gates,
                version=version,
            ))
            emit_event(
                "tool_completed",
                module="m6",
                tool="overall_score_aggregator",
                status="completed",
                message=f"Overall score {m6_scoring_summary.final_score:.1f}/5",
                elapsed_seconds=time.monotonic() - overall_started_at,
                details={
                    "version": version,
                    "score": m6_scoring_summary.final_score,
                    "raw_score": m6_scoring_summary.raw_score,
                    "applied_caps": [
                        cap.rule_id for cap in m6_scoring_summary.applied_caps
                    ],
                    "routing_uses_score": False,
                },
            )

        correction_by_id = {
            request.request_id: request.model_copy(deep=True)
            for request in state.graph_correction_requests
        }
        for review in new_reviews:
            for request in review.graph_correction_requests:
                correction_by_id[request.request_id] = request

        patch: Dict[str, Any] = {
            "reviews": state.reviews + new_reviews,
            "m6_scoring_summary": m6_scoring_summary,
            "iteration_count": version,
            "graph_correction_requests": list(correction_by_id.values()),
            "top_hypotheses": normalised_hypotheses,
            "research_plans": normalised_plans,
        }
        if experimental_validation_verdict is not None:
            patch["experimental_validation_verdict"] = experimental_validation_verdict

        # --- iteration core: scoped factual-premise verdict (fail-closed) ---
        # This audit is enabled by the same iteration switch as supplement
        # routing; it never broadens into a free-form mechanism gap judge.
        if factual_verdict is not None:
            verdict = factual_verdict
            gap_gain = state.metrics.get("m3_gap_gain") or {}
            patch["evidence_verdict"] = verdict
            merged_gaps = self._merge_evidence_gaps(
                state.evidence_gaps,
                verdict.gaps,
                version,
                verdict.sufficient,
                gap_gain=gap_gain,
            )
            merged_gaps, metrics = self._apply_zero_gain_convergence(
                state, merged_gaps
            )
            patch["evidence_gaps"] = merged_gaps
            patch["metrics"] = metrics

        return patch

    # ------------------------------------------------------------------
    # Experimental validation coverage (M4 -> M5)
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Evidence sufficiency (iteration core)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_evidence_audit_scope(
        hypothesis: Any,
        plan: Optional[ResearchPlan] = None,
    ) -> List[_AuditableClaim]:
        """Return the exclusive factual scope for evidence-gap auditing.

        M4's epistemic contract is intentionally preserved here: only explicit
        required factual premises and plan links explicitly marked
        ``unsupported`` may request literature supplementation.  Mechanisms,
        research gaps, predictions and bridge assumptions are conjectural
        content and are reviewed for logic/experimental coverage elsewhere.
        """
        claims: List[_AuditableClaim] = []
        for premise in list(getattr(hypothesis, "factual_premises", []) or []):
            claim = str(getattr(premise, "claim", "") or "").strip()
            if not getattr(premise, "required", True) or not claim:
                continue
            claims.append(_AuditableClaim(
                source_claim_type="factual_premise",
                source_claim_id=str(getattr(premise, "premise_id", "") or ""),
                claim=claim,
                required=True,
                supporting_evidence_ids=tuple(
                    str(item)
                    for item in getattr(premise, "supporting_evidence_ids", [])
                    if str(item).strip()
                ),
            ))
        if plan is not None:
            for index, link in enumerate(getattr(plan, "evidence_links", []) or []):
                if getattr(link, "support_status", "") != "unsupported":
                    continue
                claim = str(getattr(link, "claim", "") or getattr(link, "plan_element", "") or "").strip()
                if not claim:
                    continue
                claims.append(_AuditableClaim(
                    source_claim_type="plan_fact",
                    source_claim_id=f"plan:evidence_link:{index}",
                    claim=claim,
                    required=True,
                    supporting_evidence_ids=tuple(
                        str(item)
                        for item in getattr(link, "supporting_evidence_ids", [])
                        if str(item).strip()
                    ),
                ))
        return claims

    async def _audit_factual_premises(
        self,
        state: PipelineState,
        hypothesis: Any,
        plan: Any,
        version: int,
    ) -> EvidenceSufficiencyVerdict:
        """Independently audit only required factual premises.

        M6 must not turn an unproven mechanism or an innovative statement into
        a literature-search gap.  The shared ``EvidenceAuditService`` receives
        only ``evidence_backed`` premises (plus explicitly unsupported M5
        fact-links), so the source claim remains available for routing while
        conjectural text stays in the logic/coverage reviews.
        """
        graph_context = build_graph_context(state)
        scope = self._build_evidence_audit_scope(hypothesis, plan)
        started_at = time.monotonic()
        emit_event(
            "tool_started",
            module="m6",
            tool="evidence_sufficiency_judge",
            status="running",
            message="M6 factual-premise audit started",
            details={"version": version, "auditable_claim_count": len(scope)},
        )
        if not scope:
            # Checkpoints created before the structured M4 premise contract
            # used a derived ProblemCard and may carry a legacy gap ledger.
            # Preserve that ledger through one migration-only verdict call so
            # old runs remain resumable; modern M1 cards with no factual
            # premises do not enter this compatibility branch.
            legacy_card = (
                state.problem_card is not None
                and state.problem_card.task_contract.source == "derived"
            )
            if legacy_card:
                try:
                    legacy_payload = await asyncio.wait_for(
                        self.client.structured_chat(
                            system_prompt=M6_EVIDENCE_VERDICT_SYSTEM,
                            user_prompt=json.dumps({
                                "legacy_gap_ledger": [
                                    gap.model_dump(mode="json")
                                    for gap in state.evidence_gaps
                                ],
                                "instruction": "Preserve or close only these existing legacy gaps; do not invent new claims.",
                            }, ensure_ascii=False, indent=2),
                            output_schema=EvidenceSufficiencyVerdict.model_json_schema(),
                            max_tokens=4096,
                            temperature=0.0,
                            disable_thinking=True,
                        ),
                        timeout=self.reviewer_timeout_seconds,
                    )
                    legacy_verdict = EvidenceSufficiencyVerdict.model_validate(dict(legacy_payload or {}))
                    for gap in legacy_verdict.gaps:
                        gap.gap_id = make_gap_id(
                            gap.target_sub_question or gap.description,
                            gap.gap_type,
                            gap.canonical_entities,
                        )
                        gap.source_review_version = version
                        gap.status = "open" if gap.status not in {"closed", "unimprovable"} else gap.status
                    emit_event(
                        "tool_completed",
                        module="m6",
                        tool="evidence_sufficiency_judge",
                        status="completed",
                        message=f"Legacy gap ledger migrated: {len(legacy_verdict.gaps)} gap(s)",
                        elapsed_seconds=time.monotonic() - started_at,
                        details={"version": version, "sufficient": legacy_verdict.sufficient},
                    )
                    return legacy_verdict
                except Exception as exc:
                    legacy_gaps = list(state.evidence_gaps)
                    if not legacy_gaps:
                        legacy_gaps = [self._coverage_gap(state, version)]
                    verdict = EvidenceSufficiencyVerdict(
                        sufficient=False,
                        gaps=legacy_gaps,
                        rationale=f"Legacy evidence ledger migration failed closed: {type(exc).__name__}: {exc}",
                    )
                    emit_event(
                        "tool_failed",
                        module="m6",
                        tool="evidence_sufficiency_judge",
                        status="failed",
                        message=verdict.rationale,
                        elapsed_seconds=time.monotonic() - started_at,
                        details={"version": version, "legacy": True},
                    )
                    return verdict
            verdict = EvidenceSufficiencyVerdict(
                sufficient=True,
                rationale="No required factual premise or explicitly unsupported plan fact requires literature audit.",
            )
            emit_event(
                "tool_completed",
                module="m6",
                tool="evidence_sufficiency_judge",
                status="completed",
                message="M6 factual-premise audit: no auditable claims",
                elapsed_seconds=time.monotonic() - started_at,
                details={"version": version, "sufficient": True, "gap_ids": []},
            )
            return verdict

        factual_by_id = {
            str(getattr(item, "premise_id", "")): item
            for item in (getattr(hypothesis, "factual_premises", []) or [])
            if str(getattr(item, "premise_id", "")).strip()
        }
        premises: List[HypothesisPremise] = []
        for item in scope:
            existing = factual_by_id.get(item.source_claim_id)
            if existing is not None:
                premises.append(existing)
                continue
            # Plan links are deliberately converted to a narrow factual
            # premise for the shared auditor; their source identity is kept in
            # the resulting EvidenceGap as ``plan_fact``.
            premises.append(HypothesisPremise(
                premise_id=item.source_claim_id,
                claim=item.claim,
                kind="evidence_backed",
                required=item.required,
                supporting_evidence_ids=list(item.supporting_evidence_ids),
            ))

        try:
            auditor = EvidenceAuditService(
                self.client,
                timeout_seconds=self.reviewer_timeout_seconds,
                allow_legacy_node_ids=False,
            )
            results = await auditor.audit_premises(
                premises,
                state.evidence_graph,
                candidate_evidence_ids=graph_context.available_evidence_ids,
            )
        except Exception as exc:
            # A technical failure remains bound to each concrete premise; it
            # must never become a free-form gap about the whole mechanism.
            results = [PremiseAuditResult(
                premise_id=item.source_claim_id,
                claim=item.claim,
                verdict="unsupported",
                rationale=f"Factual premise audit failed closed: {type(exc).__name__}: {exc}",
            ) for item in scope]

        scope_by_id = {item.source_claim_id: item for item in scope}
        gaps: List[EvidenceGap] = []
        accepted_evidence_ids: List[str] = []
        for result in results:
            source = scope_by_id.get(result.premise_id)
            if source is None:
                continue
            accepted_evidence_ids.extend(
                str(item) for item in result.evidence_ids if str(item).strip()
            )
            if result.verdict in {"supported", "partially_supported", "not_applicable"}:
                continue
            target = (
                state.problem_card.original_question
                if state.problem_card is not None and state.problem_card.original_question
                else state.input_question
            )
            entities: List[str] = []
            if state.problem_card is not None:
                entities = list(dict.fromkeys([
                    *[
                        entity.name for entity in state.problem_card.task_contract.entities
                        if entity.required and entity.name
                    ],
                    *state.problem_card.key_entities,
                ]))[:6]
            query = " ".join([source.claim, target]).strip()
            contradicted = result.verdict == "contradicted"
            gap = EvidenceGap(
                description=(
                    f"Factual premise {source.source_claim_id} is contradicted: {source.claim}"
                    if contradicted
                    else f"Factual premise {source.source_claim_id} lacks canonical support: {source.claim}"
                ),
                gap_type="conflict" if contradicted else "coverage",
                canonical_entities=entities,
                suggested_queries=[query or target],
                target_sub_question=target,
                source_review_version=version,
                source_claim_type=source.source_claim_type,
                source_claim_id=source.source_claim_id,
                status="open",
                hypothesis_ids=[hypothesis.hypothesis_id],
                scientific_resolution="contradicted" if contradicted else "unreviewed",
                resolution_evidence_ids=list(dict.fromkeys(result.evidence_ids)) if contradicted else [],
                contradicting_evidence_ids=list(dict.fromkeys(result.evidence_ids)) if contradicted else [],
                search_completed=False,
                technical_errors=(
                    [result.rationale] if "failed closed" in result.rationale.lower() else []
                ),
                rationale=result.rationale,
            )
            gap.gap_id = make_gap_id(
                gap.target_sub_question, gap.gap_type, gap.canonical_entities,
            )
            gaps.append(gap)

        rationale = "; ".join(
            f"{result.premise_id}: {result.verdict} — {result.rationale}"
            for result in results if result.rationale
        )
        verdict = EvidenceSufficiencyVerdict(
            sufficient=not gaps,
            gaps=gaps,
            evidence_ids=list(dict.fromkeys(accepted_evidence_ids)),
            premise_audits=[FactualPremiseAudit(
                premise_id=result.premise_id,
                claim=result.claim,
                verdict=result.verdict,
                evidence_ids=list(dict.fromkeys(result.evidence_ids)),
                corrected_claim=result.corrected_claim,
                rationale=result.rationale,
            ) for result in results],
            rationale=rationale or "All audited factual premises are supported.",
        )
        emit_event(
            "tool_completed",
            module="m6",
            tool="evidence_sufficiency_judge",
            status="completed",
            message=(
                f"M6 factual-premise audit: {'sufficient' if verdict.sufficient else 'insufficient'}"
                f", {len(verdict.gaps)} gap(s)"
            ),
            elapsed_seconds=time.monotonic() - started_at,
            details={
                "version": version,
                "sufficient": verdict.sufficient,
                "auditable_claim_count": len(scope),
                "gap_ids": [gap.gap_id for gap in verdict.gaps],
            },
        )
        return verdict

    async def _judge_evidence_sufficiency(
        self,
        state: PipelineState,
        hypothesis: Any,
        plan: Any,
        version: int,
    ) -> EvidenceSufficiencyVerdict:
        return await self._audit_factual_premises(state, hypothesis, plan, version)

        # Legacy implementation retained below only as a reference for old
        # checkpoint compatibility; the return above ensures new runs use the
        # scoped factual-premise auditor.
        """Judge sufficiency against auditable graph evidence.

        Empty graphs, missing canonical citations, and judge failures are
        fail-closed. They produce a bounded, searchable coverage gap instead
        of silently allowing an unsupported plan to pass.
        """
        graph = state.evidence_graph
        graph_context = build_graph_context(state)
        context_pack = ContextPlanner().plan(
            graph_context,
            ContextRequest(
                purpose="m6_sufficiency",
                focus_evidence_ids=tuple(dict.fromkeys([
                    *hypothesis.supporting_evidence,
                    *plan.supporting_evidence_ids,
                ])),
            ),
        )
        emit_context_built(
            "m6",
            "evidence_sufficiency_judge",
            context_pack,
        )
        started_at = time.monotonic()
        emit_event(
            "tool_started",
            module="m6",
            tool="evidence_sufficiency_judge",
            status="running",
            message="Evidence sufficiency verdict started",
            details={"version": version},
        )
        try:
            plan_summary = {
                "hypothesis_id": plan.hypothesis_id,
                "study_subjects": plan.study_subjects,
                "procedures": plan.procedures[:5],
                "measurement_metrics": plan.measurement_metrics[:5],
            }
            payload = await asyncio.wait_for(
                self.client.structured_chat(
                    system_prompt=M6_EVIDENCE_VERDICT_SYSTEM,
                    user_prompt=M6_EVIDENCE_VERDICT_TEMPLATE.format(
                        original_question=state.input_question,
                        facts_count=len(graph.established_facts) if graph else 0,
                        conflicts_count=len(graph.conflicts) if graph else 0,
                        gaps_count=len(graph.knowledge_gaps) if graph else 0,
                        graph_context=context_pack.rendered,
                        hypothesis_json=json.dumps(
                            hypothesis.model_dump(mode="json"), ensure_ascii=False, indent=2
                        ),
                        plan_summary=json.dumps(plan_summary, ensure_ascii=False, indent=2),
                        version=version,
                    ),
                    output_schema=EvidenceSufficiencyVerdict.model_json_schema(),
                    max_tokens=4096,
                    temperature=getattr(self.llm_config, "temperature", 0.1),
                    disable_thinking=True,
                ),
                timeout=self.reviewer_timeout_seconds,
            )
            if not payload:
                raise ValueError("empty evidence-sufficiency payload")
            verdict = EvidenceSufficiencyVerdict.model_validate(dict(payload))
        except Exception as exc:
            logger.warning("Evidence-sufficiency judge failed closed: %s", exc)
            emit_event(
                "tool_failed",
                module="m6",
                tool="evidence_sufficiency_judge",
                status="failed",
                message=f"Evidence verdict failed (fail-closed): {type(exc).__name__}: {exc}",
                elapsed_seconds=time.monotonic() - started_at,
                details={"version": version},
            )
            return EvidenceSufficiencyVerdict(
                sufficient=False,
                gaps=[self._coverage_gap(state, version)],
                rationale=f"Evidence judge failed: {type(exc).__name__}: {exc}",
            )

        valid_evidence_ids = set(graph_context.available_evidence_ids)
        verdict.evidence_ids = list(dict.fromkeys(
            evidence_id for evidence_id in verdict.evidence_ids
            if evidence_id in valid_evidence_ids
        ))
        if verdict.sufficient and not verdict.evidence_ids:
            verdict.sufficient = False
            verdict.rationale = " ".join(filter(None, [
                verdict.rationale,
                "No canonical supporting evidence ID could be verified.",
            ]))
        if not verdict.sufficient and not verdict.gaps:
            verdict.gaps = [self._coverage_gap(state, version)]

        # Normalise gaps CODE-SIDE: gap_id is always recomputed from
        # (target_sub_question, gap_type, canonical_entities) — any id the LLM
        # emitted is discarded.  Falls back to the description when the
        # sub-question is missing so derivation never crashes.
        for gap in verdict.gaps:
            anchor = gap.target_sub_question or gap.description
            gap.gap_id = make_gap_id(anchor, gap.gap_type, gap.canonical_entities)
            gap.source_review_version = version
            gap.hypothesis_ids = [hypothesis.hypothesis_id]
            if gap.status not in ("open", "pending_grounding"):
                gap.status = "open"  # freshly reported gaps start open
        emit_event(
            "tool_completed",
            module="m6",
            tool="evidence_sufficiency_judge",
            status="completed",
            message=(
                f"Evidence verdict: {'sufficient' if verdict.sufficient else 'insufficient'}"
                f", {len(verdict.gaps)} gap(s)"
            ),
            elapsed_seconds=time.monotonic() - started_at,
            details={
                "version": version,
                "sufficient": verdict.sufficient,
                "gap_ids": [g.gap_id for g in verdict.gaps],
            },
        )
        return verdict

    @staticmethod
    def _coverage_gap(state: PipelineState, version: int) -> EvidenceGap:
        """Create a domain-neutral, searchable gap for missing provenance."""

        card = state.problem_card
        target = (
            card.original_question
            if card is not None and card.original_question
            else state.input_question
        )
        entities: List[str] = []
        if card is not None:
            required = [
                entity.name for entity in card.task_contract.entities
                if entity.required and entity.name
            ]
            entities = list(dict.fromkeys([*required, *card.key_entities]))[:6]
        query = " ".join([*entities[:3], target]).strip()
        return EvidenceGap(
            description=f"Auditable literature evidence is missing for: {target}",
            gap_type="coverage",
            canonical_entities=entities,
            target_sub_question=target,
            suggested_queries=[query or target],
            source_review_version=version,
            status="open",
            hypothesis_ids=[
                item.hypothesis_id for item in state.top_hypotheses
                if item.hypothesis_id
            ],
            scientific_resolution="unreviewed",
            search_completed=False,
        )

    @staticmethod
    def _merge_evidence_gaps(
        existing: List[EvidenceGap],
        incoming: List[EvidenceGap],
        version: int,
        sufficient: bool = False,
        gap_gain: Optional[Dict[str, int]] = None,
    ) -> List[EvidenceGap]:
        """Merge new verdict gaps into the ledger, matched by ``gap_id``.

        Semantics (v2 contract):

        * **Hit** — inherit the ledger entry's status; ``attempts`` +1 (the
          gap survived another review round — a *review-survival* counter;
          the single source for *zero-gain* attempt increments is M3's
          post-grounding ``_evaluate_pending_gaps``), refresh
          ``source_review_version`` and backfill empty
          ``suggested_queries`` / ``target_sub_question``.
        * **New gap** — recorded with status ``open``.
        * **Disappeared** — an ``open`` gap not re-reported this round is
          ``closed`` when the verdict says the evidence is now
          ``sufficient`` **or** this round's M3 gain for that gap is > 0
          (the new evidence covered it even though the verdict stayed
          conservative).
        * ``closed`` / ``unimprovable`` gaps are never re-opened.
        """
        gain_map = gap_gain or {}
        merged: Dict[str, EvidenceGap] = {
            g.gap_id: g.model_copy(deep=True) for g in existing
        }
        incoming_ids = set()
        for gap in incoming:
            incoming_ids.add(gap.gap_id)
            current = merged.get(gap.gap_id)
            if current is None:
                fresh = gap.model_copy(deep=True)
                fresh.status = "open"
                merged[gap.gap_id] = fresh
                continue
            if current.status in ("closed", "unimprovable"):
                continue  # resolved gaps stay resolved
            current.attempts += 1  # survived one more review round
            current.source_review_version = version
            if gap.hypothesis_ids:
                current.hypothesis_ids = list(dict.fromkeys(gap.hypothesis_ids))
            if gap.source_claim_type:
                current.source_claim_type = gap.source_claim_type
            if gap.source_claim_id:
                current.source_claim_id = gap.source_claim_id
            current.scientific_resolution = gap.scientific_resolution
            current.search_completed = gap.search_completed
            current.technical_errors = list(gap.technical_errors)
            if gap.suggested_queries and not current.suggested_queries:
                current.suggested_queries = list(gap.suggested_queries)
            if gap.target_sub_question and not current.target_sub_question:
                current.target_sub_question = gap.target_sub_question

        for gap in merged.values():
            if gap.status != "open" or gap.gap_id in incoming_ids:
                continue
            covered_by_evidence = int(gain_map.get(gap.gap_id, 0) or 0) > 0
            if sufficient or covered_by_evidence:
                gap.status = "closed"
        return list(merged.values())

    def _apply_zero_gain_convergence(
        self,
        state: PipelineState,
        gaps: List[EvidenceGap],
    ) -> tuple[List[EvidenceGap], Dict[str, Any]]:
        """Convergence protection over consecutive all-zero-gain rounds.

        Reads this round's ``metrics["m3_gap_gain"]`` (written by M3) and
        tracks a ``m6_zero_gain_streak`` counter in ``metrics``:

        * every gap gained 0 new effective evidence → streak + 1;
        * any positive gain (or no gain map this round) → streak resets;
        * streak ≥ ``gap_no_gain_limit`` → every still ``open`` /
          ``pending_grounding`` gap is marked ``unimprovable`` (gap-level
          marker; routing rule 2 then finds no open gap so supplement
          search stops — the ``search_round`` cap remains the backstop).
        """
        gain = state.metrics.get("m3_gap_gain")
        previous_streak = int(state.metrics.get("m6_zero_gain_streak", 0) or 0)
        if (
            isinstance(gain, dict)
            and gain
            and all(int(value or 0) == 0 for value in gain.values())
        ):
            streak = previous_streak + 1
        else:
            streak = 0
        metrics = {**state.metrics, "m6_zero_gain_streak": streak}

        if streak >= self.gap_no_gain_limit:
            marked: List[str] = []
            for gap in gaps:
                if gap.status in ("open", "pending_grounding"):
                    gap.status = "unimprovable"
                    marked.append(gap.gap_id)
            if marked:
                emit_event(
                    "evidence_gap_status_changed",
                    module="m6",
                    status="completed",
                    message=(
                        f"{streak} consecutive round(s) with zero evidence gain across "
                        f"all gaps; {len(marked)} gap(s) marked unimprovable; supplement search stopped"
                    ),
                    details={
                        "gap_ids": marked,
                        "streak": streak,
                        "limit": self.gap_no_gain_limit,
                        "status": "unimprovable",
                    },
                )
        return gaps, metrics

    @classmethod
    def get_input_fields(cls) -> List[str]:
        return [
            "top_hypotheses", "research_plans", "evidence_graph", "iteration_count",
            "problem_card", "grounding_report", "literature_results",
            "m2_knowledge_export",
        ]

    @classmethod
    def get_output_fields(cls) -> List[str]:
        return [
            "reviews", "iteration_count", "top_hypotheses", "research_plans",
            "experimental_validation_verdict",
        ]
