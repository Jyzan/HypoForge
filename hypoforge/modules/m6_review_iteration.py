"""
M6: Review & Iterative Refinement.

Three specialist reviewer agents (scientific_logic / evidence_consistency /
method_feasibility) score the top hypothesis + research plan on a 1–5 scale;
each reasons *before* it scores (rubric-anchored).  The ``overall`` score is
computed as the mean of the specialist scores.

Output: ``reviews`` appended; ``iteration_count`` incremented.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from typing import Any, Dict, List, Optional

from ..observability import emit_event
from ..graph_context import build_graph_context
from ..protocol import ModuleProtocol
from ..prompts.m6_prompts import (
    M6_EVIDENCE_VERDICT_SYSTEM,
    M6_EVIDENCE_VERDICT_TEMPLATE,
    M6_FORMAT_NOTE,
    M6_REASON_FIRST,
    M6_REVIEWER_PROMPTS,
    M6_USER_TEMPLATE,
)
from ..evaluation.rubric import review_rubric_line
from ..evaluation.scorer import _quality_gates, score_hypothesis_async, _collect_knowledge_entries
from ..evaluation.metrics import MetricRegistry
from ..registry import ModuleRegistry
from ..state import (
    EvidenceGap,
    EvidenceSufficiencyVerdict,
    GraphCorrectionRequest,
    PipelineState,
    ReviewResult,
    ReviewerDimension,
    make_gap_id,
)
from ..task_alignment import assess_task_alignment
from ..tools.qwen_client import QwenClient

logger = logging.getLogger(__name__)


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
        **kwargs,
    ):
        if semantic_alignment_timeout_seconds <= 0:
            raise ValueError(
                "semantic_alignment_timeout_seconds must be positive"
            )
        if reviewer_timeout_seconds <= 0:
            raise ValueError("reviewer_timeout_seconds must be positive")
        self.reviewer_dims = reviewers or [
            "scientific_logic",
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
            if entity.role == "primary_object" and entity.required
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
            message="M6 任务对象语义复核开始",
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
                message="M6 任务对象语义复核超时",
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
                "M6 任务对象语义复核通过"
                if consistent else "M6 任务对象语义复核未通过"
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

        hypothesis = state.top_hypotheses[0]
        plan = next(
            (p for p in state.research_plans if p.hypothesis_id == hypothesis.hypothesis_id),
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
        rendered_graph_context = graph_context.render()
        valid_evidence_ids = set(graph_context.available_evidence_ids)
        hypothesis_alignment = assess_task_alignment(
            state,
            hypothesis.model_dump_json(exclude={"task_trace"}),
            subject_text="\n".join([hypothesis.statement, hypothesis.mechanism]),
            trace=hypothesis.task_trace,
            semantic_client=None,
        )
        plan_alignment = assess_task_alignment(
            state,
            plan.model_dump_json(exclude={"task_trace"}),
            subject_text=plan.study_subjects,
            trace=plan.task_trace,
            semantic_client=None,
        )
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

        for dim in specialist_dims:
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
                message=f"评审 Agent 开始：{dim}",
                details={"version": version},
            )
            try:
                payload = await asyncio.wait_for(
                    self.client.structured_chat(
                        system_prompt=system_prompt,
                        user_prompt=M6_USER_TEMPLATE.format(
                            original_question=state.input_question,
                            problem_card_json=(
                                state.problem_card.model_dump_json(indent=2)
                                if state.problem_card else "{}"
                            ),
                            hypothesis_json=json.dumps(hypothesis.model_dump(mode="json"), ensure_ascii=False, indent=2),
                            plan_json=json.dumps(plan.model_dump(mode="json"), ensure_ascii=False, indent=2),
                            graph_context=rendered_graph_context,
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
            if dim in {"scientific_logic", "testability", "novelty"}:
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
            updates: Dict[str, Any] = {"evidence_ids": cited_evidence}
            updates["graph_correction_requests"] = (
                self._normalise_graph_corrections(
                    review,
                    valid_evidence_ids=valid_evidence_ids,
                    version=version,
                )
            )
            review = review.model_copy(update=updates)
            new_reviews.append(review)
            emit_event(
                "tool_completed",
                module="m6",
                tool=f"reviewer:{dim}",
                status="completed",
                message=f"评审 Agent 完成：{dim}，评分 {review.score:.1f}/5",
                elapsed_seconds=time.monotonic() - started_at,
                details={"version": version, "score": review.score},
            )

        # --- NEW: Objective Quality Gates & Metrics ---
        try:
            gates = _quality_gates(state)
            
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

            # 4. Metrics
            knowledge_entries = _collect_knowledge_entries(state)
            # score_hypothesis_async runs all implemented independent metrics
            metric_report = await score_hypothesis_async(
                hypothesis,
                knowledge_entries,
                llm_config=getattr(config, "llm", None),
                embed_config=getattr(config, "embedding", None),
                evidence_graph=state.evidence_graph
            )
            independent_metrics = metric_report.get("independent", {})
            for m_name, m_score in independent_metrics.items():
                m_score_scaled = m_score * 5.0
                threshold = 1.5 if m_name == "novelty" else 2.5
                m_passed = m_score_scaled >= threshold
                dim_name = f"{m_name}_metric"
                if m_name == "evidence_consistency":
                    dim_name = "objective_evidence_consistency"
                    
                m_attr = "hypothesis"
                    
                new_reviews.append(ReviewResult(
                    dimension=ReviewerDimension(dim_name),
                    attribution=m_attr,
                    reasoning=f"Calculated {m_name} score is {m_score:.2f} (scaled to {m_score_scaled:.1f}/5).",
                    score=round(m_score_scaled, 1),
                    comments=f"Objective metric {m_name} from MetricRegistry.",
                    suggestions="" if m_passed else f"Improve {m_name} to meet the minimum threshold of {threshold}/5.",
                    hard_gate_passed=m_passed,
                    version=version,
                ))
        except Exception as e:
            logger.warning(f"Failed to compute objective quality gates/metrics: {e}")

        # Compute overall as the mean of the specialist scores.
        if "overall" in self.reviewer_dims and new_reviews:
            overall_started_at = time.monotonic()
            emit_event(
                "tool_started",
                module="m6",
                tool="overall_score_aggregator",
                status="running",
                message="开始汇总总体评分",
                details={"version": version},
            )
            specialist_scores = [r.score for r in new_reviews if r.score > 0]
            avg = sum(specialist_scores) / len(specialist_scores) if specialist_scores else 3.0
            failed_gates = [
                review.dimension.value
                for review in new_reviews
                if review.hard_gate_passed is False
            ]
            if failed_gates:
                avg = min(avg, 1.9 if "task_alignment" in failed_gates else 2.9)
            new_reviews.append(ReviewResult(
                dimension=ReviewerDimension("overall"),
                reasoning=(
                    "Computed from specialist reviews; hard-gate failures: "
                    + (", ".join(failed_gates) if failed_gates else "none")
                ),
                score=round(avg, 1),
                comments="Computed from specialist reviews and capped when task/evidence hard gates fail.",
                suggestions="See individual dimension reviews for detailed suggestions.",
                evidence_ids=list(dict.fromkeys(
                    evidence_id
                    for review in new_reviews
                    for evidence_id in review.evidence_ids
                )),
                hard_gate_passed=not failed_gates,
                version=version,
            ))
            emit_event(
                "tool_completed",
                module="m6",
                tool="overall_score_aggregator",
                status="completed",
                message=f"总体评分 {avg:.1f}/5",
                elapsed_seconds=time.monotonic() - overall_started_at,
                details={"version": version, "score": round(avg, 1)},
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
            "iteration_count": version,
            "graph_correction_requests": list(correction_by_id.values()),
        }

        # --- iteration core: evidence-sufficiency verdict (fail-closed) ---
        # Exactly ONE extra structured call, only when the switch is on.
        if self.m6_evidence_revisit:
            verdict = await self._judge_evidence_sufficiency(
                state, hypothesis, plan, version
            )
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
    # Evidence sufficiency (iteration core)
    # ------------------------------------------------------------------

    async def _judge_evidence_sufficiency(
        self,
        state: PipelineState,
        hypothesis: Any,
        plan: Any,
        version: int,
    ) -> EvidenceSufficiencyVerdict:
        """Judge sufficiency against auditable graph evidence.

        Empty graphs, missing canonical citations, and judge failures are
        fail-closed. They produce a bounded, searchable coverage gap instead
        of silently allowing an unsupported plan to pass.
        """
        graph = state.evidence_graph
        graph_context = build_graph_context(state)
        started_at = time.monotonic()
        emit_event(
            "tool_started",
            module="m6",
            tool="evidence_sufficiency_judge",
            status="running",
            message="证据充足性裁决开始",
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
                        graph_context=graph_context.render(),
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
                message=f"证据裁决失败（fail-closed）：{type(exc).__name__}: {exc}",
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
            if gap.status not in ("open", "pending_grounding"):
                gap.status = "open"  # freshly reported gaps start open
        emit_event(
            "tool_completed",
            module="m6",
            tool="evidence_sufficiency_judge",
            status="completed",
            message=(
                f"证据裁决：{'sufficient' if verdict.sufficient else 'insufficient'}"
                f"，{len(verdict.gaps)} 个 gap"
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
            card.sub_questions[0]
            if card is not None and card.sub_questions
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
                        f"连续 {streak} 轮全部缺口零证据增益，"
                        f"{len(marked)} 个缺口置为 unimprovable，停止补搜"
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
        return ["reviews", "iteration_count"]
