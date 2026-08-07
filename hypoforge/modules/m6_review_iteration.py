"""
M6: Review & Iterative Refinement.

Three specialist reviewer agents (scientific_logic / evidence_consistency /
method_feasibility) score the top hypothesis + research plan on a 1–5 scale;
each reasons *before* it scores (rubric-anchored).  The ``overall`` score is
computed as the mean of the specialist scores.

Output: ``reviews`` appended; ``iteration_count`` incremented.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

from ..observability import emit_event
from ..protocol import ModuleProtocol
from ..prompts.m6_prompts import (
    M6_EVIDENCE_VERDICT_SYSTEM,
    M6_EVIDENCE_VERDICT_TEMPLATE,
    M6_REASON_FIRST,
    M6_REVIEWER_PROMPTS,
    M6_USER_TEMPLATE,
)
from ..evaluation.rubric import review_rubric_line
from ..registry import ModuleRegistry
from ..state import (
    EvidenceGap,
    EvidenceSufficiencyVerdict,
    PipelineState,
    ReviewResult,
    ReviewerDimension,
    make_gap_id,
)
from ..tools.qwen_client import QwenClient

logger = logging.getLogger(__name__)


@ModuleRegistry.register
class M6ReviewIteration(ModuleProtocol):
    module_name = "m6"
    module_version = "0.1.0"
    description = "Multi-reviewer assessment + iterative refinement decision"

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
        **kwargs,
    ):
        self.reviewer_dims = reviewers or [
            "scientific_logic",
            "evidence_consistency",
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
        self.client = QwenClient.from_config(llm_config) if llm_config else None

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

        # Nothing to review yet — just advance the iteration counter.
        if not (state.top_hypotheses and state.research_plans):
            return {"iteration_count": version}

        hypothesis = state.top_hypotheses[0]
        plan = next(
            (p for p in state.research_plans if p.hypothesis_id == hypothesis.hypothesis_id),
            state.research_plans[0],
        )
        graph = state.evidence_graph

        # Specialist reviewers (LLM); "overall" is computed, not queried.
        specialist_dims = [d for d in self.reviewer_dims if d != "overall"]
        new_reviews: List[ReviewResult] = []

        for dim in specialist_dims:
            # Anchor the score (rubric) and force reason-before-score, both
            # sourced from the single rubric definition.
            system_prompt = "\n\n".join(
                p for p in (M6_REVIEWER_PROMPTS[dim], review_rubric_line(dim), M6_REASON_FIRST) if p
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
            payload = await self.client.structured_chat(
                system_prompt=system_prompt,
                user_prompt=M6_USER_TEMPLATE.format(
                    original_question=state.input_question,
                    hypothesis_json=json.dumps(hypothesis.model_dump(mode="json"), ensure_ascii=False, indent=2),
                    plan_json=json.dumps(plan.model_dump(mode="json"), ensure_ascii=False, indent=2),
                    facts_count=len(graph.established_facts) if graph else 0,
                    conflicts_count=len(graph.conflicts) if graph else 0,
                    gaps_count=len(graph.knowledge_gaps) if graph else 0,
                ),
                output_schema=ReviewResult.model_json_schema(),
                max_tokens=8192,
                temperature=getattr(self.llm_config, "temperature", 0.1),
            )
            payload = dict(payload)
            payload["dimension"] = dim
            payload["version"] = version
            review = ReviewResult.model_validate(payload)
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
            new_reviews.append(ReviewResult(
                dimension=ReviewerDimension("overall"),
                reasoning="Computed as the mean of the specialist reviews.",
                score=round(avg, 1),
                comments="Computed from specialist reviews (scientific_logic, evidence_consistency, method_feasibility).",
                suggestions="See individual dimension reviews for detailed suggestions.",
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

        patch: Dict[str, Any] = {
            "reviews": state.reviews + new_reviews,
            "iteration_count": version,
        }

        # --- iteration core: evidence-sufficiency verdict (fail-open) ---
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
        """One structured LLM call judging evidence sufficiency.

        Fail-open by design: any failure / parse error / empty output yields
        ``sufficient=True, gaps=[]`` so the main flow is never blocked.
        """
        graph = state.evidence_graph
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
            payload = await self.client.structured_chat(
                system_prompt=M6_EVIDENCE_VERDICT_SYSTEM,
                user_prompt=M6_EVIDENCE_VERDICT_TEMPLATE.format(
                    original_question=state.input_question,
                    facts_count=len(graph.established_facts) if graph else 0,
                    conflicts_count=len(graph.conflicts) if graph else 0,
                    gaps_count=len(graph.knowledge_gaps) if graph else 0,
                    hypothesis_json=json.dumps(
                        hypothesis.model_dump(mode="json"), ensure_ascii=False, indent=2
                    ),
                    plan_summary=json.dumps(plan_summary, ensure_ascii=False, indent=2),
                    version=version,
                ),
                output_schema=EvidenceSufficiencyVerdict.model_json_schema(),
                max_tokens=4096,
                temperature=getattr(self.llm_config, "temperature", 0.1),
            )
            if not payload:
                raise ValueError("empty evidence-sufficiency payload")
            verdict = EvidenceSufficiencyVerdict.model_validate(dict(payload))
        except Exception as exc:  # fail-open: never block the main flow
            logger.warning("Evidence-sufficiency judge failed open: %s", exc)
            emit_event(
                "tool_failed",
                module="m6",
                tool="evidence_sufficiency_judge",
                status="failed",
                message=f"证据裁决失败（fail-open）：{type(exc).__name__}: {exc}",
                elapsed_seconds=time.monotonic() - started_at,
                details={"version": version},
            )
            return EvidenceSufficiencyVerdict(sufficient=True)

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
        return ["top_hypotheses", "research_plans", "evidence_graph", "iteration_count"]

    @classmethod
    def get_output_fields(cls) -> List[str]:
        return ["reviews", "iteration_count"]
