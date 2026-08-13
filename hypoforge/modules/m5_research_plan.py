"""
M5: Research Plan Design.

For each top hypothesis, generates a detailed experimental research plan
covering subjects, variables, controls, procedures, metrics, analysis,
and success/failure criteria.

Output: ``research_plans`` in state.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional

from ..observability import emit_event
from ..graph_context import GraphContext, build_graph_context
from ..protocol import ModuleProtocol
from ..prompts.m5_prompts import M5_SYSTEM_PROMPT, M5_USER_TEMPLATE
from ..registry import ModuleRegistry
from ..state import (
    PipelineState,
    ResearchPlan,
    TaskTrace,
    TaskTraceReference,
)
from ..task_alignment import _mentions_entity, assess_task_alignment
from ..tools.qwen_client import QwenClient

logger = logging.getLogger(__name__)


def _followup_requirement_block(state: PipelineState) -> str:
    """Format ``state.followup.text`` as a priority requirement block.

    Returns ``""`` on non-followup runs, so prompt construction stays
    byte-for-byte identical to the legacy behaviour when no followup exists.
    """
    followup = getattr(state, "followup", None)
    text = (followup.text or "").strip() if followup is not None else ""
    if not text:
        return ""
    return (
        "\n--- 用户追问要求 (user follow-up instruction, highest priority, must be honoured) ---\n"
        f"{text}\n"
    )


@ModuleRegistry.register
class M5ResearchPlan(ModuleProtocol):
    module_name = "m5"
    module_version = "0.1.0"
    description = "Generate detailed experimental research plans for top hypotheses"

    # ------------------------------------------------------------------
    # ModuleProtocol implementation
    # ------------------------------------------------------------------

    def __init__(
        self,
        mode: str = "llm",
        llm_config: Optional[Any] = None,
        generation_timeout_seconds: float = 180.0,
        semantic_alignment_timeout_seconds: float = 60.0,
        **kwargs,
    ):
        if generation_timeout_seconds <= 0:
            raise ValueError("generation_timeout_seconds must be positive")
        if semantic_alignment_timeout_seconds <= 0:
            raise ValueError(
                "semantic_alignment_timeout_seconds must be positive"
            )
        self.mode = mode
        self.llm_config = llm_config
        self.generation_timeout_seconds = float(generation_timeout_seconds)
        self.semantic_alignment_timeout_seconds = float(
            semantic_alignment_timeout_seconds
        )
        self.client = QwenClient.from_config(llm_config) if llm_config else None

    @staticmethod
    def _build_feedback_context(state: PipelineState) -> str:
        """Route the latest method-feasibility review to plan revision.

        Followup runs additionally inject ``state.followup.text`` so user
        instructions (e.g. "输出中文方案") shape plan generation; on
        non-followup runs the output is unchanged."""
        followup_block = _followup_requirement_block(state)
        if state.iteration_count <= 0 or not state.reviews:
            return followup_block
        latest = max(review.version for review in state.reviews)
        feedback = []
        for review in state.reviews:
            if review.version != latest:
                continue
            attr = getattr(review, "attribution", "both")
            if attr in ("plan", "both") and review.hard_gate_passed is False:
                msg = review.suggestions or review.comments or ""
                if msg.strip():
                    feedback.append(f"[{review.dimension.value}] {msg.strip()}")
        if not feedback:
            return followup_block
        lines = [
            "",
            "Research-plan revision guidance from the latest review iteration:",
            *[f"- {item}" for item in feedback],
            "Apply only suggestions relevant to this hypothesis and keep them in the research plan.",
        ]
        return followup_block + "\n".join(lines)

    @staticmethod
    def _sanitize_evidence_links(plan: ResearchPlan, context: GraphContext) -> ResearchPlan:
        """Keep only canonical M2 evidence/paper IDs and fail unsupported links closed."""

        valid_evidence = set(context.available_evidence_ids)
        valid_papers = set(context.available_paper_ids)
        evidence_to_paper = context.evidence_to_paper
        evidence_ids = list(dict.fromkeys(
            identifier
            for identifier in plan.supporting_evidence_ids
            if identifier in valid_evidence
        ))
        paper_ids = list(dict.fromkeys(
            identifier
            for identifier in plan.source_paper_ids
            if identifier in valid_papers
        ))
        links = []
        for link in plan.evidence_links:
            link_evidence = list(dict.fromkeys(
                identifier
                for identifier in link.supporting_evidence_ids
                if identifier in valid_evidence
            ))
            link_papers = list(dict.fromkeys(
                identifier
                for identifier in link.source_paper_ids
                if identifier in valid_papers
            ))
            status = link.support_status
            if status == "supported" and (not link_evidence or not link_papers):
                status = "hypothesis_to_validate"
            elif status == "supported":
                # Verify provenance: each evidence must belong to at least
                # one of the cited papers (not just be in two independent
                # whitelists).
                linked_paper_set = set(link_papers)
                paired = any(
                    evidence_to_paper.get(ev_id, "") in linked_paper_set
                    for ev_id in link_evidence
                )
                if not paired:
                    status = "hypothesis_to_validate"
            links.append(link.model_copy(update={
                "supporting_evidence_ids": link_evidence,
                "source_paper_ids": link_papers,
                "support_status": status,
            }))
            evidence_ids.extend(link_evidence)
            paper_ids.extend(link_papers)
        return plan.model_copy(update={
            "supporting_evidence_ids": list(dict.fromkeys(evidence_ids)),
            "source_paper_ids": list(dict.fromkeys(paper_ids)),
            "evidence_links": links,
        })

    @staticmethod
    def _plan_segments(plan: ResearchPlan) -> List[str]:
        values = [
            plan.study_subjects,
            *plan.independent_variables,
            *plan.dependent_variables,
            *plan.control_groups,
            *plan.procedures,
            *plan.measurement_metrics,
            *plan.analysis_methods,
            plan.expected_results_if_supported,
            plan.expected_results_if_refuted,
            plan.timeline,
            plan.risks_and_alternatives,
            *(link.plan_element for link in plan.evidence_links),
            *(link.claim for link in plan.evidence_links),
        ]
        return list(dict.fromkeys(
            str(value or "").strip()
            for value in values
            if str(value or "").strip()
        ))

    @staticmethod
    def _plan_alignment_text(plan: ResearchPlan) -> str:
        return "\n".join(M5ResearchPlan._plan_segments(plan))

    @staticmethod
    def _canonicalize_task_trace(
        state: PipelineState,
        plan: ResearchPlan,
    ) -> ResearchPlan:
        """Derive literal trace excerpts from plan content without inventing text."""

        problem_card = state.problem_card
        if problem_card is None:
            return plan
        contract = problem_card.task_contract
        if not contract.entities and not contract.requirements:
            return plan

        segments = M5ResearchPlan._plan_segments(plan)
        entity_by_id = {
            entity.entity_id: entity for entity in contract.entities
        }
        entity_mentions: List[TaskTraceReference] = []
        for entity in contract.entities:
            excerpt = next(
                (
                    segment for segment in segments
                    if _mentions_entity(segment, entity)
                ),
                "",
            )
            if excerpt:
                entity_mentions.append(TaskTraceReference(
                    contract_id=entity.entity_id,
                    output_excerpt=excerpt,
                ))

        requirement_mentions: List[TaskTraceReference] = []
        for requirement in contract.requirements:
            primary = entity_by_id.get(requirement.primary_entity_id)
            if primary is None:
                continue
            relevant_entities = [
                entity_by_id[entity_id]
                for entity_id in [
                    requirement.primary_entity_id,
                    *requirement.related_entity_ids,
                ]
                if entity_id in entity_by_id
            ]
            candidates = [
                segment for segment in segments
                if _mentions_entity(segment, primary)
            ]
            if not candidates:
                continue
            excerpt = max(
                candidates,
                key=lambda segment: (
                    sum(
                        _mentions_entity(segment, entity)
                        for entity in relevant_entities
                    ),
                    -segments.index(segment),
                ),
            )
            requirement_mentions.append(TaskTraceReference(
                contract_id=requirement.requirement_id,
                output_excerpt=excerpt,
            ))

        return plan.model_copy(update={
            "task_trace": TaskTrace(
                entity_mentions=entity_mentions,
                requirement_mentions=requirement_mentions,
            ),
        })

    async def _audit_plan_semantics(
        self,
        state: PipelineState,
        plan: ResearchPlan,
    ) -> tuple[bool, str]:
        """Check the plan's research object without blocking the event loop."""

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
            "Judge whether the research plan studies the same type of primary "
            "object required by the original task. Judge meaning, not mere word "
            "overlap. Answer ONLY yes or no, followed by one short reason.\n\n"
            "Primary task objects:\n"
            + json.dumps(primary_objects, ensure_ascii=False, indent=2)
            + "\n\nPlan study subjects:\n"
            + plan.study_subjects
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
            raise RuntimeError(
                "M5 semantic task-contract audit timed out after "
                f"{self.semantic_alignment_timeout_seconds:g} seconds"
            ) from exc
        answer = str(response or "").strip()
        return answer.casefold().startswith("yes"), answer

    async def __call__(
        self,
        state: PipelineState,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if self.client is None:
            raise RuntimeError(
                "M5 requires an LLM client — pass llm_config / set OPENAI_API_KEY."
            )

        plans: List[ResearchPlan] = []
        graph_context = build_graph_context(state)
        rendered_graph_context = graph_context.render()
        problem_card_json = (
            state.problem_card.model_dump_json(indent=2)
            if state.problem_card
            else "{}"
        )
        for h in state.top_hypotheses:
            started_at = time.monotonic()
            emit_event(
                "tool_started",
                module="m5",
                tool="research_plan_designer",
                status="running",
                message=f"开始为假设 {h.hypothesis_id} 设计研究方案",
                details={"hypothesis_id": h.hypothesis_id},
            )
            alignment_feedback = ""
            plan: ResearchPlan | None = None
            plan_accepted = False
            last_alignment_rationale = "plan did not pass task alignment"
            for attempt in range(2):
                try:
                    payload = await asyncio.wait_for(
                        self.client.structured_chat(
                            system_prompt=M5_SYSTEM_PROMPT,
                            user_prompt=M5_USER_TEMPLATE.format(
                                original_question=(
                                    state.problem_card.original_question
                                    if state.problem_card else state.input_question
                                ),
                                problem_card_json=problem_card_json,
                                graph_context=rendered_graph_context,
                                statement=h.statement,
                                mechanism=h.mechanism,
                                predictions="\n".join(h.observable_predictions),
                                falsification_conditions="\n".join(h.falsification_conditions),
                                hypothesis_evidence="\n".join(h.supporting_evidence) or "(none)",
                                feedback_context=(
                                    self._build_feedback_context(state)
                                    + alignment_feedback
                                ),
                            ),
                            output_schema=ResearchPlan.model_json_schema(),
                            max_tokens=16384,
                            temperature=getattr(self.llm_config, "temperature", 0.1),
                            disable_thinking=True,
                        ),
                        timeout=self.generation_timeout_seconds,
                    )
                except asyncio.TimeoutError as exc:
                    if attempt == 0:
                        emit_event(
                            "tool_retrying",
                            module="m5",
                            tool="research_plan_designer",
                            status="retrying",
                            message=(
                                f"假设 {h.hypothesis_id} 的方案生成超时，正在重试"
                            ),
                            details={
                                "hypothesis_id": h.hypothesis_id,
                                "timeout_seconds": self.generation_timeout_seconds,
                            },
                        )
                        continue
                    raise RuntimeError(
                        "M5 research-plan generation timed out after "
                        f"{self.generation_timeout_seconds:g} seconds"
                    ) from exc
                plan = ResearchPlan.model_validate(payload).model_copy(
                    update={"hypothesis_id": h.hypothesis_id}
                )
                plan = self._sanitize_evidence_links(plan, graph_context)
                plan = self._canonicalize_task_trace(state, plan)
                plan_text = self._plan_alignment_text(plan)
                alignment = assess_task_alignment(
                    state,
                    plan_text,
                    subject_text=plan.study_subjects,
                    trace=plan.task_trace,
                    semantic_client=None,
                )
                if alignment.passed:
                    semantic_consistent, semantic_rationale = (
                        await self._audit_plan_semantics(state, plan)
                    )
                    if semantic_consistent:
                        plan_accepted = True
                        break
                    last_alignment_rationale = (
                        "semantic object mismatch: " + semantic_rationale
                    )
                else:
                    last_alignment_rationale = alignment.rationale
                alignment_feedback = (
                    "\nHARD ALIGNMENT RETRY: The previous plan was rejected because "
                    f"{last_alignment_rationale} Rewrite it for the original research object "
                    "and return valid task-entity and requirement traces from the "
                    "binding M1 task contract.\n"
                )
            assert plan is not None
            if not plan_accepted:
                raise ValueError(
                    f"M5 blocked task-misaligned plan for {h.hypothesis_id}: "
                    f"{last_alignment_rationale}"
                )
            plans.append(plan)
            emit_event(
                "tool_completed",
                module="m5",
                tool="research_plan_designer",
                status="completed",
                message=f"假设 {h.hypothesis_id} 的研究方案完成",
                elapsed_seconds=time.monotonic() - started_at,
                details={
                    "hypothesis_id": h.hypothesis_id,
                    "procedures": len(plan.procedures),
                    "analysis_methods": len(plan.analysis_methods),
                },
            )

        history = dict(state.research_plan_history)
        history[state.iteration_count + 1] = [plan.model_copy(deep=True) for plan in plans]
        return {"research_plans": plans, "research_plan_history": history}

    @classmethod
    def get_input_fields(cls) -> List[str]:
        return [
            "top_hypotheses", "reviews", "iteration_count", "problem_card",
            "evidence_graph", "grounding_report", "literature_results",
            "m2_knowledge_export",
        ]

    @classmethod
    def get_output_fields(cls) -> List[str]:
        return ["research_plans", "research_plan_history"]
