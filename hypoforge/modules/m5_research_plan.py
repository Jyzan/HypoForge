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
from ..context import ContextPlanner, ContextRequest, emit_context_built
from ..experimental_validation import ExperimentalValidationAuditor
from ..graph_context import GraphContext, build_graph_context
from ..protocol import ModuleProtocol
from ..prompts.m5_prompts import M5_SYSTEM_PROMPT, M5_USER_TEMPLATE
from ..registry import ModuleRegistry
from ..state import (
    ExperimentalValidationVerdict,
    PipelineState,
    ResearchPlan,
    ResearchPlanEvidenceLink,
    TaskTrace,
    TaskTraceReference,
    WorkingAssumptionValidation,
)
from ..synthesis_contract import (
    SYNTHESIS_REQUIREMENT_ID,
    synthesis_contract_for_state,
    synthesis_problem_payload,
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
        "\n--- User follow-up requirement (highest priority, must be honoured) ---\n"
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
        validation_timeout_seconds: float = 60.0,
        fast_mode: bool = False,
        **kwargs,
    ):
        self.fast_mode = bool(fast_mode)
        if generation_timeout_seconds <= 0:
            raise ValueError("generation_timeout_seconds must be positive")
        if semantic_alignment_timeout_seconds <= 0:
            raise ValueError(
                "semantic_alignment_timeout_seconds must be positive"
            )
        if validation_timeout_seconds <= 0:
            raise ValueError("validation_timeout_seconds must be positive")
        self.mode = mode
        self.llm_config = llm_config
        self.generation_timeout_seconds = float(generation_timeout_seconds)
        self.semantic_alignment_timeout_seconds = float(
            semantic_alignment_timeout_seconds
        )
        self.validation_timeout_seconds = float(validation_timeout_seconds)
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
            if review.dimension.value == "overall":
                continue  # aggregate review adds no plan-revision signal
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
    def _build_coverage_retry_context(
        plan: ResearchPlan,
        verdict: Any,
    ) -> str:
        """Render one explicit, bounded repair request for an incomplete plan."""
        unresolved = [
            item for item in verdict.items
            if getattr(item, "verdict", "missing") != "covered"
        ]
        lines = [
            "",
            "M5 INTERNAL COVERAGE RETRY — this is the only allowed coverage rewrite.",
            "Preserve every already-covered part of the previous plan.",
            "Repair every item below; do not omit any target.",
        ]
        for item in unresolved:
            lines.extend([
                "",
                f"[{item.verdict}] {item.target_id}",
                f"Target: {item.target_text}",
                f"Reason: {item.rationale or 'The target lacks complete experimental coverage.'}",
            ])
        if not unresolved:
            lines.extend([
                "",
                f"Reason: {getattr(verdict, 'rationale', '') or 'The coverage verdict is insufficient.'}",
            ])
        lines.extend([
            "",
            "Previous ResearchPlan JSON:",
            json.dumps(plan.model_dump(mode="json"), ensure_ascii=False, indent=2),
        ])
        return "\n".join(lines)

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
        existing_bridge_claims = {link.claim for link in links}
        for bridge in context.bridge_hypotheses:
            claim = f"[{bridge.entry_id}] {bridge.text}"
            if claim in existing_bridge_claims:
                continue
            links.append(ResearchPlanEvidenceLink(
                plan_element=f"Validate unverified bridge {bridge.entry_id}",
                claim=claim,
                support_status="hypothesis_to_validate",
            ))
        return plan.model_copy(update={
            "supporting_evidence_ids": list(dict.fromkeys(evidence_ids)),
            "source_paper_ids": list(dict.fromkeys(paper_ids)),
            "evidence_links": links,
        })

    @staticmethod
    def _inherit_audited_premise_evidence(
        plan: ResearchPlan,
        hypothesis: Any,
        context: GraphContext,
    ) -> ResearchPlan:
        """Carry canonical support for audited M4 facts into every M5 plan.

        The inheritance boundary is deliberately narrow: only factual
        premises already audited as supported or partially supported may add
        provenance.  Hypothesis statements, mechanisms and working
        assumptions remain experiment targets and are never upgraded to
        literature-supported claims by this helper.
        """

        valid_evidence = set(context.available_evidence_ids)
        valid_papers = set(context.available_paper_ids)
        evidence_to_paper = context.evidence_to_paper
        evidence_ids = list(plan.supporting_evidence_ids)
        paper_ids = list(plan.source_paper_ids)
        links = list(plan.evidence_links)
        existing_claims = {
            link.claim.strip().casefold()
            for link in links
            if link.claim.strip()
        }

        for premise in getattr(hypothesis, "factual_premises", []) or []:
            if (
                getattr(premise, "kind", "") != "evidence_backed"
                or getattr(premise, "audit_verdict", "")
                not in {"supported", "partially_supported"}
            ):
                continue
            premise_evidence = list(dict.fromkeys(
                identifier
                for identifier in premise.supporting_evidence_ids
                if identifier in valid_evidence
            ))
            premise_papers = list(dict.fromkeys(
                evidence_to_paper[identifier]
                for identifier in premise_evidence
                if evidence_to_paper.get(identifier) in valid_papers
            ))
            if not premise_evidence or not premise_papers:
                continue

            evidence_ids.extend(premise_evidence)
            paper_ids.extend(premise_papers)
            claim_key = premise.claim.strip().casefold()
            if not claim_key or claim_key in existing_claims:
                continue
            links.append(ResearchPlanEvidenceLink(
                plan_element=f"Factual premise {premise.premise_id}",
                claim=premise.claim.strip(),
                supporting_evidence_ids=premise_evidence,
                source_paper_ids=premise_papers,
                support_status="supported",
            ))
            existing_claims.add(claim_key)

        return plan.model_copy(update={
            "supporting_evidence_ids": list(dict.fromkeys(evidence_ids)),
            "source_paper_ids": list(dict.fromkeys(paper_ids)),
            "evidence_links": links,
        })

    @staticmethod
    def _ensure_bridge_validations(
        plan: ResearchPlan,
        hypothesis: Any,
    ) -> ResearchPlan:
        """Make every M3 bridge used by M4 an explicit M5 test target."""
        assumptions = list(getattr(hypothesis, "working_assumptions", []) or [])
        if not assumptions:
            return plan
        existing = {
            item.bridge_hypothesis_node_id: item
            for item in plan.bridge_validations
            if item.bridge_hypothesis_node_id
        }
        procedure = next((item.strip() for item in plan.procedures if item.strip()), "Run the proposed intervention with a matched control.")
        measurement = next((item.strip() for item in plan.measurement_metrics if item.strip()), "Measure the bridge's predicted downstream outcome.")
        falsification = (
            plan.expected_results_if_refuted.strip()
            or next((item.strip() for item in hypothesis.falsification_conditions if item.strip()), "The predicted effect is absent or reverses under the intervention.")
        )
        validations: list[WorkingAssumptionValidation] = []
        for assumption in assumptions:
            bridge_id = assumption.bridge_hypothesis_node_id
            current = existing.get(bridge_id)
            validations.append(WorkingAssumptionValidation(
                bridge_hypothesis_node_id=bridge_id,
                procedure=(current.procedure.strip() if current and current.procedure.strip() else procedure),
                measurement=(current.measurement.strip() if current and current.measurement.strip() else measurement),
                falsification_condition=(current.falsification_condition.strip() if current and current.falsification_condition.strip() else falsification),
            ))
        return plan.model_copy(update={"bridge_validations": validations})

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
        requirement_ids: set[str] | None = None,
    ) -> ResearchPlan:
        """Derive literal trace excerpts from plan content without inventing text."""

        contract = synthesis_contract_for_state(state)
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
            if (
                requirement_ids is not None
                and requirement.requirement_id not in requirement_ids
            ):
                continue
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
            if (
                not candidates
                and requirement.requirement_id == SYNTHESIS_REQUIREMENT_ID
            ):
                candidates = list(segments)
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
        contract = synthesis_contract_for_state(state)
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
            "Judge whether the research plan studies the same type of primary "
            "object required by the original task. Judge meaning, not mere word "
            "overlap. Answer ONLY yes or no, followed by one short reason.\n\n"
            "Primary task objects:\n"
            + json.dumps(primary_objects, ensure_ascii=False, indent=2)
            + "\n\nA plan may use animals, cell lines, viruses, simulators, "
            + "computational systems, patient cohorts, or other proxies as "
            + "experimental means. It should still be accepted as aligned when "
            + "its study_subjects explicitly names the primary research object "
            + "and explains that those proxies are means to study it, not "
            + "substitutes for it.\n\nPlan study subjects:\n"
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

    @staticmethod
    def _fast_fallback_plan(
        state: PipelineState,
        hypothesis: Any,
        rationale: str = "",
    ) -> ResearchPlan:
        """Minimal research plan used by fast mode when plan generation cannot
        pass alignment; keeps the pipeline able to always return a plan."""
        q = (
            state.problem_card.original_question
            if state.problem_card and state.problem_card.original_question
            else state.input_question
        )
        return ResearchPlan(
            hypothesis_id=hypothesis.hypothesis_id,
            study_subjects=(
                "Target system(s) relevant to: " + (q or "the original question")
            ),
            independent_variables=[
                "Candidate mechanism/feature proposed by the hypothesis",
            ],
            dependent_variables=[
                "Primary measurable outcome(s) expected by the hypothesis",
            ],
            control_groups=[
                "Baseline/control condition without the proposed intervention",
            ],
            procedures=[
                "Define the target system and baseline; apply the proposed intervention; "
                "measure the primary outcome under controlled conditions; "
                "compare against baseline and replicate."
            ],
            measurement_metrics=[
                "Quantitative outcome metric; statistical effect size; reproducibility measure",
            ],
            analysis_methods=[
                "Descriptive statistics and hypothesis test; pre-registered analysis",
            ],
            expected_results_if_supported=(
                "If the hypothesis is supported, the proposed intervention produces "
                "a measurable improvement over baseline."
            ),
            expected_results_if_refuted=(
                "If the hypothesis is refuted, no reliable difference is observed "
                "across replicated runs."
            ),
            timeline="4-8 weeks: setup, intervention, measurement, analysis.",
            risks_and_alternatives=(
                "Risk: limited prior evidence; Alternative: use a simpler pilot "
                "or a larger-scale observational comparison."
            ),
            evidence_links=[],
            task_trace=TaskTrace(),
        )

    @staticmethod
    def _fast_repair_plan_alignment(
        state: PipelineState,
        plan: ResearchPlan,
    ) -> ResearchPlan:
        """Bind an otherwise usable quick plan back to the original task.

        Fast mode already skips the expensive semantic audit.  When the only
        remaining gate is a missing lexical task anchor, preserve every
        generated procedure and measurement while making the original target
        explicit in ``study_subjects`` instead of paying for a second full
        research-plan generation.
        """
        original_question = (
            state.problem_card.original_question
            if state.problem_card and state.problem_card.original_question
            else state.input_question
        )
        target = " ".join(str(original_question or "").split())
        existing = " ".join(str(plan.study_subjects or "").split())
        contract = synthesis_contract_for_state(state)
        primary_objects = [
            entity for entity in contract.entities
            if entity.role == "primary_object"
        ]
        missing_primary = [
            entity for entity in primary_objects
            if not _mentions_entity(existing, entity)
        ]
        if not target and not primary_objects:
            return plan
        if missing_primary:
            primary_hint = "; ".join(
                ", ".join(dict.fromkeys([
                    entity.name,
                    *entity.aliases,
                ]))
                for entity in primary_objects
            )
            study_subjects = (
                f"Original research target: {target}\n"
                f"Primary research object(s): {primary_hint}\n"
                f"Proposed study system: {existing or 'target system from the original question'}"
            )
        else:
            study_subjects = (
                f"Original research target: {target}\n"
                f"Proposed study system: {existing or 'target system from the original question'}"
            )
        return plan.model_copy(update={"study_subjects": study_subjects})
    @staticmethod
    def _is_supplement_reentry(state: PipelineState) -> bool:
        """Only reuse plans for a contiguous M2 supplement return."""
        history = list(state.routing_history or [])
        if not history:
            return False
        latest = history[-1]
        if latest.to_module == "supplement_m2" and latest.from_module in {"m4", "m6"}:
            return True
        if latest.from_module != "m3" or latest.to_module not in {"m4", "m5", "m2"} or len(history) < 2:
            return False
        origin = history[-2]
        return bool(
            origin.to_module == "supplement_m2"
            and origin.from_module in {"m4", "m6"}
            and (
                not latest.gap_ids
                or not origin.gap_ids
                or set(latest.gap_ids) & set(origin.gap_ids)
            )
        )

    @staticmethod
    def _reusable_plans(state: PipelineState) -> dict[str, ResearchPlan]:
        """Return plans whose hypothesis IDs remain in the active M4 portfolio."""
        active_ids = {
            hypothesis.hypothesis_id
            for hypothesis in state.top_hypotheses
            if hypothesis.hypothesis_id
        }
        return {
            plan.hypothesis_id: plan.model_copy(deep=True)
            for plan in state.research_plans
            if plan.hypothesis_id in active_ids
        }

    async def _generate_plan_candidate(
        self,
        *,
        state: PipelineState,
        hypothesis: Any,
        graph_context: GraphContext,
        rendered_graph_context: str,
        problem_card_json: str,
        extra_feedback: str,
        coverage_attempt: int = 1,
    ) -> ResearchPlan:
        """Generate and locally validate one ResearchPlan candidate."""
        started_at = time.monotonic()
        emit_event(
            "tool_started",
            module="m5",
            tool="research_plan_designer",
            status="running",
            message=f"Designing research plan for hypothesis {hypothesis.hypothesis_id}",
            details={
                "hypothesis_id": hypothesis.hypothesis_id,
                "coverage_attempt": coverage_attempt,
            },
        )
        alignment_feedback = ""
        plan: ResearchPlan | None = None
        plan_accepted = False
        last_alignment_rationale = "plan did not pass task alignment"
        for attempt in range(1 if self.fast_mode else 2):
            try:
                working_assumptions_text = (
                    json.dumps(
                        [item.model_dump(mode="json") for item in hypothesis.working_assumptions],
                        ensure_ascii=False,
                        indent=2,
                    ) if hypothesis.working_assumptions else "(none)"
                )
                m5_user_prompt = M5_USER_TEMPLATE.format(
                    original_question=(
                        state.problem_card.original_question
                        if state.problem_card else state.input_question
                    ),
                    problem_card_json=problem_card_json,
                    graph_context=rendered_graph_context,
                    statement=hypothesis.statement,
                    mechanism=hypothesis.mechanism,
                    predictions="\n".join(hypothesis.observable_predictions),
                    falsification_conditions="\n".join(hypothesis.falsification_conditions),
                    hypothesis_evidence="\n".join(hypothesis.supporting_evidence) or "(none)",
                    feedback_context=(
                        self._build_feedback_context(state)
                        + extra_feedback
                        + alignment_feedback
                    ),
                )
                m5_user_prompt += (
                    "\nWorking assumptions that must be tested:\n"
                    + working_assumptions_text
                )
                payload = await asyncio.wait_for(
                    self.client.structured_chat(
                        system_prompt=M5_SYSTEM_PROMPT,
                        user_prompt=m5_user_prompt,
                        output_schema=ResearchPlan.model_json_schema(),
                        max_tokens=16384,
                        temperature=getattr(self.llm_config, "temperature", 0.1),
                        disable_thinking=True,
                    ),
                    timeout=self.generation_timeout_seconds,
                )
            except asyncio.TimeoutError as exc:
                if attempt == 0 and not self.fast_mode:
                    emit_event(
                        "tool_retrying",
                        module="m5",
                        tool="research_plan_designer",
                        status="retrying",
                        message=(
                            f"Plan generation for hypothesis {hypothesis.hypothesis_id} timed out; retrying"
                        ),
                        details={
                            "hypothesis_id": hypothesis.hypothesis_id,
                            "timeout_seconds": self.generation_timeout_seconds,
                            "coverage_attempt": coverage_attempt,
                        },
                    )
                    continue
                if self.fast_mode:
                    emit_event(
                        "tool_failed",
                        module="m5",
                        tool="research_plan_designer",
                        status="failed",
                        message=(
                            f"Plan generation for hypothesis {hypothesis.hypothesis_id} "
                            "timed out in fast mode; using fallback plan"
                        ),
                        details={
                            "hypothesis_id": hypothesis.hypothesis_id,
                            "timeout_seconds": self.generation_timeout_seconds,
                            "coverage_attempt": coverage_attempt,
                        },
                    )
                    break
                raise RuntimeError(
                    "M5 research-plan generation timed out after "
                    f"{self.generation_timeout_seconds:g} seconds"
                ) from exc
            except Exception as exc:
                if not self.fast_mode:
                    raise
                emit_event(
                    "tool_failed",
                    module="m5",
                    tool="research_plan_designer",
                    status="failed",
                    message=(
                        f"Plan generation for hypothesis {hypothesis.hypothesis_id} "
                        f"failed in fast mode: {type(exc).__name__}"
                    ),
                    details={
                        "hypothesis_id": hypothesis.hypothesis_id,
                        "error": type(exc).__name__,
                        "coverage_attempt": coverage_attempt,
                    },
                )
                break
            try:
                plan = ResearchPlan.model_validate(payload).model_copy(
                    update={"hypothesis_id": hypothesis.hypothesis_id}
                )
                plan = self._sanitize_evidence_links(plan, graph_context)
                plan = self._inherit_audited_premise_evidence(
                    plan, hypothesis, graph_context,
                )
                plan = self._ensure_bridge_validations(plan, hypothesis)
                # Always anchor the primary research object explicitly so
                # animal/cell/virus models are not mistaken for the target.
                plan = self._fast_repair_plan_alignment(state, plan)
                if self.fast_mode:
                    # Fast mode deliberately skips the semantic LLM audit.
                    plan = self._fast_repair_plan_alignment(state, plan)
                # A final research plan should always answer the original
                # whole-question Q0 contract.  Only force Q0 when the binding
                # synthesis contract actually contains it; otherwise do not
                # invent a non-existent requirement.
                synthesis_contract = synthesis_contract_for_state(state)
                if any(
                    requirement.requirement_id == SYNTHESIS_REQUIREMENT_ID
                    for requirement in synthesis_contract.requirements
                ):
                    hypothesis_requirement_ids = {SYNTHESIS_REQUIREMENT_ID}
                else:
                    hypothesis_requirement_ids = set()
                plan = self._canonicalize_task_trace(
                    state,
                    plan,
                    requirement_ids=(hypothesis_requirement_ids or None),
                )
                plan_text = self._plan_alignment_text(plan)
                alignment = assess_task_alignment(
                    state,
                    plan_text,
                    subject_text=plan.study_subjects,
                    trace=plan.task_trace,
                    semantic_client=None,
                    required_requirement_ids=(hypothesis_requirement_ids or None),
                    contract_override=synthesis_contract_for_state(state),
                )
            except Exception as exc:
                if not self.fast_mode:
                    raise
                emit_event(
                    "tool_failed",
                    module="m5",
                    tool="research_plan_designer",
                    status="failed",
                    message=(
                        f"Plan output for hypothesis {hypothesis.hypothesis_id} "
                        f"could not be validated in fast mode: {type(exc).__name__}"
                    ),
                    details={
                        "hypothesis_id": hypothesis.hypothesis_id,
                        "error": type(exc).__name__,
                        "coverage_attempt": coverage_attempt,
                    },
                )
                plan = None
                break
            if alignment.passed:
                if self.fast_mode:
                    semantic_consistent = True
                    semantic_rationale = "fast mode: semantic audit skipped"
                else:
                    semantic_consistent, semantic_rationale = await self._audit_plan_semantics(
                        state, plan
                    )
                if semantic_consistent:
                    plan_accepted = True
                    break
                last_alignment_rationale = "semantic object mismatch: " + semantic_rationale
            else:
                last_alignment_rationale = alignment.rationale
                if self.fast_mode:
                    repaired_plan = self._fast_repair_plan_alignment(state, plan)
                    repaired_plan = self._canonicalize_task_trace(
                        state,
                        repaired_plan,
                        requirement_ids=(hypothesis_requirement_ids or None),
                    )
                    repaired_alignment = assess_task_alignment(
                        state,
                        self._plan_alignment_text(repaired_plan),
                        subject_text=repaired_plan.study_subjects,
                        trace=repaired_plan.task_trace,
                        semantic_client=None,
                        required_requirement_ids=(hypothesis_requirement_ids or None),
                        contract_override=synthesis_contract_for_state(state),
                    )
                    emit_event(
                        "tool_result",
                        module="m5",
                        tool="research_plan_designer",
                        status=("completed" if repaired_alignment.passed else "warning"),
                        message=(
                            "Fast-mode task alignment repaired locally"
                            if repaired_alignment.passed
                            else "Fast-mode local alignment repair was insufficient; using fallback plan"
                        ),
                        details={
                            "hypothesis_id": hypothesis.hypothesis_id,
                            "initial_rationale": last_alignment_rationale,
                            "repaired": repaired_alignment.passed,
                            "coverage_attempt": coverage_attempt,
                        },
                    )
                    if repaired_alignment.passed:
                        plan = repaired_plan
                        plan_accepted = True
                    break
            alignment_feedback = (
                "\nHARD ALIGNMENT RETRY: The previous plan was rejected because "
                f"{last_alignment_rationale} Rewrite it for the original research object "
                "and return valid task-entity and requirement traces from the "
                "binding M1 task contract.\n"
            )
        if plan is None and self.fast_mode:
            plan = self._fast_fallback_plan(
                state,
                hypothesis,
                "fast-mode plan generation timed out",
            )
        assert plan is not None
        if not plan_accepted and self.fast_mode:
            plan = self._fast_fallback_plan(state, hypothesis, last_alignment_rationale)
        if not plan_accepted and not self.fast_mode:
            # Soft-fail: keep the generated plan so the pipeline can continue,
            # but surface a visible object-alignment warning for human review.
            emit_event(
                "tool_result",
                module="m5",
                tool="research_plan_designer",
                status="warning",
                message=(
                    f"Research plan for {hypothesis.hypothesis_id} could not "
                    "pass semantic object alignment; continuing with a warning"
                ),
                details={
                    "hypothesis_id": hypothesis.hypothesis_id,
                    "alignment_warning": last_alignment_rationale,
                    "coverage_attempt": coverage_attempt,
                },
            )
            plan = plan.model_copy(update={
                "alignment_warning": last_alignment_rationale,
            })
            plan_accepted = True
        plan = self._sanitize_evidence_links(plan, graph_context)
        plan = self._inherit_audited_premise_evidence(
            plan, hypothesis, graph_context,
        )
        plan = self._ensure_bridge_validations(plan, hypothesis)
        emit_event(
            "tool_completed",
            module="m5",
            tool="research_plan_designer",
            status="completed",
            message=f"Research plan for hypothesis {hypothesis.hypothesis_id} completed",
            elapsed_seconds=time.monotonic() - started_at,
            details={
                "hypothesis_id": hypothesis.hypothesis_id,
                "procedures": len(plan.procedures),
                "analysis_methods": len(plan.analysis_methods),
                "coverage_attempt": coverage_attempt,
            },
        )
        return plan

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
        validation_verdicts: List[ExperimentalValidationVerdict] = []
        supplement_reentry = self._is_supplement_reentry(state)
        reusable = self._reusable_plans(state) if supplement_reentry else {}
        graph_context = build_graph_context(state)
        problem_card_json = json.dumps(
            synthesis_problem_payload(state.problem_card),
            ensure_ascii=False,
            indent=2,
        )
        for h in state.top_hypotheses:
            if h.hypothesis_id in reusable:
                prior = reusable[h.hypothesis_id]
                plan = prior.model_copy(update={
                    "supporting_evidence_ids": list(dict.fromkeys([
                        *prior.supporting_evidence_ids,
                        *h.supporting_evidence,
                    ])),
                    "source_paper_ids": list(dict.fromkeys([
                        *prior.source_paper_ids,
                        *h.source_paper_ids,
                    ])),
                })
                plan = self._sanitize_evidence_links(plan, graph_context)
                plan = self._inherit_audited_premise_evidence(
                    plan, h, graph_context,
                )
                plan = self._ensure_bridge_validations(plan, h)
                plans.append(plan)
                emit_event(
                    "tool_completed",
                    module="m5",
                    tool="research_plan_reuser",
                    status="completed",
                    message=f"Reused research plan for hypothesis {h.hypothesis_id}",
                    details={
                        "hypothesis_id": h.hypothesis_id,
                        "bridge_validations": len(plan.bridge_validations),
                    },
                )
                continue
            context_pack = ContextPlanner().plan(
                graph_context,
                ContextRequest(
                    purpose="m5_plan",
                    focus_evidence_ids=tuple(h.supporting_evidence),
                ),
            )
            emit_context_built(
                "m5",
                "research_plan_designer",
                context_pack,
            )
            rendered_graph_context = context_pack.rendered
            coverage_retry_context = ""
            final_plan: ResearchPlan | None = None
            final_validation_verdict: ExperimentalValidationVerdict | None = None
            validation_auditor = None if self.fast_mode else ExperimentalValidationAuditor(
                client=self.client,
                llm_config=self.llm_config,
                timeout_seconds=self.validation_timeout_seconds,
            )
            for coverage_attempt in range(1 if self.fast_mode else 2):
                final_plan = await self._generate_plan_candidate(
                    state=state,
                    hypothesis=h,
                    graph_context=graph_context,
                    rendered_graph_context=rendered_graph_context,
                    problem_card_json=problem_card_json,
                    extra_feedback=coverage_retry_context,
                    coverage_attempt=coverage_attempt + 1,
                )
                if validation_auditor is None:
                    break
                emit_event(
                    "tool_started",
                    module="m5",
                    tool="experimental_validation_precheck",
                    status="running",
                    message=f"Prechecking M5 coverage for hypothesis {h.hypothesis_id}",
                    details={
                        "hypothesis_id": h.hypothesis_id,
                        "attempt": coverage_attempt + 1,
                    },
                )
                outcome = await validation_auditor.audit(
                    h,
                    final_plan,
                    version=state.iteration_count + 1,
                )
                if outcome.error:
                    emit_event(
                        "tool_result",
                        module="m5",
                        tool="experimental_validation_precheck",
                        status="warning",
                        message=(
                            "M5 precheck failed; plan preserved for formal M6 review"
                        ),
                        details={
                            "hypothesis_id": h.hypothesis_id,
                            "attempt": coverage_attempt + 1,
                            "error": outcome.error,
                        },
                    )
                    break
                final_validation_verdict = outcome.verdict
                unresolved = [
                    item for item in outcome.verdict.items
                    if item.verdict != "covered"
                ]
                emit_event(
                    "tool_completed",
                    module="m5",
                    tool="experimental_validation_precheck",
                    status="completed",
                    message=(
                        "M5 precheck passed"
                        if outcome.verdict.sufficient
                        else "M5 precheck found incomplete coverage"
                    ),
                    details={
                        "hypothesis_id": h.hypothesis_id,
                        "attempt": coverage_attempt + 1,
                        "sufficient": outcome.verdict.sufficient,
                        "missing": sum(item.verdict == "missing" for item in unresolved),
                        "partial": sum(item.verdict == "partial" for item in unresolved),
                    },
                )
                if outcome.verdict.sufficient or coverage_attempt == 1:
                    break
                coverage_retry_context = self._build_coverage_retry_context(
                    final_plan,
                    outcome.verdict,
                )
                emit_event(
                    "tool_retrying",
                    module="m5",
                    tool="research_plan_designer",
                    status="retrying",
                    message=(
                        f"Coverage incomplete for hypothesis {h.hypothesis_id}; "
                        "performing the only allowed internal rewrite"
                    ),
                    details={
                        "hypothesis_id": h.hypothesis_id,
                        "reason": "coverage_incomplete",
                        "missing": sum(item.verdict == "missing" for item in unresolved),
                        "partial": sum(item.verdict == "partial" for item in unresolved),
                    },
                )
            assert final_plan is not None
            plans.append(final_plan)
            if final_validation_verdict is not None:
                validation_verdicts.append(final_validation_verdict)
            continue

        history = dict(state.research_plan_history)
        history[state.iteration_count + 1] = [plan.model_copy(deep=True) for plan in plans]
        if validation_verdicts:
            validation_verdict = ExperimentalValidationVerdict(
                sufficient=all(item.sufficient for item in validation_verdicts),
                items=[
                    coverage
                    for item in validation_verdicts
                    for coverage in item.items
                ],
                rationale=" ".join(
                    item.rationale for item in validation_verdicts if item.rationale
                ),
            )
        elif reusable and state.experimental_validation_verdict is not None:
            validation_verdict = state.experimental_validation_verdict.model_copy(deep=True)
        else:
            validation_verdict = None
        return {
            "research_plans": plans,
            "research_plan_history": history,
            "experimental_validation_verdict": validation_verdict,
        }

    @classmethod
    def get_input_fields(cls) -> List[str]:
        return [
            "top_hypotheses", "reviews", "iteration_count", "problem_card",
            "evidence_graph", "grounding_report", "literature_results",
            "m2_knowledge_export",
        ]

    @classmethod
    def get_output_fields(cls) -> List[str]:
        return [
            "research_plans",
            "research_plan_history",
            "experimental_validation_verdict",
        ]
