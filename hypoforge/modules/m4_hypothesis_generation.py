"""
M4: Hypothesis Generation & Screening (core innovation module).

Multi-agent pipeline: Generator → Ranker.  On revision rounds it folds in the
latest reviewer feedback and optional human guidance, and keeps the best
hypotheses seen across iterations.

Output: ``candidate_hypotheses`` + ``top_hypotheses`` in state.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import sys
import time
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

from ..observability import emit_event
from ..context import ContextPlanner, ContextRequest, emit_context_built
from ..evidence_audit import (
    ClaimEvidenceVerdict,
    EvidenceAuditService,
    PremiseAuditResult,
)
from ..epistemic_contract import (
    EpistemicContractDiagnostic,
    normalize_hypothesis_grounding,
    validate_epistemic_contract,
)
from ..graph_context import (
    GraphContext,
    active_bridge_ids_for_state,
    build_graph_context,
)
from ..protocol import ModuleProtocol
from ..prompts.m4_prompts import (
    M4_CONTRACT_REPAIR_SYSTEM_PROMPT,
    M4_CONTRACT_REPAIR_USER_TEMPLATE,
    M4_CRITIC_SYSTEM_PROMPT,
    M4_CRITIC_USER_TEMPLATE,
    M4_FALSIFIABILITY_SYSTEM_PROMPT,
    M4_FALSIFIABILITY_USER_TEMPLATE,
    M4_GENERATOR_SYSTEM_PROMPT,
    M4_GENERATOR_USER_TEMPLATE,
    M4_EPISTEMIC_AUDITOR_SYSTEM_PROMPT,
    M4_EPISTEMIC_AUDITOR_USER_TEMPLATE,
    M4_RANKER_SYSTEM_PROMPT,
    M4_RANKER_USER_TEMPLATE,
)
from ..evaluation.rubric import (
    HYPOTHESIS_DIMENSIONS,
    composite_score,
    hypothesis_rubric_block,
    normalise_weights,
    weights_summary,
)
from ..registry import ModuleRegistry
from ..state import (
    ClarificationRequest,
    EvidenceGapRequest,
    HypothesisCard,
    HypothesisPremise,
    PipelineState,
    TaskTrace,
    TaskTraceReference,
)
from ..synthesis_contract import (
    SYNTHESIS_REQUIREMENT_ID,
    synthesis_contract_for_state,
)
from ..task_alignment import _mentions_entity, assess_task_alignment
from ..tools.qwen_client import QwenClient, track_token_scope

logger = logging.getLogger(__name__)


class M4RefinementRequired(Exception):
    """Raised when M4 cannot form an acceptable whole-question hypothesis.

    The pipeline should stop and ask the user to refine the broad original
    question into a more specific research direction instead of failing hard.
    """

    def __init__(
        self,
        clarification: ClarificationRequest,
        candidates: List[HypothesisCard],
    ) -> None:
        self.clarification = clarification
        self.candidates = list(candidates)
        super().__init__(clarification.message or "M4 requires user refinement")


_EDITORIAL_STATEMENT_RE = re.compile(
    r"^\s*(?:"
    r"(?:please\s+)?consider\b|"
    r"to\s+(?:better\s+)?(?:strengthen|improve|clarify|address|validate)\b|"
    r"(?:we|the authors?)\s+(?:hypothesize|suggest|recommend|propose|should|need\s+to)\b|"
    r"(?:this|the)\s+hypothesis\s+(?:should|must|needs?\s+to|would\s+benefit)\b|"
    r"(?:a|the)\s+(?:stronger|revised)\s+hypothesis\s+(?:would|should)\b|"
    r"it\s+(?:is|would\s+be)\s+(?:important|helpful|useful|necessary|recommended)\b|"
    r"it\s+would\s+(?:strengthen|improve|clarify|address|validate)\b|"
    r"(?:future|further)\s+(?:work|research)\b|"
    r"(?:add|include|provide|clarify|acknowledge|discuss|investigate|explore)\b|"
    r"(?:建议|请考虑|考虑(?:增加|加入|补充)|应当|应该|需要(?:增加|加入|补充)|"
    r"为(?:了)?(?:加强|增强|改进|验证)|未来(?:工作|研究)|(?:本|该)假设(?:应|应该|需要))"
    r")",
    re.IGNORECASE,
)

_TRACE_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "does", "for",
    "from", "how", "in", "is", "of", "on", "or", "the", "to", "use",
    "uses", "via", "what", "which", "with",
})


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
        "\n--- User follow-up instruction (highest priority, must be honoured) ---\n"
        f"{text}\n"
    )


@ModuleRegistry.register
class M4HypothesisGeneration(ModuleProtocol):
    module_name = "m4"
    module_version = "0.1.0"
    description = "Multi-agent hypothesis generation: Generator → Critic → Falsifiability → Ranker"

    # ------------------------------------------------------------------
    # Configurable parameters
    # ------------------------------------------------------------------

    def __init__(
        self,
        num_candidates: int = 7,
        top_k: int = 3,
        mode: str = "multi_agent",  # "direct" | "multi_agent"
        llm_config: Optional[Any] = None,
        ranker_llm_config: Optional[Any] = None,
        weights: Optional[Dict[str, float]] = None,
        interactive: bool = False,
        llm_call_timeout: float = 240.0,
        total_time_budget_seconds: float = 540.0,
        semantic_alignment_timeout_seconds: float = 60.0,
        fast_mode: bool = False,
        **kwargs,
    ):
        self.num_candidates = num_candidates
        self.top_k = top_k
        self.mode = mode
        self.fast_mode = bool(fast_mode)
        self.llm_config = llm_config
        # Composite weights come from a single source (PipelineConfig.scoring →
        # rubric defaults); the ranker prompt and the recompute below share them.
        self.weights = normalise_weights(weights)
        # When True (and on a real TTY), pause between iterations to let the user
        # type guidance for the next revision — Claude-Code style.
        self.interactive = interactive
        if semantic_alignment_timeout_seconds <= 0:
            raise ValueError("semantic_alignment_timeout_seconds must be positive")
        self.semantic_alignment_timeout_seconds = float(
            semantic_alignment_timeout_seconds
        )
        # Per-LLM-call ceiling: a single hung or queued call must not silently
        # consume the whole node budget (mirrors m3's llm_call_timeout).
        self.llm_call_timeout = max(10.0, float(llm_call_timeout))
        if total_time_budget_seconds < 0:
            raise ValueError("total_time_budget_seconds cannot be negative")
        self.total_time_budget_seconds = float(total_time_budget_seconds)
        self._run_deadline: float | None = None
        self.client = QwenClient.from_config(llm_config) if llm_config else None
        self.evidence_auditor = EvidenceAuditService(self.client)
        # Ranker uses a separate model tier to reduce self-scoring bias.
        # Falls back to the main client when no separate config is provided.
        _ranker_cfg = ranker_llm_config if ranker_llm_config is not None else llm_config
        self.ranker_client = QwenClient.from_config(_ranker_cfg) if _ranker_cfg else None

    # ------------------------------------------------------------------
    # LLM helpers
    # ------------------------------------------------------------------

    def _tool_call(
        self,
        tool: str,
        operation,
        *,
        details: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Observe one LLM tool call under the module's per-call timeout."""
        timeout = self.llm_call_timeout
        if self._run_deadline is not None:
            remaining = self._run_deadline - time.monotonic()
            if remaining <= 0:
                close = getattr(operation, "close", None)
                if callable(close):
                    close()
                raise RuntimeError("M4 exhausted its shared LLM time budget")
            timeout = min(timeout, remaining)
        return self._observe_tool(
            tool,
            operation,
            details=details,
            timeout=timeout,
        )

    @staticmethod
    async def _observe_tool(
        tool: str,
        operation,
        *,
        details: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Any:
        started_at = time.monotonic()
        tool_usage = {"input": 0, "output": 0, "calls": 0}

        async def tracked_operation() -> Any:
            with track_token_scope() as scope:
                try:
                    return await operation
                finally:
                    tool_usage.update(scope.snapshot())

        emit_event(
            "tool_started",
            module="m4",
            tool=tool,
            status="running",
            message=f"M4 Agent started: {tool}",
            details=details,
        )
        try:
            if timeout is not None:
                result = await asyncio.wait_for(
                    tracked_operation(), timeout=timeout
                )
            else:
                result = await tracked_operation()
        except asyncio.CancelledError:
            emit_event(
                "tool_cancelled",
                module="m4",
                tool=tool,
                status="cancelled",
                message=f"M4 Agent cancelled: {tool}",
                elapsed_seconds=time.monotonic() - started_at,
                details={**(details or {}), "token_usage": dict(tool_usage)},
            )
            raise
        except BaseException as exc:
            emit_event(
                "tool_failed",
                module="m4",
                tool=tool,
                status="failed",
                message=f"M4 Agent failed: {tool}: {type(exc).__name__}: {exc}",
                elapsed_seconds=time.monotonic() - started_at,
                details={**(details or {}), "token_usage": dict(tool_usage)},
            )
            raise
        emit_event(
            "tool_completed",
            module="m4",
            tool=tool,
            status="completed",
            message=f"M4 Agent completed: {tool}",
            elapsed_seconds=time.monotonic() - started_at,
            details={**(details or {}), "token_usage": dict(tool_usage)},
        )
        return result

    @staticmethod
    def _is_scientific_statement(statement: str) -> bool:
        """Return whether *statement* has the form of a scientific claim.

        This is deliberately a narrow guard against editorial instructions,
        not a substitute for scientific evaluation.  It protects state
        boundaries even when an LLM ignores prompt-level requirements.
        """
        text = str(statement or "").strip().lstrip("-*•").strip()
        if not text or text.endswith(("?", "？")):
            return False
        return _EDITORIAL_STATEMENT_RE.match(text) is None

    def _graph_bucket_text(self, state: PipelineState, bucket: str) -> str:
        context = build_graph_context(state)
        items = getattr(context, bucket, [])
        lines = [
            f"- {item.entry_id}: {item.text}"
            + (f" [evidence: {', '.join(item.evidence_ids)}]" if item.evidence_ids else "")
            for item in items
        ]
        return "\n".join(lines) if lines else "- None available"

    @staticmethod
    def _requirement_match_score(segment: str, requirement: Any) -> int:
        """Domain-neutral lexical evidence for a requirement/segment pair."""

        requirement_tokens = {
            token for token in re.findall(
                r"[a-z0-9]+",
                f"{requirement.relation} {requirement.sub_question}".casefold(),
            )
            if len(token) > 2 and token not in _TRACE_STOPWORDS
        }
        segment_tokens = set(re.findall(r"[a-z0-9]+", segment.casefold()))
        return len(requirement_tokens.intersection(segment_tokens))

    @staticmethod
    def _canonicalize_task_traces(
        state: PipelineState,
        cards: List[HypothesisCard],
    ) -> List[HypothesisCard]:
        """Rebuild task traces from literal hypothesis output segments.

        Models occasionally cover every binding task entity in the generated
        hypothesis but attach a trace excerpt that points at the wrong words.
        The trace is audit metadata, so derive it deterministically from the
        actual output before applying the hard alignment gate.  Missing content
        is never synthesized: an entity or requirement gets no trace unless a
        literal output segment contains its required entity text.
        """

        problem_card = state.problem_card
        if problem_card is None:
            return cards
        contract = synthesis_contract_for_state(state)
        if not contract.entities and not contract.requirements:
            return cards

        entity_by_id = {
            entity.entity_id: entity for entity in contract.entities
        }
        canonical: List[HypothesisCard] = []
        for card in cards:
            segments = list(dict.fromkeys(
                segment.strip()
                for segment in [
                    card.statement,
                    card.mechanism,
                    *card.observable_predictions,
                    *card.falsification_conditions,
                ]
                if str(segment or "").strip()
            ))

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

            declared_requirement_ids = list(dict.fromkeys(
                reference.contract_id
                for reference in card.task_trace.requirement_mentions
                if any(
                    requirement.requirement_id == reference.contract_id
                    for requirement in contract.requirements
                )
            ))
            if (
                not declared_requirement_ids
                and any(
                    requirement.requirement_id == SYNTHESIS_REQUIREMENT_ID
                    for requirement in contract.requirements
                )
            ):
                # Legacy and imperfect model outputs may still carry retrieval
                # IDs (R1...Rn).  Those IDs are intentionally unavailable to
                # M4, so rebuild the trace as the whole-question requirement
                # when the actual hypothesis text names the primary object.
                q0 = next(
                    requirement for requirement in contract.requirements
                    if requirement.requirement_id == SYNTHESIS_REQUIREMENT_ID
                )
                if segments:
                    declared_requirement_ids = [SYNTHESIS_REQUIREMENT_ID]
            if not declared_requirement_ids and contract.requirements:
                # Recover at most one omitted trace from actual relation words.
                # The previous implementation assigned every requirement that
                # shared a primary entity, allowing one generic sentence to
                # masquerade as R1…Rn.
                scored = [
                    (
                        max(
                            (
                                M4HypothesisGeneration._requirement_match_score(
                                    segment, requirement
                                )
                                for segment in segments
                            ),
                            default=0,
                        ),
                        requirement.requirement_id,
                    )
                    for requirement in contract.requirements
                ]
                best_score, best_id = max(scored, default=(0, ""))
                if best_score > 0:
                    declared_requirement_ids = [best_id]

            requirement_mentions: List[TaskTraceReference] = []
            for requirement in contract.requirements:
                if requirement.requirement_id not in declared_requirement_ids:
                    continue
                primary = entity_by_id.get(requirement.primary_entity_id)
                if primary is None:
                    continue
                requirement_entities = [
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
                        M4HypothesisGeneration._requirement_match_score(
                            segment, requirement
                        ),
                        sum(
                            _mentions_entity(segment, entity)
                            for entity in requirement_entities
                        ),
                        -segments.index(segment),
                    ),
                )
                requirement_mentions.append(TaskTraceReference(
                    contract_id=requirement.requirement_id,
                    output_excerpt=excerpt,
                ))

            canonical.append(card.model_copy(update={
                "task_trace": TaskTrace(
                    entity_mentions=entity_mentions,
                    requirement_mentions=requirement_mentions,
                ),
            }))
        return canonical

    @staticmethod
    def _render_task_contract_block(
        state: PipelineState,
        *,
        require_all: bool = False,
    ) -> str:
        """Render the binding M1 task contract for literal reproduction.

        The alignment gate matches entity names character-for-character, so
        the generator must see the exact names/aliases it has to quote.
        Returns an empty string when there is no binding contract.
        """
        card = state.problem_card
        if card is None:
            return ""
        contract = synthesis_contract_for_state(state)
        if not contract.entities and not contract.requirements:
            return ""
        lines = [
            "Binding task contract (hard alignment gate):",
            "Task entity hints for semantic scope. They may be translated or "
            "paraphrased to match the language of the original question:",
        ]
        for entity in contract.entities:
            terms = list(dict.fromkeys(
                term for term in [entity.name, *entity.aliases]
                if str(term).strip()
            ))
            mark = "required" if entity.required else "optional"
            rendered = (
                f"- {entity.entity_id}: {entity.name} ({entity.role}, {mark})"
            )
            if len(terms) > 1:
                rendered += f" — exact names/aliases: {' | '.join(terms)}"
            lines.append(rendered)
        if contract.requirements:
            lines.append(
                "Whole-question synthesis requirement — every candidate must "
                "independently answer the original question as a whole and "
                "include one literal task_trace excerpt for Q0. Retrieval "
                "sub-questions are intentionally unavailable and no part of the "
                "original question may be deferred to another candidate:"
            )
            for requirement in contract.requirements:
                lines.append(
                    f"- {requirement.requirement_id}: {requirement.relation}; "
                    f"original question: {requirement.sub_question} "
                    f"(primary entity: {requirement.primary_entity_id})"
                )
        lines.append(
            "The original question and Q0 define the binding answer scope; "
            "entity hints are not literal-string hard gates."
        )
        return "\n".join(lines)

    @staticmethod
    def _task_contract_reminder(state: PipelineState) -> str:
        """Short reminder of the required task scope for feedback passes.

        Iteration feedback (e.g. "be more novel") can push the generator away
        from task alignment; the reminder re-anchors it to the required
        entities and requirement traces before the feedback text.
        """
        card = state.problem_card
        if card is None:
            return ""
        contract = synthesis_contract_for_state(state)
        if not contract.entities and not contract.requirements:
            return ""
        lines = ["Task-contract reminder: iteration feedback must not break task alignment."]
        entities = [
            entity.name for entity in contract.entities if entity.required
        ]
        if entities:
            lines.append(
                "Required task entities that must stay in the hypothesis "
                "output scope: " + ", ".join(entities) + "."
            )
        requirements = [
            requirement.requirement_id
            for requirement in contract.requirements
            if requirement.required
        ]
        if requirements:
            lines.append(
                "Required whole-question synthesis ID: " + ", ".join(requirements)
                + ". Every candidate must continue to answer the original "
                "question as a whole."
            )
        return "\n".join(lines) + "\n\n"

    @staticmethod
    def _check_context_contract(
        state: PipelineState,
        context: GraphContext,
        cards: List[HypothesisCard],
        *,
        semantic_client: object | None = None,
    ) -> tuple[List[HypothesisCard], List[Dict[str, Any]]]:
        """Return accepted cards and machine-readable contract diagnostics."""

        valid_references = set(context.available_evidence_ids)
        accepted: List[HypothesisCard] = []
        failures: List[Dict[str, Any]] = []
        contract = synthesis_contract_for_state(state) if state.problem_card else None
        known_required_ids = {
            requirement.requirement_id
            for requirement in (contract.requirements if contract else [])
            if requirement.required
        } if contract is not None and contract.source == "m1" else set()
        for card in cards:
            candidate_text = "\n".join([
                card.statement,
                card.mechanism,
                *card.observable_predictions,
                *card.falsification_conditions,
            ])
            scoped_requirement_ids = {
                reference.contract_id
                for reference in card.task_trace.requirement_mentions
                if reference.contract_id in known_required_ids
            }
            alignment = assess_task_alignment(
                state,
                candidate_text,
                subject_text="\n".join([card.statement, card.mechanism]),
                trace=card.task_trace,
                semantic_client=semantic_client,
                required_requirement_ids=scoped_requirement_ids,
                contract_override=contract,
            )
            if not alignment.passed:
                failures.append({
                    "hypothesis_id": card.hypothesis_id,
                    "rationale": alignment.rationale,
                    "missing_task_entities": list(alignment.missing_anchors),
                    "missing_requirement_ids": list(
                        alignment.missing_requirement_ids
                    ),
                    "invalid_task_trace_references": list(
                        alignment.invalid_trace_references
                    ),
                })
                logger.warning(
                    "M4 discarded task-misaligned hypothesis %s: %s",
                    card.hypothesis_id,
                    alignment.rationale,
                )
                continue
            if known_required_ids and not scoped_requirement_ids:
                failures.append({
                    "hypothesis_id": card.hypothesis_id,
                    "rationale": (
                        "hypothesis does not declare any auditable atomic "
                        "requirement scope"
                    ),
                    "missing_task_entities": [],
                    "missing_requirement_ids": sorted(known_required_ids),
                    "invalid_task_trace_references": [],
                })
                continue
            references = list(dict.fromkeys(
                reference
                for reference in card.supporting_evidence
                if reference in valid_references
            ))
            if len(references) != len(card.supporting_evidence):
                logger.warning(
                    "M4 removed unresolved evidence references from %s",
                    card.hypothesis_id,
                )
            valid_papers = set(context.available_paper_ids)
            paper_ids = list(dict.fromkeys(
                pid for pid in card.source_paper_ids if pid in valid_papers
            ))
            if len(paper_ids) != len(card.source_paper_ids):
                logger.warning(
                    "M4 removed unresolved paper references from %s",
                    card.hypothesis_id,
                )
            accepted.append(card.model_copy(update={
                "supporting_evidence": references,
                "source_paper_ids": paper_ids,
            }))
        return accepted, failures

    async def _audit_context_contract_semantics(
        self,
        state: PipelineState,
        candidates: List[HypothesisCard],
    ) -> tuple[List[HypothesisCard], List[Dict[str, Any]]]:
        """Batch-check candidate study objects without blocking the event loop."""

        if not candidates:
            return [], []
        assert self.client is not None
        contract = synthesis_contract_for_state(state) if state.problem_card else None
        primary_objects = [
            {
                "entity_id": entity.entity_id,
                "name": entity.name,
                "aliases": list(entity.aliases),
            }
            for entity in (contract.entities if contract is not None else [])
            if entity.role == "primary_object"
        ]
        entity_by_id = {
            entity.entity_id: entity
            for entity in (contract.entities if contract is not None else [])
        }
        requirements = [
            {
                "requirement_id": requirement.requirement_id,
                "sub_question": requirement.sub_question,
                "relation": requirement.relation,
                "primary_entity": (
                    entity_by_id[requirement.primary_entity_id].name
                    if requirement.primary_entity_id in entity_by_id else ""
                ),
                "related_entities": [
                    entity_by_id[entity_id].name
                    for entity_id in requirement.related_entity_ids
                    if entity_id in entity_by_id
                ],
            }
            for requirement in (contract.requirements if contract is not None else [])
            if requirement.required
        ] if contract is not None and contract.source == "m1" else []
        if not primary_objects:
            return list(candidates), []

        schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "hypothesis_id": {"type": "string"},
                    "consistent": {"type": "boolean"},
                    "rationale": {"type": "string"},
                    "covered_requirement_ids": {
                        "type": "array", "items": {"type": "string"},
                    },
                },
                "required": [
                    "hypothesis_id", "consistent", "rationale",
                    "covered_requirement_ids",
                ],
            },
        }
        user_prompt = (
            "Check whether every candidate hypothesis studies the same type of "
            "primary object required by the original task. Judge meaning, not "
            "mere word overlap. Return exactly one verdict per hypothesis_id.\n\n"
            "Primary task objects:\n"
            + json.dumps(primary_objects, ensure_ascii=False, indent=2)
            + "\n\nAtomic task requirements:\n"
            + json.dumps(requirements, ensure_ascii=False, indent=2)
            + "\n\nCandidate hypotheses:\n"
            + json.dumps(
                [
                    {
                        "hypothesis_id": card.hypothesis_id,
                        "statement": card.statement,
                        "mechanism": card.mechanism,
                        "declared_requirement_ids": [
                            reference.contract_id
                            for reference in card.task_trace.requirement_mentions
                        ],
                    }
                    for card in candidates
                ],
                ensure_ascii=False,
                indent=2,
            )
        )
        try:
            verdicts = await self._tool_call(
                "hypothesis_contract_auditor",
                asyncio.wait_for(
                    self.client.structured_chat(
                        system_prompt=(
                            "You are a strict task-alignment auditor. Fail a "
                            "candidate when its research object drifts from the "
                            "required primary object. For each candidate, return "
                            "only requirement IDs whose stated relation is "
                            "actually expressed; entity mention alone is not "
                            "requirement coverage."
                        ),
                        user_prompt=user_prompt,
                        output_schema=schema,
                        max_tokens=2048,
                        temperature=0.0,
                        disable_thinking=True,
                    ),
                    timeout=self.semantic_alignment_timeout_seconds,
                ),
                details={
                    "candidates": len(candidates),
                    "primary_objects": len(primary_objects),
                    "timeout_seconds": self.semantic_alignment_timeout_seconds,
                },
            )
        except asyncio.TimeoutError as exc:
            raise RuntimeError(
                "M4 semantic task-contract audit timed out after "
                f"{self.semantic_alignment_timeout_seconds:g} seconds"
            ) from exc

        by_id = {card.hypothesis_id: card for card in candidates}
        seen: set[str] = set()
        accepted: List[HypothesisCard] = []
        failures: List[Dict[str, Any]] = []
        for verdict in verdicts if isinstance(verdicts, list) else []:
            if not isinstance(verdict, dict):
                continue
            hypothesis_id = str(verdict.get("hypothesis_id") or "")
            if hypothesis_id not in by_id or hypothesis_id in seen:
                continue
            consistent = verdict.get("consistent")
            if not isinstance(consistent, bool):
                continue
            seen.add(hypothesis_id)
            if consistent:
                card = by_id[hypothesis_id]
                declared = {
                    reference.contract_id
                    for reference in card.task_trace.requirement_mentions
                }
                raw_covered = verdict.get("covered_requirement_ids")
                covered = (
                    {
                        str(item) for item in raw_covered
                        if str(item) in declared
                    }
                    if isinstance(raw_covered, list)
                    else (set() if requirements else declared)
                )
                if requirements and not covered:
                    failures.append({
                        "hypothesis_id": hypothesis_id,
                        "rationale": (
                            "semantic task-contract auditor found no covered "
                            "atomic requirement"
                        ),
                        "semantic_requirement_mismatch": True,
                    })
                    continue
                accepted.append(card.model_copy(update={
                    "task_trace": card.task_trace.model_copy(update={
                        "requirement_mentions": [
                            reference
                            for reference in card.task_trace.requirement_mentions
                            if reference.contract_id in covered
                        ],
                    }),
                }))
            else:
                failures.append({
                    "hypothesis_id": hypothesis_id,
                    "rationale": str(
                        verdict.get("rationale")
                        or "semantic task-object mismatch"
                    ),
                    "semantic_object_mismatch": True,
                })
        for hypothesis_id in by_id.keys() - seen:
            failures.append({
                "hypothesis_id": hypothesis_id,
                "rationale": (
                    "semantic task-contract auditor returned no valid verdict"
                ),
                "semantic_verdict_missing": True,
            })
        return accepted, failures

    @staticmethod
    def _enforce_context_contract(
        state: PipelineState,
        context: GraphContext,
        cards: List[HypothesisCard],
        *,
        semantic_client: object | None = None,
    ) -> List[HypothesisCard]:
        """Compatibility wrapper returning only contract-compliant cards."""

        accepted, _ = M4HypothesisGeneration._check_context_contract(
            state, context, cards, semantic_client=semantic_client,
        )
        return accepted

    async def _repair_context_contract(
        self,
        *,
        state: PipelineState,
        context: GraphContext,
        candidates: List[HypothesisCard],
        failures: List[Dict[str, Any]],
        feedback_context: str,
        requested_count: Optional[int] = None,
        disable_thinking: bool = True,
    ) -> tuple[List[HypothesisCard], List[Dict[str, Any]]]:
        """Make one deterministic repair attempt without relaxing hard gates."""

        assert self.client is not None
        requested = (
            self.top_k
            if requested_count is None
            else max(1, int(requested_count))
        )
        question = (
            state.problem_card.original_question
            if state.problem_card else state.input_question
        )
        context_pack = ContextPlanner().plan(
            context,
            ContextRequest(
                purpose="m4_generate",
                focus_evidence_ids=tuple(
                    evidence_id
                    for candidate in candidates
                    for evidence_id in candidate.supporting_evidence
                ),
            ),
        )
        emit_context_built(
            "m4",
            "hypothesis_contract_repair",
            context_pack,
        )
        repaired = await self._tool_call(
            "hypothesis_contract_repair",
            self.client.structured_chat(
                system_prompt=M4_CONTRACT_REPAIR_SYSTEM_PROMPT.format(
                    num_candidates=requested,
                ),
                user_prompt=M4_CONTRACT_REPAIR_USER_TEMPLATE.format(
                    graph_context=context_pack.rendered,
                    candidate_json=json.dumps(
                        [card.model_dump(mode="json") for card in candidates],
                        ensure_ascii=False,
                        indent=2,
                    ),
                    failure_json=json.dumps(
                        failures or [{
                            "rationale": (
                                "No schema-valid scientific hypothesis was returned."
                            ),
                        }],
                        ensure_ascii=False,
                        indent=2,
                    ),
                    original_question=question,
                    feedback_context=feedback_context,
                    task_contract_block=self._render_task_contract_block(
                        state,
                        require_all=self.fast_mode,
                    ),
                ),
                output_schema=self._hypothesis_generation_schema(),
                max_tokens=8192,
                temperature=0.0,
                disable_thinking=disable_thinking,
            ),
            details={
                "candidate_count": len(candidates),
                "failure_count": len(failures),
                "requested_candidates": requested,
                "disable_thinking": disable_thinking,
                "iteration": state.iteration_count + 1,
            },
        )
        repaired_candidates = self._canonicalize_task_traces(
            state, self._normalise_hypotheses(repaired),
        )
        accepted, deterministic_failures = self._check_context_contract(
            state, context, repaired_candidates, semantic_client=None,
        )
        accepted, semantic_failures = await self._audit_context_contract_semantics(
            state, accepted
        )
        return accepted, [*deterministic_failures, *semantic_failures]

    @staticmethod
    def _generator_payload_failure(payload: Any) -> Optional[Dict[str, Any]]:
        """Diagnose a generator payload that cannot contain candidate cards.

        QwenClient may salvage flattened string arrays from a truncated JSON
        response.  That payload is useful for diagnostics, but it is not a
        list of ``HypothesisCard`` objects and must not be mistaken for an
        empty, valid generation.  Legacy wrapper dictionaries containing a
        list of object candidates remain accepted.
        """

        if isinstance(payload, list):
            return None
        if isinstance(payload, dict):
            for key in ("hypotheses", "items", "data"):
                value = payload.get(key)
                if isinstance(value, list) and all(
                    isinstance(item, dict) for item in value
                ):
                    return None
            if payload.get("_parse_error") is True:
                return {
                    "failure_type": "parse_error",
                    "payload_keys": sorted(str(key) for key in payload),
                    "rationale": (
                        "Generator response JSON was truncated or could not be parsed."
                    ),
                }
            return {
                "failure_type": "non_candidate_payload",
                "payload_keys": sorted(str(key) for key in payload),
                "rationale": (
                    "Generator returned a dict that does not contain a list of "
                    "hypothesis objects."
                ),
            }
        return {
            "failure_type": "unexpected_payload_type",
            "payload_keys": [],
            "rationale": (
                "Generator returned unsupported payload type "
                f"{type(payload).__name__}."
            ),
        }

    @staticmethod
    def _hypothesis_generation_schema() -> Dict[str, Any]:
        """Build one strict schema for every new M4 hypothesis response."""

        card_schema = HypothesisCard.model_json_schema()
        item_schema = dict(card_schema)
        item_schema["type"] = "object"
        item_schema["properties"] = dict(card_schema.get("properties", {}))
        item_schema["required"] = [
            "hypothesis_id",
            "statement",
            "mechanism",
            "factual_premises",
            "research_gap",
            "working_assumptions",
            "observable_predictions",
            "falsification_conditions",
            "supporting_evidence",
            "source_paper_ids",
            "task_trace",
            "grounding_status",
        ]
        return {
            "$defs": card_schema.get("$defs", {}),
            "type": "array",
            "items": item_schema,
        }

    @staticmethod
    def _epistemic_audit_schema() -> Dict[str, Any]:
        card_schema = HypothesisCard.model_json_schema()
        return {
            "$defs": card_schema.get("$defs", {}),
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "hypothesis_id": {"type": "string"},
                            "passed": {"type": "boolean"},
                            "hidden_factual_claims": {
                                "type": "array", "items": {"type": "string"}
                            },
                            "overclaim_segments": {
                                "type": "array", "items": {"type": "string"}
                            },
                            "repair_summary": {"type": "string"},
                            "corrected_hypothesis": {
                                "type": "object",
                                "properties": card_schema.get("properties", {}),
                                "required": [
                                    "hypothesis_id",
                                    "statement",
                                    "mechanism",
                                    "factual_premises",
                                    "research_gap",
                                    "working_assumptions",
                                    "observable_predictions",
                                    "falsification_conditions",
                                    "supporting_evidence",
                                    "source_paper_ids",
                                    "task_trace",
                                    "grounding_status",
                                ],
                            },
                        },
                        "required": [
                            "hypothesis_id",
                            "passed",
                            "hidden_factual_claims",
                            "overclaim_segments",
                            "repair_summary",
                            "corrected_hypothesis",
                        ],
                    },
                }
            },
            "required": ["items"],
        }

    def _normalise_hypotheses(self, payload: Any) -> List[HypothesisCard]:
        if isinstance(payload, dict):
            payload = payload.get("hypotheses") or payload.get("items") or payload.get("data") or []

        cards: List[HypothesisCard] = []
        for idx, item in enumerate(payload or [], start=1):
            if not isinstance(item, dict):
                continue
            item = dict(item)
            item.setdefault("hypothesis_id", f"H{idx}")
            item.setdefault("scores", {})
            card = HypothesisCard.model_validate(item)
            if not self._is_scientific_statement(card.statement):
                logger.warning(
                    "M4 discarded editorial/non-claim statement for %s: %s",
                    card.hypothesis_id,
                    card.statement[:160],
                )
                continue
            # A generator payload may contain preliminary scores. Recompute any
            # composite from the four dimensions via the shared rubric
            # (single source of truth for the weights) whenever they are all
            # present, so the displayed score and ranking match the documented
            # formula.
            if all(key in card.scores for key in HYPOTHESIS_DIMENSIONS):
                card.scores["composite"] = composite_score(card.scores, self.weights)
            cards.append(card)
        return cards

    async def _audit_and_repair_epistemic_structure(
        self,
        state: PipelineState,
        cards: List[HypothesisCard],
        context: GraphContext,
    ) -> tuple[List[HypothesisCard], List[EpistemicContractDiagnostic]]:
        """Run exactly one LLM epistemic audit, then deterministic normalization."""

        normalized = [
            normalize_hypothesis_grounding(card, context)
            for card in cards
        ]
        initial = [
            validate_epistemic_contract(card, context)
            for card in normalized
        ]
        emit_event(
            "epistemic_contract_checked",
            module="m4",
            tool="epistemic_boundary_auditor",
            status="running",
            message="M4 epistemic boundary audit started",
            details={
                "candidates": len(normalized),
                "initial_invalid": sum(not item.valid for item in initial),
            },
        )
        if not normalized:
            return [], []

        context_pack = ContextPlanner().plan(
            context,
            ContextRequest(
                purpose="m4_epistemic_audit",
                focus_evidence_ids=tuple(
                    evidence_id
                    for card in normalized
                    for evidence_id in card.supporting_evidence
                ),
            ),
        )
        emit_context_built("m4", "epistemic_boundary_auditor", context_pack)
        try:
            payload = await self._tool_call(
                "epistemic_boundary_auditor",
                self.client.structured_chat(
                    system_prompt=M4_EPISTEMIC_AUDITOR_SYSTEM_PROMPT,
                    user_prompt=M4_EPISTEMIC_AUDITOR_USER_TEMPLATE.format(
                        original_question=(
                            state.problem_card.original_question
                            if state.problem_card else state.input_question
                        ),
                        graph_context=context_pack.rendered,
                        hypotheses_json=json.dumps(
                            [card.model_dump(mode="json") for card in normalized],
                            ensure_ascii=False,
                            indent=2,
                        ),
                        diagnostics_json=json.dumps(
                            [item.model_dump(mode="json") for item in initial],
                            ensure_ascii=False,
                            indent=2,
                        ),
                    ),
                    output_schema=self._epistemic_audit_schema(),
                    max_tokens=8192,
                    temperature=0.0,
                    disable_thinking=True,
                ),
                details={"candidates": len(normalized), "repair_attempt": 1},
            )
        except Exception as exc:
            emit_event(
                "epistemic_contract_failed",
                module="m4",
                tool="epistemic_boundary_auditor",
                status="failed",
                message="M4 epistemic audit had a technical failure",
                details={"error_type": type(exc).__name__},
            )
            return normalized, initial

        raw_items = payload.get("items", []) if isinstance(payload, dict) else []
        repaired_by_id: Dict[str, HypothesisCard] = {}
        repaired_count = 0
        audit_by_id: Dict[str, Dict[str, Any]] = {}
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            hypothesis_id = str(item.get("hypothesis_id") or "")
            corrected = item.get("corrected_hypothesis")
            if not hypothesis_id or not isinstance(corrected, dict):
                continue
            try:
                parsed = self._normalise_hypotheses([corrected])
            except ValidationError as exc:
                logger.warning(
                    "M4 discarded malformed epistemic repair for %s: %s",
                    hypothesis_id,
                    exc,
                )
                emit_event(
                    "epistemic_repair_rejected",
                    module="m4",
                    tool="epistemic_boundary_auditor",
                    status="degraded",
                    message=(
                        "M4 discarded one malformed epistemic repair and "
                        "preserved the original valid hypothesis"
                    ),
                    details={
                        "hypothesis_id": hypothesis_id,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
                continue
            if len(parsed) != 1 or parsed[0].hypothesis_id != hypothesis_id:
                continue
            original = next(
                (card for card in normalized if card.hypothesis_id == hypothesis_id),
                None,
            )
            if original is None:
                continue
            repaired_by_id[hypothesis_id] = normalize_hypothesis_grounding(
                parsed[0], context
            )
            audit_by_id[hypothesis_id] = item
            if parsed[0].model_dump(mode="json") != original.model_dump(mode="json"):
                repaired_count += 1

        output_cards = [
            repaired_by_id.get(card.hypothesis_id, card)
            for card in normalized
        ]
        final_diagnostics: List[EpistemicContractDiagnostic] = []
        for card in output_cards:
            diagnostic = validate_epistemic_contract(card, context)
            audit_item = audit_by_id.get(card.hypothesis_id, {})
            if audit_item.get("hidden_factual_claims") or audit_item.get(
                "overclaim_segments"
            ):
                diagnostic = diagnostic.model_copy(update={
                    "valid": False,
                    "codes": list(dict.fromkeys([
                        *diagnostic.codes,
                        "hidden_factual_overclaim",
                    ])),
                    "messages": list(dict.fromkeys([
                        *diagnostic.messages,
                        str(audit_item.get("repair_summary") or "LLM found hidden factual overclaim"),
                    ])),
                })
            final_diagnostics.append(diagnostic)

        emit_event(
            "epistemic_contract_repaired" if repaired_count else "epistemic_contract_checked",
            module="m4",
            tool="epistemic_boundary_auditor",
            status="completed",
            message="M4 epistemic boundary audit completed",
            details={
                "candidates": len(output_cards),
                "repaired": repaired_count,
                "invalid_after_audit": sum(
                    not item.valid for item in final_diagnostics
                ),
            },
        )
        emit_event(
            "hypothesis_grounding_normalized",
            module="m4",
            tool="grounding_normalizer",
            status="completed",
            message="M4 hypothesis provenance normalized",
            details={
                "statuses": [card.grounding_status for card in output_cards],
                "evidence_counts": [len(card.supporting_evidence) for card in output_cards],
            },
        )
        return output_cards, final_diagnostics

    @staticmethod
    def _merge_generation_candidates(
        existing: List[HypothesisCard],
        incoming: List[HypothesisCard],
    ) -> List[HypothesisCard]:
        """Merge one generator retry without losing distinct hypotheses.

        Models commonly restart IDs at ``H1`` on every call.  IDs are therefore
        made unique here, while repeated statements are discarded so a retry
        cannot inflate the candidate count with duplicates.
        """

        merged = list(existing)
        seen_statements = {
            " ".join(card.statement.casefold().split())
            for card in merged
        }
        used_ids = {card.hypothesis_id for card in merged}
        for card in incoming:
            statement_key = " ".join(card.statement.casefold().split())
            if not statement_key or statement_key in seen_statements:
                continue
            candidate = card
            base_id = str(candidate.hypothesis_id or "H")
            unique_id = base_id
            suffix = 2
            while unique_id in used_ids:
                unique_id = f"{base_id}-retry{suffix}"
                suffix += 1
            if unique_id != candidate.hypothesis_id:
                candidate = candidate.model_copy(update={"hypothesis_id": unique_id})
            merged.append(candidate)
            used_ids.add(unique_id)
            seen_statements.add(statement_key)
        return merged

    def _rank_top(self, cards: List[HypothesisCard]) -> List[HypothesisCard]:
        return sorted(
            cards,
            key=lambda h: h.scores.get("composite", 0),
            reverse=True,
        )[: self.top_k]

    def _attach_rankings(
        self,
        candidates: List[HypothesisCard],
        payload: Any,
        context: GraphContext | None = None,
    ) -> List[HypothesisCard]:
        """Attach ranker-only output to the original candidate objects.

        The ranker is intentionally not trusted to return hypothesis content.
        This makes statements, mechanisms, predictions, and evidence immutable
        across the ranking boundary.
        """
        if isinstance(payload, dict):
            payload = payload.get("rankings") or payload.get("items") or payload.get("data") or []

        by_id = {card.hypothesis_id: card for card in candidates}
        ranked_cards: List[HypothesisCard] = []
        seen: set[str] = set()
        for item in payload or []:
            if not isinstance(item, dict):
                continue
            hypothesis_id = str(item.get("hypothesis_id") or "")
            if hypothesis_id in seen or hypothesis_id not in by_id:
                continue
            raw_scores = item.get("scores")
            if not isinstance(raw_scores, dict):
                continue
            try:
                scores = {
                    dimension: float(raw_scores[dimension])
                    for dimension in HYPOTHESIS_DIMENSIONS
                }
            except (KeyError, TypeError, ValueError):
                continue
            if not all(0.0 <= value <= 1.0 for value in scores.values()):
                continue
            if context is not None and not by_id[hypothesis_id].supporting_evidence:
                # A self-assigned high evidence score cannot survive without a
                # concrete canonical citation. This is intentionally separate
                # from novelty/soundness/testability.
                scores["evidence_consistency"] = min(
                    scores["evidence_consistency"],
                    0.0 if not context.available_evidence_ids else 0.2,
                )
            scores["composite"] = composite_score(scores, self.weights)
            ranked_cards.append(by_id[hypothesis_id].model_copy(update={
                "ranking_rationale": str(item.get("ranking_rationale") or "").strip(),
                "scores": scores,
            }))
            seen.add(hypothesis_id)

        return self._rank_top(ranked_cards)

    # ------------------------------------------------------------------
    # Iteration feedback loop (C): reviews + human guidance → next round
    # ------------------------------------------------------------------

    async def _prompt_user_guidance(self, state: PipelineState) -> str:
        """Claude-Code-style pause: let the user type guidance for this revision
        round, or press Enter to skip.

        No-op (returns "") unless ``interactive`` is set *and* we are attached to
        a real TTY — so tests, pipes, and ``--quiet`` runs never block on stdin.
        """
        if not self.interactive or not sys.stdin.isatty():
            return ""
        try:
            from ..display.panels import render_guidance_prompt, render_guidance_result
            render_guidance_prompt(state.iteration_count)
            text = await asyncio.to_thread(input, "  > ")
        except (EOFError, KeyboardInterrupt):
            return ""
        guidance = text.strip()
        render_guidance_result(guidance)
        return guidance

    async def collect_user_guidance(self, state: PipelineState) -> Dict[str, Any]:
        """Collect the next revision instruction before the M4 header renders.

        ``PipelineRunner`` calls this pre-phase hook before printing the M4
        phase header.  Returning a state patch keeps the captured instruction
        in the normal LangGraph state flow and checkpoint output.
        """
        if state.iteration_count <= 0:
            return {}
        extra = await self._prompt_user_guidance(state)
        if not extra:
            return {}
        guidance = list(state.user_guidance)
        guidance.append(f"[iter {state.iteration_count}] {extra}")
        return {"user_guidance": guidance}

    def _build_feedback_context(self, state: PipelineState, guidance: List[str]) -> str:
        """Assemble reviewer feedback + prior hypotheses + human guidance into a
        revision block for the generator (empty string on the first round,
        except in followup runs where the follow-up instruction is always
        injected)."""
        followup_block = _followup_requirement_block(state)
        if state.iteration_count <= 0:
            return followup_block

        parts: List[str] = [f"\n--- Revision guidance (iteration {state.iteration_count}) ---"]

        prior_top = [
            hypothesis
            for hypothesis in state.top_hypotheses
            if self._is_scientific_statement(hypothesis.statement)
        ]
        if prior_top:
            parts.append("Prior top hypotheses (revise these, don't restart):")
            parts.extend(f"- [{h.hypothesis_id}] {h.statement}" for h in prior_top)

        latest = max((r.version for r in state.reviews), default=0)
        recent = [
            r for r in state.reviews
            if r.version == latest
            and r.dimension.value != "overall"
            and getattr(r, "attribution", "both") in {"hypothesis", "both"}
        ]
        if recent:
            parts.append("\nReviewer feedback (address these):")
            for r in recent:
                fix = " ".join(filter(None, [r.reasoning, r.suggestions, r.comments]))
                parts.append(f"- [{r.dimension.value}] {r.score:.1f}/5 — {fix}")

        if guidance:
            parts.append("\nUser guidance (highest priority — follow closely):")
            parts.extend(f"- {g}" for g in guidance)

        parts.append("")
        return followup_block + "\n".join(parts)

    @staticmethod
    def _is_supplement_reentry(state: PipelineState) -> bool:
        history = list(state.routing_history or [])
        if not history:
            return False
        latest = history[-1]
        if (
            latest.to_module == "supplement_m2"
            and latest.from_module in {"m4", "m6"}
        ):
            return True
        if (
            latest.from_module != "m3"
            or latest.to_module not in {"m4", "m5", "m2"}
            or len(history) < 2
        ):
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
    def _reconcile_reentry_portfolio(state: PipelineState):
        """Apply current gap outcomes to the active portfolio in place of a reset."""
        gaps = [*state.evidence_gap_requests, *state.evidence_gaps]
        # Re-entry mutates premise provenance and may attach an explicit M3
        # bridge node.  Rebuild the authoritative context once so every card
        # is normalized against the same evidence/bridge whitelist before it
        # is handed to M5/M6; otherwise stale card-level citations can survive
        # a supplement round.
        context = build_graph_context(state)
        contradicted: set[str] = set()
        by_hypothesis: dict[str, list[Any]] = {}
        for gap in gaps:
            ids = list(getattr(gap, "hypothesis_ids", []) or [])
            hid = str(getattr(gap, "hypothesis_id", "") or "")
            if hid:
                ids.append(hid)
            for item in dict.fromkeys(ids):
                by_hypothesis.setdefault(item, []).append(gap)
            if getattr(gap, "scientific_resolution", "unreviewed") == "contradicted":
                contradicted.update(ids)

        def update(cards):
            output = []
            for card in cards:
                if card.hypothesis_id in contradicted:
                    continue
                new = card.model_copy(deep=True)
                touched = False
                for gap in by_hypothesis.get(card.hypothesis_id, []):
                    resolution = getattr(gap, "scientific_resolution", "unreviewed")
                    premise_id = str(getattr(gap, "premise_id", "") or "")
                    if resolution == "supported" and premise_id:
                        touched = True
                        for index, premise in enumerate(new.factual_premises):
                            if premise.premise_id != premise_id:
                                continue
                            new.factual_premises[index] = premise.model_copy(update={
                                "supporting_evidence_ids": list(dict.fromkeys(
                                    getattr(gap, "resolution_evidence_ids", [])
                                )),
                                "claim": getattr(gap, "corrected_claim", "") or premise.claim,
                                "audit_verdict": "supported",
                            })
                        new.supporting_evidence = list(dict.fromkeys([
                            *new.supporting_evidence,
                            *getattr(gap, "resolution_evidence_ids", []),
                        ]))
                    elif resolution == "unresolved":
                        bridge_id = getattr(gap, "bridge_hypothesis_node_id", "")
                        if not bridge_id:
                            continue
                        touched = True
                        new.factual_premises = [
                            premise for premise in new.factual_premises
                            if premise.premise_id != premise_id
                        ]
                        if not any(item.bridge_hypothesis_node_id == bridge_id for item in new.working_assumptions):
                            new.working_assumptions.append(HypothesisPremise(
                                premise_id=f"BRIDGE_{bridge_id}",
                                claim=getattr(gap, "audit_claim", "") or getattr(gap, "description", "") or getattr(gap, "sub_question", ""),
                                kind="unverified_bridge",
                                bridge_hypothesis_node_id=bridge_id,
                                origin_gap_id=gap.gap_id,
                                required=True,
                            ))
                output.append(
                    normalize_hypothesis_grounding(new, context)
                    if touched else card
                )
            return output

        return update(state.candidate_hypotheses), update(state.top_hypotheses), contradicted

    @staticmethod
    def _supplement_focus_evidence_ids(state: PipelineState) -> list[str]:
        ids: list[str] = []
        history = list(state.routing_history or [])
        latest = history[-1] if history and history[-1].to_module == "supplement_m2" else (
            history[-2] if len(history) >= 2 and history[-1].from_module == "m3" and history[-2].to_module == "supplement_m2" else None
        )
        current_gap_ids = set(latest.gap_ids) if latest and latest.gap_ids else set()
        gaps = [
            gap for gap in [*state.evidence_gap_requests, *state.evidence_gaps]
            if not current_gap_ids or str(getattr(gap, "gap_id", "")) in current_gap_ids
        ]
        for gap in gaps:
            ids.extend(getattr(gap, "resolution_evidence_ids", []) or [])
            ids.extend(getattr(gap, "contradicting_evidence_ids", []) or [])
            gap_id = str(getattr(gap, "gap_id", "") or "")
            for result in state.literature_results:
                if gap_id not in result.origin_gap_ids and gap_id not in result.origin_gap_request_ids:
                    continue
                for entry in result.knowledge_entries:
                    ids.extend(entry.evidence_ids)
        if state.m2_knowledge_export:
            for run in state.m2_knowledge_export.runs:
                if any(
                    str(getattr(gap, "gap_id", "")) in (*run.origin_gap_ids, *run.origin_gap_request_ids)
                    for gap in gaps
                ):
                    ids.extend(item.evidence_id for item in run.evidence)
                    for entry in run.knowledge_entries:
                        ids.extend(entry.evidence_ids)
        return list(dict.fromkeys(str(item) for item in ids if str(item).strip()))

    def _update_best(self, state: PipelineState, top: List[HypothesisCard]) -> List[HypothesisCard]:
        """Keep the best hypotheses seen across all iterations, so a weaker
        revision cannot discard a stronger earlier result."""
        pool = [
            card
            for card in list(top) + list(state.best_hypotheses or [])
            if self._is_scientific_statement(card.statement)
        ]
        dedup: Dict[str, HypothesisCard] = {}
        for card in pool:
            key = card.statement.strip()
            existing = dedup.get(key)
            if existing is None or card.scores.get("composite", 0) > existing.scores.get("composite", 0):
                dedup[key] = card
        return self._rank_top(list(dedup.values()))

    @staticmethod
    def _sync_audited_top_to_candidates(
        candidates: List[HypothesisCard],
        top: List[HypothesisCard],
    ) -> List[HypothesisCard]:
        """Replace stale ranked candidates with their final audited copies."""

        audited_by_id = {card.hypothesis_id: card for card in top}
        synchronized = [
            audited_by_id.get(card.hypothesis_id, card)
            for card in candidates
        ]
        known_ids = {card.hypothesis_id for card in synchronized}
        synchronized.extend(
            card for card in top if card.hypothesis_id not in known_ids
        )
        return synchronized

    @staticmethod
    def _attach_bridge_assumptions(
        state: PipelineState,
        hypotheses: List[HypothesisCard],
        context: GraphContext,
    ) -> List[HypothesisCard]:
        """Expose M3's unverified bridge nodes as explicit M4 assumptions.

        The bridge text is copied from the graph context and is never given an
        Evidence ID.  This deterministic attachment keeps a model that omits
        the optional field from silently losing the unresolved relation.
        """
        if not context.bridge_hypotheses:
            return hypotheses
        active_bridge_ids = active_bridge_ids_for_state(state)
        gap_by_bridge = {
            gap.bridge_hypothesis_node_id: gap
            for gap in state.evidence_gap_requests
            if gap.bridge_hypothesis_node_id
        }
        for hypothesis in hypotheses:
            existing_ids = {
                item.bridge_hypothesis_node_id
                for item in hypothesis.working_assumptions
            }
            assumptions = list(hypothesis.working_assumptions)
            for item in context.bridge_hypotheses:
                if item.entry_id not in active_bridge_ids:
                    continue
                if item.entry_id in existing_ids:
                    continue
                gap = gap_by_bridge.get(item.entry_id)
                assumptions.append(HypothesisPremise(
                    premise_id=f"BRIDGE_{item.entry_id}",
                    claim=item.text,
                    kind="unverified_bridge",
                    required=True,
                    bridge_hypothesis_node_id=item.entry_id,
                    origin_gap_id=gap.gap_id if gap else "",
                ))
                existing_ids.add(item.entry_id)
            hypothesis.working_assumptions = assumptions
        return hypotheses

    async def _audit_hypotheses(
        self,
        state: PipelineState,
        hypotheses: List[HypothesisCard],
        context: GraphContext,
    ) -> List[EvidenceGapRequest]:
        """Audit explicit factual premises and create bounded search gaps.

        The new contract audits only ``factual_premises``.  Mechanisms,
        predictions and the research gap are proposed research content and do
        not trigger M2.  A compatibility branch keeps the old claim auditor
        usable for pre-contract test/checkpoint objects that have no premise
        list and expose only ``audit_claims``.
        """
        existing = {
            gap.gap_id: gap.model_copy(deep=True)
            for gap in state.evidence_gap_requests
        }
        by_premise = {
            (gap.hypothesis_id, gap.premise_id): gap
            for gap in existing.values()
            if gap.premise_id.strip()
        }
        by_claim = {
            (gap.hypothesis_id, gap.audit_claim.strip().casefold()): gap
            for gap in existing.values()
            if gap.audit_claim.strip()
        }
        auditor = getattr(self, "evidence_auditor", None)
        if auditor is None:
            auditor = EvidenceAuditService(getattr(self, "client", None))

        for hypothesis in hypotheses:
            premises = list(hypothesis.factual_premises)
            if premises and hasattr(auditor, "audit_premises"):
                verdicts = await auditor.audit_premises(
                    premises,
                    state.evidence_graph,
                    candidate_evidence_ids=context.available_evidence_ids,
                )
                verdict_by_id = {item.premise_id: item for item in verdicts}
                for premise in premises:
                    if not premise.required or premise.kind != "evidence_backed":
                        continue
                    claim = premise.claim.strip()
                    if not claim:
                        continue
                    previous = by_premise.get((hypothesis.hypothesis_id, premise.premise_id))
                    verdict = verdict_by_id.get(premise.premise_id, PremiseAuditResult(
                        premise_id=premise.premise_id,
                        claim=claim,
                        verdict="unsupported",
                        rationale="Premise auditor returned no matching verdict.",
                    ))
                    audit_verdict = verdict.verdict
                    corrected_claim = verdict.corrected_claim.strip()
                    supporting_ids = (
                        list(dict.fromkeys(verdict.evidence_ids))
                        if audit_verdict in {"supported", "partially_supported"}
                        else []
                    )
                    premise_update: Dict[str, Any] = {
                        "audit_verdict": audit_verdict,
                        "audit_reason": verdict.rationale,
                        "supporting_evidence_ids": supporting_ids,
                    }
                    if audit_verdict == "partially_supported" and corrected_claim:
                        premise_update.update({
                            "original_claim": premise.claim,
                            "claim": corrected_claim,
                        })
                    hypothesis.factual_premises = [
                        item.model_copy(update=(
                            premise_update if item.premise_id == premise.premise_id else {}
                        ))
                        for item in hypothesis.factual_premises
                    ]
                    if previous is not None and previous.status == "hypothesized":
                        existing[previous.gap_id] = previous
                        continue
                    fingerprint = "|".join([
                        hypothesis.hypothesis_id,
                        premise.premise_id,
                        "supporting_evidence",
                    ])
                    gap_id = previous.gap_id if previous is not None else (
                        "GAP_" + hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:12]
                    )
                    if verdict.verdict in {"supported", "partially_supported"} and verdict.evidence_ids:
                        if previous is not None:
                            existing[gap_id] = previous.model_copy(update={
                                "status": "resolved",
                                "scientific_resolution": "supported",
                                "search_completed": True,
                                "resolution_evidence_ids": list(dict.fromkeys(verdict.evidence_ids)),
                                "corrected_claim": verdict.corrected_claim,
                                "hypothesis_id": hypothesis.hypothesis_id,
                                "rationale": verdict.rationale,
                            })
                        continue
                    if verdict.verdict == "contradicted":
                        existing[gap_id] = EvidenceGapRequest(
                            gap_id=gap_id,
                            hypothesis_id=hypothesis.hypothesis_id,
                            premise_id=premise.premise_id,
                            sub_question=f"What evidence tests whether {claim}",
                            audit_claim=claim,
                            suggested_queries=[claim],
                            executed_queries=(previous.executed_queries if previous else []),
                            status="resolved",
                            attempts=(previous.attempts if previous else 0),
                            created_iteration=(
                                previous.created_iteration
                                if previous is not None else state.iteration_count
                            ),
                            scientific_resolution="contradicted",
                            search_completed=True,
                            contradicting_evidence_ids=list(dict.fromkeys(verdict.evidence_ids)),
                            rationale=verdict.rationale or "The factual premise is contradicted by canonical evidence.",
                        )
                        by_premise[(hypothesis.hypothesis_id, premise.premise_id)] = existing[gap_id]
                        continue
                    attempts = previous.attempts if previous is not None else 0
                    status = "exhausted" if attempts >= state.max_evidence_gap_rounds else "pending"
                    existing[gap_id] = EvidenceGapRequest(
                        gap_id=gap_id,
                        hypothesis_id=hypothesis.hypothesis_id,
                        premise_id=premise.premise_id,
                        sub_question=f"What evidence tests whether {claim}",
                        audit_claim=claim,
                        suggested_queries=[claim],
                        executed_queries=(previous.executed_queries if previous else []),
                        status=status,
                        attempts=attempts,
                        created_iteration=(
                            previous.created_iteration
                            if previous is not None else state.iteration_count
                        ),
                        scientific_resolution="unreviewed",
                        search_completed=False,
                        rationale=verdict.rationale or (
                            "The supplied literature is related to the premise but does not "
                            "entail the complete factual claim."
                        ),
                    )
                    by_premise[(hypothesis.hypothesis_id, premise.premise_id)] = existing[gap_id]
                continue

            # Compatibility for old cards and old test doubles.  The real new
            # auditor has ``audit_premises``; new M4 output therefore cannot
            # route a mechanism or prediction as an evidence gap.
            if premises or hasattr(auditor, "audit_premises") or not hasattr(auditor, "audit_claims"):
                continue
            claims = auditor.decompose_hypothesis(hypothesis)
            if not claims:
                claims = [hypothesis.mechanism or hypothesis.statement]
            verdicts = await auditor.audit_claims(
                claims,
                state.evidence_graph,
                candidate_evidence_ids=context.available_evidence_ids,
            )
            for verdict in verdicts:
                claim = verdict.claim.strip()
                if not claim:
                    continue
                previous = by_claim.get((hypothesis.hypothesis_id, claim.casefold()))
                if previous is not None and previous.status == "hypothesized":
                    existing[previous.gap_id] = previous
                    continue
                if verdict.support_status in {"direct_support", "partial_support"} and verdict.evidence_ids:
                    if previous is not None:
                        existing[previous.gap_id] = previous.model_copy(update={
                            "status": "resolved",
                            "scientific_resolution": "supported",
                            "search_completed": True,
                            "resolution_evidence_ids": list(dict.fromkeys(verdict.evidence_ids)),
                            "hypothesis_id": hypothesis.hypothesis_id,
                            "rationale": verdict.rationale,
                        })
                    continue
                fingerprint = "|".join([
                    hypothesis.hypothesis_id,
                    claim.casefold(),
                    "supporting_evidence",
                ])
                gap_id = previous.gap_id if previous is not None else (
                    "GAP_" + hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:12]
                )
                attempts = previous.attempts if previous is not None else 0
                status = "exhausted" if attempts >= state.max_evidence_gap_rounds else "pending"
                existing[gap_id] = EvidenceGapRequest(
                    gap_id=gap_id,
                    hypothesis_id=hypothesis.hypothesis_id,
                    sub_question=f"What evidence tests whether {claim}",
                    audit_claim=claim,
                    suggested_queries=[claim],
                    executed_queries=(previous.executed_queries if previous else []),
                    status=status,
                    attempts=attempts,
                    created_iteration=(
                        previous.created_iteration
                        if previous is not None else state.iteration_count
                    ),
                    rationale=verdict.rationale or (
                        "The supplied literature is related to the claim but does not "
                        "entail the complete causal step."
                    ),
                )
                by_claim[(hypothesis.hypothesis_id, claim.casefold())] = existing[gap_id]
        return list(existing.values())

    @staticmethod
    def _update_evidence_gaps(
        state: PipelineState,
        hypotheses: List[HypothesisCard],
        context: GraphContext,
    ) -> List[EvidenceGapRequest]:
        """Create/deduplicate bounded search requests before M5 is allowed to run."""

        existing = {
            gap.gap_id: gap.model_copy(deep=True)
            for gap in state.evidence_gap_requests
        }
        card = state.problem_card
        synthesis_contract = synthesis_contract_for_state(state)
        requirements = {
            item.requirement_id: item
            for item in synthesis_contract.requirements
            if item.required
        }
        valid_evidence = set(context.available_evidence_ids)

        for hypothesis in hypotheses:
            cited = list(dict.fromkeys(
                evidence_id for evidence_id in hypothesis.supporting_evidence
                if evidence_id in valid_evidence
            ))
            traced_requirement_ids = [
                reference.contract_id
                for reference in hypothesis.task_trace.requirement_mentions
                if reference.contract_id in requirements
            ]
            if not traced_requirement_ids:
                traced_requirement_ids = list(requirements)

            targets = traced_requirement_ids or [""]
            for requirement_id in targets:
                requirement = requirements.get(requirement_id)
                sub_question = (
                    requirement.sub_question if requirement
                    else (card.original_question if card else state.input_question)
                )
                relation = requirement.relation if requirement else hypothesis.mechanism
                entity_ids = list(dict.fromkeys(filter(None, [
                    requirement.primary_entity_id if requirement else "",
                    *(requirement.related_entity_ids if requirement else []),
                ])))
                fingerprint = "|".join([
                    requirement_id or sub_question.casefold(),
                    "supporting_evidence",
                ])
                gap_id = "GAP_" + hashlib.sha256(
                    fingerprint.encode("utf-8")
                ).hexdigest()[:12]
                previous = existing.get(gap_id)
                if cited:
                    if previous is not None:
                        existing[gap_id] = previous.model_copy(update={
                            "status": "resolved",
                            "resolution_evidence_ids": cited,
                            "hypothesis_id": hypothesis.hypothesis_id,
                        })
                    continue

                attempts = previous.attempts if previous else 0
                status = (
                    "exhausted"
                    if attempts >= state.max_evidence_gap_rounds
                    else "pending"
                )
                existing[gap_id] = EvidenceGapRequest(
                    gap_id=gap_id,
                    hypothesis_id=hypothesis.hypothesis_id,
                    requirement_id=requirement_id,
                    sub_question=sub_question,
                    task_entity_ids=entity_ids,
                    relation=relation,
                    suggested_queries=[sub_question],
                    executed_queries=(previous.executed_queries if previous else []),
                    status=status,
                    attempts=attempts,
                    created_iteration=(
                        previous.created_iteration
                        if previous else state.iteration_count
                    ),
                    rationale=(
                        "No canonical supporting evidence ID was attached to a "
                        "hypothesis that addresses this task requirement."
                    ),
                )

        return list(existing.values())

    async def _generate_hypothesis_batch(
        self,
        state: PipelineState,
        *,
        question: str,
        graph_context: GraphContext,
        feedback_context: str,
        tool_name: str,
        attempt: int,
        requested_count: Optional[int] = None,
        disable_thinking: Optional[bool] = None,
    ) -> tuple[Any, List[HypothesisCard], List[Dict[str, Any]]]:
        """Run one generator pass and apply the normal context gates.

        ``requested_count`` overrides ``self.num_candidates`` (used by gate
        recovery to keep the regeneration pass small).
        """

        assert self.client is not None
        requested = (
            self.num_candidates if requested_count is None else int(requested_count)
        )
        focus_ids = [
            *self._supplement_focus_evidence_ids(state),
            *(
                evidence_id
                for card in [*state.candidate_hypotheses, *state.top_hypotheses]
                for evidence_id in card.supporting_evidence
            ),
        ]
        context_pack = ContextPlanner().plan(
            graph_context,
            ContextRequest(
                purpose="m4_generate",
                focus_evidence_ids=tuple(dict.fromkeys(focus_ids)),
            ),
        )
        emit_context_built("m4", tool_name, context_pack)
        generated = await self._tool_call(
            tool_name,
            self.client.structured_chat(
                system_prompt=M4_GENERATOR_SYSTEM_PROMPT.format(
                    num_candidates=requested,
                    rubric_block=hypothesis_rubric_block(),
                ),
                user_prompt=M4_GENERATOR_USER_TEMPLATE.format(
                    graph_context=context_pack.rendered,
                    knowledge_gaps="(included in the audited context pack above)",
                    established_facts="(included in the audited context pack above)",
                    conflicts="(included in the audited context pack above)",
                    original_question=question,
                    num_candidates=requested,
                    feedback_context=feedback_context,
                    task_contract_block=self._render_task_contract_block(state),
                ),
                    output_schema=self._hypothesis_generation_schema(),
                max_tokens=8192,
                temperature=max(getattr(self.llm_config, "temperature", 0.1), 0.4),
                disable_thinking=(
                    # The first standard pass keeps open-ended reasoning.
                    # Later passes already have prior hypotheses, reviewer
                    # feedback, and an updated graph; avoid repeating the
                    # observed 360-second reasoning tail.
                    (self.fast_mode or state.iteration_count > 0)
                    if disable_thinking is None
                    else bool(disable_thinking)
                ),
            ),
            details={
                "requested_candidates": requested,
                "iteration": state.iteration_count + 1,
                "attempt": attempt,
            },
        )
        candidates = self._canonicalize_task_traces(
            state, self._normalise_hypotheses(generated),
        )
        candidates, contract_failures = self._check_context_contract(
            state, graph_context, candidates, semantic_client=None,
        )
        if self.fast_mode:
            # Fast mode skips the extra semantic-object LLM audit; the
            # deterministic contract check is enough to keep the run moving.
            semantic_failures = []
        else:
            candidates, semantic_failures = (
                await self._audit_context_contract_semantics(
                    state, candidates
                )
            )
        contract_failures.extend(semantic_failures)
        payload_failure = self._generator_payload_failure(generated)
        if payload_failure is not None:
            contract_failures.insert(0, payload_failure)
        return generated, candidates, contract_failures

    def _fast_fallback_hypotheses(
        self,
        state: PipelineState,
        question: str,
    ) -> List[HypothesisCard]:
        """Deterministic last-resort hypotheses so fast mode never stalls.

        These are deliberately generic and marked as fast-mode fallback; they
        are only used when the LLM/contract pipeline produced no usable
        hypothesis at all.
        """
        q = " ".join(str(question or state.input_question or "").split())
        if not q:
            q = "the given scientific question"
        statements = [
            (
                f"A systematic experimental investigation of '{q}' is needed to "
                "identify the primary causal or mechanistic factors."
            ),
            (
                f"A controlled multi-metric comparison of candidate approaches to "
                "'{q}' can reveal which strategy yields the most robust improvement."
            ),
            (
                f"An iterative model-building and validation cycle is the most "
                "practical route to making progress on '{q}'."
            ),
        ]
        return [
            HypothesisCard(
                hypothesis_id=f"FH{index}",
                statement=statement,
                mechanism=(
                    "Fast-mode fallback: systematic investigation, controlled "
                    "comparison, and iterative validation."
                ),
                observable_predictions=[
                    "The proposed approach produces measurable, reproducible progress.",
                    "Comparator baselines differ in at least one clearly defined metric.",
                    "Iteration improves the primary outcome compared with a single-pass baseline.",
                ],
                falsification_conditions=[
                    "No measurable difference is observed between intervention and control.",
                    "The effect cannot be reproduced in an independent run.",
                ],
                scores={"composite": 0.5 - index * 0.01},
                ranking_rationale="Fast-mode deterministic fallback hypothesis.",
            )
            for index, statement in enumerate(statements, start=1)
        ]

    def _continuity_fallback_hypotheses(
        self,
        state: PipelineState,
        question: str,
        reason: str,
    ) -> List[HypothesisCard]:
        """Create an auditable low-confidence candidate after M4 recovery.

        This is a pipeline-continuity mechanism, not a scientific conclusion.
        It deliberately carries no evidence or paper identifiers and is marked
        in both its score and ranking rationale so downstream M5/M6 can expose
        the missing grounding instead of treating it as a normal result.
        """

        base = self._fast_fallback_hypotheses(state, question)
        safe_reason = " ".join(str(reason or "recovery exhausted").split())
        cards = [
            card.model_copy(update={
                "scores": {
                    **card.scores,
                    "composite": min(
                        float(card.scores.get("composite", 0.2)), 0.2
                    ),
                },
                "ranking_rationale": (
                    "Deterministic continuity fallback after bounded M4 recovery "
                    f"was exhausted: {safe_reason}"
                ),
                "supporting_evidence": [],
                "source_paper_ids": [],
            })
            for card in base[: max(1, self.top_k)]
        ]
        return self._canonicalize_task_traces(state, cards)

    async def _run_llm(self, state: PipelineState, feedback_context: str = "") -> Dict[str, Any]:
        assert self.client is not None
        question = state.problem_card.original_question if state.problem_card else state.input_question
        graph_context = build_graph_context(state)

        # Iteration feedback (e.g. "be more novel") must not push the
        # generator away from task alignment: re-anchor it to the required
        # task scope first.
        if feedback_context and state.problem_card is not None:
            feedback_context = self._task_contract_reminder(state) + feedback_context

        # ── Step 1: Generator ──────────────────────────────────────────
        generator_timeout_recovery_failed = False
        try:
            generated, candidates, contract_failures = (
                await self._generate_hypothesis_batch(
                    state,
                    question=question,
                    graph_context=graph_context,
                    feedback_context=feedback_context,
                    tool_name="hypothesis_generator",
                    attempt=1,
                )
            )
        except TimeoutError as exc:
            if self.fast_mode:
                logger.warning(
                    "M4 fast-mode generator timed out; using deterministic "
                    "fallback hypotheses so the run can continue."
                )
                generated = []
                candidates = self._fast_fallback_hypotheses(state, question)
                contract_failures = []
            else:
                # A per-call timeout is recoverable; exhaustion of the module's
                # shared budget is not.  Preserve the outer deadline contract
                # instead of starting a recovery call with no time remaining.
                if (
                    self._run_deadline is not None
                    and self._run_deadline <= time.monotonic()
                ):
                    raise
                logger.warning(
                    "M4 reasoning-enabled generator timed out (%s); retrying "
                    "once without extended thinking.",
                    exc,
                )
                try:
                    generated, candidates, contract_failures = (
                        await self._generate_hypothesis_batch(
                            state,
                            question=question,
                            graph_context=graph_context,
                            feedback_context=feedback_context,
                            tool_name="hypothesis_generator_timeout_recovery",
                            attempt=2,
                            disable_thinking=True,
                        )
                    )
                except Exception as recovery_exc:
                    generator_timeout_recovery_failed = True
                    generated = []
                    candidates = []
                    contract_failures = [{
                        "failure_type": "generator_timeout_recovery_failed",
                        "rationale": (
                            "Generator timeout recovery failed: "
                            f"{type(recovery_exc).__name__}: "
                            f"{str(recovery_exc).strip()}"
                        ).rstrip(": "),
                    }]
                    emit_event(
                        "tool_result",
                        module="m4",
                        tool="hypothesis_generator_timeout_recovery",
                        status="warning",
                        message=(
                            "M4 generator timeout recovery failed; continuing to "
                            "bounded contract repair/degradation"
                        ),
                        details={"error_type": type(recovery_exc).__name__},
                    )
        except Exception as exc:
            if not self.fast_mode:
                raise
            logger.warning(
                "M4 fast-mode generator failed (%s: %s); using deterministic "
                "fallback hypotheses so the run can continue.",
                type(exc).__name__,
                exc,
            )
            generated = []
            candidates = self._fast_fallback_hypotheses(state, question)
            contract_failures = []
        all_generated = list(self._normalise_hypotheses(generated))

        # A short — or empty — generator response is recoverable.  Give the
        # model exactly one additional pass, then accept the smaller set if
        # the problem genuinely does not yield the requested number.  When
        # contract validation discarded candidates, their reasons are fed back
        # so the retry can fix task alignment.  An empty set that survives the
        # retry still falls through to the deterministic repair path below.
        # A smaller-than-requested portfolio is usable once it can satisfy the
        # configured output cardinality.  Retrying merely to fill the optional
        # candidate pool consumed a second generator + semantic-audit pass and
        # was a major source of M4 timeouts.
        if not generator_timeout_recovery_failed and len(candidates) < self.top_k:
            missing_count = max(1, self.top_k - len(candidates))
            retry_feedback = (
                f"The previous pass produced only {len(candidates)} valid hypotheses. "
                f"Generate exactly {missing_count} additional hypothesis object(s). "
                "Return concise valid JSON; preserve the original-question scope "
                "and do not repeat existing hypotheses."
            )
            if contract_failures:
                rationale_lines = [
                    str(item.get("rationale") or "contract validation failed")
                    for item in contract_failures[:5]
                ]
                retry_feedback += (
                    "\nContract validation rejected some candidates. Address "
                    "these reasons explicitly (keep required task entities and "
                    "auditable requirement traces in scope):\n- "
                    + "\n- ".join(rationale_lines)
                )
            try:
                retry_generated, retry_candidates, retry_failures = (
                    await self._generate_hypothesis_batch(
                        state,
                        question=question,
                        graph_context=graph_context,
                        feedback_context="\n".join(
                            item for item in [feedback_context, retry_feedback] if item
                        ),
                        tool_name="hypothesis_generator_retry",
                        attempt=2,
                        requested_count=missing_count,
                        disable_thinking=True,
                    )
                )
            except Exception as exc:
                contract_failures.append({
                    "failure_type": "shortfall_retry_failed",
                    "rationale": (
                        "Shortfall retry failed: "
                        f"{type(exc).__name__}: {str(exc).strip()}"
                    ).rstrip(": "),
                })
                emit_event(
                    "tool_result",
                    module="m4",
                    tool="hypothesis_generator_shortfall_recovery",
                    status="warning",
                    message=(
                        "M4 shortfall retry failed; preserving valid candidates "
                        "and continuing to bounded repair/degradation"
                    ),
                    details={
                        "valid_candidates": len(candidates),
                        "missing_candidates": missing_count,
                        "error_type": type(exc).__name__,
                    },
                )
                if self.fast_mode:
                    logger.warning(
                        "M4 fast-mode generator retry failed (%s: %s); using "
                        "deterministic fallback hypotheses.",
                        type(exc).__name__,
                        exc,
                    )
                    candidates = self._fast_fallback_hypotheses(state, question)
            else:
                all_generated.extend(self._normalise_hypotheses(retry_generated))
                candidates = self._merge_generation_candidates(
                    candidates, retry_candidates
                )
                contract_failures.extend(retry_failures)
        generation_shortfall = len(candidates) < self.top_k

        degraded_generation = False
        if not candidates and self.fast_mode:
            logger.warning(
                "M4 fast mode found no contract-valid hypotheses after one "
                "generator retry; skipping the additional LLM repair call and "
                "using deterministic fallback hypotheses."
            )
            candidates = self._fast_fallback_hypotheses(state, question)
        elif not candidates:
            try:
                candidates, repair_failures = await self._repair_context_contract(
                    state=state,
                    context=graph_context,
                    candidates=all_generated,
                    failures=contract_failures,
                    feedback_context=feedback_context,
                    requested_count=self.top_k,
                    disable_thinking=True,
                )
            except Exception as exc:
                contract_failures.append({
                    "failure_type": "contract_repair_failed",
                    "rationale": (
                        "Contract repair failed: "
                        f"{type(exc).__name__}: {str(exc).strip()}"
                    ).rstrip(": "),
                })
                emit_event(
                    "tool_result",
                    module="m4",
                    tool="hypothesis_contract_repair",
                    status="warning",
                    message=(
                        "M4 contract repair failed; using deterministic continuity "
                        "degradation instead of terminating the pipeline"
                    ),
                    details={"error_type": type(exc).__name__},
                )
                candidates = []
            else:
                contract_failures.extend(repair_failures)
        if not candidates:
            if self.fast_mode:
                logger.warning(
                    "M4 fast mode found no contract-valid hypotheses; using "
                    "deterministic fallback hypotheses so the run can continue."
                )
                candidates = self._fast_fallback_hypotheses(state, question)
            else:
                diagnostics = "; ".join(
                    str(item.get("rationale") or "contract validation failed")
                    for item in contract_failures[-3:]
                )
                candidates = self._continuity_fallback_hypotheses(
                    state,
                    question,
                    diagnostics or "no contract-valid hypothesis was produced",
                )
                degraded_generation = True
                emit_event(
                    "tool_result",
                    module="m4",
                    tool="hypothesis_generator_continuity_fallback",
                    status="warning",
                    message=(
                        "M4 bounded generator recovery was exhausted; continuing "
                        "with an auditable low-confidence hypothesis"
                    ),
                    details={
                        "candidate_count": len(candidates),
                        "top_k": self.top_k,
                        "evidence_links": 0,
                    },
                )

        # The boundary audit is deliberately after generator recovery and
        # before Critic/Ranker.  Fast mode keeps its bounded deterministic path;
        # normal mode gets exactly one semantic check for every new batch.
        has_audit_context = bool(
            graph_context.established_facts
            or graph_context.conflicts
            or graph_context.knowledge_gaps
            or graph_context.bridge_hypotheses
            or graph_context.available_evidence_ids
        )
        if (
            candidates
            and not self.fast_mode
            and not degraded_generation
            and has_audit_context
        ):
            candidates, epistemic_diagnostics = (
                await self._audit_and_repair_epistemic_structure(
                    state, candidates, graph_context
                )
            )
            contract_failures.extend(
                {
                    "failure_type": "epistemic_contract",
                    "hypothesis_id": diagnostic.hypothesis_id,
                    "rationale": "; ".join(diagnostic.messages),
                    "codes": list(diagnostic.codes),
                }
                for diagnostic in epistemic_diagnostics
                if not diagnostic.valid
            )

        if degraded_generation:
            top = self._rank_top(candidates)[: self.top_k]
            return {
                "candidate_hypotheses": candidates,
                "top_hypotheses": top,
            }

        if self.mode != "multi_agent":
            top = self._rank_top(candidates)
            return {
                "candidate_hypotheses": candidates,
                "top_hypotheses": top[: self.top_k],
            }

        # ── Steps 2–3: Critic + Falsifiability quality gates ──────────
        candidates = await self._run_quality_gates(
            state,
            candidates,
            generation_shortfall=generation_shortfall,
            question=question,
            graph_context=graph_context,
            feedback_context=feedback_context,
        )

        # ── Step 4: Ranker (separate model tier) ───────────────────────
        ranker_client = self.ranker_client or self.client
        ranker_item_schema = {
            "type": "object",
            "properties": {
                "hypothesis_id": {"type": "string"},
                "ranking_rationale": {"type": "string"},
                "scores": {
                    "type": "object",
                    "properties": {
                        dimension: {"type": "number", "minimum": 0, "maximum": 1}
                        for dimension in HYPOTHESIS_DIMENSIONS
                    },
                    "required": list(HYPOTHESIS_DIMENSIONS),
                },
            },
            "required": ["hypothesis_id", "ranking_rationale", "scores"],
        }
        ranker_schema = {"type": "array", "items": ranker_item_schema}
        rank_context = ContextPlanner().plan(
            graph_context,
            ContextRequest(
                purpose="m4_rank",
                focus_evidence_ids=tuple(
                    dict.fromkeys([
                        *self._supplement_focus_evidence_ids(state),
                        *(
                            evidence_id
                            for candidate in candidates
                            for evidence_id in candidate.supporting_evidence
                        ),
                    ])
                ),
            ),
        )
        emit_context_built("m4", "hypothesis_ranker", rank_context)
        try:
            ranked = await self._tool_call(
                "hypothesis_ranker",
                ranker_client.structured_chat(
                    system_prompt=M4_RANKER_SYSTEM_PROMPT.format(
                        top_k=self.top_k,
                        weights_formula=weights_summary(self.weights),
                        rubric_block=hypothesis_rubric_block(),
                    ),
                    user_prompt=M4_RANKER_USER_TEMPLATE.format(
                        graph_context=rank_context.rendered,
                        hypotheses_json=json.dumps(
                            [h.model_dump(mode="json") for h in candidates],
                            ensure_ascii=False,
                            indent=2,
                        ),
                        top_k=self.top_k,
                    ),
                    output_schema=ranker_schema,
                    max_tokens=4096,
                    temperature=getattr(self.llm_config, "temperature", 0.1),
                    disable_thinking=True,
                ),
                details={"candidates": len(candidates), "top_k": self.top_k},
            )
        except Exception:
            if not generation_shortfall:
                raise
            logger.warning(
                "M4 Ranker failed after the one retry still left too few "
                "hypotheses; continuing with deterministic ranking."
            )
            ranked = []
        try:
            top = self._attach_rankings(candidates, ranked, graph_context)
        except Exception:
            if not generation_shortfall:
                raise
            logger.warning(
                "M4 Ranker returned an incomplete result after the one retry "
                "still left too few hypotheses; using deterministic ranking."
            )
            top = self._rank_top(candidates)
        if not top:
            logger.warning("M4 ranker returned incomplete scores; ranking generated candidates instead")
            top = self._rank_top(candidates)

        return {
            "candidate_hypotheses": candidates,
            "top_hypotheses": top[: self.top_k],
        }

    async def _run_quality_gates(
        self,
        state: PipelineState,
        candidates: List[HypothesisCard],
        *,
        generation_shortfall: bool,
        question: str = "",
        graph_context: Optional[Any] = None,
        feedback_context: str = "",
    ) -> List[HypothesisCard]:
        """Run the multi-agent quality gates (Critic → Falsifiability).

        ``question`` / ``graph_context`` / ``feedback_context`` are unused
        here; they are reserved for strict subclasses that regenerate
        candidates with gate feedback when every candidate is rejected.
        """
        # ── Step 2: Critic ─────────────────────────────────────────────
        try:
            candidates = await self._run_critic(state, candidates)
        except Exception:
            if not generation_shortfall:
                raise
            logger.warning(
                "M4 Critic failed after the one retry still left too few "
                "hypotheses; continuing with the generated candidates."
            )

        # ── Step 3: Falsifiability Checker ─────────────────────────────
        try:
            candidates = await self._run_falsifiability(state, candidates)
        except Exception:
            if not generation_shortfall:
                raise
            logger.warning(
                "M4 Falsifiability Checker failed after the one retry still "
                "left too few hypotheses; continuing with the current candidates."
            )
        return candidates

    # ------------------------------------------------------------------
    # Multi-agent sub-steps (Critic, Falsifiability)
    # ------------------------------------------------------------------

    async def _run_critic(
        self, state: PipelineState, candidates: List[HypothesisCard]
    ) -> List[HypothesisCard]:
        """Filter candidates through the Critic agent.

        Drops hypotheses with logical gaps or external inconsistencies.  The
        critic cannot rewrite candidate content; revisions belong to the next
        Generator pass.
        Falls back to the original list if every candidate is rejected.
        """
        critic_schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "hypothesis_id": {"type": "string"},
                    "pass": {"type": "boolean"},
                    "critique": {"type": "string"},
                    "issues": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["hypothesis_id", "pass", "critique", "issues"],
            },
        }
        graph_context = build_graph_context(state)
        context_pack = ContextPlanner().plan(
            graph_context,
            ContextRequest(
                purpose="m4_critic",
                focus_evidence_ids=tuple(
                    dict.fromkeys([
                        *self._supplement_focus_evidence_ids(state),
                        *(
                            evidence_id
                            for candidate in candidates
                            for evidence_id in candidate.supporting_evidence
                        ),
                    ])
                ),
            ),
        )
        emit_context_built("m4", "hypothesis_critic", context_pack)
        try:
            result = await self._tool_call(
                "hypothesis_critic",
                self.client.structured_chat(
                    system_prompt=M4_CRITIC_SYSTEM_PROMPT,
                    user_prompt=M4_CRITIC_USER_TEMPLATE.format(
                        graph_context=context_pack.rendered,
                        established_facts="(included in the audited context pack above)",
                        hypotheses_json=json.dumps(
                            [h.model_dump(mode="json") for h in candidates],
                            ensure_ascii=False,
                            indent=2,
                        ),
                    ),
                    output_schema=critic_schema,
                    max_tokens=4096,
                    temperature=getattr(self.llm_config, "temperature", 0.1),
                    disable_thinking=True,
                ),
                details={"candidates": len(candidates)},
            )
        except Exception as exc:
            raise RuntimeError(
                "M4 Critic stage failed — when multi-agent mode is active "
                "the Critic is a required quality gate and cannot be skipped."
            ) from exc

        reviews = result if isinstance(result, list) else []
        id_to_candidate = {h.hypothesis_id: h for h in candidates}
        survivors: List[HypothesisCard] = []

        for review in reviews:
            if not isinstance(review, dict):
                continue
            if not review.get("pass", False):
                logger.info("M4 critic rejected %s: %s", review.get("hypothesis_id"), review.get("critique", "")[:120])
                continue
            hid = review.get("hypothesis_id", "")
            card = id_to_candidate.get(hid)
            if card is None:
                continue
            survivors.append(card)

        if len(survivors) < 2:
            logger.warning("M4 critic left %d survivors (need ≥2); keeping all %d candidates",
                           len(survivors), len(candidates))
            return candidates
        logger.info("M4 critic: %d/%d passed", len(survivors), len(candidates))
        return survivors

    async def _run_falsifiability(
        self, state: PipelineState, candidates: List[HypothesisCard]
    ) -> List[HypothesisCard]:
        """Filter candidates through the Falsifiability Checker.

        Drops hypotheses that cannot be empirically falsified.
        Falls back to the original list if every candidate is rejected.
        """
        falsifiability_schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "hypothesis_id": {"type": "string"},
                    "is_falsifiable": {"type": "boolean"},
                    "specificity_score": {"type": "number", "minimum": 0, "maximum": 1},
                    "assessment": {"type": "string"},
                },
                "required": ["hypothesis_id", "is_falsifiable", "assessment"],
            },
        }
        context_pack = ContextPlanner().plan(
            build_graph_context(state),
            ContextRequest(
                purpose="m4_critic",
                focus_evidence_ids=tuple(
                    dict.fromkeys([
                        *self._supplement_focus_evidence_ids(state),
                        *(
                            evidence_id
                            for candidate in candidates
                            for evidence_id in candidate.supporting_evidence
                        ),
                    ])
                ),
            ),
        )
        emit_context_built("m4", "falsifiability_checker", context_pack)
        try:
            result = await self._tool_call(
                "falsifiability_checker",
                self.client.structured_chat(
                    system_prompt=M4_FALSIFIABILITY_SYSTEM_PROMPT,
                    user_prompt=M4_FALSIFIABILITY_USER_TEMPLATE.format(
                        graph_context=context_pack.rendered,
                        hypotheses_json=json.dumps(
                            [h.model_dump(mode="json") for h in candidates],
                            ensure_ascii=False,
                            indent=2,
                        ),
                    ),
                    output_schema=falsifiability_schema,
                    max_tokens=4096,
                    temperature=getattr(self.llm_config, "temperature", 0.1),
                    disable_thinking=True,
                ),
                details={"candidates": len(candidates)},
            )
        except Exception as exc:
            raise RuntimeError(
                "M4 Falsifiability Checker failed — when multi-agent mode "
                "is active the Falsifiability check is a required quality "
                "gate and cannot be skipped."
            ) from exc

        reviews = result if isinstance(result, list) else []
        id_to_candidate = {h.hypothesis_id: h for h in candidates}
        survivors: List[HypothesisCard] = []

        for review in reviews:
            if not isinstance(review, dict):
                continue
            if not review.get("is_falsifiable", False):
                logger.info("M4 falsifiability rejected %s: %s", review.get("hypothesis_id"), review.get("assessment", "")[:120])
                continue
            hid = review.get("hypothesis_id", "")
            card = id_to_candidate.get(hid)
            if card is None:
                continue
            survivors.append(card)

        if len(survivors) < 2:
            logger.warning("M4 falsifiability left %d survivors (need ≥2); keeping incoming candidates",
                           len(survivors))
            return candidates
        logger.info("M4 falsifiability: %d/%d passed", len(survivors), len(candidates))
        return survivors

    # ------------------------------------------------------------------
    # ModuleProtocol implementation
    # ------------------------------------------------------------------

    async def __call__(
        self,
        state: PipelineState,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if self.client is None:
            raise RuntimeError(
                "M4 requires an LLM client — pass llm_config / set OPENAI_API_KEY."
            )

        # Supplement re-entry is a portfolio update, not a fresh M4 run.
        # Preserve unaffected hypotheses and only invoke the generator when a
        # gap was actually contradicted.
        if self._is_supplement_reentry(state):
            preserved_candidates, preserved_top, contradicted = (
                self._reconcile_reentry_portfolio(state)
            )
            if not contradicted:
                result = {
                    "candidate_hypotheses": preserved_candidates,
                    "top_hypotheses": preserved_top,
                    "best_hypotheses": self._update_best(state, preserved_top),
                    "evidence_gap_requests": list(state.evidence_gap_requests),
                    "user_guidance": list(state.user_guidance),
                }
                emit_event(
                    "tool_completed",
                    module="m4",
                    tool="hypothesis_portfolio_reconciler",
                    status="completed",
                    message="M4 preserved the existing hypothesis portfolio after supplement search",
                    details={"preserved": len(preserved_top), "replacements": 0},
                )
                return result

            guidance = list(state.user_guidance)
            feedback_context = self._build_feedback_context(state, guidance)
            feedback_context += (
                "\n--- Targeted contradiction revision ---\n"
                "Replace only the hypotheses contradicted by the current supplement evidence. "
                "Keep unaffected hypotheses unchanged and return new IDs for replacements.\n"
                + "\n".join(
                    f"- {gap.gap_id}: {getattr(gap, 'rationale', '') or getattr(gap, 'description', '')}"
                    for gap in [*state.evidence_gap_requests, *state.evidence_gaps]
                    if getattr(gap, "scientific_resolution", "") == "contradicted"
                )
            )
            self._run_deadline = (
                time.monotonic() + self.total_time_budget_seconds
                if self.total_time_budget_seconds > 0 else None
            )
            try:
                generated_result = await self._run_llm(state, feedback_context)
            finally:
                self._run_deadline = None
            fresh_candidates = list(generated_result.get("candidate_hypotheses", []))
            fresh_top = list(generated_result.get("top_hypotheses", []))
            taken = {card.hypothesis_id for card in preserved_candidates}
            normalized_fresh = []
            renamed_by_object = {}
            for card in fresh_candidates:
                original_object_id = id(card)
                if card.hypothesis_id in taken:
                    card = card.model_copy(update={
                        "hypothesis_id": f"{card.hypothesis_id}-revision-{state.search_round + 1}",
                    })
                taken.add(card.hypothesis_id)
                normalized_fresh.append(card)
                renamed_by_object[original_object_id] = card
            candidates = self._merge_generation_candidates(
                preserved_candidates, normalized_fresh,
            )
            fresh_by_id = {card.hypothesis_id: card for card in normalized_fresh}
            top = list(preserved_top)
            top.extend(
                renamed_by_object.get(id(card), fresh_by_id.get(card.hypothesis_id, card))
                for card in fresh_top
            )
            dedup_top = []
            seen_top = set()
            for card in top:
                if card.hypothesis_id not in seen_top:
                    dedup_top.append(card)
                    seen_top.add(card.hypothesis_id)
            result = {
                "candidate_hypotheses": candidates,
                "top_hypotheses": dedup_top[: self.top_k],
                "user_guidance": guidance,
            }
            graph_context = build_graph_context(state)
            result["evidence_gap_requests"] = await self._audit_hypotheses(
                state, result["top_hypotheses"], graph_context,
            )
            result["top_hypotheses"] = [
                normalize_hypothesis_grounding(card, graph_context)
                for card in result["top_hypotheses"]
            ]
            result["candidate_hypotheses"] = self._sync_audited_top_to_candidates(
                result["candidate_hypotheses"], result["top_hypotheses"],
            )
            result["best_hypotheses"] = self._update_best(
                state, result["top_hypotheses"],
            )
            return result

        # PipelineRunner collects optional human guidance before rendering the
        # M4 phase header.  Fold that state — plus reviewer feedback — into the
        # generator here.
        guidance = list(state.user_guidance)
        feedback_context = self._build_feedback_context(state, guidance)

        self._run_deadline = (
            time.monotonic() + self.total_time_budget_seconds
            if self.total_time_budget_seconds > 0
            else None
        )
        try:
            result = await self._run_llm(state, feedback_context)
        except M4RefinementRequired as exc:
            emit_event(
                "m4_refinement_required",
                module="m4",
                tool="hypothesis_generator_gate_recovery",
                status="warning",
                message=exc.clarification.message,
                details={
                    "original_question": exc.clarification.original_question,
                    "suggested_directions": exc.clarification.suggested_directions,
                    "rejected_hypothesis_ids": exc.clarification.rejected_hypothesis_ids,
                },
            )
            return {
                "candidate_hypotheses": exc.candidates,
                "top_hypotheses": [],
                "best_hypotheses": [],
                "evidence_gap_requests": [],
                "clarification_request": exc.clarification,
                "user_guidance": guidance,
            }
        finally:
            self._run_deadline = None
        result["user_guidance"] = guidance
        graph_context = build_graph_context(state)
        self._attach_bridge_assumptions(
            state,
            result.get("candidate_hypotheses", []),
            graph_context,
        )
        self._attach_bridge_assumptions(
            state,
            result.get("top_hypotheses", []),
            graph_context,
        )
        if self.fast_mode:
            # Fast mode never routes M4 -> M2; keep the audit field clean.
            gaps = []
        else:
            gaps = await self._audit_hypotheses(
                state,
                result.get("top_hypotheses", []),
                graph_context,
            )
            result["top_hypotheses"] = [
                normalize_hypothesis_grounding(card, graph_context)
                for card in result.get("top_hypotheses", [])
            ]
        result["candidate_hypotheses"] = self._sync_audited_top_to_candidates(
            result.get("candidate_hypotheses", []),
            result.get("top_hypotheses", []),
        )
        result["best_hypotheses"] = self._update_best(
            state, result.get("top_hypotheses", []),
        )
        result["evidence_gap_requests"] = gaps
        pending = [gap for gap in gaps if gap.status == "pending"]
        if pending:
            emit_event(
                "evidence_gap_detected",
                module="m4",
                tool="evidence_gap_detector",
                status="pending",
                message=f"M4 detected {len(pending)} searchable evidence gap(s)",
                details={
                    "route": "m4->m2->m3->m4",
                    "gap_ids": [gap.gap_id for gap in pending],
                    "premise_ids": [gap.premise_id for gap in pending if gap.premise_id],
                    "sub_questions": [gap.sub_question for gap in pending],
                    "audit_claims": [gap.audit_claim for gap in pending],
                },
            )
        return result

    @classmethod
    def get_input_fields(cls) -> List[str]:
        return [
            "evidence_graph", "problem_card", "literature_results",
            "m2_knowledge_export", "grounding_report",
        ]

    @classmethod
    def get_output_fields(cls) -> List[str]:
        return ["candidate_hypotheses", "top_hypotheses"]
