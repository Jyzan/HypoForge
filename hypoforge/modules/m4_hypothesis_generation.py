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

from ..observability import emit_event
from ..context import ContextPlanner, ContextRequest, emit_context_built
from ..graph_context import GraphContext, build_graph_context
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
    EvidenceGapRequest,
    HypothesisCard,
    PipelineState,
    TaskTrace,
    TaskTraceReference,
)
from ..task_alignment import _mentions_entity, assess_task_alignment
from ..tools.qwen_client import QwenClient

logger = logging.getLogger(__name__)


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
                result = await asyncio.wait_for(operation, timeout=timeout)
            else:
                result = await operation
        except asyncio.CancelledError:
            emit_event(
                "tool_cancelled",
                module="m4",
                tool=tool,
                status="cancelled",
                message=f"M4 Agent cancelled: {tool}",
                elapsed_seconds=time.monotonic() - started_at,
                details=details,
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
                details=details,
            )
            raise
        emit_event(
            "tool_completed",
            module="m4",
            tool=tool,
            status="completed",
            message=f"M4 Agent completed: {tool}",
            elapsed_seconds=time.monotonic() - started_at,
            details=details,
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
        contract = problem_card.task_contract
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
    def _render_task_contract_block(state: PipelineState) -> str:
        """Render the binding M1 task contract for literal reproduction.

        The alignment gate matches entity names character-for-character, so
        the generator must see the exact names/aliases it has to quote.
        Returns an empty string when there is no binding contract.
        """
        card = state.problem_card
        if card is None:
            return ""
        contract = card.task_contract
        if not contract.entities and not contract.requirements:
            return ""
        lines = [
            "Binding task contract (hard alignment gate):",
            "Required task entities — quote one of the exact names below "
            "verbatim (same characters, same language) in the hypothesis "
            "statement or mechanism:",
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
                "Atomic auditable requirements — each hypothesis must select "
                "one or more that it genuinely addresses and trace only those "
                "via a literal excerpt; do not claim every ID merely because "
                "the primary entity is mentioned:"
            )
            for requirement in contract.requirements:
                lines.append(
                    f"- {requirement.requirement_id}: {requirement.relation} "
                    f"(primary entity: {requirement.primary_entity_id})"
                )
        lines.append(
            "Do not translate or paraphrase these names — the validator "
            "matches them literally (character-for-character)."
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
        contract = card.task_contract
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
                "Available atomic requirement IDs: " + ", ".join(requirements)
                + ". Preserve the current hypothesis's selected subset; do not "
                "attach unrelated IDs."
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
        contract = state.problem_card.task_contract if state.problem_card else None
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
    ) -> tuple[List[HypothesisCard], List[Dict[str, Any]]]:
        """Make one deterministic repair attempt without relaxing hard gates."""

        assert self.client is not None
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
                    num_candidates=self.num_candidates,
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
                    task_contract_block=self._render_task_contract_block(state),
                ),
                output_schema={
                    "type": "array",
                    "items": HypothesisCard.model_json_schema(),
                },
                max_tokens=8192,
                temperature=0.0,
            ),
            details={
                "candidate_count": len(candidates),
                "failure_count": len(failures),
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
        requirements = {
            item.requirement_id: item
            for item in (card.task_contract.requirements if card else [])
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
    ) -> tuple[Any, List[HypothesisCard], List[Dict[str, Any]]]:
        """Run one generator pass and apply the normal context gates.

        ``requested_count`` overrides ``self.num_candidates`` (used by gate
        recovery to keep the regeneration pass small).
        """

        assert self.client is not None
        requested = (
            self.num_candidates if requested_count is None else int(requested_count)
        )
        context_pack = ContextPlanner().plan(
            graph_context,
            ContextRequest(purpose="m4_generate"),
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
                output_schema={
                    "type": "array",
                    "items": HypothesisCard.model_json_schema(),
                },
                max_tokens=8192,
                temperature=max(getattr(self.llm_config, "temperature", 0.1), 0.4),
                disable_thinking=self.fast_mode,
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
        if len(candidates) < self.top_k:
            retry_feedback = (
                f"The previous pass produced only {len(candidates)} valid hypotheses, "
                f"but {self.num_candidates} were requested. Generate additional "
                "distinct scientific hypotheses if the evidence supports them. "
                "Do not invent unsupported claims; returning fewer is acceptable "
                "if no further defensible hypothesis exists."
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
                    )
                )
            except Exception as exc:
                if not self.fast_mode:
                    raise
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
        generation_shortfall = bool(candidates) and len(candidates) < self.top_k

        if not candidates:
            try:
                candidates, contract_failures = await self._repair_context_contract(
                    state=state,
                    context=graph_context,
                    candidates=all_generated,
                    failures=contract_failures,
                    feedback_context=feedback_context,
                )
            except Exception as exc:
                if not self.fast_mode:
                    raise
                logger.warning(
                    "M4 fast-mode contract repair failed (%s); using "
                    "deterministic fallback hypotheses.",
                    type(exc).__name__,
                )
                candidates = self._fast_fallback_hypotheses(state, question)
        if not candidates:
            if self.fast_mode:
                logger.warning(
                    "M4 fast mode found no contract-valid hypotheses; using "
                    "deterministic fallback hypotheses so the run can continue."
                )
                candidates = self._fast_fallback_hypotheses(state, question)
            elif generation_shortfall:
                logger.warning(
                    "M4 produced no task-contract-compliant hypotheses after "
                    "one retry; continuing with an empty candidate set."
                )
                return {
                    "candidate_hypotheses": [],
                    "top_hypotheses": [],
                }
            else:
                diagnostics = "; ".join(
                    str(item.get("rationale") or "contract validation failed")
                    for item in contract_failures[:3]
                )
                raise ValueError(
                    "M4 generator and one deterministic repair attempt returned no "
                    f"task-contract-compliant hypotheses: {diagnostics}"
                )

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
                    evidence_id
                    for candidate in candidates
                    for evidence_id in candidate.supporting_evidence
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
                    evidence_id
                    for candidate in candidates
                    for evidence_id in candidate.supporting_evidence
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
                    evidence_id
                    for candidate in candidates
                    for evidence_id in candidate.supporting_evidence
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
        finally:
            self._run_deadline = None
        result["user_guidance"] = guidance
        result["best_hypotheses"] = self._update_best(state, result.get("top_hypotheses", []))
        graph_context = build_graph_context(state)
        if self.fast_mode:
            # Fast mode never routes M4 -> M2; keep the audit field clean.
            gaps = []
        else:
            gaps = self._update_evidence_gaps(
                state,
                result.get("top_hypotheses", []),
                graph_context,
            )
        result["evidence_gap_requests"] = gaps
        pending = [gap for gap in gaps if gap.status == "pending"]
        if pending:
            emit_event(
                "routing_decision",
                module="m4",
                tool="evidence_gap_detector",
                status="pending",
                message=f"M4 detected {len(pending)} searchable evidence gap(s)",
                details={
                    "route": "m4->m2->m3->m4",
                    "gap_ids": [gap.gap_id for gap in pending],
                    "sub_questions": [gap.sub_question for gap in pending],
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
