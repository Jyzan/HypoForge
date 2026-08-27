"""Entity must/unmust grouped round strategy for M2 agentic search.

Replaces coverage-driven iteration for the primary (agentic) search path:

1. **Entity classification** — one LLM call splits the sub-question's
   entities (M1 bound + M2-local supplements) into ``must_entities``
   (important, should ride along in queries) and ``unmust_entities``
   (optional extras).  The prompt asks for at most 3 must entities; this
   is a soft constraint — an oversized must list never raises.
2. **Round planning** — a deterministic rule branches on entity counts:

   - *Case A* (total ≤ 4): round 1 searches all entities; when results are
     poor, round 2 keeps only the must entities.  ≤ 2 rounds.
   - *Case B1* (total ≥ 5, must ≤ 2): every round carries all must
     entities plus 1-2 unmust entities; the LLM assigns which unmust
     entities ride in which round.  ≤ 2 rounds.
   - *Case B2* (total ≥ 5, must ≥ 3): must entities split into two groups
     (first 2 vs. the rest — must never exceeds 4 in practice) and unmust
     entities split into two groups; round *i* = must group *i* + unmust
     group *i*.  ≤ 2 rounds.

3. **Zero-result rescue** — handled by the agent: after any round, when at
   least ``MIN_ZERO_SOURCES_FOR_RESCUE`` sources returned zero hits or the
   round produced ≤ ``DEFAULT_POOR_RESULT_THRESHOLD`` raw records, one
   extra must-only round is appended (at most once per sub-question), for a
   hard cap of 3 rounds (matching ``SearchBudget.max_rounds``).

Every LLM failure degrades to a deterministic fallback plus a warning
event; the search itself always proceeds.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

from pydantic import BaseModel, Field

from ....observability import emit_event

logger = logging.getLogger(__name__)

# --- tunables (user-specified spec constants) -------------------------------

#: Soft prompt constraint — keep must entities few.
MUST_ENTITY_SOFT_LIMIT = 3
#: "Results are poor" when a round returns at most this many raw records.
DEFAULT_POOR_RESULT_THRESHOLD = 2
#: "Results are poor" when at least this many sources returned zero hits.
MIN_ZERO_SOURCES_FOR_RESCUE = 2
#: Experience-based per-round entity cap used by the grouping rules.
MAX_ENTITIES_PER_ROUND = 4

_CLASSIFY_TOOL = "m2_entity_classification"
_PLAN_TOOL = "m2_round_plan"


# --- structured output schemas ----------------------------------------------


class EntityClassificationSchema(BaseModel):
    """LLM output of the must/unmust entity split."""

    must_entities: List[str] = Field(
        default_factory=list,
        description=(
            "Entities that are indispensable for answering the sub-question; "
            "queries should always include them. Keep this list short — "
            "ideally 3 or fewer."
        ),
    )
    unmapped_entities: List[str] = Field(
        default_factory=list,
        description=(
            "Supporting entities that improve recall but may be dropped "
            "when a query is too narrow. Everything not must belongs here."
        ),
    )


class _RoundGroupingSchema(BaseModel):
    """LLM output assigning unmust entities to search rounds."""

    rounds: List[List[str]] = Field(
        default_factory=list,
        description=(
            "Up to 2 groups; each group lists the unmust entity names that "
            "accompany the must entities in one search round (1-2 each)."
        ),
    )
    reasoning: str = ""


_CLASSIFY_SYSTEM = (
    "You are a scientific literature search assistant. For one atomic "
    "research sub-question you receive the candidate search entities. "
    "Split them into must_entities and unmapped_entities.\n\n"
    "Rules:\n"
    "1. must_entities are the entities the search should always try to "
    "include — the core research object and its central relation partner. "
    "Keep this list SHORT: ideally at most 3 entities. If you cannot keep "
    "it that small, still return your best split; do not invent new "
    "entities.\n"
    "2. unmapped_entities are supporting concepts (methods, co-factors, "
    "context terms) that can be dropped from a query when needed.\n"
    "3. Every supplied entity must appear in exactly one list, using the "
    "exact names given (aliases allowed only alongside the given name).\n"
    "4. Prefer the more central, domain-specific terms as must; generic "
    "modifiers belong in unmapped."
)

_CLASSIFY_USER = (
    "## Sub-question\n{sub_question}\n\n"
    "## Research domains\n{domains}\n\n"
    "## Candidate entities\n{entities}\n\n"
    "## Task\n"
    "Split the candidate entities into must_entities (keep this list as "
    "short as possible, ideally at most 3) and unmapped_entities. Return "
    "the JSON object only."
)

_GROUPING_SYSTEM = (
    "You are a scientific literature search planner. Must entities are "
    "already fixed for every search round. Assign the supplied optional "
    "(unmust) entities to at most 2 rounds, 1-2 entities per round, so "
    "that each round stays focused (at most 4 concepts overall). Use the "
    "exact entity names supplied; never invent new ones. Optional entities "
    "that do not fit can be left out."
)

_GROUPING_USER = (
    "## Sub-question\n{sub_question}\n\n"
    "## Must entities (present in every round)\n{must}\n\n"
    "## Optional entities to distribute\n{unmapped}\n\n"
    "## Task\n"
    "Return up to 2 rounds; each round is a list of 1-2 optional entity "
    "names to search together with the must entities."
)


# --- data classes -------------------------------------------------------------


@dataclass(frozen=True)
class EntityClassification:
    """Result of the must/unmust split for one sub-question."""

    must_entities: List[str] = field(default_factory=list)
    unmapped_entities: List[str] = field(default_factory=list)
    degraded: bool = False

    @property
    def total(self) -> int:
        return len(self.must_entities) + len(self.unmapped_entities)


@dataclass(frozen=True)
class RoundSpec:
    """One planned search round: the focus entity subset for the planner."""

    focus_entities: List[str]
    label: str


@dataclass(frozen=True)
class RoundPlan:
    """Deterministic round organisation for one sub-question."""

    case: str
    rounds: List[RoundSpec]
    rescue_focus: List[str]


# --- helpers ------------------------------------------------------------------


def normalise_entity_name(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def _clean_names(values: Sequence[Any], known: set[str]) -> List[str]:
    """Map raw LLM strings back onto the supplied canonical entity names."""

    by_norm: Dict[str, str] = {}
    for name in known:
        by_norm.setdefault(normalise_entity_name(name), name)
    output: List[str] = []
    seen: set[str] = set()
    for value in values:
        norm = normalise_entity_name(value)
        canonical = by_norm.get(norm)
        if canonical is None or norm in seen:
            continue
        seen.add(norm)
        output.append(canonical)
    return output


def deterministic_classification(entities: Sequence[str]) -> EntityClassification:
    """Fail-safe split: the first two entities are must, the rest unmapped."""

    ordered = [str(entity).strip() for entity in entities if str(entity).strip()]
    must = ordered[:2]
    return EntityClassification(
        must_entities=must,
        unmapped_entities=ordered[len(must):],
        degraded=True,
    )


async def classify_entities(
    client: Any,
    sub_question: str,
    entities: Sequence[str],
    domains: Sequence[str] = (),
) -> EntityClassification:
    """Split *entities* into must/unmapped via one LLM call (fail-safe).

    Any LLM or validation failure degrades to
    :func:`deterministic_classification` plus a warning event; the search
    always proceeds.  An oversized must list is tolerated (soft limit).
    """

    ordered: List[str] = []
    seen: set[str] = set()
    for entity in entities:
        name = " ".join(str(entity or "").split())
        norm = normalise_entity_name(name)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        ordered.append(name)

    if not ordered:
        return EntityClassification()
    if client is None:
        classification = deterministic_classification(ordered)
        emit_event(
            "tool_result",
            module="m2",
            tool=_CLASSIFY_TOOL,
            status="warning",
            message=(
                "No entity-classification LLM configured; using the first "
                "2 entities as must entities (deterministic fallback)"
            ),
            details={
                "sub_question": sub_question,
                "must_entities": list(classification.must_entities),
                "unmapped_entities": list(classification.unmapped_entities),
            },
        )
        return classification

    emit_event(
        "tool_started",
        module="m2",
        tool=_CLASSIFY_TOOL,
        status="running",
        message="M2 entity must/unmust classification started",
        details={"sub_question": sub_question, "entities": list(ordered)},
    )
    try:
        payload = await client.structured_chat(
            system_prompt=_CLASSIFY_SYSTEM,
            user_prompt=_CLASSIFY_USER.format(
                sub_question=sub_question,
                domains=", ".join(domains) if domains else "unknown",
                entities="\n".join(f"- {name}" for name in ordered),
            ),
            output_schema=EntityClassificationSchema.model_json_schema(),
            max_tokens=1024,
            temperature=0.0,
            disable_thinking=True,
        )
        parsed = EntityClassificationSchema.model_validate(payload)
    except Exception as exc:
        classification = deterministic_classification(ordered)
        logger.warning(
            "M2 entity classification failed for %r: %s: %s",
            sub_question[:60],
            type(exc).__name__,
            exc,
        )
        emit_event(
            "tool_result",
            module="m2",
            tool=_CLASSIFY_TOOL,
            status="warning",
            message=(
                "Entity-classification LLM failed; degraded to the "
                f"deterministic fallback (first 2 entities treated as "
                f"must): {type(exc).__name__}"
            ),
            details={
                "sub_question": sub_question,
                "must_entities": list(classification.must_entities),
                "unmapped_entities": list(classification.unmapped_entities),
            },
        )
        return classification

    known = set(ordered)
    must = _clean_names(parsed.must_entities, known)
    unmapped = _clean_names(parsed.unmapped_entities, known)
    must_norms = {normalise_entity_name(name) for name in must}
    # Every supplied entity must land somewhere; anything the LLM dropped or
    # double-listed falls into the unmapped bucket deterministically.
    for name in ordered:
        norm = normalise_entity_name(name)
        if norm not in must_norms and name not in unmapped:
            unmapped.append(name)
    classification = EntityClassification(
        must_entities=must, unmapped_entities=unmapped, degraded=False
    )
    emit_event(
        "tool_completed",
        module="m2",
        tool=_CLASSIFY_TOOL,
        status="completed",
        message=(
            f"Entity classification completed: {len(must)} must / "
            f"{len(unmapped)} unmapped"
        ),
        details={
            "sub_question": sub_question,
            "must_entities": list(must),
            "unmapped_entities": list(unmapped),
            "soft_limit": MUST_ENTITY_SOFT_LIMIT,
            "over_soft_limit": len(must) > MUST_ENTITY_SOFT_LIMIT,
        },
    )
    return classification


def _deterministic_unmapped_chunks(
    unmapped: Sequence[str], chunk_size: int = 2, max_groups: int = 2
) -> List[List[str]]:
    groups: List[List[str]] = []
    for start in range(0, len(unmapped), chunk_size):
        groups.append(list(unmapped[start:start + chunk_size]))
        if len(groups) >= max_groups:
            break
    return groups


async def _assign_unmapped_with_llm(
    client: Any,
    sub_question: str,
    must: Sequence[str],
    unmapped: Sequence[str],
    *,
    max_groups: int,
    max_group_size: int,
) -> List[List[str]]:
    """Ask the LLM which unmapped entities ride in which round; fail-safe
    to deterministic chunking."""

    fallback = _deterministic_unmapped_chunks(
        unmapped, chunk_size=max_group_size, max_groups=max_groups
    )
    if client is None or len(unmapped) <= max_group_size:
        return fallback
    try:
        payload = await client.structured_chat(
            system_prompt=_GROUPING_SYSTEM,
            user_prompt=_GROUPING_USER.format(
                sub_question=sub_question,
                must=", ".join(must) if must else "(none)",
                unmapped="\n".join(f"- {name}" for name in unmapped),
            ),
            output_schema=_RoundGroupingSchema.model_json_schema(),
            max_tokens=1024,
            temperature=0.0,
            disable_thinking=True,
        )
        parsed = _RoundGroupingSchema.model_validate(payload)
    except Exception as exc:
        logger.warning(
            "M2 round grouping failed for %r: %s: %s",
            sub_question[:60],
            type(exc).__name__,
            exc,
        )
        emit_event(
            "tool_result",
            module="m2",
            tool=_PLAN_TOOL,
            status="warning",
            message=(
                "Unmust entity grouping LLM failed; degraded to "
                f"deterministic even split: {type(exc).__name__}"
            ),
            details={"sub_question": sub_question},
        )
        return fallback

    known = set(unmapped)
    groups: List[List[str]] = []
    used: set[str] = set()
    for raw_group in parsed.rounds:
        if not isinstance(raw_group, (list, tuple)):
            continue
        cleaned = [
            name
            for name in _clean_names(raw_group, known)
            if normalise_entity_name(name) not in used
        ]
        used.update(normalise_entity_name(name) for name in cleaned)
        if cleaned:
            groups.append(cleaned[:max_group_size])
        if len(groups) >= max_groups:
            break
    if not groups:
        return fallback
    return groups


async def plan_entity_rounds(
    client: Any,
    sub_question: str,
    classification: EntityClassification,
) -> RoundPlan:
    """Produce the deterministic round organisation for one sub-question.

    Branches on ``total = len(must) + len(unmust)``:

    - Case A  (total ≤ 4): one round with all entities; the poor-result
      follow-up keeps must entities only (appended lazily by the agent).
    - Case B1 (total ≥ 5, must ≤ 2): every round = all must + 1-2 unmapped
      (LLM-assigned), ≤ 2 rounds.
    - Case B2 (total ≥ 5, must ≥ 3): must split into [first 2 | rest];
      unmapped split into two groups; round i pairs group i. ≤ 2 rounds.
    """

    must = list(classification.must_entities)
    unmapped = list(classification.unmapped_entities)
    total = len(must) + len(unmapped)
    rescue_focus = list(must)

    if total == 0:
        return RoundPlan(case="none", rounds=[], rescue_focus=[])

    if total <= MAX_ENTITIES_PER_ROUND:
        plan = RoundPlan(
            case="A",
            rounds=[RoundSpec(focus_entities=[*must, *unmapped], label="all_entities")],
            rescue_focus=rescue_focus,
        )
    elif len(must) <= 2:
        groups = await _assign_unmapped_with_llm(
            client,
            sub_question,
            must,
            unmapped,
            max_groups=2,
            max_group_size=2,
        )
        rounds = [
            RoundSpec(focus_entities=[*must, *group], label=f"must_plus_unmust_{index + 1}")
            for index, group in enumerate(groups)
        ]
        if not rounds:
            rounds = [RoundSpec(focus_entities=list(must), label="must_only")]
        plan = RoundPlan(case="B1", rounds=rounds, rescue_focus=rescue_focus)
    else:
        must_groups = [must[:2], must[2:]]
        # Even split: two groups as equal in size as possible (ceil(n/2)).
        split_size = max(1, math.ceil(len(unmapped) / 2)) if unmapped else 2
        unmapped_groups = await _assign_unmapped_with_llm(
            client,
            sub_question,
            must,
            unmapped,
            max_groups=2,
            max_group_size=split_size,
        )
        if len(unmapped_groups) < 2:
            unmapped_groups = _deterministic_unmapped_chunks(
                unmapped,
                chunk_size=split_size,
                max_groups=2,
            )
        # Keep the second round useful when pairing leaves it empty.
        if (
            len(unmapped_groups) == 2
            and not unmapped_groups[1]
            and len(unmapped_groups[0]) >= 2
        ):
            unmapped_groups[1] = [unmapped_groups[0].pop()]
        rounds = []
        for index, must_group in enumerate(must_groups):
            if not must_group:
                continue
            group = unmapped_groups[index] if index < len(unmapped_groups) else []
            rounds.append(
                RoundSpec(
                    focus_entities=[*must_group, *group],
                    label=f"must_group_{index + 1}",
                )
            )
        plan = RoundPlan(case="B2", rounds=rounds, rescue_focus=rescue_focus)

    emit_event(
        "tool_result",
        module="m2",
        tool=_PLAN_TOOL,
        status="completed",
        message=(
            f"Round planning completed: Case {plan.case}, "
            f"{len(plan.rounds)} base round(s)"
        ),
        details={
            "sub_question": sub_question,
            "case": plan.case,
            "must_entities": must,
            "unmapped_entities": unmapped,
            "rounds": [
                {"label": spec.label, "focus_entities": spec.focus_entities}
                for spec in plan.rounds
            ],
            "rescue_focus": rescue_focus,
        },
    )
    return plan


def poor_round_reason(
    round_source_counts: Dict[str, int],
    total_papers: int,
    threshold: int = DEFAULT_POOR_RESULT_THRESHOLD,
    min_zero_sources: int = MIN_ZERO_SOURCES_FOR_RESCUE,
) -> str:
    """Return a non-empty reason when the round's results count as "poor".

    Poor means: at least *min_zero_sources* successfully queried sources
    returned zero hits, or the round produced ≤ *threshold* raw records.
    """

    zero_sources = sorted(
        source for source, count in round_source_counts.items() if count == 0
    )
    if len(zero_sources) >= min_zero_sources:
        return f"{len(zero_sources)} sources returned zero results"
    if total_papers <= threshold:
        return f"total results {total_papers} <= threshold {threshold}"
    return ""
