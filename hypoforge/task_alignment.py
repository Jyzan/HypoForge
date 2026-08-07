"""Domain-neutral, auditable task-contract checks.

The guard never contains domain vocabulary.  M1 supplies the current task's
entities, aliases, roles, and atomic requirements; downstream modules supply
literal excerpts that show where those contract items were addressed.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable

from .state import PipelineState, TaskContract, TaskEntity, TaskTrace


def _normalise(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return re.sub(r"\s+", " ", value).strip()


def _contains(text: str, phrase: str) -> bool:
    """Match a task-provided phrase without assuming a language or domain."""

    text = _normalise(text)
    phrase = _normalise(phrase)
    if not phrase:
        return False
    if re.search(r"[^\x00-\x7f]", phrase):
        return phrase in text
    tokens = re.findall(r"[a-z0-9]+", phrase)
    return bool(tokens) and all(
        re.search(rf"\b{re.escape(token)}\b", text) for token in tokens
    )


def _entity_terms(entity: TaskEntity) -> list[str]:
    return list(dict.fromkeys(
        term for term in [entity.name, *entity.aliases] if str(term).strip()
    ))


def _mentions_entity(text: str, entity: TaskEntity) -> bool:
    return any(_contains(text, term) for term in _entity_terms(entity))


def _validated_trace_ids(
    candidate_text: str,
    trace: TaskTrace,
    contract: TaskContract,
) -> tuple[set[str], set[str], list[str]]:
    """Accept only known IDs backed by a literal, semantically relevant excerpt."""

    entity_by_id = {entity.entity_id: entity for entity in contract.entities}
    requirement_by_id = {
        requirement.requirement_id: requirement
        for requirement in contract.requirements
    }
    entity_ids: set[str] = set()
    requirement_ids: set[str] = set()
    invalid: list[str] = []

    for reference in trace.entity_mentions:
        entity = entity_by_id.get(reference.contract_id)
        excerpt = reference.output_excerpt
        if (
            entity is None
            or not excerpt
            or not _contains(candidate_text, excerpt)
            or not _mentions_entity(excerpt, entity)
        ):
            invalid.append(f"invalid entity trace {reference.contract_id!r}")
            continue
        entity_ids.add(reference.contract_id)

    for reference in trace.requirement_mentions:
        requirement = requirement_by_id.get(reference.contract_id)
        excerpt = reference.output_excerpt
        if requirement is None or not excerpt or not _contains(candidate_text, excerpt):
            invalid.append(f"invalid requirement trace {reference.contract_id!r}")
            continue
        primary = entity_by_id.get(requirement.primary_entity_id)
        if primary is not None and not _mentions_entity(excerpt, primary):
            invalid.append(
                f"requirement trace {reference.contract_id!r} omits its primary entity"
            )
            continue
        requirement_ids.add(reference.contract_id)

    return entity_ids, requirement_ids, invalid


@dataclass(frozen=True)
class AlignmentAssessment:
    passed: bool
    score: float
    matched_anchors: tuple[str, ...] = ()
    missing_anchors: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()
    matched_entity_ids: tuple[str, ...] = ()
    missing_requirement_ids: tuple[str, ...] = ()
    invalid_trace_references: tuple[str, ...] = ()
    rationale: str = ""


def assess_task_alignment(
    state: PipelineState,
    candidate_text: str,
    *,
    subject_text: str | None = None,
    trace: TaskTrace | None = None,
) -> AlignmentAssessment:
    """Validate an output against the task-local M1 contract.

    ``subject_text`` is the narrow field that names the actual study object
    (for example a hypothesis statement or ``ResearchPlan.study_subjects``).
    Requiring the primary object there prevents an incidental mention elsewhere
    from satisfying the hard gate.
    """

    card = state.problem_card
    if card is None:
        return AlignmentAssessment(
            passed=True,
            score=0.5,
            rationale="No M1 ProblemCard was available; semantic review is required.",
        )

    contract = card.task_contract
    if not contract.entities and not contract.requirements:
        return AlignmentAssessment(
            passed=True,
            score=0.5,
            rationale="No task contract anchors were available; semantic review is required.",
        )

    trace = trace or TaskTrace()
    traced_entities, traced_requirements, invalid_trace = _validated_trace_ids(
        candidate_text, trace, contract
    )
    subject = subject_text if subject_text is not None else candidate_text
    matched_entity_ids: set[str] = set(traced_entities)
    for entity in contract.entities:
        scope = subject if entity.role == "primary_object" else candidate_text
        if _mentions_entity(scope, entity):
            matched_entity_ids.add(entity.entity_id)

    required_entities = [entity for entity in contract.entities if entity.required]
    missing_entities = [
        entity for entity in required_entities
        if entity.entity_id not in matched_entity_ids
    ]
    required_requirements = [
        requirement for requirement in contract.requirements if requirement.required
    ]
    # Old snapshots did not produce task traces.  Their derived contracts retain
    # lexical object protection while M6 performs the semantic requirement check.
    missing_requirements = (
        [
            requirement.requirement_id
            for requirement in required_requirements
            if requirement.requirement_id not in traced_requirements
        ]
        if contract.source == "m1"
        else []
    )

    entity_coverage = (
        (len(required_entities) - len(missing_entities)) / len(required_entities)
        if required_entities else 1.0
    )
    requirement_coverage = (
        (len(required_requirements) - len(missing_requirements))
        / len(required_requirements)
        if required_requirements else 1.0
    )
    score = round(0.7 * entity_coverage + 0.3 * requirement_coverage, 4)
    passed = not missing_entities and not missing_requirements and not invalid_trace
    matched_names = [
        entity.name for entity in contract.entities
        if entity.entity_id in matched_entity_ids
    ]
    problems: list[str] = []
    if missing_entities:
        problems.append(
            "missing required task entities in their required output scope: "
            + ", ".join(entity.name for entity in missing_entities)
        )
    if missing_requirements:
        problems.append(
            "missing auditable requirement traces: " + ", ".join(missing_requirements)
        )
    problems.extend(invalid_trace)
    return AlignmentAssessment(
        passed=passed,
        score=score,
        matched_anchors=tuple(matched_names),
        missing_anchors=tuple(entity.name for entity in missing_entities),
        conflicts=tuple(invalid_trace),
        matched_entity_ids=tuple(sorted(matched_entity_ids)),
        missing_requirement_ids=tuple(missing_requirements),
        invalid_trace_references=tuple(invalid_trace),
        rationale=(
            "Task contract is covered by scoped text and auditable traces."
            if passed else "; ".join(problems)
        ),
    )


def search_entities_for_sub_question(
    state: PipelineState,
    sub_question: str,
) -> list[str]:
    """Return one search-ready alias per entity relevant to this sub-question."""

    card = state.problem_card
    if card is None:
        return []
    contract = card.task_contract
    entity_by_id = {entity.entity_id: entity for entity in contract.entities}
    relevant_ids: list[str] = []
    for requirement in contract.requirements:
        if _normalise(requirement.sub_question) != _normalise(sub_question):
            continue
        relevant_ids.extend([
            requirement.primary_entity_id,
            *requirement.related_entity_ids,
        ])

    if not relevant_ids:
        relevant_ids.extend(
            entity.entity_id
            for entity in contract.entities
            if _mentions_entity(sub_question, entity)
        )
    if not relevant_ids:
        relevant_ids.extend(
            entity.entity_id
            for entity in contract.entities
            if entity.role == "primary_object" and entity.required
        )

    output: list[str] = []
    for entity_id in dict.fromkeys(item for item in relevant_ids if item):
        entity = entity_by_id.get(entity_id)
        if entity is None:
            continue
        terms = _entity_terms(entity)
        # Literature APIs generally index English terminology. Prefer a supplied
        # Latin-script alias, but preserve the canonical name when no such alias exists.
        selected = next(
            (term for term in terms if re.search(r"[a-zA-Z]", term)),
            entity.name,
        )
        if selected and selected not in output:
            output.append(selected)
    return output


def contract_text(state: PipelineState) -> str:
    """Serialize the binding task contract for prompts."""

    if state.problem_card is None:
        return "{}"
    return state.problem_card.task_contract.model_dump_json(indent=2)
