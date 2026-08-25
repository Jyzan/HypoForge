"""Whole-question task view used by synthesis modules M4-M6.

M1's atomic requirements are retrieval controls for M2/M3.  This module
derives a separate, non-persisted contract so downstream synthesis cannot
accidentally treat those search questions as independent answer targets.
"""

from __future__ import annotations

from .state import PipelineState, ProblemCard, TaskContract, TaskRequirement


SYNTHESIS_REQUIREMENT_ID = "Q0"


def build_synthesis_contract(card: ProblemCard | None) -> TaskContract:
    """Return a deterministic whole-question view without retrieval questions."""

    if card is None:
        return TaskContract(source="derived")

    source_entities = list(card.task_contract.entities)
    entities = [
        entity.model_copy(deep=True, update={"required": False})
        for entity in source_entities
    ]
    primary_entity_id = next(
        (
            entity.entity_id
            for entity in entities
            if entity.role == "primary_object"
        ),
        entities[0].entity_id if entities else "",
    )
    related_entity_ids = [
        entity.entity_id
        for entity in source_entities
        if entity.required and entity.entity_id != primary_entity_id
    ]
    return TaskContract(
        source=card.task_contract.source,
        entities=entities,
        requirements=[
            TaskRequirement(
                requirement_id=SYNTHESIS_REQUIREMENT_ID,
                sub_question=card.original_question,
                primary_entity_id=primary_entity_id,
                related_entity_ids=related_entity_ids,
                relation="answer the original question as a whole",
                required=True,
            )
        ],
    )


def synthesis_contract_for_state(state: PipelineState) -> TaskContract:
    """Derive the synthesis contract for one pipeline state."""

    return build_synthesis_contract(state.problem_card)


def synthesis_problem_payload(card: ProblemCard | None) -> dict[str, object]:
    """Return the M4-M6 problem view without M1 retrieval sub-questions."""

    if card is None:
        return {}
    return {
        "original_question": card.original_question,
        "domain": list(card.domain),
        "key_entities": list(card.key_entities),
        "task_contract": build_synthesis_contract(card).model_dump(mode="json"),
    }
