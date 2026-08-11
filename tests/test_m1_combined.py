"""Behavior tests for the combined M1 decomposition and task contract."""

from __future__ import annotations

from typing import Any

import pytest

from hypoforge.modules.m1_problem_understanding import M1ProblemUnderstanding
from hypoforge.state import ProblemCard, TaskEntity


class EntityClient:
    def __init__(self, payloads: list[dict[str, Any]]):
        self.payloads = list(payloads)
        self.calls: list[dict[str, Any]] = []

    async def structured_chat(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if not self.payloads:
            raise AssertionError("unexpected extra entity LLM call")
        return self.payloads.pop(0)


def entity_module(payloads: list[dict[str, Any]]) -> M1ProblemUnderstanding:
    module = M1ProblemUnderstanding(mode="llm", entity_repair_attempts=1)
    module.client = EntityClient(payloads)
    return module


def candidate(
    name: str,
    *,
    source_mention: str | None = None,
    role: str = "other",
    required: bool = False,
    aliases: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "source_mention": source_mention or name,
        "aliases": aliases or [],
        "role": role,
        "required": required,
        "extraction_reason": "用户原文明确出现",
    }


def audit_item(
    name: str,
    accepted: bool = True,
    *,
    source_mention: str | None = None,
) -> dict[str, Any]:
    return {
        "candidate_name": name,
        "accepted": accepted,
        "source_mention": source_mention or name,
        "reason": "可在用户原文定位" if accepted else "原文没有该实体",
    }


def test_problem_card_ignores_legacy_question_type_and_derives_key_entities() -> None:
    """A stale UI list must not override the audited task-contract entities."""

    card = ProblemCard.model_validate({
        "original_question": "如何使用离线强化学习控制机械臂？",
        "question_type": "method_development",
        "key_entities": ["错误的旧列表"],
        "task_contract": {
            "source": "m1",
            "entities": [
                {
                    "entity_id": "E1",
                    "name": "机械臂",
                    "source_mention": "机械臂",
                    "role": "primary_object",
                    "required": True,
                },
                {
                    "entity_id": "E2",
                    "name": "离线强化学习",
                    "source_mention": "离线强化学习",
                    "role": "method",
                    "required": True,
                },
            ],
            "requirements": [],
        },
    })

    assert not hasattr(card, "question_type")
    assert card.key_entities == ["机械臂", "离线强化学习"]


def test_legacy_problem_card_without_contract_still_derives_compatibility_contract() -> None:
    """Historical snapshots remain readable after the new contract is introduced."""

    card = ProblemCard.model_validate({
        "original_question": "Hsp70 如何识别底物？",
        "sub_questions": ["Hsp70 如何识别底物？"],
        "key_entities": ["Hsp70", "底物"],
        "question_type": "mechanism_explanation",
    })

    assert card.task_contract.source == "derived"
    assert [entity.name for entity in card.task_contract.entities] == ["Hsp70", "底物"]
    assert card.task_contract.requirements[0].sub_question == "Hsp70 如何识别底物？"


@pytest.mark.asyncio
async def test_entity_gate_rejects_llm_accepted_term_absent_from_user_text() -> None:
    """An agreeable audit model cannot legitimize a sub-question invention."""

    module = entity_module([
        {"entities": [
            candidate(
                "robot arm", source_mention="机械臂",
                role="primary_object", required=True,
            ),
            candidate(
                "domain randomization", source_mention="域随机化", role="method",
            ),
        ]},
        {
            "items": [
                audit_item("robot arm", source_mention="机械臂"),
                audit_item("domain randomization", source_mention="域随机化"),
            ],
            "missing_explicit_entities": [],
            "complete": True,
        },
    ])

    entities = await module._extract_and_audit_entities(
        "如何改进机械臂从仿真到现实的迁移？"
    )

    assert [entity.name for entity in entities] == ["robot arm"]
    assert entities[0].source_mention == "机械臂"
    assert entities[0].entity_id == "E1"


@pytest.mark.asyncio
async def test_entity_audit_missing_term_triggers_one_repair_and_reaudit() -> None:
    module = entity_module([
        {"entities": [candidate(
            "robot arm", source_mention="机械臂",
            role="primary_object", required=True,
        )]},
        {
            "items": [audit_item("robot arm", source_mention="机械臂")],
            "missing_explicit_entities": ["离线强化学习"],
            "complete": False,
        },
        {"entities": [
            candidate(
                "robot arm", source_mention="机械臂",
                role="primary_object", required=True,
            ),
            candidate(
                "offline reinforcement learning",
                source_mention="离线强化学习",
                role="method",
                required=True,
                aliases=["offline RL"],
            ),
        ]},
        {
            "items": [
                audit_item("robot arm", source_mention="机械臂"),
                audit_item(
                    "offline reinforcement learning",
                    source_mention="离线强化学习",
                ),
            ],
            "missing_explicit_entities": [],
            "complete": True,
        },
    ])

    entities = await module._extract_and_audit_entities(
        "如何使用离线强化学习控制机械臂？"
    )

    assert [entity.name for entity in entities] == [
        "robot arm", "offline reinforcement learning",
    ]
    assert entities[1].aliases == ["offline RL"]
    assert len(module.client.calls) == 4


@pytest.mark.asyncio
async def test_entity_prompts_never_receive_generated_subquestions() -> None:
    module = entity_module([
        {"entities": [candidate(
            "robot arm", source_mention="机械臂",
            role="primary_object", required=True,
        )]},
        {
            "items": [audit_item("robot arm", source_mention="机械臂")],
            "missing_explicit_entities": [],
            "complete": True,
        },
    ])

    await module._extract_and_audit_entities("如何改进机械臂迁移？")

    forbidden_generated_question = "域随机化如何提高机械臂迁移鲁棒性？"
    assert all(
        forbidden_generated_question not in call["user_prompt"]
        for call in module.client.calls
    )


@pytest.mark.asyncio
async def test_requirement_builder_maps_each_final_question_exactly_once() -> None:
    questions = [
        "How can offline reinforcement learning transfer a robot arm policy?",
        "How should real-world transfer performance be evaluated?",
    ]
    entities = [
        TaskEntity(
            entity_id="E1",
            name="robot arm",
            source_mention="robot arm",
            role="primary_object",
            required=True,
        ),
        TaskEntity(
            entity_id="E2",
            name="offline reinforcement learning",
            source_mention="offline reinforcement learning",
            role="method",
            required=True,
        ),
    ]
    module = entity_module([{
        "requirements": [
            {
                "requirement_id": "R1",
                "sub_question": questions[0],
                "primary_entity_id": "E1",
                "related_entity_ids": ["E2"],
                "relation": "transfer policy",
                "required": True,
            },
            {
                "requirement_id": "R2",
                "sub_question": questions[1],
                "primary_entity_id": "E1",
                "related_entity_ids": [],
                "relation": "evaluate performance",
                "required": True,
            },
        ]
    }])

    requirements = await module._build_requirements(questions, entities)

    assert [item.sub_question for item in requirements] == questions
    assert {item.primary_entity_id for item in requirements} == {"E1"}
    prompt = module.client.calls[0]["user_prompt"]
    assert "E1" in prompt and "E2" in prompt


@pytest.mark.asyncio
async def test_requirement_builder_repairs_unknown_entity_without_changing_entities() -> None:
    question = "How can a robot arm transfer from simulation to reality?"
    entities = [TaskEntity(
        entity_id="E1",
        name="robot arm",
        source_mention="robot arm",
        role="primary_object",
        required=True,
    )]
    module = entity_module([
        {"requirements": [{
            "requirement_id": "R1",
            "sub_question": question,
            "primary_entity_id": "E99",
            "related_entity_ids": [],
            "relation": "transfer",
            "required": True,
        }]},
        {"requirements": [{
            "requirement_id": "R1",
            "sub_question": question,
            "primary_entity_id": "E1",
            "related_entity_ids": [],
            "relation": "transfer",
            "required": True,
        }]},
    ])

    requirements = await module._build_requirements([question], entities)

    assert requirements[0].primary_entity_id == "E1"
    assert len(module.client.calls) == 2
    assert "E99" not in module.client.calls[1]["user_prompt"]


@pytest.mark.asyncio
async def test_missing_primary_object_after_repair_fails_closed() -> None:
    invalid_round = {"entities": [candidate(
        "domain randomization", source_mention="域随机化", role="method",
    )]}
    invalid_audit = {
        "items": [audit_item(
            "domain randomization", source_mention="域随机化",
        )],
        "missing_explicit_entities": [],
        "complete": True,
    }
    module = entity_module([
        invalid_round,
        invalid_audit,
        invalid_round,
        invalid_audit,
    ])

    with pytest.raises(ValueError, match="required primary object"):
        await module._extract_and_audit_entities("如何改进机械臂迁移？")
