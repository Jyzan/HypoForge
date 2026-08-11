"""Behavior tests for M1 atomic decomposition and semantic coverage repair."""

from __future__ import annotations

from typing import Any

import pytest

from hypoforge.modules.m1_problem_understanding import M1ProblemUnderstanding


class SequenceClient:
    def __init__(self, payloads: list[dict[str, Any]]):
        self.payloads = list(payloads)
        self.calls: list[dict[str, Any]] = []

    async def structured_chat(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if not self.payloads:
            raise AssertionError("unexpected extra M1 structured call")
        return self.payloads.pop(0)


def module_with(payloads: list[dict[str, Any]], *, rounds: int = 2):
    module = M1ProblemUnderstanding(
        mode="llm",
        coverage_max_rounds=rounds,
    )
    module.client = SequenceClient(payloads)
    return module


@pytest.mark.asyncio
async def test_missing_core_intent_is_supplemented_and_rechecked() -> None:
    module = module_with([
        {
            "sufficient": False,
            "core_intent_covered": False,
            "missing_aspects": ["use of offline reinforcement learning for transfer"],
            "over_fragmented": False,
            "merge_instructions": [],
            "reason": "核心方法缺失",
        },
        {
            "sub_questions": [
                "How can offline reinforcement learning transfer a robot arm policy from simulation to reality?"
            ]
        },
        {
            "sufficient": True,
            "core_intent_covered": True,
            "missing_aspects": [],
            "over_fragmented": False,
            "merge_instructions": [],
            "reason": "已覆盖",
        },
    ])

    result = await module._check_subquestion_coverage(
        "如何使用离线强化学习实现机械臂从仿真到现实的迁移？",
        ["How do simulation and real-world environments differ?"],
    )

    assert result == [
        "How do simulation and real-world environments differ?",
        "How can offline reinforcement learning transfer a robot arm policy from simulation to reality?",
    ]
    assert len(module.client.calls) == 3


@pytest.mark.asyncio
async def test_over_fragmented_questions_are_merged_and_rechecked() -> None:
    module = module_with([
        {
            "sufficient": True,
            "core_intent_covered": True,
            "missing_aspects": [],
            "over_fragmented": True,
            "merge_instructions": ["Merge the two environment-difference questions."],
            "reason": "同一关系被拆碎",
        },
        {
            "sub_questions": ["How do simulation-to-reality environment differences affect robot arm policy transfer?"]
        },
        {
            "sufficient": True,
            "core_intent_covered": True,
            "missing_aspects": [],
            "over_fragmented": False,
            "merge_instructions": [],
            "reason": "粒度合适",
        },
    ])

    result = await module._check_subquestion_coverage(
        "如何解决机械臂从仿真到现实的策略迁移？",
        [
            "How do friction differences affect robot arm policy transfer?",
            "How do sensor-noise differences affect robot arm policy transfer?",
        ],
    )

    assert result == [
        "How do simulation-to-reality environment differences affect robot arm policy transfer?"
    ]


@pytest.mark.asyncio
async def test_last_coverage_round_can_apply_its_merge_repair() -> None:
    split_questions = [
        "How can a model predict object position from one image?",
        "How can a model predict object rotation from one image?",
    ]
    merged_question = (
        "How can a model jointly predict object position and rotation from one image?"
    )
    over_fragmented = {
        "sufficient": True,
        "core_intent_covered": True,
        "missing_aspects": [],
        "over_fragmented": True,
        "merge_instructions": ["Merge position and rotation into one question."],
        "reason": "Parallel outputs of the same prediction task were split.",
    }
    module = module_with([
        over_fragmented,
        {"sub_questions": split_questions},  # First merge attempt is ineffective.
        over_fragmented,
        {"sub_questions": [merged_question]},
        {
            "sufficient": True,
            "core_intent_covered": True,
            "missing_aspects": [],
            "over_fragmented": False,
            "merge_instructions": [],
            "reason": "The parallel outputs are now merged.",
        },
    ], rounds=2)

    result = await module._check_subquestion_coverage(
        "How can a model infer object position and rotation from one image?",
        split_questions,
    )

    assert result == [merged_question]
    assert len(module.client.calls) == 5


@pytest.mark.asyncio
async def test_core_intent_missing_after_budget_fails_closed() -> None:
    missing = {
        "sufficient": False,
        "core_intent_covered": False,
        "missing_aspects": ["the core implementation method"],
        "over_fragmented": False,
        "merge_instructions": [],
        "reason": "仍未回答如何实现",
    }
    module = module_with([
        missing,
        {"sub_questions": ["What is the new background question?"]},
        missing,
        {"sub_questions": ["What is another background question?"]},
        missing,
    ])

    with pytest.raises(ValueError, match="core user intent"):
        await module._check_subquestion_coverage(
            "如何实现机械臂策略迁移？",
            ["What is the simulation environment?"],
        )


def test_atomic_validator_rejects_packed_parallel_question() -> None:
    violations = M1ProblemUnderstanding._sub_question_violations([
        "离线强化学习如何迁移；摩擦力变化又如何处理？？"
    ])

    assert len(violations) == 1
    assert "multiple question marks" in violations[0]
    assert "semicolon" in violations[0]


def test_atomic_validator_rejects_more_than_five_questions() -> None:
    violations = M1ProblemUnderstanding._sub_question_violations([
        f"Atomic research question {index}?" for index in range(1, 7)
    ])

    assert any("at most 5" in violation for violation in violations)


@pytest.mark.asyncio
async def test_coverage_supplement_is_semantically_merged_when_it_exceeds_five() -> None:
    initial = [f"Atomic research question {index}?" for index in range(1, 5)]
    module = module_with([
        {
            "sufficient": False,
            "core_intent_covered": True,
            "missing_aspects": ["missing aspect A", "missing aspect B"],
            "over_fragmented": False,
            "merge_instructions": [],
            "reason": "two indispensable aspects are missing",
        },
        {
            "sub_questions": [
                "Atomic missing aspect A?",
                "Atomic missing aspect B?",
            ]
        },
        {
            "sub_questions": [
                "Atomic research question 1?",
                "Atomic research question 2?",
                "Atomic research question 3?",
                "Atomic research question 4 with missing aspect A?",
                "Atomic missing aspect B?",
            ]
        },
        {
            "sufficient": True,
            "core_intent_covered": True,
            "missing_aspects": [],
            "over_fragmented": False,
            "merge_instructions": [],
            "reason": "complete within the fixed limit",
        },
    ])

    result = await module._check_subquestion_coverage(
        "How can the complete research objective be addressed?",
        initial,
    )

    assert len(result) == 5
    assert result[-2:] == [
        "Atomic research question 4 with missing aspect A?",
        "Atomic missing aspect B?",
    ]


@pytest.mark.asyncio
async def test_initial_decomposition_call_cannot_generate_entities_or_contract() -> None:
    """The first model call must not contaminate task entities via its own questions."""

    module = module_with([
        {
            "domain": ["robotics", "reinforcement learning"],
            "sub_questions": [
                "How can offline reinforcement learning transfer a robot arm policy from simulation to reality?",
                "How should real-world robot arm transfer performance be evaluated?",
                "How do simulation-to-reality environment differences affect policy transfer?",
            ],
        },
        {
            "sufficient": True,
            "core_intent_covered": True,
            "missing_aspects": [],
            "over_fragmented": False,
            "merge_instructions": [],
            "reason": "已覆盖",
        },
        {
            "entities": [
                {
                    "name": "robot arm",
                    "source_mention": "机械臂",
                    "aliases": [],
                    "role": "primary_object",
                    "required": True,
                    "extraction_reason": "literal source mention",
                },
                {
                    "name": "offline reinforcement learning",
                    "source_mention": "离线强化学习",
                    "aliases": ["offline RL"],
                    "role": "method",
                    "required": True,
                    "extraction_reason": "literal source mention",
                },
            ]
        },
        {
            "items": [
                {
                    "candidate_name": "robot arm",
                    "accepted": True,
                    "source_mention": "机械臂",
                    "reason": "literal source mention",
                },
                {
                    "candidate_name": "offline reinforcement learning",
                    "accepted": True,
                    "source_mention": "离线强化学习",
                    "reason": "literal source mention",
                },
            ],
            "missing_explicit_entities": [],
            "complete": True,
        },
        {
            "requirements": [
                {
                    "requirement_id": f"R{index}",
                    "sub_question": question,
                    "primary_entity_id": "E1",
                    "related_entity_ids": ["E2"],
                    "relation": "address atomic question",
                    "required": True,
                }
                for index, question in enumerate([
                    "How can offline reinforcement learning transfer a robot arm policy from simulation to reality?",
                    "How should real-world robot arm transfer performance be evaluated?",
                    "How do simulation-to-reality environment differences affect policy transfer?",
                ], start=1)
            ]
        },
    ])

    card = await module._understand_question(
        "如何使用离线强化学习实现机械臂从仿真到现实的迁移？"
    )

    schema_properties = module.client.calls[0]["output_schema"]["properties"]
    assert set(schema_properties) == {"domain", "sub_questions"}
    assert card.key_entities == ["robot arm", "offline reinforcement learning"]
    assert [entity.name for entity in card.task_contract.entities] == [
        "robot arm",
        "offline reinforcement learning",
    ]


def test_sub_question_validator_rejects_non_english_output() -> None:
    violations = M1ProblemUnderstanding._sub_question_violations([
        "如何训练旋转目标检测模型？",
    ])

    assert any("entirely in English" in item for item in violations)
