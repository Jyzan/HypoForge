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
            "missing_aspects": ["离线强化学习如何用于迁移"],
            "over_fragmented": False,
            "merge_instructions": [],
            "reason": "核心方法缺失",
        },
        {
            "sub_questions": [
                "离线强化学习如何实现机械臂从仿真到现实的策略迁移？"
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
        ["仿真环境与现实环境存在哪些差异？"],
    )

    assert result == [
        "仿真环境与现实环境存在哪些差异？",
        "离线强化学习如何实现机械臂从仿真到现实的策略迁移？",
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
            "merge_instructions": ["合并两个环境差异问题"],
            "reason": "同一关系被拆碎",
        },
        {
            "sub_questions": ["仿真与现实环境通过哪些差异影响机械臂策略迁移？"]
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
        ["摩擦力差异有什么影响？", "传感器噪声差异有什么影响？"],
    )

    assert result == ["仿真与现实环境通过哪些差异影响机械臂策略迁移？"]


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
        "missing_aspects": ["核心实现方法"],
        "over_fragmented": False,
        "merge_instructions": [],
        "reason": "仍未回答如何实现",
    }
    module = module_with([
        missing,
        {"sub_questions": ["新的背景问题是什么？"]},
        missing,
        {"sub_questions": ["另一个背景问题是什么？"]},
        missing,
    ])

    with pytest.raises(ValueError, match="core user intent"):
        await module._check_subquestion_coverage(
            "如何实现机械臂策略迁移？",
            ["仿真环境是什么？"],
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
                "离线强化学习如何实现机械臂从仿真到现实的策略迁移？",
                "如何评估机械臂策略迁移的现实性能？",
                "仿真与现实环境差异如何影响策略迁移？",
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
                    "name": "机械臂",
                    "source_mention": "机械臂",
                    "aliases": [],
                    "role": "primary_object",
                    "required": True,
                    "extraction_reason": "literal source mention",
                },
                {
                    "name": "离线强化学习",
                    "source_mention": "离线强化学习",
                    "aliases": ["offline reinforcement learning"],
                    "role": "method",
                    "required": True,
                    "extraction_reason": "literal source mention",
                },
            ]
        },
        {
            "items": [
                {
                    "candidate_name": "机械臂",
                    "accepted": True,
                    "source_mention": "机械臂",
                    "reason": "literal source mention",
                },
                {
                    "candidate_name": "离线强化学习",
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
                    "离线强化学习如何实现机械臂从仿真到现实的策略迁移？",
                    "如何评估机械臂策略迁移的现实性能？",
                    "仿真与现实环境差异如何影响策略迁移？",
                ], start=1)
            ]
        },
    ])

    card = await module._understand_question(
        "如何使用离线强化学习实现机械臂从仿真到现实的迁移？"
    )

    schema_properties = module.client.calls[0]["output_schema"]["properties"]
    assert set(schema_properties) == {"domain", "sub_questions"}
    assert card.key_entities == ["机械臂", "离线强化学习"]
    assert [entity.name for entity in card.task_contract.entities] == [
        "机械臂",
        "离线强化学习",
    ]
