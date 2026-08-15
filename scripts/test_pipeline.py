"""整合测试集：M1-M6 各模块流程 + 完整管线回归。
每次修改后运行：pytest tests/test_pipeline.py -q
"""

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
    role: str = "other",
    required: bool = False,
    aliases: list[str] | None = None,
    mention: str | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "source_mention": mention or name,
        "aliases": aliases or [],
        "role": role,
        "required": required,
        "extraction_reason": "用户原文明确出现",
    }


def audit_item(name: str, accepted: bool = True, mention: str | None = None) -> dict[str, Any]:
    return {
        "candidate_name": name,
        "accepted": accepted,
        "source_mention": mention or name,
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
            candidate("robot arm", role="primary_object", required=True, mention="机械臂"),
            candidate("domain randomization", role="method", mention="域随机化"),
        ]},
        {
            "items": [audit_item("robot arm", mention="机械臂"), audit_item("domain randomization", mention="域随机化")],
            "missing_explicit_entities": [],
            "complete": True,
        },
    ])

    entities = await module._extract_and_audit_entities(
        "如何改进机械臂从仿真到现实的迁移？"
    )

    assert [entity.name for entity in entities] == ["robot arm"]
    assert entities[0].entity_id == "E1"


@pytest.mark.asyncio
async def test_entity_audit_missing_term_triggers_one_repair_and_reaudit() -> None:
    module = entity_module([
        {"entities": [candidate("robot arm", role="primary_object", required=True, mention="机械臂")]},
        {
            "items": [audit_item("robot arm", mention="机械臂")],
            "missing_explicit_entities": ["offline reinforcement learning"],
            "complete": False,
        },
        {"entities": [
            candidate("robot arm", role="primary_object", required=True, mention="机械臂"),
            candidate(
                "offline reinforcement learning",
                role="method",
                required=True,
                aliases=["offline RL"],
                mention="离线强化学习",
            ),
        ]},
        {
            "items": [audit_item("robot arm", mention="机械臂"), audit_item("offline reinforcement learning", mention="离线强化学习")],
            "missing_explicit_entities": [],
            "complete": True,
        },
    ])

    entities = await module._extract_and_audit_entities(
        "如何使用离线强化学习控制机械臂？"
    )

    assert [entity.name for entity in entities] == ["robot arm", "offline reinforcement learning"]
    assert entities[1].aliases == ["offline RL"]
    assert len(module.client.calls) == 4


@pytest.mark.asyncio
async def test_entity_prompts_never_receive_generated_subquestions() -> None:
    module = entity_module([
        {"entities": [candidate("robot arm", role="primary_object", required=True, mention="机械臂")]},
        {
            "items": [audit_item("robot arm", mention="机械臂")],
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
    invalid_round = {"entities": [candidate("域随机化", role="method")]}
    invalid_audit = {
        "items": [audit_item("域随机化")],
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


# --- merged from test_m1_followup_triage.py ---

from typing import Any

import pytest

from hypoforge.modules.m1_problem_understanding import M1ProblemUnderstanding
from hypoforge.state import (
    EvidenceGraph,
    EvidenceNode,
    FollowupRequest,
    HypothesisCard,
    PipelineState,
    ProblemCard,
)


class ScriptedClient:
    def __init__(self, *payloads: Any):
        self.payloads = list(payloads)
        self.calls: list[dict[str, Any]] = []

    async def structured_chat(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if not self.payloads:
            raise AssertionError("unexpected extra M1 LLM call")
        payload = self.payloads.pop(0)
        if isinstance(payload, BaseException):
            raise payload
        return payload


def make_module(client: Any, *, threshold: float = 0.75) -> M1ProblemUnderstanding:
    module = M1ProblemUnderstanding(
        mode="llm",
        followup_routing=True,
        followup_triage_confidence_threshold=threshold,
    )
    module.client = client
    return module


def triage_payload(
    category: str = "presentation_adjustment",
    *,
    skip_search: bool = True,
    rebuild_problem_card: bool = False,
    confidence: float = 0.99,
) -> dict[str, Any]:
    return {
        "category": category,
        "skip_search": skip_search,
        "rebuild_problem_card": rebuild_problem_card,
        "rationale": "仅调整语言和呈现方式。",
        "confidence": confidence,
    }


def parent_state(text: str, *, graph: bool = True) -> PipelineState:
    evidence_graph = None
    if graph:
        evidence_graph = EvidenceGraph(nodes=[
            EvidenceNode(id="n1", type="claim", label="NRF2 evidence")
        ])
    return PipelineState(
        input_question=text,
        problem_card=ProblemCard(
            original_question="Nrf2 激动剂在阿尔茨海默病中的神经保护机制",
            domain=["neuroscience"],
            sub_questions=["Nrf2 通路如何调控氧化应激？"],
            key_entities=["Nrf2 激动剂", "阿尔茨海默病"],
        ),
        followup=FollowupRequest(text=text, parent_run_id="parent-run"),
        evidence_graph=evidence_graph,
        best_hypotheses=[HypothesisCard(hypothesis_id="h1", statement="s1")],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    ["给我中文方案", "翻译成英文", "改成表格", "写得更简洁"],
)
async def test_presentation_followup_always_calls_llm_and_reuses_parent_card(
    text: str,
) -> None:
    """Removing the LLM triage call would silently bypass user-intent review."""

    client = ScriptedClient(triage_payload())
    module = make_module(client)
    state = parent_state(text)

    patch = await module(state)

    assert len(client.calls) == 1
    assert patch["followup"].skip_search is True
    assert patch["problem_card"] == state.problem_card
    assert patch["problem_card"] is not state.problem_card


@pytest.mark.asyncio
async def test_presentation_followup_without_reusable_graph_fails_open_to_search() -> None:
    """A formatting decision cannot route into an evidence-free M4 run."""

    client = ScriptedClient(triage_payload())
    module = make_module(client)

    patch = await module(parent_state("给我中文方案", graph=False))

    assert len(client.calls) == 1
    assert patch["followup"].skip_search is False


@pytest.mark.asyncio
async def test_low_confidence_followup_fails_open_to_search() -> None:
    """Uncertain triage must spend search cost rather than reuse wrong evidence."""

    client = ScriptedClient(triage_payload(confidence=0.40))
    module = make_module(client)

    patch = await module(parent_state("帮我调整一下"))

    assert patch["followup"].skip_search is False


@pytest.mark.asyncio
async def test_inconsistent_presentation_flags_fail_open_to_search() -> None:
    """A malformed decision cannot claim presentation-only while rebuilding."""

    client = ScriptedClient(triage_payload(rebuild_problem_card=True))
    module = make_module(client)

    patch = await module(parent_state("给我中文方案"))

    assert patch["followup"].skip_search is False


@pytest.mark.asyncio
async def test_research_change_requests_full_m1_path() -> None:
    client = ScriptedClient(triage_payload(
        "research_change",
        skip_search=False,
        rebuild_problem_card=True,
    ))
    module = make_module(client)
    rebuilt = ProblemCard(
        original_question="combined research task",
        domain=["neuroscience"],
        sub_questions=["new atomic question"],
        key_entities=["new object"],
    )
    received: list[str] = []

    async def rebuild(question: str) -> ProblemCard:
        received.append(question)
        return rebuilt

    module._understand_question = rebuild

    patch = await module(parent_state("加入帕金森病模型重新研究"))

    assert patch["followup"].skip_search is False
    assert patch["problem_card"] is rebuilt
    assert len(received) == 1
    assert "Original research question:" in received[0]
    assert "research-changing follow-up:" in received[0]


@pytest.mark.asyncio
async def test_triage_event_records_llm_rationale(monkeypatch) -> None:
    events: list[dict[str, Any]] = []

    def record(event_type: str, **kwargs: Any) -> None:
        events.append({"event_type": event_type, **kwargs})

    monkeypatch.setattr(
        "hypoforge.modules.m1_problem_understanding.emit_event",
        record,
    )
    module = make_module(ScriptedClient(triage_payload()))

    await module(parent_state("给我中文方案"))

    completed = [
        event for event in events
        if event["event_type"] == "tool_completed"
        and event.get("tool") == "qwen_followup_triage"
    ]
    assert len(completed) == 1
    assert completed[0]["details"]["decision_source"] == "llm_triage"
    assert completed[0]["details"]["rationale"] == "仅调整语言和呈现方式。"


@pytest.mark.asyncio
async def test_triage_receives_parent_artifact_summary() -> None:
    client = ScriptedClient(triage_payload())
    module = make_module(client)

    await module(parent_state("给我中文方案"))

    prompt = client.calls[0]["user_prompt"]
    assert "Parent run artefacts:" in prompt
    assert "best_hypotheses: already produced" in prompt
    assert "给我中文方案" in prompt


# ==================== test_m2 ====================

from rich.console import Console

from hypoforge.display.m2_progress import M2ProgressReporter
from hypoforge.observability import bind_event_sink, notify_event


def _event(event_type: str, *, tool: str = "", details: dict | None = None, **extra) -> dict:
    return {
        "event_type": event_type,
        "module": "m2",
        "tool": tool,
        "details": details or {},
        **extra,
    }


def test_m2_progress_ignores_other_modules_and_disabled_output() -> None:
    output = Console(record=True, force_terminal=False, width=100)
    reporter = M2ProgressReporter(output=output)

    reporter.handle_event({"event_type": "module_started", "module": "m1"})
    reporter.handle_event(_event("module_started"))
    disabled_output = Console(record=True, force_terminal=False, width=100)
    M2ProgressReporter(enabled=False, output=disabled_output).handle_event(
        _event("module_started")
    )

    assert "M2 / Agentic" in output.export_text()
    assert disabled_output.export_text() == ""


def test_m2_progress_renders_reading_and_grounding_summary() -> None:
    output = Console(record=True, force_terminal=False, width=100)
    reporter = M2ProgressReporter(output=output)

    reporter.handle_event(_event("module_started"))
    reporter.handle_event(
        _event("tool_started", tool="query_planner", details={"round": 1})
    )
    reporter.handle_event(
        _event(
            "tool_result",
            tool="m2_export",
            details={"papers": 5, "evidence": 18, "knowledge_entries": 12},
            elapsed_seconds=1.25,
        )
    )
    reporter.handle_event(
        _event(
            "module_completed",
            details={
                "sub_questions": 1,
                "papers_retrieved": 5,
                "knowledge_entries": 12,
                "export_runs": 1,
            },
        )
    )

    rendered = output.export_text()
    assert "Query planning" in rendered
    assert "M2 -> M3 evidence export" in rendered
    assert "papers 5 | evidence 18 | knowledge 12" in rendered
    assert "sub-questions 1 | papers 5 | knowledge 12 | exports 1" in rendered
    assert "M2 complete" in rendered


def test_event_sink_presentation_failure_is_isolated() -> None:
    class BrokenSink:
        def handle_event(self, event):
            raise RuntimeError("terminal failure")

    with bind_event_sink(BrokenSink()):
        notify_event(_event("module_started"))


def test_m2_progress_failure_keeps_exception_diagnostic_readable() -> None:
    output = Console(record=True, force_terminal=False, width=100)
    reporter = M2ProgressReporter(output=output)

    reporter.handle_event(_event("module_started"))
    reporter.handle_event(
        _event(
            "tool_failed",
            tool="source:pubmed",
            message="M2 Tool failure: source:pubmed: TimeoutError: upstream unavailable",
        )
    )

    rendered = output.export_text()
    assert "Source search / pubmed" in rendered
    assert "TimeoutError: upstream unavailable" in rendered
    assert "M2 Tool failure" not in rendered


def test_m2_progress_escapes_untrusted_query_markup() -> None:
    output = Console(record=True, force_terminal=False, width=100)
    reporter = M2ProgressReporter(output=output)

    reporter.handle_event(_event("module_started"))
    reporter.handle_event(
        _event(
            "tool_started",
            tool="query_planner",
            details={"round": 1, "query": "[malformed markup"},
        )
    )

    assert "[malformed markup" in output.export_text()


def test_m2_progress_merges_completion_and_result_into_one_status_line() -> None:
    output = Console(record=True, force_terminal=False, width=100)
    reporter = M2ProgressReporter(output=output)

    reporter.handle_event(_event("module_started"))
    reporter.handle_event(
        _event(
            "tool_completed",
            tool="query_planner",
            details={"round": 1},
            elapsed_seconds=0.4,
        )
    )
    reporter.handle_event(
        _event(
            "tool_result",
            tool="query_planner",
            details={"round": 1, "queries": [{"text": "robot arm"}]},
        )
    )

    rendered = output.export_text()
    assert rendered.count("[OK]") == 1
    assert "01/13" in rendered
    assert "1 query" in rendered


def test_m2_progress_keeps_same_stage_completions_for_each_sub_question() -> None:
    output = Console(record=True, force_terminal=False, width=100)
    reporter = M2ProgressReporter(output=output)

    reporter.handle_event(_event("module_started"))
    reporter.handle_event(
        _event(
            "tool_completed",
            tool="reading_workflow",
            details={"sub_question": "question one"},
            elapsed_seconds=1.0,
        )
    )
    reporter.handle_event(
        _event(
            "tool_completed",
            tool="reading_workflow",
            details={"sub_question": "question two"},
            elapsed_seconds=2.0,
        )
    )
    reporter.handle_event(_event("module_completed", details={}))

    rendered = output.export_text()
    assert rendered.count("Full-text / abstract reading") == 2


# --- merged from test_m2_supplement.py ---

import pytest

from hypoforge.literature.adapter import AgenticM2Adapter
from hypoforge.literature.models import (
    CoverageReport,
    EvidenceChunk,
    EvidenceLinkedKnowledge,
    PaperReadingResult,
    PaperRecord,
    SearchRunResult,
    StopReason,
)
from hypoforge.literature.search import IterativeSearchAgent
from hypoforge.memory import PaperStore, paper_key
from hypoforge.observability import RunEventRecorder, bind_recorder
from hypoforge.state import (
    ConfidenceLevel,
    EvidenceGap,
    KnowledgeEntry,
    KnowledgeEntryType,
    LiteratureResult,
    M2KnowledgeExport,
    M2KnowledgeRun,
    M2PaperExport,
    PipelineState,
    ProblemCard,
    SearchLedger,
)



# ==================== 测试替身（原 scripts/test_doubles.py） ====================

import asyncio
from collections.abc import Sequence

from hypoforge.literature.models import (
    CoverageReport,
    PaperReadingResult,
    PaperRecord,
    ScoutNote,
    SearchQuery,
    SearchRunResult,
    SearchState,
)
from hypoforge.literature.protocols import (
    CoverageEvaluatorProtocol,
    LiteratureSourceProtocol,
    PaperDeduplicatorProtocol,
    PaperRankerProtocol,
    QueryPlannerProtocol,
    ReadingExtractionWorkflowProtocol,
    ScoutReaderProtocol,
)


class FakePlanner(QueryPlannerProtocol):
    def __init__(
        self,
        plans: Sequence[Sequence[SearchQuery]],
        error: Exception | None = None,
    ) -> None:
        self.plans = [list(plan) for plan in plans]
        self.error = error
        self.states: list[SearchState] = []

    async def plan(
        self,
        sub_question: str,
        key_entities: Sequence[str] = (),
        domains: Sequence[str] = (),
        question_type: str = "",
        state: SearchState | None = None,
    ) -> list[SearchQuery]:
        if self.error:
            raise self.error
        self.states.append((state or SearchState()).model_copy(deep=True))
        index = min(len(self.states) - 1, len(self.plans) - 1)
        return [query.model_copy(deep=True) for query in self.plans[index]]


class FakeSource(LiteratureSourceProtocol):
    def __init__(
        self,
        source_name: str,
        results: dict[str, Sequence[PaperRecord]] | None = None,
        errors: dict[str, Exception] | None = None,
        delays: dict[str, float] | None = None,
    ) -> None:
        self.source_name = source_name
        self.results = {key: list(value) for key, value in (results or {}).items()}
        self.errors = errors or {}
        self.delays = delays or {}
        self.calls: list[SearchQuery] = []

    async def search(self, query: SearchQuery, limit: int = 20) -> list[PaperRecord]:
        self.calls.append(query.model_copy(deep=True))
        if query.text in self.delays:
            await asyncio.sleep(self.delays[query.text])
        if query.text in self.errors:
            raise self.errors[query.text]
        return [paper.model_copy(deep=True) for paper in self.results.get(query.text, [])][
            :limit
        ]


class FakeDeduplicator(PaperDeduplicatorProtocol):
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error

    async def deduplicate(
        self,
        papers: Sequence[PaperRecord],
        existing_papers: Sequence[PaperRecord] = (),
    ) -> list[PaperRecord]:
        if self.error:
            raise self.error
        canonical = {paper.paper_id: paper for paper in existing_papers}
        unique: dict[str, PaperRecord] = {}
        for paper in papers:
            unique.setdefault(paper.paper_id, canonical.get(paper.paper_id, paper))
        return list(unique.values())


class FakeRanker(PaperRankerProtocol):
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error

    async def rank(
        self,
        sub_question: str,
        papers: Sequence[PaperRecord],
        limit: int,
    ) -> list[PaperRecord]:
        if self.error:
            raise self.error
        return list(papers)[:limit]


class FakeScoutReader(ScoutReaderProtocol):
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[list[str]] = []

    async def read(
        self,
        sub_question: str,
        papers: Sequence[PaperRecord],
    ) -> list[ScoutNote]:
        if self.error:
            raise self.error
        self.calls.append([paper.paper_id for paper in papers])
        return [
            ScoutNote(
                paper_id=paper.paper_id,
                key_terms=[f"term-{paper.paper_id}"],
                relevance_to_question=0.8,
            )
            for paper in papers
        ]


class FakeCoverageEvaluator(CoverageEvaluatorProtocol):
    def __init__(
        self,
        reports: Sequence[CoverageReport],
        error: Exception | None = None,
    ) -> None:
        self.reports = list(reports)
        self.error = error
        self.states: list[SearchState] = []
        self.note_calls: list[list[str]] = []

    async def evaluate(
        self,
        sub_question: str,
        papers: Sequence[PaperRecord],
        scout_notes: Sequence[ScoutNote],
        state: SearchState,
    ) -> CoverageReport:
        if self.error:
            raise self.error
        self.states.append(state.model_copy(deep=True))
        self.note_calls.append([note.paper_id for note in scout_notes])
        index = min(len(self.states) - 1, len(self.reports) - 1)
        return self.reports[index].model_copy(deep=True)


class FakeReadingWorkflow(ReadingExtractionWorkflowProtocol):
    def __init__(self, results: Sequence[PaperReadingResult]) -> None:
        self.results = list(results)
        self.calls: list[list[str]] = []
        self.search_contexts: list[SearchRunResult | None] = []

    async def run(
        self,
        sub_question: str,
        papers: Sequence[PaperRecord],
        search_context: SearchRunResult | None = None,
    ) -> list[PaperReadingResult]:
        self.calls.append([paper.paper_id for paper in papers])
        self.search_contexts.append(search_context)
        return [result.model_copy(deep=True) for result in self.results]


SUB_QUESTION = "What is the Hsp70 mechanism?"


# ---------------------------------------------------------------------------
# Local fakes / helpers
# ---------------------------------------------------------------------------


class RecordingSearchAgent:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    async def run(self, sub_question, **kwargs):
        self.calls.append((sub_question, kwargs))
        return self.results[len(self.calls) - 1].model_copy(deep=True)


class ExplodingSearchAgent:
    async def run(self, *args, **kwargs):
        raise AssertionError("search agent must not run on a cache hit")


class ExplodingReadingWorkflow:
    async def run(self, *args, **kwargs):
        raise AssertionError("reading workflow must not run on a cache hit")


class EchoReadingWorkflow:
    """Returns one reading per input paper (optionally with evidence)."""

    def __init__(self, with_evidence: bool = False) -> None:
        self.with_evidence = with_evidence
        self.calls = []

    async def run(self, sub_question, papers, search_context=None):
        self.calls.append([paper.paper_id for paper in papers])
        results = []
        for paper in papers:
            if not self.with_evidence:
                results.append(PaperReadingResult(paper_id=paper.paper_id))
                continue
            results.append(
                PaperReadingResult(
                    paper_id=paper.paper_id,
                    evidence=[
                        EvidenceChunk(
                            evidence_id=f"ev-{paper.paper_id}",
                            paper_id=paper.paper_id,
                            chunk_id=f"chunk-{paper.paper_id}",
                            quote="observed binding",
                            normalized_claim="observed binding",
                            relevance_score=0.9,
                        )
                    ],
                    knowledge_entries=[
                        EvidenceLinkedKnowledge(
                            entry_id=f"ke-{paper.paper_id}",
                            entry_type=KnowledgeEntryType.MECHANISTIC_CONCLUSION,
                            content=f"Finding from {paper.paper_id}",
                            confidence=ConfidenceLevel.HIGH,
                            evidence_ids=[f"ev-{paper.paper_id}"],
                        )
                    ],
                )
            )
        return results


def make_problem_card() -> ProblemCard:
    return ProblemCard(
        original_question="Q",
        sub_questions=[SUB_QUESTION],
        key_entities=["Hsp70"],
        domain=["biology"],
    )


def make_prior_run() -> M2KnowledgeRun:
    return M2KnowledgeRun(
        sub_question=SUB_QUESTION,
        papers=[M2PaperExport(paper_id="old-1", title="Old paper", doi="10.1000/old")],
    )


def make_prior_entry() -> KnowledgeEntry:
    return KnowledgeEntry(
        id="KE_old",
        type=KnowledgeEntryType.ESTABLISHED_FACT,
        content="old fact",
        source_paper_id="old-1",
        evidence_ids=["ev-old"],
    )


def make_supplement_state(**overrides) -> PipelineState:
    base = dict(
        input_question="Q",
        problem_card=make_problem_card(),
        search_round=1,
        run_id="run-1",
        literature_results=[
            LiteratureResult(
                sub_question=SUB_QUESTION,
                papers_retrieved=1,
                knowledge_entries=[make_prior_entry()],
            )
        ],
        m2_knowledge_export=M2KnowledgeExport(runs=[make_prior_run()]),
        search_ledger=SearchLedger(
            queries_issued=["initial query"], paper_keys=["doi:10.1000/old"]
        ),
    )
    base.update(overrides)
    return PipelineState(**base)


def make_base_agent(source_results):
    source = FakeSource("pubmed", results=source_results)
    agent = IterativeSearchAgent(
        query_planner=FakePlanner([[]]),
        sources=[source],
        deduplicator=FakeDeduplicator(),
        ranker=FakeRanker(),
        scout_reader=FakeScoutReader(),
        coverage_evaluator=FakeCoverageEvaluator(
            [CoverageReport(sufficient=True)]
        ),
    )
    return agent, source


def make_gap(**overrides) -> EvidenceGap:
    base = dict(
        description="Missing Hsp70 co-chaperone data",
        suggested_queries=["Hsp70 co-chaperone binding"],
        target_sub_question=SUB_QUESTION,
    )
    base.update(overrides)
    return EvidenceGap(**base)


# ---------------------------------------------------------------------------
# Trigger semantics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_first_round_runs_full_flow() -> None:
    paper = PaperRecord(paper_id="p1", title="T", sources=["pubmed"])
    agent = RecordingSearchAgent(
        [
            SearchRunResult(
                sub_question=SUB_QUESTION,
                final_papers=[paper],
                stop_reason=StopReason.COVERAGE_SATISFIED,
            )
        ]
    )
    adapter = AgenticM2Adapter(
        search_agent=agent,
        reading_workflow=FakeReadingWorkflow([PaperReadingResult(
            paper_id="p1",
            evidence=[EvidenceChunk(
                evidence_id="ev-p1", paper_id="p1", chunk_id="chunk-p1",
                quote="Observed binding.", normalized_claim="Observed binding.", relevance_score=0.9,
            )],
            knowledge_entries=[EvidenceLinkedKnowledge(
                entry_id="ke-p1",
                entry_type=KnowledgeEntryType.MECHANISTIC_CONCLUSION,
                content="Observed binding.",
                confidence=ConfidenceLevel.HIGH,
                evidence_ids=["ev-p1"],
            )],
        )]),
    )
    state = PipelineState(input_question="Q", problem_card=make_problem_card())

    output = await adapter(state)

    assert len(agent.calls) == 1
    # fresh flow returns the Agentic literature and grounded export fields
    assert set(output) == {"literature_results", "m2_knowledge_export"}


@pytest.mark.asyncio
async def test_no_open_gap_runs_full_flow_even_on_later_round() -> None:
    paper = PaperRecord(paper_id="p1", title="T", sources=["pubmed"])
    agent = RecordingSearchAgent(
        [
            SearchRunResult(
                sub_question=SUB_QUESTION,
                final_papers=[paper],
                stop_reason=StopReason.COVERAGE_SATISFIED,
            )
        ]
    )
    adapter = AgenticM2Adapter(
        search_agent=agent,
        reading_workflow=FakeReadingWorkflow([PaperReadingResult(
            paper_id="p1",
            evidence=[EvidenceChunk(
                evidence_id="ev-p1", paper_id="p1", chunk_id="chunk-p1",
                quote="Observed binding.", normalized_claim="Observed binding.", relevance_score=0.9,
            )],
            knowledge_entries=[EvidenceLinkedKnowledge(
                entry_id="ke-p1",
                entry_type=KnowledgeEntryType.MECHANISTIC_CONCLUSION,
                content="Observed binding.",
                confidence=ConfidenceLevel.HIGH,
                evidence_ids=["ev-p1"],
            )],
        )]),
    )
    state = make_supplement_state(
        evidence_gaps=[make_gap(status="closed")],
    )

    output = await adapter(state)

    assert len(agent.calls) == 1
    assert set(output) == {"literature_results", "m2_knowledge_export"}


# ---------------------------------------------------------------------------
# Cache-hit path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_supplement_cache_hit_rebuilds_paper_level_increment(
    tmp_path,
) -> None:
    cached = M2PaperExport(
        paper_id="cached-1", title="Cached paper", doi="10.1000/cached"
    )
    store = PaperStore(tmp_path)
    keys = store.upsert_papers([cached], run_id="run-0", round=1)
    store.record_query("hsp70 chaperone mechanism", keys, run_id="run-0", round=1)

    state = make_supplement_state(
        memory_cache_dir=str(tmp_path),
        evidence_gaps=[
            make_gap(suggested_queries=["Hsp70 chaperone mechanism"])
        ],
    )
    adapter = AgenticM2Adapter(
        search_agent=ExplodingSearchAgent(),
        reading_workflow=ExplodingReadingWorkflow(),
    )
    recorder = RunEventRecorder(tmp_path / "run", "run-1")

    with bind_recorder(recorder):
        output = await adapter(state)

    # gap attempted → pending_grounding
    gaps = output["evidence_gaps"]
    assert len(gaps) == 1
    assert gaps[0].status == "open"

    # literature_results merged monotonically (old entry preserved)
    results = output["literature_results"]
    assert len(results) == 1
    assert results[0].papers_retrieved == 2
    assert [entry.id for entry in results[0].knowledge_entries] == ["KE_old"]

    # export appends a new, provenance-valid paper-level run
    export = output["m2_knowledge_export"]
    assert len(export.runs) == 2
    new_run = export.runs[1]
    assert new_run.sub_question == SUB_QUESTION
    assert [paper.paper_id for paper in new_run.papers] == ["cached-1"]
    assert new_run.evidence == []
    assert new_run.knowledge_entries == []  # metadata cache hits are not grounding
    assert new_run.search_provenance.stop_reason == "cache_hit"
    assert (
        new_run.search_provenance.queries[0].target_gap
        == "Missing Hsp70 co-chaperone data"
    )

    # ledger gains the cached paper key but no new live query
    ledger = output["search_ledger"]
    assert ledger.queries_issued == ["initial query"]
    assert "doi:10.1000/cached" in ledger.paper_keys

    # memory_hit event carries gap_id + hit count
    hits = [
        event
        for event in recorder.read_events()
        if event["event_type"] == "memory_hit"
    ]
    assert len(hits) == 1
    assert hits[0]["details"]["gap_id"] == gaps[0].gap_id
    assert hits[0]["details"]["hits"] == 1


# ---------------------------------------------------------------------------
# Live gap search path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_supplement_search_merges_and_updates_ledger() -> None:
    new_paper = PaperRecord(
        paper_id="new-1", title="New paper", doi="10.1000/new", sources=["pubmed"]
    )
    agent, source = make_base_agent({"Hsp70 co-chaperone binding": [new_paper]})
    adapter = AgenticM2Adapter(
        search_agent=agent, reading_workflow=EchoReadingWorkflow(with_evidence=True)
    )
    state = make_supplement_state(evidence_gaps=[make_gap()])

    output = await adapter(state)

    # exactly the suggested query hit the source (no cache dir → no lookup)
    assert [query.text for query in source.calls] == ["Hsp70 co-chaperone binding"]

    results = output["literature_results"]
    assert len(results) == 1
    assert results[0].papers_retrieved == 2
    entry_ids = [entry.id for entry in results[0].knowledge_entries]
    assert "KE_old" in entry_ids and "ke-new-1" in entry_ids

    export = output["m2_knowledge_export"]
    assert len(export.runs) == 2
    new_run = export.runs[1]
    assert [paper.paper_id for paper in new_run.papers] == ["new-1"]
    # target_gap activated on the issued queries
    assert new_run.search_provenance.queries
    assert all(
        query.target_gap == "Missing Hsp70 co-chaperone data"
        for query in new_run.search_provenance.queries
    )
    # validate_provenance holds (evidence/knowledge reference the new paper)
    assert new_run.evidence[0].paper_id == "new-1"
    assert new_run.knowledge_entries[0].source_paper_id == "new-1"

    ledger = output["search_ledger"]
    assert ledger.queries_issued == ["initial query", "Hsp70 co-chaperone binding"]
    assert ledger.paper_keys == ["doi:10.1000/old", "doi:10.1000/new"]

    assert output["evidence_gaps"][0].status == "pending_grounding"


@pytest.mark.asyncio
async def test_supplement_dedups_queries_already_in_ledger() -> None:
    new_paper = PaperRecord(
        paper_id="new-1", title="New paper", doi="10.1000/new", sources=["pubmed"]
    )
    agent, source = make_base_agent({"Hsp70 co-chaperone binding": [new_paper]})
    adapter = AgenticM2Adapter(
        search_agent=agent, reading_workflow=EchoReadingWorkflow(with_evidence=True)
    )
    state = make_supplement_state(
        evidence_gaps=[make_gap()],
        search_ledger=SearchLedger(
            queries_issued=["hsp70   co-chaperone binding!!"],
            paper_keys=["doi:10.1000/old"],
        ),
    )

    output = await adapter(state)

    assert source.calls == []  # duplicate query dropped before any search
    assert len(output["m2_knowledge_export"].runs) == 1  # no new run
    assert output["literature_results"][0].papers_retrieved == 1
    assert output["search_ledger"].queries_issued == [
        "hsp70   co-chaperone binding!!"
    ]
    assert output["evidence_gaps"][0].status == "open"


@pytest.mark.asyncio
async def test_supplement_skips_papers_already_in_prior_export() -> None:
    duplicate = PaperRecord(
        paper_id="dup-1", title="Old paper", doi="10.1000/old", sources=["pubmed"]
    )
    agent, source = make_base_agent({"Hsp70 co-chaperone binding": [duplicate]})
    adapter = AgenticM2Adapter(
        search_agent=agent, reading_workflow=EchoReadingWorkflow(with_evidence=True)
    )
    state = make_supplement_state(evidence_gaps=[make_gap()])

    output = await adapter(state)

    assert len(source.calls) == 1
    assert len(output["m2_knowledge_export"].runs) == 1  # duplicate discarded
    assert output["literature_results"][0].papers_retrieved == 1
    assert output["search_ledger"].paper_keys == ["doi:10.1000/old"]
    assert output["evidence_gaps"][0].status == "open"


@pytest.mark.asyncio
async def test_supplement_paper_budget_caps_new_papers() -> None:
    papers = [
        PaperRecord(
            paper_id=f"new-{index}",
            title=f"New paper {index}",
            doi=f"10.1000/new{index}",
            sources=["pubmed"],
        )
        for index in range(3)
    ]
    agent, source = make_base_agent({"Hsp70 co-chaperone binding": papers})
    adapter = AgenticM2Adapter(
        search_agent=agent,
        reading_workflow=EchoReadingWorkflow(with_evidence=True),
        supplement_paper_budget=1,
    )
    state = make_supplement_state(evidence_gaps=[make_gap()])

    output = await adapter(state)

    new_keys = [
        key
        for key in output["search_ledger"].paper_keys
        if key != "doi:10.1000/old"
    ]
    assert len(new_keys) == 1
    assert output["literature_results"][0].papers_retrieved == 2
    assert len(output["m2_knowledge_export"].runs[1].papers) == 1


@pytest.mark.asyncio
async def test_supplement_without_cache_dir_still_searches(tmp_path) -> None:
    new_paper = PaperRecord(
        paper_id="new-1", title="New paper", doi="10.1000/new", sources=["pubmed"]
    )
    agent, source = make_base_agent({"Hsp70 co-chaperone binding": [new_paper]})
    adapter = AgenticM2Adapter(
        search_agent=agent, reading_workflow=EchoReadingWorkflow(with_evidence=True)
    )
    # memory_cache_dir left empty → degraded but functional
    state = make_supplement_state(evidence_gaps=[make_gap()])

    output = await adapter(state)

    assert len(source.calls) == 1
    assert output["literature_results"][0].papers_retrieved == 2


# ---------------------------------------------------------------------------
# PaperStore unit tests
# ---------------------------------------------------------------------------


def test_paper_key_priority_and_normalisation() -> None:
    rich = {
        "doi": "HTTPS://DOI.ORG/10.1000/ABC",
        "pmid": "123",
        "external_ids": {"arxiv": "2301.00001"},
        "title": "Some title",
    }
    assert paper_key(rich) == "doi:10.1000/abc"
    assert paper_key({"pmid": "123", "title": "Some title"}) == "pmid:123"
    assert (
        paper_key({"external_ids": {"arXiv": "arXiv:2301.00001"}, "title": "T"})
        == "arxiv:2301.00001"
    )
    title_key = paper_key({"title": "  A Study   of Chaperones "})
    assert title_key == paper_key({"title": "a study of chaperones"})
    assert title_key.startswith("title:")


def test_paper_store_upsert_idempotent_and_first_seen(tmp_path) -> None:
    store = PaperStore(tmp_path / "cache")
    paper = {"doi": "10.1/x", "title": "T", "abstract": "old"}
    keys_first = store.upsert_papers([paper], run_id="run-1", round=1)
    keys_second = store.upsert_papers(
        [{"doi": "10.1/x", "title": "T", "abstract": "new"}],
        run_id="run-2",
        round=2,
    )
    assert keys_first == keys_second
    assert len(store.lookup_by_keys(keys_first)) == 1

    # reload from disk: one logical record, first_seen preserved
    reloaded = PaperStore(tmp_path / "cache")
    papers = reloaded.lookup_by_keys(keys_first)
    assert len(papers) == 1
    assert papers[0]["abstract"] == "new"
    reloaded._ensure_loaded()
    record = reloaded._papers[keys_first[0]]
    assert record["first_seen_run"] == "run-1"
    assert record["first_seen_round"] == 1
    assert record["last_seen_run"] == "run-2"


def test_paper_store_query_cache_roundtrip_and_normalisation(tmp_path) -> None:
    store = PaperStore(tmp_path / "cache")
    store.record_query("Hsp70  chaperone mechanism!", ["doi:10.1/x"])

    assert store.lookup_query("hsp70 chaperone mechanism") == ["doi:10.1/x"]
    assert store.lookup_query("  HSP70   chaperone, mechanism? ") == [
        "doi:10.1/x"
    ]
    assert store.lookup_query("something else") == []

    # re-record wins (append-only, last write per hash)
    store.record_query("hsp70 chaperone mechanism", ["doi:10.1/y"])
    reloaded = PaperStore(tmp_path / "cache")
    assert reloaded.lookup_query("hsp70 chaperone mechanism") == ["doi:10.1/y"]


def test_paper_store_missing_files_are_graceful(tmp_path) -> None:
    store = PaperStore(tmp_path / "fresh-cache")
    assert store.lookup_query("anything") == []
    assert store.lookup_by_keys(["doi:10.1/x", "pmid:9"]) == []



# ==================== test_m3 ====================

from hypoforge.modules.m3_grounding.evidence_gams import EvidenceGraphGAMS
from hypoforge.modules.m3_grounding.models import RelationCandidate


def _gams_candidate(
    candidate_id: str,
    source: str,
    target: str,
    relation: str,
    confidence: float,
    evidence_ids: list[str],
    source_paper: str,
    target_paper: str,
    comparability: float = 0.9,
) -> RelationCandidate:
    return RelationCandidate(
        id=candidate_id,
        source=source,
        target=target,
        relation=relation,
        confidence=confidence,
        rationale=f"{source} {relation} {target}",
        evidence_ids=evidence_ids,
        source_paper_ids=[source_paper],
        target_paper_ids=[target_paper],
        retrieval_score=0.8,
        condition_comparability=comparability,
        candidate_origin=["shared_entity", "cross_source"],
    )


def test_evidence_gams_removes_low_quality_edges() -> None:
    candidates = [
        _gams_candidate("R1", "C1", "C4", "supports", 0.92, ["E1"], "P1", "P4"),
        _gams_candidate("R2", "C2", "C4", "supports", 0.88, ["E2"], "P2", "P4"),
        _gams_candidate("R3", "C3", "C4", "contradicts", 0.82, ["E3"], "P3", "P4"),
        _gams_candidate("R4", "C2", "C4", "extends", 0.72, ["E4"], "P2", "P4"),
        _gams_candidate("R5", "C3", "C4", "limits", 0.76, ["E5"], "P3", "P4"),
        _gams_candidate("BAD", "C1", "C4", "contradicts", 0.18, [], "P1", "P4"),
        _gams_candidate("UNSOURCED", "C1", "C2", "supports", 0.25, [], "P1", "P2"),
    ]
    search = EvidenceGraphGAMS(
        minimum_confidence=0.35,
        exploration_weight=0.35,
        random_seed=42,
    )

    selected, trace = search.search(candidates, iterations=128)
    selected_ids = {item.id for item in selected}

    assert set(trace["operator_counts"]) == set(EvidenceGraphGAMS.OPERATORS)
    assert all(count > 0 for count in trace["operator_counts"].values())
    assert trace["states_explored"] > 6
    assert trace["best_reward"] >= trace["direct_accept_all_reward"]
    assert "BAD" not in selected_ids
    assert "UNSOURCED" not in selected_ids
    assert {"R1", "R2", "R3", "R5"}.issubset(selected_ids)
    assert all(
        item.evidence_ids and item.confidence >= 0.35 for item in selected
    )


def test_evidence_gams_reproducible_with_fixed_seed() -> None:
    candidates = [
        _gams_candidate("R1", "C1", "C2", "supports", 0.9, ["E1"], "P1", "P2"),
        _gams_candidate("R2", "C2", "C3", "extends", 0.8, ["E2"], "P2", "P3"),
        _gams_candidate("R3", "C1", "C3", "limits", 0.7, ["E3"], "P1", "P3"),
    ]
    first, first_trace = EvidenceGraphGAMS(random_seed=7).search(
        candidates, iterations=64
    )
    second, second_trace = EvidenceGraphGAMS(random_seed=7).search(
        candidates, iterations=64
    )

    assert [item.id for item in first] == [item.id for item in second]
    assert first_trace["best_reward"] == second_trace["best_reward"]


# --- merged from test_m3_relation_retrieval.py ---

from hypoforge.modules.m3_grounding.models import AtomicClaim, EvidenceRecord, RelationPair
from hypoforge.modules.m3_grounding.relation_retrieval import RelationCandidateRetriever


def _record(
    record_id: str,
    paper_id: str,
    claim: str,
    entities: list[str],
    query: str,
) -> EvidenceRecord:
    return EvidenceRecord(
        id="ER_" + record_id,
        evidence_id=record_id,
        paper_id=paper_id,
        query=query,
        quote=claim,
        normalized_claim=claim,
        summary=claim,
        excerpt=claim,
        relevance_score=8.0,
        retrieval_score=0.8,
        claims=[claim],
        entities=entities,
        context={"population": "human cells", "outcome": "Hsp70 activity"},
    )


def test_retrieval_prioritises_shared_entity_cross_paper_pairs() -> None:
    records = [
        _record("E1", "P1", "NAD+ increases Hsp70 ATPase activity.", ["NAD+", "Hsp70"], "Hsp70 mechanism"),
        _record("E2", "P2", "NAD+ supplementation does not change Hsp70 activity.", ["NAD+", "Hsp70"], "Hsp70 mechanism"),
        _record("E3", "P3", "Kinase A phosphorylates substrate B.", ["Kinase A", "substrate B"], "kinase signalling"),
    ]
    claims = [
        AtomicClaim(id="C1", statement=records[0].claims[0], evidence_ids=["E1"], entities=records[0].entities),
        AtomicClaim(id="C2", statement=records[1].claims[0], evidence_ids=["E2"], entities=records[1].entities),
        AtomicClaim(id="C3", statement=records[2].claims[0], evidence_ids=["E3"], entities=records[2].entities),
    ]

    pairs = RelationCandidateRetriever(max_candidates_per_claim=2).recall(claims, records)
    target = next(pair for pair in pairs if {pair.source, pair.target} == {"C1", "C2"})

    assert "shared_entity" in target.candidate_origin
    assert "cross_source" in target.candidate_origin
    assert "same_query" in target.candidate_origin
    assert {entity.lower() for entity in target.entity_overlap} == {"nad+", "hsp70"}
    assert target.retrieval_score > 0.4


def test_retrieval_avoids_unrelated_low_signal_pairs() -> None:
    records = [
        _record("E1", "P1", "NAD+ increases Hsp70 ATPase activity.", ["NAD+", "Hsp70"], "Hsp70 mechanism"),
        _record("E2", "P2", "Kinase A phosphorylates substrate B.", ["Kinase A", "substrate B"], "kinase signalling"),
    ]
    records[1].context = {"population": "bacteria", "outcome": "substrate phosphorylation"}
    claims = [
        AtomicClaim(id="C1", statement=records[0].claims[0], evidence_ids=["E1"], entities=records[0].entities),
        AtomicClaim(id="C2", statement=records[1].claims[0], evidence_ids=["E2"], entities=records[1].entities),
    ]

    pairs = RelationCandidateRetriever(max_candidates_per_claim=2).recall(claims, records)
    assert pairs == []


def test_retrieval_reserves_capacity_for_cross_source_pairs() -> None:
    records = [
        _record("E1", "P1", "Hsp70 assists folding of client protein A.", ["Hsp70"], "protein folding"),
        _record("E2", "P1", "Hsp70 assists folding of client protein B.", ["Hsp70"], "protein folding"),
        _record("E3", "P1", "Hsp70 assists folding of client protein C.", ["Hsp70"], "protein folding"),
        _record("E4", "P2", "Hsp70 limits aggregation of damaged proteins.", ["Hsp70"], "protein folding"),
    ]
    claims = [
        AtomicClaim(id=f"C{idx}", statement=item.claims[0], evidence_ids=[item.evidence_id], entities=item.entities)
        for idx, item in enumerate(records, start=1)
    ]

    pairs = RelationCandidateRetriever(
        max_candidates_per_claim=4, max_pairs_total=2, cross_source_ratio=0.5,
    ).recall(claims, records)

    assert len(pairs) == 2
    assert any(
        set(pair.source_paper_ids).isdisjoint(pair.target_paper_ids)
        for pair in pairs
    )


# --- merged from test_m3_gams_integration.py ---

import pytest

from hypoforge.modules.m3_grounding.workflow import GroundingWorkflow
from hypoforge.modules.m3_grounding.models import (
    AtomicClaim,
    EvidenceRecord,
    GroundingReport,
    RelationCandidate,
)
from hypoforge.state import EvidenceEdgeRelation, EvidenceEdge, EvidenceGraph, EvidenceNode, EvidenceNodeType


def _evidence(record_id: str, paper_id: str) -> EvidenceRecord:
    return EvidenceRecord(
        id="ER_" + record_id,
        evidence_id=record_id,
        paper_id=paper_id,
        query="test query",
        summary=f"summary {record_id}",
        excerpt=f"excerpt {record_id}",
        relevance_score=8.0,
        retrieval_score=0.8,
        claims=[f"Scientific claim from {record_id} with sufficient length for testing purposes."],
    )


@pytest.mark.asyncio
async def test_direct_and_gams_select_differently(tmp_path) -> None:
    """GAMS should filter out low-confidence edges that direct mode accepts."""
    records = [_evidence("E1", "P1"), _evidence("E2", "P2")]
    claims = [
        AtomicClaim(
            id="C1",
            statement="Claim one has enough scientific content for testing.",
            evidence_ids=["E1"],
        ),
        AtomicClaim(
            id="C2",
            statement="Claim two has enough scientific content for testing.",
            evidence_ids=["E2"],
        ),
    ]
    candidates = [
        RelationCandidate(
            id="GOOD",
            source="C1",
            target="C2",
            relation="supports",
            confidence=0.9,
            rationale="Independent evidence agrees.",
            evidence_ids=["E1", "E2"],
            source_paper_ids=["P1"],
            target_paper_ids=["P2"],
            retrieval_score=0.8,
            condition_comparability=0.9,
            candidate_origin=["shared_entity"],
        ),
        RelationCandidate(
            id="BAD",
            source="C1",
            target="C2",
            relation="contradicts",
            confidence=0.2,
            rationale="Unsupported conflict.",
            evidence_ids=[],
            source_paper_ids=["P1"],
            target_paper_ids=["P2"],
            retrieval_score=0.2,
            condition_comparability=0.2,
            candidate_origin=[],
        ),
    ]
    state = {
        "claims": claims,
        "evidence_records": records,
        "relation_candidates": candidates,
        "report": GroundingReport(),
    }

    direct = GroundingWorkflow(
        mode="rule",
        cache_dir=str(tmp_path / "direct"),
        relation_selection_mode="direct",
        relation_min_confidence=0.0,
    )
    gams = GroundingWorkflow(
        mode="rule",
        cache_dir=str(tmp_path / "gams"),
        relation_selection_mode="evidence_gams",
        relation_min_confidence=0.35,
        evidence_gams_iterations=64,
    )

    import asyncio
    direct_output = await direct._select_relations(state)  # type: ignore[arg-type]
    gams_output = await gams._select_relations(state)  # type: ignore[arg-type]

    direct_semantic_ids = {
        r.id for r in direct_output["relations"] if r.id in {"GOOD", "BAD"}
    }
    gams_semantic_ids = {
        r.id for r in gams_output["relations"] if r.id in {"GOOD", "BAD"}
    }

    # Direct mode accepts both (BAD has confidence 0.2 which is >= 0.0)
    assert direct_semantic_ids == {"GOOD", "BAD"}
    # GAMS should filter out BAD (low confidence, no evidence, low retrieval)
    assert gams_semantic_ids == {"GOOD"}
    assert gams_output["report"].relation_selection_mode == "evidence_gams"


def test_merge_grounding_creates_claim_nodes():
    """Verify that _merge_grounding adds CLAIM and EVIDENCE nodes to graph."""
    from hypoforge.modules.m3_evidence_graph import M3EvidenceGraph

    records = [
        EvidenceRecord(
            id="ER_E1",
            evidence_id="E1",
            paper_id="P1",
            query="q",
            quote="Test quote.",
            normalized_claim="Test claim.",
            summary="Summary of test.",
            excerpt="Test quote.",
            relevance_score=7.0,
        ),
    ]
    claims = [
        AtomicClaim(
            id="CLM_test",
            statement="A test scientific claim with sufficient detail for the evidence graph.",
            evidence_ids=["E1"],
            paper_ids=["P1"],
            entities=["Protein A", "pathway B"],
            confidence=0.8,
        ),
    ]

    graph = EvidenceGraph()
    result = M3EvidenceGraph._merge_grounding(
        graph, records, claims, [], GroundingReport()
    )

    # Should have 1 EVIDENCE node and 1 CLAIM node
    evidence_nodes = [n for n in result.nodes if n.type == EvidenceNodeType.EVIDENCE]
    claim_nodes = [n for n in result.nodes if n.type == EvidenceNodeType.CLAIM]
    assert len(evidence_nodes) == 1
    assert len(claim_nodes) == 1
    assert evidence_nodes[0].metadata["evidence_id"] == "E1"
    assert claim_nodes[0].metadata["claim_id"] == "CLM_test"
    assert claim_nodes[0].metadata["confidence"] == 0.8

    # Should have 1 provenance edge (evidence → claim)
    prov_edges = [
        e for e in result.edges
        if e.relation == EvidenceEdgeRelation.SUPPORTS
    ]
    assert len(prov_edges) == 1


# --- merged from test_m3_grounding.py ---

import pytest

from hypoforge.modules.m3_grounding.workflow import GroundingWorkflow
from hypoforge.modules.m3_grounding.models import (
    AtomicClaim,
    EvidenceRecord,
    GroundingReport,
    RelationCandidate,
)
from hypoforge.modules.m3_evidence_graph import (
    M3EvidenceGraph,
    add_entity_synonym,
    evidence_graph_to_mermaid,
    normalize_entity,
)
from hypoforge.state import (
    EvidenceEdge,
    EvidenceEdgeRelation,
    EvidenceGraph,
    EvidenceNode,
    EvidenceNodeType,
    M2EvidenceExport,
    M2KnowledgeExport,
    M2KnowledgeRun,
    M2PaperExport,
    PipelineState,
    ProblemCard,
)


def _make_m2_export(
    evidence_items: list[tuple[str, str, str, str]],
) -> M2KnowledgeExport:
    """Build a minimal M2KnowledgeExport with given evidence items.

    Each tuple: (evidence_id, paper_id, quote, normalized_claim)
    """
    paper_ids = sorted({pid for (_, pid, _, _) in evidence_items})
    return M2KnowledgeExport(
        runs=[
            M2KnowledgeRun(
                sub_question="test question",
                papers=[
                    M2PaperExport(paper_id=pid, title=f"Paper {pid}")
                    for pid in paper_ids
                ],
                evidence=[
                    M2EvidenceExport(
                        evidence_id=eid,
                        paper_id=pid,
                        chunk_id=f"CHK_{eid}",
                        quote=quote,
                        normalized_claim=nclaim,
                        relevance_score=0.9,
                        citable=True,
                    )
                    for (eid, pid, quote, nclaim) in evidence_items
                ],
            )
        ]
    )


def test_collect_m2_evidence_filters_citable_only():
    """Evidence items with citable=False should be excluded."""
    from hypoforge.state import M2PaperExport as Paper
    export = M2KnowledgeExport(
        runs=[
            M2KnowledgeRun(
                sub_question="q",
                papers=[
                    Paper(paper_id="P1", title="Test Paper"),
                ],
                evidence=[
                    M2EvidenceExport(
                        evidence_id="E1", paper_id="P1",
                        chunk_id="C1", quote="good", normalized_claim="good",
                        relevance_score=0.8, citable=True,
                    ),
                    M2EvidenceExport(
                        evidence_id="E2", paper_id="P1",
                        chunk_id="C2", quote="bad", normalized_claim="bad",
                        relevance_score=0.5, citable=False,
                    ),
                ],
            )
        ]
    )
    state = PipelineState(
        input_question="test",
        m2_knowledge_export=export,
    )
    wf = GroundingWorkflow(mode="rule")
    gs = {"pipeline_state": state, "report": GroundingReport()}

    import asyncio
    result = asyncio.run(wf._collect_m2_evidence(gs))  # type: ignore[arg-type]

    items = result["m2_evidence_items"]
    assert len(items) == 1
    assert items[0].evidence_id == "E1"


def test_collect_m2_evidence_warns_on_empty_export():
    """When m2_knowledge_export is None, should warn and return empty."""
    state = PipelineState(input_question="test")
    wf = GroundingWorkflow(mode="rule")
    gs = {"pipeline_state": state, "report": GroundingReport()}

    import asyncio
    result = asyncio.run(wf._collect_m2_evidence(gs))  # type: ignore[arg-type]

    assert result["m2_evidence_items"] == []
    assert any("empty" in w.lower() or "none" in w.lower() for w in result["report"].warnings)


def test_plan_queries_harvests_m2_knowledge_gaps():
    """Gaps and conflicts from M2KnowledgeExport should become queries."""
    from hypoforge.state import KnowledgeEntry, KnowledgeEntryType, M2PaperExport as Paper
    export = M2KnowledgeExport(
        runs=[
            M2KnowledgeRun(
                sub_question="q",
                papers=[
                    Paper(paper_id="P1", title="Test Paper"),
                ],
                knowledge_entries=[
                    KnowledgeEntry(
                        id="GAP1",
                        type=KnowledgeEntryType.KNOWLEDGE_GAP,
                        content="Unknown interaction between X and Y remains to be explored.",
                        source_paper_id="P1",
                        evidence_ids=["E1"],
                    ),
                ],
                evidence=[
                    M2EvidenceExport(
                        evidence_id="E1", paper_id="P1",
                        chunk_id="C1", quote="X interacts with Y is unknown.",
                        normalized_claim="X-Y interaction unknown",
                        relevance_score=0.8,
                    ),
                ],
            )
        ]
    )
    state = PipelineState(
        input_question="What is the role of X?",
        m2_knowledge_export=export,
        problem_card=ProblemCard(
            original_question="What is the role of X?",
            key_entities=["X", "Y"],
        ),
    )
    wf = GroundingWorkflow(mode="rule")
    gs = {"pipeline_state": state, "report": GroundingReport()}

    import asyncio
    result = asyncio.run(wf._plan_queries(gs))  # type: ignore[arg-type]

    queries = result["queries"]
    # Should include the gap content
    gap_queries = [q for q in queries if "Unknown interaction" in q]
    assert len(gap_queries) >= 1


def test_evidence_record_has_evidence_id():
    """EvidenceRecord should bridge to M2EvidenceExport via evidence_id."""
    record = EvidenceRecord(
        id="ER_test",
        evidence_id="E1",
        paper_id="P1",
        query="test",
        quote="Hsp70 binds ATP.",
        normalized_claim="Hsp70 has ATP binding activity.",
        relevance_score=7.5,
    )
    assert record.evidence_id == "E1"
    assert record.paper_id == "P1"


def test_relation_candidate_to_relation():
    """RelationCandidate.to_relation() should produce a canonical EvidenceRelation."""
    candidate = RelationCandidate(
        id="REL_1",
        source="C1",
        target="C2",
        relation="supports",
        confidence=0.85,
        rationale="Both point to the same mechanism.",
        evidence_ids=["E1", "E2"],
        source_paper_ids=["P1"],
        target_paper_ids=["P2"],
        retrieval_score=0.75,
        condition_comparability=0.9,
        candidate_origin=["shared_entity"],
    )
    rel = candidate.to_relation()
    assert rel.id == "REL_1"
    assert rel.source == "C1"
    assert rel.target == "C2"
    assert rel.relation == "supports"
    assert rel.confidence == 0.85
    assert rel.evidence_ids == ["E1", "E2"]


def test_unrelated_cannot_become_relation():
    """Candidates marked 'unrelated' should raise when converting to relation."""
    candidate = RelationCandidate(
        id="REL_BAD",
        source="C1",
        target="C2",
        relation="unrelated",
        confidence=0.1,
        rationale="No connection.",
    )
    with pytest.raises(ValueError, match="unrelated"):
        candidate.to_relation()


# ============================================================================
# Entity normalisation tests
# ============================================================================


def test_normalize_entity_maps_synonyms():
    """Known variants should map to the same canonical form."""
    assert normalize_entity("Hsp70") == normalize_entity("HSP70")
    assert normalize_entity("HSPA1A") == normalize_entity("hsp70")
    assert normalize_entity("heat shock protein 70") == "hsp70"


def test_normalize_entity_falls_back_to_cleaned():
    """Unknown names should be lowercased and stripped but not lost."""
    result = normalize_entity("  Unknown-Protein-X  ")
    assert result == "unknown-protein-x"


def test_normalize_entity_strips_punctuation():
    """Trailing commas, periods should be removed."""
    assert normalize_entity("Hsp70,") == "hsp70"
    assert normalize_entity("(p53)") == "p53"


def test_add_entity_synonym_runtime():
    """Runtime-registered synonyms should be picked up immediately."""
    add_entity_synonym("MyNewProtein", "MNP")
    assert normalize_entity("MNP") == "mynewprotein"
    assert normalize_entity("MyNewProtein") == "mynewprotein"


# ============================================================================
# Mermaid export tests
# ============================================================================


def test_evidence_graph_to_mermaid_produces_valid_structure():
    """Mermaid output should have nodes, edges, and title."""
    graph = EvidenceGraph(
        nodes=[
            EvidenceNode(id="N1", type=EvidenceNodeType.CLAIM, label="Claim A"),
            EvidenceNode(id="N2", type=EvidenceNodeType.EVIDENCE, label="Evidence B"),
            EvidenceNode(id="SRC_P1", type=EvidenceNodeType.SOURCE, label="Paper 1"),
        ],
        edges=[
            EvidenceEdge(
                source="N2", target="N1",
                relation=EvidenceEdgeRelation.SUPPORTS,
                confidence=0.88, rationale="Strong evidence.",
            ),
            EvidenceEdge(
                source="SRC_P1", target="N2",
                relation=EvidenceEdgeRelation.INVOLVES,
            ),
        ],
    )
    mermaid = evidence_graph_to_mermaid(graph)
    assert "flowchart LR" in mermaid
    assert "N1" in mermaid
    assert "N2" in mermaid
    assert "supports" in mermaid
    assert "involves" in mermaid
    assert "c=0.88" in mermaid


def test_evidence_graph_to_mermaid_escapes_quotes():
    """Double-quotes in labels should be turned into single quotes."""
    graph = EvidenceGraph(
        nodes=[
            EvidenceNode(
                id="N1", type=EvidenceNodeType.CLAIM,
                label='The "key" finding',
            ),
        ],
    )
    mermaid = evidence_graph_to_mermaid(graph)
    # The label's double-quotes become single quotes in the mermaid output
    assert "\"key\"" not in mermaid
    assert "'key'" in mermaid


# ============================================================================
# Incremental update tests
# ============================================================================


def test_find_new_entries_filters_existing():
    """_find_new_entries should return only entries not already in the graph."""
    from hypoforge.state import KnowledgeEntry, KnowledgeEntryType, ConfidenceLevel

    existing = EvidenceGraph(
        nodes=[
            EvidenceNode(
                id="N_K1", type=EvidenceNodeType.CLAIM,
                label="Entry 1", metadata={"entry_type": "established_fact"},
            ),
            EvidenceNode(
                id="N_K2", type=EvidenceNodeType.EVIDENCE,
                label="Entry 2", metadata={"entry_type": "established_fact"},
            ),
        ],
    )
    entries = [
        KnowledgeEntry(id="K1", type=KnowledgeEntryType.ESTABLISHED_FACT, content="old"),
        KnowledgeEntry(id="K2", type=KnowledgeEntryType.ESTABLISHED_FACT, content="old"),
        KnowledgeEntry(id="K3", type=KnowledgeEntryType.ESTABLISHED_FACT, content="new"),
    ]
    new_entries = M3EvidenceGraph._find_new_entries(entries, existing)
    assert len(new_entries) == 1
    assert new_entries[0].id == "K3"


def test_merge_graphs_deduplicates():
    """Merging two graphs should not create duplicate nodes or edges."""
    base = EvidenceGraph(
        nodes=[
            EvidenceNode(id="N1", type=EvidenceNodeType.CLAIM, label="A"),
        ],
        edges=[
            EvidenceEdge(source="SRC_P1", target="N1", relation=EvidenceEdgeRelation.INVOLVES),
        ],
        established_facts=["K1"],
    )
    additions = EvidenceGraph(
        nodes=[
            EvidenceNode(id="N1", type=EvidenceNodeType.CLAIM, label="A"),  # duplicate
            EvidenceNode(id="N2", type=EvidenceNodeType.CLAIM, label="B"),  # new
        ],
        edges=[
            EvidenceEdge(source="SRC_P1", target="N1", relation=EvidenceEdgeRelation.INVOLVES),  # dup
            EvidenceEdge(source="N2", target="N1", relation=EvidenceEdgeRelation.EXTENDS),  # new
        ],
        established_facts=["K1", "K2"],
    )
    merged = M3EvidenceGraph._merge_graphs(base, additions)
    # 2 unique nodes (N1, N2)
    assert len(merged.nodes) == 2
    # 2 unique edges
    assert len(merged.edges) == 2
    # 2 unique facts (K1, K2), no duplicates
    assert len(merged.established_facts) == 2
    assert set(merged.established_facts) == {"K1", "K2"}


# ==================== test_memory_merge_and_m3 ====================

import json

import pytest

from hypoforge.memory import (
    KnowledgeGraphManager,
    load_latest_graph_round,
    save_graph_round_snapshot,
    stable_entry_id,
)
from hypoforge.modules.m3_evidence_graph import M3EvidenceGraph
from hypoforge.state import (
    EvidenceEdge,
    EvidenceEdgeRelation,
    EvidenceGap,
    EvidenceGraph,
    EvidenceNode,
    EvidenceNodeType,
    KnowledgeEntry,
    KnowledgeEntryType,
    LiteratureResult,
    PipelineState,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _entry(entry_id: str, content: str, paper_id: str = "PMID:1") -> KnowledgeEntry:
    return KnowledgeEntry(
        id=entry_id,
        type=KnowledgeEntryType.ESTABLISHED_FACT,
        content=content,
        source_paper_id=paper_id,
        source_paper_title=f"Paper {paper_id}",
        entities=["hsp70"],
    )


def _graph_with(node_id: str, label: str = "node") -> EvidenceGraph:
    return EvidenceGraph(
        nodes=[EvidenceNode(id=node_id, type=EvidenceNodeType.CLAIM, label=label)],
        edges=[],
        established_facts=[],
        conflicts=[],
        knowledge_gaps=[],
    )


# ---------------------------------------------------------------------------
# Merge API
# ---------------------------------------------------------------------------


def test_merge_entities_by_normalised_name_and_union_observations(tmp_path):
    mgr = KnowledgeGraphManager(cache_dir=tmp_path / "kg")

    eg1 = EvidenceGraph(nodes=[
        EvidenceNode(id="ENT_HSP70", type=EvidenceNodeType.ENTITY, label="HSP70 node"),
    ])
    eg2 = EvidenceGraph(nodes=[
        # Same entity after normalisation (case / whitespace variants)
        EvidenceNode(id=" ent_hsp70 ", type=EvidenceNodeType.ENTITY, label="alias obs"),
        EvidenceNode(id="ENT_P53", type=EvidenceNodeType.ENTITY, label="p53 node"),
    ])

    mgr.merge_from_evidence_graph(eg1)
    merged = mgr.merge_from_evidence_graph(eg2)

    names = {e.name for e in merged.entities}
    assert names == {"ENT_HSP70", "ENT_P53"}
    hsp70 = next(e for e in merged.entities if e.name == "ENT_HSP70")
    # Observations are unioned (deduplicated)
    assert "HSP70 node" in hsp70.observations
    assert "alias obs" in hsp70.observations

    # Round-trip through disk matches the returned object
    on_disk = mgr.read_graph()
    assert {e.name for e in on_disk.entities} == names


def test_merge_relations_keep_higher_confidence_and_merge_rationale(tmp_path):
    mgr = KnowledgeGraphManager(cache_dir=tmp_path / "kg")

    def eg_with_edge(confidence, rationale):
        return EvidenceGraph(
            nodes=[
                EvidenceNode(id="N_a", type=EvidenceNodeType.CLAIM, label="a"),
                EvidenceNode(id="N_b", type=EvidenceNodeType.CLAIM, label="b"),
            ],
            edges=[EvidenceEdge(
                source="N_a", target="N_b",
                relation=EvidenceEdgeRelation.SUPPORTS,
                confidence=confidence, rationale=rationale,
            )],
        )

    mgr.merge_from_evidence_graph(eg_with_edge(0.6, "first rationale"))
    merged = mgr.merge_from_evidence_graph(eg_with_edge(0.9, "second rationale"))

    supports = [r for r in merged.relations if r.relation_type == "supports"]
    assert len(supports) == 1  # dedup by (from, relation_type, to)
    assert supports[0].confidence == pytest.approx(0.9)
    assert "first rationale" in supports[0].rationale
    assert "second rationale" in supports[0].rationale

    # Merging a lower-confidence duplicate afterwards keeps the higher one
    merged2 = mgr.merge_from_evidence_graph(eg_with_edge(0.2, "first rationale"))
    supports2 = [r for r in merged2.relations if r.relation_type == "supports"]
    assert len(supports2) == 1
    assert supports2[0].confidence == pytest.approx(0.9)


def test_overwrite_semantics_not_broken(tmp_path):
    """save_from_evidence_graph / clear_graph / delete_entities keep
    overwrite semantics (the merge API must not interfere)."""
    mgr = KnowledgeGraphManager(cache_dir=tmp_path / "kg")

    big = EvidenceGraph(nodes=[
        EvidenceNode(id="N_keep_old", type=EvidenceNodeType.CLAIM, label="old"),
        EvidenceNode(id="N_gone", type=EvidenceNodeType.CLAIM, label="gone"),
    ])
    small = EvidenceGraph(nodes=[
        EvidenceNode(id="N_keep_new", type=EvidenceNodeType.CLAIM, label="new"),
    ])

    # save_from_evidence_graph overwrites completely
    mgr.save_from_evidence_graph(big)
    mgr.save_from_evidence_graph(small)
    assert {e.name for e in mgr.read_graph().entities} == {"N_keep_new"}

    # clear_graph empties the store
    mgr.clear_graph()
    assert mgr.read_graph().is_empty

    # merge after clear must NOT resurrect cleared data
    mgr.merge_from_evidence_graph(small)
    assert {e.name for e in mgr.read_graph().entities} == {"N_keep_new"}

    # delete_entities still removes entities
    mgr.delete_entities(["N_keep_new"])
    assert mgr.read_graph().is_empty


# ---------------------------------------------------------------------------
# Round snapshots
# ---------------------------------------------------------------------------


def test_round_snapshot_write_and_load_latest(tmp_path):
    out = tmp_path / "run-output"

    eg0 = _graph_with("N_round0", "round zero")
    eg2 = _graph_with("N_round2", "round two")

    assert save_graph_round_snapshot(eg0, out, 0) is not None
    assert save_graph_round_snapshot(eg2, out, 2) is not None
    # noise file must be ignored by the loader
    (out / "graph-round-notes.json").write_text("{}", encoding="utf-8")

    latest = load_latest_graph_round(out)
    assert latest is not None
    assert [n["id"] for n in latest["nodes"]] == ["N_round2"]

    # Same round overwrites its own file
    eg2b = _graph_with("N_round2_updated", "round two v2")
    save_graph_round_snapshot(eg2b, out, 2)
    latest2 = load_latest_graph_round(out)
    assert [n["id"] for n in latest2["nodes"]] == ["N_round2_updated"]

    # Snapshot is a lossless EvidenceGraph dump (re-parseable)
    parsed = EvidenceGraph(**latest2)
    assert parsed.nodes[0].id == "N_round2_updated"


def test_load_latest_graph_round_missing(tmp_path):
    assert load_latest_graph_round(tmp_path / "does-not-exist") is None
    empty = tmp_path / "empty"
    empty.mkdir()
    assert load_latest_graph_round(empty) is None


# ---------------------------------------------------------------------------
# M3 — empty entries must not destroy the graph
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_m3_empty_entries_preserves_existing_graph():
    existing = _graph_with("N_old", "old claim")
    state = PipelineState(
        input_question="q",
        literature_results=[],
        evidence_graph=existing,
    )
    module = M3EvidenceGraph(mode="rule")
    result = await module(state)
    graph = result["evidence_graph"]
    assert [n.id for n in graph.nodes] == ["N_old"]


@pytest.mark.asyncio
async def test_m3_empty_entries_and_no_graph_returns_empty():
    state = PipelineState(input_question="q", literature_results=[])
    module = M3EvidenceGraph(mode="rule")
    with pytest.raises(RuntimeError, match="no usable knowledge entries"):
        await module(state)


@pytest.mark.asyncio
async def test_m3_empty_entries_loads_persisted_graph_from_memory(tmp_path):
    """No entries + no in-state graph → restore from the JSONL store."""
    cache_dir = tmp_path / "kg-cache"

    # Round 1: build + persist a graph
    entries = [_entry("KE_persist", "Persisted finding about AMPK.")]
    first = await M3EvidenceGraph(mode="rule")(PipelineState(
        input_question="q",
        literature_results=[LiteratureResult(
            sub_question="sq", papers_retrieved=1, knowledge_entries=entries,
        )],
        memory_cache_dir=str(cache_dir),
    ))
    assert any(n.id == "N_KE_persist" for n in first["evidence_graph"].nodes)

    # Round 2: fresh state with NO entries and NO graph → must recover
    # the persisted graph instead of returning an empty one.
    result = await M3EvidenceGraph(mode="rule")(PipelineState(
        input_question="q",
        literature_results=[],
        memory_cache_dir=str(cache_dir),
    ))
    graph = result["evidence_graph"]
    assert len(graph.nodes) > 0
    names = {n.label for n in graph.nodes} | {n.id for n in graph.nodes}
    assert any("KE_persist" in str(n) for n in names)


@pytest.mark.asyncio
async def test_m3_empty_entries_prefers_lossless_round_snapshot(tmp_path):
    """When both a snapshot and the JSONL store exist, the lossless
    round snapshot wins."""
    out_dir = tmp_path / "run-out"
    original = _graph_with("N_snapshot_only", "snapshot claim")
    assert save_graph_round_snapshot(original, out_dir, 3) is not None

    result = await M3EvidenceGraph(mode="rule", output_dir=str(out_dir))(
        PipelineState(input_question="q", literature_results=[])
    )
    assert [n.id for n in result["evidence_graph"].nodes] == ["N_snapshot_only"]


# ---------------------------------------------------------------------------
# M3 — no-new-entry rounds still persist + snapshot
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_m3_no_new_entries_still_persists_and_snapshots(tmp_path):
    entries = [_entry("KE_a", "Hsp70 assists protein folding.")]
    lr = LiteratureResult(
        sub_question="How does Hsp70 work?",
        papers_retrieved=1,
        knowledge_entries=entries,
    )

    module = M3EvidenceGraph(mode="rule")
    first = await module(PipelineState(
        input_question="q", literature_results=[lr],
    ))
    graph = first["evidence_graph"]
    assert any(n.id == "N_KE_a" for n in graph.nodes)

    cache_dir = tmp_path / "kg-cache"
    out_dir = tmp_path / "run-out"
    state2 = PipelineState(
        input_question="q",
        literature_results=[lr],
        evidence_graph=graph,
        memory_cache_dir=str(cache_dir),
        iteration_count=1,
    )
    module2 = M3EvidenceGraph(mode="rule", output_dir=str(out_dir))
    second = await module2(state2)

    # Graph unchanged (no new entries → no re-construction)
    assert {n.id for n in second["evidence_graph"].nodes} == {
        n.id for n in graph.nodes
    }
    # Persistent JSONL store written via merge
    assert (cache_dir / "memory-evidence_graph.jsonl").exists()
    stored = KnowledgeGraphManager(cache_dir=cache_dir).read_graph()
    assert any(e.name == "N_KE_a" for e in stored.entities)
    # Lossless round snapshot written to the run output dir
    snapshot = out_dir / "graph-round-1.json"
    assert snapshot.exists()
    payload = json.loads(snapshot.read_text(encoding="utf-8"))
    assert any(n["id"] == "N_KE_a" for n in payload["nodes"])


@pytest.mark.asyncio
async def test_m3_snapshot_round_prefers_search_round(tmp_path):
    """Supplement rounds number snapshots by search_round."""
    entries = [_entry("KE_c", "Supplement-round evidence about ROS.")]
    out_dir = tmp_path / "run-out"
    module = M3EvidenceGraph(mode="rule", output_dir=str(out_dir))
    await module(PipelineState(
        input_question="q",
        literature_results=[LiteratureResult(
            sub_question="sq", papers_retrieved=1, knowledge_entries=entries,
        )],
        search_round=2,
        iteration_count=1,
    ))
    assert (out_dir / "graph-round-2.json").exists()


@pytest.mark.asyncio
async def test_m3_snapshot_falls_back_to_memory_cache_dir(tmp_path):
    entries = [_entry("KE_b", "NAD+ boosts mitochondrial function.")]
    lr = LiteratureResult(
        sub_question="What does NAD+ do?",
        papers_retrieved=1,
        knowledge_entries=entries,
    )
    cache_dir = tmp_path / "cache"
    module = M3EvidenceGraph(mode="rule")  # no output_dir
    await module(PipelineState(
        input_question="q",
        literature_results=[lr],
        memory_cache_dir=str(cache_dir),
        iteration_count=0,
    ))
    assert (cache_dir / "graph-round-0.json").exists()


# ---------------------------------------------------------------------------
# M3 — incremental append across rounds (dedup via graph, not instance state)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_m3_incremental_appends_new_entries_across_instances(tmp_path):
    old_entry = _entry("KE_old", "Hsp70 is a heat shock protein.")
    new_entry = _entry("KE_new", "Hsp70 inhibition sensitises tumours.", "PMID:2")

    module1 = M3EvidenceGraph(mode="rule")
    first = await module1(PipelineState(
        input_question="q",
        literature_results=[LiteratureResult(
            sub_question="sq", papers_retrieved=1, knowledge_entries=[old_entry],
        )],
    ))
    graph = first["evidence_graph"]

    # A *fresh* module instance (no shared instance state) must still dedup
    module2 = M3EvidenceGraph(mode="rule")
    second = await module2(PipelineState(
        input_question="q",
        literature_results=[LiteratureResult(
            sub_question="sq", papers_retrieved=2,
            knowledge_entries=[old_entry, new_entry],
        )],
        evidence_graph=graph,
    ))
    ids = {n.id for n in second["evidence_graph"].nodes}
    assert "N_KE_old" in ids
    assert "N_KE_new" in ids
    # old entry not duplicated
    assert sum(1 for i in ids if i == "N_KE_old") == 1


# ---------------------------------------------------------------------------
# M3 — pending_grounding gap confirmation
# ---------------------------------------------------------------------------


def _pending_gap(target: str) -> EvidenceGap:
    return EvidenceGap(
        description=f"Missing evidence for: {target}",
        target_sub_question=target,
        status="pending_grounding",
    )


@pytest.mark.asyncio
async def test_gap_with_gain_is_closed():
    old_entry = _entry("KE_old2", "Baseline finding about p53.")
    new_entry = _entry("KE_new2", "New p53 mechanism found.", "PMID:9")

    module = M3EvidenceGraph(mode="rule")
    first = await module(PipelineState(
        input_question="q",
        literature_results=[LiteratureResult(
            sub_question="How does p53 work?",
            papers_retrieved=1,
            knowledge_entries=[old_entry],
        )],
    ))
    graph = first["evidence_graph"]

    gap = _pending_gap("How does p53 work?")
    state = PipelineState(
        input_question="q",
        literature_results=[
            LiteratureResult(
                sub_question="How does p53 work?",
                papers_retrieved=1,
                knowledge_entries=[old_entry],
            ),
            LiteratureResult(
                sub_question="How does p53 work?",
                papers_retrieved=1,
                knowledge_entries=[new_entry],
            ),
        ],
        evidence_graph=graph,
        evidence_gaps=[gap],
    )
    result = await module(state)

    assert "evidence_gaps" in result
    updated = result["evidence_gaps"]
    assert len(updated) == 1
    assert updated[0].status == "closed"
    # state object untouched (copy-on-write patch)
    assert state.evidence_gaps[0].status == "pending_grounding"
    # CONTRACT (P2): per-gap gain counters travel via metrics["m3_gap_gain"]
    # (PipelineState has no dedicated gap_gain field; pipeline.py rejects
    # unknown patch keys).
    assert result["metrics"]["m3_gap_gain"] == {gap.gap_id: 1}


@pytest.mark.asyncio
async def test_gap_without_gain_becomes_unimprovable_at_limit():
    entry = _entry("KE_same", "No change in evidence base.")
    module = M3EvidenceGraph(mode="rule")
    first = await module(PipelineState(
        input_question="q",
        literature_results=[LiteratureResult(
            sub_question="unrelated sub-question",
            papers_retrieved=1,
            knowledge_entries=[entry],
        )],
    ))
    graph = first["evidence_graph"]

    gap = _pending_gap("target that matches nothing new")
    state = PipelineState(
        input_question="q",
        literature_results=[LiteratureResult(
            sub_question="unrelated sub-question",
            papers_retrieved=1,
            knowledge_entries=[entry],
        )],
        evidence_graph=graph,
        evidence_gaps=[gap],
    )
    # default gap_no_improvement_limit == 1 → first no-gain round closes it
    result = await module(state)
    updated = result["evidence_gaps"]
    assert updated[0].attempts == 1
    assert updated[0].status == "unimprovable"


@pytest.mark.asyncio
async def test_gap_without_gain_increments_attempts_below_limit():
    entry = _entry("KE_same3", "Still no relevant new papers.")
    module = M3EvidenceGraph(mode="rule", gap_no_improvement_limit=3)
    first = await module(PipelineState(
        input_question="q",
        literature_results=[LiteratureResult(
            sub_question="something else",
            papers_retrieved=1,
            knowledge_entries=[entry],
        )],
    ))
    graph = first["evidence_graph"]

    state = PipelineState(
        input_question="q",
        literature_results=[LiteratureResult(
            sub_question="something else",
            papers_retrieved=1,
            knowledge_entries=[entry],
        )],
        evidence_graph=graph,
        evidence_gaps=[_pending_gap("still unmatched target")],
    )
    result = await module(state)
    updated = result["evidence_gaps"]
    assert updated[0].attempts == 1
    assert updated[0].status == "pending_grounding"


@pytest.mark.asyncio
async def test_no_pending_gaps_means_no_gap_patch():
    entry = _entry("KE_x", "Some content about mTOR signalling.")
    module = M3EvidenceGraph(mode="rule")
    result = await module(PipelineState(
        input_question="q",
        literature_results=[LiteratureResult(
            sub_question="sq", papers_retrieved=1, knowledge_entries=[entry],
        )],
        evidence_gaps=[EvidenceGap(
            description="an open gap", target_sub_question="sq", status="open",
        )],
    ))
    assert "evidence_gaps" not in result


# ---------------------------------------------------------------------------
# Stable KnowledgeEntry ids
# ---------------------------------------------------------------------------


def test_stable_entry_id_is_content_based():
    a = stable_entry_id("Hsp70   folds proteins.", "PMID:1")
    b = stable_entry_id("hsp70 folds proteins.", "PMID:1")  # case/whitespace
    c = stable_entry_id("Hsp70 folds proteins.", "PMID:2")  # other paper
    assert a == b
    assert a != c
    assert a.startswith("KE_")
    assert len(a) == len("KE_") + 12


def test_stable_entry_id_matches_canonical_memory_helper():
    """M2's local helper and the canonical memory-layer helper must agree
    (same evidence → same id regardless of producer)."""
    assert stable_entry_id("  Hsp70 folds\tproteins. ", "PMID:7") == \
        stable_entry_id("Hsp70 folds proteins.", "PMID:7")


@pytest.mark.asyncio
async def test_m3_blank_entry_ids_get_stable_content_based_node_ids():
    """Entries with empty ids must get stable content-based node ids, so
    two separate runs (fresh module instances) produce identical graphs
    and incremental dedup still works."""
    def blank_entry() -> KnowledgeEntry:
        return KnowledgeEntry(
            id="",
            type=KnowledgeEntryType.ESTABLISHED_FACT,
            content="Hsp90 stabilises client kinases.",
            source_paper_id="PMID:42",
            entities=["hsp90"],
        )

    first = await M3EvidenceGraph(mode="rule")(PipelineState(
        input_question="q",
        literature_results=[LiteratureResult(
            sub_question="sq", papers_retrieved=1,
            knowledge_entries=[blank_entry()],
        )],
    ))
    second = await M3EvidenceGraph(mode="rule")(PipelineState(
        input_question="q",
        literature_results=[LiteratureResult(
            sub_question="sq", papers_retrieved=1,
            knowledge_entries=[blank_entry()],
        )],
    ))
    ids_first = sorted(n.id for n in first["evidence_graph"].nodes)
    ids_second = sorted(n.id for n in second["evidence_graph"].nodes)
    assert ids_first == ids_second  # deterministic across runs

    expected = "N_" + stable_entry_id(
        "Hsp90 stabilises client kinases.", "PMID:42"
    )
    assert expected in ids_first

    # Incremental dedup: feeding the same blank-id entry back together with
    # the existing graph adds nothing.
    third = await M3EvidenceGraph(mode="rule")(PipelineState(
        input_question="q",
        literature_results=[LiteratureResult(
            sub_question="sq", papers_retrieved=1,
            knowledge_entries=[blank_entry()],
        )],
        evidence_graph=first["evidence_graph"],
    ))
    assert len(third["evidence_graph"].nodes) == len(first["evidence_graph"].nodes)


@pytest.mark.asyncio
async def test_gap_gain_present_for_all_gaps_including_zero():
    """metrics['m3_gap_gain'] covers every gap, zeros included."""
    entry = _entry("KE_g1", "Evidence only for the first gap.")
    matched = _pending_gap("matched sub-question")
    unmatched = _pending_gap("never matched sub-question")

    module = M3EvidenceGraph(mode="rule", gap_no_improvement_limit=3)
    first = await module(PipelineState(
        input_question="q",
        literature_results=[LiteratureResult(
            sub_question="matched sub-question",
            papers_retrieved=1,
            knowledge_entries=[entry],
        )],
    ))
    result = await module(PipelineState(
        input_question="q",
        literature_results=[LiteratureResult(
            sub_question="matched sub-question",
            papers_retrieved=1,
            knowledge_entries=[entry],
        )],
        evidence_graph=first["evidence_graph"],
        evidence_gaps=[matched, unmatched],
    ))
    gain = result["metrics"]["m3_gap_gain"]
    assert set(gain) == {matched.gap_id, unmatched.gap_id}
    # entry already in graph → not new → zero gain for both gaps
    assert gain[matched.gap_id] == 0
    assert gain[unmatched.gap_id] == 0


# ==================== test_m4_revision_safety ====================

import asyncio
from typing import Any

import pytest

from hypoforge.modules.m4_hypothesis_generation import M4HypothesisGeneration
from hypoforge.modules.m5_research_plan import M5ResearchPlan
from hypoforge.observability import RunEventRecorder, bind_recorder
from hypoforge.prompts.m5_prompts import M5_USER_TEMPLATE
from hypoforge.state import (
    FollowupRequest,
    HypothesisCard,
    PipelineState,
    ProblemCard,
    ReviewResult,
    ReviewerDimension,
)


def _card(hypothesis_id: str, statement: str) -> HypothesisCard:
    return HypothesisCard(
        hypothesis_id=hypothesis_id,
        statement=statement,
        mechanism="PBK activation -> bypass signaling -> persister survival",
        observable_predictions=["PBK inhibition will reduce persister survival."],
        falsification_conditions=["PBK inhibition has no effect on survival."],
    )


@pytest.mark.parametrize(
    "statement",
    [
        "Consider adding orthogonal validation of PBK dependency.",
        "To strengthen the hypothesis, include preliminary evidence.",
        "Future work should test compensatory pathways.",
        "We hypothesize that PBK regulates persister survival.",
        "The hypothesis should include a rescue experiment.",
        "A stronger hypothesis would acknowledge alternative pathways.",
        "建议考虑增加PBK过表达救援实验。",
        "该假设应该增加更多对照。",
        "Does PBK regulate persister survival?",
    ],
)
def test_editorial_or_question_statements_are_rejected(statement: str) -> None:
    assert not M4HypothesisGeneration._is_scientific_statement(statement)


@pytest.mark.parametrize(
    "statement",
    [
        "PBK activation sustains survival signaling in drug-tolerant persister cells.",
        "Adding genetic PBK knockdown to IOX2 treatment will reduce persister survival.",
        "PBK激活通过AXL旁路信号维持耐药持留细胞存活。",
    ],
)
def test_objective_scientific_statements_are_accepted(statement: str) -> None:
    assert M4HypothesisGeneration._is_scientific_statement(statement)


def test_normalisation_discards_editorial_statements() -> None:
    module = M4HypothesisGeneration()
    cards = module._normalise_hypotheses([
        _card("H1", "PBK activation sustains persister survival.").model_dump(),
        _card("H2", "Consider adding a PBK rescue experiment.").model_dump(),
    ])

    assert [card.hypothesis_id for card in cards] == ["H1"]


class _SequenceClient:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)

    async def structured_chat(self, **kwargs: Any) -> Any:
        return self.responses.pop(0)


class _RecordingClient:
    def __init__(self, response: Any) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    async def structured_chat(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.response


class _BlockingClient:
    async def structured_chat(self, **kwargs: Any) -> Any:
        await asyncio.Event().wait()


@pytest.mark.asyncio
async def test_semantic_contract_audit_batches_all_candidates_in_one_call() -> None:
    client = _RecordingClient([
        {"hypothesis_id": "H1", "consistent": True, "rationale": "same object"},
        {"hypothesis_id": "H2", "consistent": True, "rationale": "same object"},
        {"hypothesis_id": "H3", "consistent": True, "rationale": "same object"},
    ])
    module = M4HypothesisGeneration(semantic_alignment_timeout_seconds=0.5)
    module.client = client
    state = PipelineState(
        input_question="如何训练视觉模型检测深度？",
        problem_card=ProblemCard(
            original_question="如何训练视觉模型检测深度？",
            sub_questions=["如何训练视觉模型检测深度？"],
            key_entities=["视觉模型"],
            domain=["computer vision"],
        ),
    )
    candidates = [
        _card("H1", "视觉模型可通过单目深度监督学习距离。"),
        _card("H2", "视觉模型可联合预测深度与不确定性。"),
        _card("H3", "视觉模型可利用几何一致性学习相对距离。"),
    ]

    accepted, failures = await module._audit_context_contract_semantics(
        state, candidates
    )

    assert [card.hypothesis_id for card in accepted] == ["H1", "H2", "H3"]
    assert failures == []
    assert len(client.calls) == 1
    assert "H1" in client.calls[0]["user_prompt"]
    assert "H2" in client.calls[0]["user_prompt"]
    assert "H3" in client.calls[0]["user_prompt"]


@pytest.mark.asyncio
async def test_semantic_contract_audit_has_a_hard_timeout() -> None:
    module = M4HypothesisGeneration(semantic_alignment_timeout_seconds=0.03)
    module.client = _BlockingClient()
    state = PipelineState(
        input_question="如何训练视觉模型检测深度？",
        problem_card=ProblemCard(
            original_question="如何训练视觉模型检测深度？",
            sub_questions=["如何训练视觉模型检测深度？"],
            key_entities=["视觉模型"],
            domain=["computer vision"],
        ),
    )

    with pytest.raises(RuntimeError, match="timed out after 0.03 seconds"):
        await asyncio.wait_for(
            module._audit_context_contract_semantics(
                state,
                [_card("H1", "视觉模型可通过单目深度监督学习距离。")],
            ),
            timeout=0.25,
        )


@pytest.mark.asyncio
async def test_observed_m4_tool_cancellation_is_not_reported_as_failure(
    tmp_path,
) -> None:
    recorder = RunEventRecorder(tmp_path, "m4-cancel")

    async def blocked_operation() -> None:
        await asyncio.Event().wait()

    with bind_recorder(recorder):
        task = asyncio.create_task(
            M4HypothesisGeneration._observe_tool(
                "hypothesis_contract_auditor", blocked_operation()
            )
        )
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    event_types = [event["event_type"] for event in recorder.read_events()]
    assert event_types == ["tool_started", "tool_cancelled"]


@pytest.mark.asyncio
async def test_critic_cannot_overwrite_candidate_statements() -> None:
    module = M4HypothesisGeneration(num_candidates=2, top_k=1, mode="multi_agent")
    module.client = _SequenceClient([[
        {
            "hypothesis_id": "H1",
            "pass": True,
            "critique": "Add orthogonal validation.",
            "issues": ["off-target risk"],
            "suggested_revision": "Consider adding orthogonal validation.",
        },
        {
            "hypothesis_id": "H2",
            "pass": True,
            "critique": "Clarify evidence.",
            "issues": ["limited evidence"],
            "suggested_revision": "To strengthen the hypothesis, cite more evidence.",
        },
    ]])
    candidates = [
        _card("H1", "PBK activation sustains persister survival."),
        _card("H2", "PBK inhibition suppresses bypass signaling."),
    ]

    survivors = await module._run_critic(PipelineState(input_question="q"), candidates)

    assert [card.statement for card in survivors] == [
        "PBK activation sustains persister survival.",
        "PBK inhibition suppresses bypass signaling.",
    ]


def test_ranker_attaches_scores_without_accepting_rewritten_content() -> None:
    module = M4HypothesisGeneration(top_k=1)
    candidate = _card("H1", "PBK activation sustains persister survival.")
    ranked = module._attach_rankings([candidate], [{
        "hypothesis_id": "H1",
        "statement": "Consider adding orthogonal validation.",
        "mechanism": "rewritten content",
        "ranking_rationale": "Strong and testable.",
        "scores": {
            "novelty": 0.9,
            "scientific_soundness": 0.9,
            "testability": 0.9,
            "evidence_consistency": 0.9,
            "composite": 1.0,
        },
    }])

    assert len(ranked) == 1
    assert ranked[0].statement == candidate.statement
    assert ranked[0].mechanism == candidate.mechanism
    assert ranked[0].scores["composite"] == pytest.approx(0.9)


def test_invalid_high_scoring_candidate_cannot_replace_previous_best() -> None:
    module = M4HypothesisGeneration(top_k=1)
    previous = _card("H1", "PBK activation sustains persister survival.")
    previous.scores = {"composite": 0.8}
    editorial = _card("H2", "Consider adding orthogonal validation.")
    editorial.scores = {"composite": 0.99}
    state = PipelineState(input_question="q", best_hypotheses=[previous])

    best = module._update_best(state, [editorial])

    assert [card.hypothesis_id for card in best] == ["H1"]


@pytest.mark.asyncio
async def test_multi_agent_round_preserves_generator_statements_end_to_end() -> None:
    h1 = _card("H1-rev", "PBK activation sustains persister survival.")
    h2 = _card("H2-rev", "PBK inhibition suppresses AXL bypass signaling.")
    client = _SequenceClient([
        [h1.model_dump(), h2.model_dump()],
        [
            {"hypothesis_id": "H1-rev", "consistent": True, "rationale": "same object"},
            {"hypothesis_id": "H2-rev", "consistent": True, "rationale": "same object"},
        ],
        [
            {
                "hypothesis_id": "H1-rev",
                "pass": True,
                "critique": "Add validation.",
                "issues": [],
                "suggested_revision": "Consider adding orthogonal validation.",
            },
            {
                "hypothesis_id": "H2-rev",
                "pass": True,
                "critique": "Add evidence.",
                "issues": [],
                "suggested_revision": "To strengthen the hypothesis, add evidence.",
            },
        ],
        [
            {"hypothesis_id": "H1-rev", "is_falsifiable": True, "assessment": "yes"},
            {"hypothesis_id": "H2-rev", "is_falsifiable": True, "assessment": "yes"},
        ],
        [{
            "hypothesis_id": "H1-rev",
            "statement": "Consider adding orthogonal validation.",
            "ranking_rationale": "Best candidate.",
            "scores": {
                "novelty": 0.9,
                "scientific_soundness": 0.9,
                "testability": 0.9,
                "evidence_consistency": 0.9,
            },
        }],
    ])
    module = M4HypothesisGeneration(num_candidates=2, top_k=1, mode="multi_agent")
    module.client = client
    module.ranker_client = client

    result = await module._run_llm(
        PipelineState(
            input_question="q",
            problem_card=ProblemCard(
                original_question="q",
                sub_questions=["q"],
                key_entities=["PBK"],
                domain=[],
            ),
        ),
        "revision",
    )

    assert [card.statement for card in result["candidate_hypotheses"]] == [
        h1.statement,
        h2.statement,
    ]
    assert result["top_hypotheses"][0].statement == h1.statement


def test_method_feedback_is_routed_to_m5_not_m4() -> None:
    state = PipelineState(
        input_question="q",
        iteration_count=1,
        reviews=[
            ReviewResult(
                dimension=ReviewerDimension("scientific_logic"),
                attribution="hypothesis",
                score=3.0,
                suggestions="clarify the causal mechanism",
                hard_gate_passed=False,
                version=1,
            ),
            ReviewResult(
                dimension=ReviewerDimension("method_feasibility"),
                attribution="plan",
                score=3.0,
                suggestions="add a power analysis",
                hard_gate_passed=False,
                version=1,
            ),
        ],
    )
    m4_context = M4HypothesisGeneration()._build_feedback_context(state, [])
    m5_context = M5ResearchPlan._build_feedback_context(state)

    assert "causal mechanism" in m4_context
    assert "power analysis" not in m4_context
    assert "power analysis" in m5_context
    assert "causal mechanism" not in m5_context


# ---------------------------------------------------------------------------
# Followup text injection into M4/M5 prompts (task #16 fix 2)
# ---------------------------------------------------------------------------

def test_followup_text_is_injected_into_m4_and_m5_context() -> None:
    state = PipelineState(
        input_question="q",
        followup=FollowupRequest(text="请用中文输出方案", parent_run_id="p1"),
    )
    m4_context = M4HypothesisGeneration()._build_feedback_context(state, [])
    m5_context = M5ResearchPlan._build_feedback_context(state)

    for context in (m4_context, m5_context):
        assert "请用中文输出方案" in context
        assert "用户追问要求" in context


def test_followup_block_coexists_with_m4_revision_context() -> None:
    state = PipelineState(
        input_question="q",
        iteration_count=1,
        followup=FollowupRequest(text="输出中文方案", parent_run_id="p1"),
        reviews=[
            ReviewResult(
                dimension=ReviewerDimension("scientific_logic"),
                score=3.0,
                suggestions="clarify the causal mechanism",
                version=1,
            ),
        ],
    )
    context = M4HypothesisGeneration()._build_feedback_context(state, [])
    assert "输出中文方案" in context
    assert "causal mechanism" in context  # revision feedback still present


def test_m5_user_prompt_carries_followup_text() -> None:
    state = PipelineState(
        input_question="q",
        followup=FollowupRequest(text="请用中文输出方案", parent_run_id="p1"),
    )
    prompt = M5_USER_TEMPLATE.format(
        original_question=state.input_question,
        problem_card_json="{}",
        graph_context="(no evidence)",
        statement="PBK activation sustains persister survival.",
        mechanism="PBK -> bypass signaling",
        predictions="p1",
        falsification_conditions="f1",
        hypothesis_evidence="[]",
        feedback_context=M5ResearchPlan._build_feedback_context(state),
    )
    assert "请用中文输出方案" in prompt


def test_no_followup_keeps_prompt_context_unchanged() -> None:
    """Incremental injection: without a followup the context is exactly the
    legacy empty string (first round) in both modules."""
    plain = PipelineState(input_question="q")
    assert M4HypothesisGeneration()._build_feedback_context(plain, []) == ""
    assert M5ResearchPlan._build_feedback_context(plain) == ""

    # empty followup text is treated the same as no followup
    blank = PipelineState(input_question="q", followup=FollowupRequest(text="  "))
    assert M4HypothesisGeneration()._build_feedback_context(blank, []) == ""
    assert M5ResearchPlan._build_feedback_context(blank) == ""


# ==================== test_alignment_evidence_context ====================

import asyncio
import pytest
from pathlib import Path

from hypoforge.graph_context import build_graph_context
from hypoforge.modules.m1_problem_understanding import M1ProblemUnderstanding
from hypoforge.modules.m4_hypothesis_generation import M4HypothesisGeneration
from hypoforge.modules.m5_research_plan import M5ResearchPlan
from hypoforge.modules.m6_review_iteration import M6ReviewIteration
from hypoforge.modules.m3_evidence_graph import M3EvidenceGraph
from hypoforge.prompts.m5_prompts import M5_SYSTEM_PROMPT
from hypoforge.pipeline import _route_after_m4, _should_continue_iterating
from hypoforge.state import (
    EvidenceGapRequest,
    EvidenceGraph,
    HypothesisCard,
    KnowledgeEntry,
    M2EvidenceExport,
    M2KnowledgeExport,
    M2KnowledgeRun,
    M2PaperExport,
    PipelineState,
    ProblemCard,
    ResearchPlan,
    LiteratureResult,
    TaskContract,
    TaskEntity,
    TaskRequirement,
    TaskTrace,
    TaskTraceReference,
)
from hypoforge.task_alignment import (
    assess_task_alignment,
    search_entities_for_sub_question,
)


def robot_state() -> PipelineState:
    export = M2KnowledgeExport(runs=[M2KnowledgeRun(
        sub_question="How does domain randomization improve arm transfer?",
        papers=[M2PaperExport(paper_id="S2:p1", title="Manipulator transfer")],
        evidence=[M2EvidenceExport(
            evidence_id="ev-arm-1",
            paper_id="S2:p1",
            chunk_id="chunk-1",
            quote="Domain randomization improved robotic-arm transfer success.",
            normalized_claim="Domain randomization improves robotic-arm transfer success.",
            relevance_score=0.95,
        )],
        knowledge_entries=[KnowledgeEntry(
            id="fact-arm-1",
            type="established_fact",
            content="Domain randomization improves robotic-arm transfer success.",
            source_paper_id="S2:p1",
            source_paper_title="Manipulator transfer",
            entities=["robotic arm", "domain randomization"],
            evidence_ids=["ev-arm-1"],
        )],
    )])
    return PipelineState(
        input_question="如何改进机械臂 sim-to-real 操控？",
        problem_card=ProblemCard(
            original_question="如何改进机械臂 sim-to-real 操控？",
            domain=["robotics", "robot manipulation"],
            key_entities=["机械臂", "sim-to-real", "域随机化"],
            sub_questions=["域随机化如何影响机械臂 sim-to-real 成功率？"],
            question_type="method_development",
            task_contract=TaskContract(
                entities=[
                    TaskEntity(
                        entity_id="E1", name="机械臂",
                        aliases=["robotic arm", "manipulator"],
                        role="primary_object", required=True,
                    ),
                    TaskEntity(
                        entity_id="E2", name="sim-to-real",
                        aliases=["simulation-to-reality transfer"],
                        role="method", required=True,
                    ),
                    TaskEntity(
                        entity_id="E3", name="域随机化",
                        aliases=["domain randomization"],
                        role="intervention", required=False,
                    ),
                ],
                requirements=[TaskRequirement(
                    requirement_id="R1",
                    sub_question="域随机化如何影响机械臂 sim-to-real 成功率？",
                    primary_entity_id="E1",
                    related_entity_ids=["E2", "E3"],
                    relation="evaluate transfer success",
                )],
            ),
        ),
        m2_knowledge_export=export,
        evidence_graph=EvidenceGraph(established_facts=["fact-arm-1"]),
    )


def hypothesis() -> HypothesisCard:
    return HypothesisCard(
        hypothesis_id="H1",
        statement="域随机化可提高机械臂 sim-to-real 操控成功率。",
        mechanism="域随机化 → 鲁棒表征 → 机械臂迁移成功",
        observable_predictions=["机械臂真实抓取成功率提高。"],
        falsification_conditions=["真实抓取成功率不变。"],
        supporting_evidence=["ev-arm-1"],
        task_trace=TaskTrace(
            entity_mentions=[
                TaskTraceReference(contract_id="E1", output_excerpt="机械臂"),
                TaskTraceReference(contract_id="E2", output_excerpt="sim-to-real"),
            ],
            requirement_mentions=[TaskTraceReference(
                contract_id="R1",
                output_excerpt="域随机化可提高机械臂 sim-to-real 操控成功率。",
            )],
        ),
    )


def test_graph_context_resolves_export_history_to_text_and_quote() -> None:
    context = build_graph_context(robot_state())

    assert context.established_facts[0].text.startswith("Domain randomization")
    assert context.established_facts[0].evidence_ids == ["ev-arm-1"]
    assert "robotic-arm transfer" in context.established_facts[0].quotes[0]
    assert context.unresolved_entry_ids == []


def test_task_alignment_rejects_legged_robot_substitution() -> None:
    assessment = assess_task_alignment(
        robot_state(),
        "使用四足机器人研究行走和防摔倒，并优化腿式机器人控制。",
    )

    assert assessment.passed is False
    assert assessment.missing_anchors
    assert assessment.missing_requirement_ids


def _contract_state(
    primary: str,
    alias: str,
    relation: str,
    sub_question: str,
) -> PipelineState:
    return PipelineState(
        input_question=sub_question,
        problem_card=ProblemCard(
            original_question=sub_question,
            domain=["test-domain"],
            sub_questions=[sub_question],
            key_entities=[primary],
            task_contract=TaskContract(
                entities=[TaskEntity(
                    entity_id="E1",
                    name=primary,
                    aliases=[alias],
                    role="primary_object",
                    required=True,
                )],
                requirements=[TaskRequirement(
                    requirement_id="R1",
                    sub_question=sub_question,
                    primary_entity_id="E1",
                    relation=relation,
                )],
            ),
        ),
    )


@pytest.mark.parametrize(("primary", "alias", "relation", "output"), [
    (
        "锂金属电池", "lithium-metal battery", "reduce dendrite growth",
        "A lithium-metal battery electrolyte reduces dendrite growth.",
    ),
    (
        "区域气候模型", "regional climate model", "estimate rainfall bias",
        "The regional climate model estimates rainfall bias after calibration.",
    ),
    (
        "高血压患者队列", "hypertension patient cohort", "compare treatment response",
        "The hypertension patient cohort shows a different treatment response.",
    ),
])
def test_task_contract_alignment_is_domain_neutral(
    primary: str,
    alias: str,
    relation: str,
    output: str,
) -> None:
    question = f"How can we {relation} in {alias}?"
    state = _contract_state(primary, alias, relation, question)
    trace = TaskTrace(
        entity_mentions=[TaskTraceReference(
            contract_id="E1", output_excerpt=alias,
        )],
        requirement_mentions=[TaskTraceReference(
            contract_id="R1", output_excerpt=output,
        )],
    )

    accepted = assess_task_alignment(
        state, output, subject_text=output, trace=trace,
    )
    rejected = assess_task_alignment(
        state,
        "An unrelated system is evaluated under the same protocol.",
        subject_text="An unrelated system",
        trace=trace,
    )

    assert accepted.passed is True
    assert rejected.passed is False
    assert rejected.invalid_trace_references


def test_alignment_implementation_contains_no_case_domain_vocabulary() -> None:
    source = (
        Path(__file__).parents[1] / "hypoforge" / "task_alignment.py"
    ).read_text(encoding="utf-8").casefold()

    for forbidden in ("quadruped", "biped", "robot arm", "biomedical", "tumor"):
        assert forbidden not in source


def test_search_entities_are_scoped_to_atomic_requirement() -> None:
    state = robot_state()
    state.problem_card.sub_questions.append("如何测量 sim-to-real 成功率？")
    state.problem_card.task_contract.requirements.append(TaskRequirement(
        requirement_id="R2",
        sub_question="如何测量 sim-to-real 成功率？",
        primary_entity_id="E1",
        related_entity_ids=["E2"],
        relation="measure transfer success",
    ))

    anchors = search_entities_for_sub_question(
        state, "如何测量 sim-to-real 成功率？"
    )

    assert anchors == ["robotic arm", "sim-to-real"]
    assert "domain randomization" not in anchors


def test_m4_evidence_gap_lifecycle_is_deduplicated_and_bounded() -> None:
    state = robot_state()
    unsupported = hypothesis().model_copy(update={"supporting_evidence": []})
    context = build_graph_context(state)

    first = M4HypothesisGeneration._update_evidence_gaps(
        state, [unsupported], context,
    )
    assert len(first) == 1
    assert first[0].status == "pending"
    assert _route_after_m4(state.model_copy(update={
        "evidence_gap_requests": first,
    })) == "search_gap"

    indexed = first[0].model_copy(update={"status": "indexed", "attempts": 1})
    exhausted = M4HypothesisGeneration._update_evidence_gaps(
        state.model_copy(update={"evidence_gap_requests": [indexed]}),
        [unsupported],
        context,
    )
    assert [gap.gap_id for gap in exhausted] == [first[0].gap_id]
    assert exhausted[0].status == "exhausted"
    assert _route_after_m4(state.model_copy(update={
        "evidence_gap_requests": exhausted,
    })) == "continue"

    resolved = M4HypothesisGeneration._update_evidence_gaps(
        state.model_copy(update={"evidence_gap_requests": [indexed]}),
        [hypothesis()],
        context,
    )
    assert resolved[0].status == "resolved"
    assert resolved[0].resolution_evidence_ids == ["ev-arm-1"]


def test_m4_gap_router_ignores_non_pending_requests() -> None:
    state = robot_state().model_copy(update={
        "evidence_gap_requests": [EvidenceGapRequest(
            gap_id="GAP_test",
            sub_question="generic question",
            status="searched",
            attempts=1,
        )],
    })

    assert _route_after_m4(state) == "continue"


class HypothesisRepairClient:
    def __init__(self, responses) -> None:
        self.responses = list(responses)
        self.calls = []

    async def structured_chat(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)

    async def chat(self, prompt: str = "", **kwargs) -> str:
        return "yes — test semantic client is consistent"


@pytest.mark.asyncio
async def test_m4_repairs_contract_diagnostics_once_without_weakening_gate() -> None:
    state = _contract_state(
        "锂金属电池",
        "lithium-metal battery",
        "reduce dendrite growth",
        "How can an electrolyte reduce dendrite growth in a lithium-metal battery?",
    )
    invalid = {
        "hypothesis_id": "H1",
        "statement": "A generic device becomes more stable after treatment.",
        "mechanism": "treatment -> stability",
        "observable_predictions": ["Failure frequency decreases."],
        "falsification_conditions": ["Failure frequency does not change."],
        "supporting_evidence": ["invented-evidence"],
        "task_trace": {},
    }
    valid_statement = (
        "A lithium-metal battery electrolyte reduces dendrite growth during cycling."
    )
    repaired = {
        "hypothesis_id": "H1",
        "statement": valid_statement,
        "mechanism": (
            "The lithium-metal battery electrolyte homogenizes ion flux and "
            "thereby reduces dendrite growth."
        ),
        "observable_predictions": [
            "Lithium-metal battery cells show lower dendrite coverage."
        ],
        "falsification_conditions": [
            "Dendrite growth is unchanged in the lithium-metal battery."
        ],
        "supporting_evidence": [],
        "task_trace": {
            "entity_mentions": [{
                "contract_id": "E1",
                "output_excerpt": "lithium-metal battery",
            }],
            "requirement_mentions": [{
                "contract_id": "R1",
                "output_excerpt": valid_statement,
            }],
        },
    }
    client = HypothesisRepairClient([
        [invalid],
        [repaired],
        [{
            "hypothesis_id": "H1",
            "consistent": True,
            "rationale": "same lithium-metal battery research object",
        }],
    ])
    module = M4HypothesisGeneration(num_candidates=1, top_k=1, mode="direct")
    module.client = client

    result = await module._run_llm(state)

    assert len(client.calls) == 3
    assert client.calls[1]["temperature"] == 0.0
    assert "missing required task entities" in client.calls[1]["user_prompt"]
    assert "hypothesis_contract_auditor" not in client.calls[1]["user_prompt"]
    assert "Primary task objects" in client.calls[2]["user_prompt"]
    assert result["top_hypotheses"][0].statement == valid_statement


@pytest.mark.asyncio
async def test_m4_generator_prompt_carries_literal_contract_names() -> None:
    """The generator must see the exact contract names/aliases so it can quote
    them verbatim — the alignment gate matches them character-for-character."""
    state = _contract_state(
        "锂金属电池",
        "lithium-metal battery",
        "reduce dendrite growth",
        "How can an electrolyte reduce dendrite growth in a lithium-metal battery?",
    )
    valid = {
        "hypothesis_id": "H1",
        "statement": (
            "A 锂金属电池 electrolyte reduces dendrite growth during cycling."
        ),
        "mechanism": "锂金属电池 ion flux homogenization -> less dendrite growth",
        "observable_predictions": ["Dendrite coverage decreases."],
        "falsification_conditions": ["Dendrite growth is unchanged."],
        "supporting_evidence": [],
        "task_trace": {
            "entity_mentions": [{
                "contract_id": "E1",
                "output_excerpt": "锂金属电池",
            }],
            "requirement_mentions": [{
                "contract_id": "R1",
                "output_excerpt": "A 锂金属电池 electrolyte reduces dendrite growth during cycling.",
            }],
        },
    }
    client = HypothesisRepairClient([
        [valid],
        [{
            "hypothesis_id": "H1",
            "consistent": True,
            "rationale": "same research object",
        }],
    ])
    module = M4HypothesisGeneration(num_candidates=1, top_k=1, mode="direct")
    module.client = client

    result = await module._run_llm(state)

    generator_prompt = client.calls[0]["user_prompt"]
    assert "锂金属电池" in generator_prompt
    assert "lithium-metal battery" in generator_prompt
    assert "R1" in generator_prompt
    assert "Do not translate or paraphrase" in generator_prompt
    assert result["top_hypotheses"][0].statement == valid["statement"]


@pytest.mark.asyncio
async def test_m4_repair_prompt_carries_literal_contract_names() -> None:
    """The deterministic repair must also see the exact contract names."""
    state = _contract_state(
        "锂金属电池",
        "lithium-metal battery",
        "reduce dendrite growth",
        "How can an electrolyte reduce dendrite growth in a lithium-metal battery?",
    )
    invalid = {
        "hypothesis_id": "H1",
        "statement": "A generic device becomes more stable after treatment.",
        "mechanism": "treatment -> stability",
        "observable_predictions": ["Failure frequency decreases."],
        "falsification_conditions": ["Failure frequency does not change."],
        "supporting_evidence": [],
        "task_trace": {},
    }
    valid = {
        "hypothesis_id": "H1",
        "statement": (
            "A 锂金属电池 electrolyte reduces dendrite growth during cycling."
        ),
        "mechanism": "锂金属电池 ion flux homogenization -> less dendrite growth",
        "observable_predictions": ["Dendrite coverage decreases."],
        "falsification_conditions": ["Dendrite growth is unchanged."],
        "supporting_evidence": [],
        "task_trace": {
            "entity_mentions": [{
                "contract_id": "E1",
                "output_excerpt": "锂金属电池",
            }],
            "requirement_mentions": [{
                "contract_id": "R1",
                "output_excerpt": "A 锂金属电池 electrolyte reduces dendrite growth during cycling.",
            }],
        },
    }
    client = HypothesisRepairClient([
        [invalid],
        [valid],
        [{
            "hypothesis_id": "H1",
            "consistent": True,
            "rationale": "same research object",
        }],
    ])
    module = M4HypothesisGeneration(num_candidates=1, top_k=1, mode="direct")
    module.client = client

    result = await module._run_llm(state)

    repair_prompt = client.calls[1]["user_prompt"]
    assert "锂金属电池" in repair_prompt
    assert "lithium-metal battery" in repair_prompt
    assert "Binding task contract" in repair_prompt
    assert result["top_hypotheses"][0].statement == valid["statement"]


def _pose_estimation_contract_state() -> PipelineState:
    question = "如何训练模型使之可以从单张图片识别出物体的位置、旋转等信息？"
    return PipelineState(
        input_question=question,
        problem_card=ProblemCard(
            original_question=question,
            domain=["computer vision"],
            sub_questions=[
                "如何训练模型从单张图片联合估计物体的位置和旋转？",
            ],
            key_entities=["模型", "单张图片", "物体的位置", "旋转", "训练"],
            task_contract=TaskContract(
                entities=[
                    TaskEntity(
                        entity_id="E1", name="模型", role="primary_object",
                        required=True,
                    ),
                    TaskEntity(
                        entity_id="E2", name="单张图片", role="context",
                        required=True,
                    ),
                    TaskEntity(
                        entity_id="E3", name="物体的位置", role="outcome",
                        required=True,
                    ),
                    TaskEntity(
                        entity_id="E4", name="旋转", role="outcome",
                        required=True,
                    ),
                    TaskEntity(
                        entity_id="E5", name="训练", role="method",
                        required=True,
                    ),
                ],
                requirements=[
                    TaskRequirement(
                        requirement_id="R1",
                        sub_question="如何训练模型从单张图片联合估计物体的位置和旋转？",
                        primary_entity_id="E1",
                        related_entity_ids=["E2", "E3", "E4", "E5"],
                        relation="联合估计",
                    ),
                    TaskRequirement(
                        requirement_id="R2",
                        sub_question="如何评价模型的位置和旋转预测？",
                        primary_entity_id="E1",
                        related_entity_ids=["E3", "E4"],
                        relation="评价预测",
                    ),
                ],
            ),
        ),
    )


def test_m4_canonicalizes_wrong_trace_excerpts_from_literal_content() -> None:
    state = _pose_estimation_contract_state()
    statement = "训练模型从单张图片联合预测物体的位置和旋转。"
    candidate = HypothesisCard(
        hypothesis_id="H1",
        statement=statement,
        mechanism="模型用共享视觉特征同时回归物体的位置与旋转。",
        observable_predictions=["模型的位置和旋转误差均低于基线。"],
        falsification_conditions=["模型在单张图片上的位置或旋转误差不下降。"],
        task_trace=TaskTrace(
            entity_mentions=[TaskTraceReference(
                contract_id="E5", output_excerpt="模型",
            )],
            requirement_mentions=[
                TaskTraceReference(contract_id="R1", output_excerpt="训练"),
                TaskTraceReference(contract_id="R2", output_excerpt="旋转"),
            ],
        ),
    )

    canonical = M4HypothesisGeneration._canonicalize_task_traces(
        state, [candidate],
    )
    accepted, failures = M4HypothesisGeneration._check_context_contract(
        state, build_graph_context(state), canonical,
    )

    assert [card.hypothesis_id for card in accepted] == ["H1"]
    assert failures == []
    assert canonical[0].task_trace.entity_mentions[4].output_excerpt == statement
    assert {
        ref.contract_id for ref in canonical[0].task_trace.requirement_mentions
    } == {"R1", "R2"}


def test_m4_trace_canonicalization_does_not_invent_missing_content() -> None:
    state = _pose_estimation_contract_state()
    candidate = HypothesisCard(
        hypothesis_id="H1",
        statement="模型从单张图片联合预测物体的位置和旋转。",
        mechanism="模型用共享视觉特征执行多任务预测。",
        observable_predictions=["模型的位置和旋转误差均低于基线。"],
        falsification_conditions=["模型的位置或旋转误差不下降。"],
    )

    canonical = M4HypothesisGeneration._canonicalize_task_traces(
        state, [candidate],
    )
    accepted, failures = M4HypothesisGeneration._check_context_contract(
        state, build_graph_context(state), canonical,
    )

    assert accepted == []
    assert "训练" in failures[0]["missing_task_entities"]
    assert all(
        ref.contract_id != "E5"
        for ref in canonical[0].task_trace.entity_mentions
    )


@pytest.mark.asyncio
async def test_m4_repair_does_not_admit_a_second_invalid_result() -> None:
    state = _contract_state(
        "regional climate model",
        "RCM",
        "estimate rainfall bias",
        "How can an RCM estimate rainfall bias?",
    )
    invalid = {
        "hypothesis_id": "H1",
        "statement": "An unrelated system produces a measurable output.",
        "mechanism": "input -> output",
        "observable_predictions": ["The output changes."],
        "falsification_conditions": ["The output does not change."],
        "task_trace": {},
    }
    client = HypothesisRepairClient([[invalid], [invalid]])
    module = M4HypothesisGeneration(num_candidates=1, top_k=1, mode="direct")
    module.client = client

    with pytest.raises(ValueError, match="one deterministic repair attempt"):
        await module._run_llm(state)

    assert len(client.calls) == 2


def test_m1_atomic_validator_flags_parallel_questions() -> None:
    violations = M1ProblemUnderstanding._sub_question_violations([
        "What is the mechanism; how does environmental stress affect it??",
        "How does domain randomization affect robot arm transfer?",
    ])

    assert len(violations) == 1
    assert "multiple question marks" in violations[0]


class PlanClient:
    def __init__(self) -> None:
        self.calls = []

    async def chat(self, prompt: str = "", **kwargs) -> str:
        return "yes — test semantic client is consistent"

    async def structured_chat(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            return {
                "study_subjects": "四足机器人行走平台",
                "procedures": ["测试行走和防摔倒。"],
                "supporting_evidence_ids": ["invented"],
            }
        return {
            "study_subjects": "机械臂 sim-to-real 抓取平台",
            "procedures": ["在域随机化仿真中训练机械臂，再测试真实抓取。"],
            "supporting_evidence_ids": ["ev-arm-1", "invented"],
            "source_paper_ids": ["S2:p1", "missing"],
            "evidence_links": [{
                "plan_element": "procedure:1",
                "claim": "域随机化支持机械臂迁移。",
                "supporting_evidence_ids": ["ev-arm-1"],
                "source_paper_ids": ["S2:p1"],
                "support_status": "supported",
            }],
            "task_trace": {
                "entity_mentions": [
                    {"contract_id": "E1", "output_excerpt": "机械臂"},
                    {"contract_id": "E2", "output_excerpt": "sim-to-real"},
                ],
                "requirement_mentions": [{
                    "contract_id": "R1",
                    "output_excerpt": "在域随机化仿真中训练机械臂，再测试真实抓取。",
                }],
            },
        }


class HangingPlanGenerationClient:
    async def structured_chat(self, **kwargs):
        await asyncio.Event().wait()


class HangingPlanSemanticClient:
    async def chat(self, **kwargs):
        await asyncio.Event().wait()


@pytest.mark.asyncio
async def test_m5_plan_generation_times_out_instead_of_hanging() -> None:
    state = robot_state().model_copy(update={"top_hypotheses": [hypothesis()]})
    module = M5ResearchPlan(generation_timeout_seconds=0.01)
    module.client = HangingPlanGenerationClient()

    with pytest.raises(RuntimeError, match="generation timed out"):
        await asyncio.wait_for(module(state), timeout=0.5)


@pytest.mark.asyncio
async def test_m5_semantic_alignment_times_out_without_blocking_event_loop() -> None:
    state = robot_state()
    plan = ResearchPlan(study_subjects="机械臂 sim-to-real 抓取平台")
    module = M5ResearchPlan(semantic_alignment_timeout_seconds=0.01)
    module.client = HangingPlanSemanticClient()

    with pytest.raises(RuntimeError, match="semantic task-contract audit timed out"):
        await asyncio.wait_for(
            module._audit_plan_semantics(state, plan),
            timeout=0.5,
        )


def test_m5_requires_original_language_and_literal_contract_terms() -> None:
    prompt = M5_SYSTEM_PROMPT.casefold()

    assert "same language as the original question" in prompt
    assert "verbatim" in prompt
    assert "never embed a chinese" in prompt
    assert "contract name inside an english sentence" in prompt
    assert "never paraphrase an entity" in prompt


def test_m5_canonicalizes_trace_excerpts_from_plan_content() -> None:
    state = _pose_estimation_contract_state()
    plan = ResearchPlan(
        hypothesis_id="H1",
        study_subjects="训练模型从单张图片识别物体的位置和旋转。",
        procedures=["使用单张图片训练模型并联合预测物体的位置和旋转。"],
        measurement_metrics=["分别测量模型的位置误差和旋转误差。"],
        task_trace=TaskTrace(
            entity_mentions=[TaskTraceReference(
                contract_id="E5", output_excerpt="模型",
            )],
            requirement_mentions=[TaskTraceReference(
                contract_id="R1", output_excerpt="训练",
            )],
        ),
    )

    canonical = M5ResearchPlan._canonicalize_task_trace(state, plan)
    assessment = assess_task_alignment(
        state,
        M5ResearchPlan._plan_alignment_text(canonical),
        subject_text=canonical.study_subjects,
        trace=canonical.task_trace,
        semantic_client=None,
    )

    assert assessment.passed is True
    assert {
        item.contract_id for item in canonical.task_trace.entity_mentions
    } == {"E1", "E2", "E3", "E4", "E5"}
    assert {
        item.contract_id for item in canonical.task_trace.requirement_mentions
    } == {"R1", "R2"}


@pytest.mark.asyncio
async def test_m5_retries_object_drift_and_sanitizes_citations() -> None:
    state = robot_state().model_copy(update={"top_hypotheses": [hypothesis()]})
    module = M5ResearchPlan()
    module.client = PlanClient()

    result = await module(state)

    assert len(module.client.calls) == 2
    plan = result["research_plans"][0]
    assert "机械臂" in plan.study_subjects
    assert plan.supporting_evidence_ids == ["ev-arm-1"]
    assert plan.source_paper_ids == ["S2:p1"]
    assert "Original question" in module.client.calls[1]["user_prompt"]
    assert "ev-arm-1" in module.client.calls[1]["user_prompt"]


class ReviewClient:
    def __init__(self, chat_response: str = "yes — consistent") -> None:
        self._chat_response = chat_response

    async def chat(self, prompt: str = "", **kwargs) -> str:
        return self._chat_response

    async def structured_chat(self, **kwargs):
        prompt = kwargs["system_prompt"]
        base = {
            "reasoning": "Reviewed against the supplied task and graph context.",
            "score": 5.0,
            "comments": "",
            "suggestions": "",
            "hard_gate_passed": True,
            "evidence_ids": [],
        }
        if "evidence consistency" in prompt:
            # Deliberately claims a high score without an auditable citation.
            return base
        return base


class HangingReviewSemanticClient:
    async def chat(self, **kwargs):
        await asyncio.Event().wait()


@pytest.mark.asyncio
async def test_m6_pair_semantic_audit_times_out_without_blocking() -> None:
    state = robot_state()
    plan = ResearchPlan(study_subjects="机械臂 sim-to-real 抓取平台")
    module = M6ReviewIteration(semantic_alignment_timeout_seconds=0.01)
    module.client = HangingReviewSemanticClient()

    with pytest.raises(RuntimeError, match="semantic task-contract audit timed out"):
        await asyncio.wait_for(
            module._audit_pair_semantics(state, hypothesis(), plan),
            timeout=0.5,
        )


@pytest.mark.asyncio
async def test_m6_fails_closed_when_evidence_review_cites_no_id() -> None:
    state = robot_state().model_copy(update={
        "top_hypotheses": [hypothesis()],
        "research_plans": [ResearchPlan(
            hypothesis_id="H1",
            study_subjects="机械臂 sim-to-real 抓取平台",
            procedures=["执行机械臂真实抓取测试。"],
            task_trace=TaskTrace(
                entity_mentions=[
                    TaskTraceReference(contract_id="E1", output_excerpt="机械臂"),
                    TaskTraceReference(contract_id="E2", output_excerpt="sim-to-real"),
                ],
                requirement_mentions=[TaskTraceReference(
                    contract_id="R1",
                    output_excerpt="执行机械臂真实抓取测试。",
                )],
            ),
        )],
        "max_iterations": 3,
    })
    module = M6ReviewIteration()
    module.client = ReviewClient()

    result = await module(state)
    reviewed = state.model_copy(update=result)
    coverage_review = next(
        review for review in reviewed.reviews
        if review.dimension.value == "evidence_coverage_gate"
    )
    overall = next(
        review for review in reviewed.reviews
        if review.dimension.value == "overall"
    )

    # The evidence_consistency reviewer is gone; the objective coverage gate
    # now fails closed when a unit cites no auditable evidence ID.
    assert coverage_review.score < 4.5  # below the 90% pass line
    assert coverage_review.hard_gate_passed is False
    assert overall.hard_gate_passed is False
    assert _should_continue_iterating(reviewed) == "iterate"


@pytest.mark.asyncio
async def test_m6_rejects_legged_plan_even_if_llm_scores_it_five() -> None:
    state = robot_state().model_copy(update={
        "top_hypotheses": [hypothesis()],
        "research_plans": [ResearchPlan(
            hypothesis_id="H1",
            study_subjects="四足机器人行走平台",
            procedures=["优化腿式机器人防摔倒控制。"],
            supporting_evidence_ids=["ev-arm-1"],
        )],
    })
    module = M6ReviewIteration()
    module.client = ReviewClient(chat_response="no — semantic mismatch for legged robot")

    result = await module(state)
    task_review = next(
        review for review in result["reviews"]
        if review.dimension.value == "task_alignment"
    )
    overall = next(
        review for review in result["reviews"]
        if review.dimension.value == "overall"
    )

    assert task_review.hard_gate_passed is False
    assert task_review.score <= 2.0
    assert overall.score == 1.9


@pytest.mark.asyncio
async def test_m3_loads_task_relevant_persistent_graph(tmp_path) -> None:
    entry = KnowledgeEntry(
        id="persistent-arm-fact",
        type="established_fact",
        content="Robot arm domain randomization improves transfer.",
        source_paper_id="p-persistent",
        source_paper_title="Persistent paper",
        entities=["robot arm", "domain randomization"],
        evidence_ids=["ev-persistent"],
    )
    first_state = robot_state().model_copy(update={
        "literature_results": [LiteratureResult(
            sub_question="robot arm transfer",
            papers_retrieved=1,
            knowledge_entries=[entry],
        )],
        "m2_knowledge_export": None,
        "evidence_graph": None,
        "memory_cache_dir": str(tmp_path),
    })
    first = await M3EvidenceGraph(mode="rule")(first_state)
    assert first["evidence_graph"].nodes

    second_state = robot_state().model_copy(update={
        "literature_results": [],
        "m2_knowledge_export": None,
        "evidence_graph": None,
        "memory_cache_dir": str(tmp_path),
    })
    second = await M3EvidenceGraph(mode="rule")(second_state)

    assert any(
        node.metadata.get("entry_id") == "persistent-arm-fact"
        for node in second["evidence_graph"].nodes
    )


# ---- fail-closed regression tests ----


def test_assess_task_alignment_raises_on_missing_card() -> None:
    """Missing ProblemCard with require_contract=True must raise."""
    from hypoforge.task_alignment import assess_task_alignment
    from hypoforge.state import PipelineState
    state = PipelineState(input_question="test?")
    state.problem_card = None
    with pytest.raises(RuntimeError, match="no M1 ProblemCard"):
        assess_task_alignment(state, "test candidate")


def test_assess_task_alignment_permissive_when_require_contract_false() -> None:
    """Missing ProblemCard with require_contract=False is backward-compatible."""
    from hypoforge.task_alignment import assess_task_alignment
    from hypoforge.state import PipelineState
    state = PipelineState(input_question="test?")
    state.problem_card = None
    assessment = assess_task_alignment(
        state, "test candidate", require_contract=False,
    )
    assert assessment.passed
    assert assessment.score == 0.5


class FailingSemanticClient:
    async def chat(self, prompt: str) -> str:
        raise ConnectionError("simulated semantic judge failure")


class InconsistentSemanticClient:
    async def chat(self, prompt: str) -> str:
        return "no — subject text describes a different research object"


class ConsistentSemanticClient:
    async def chat(self, prompt: str) -> str:
        return "yes — subject text is about the same research object"


def test_semantic_client_consistency_passes() -> None:
    """Semantic judge returns yes, contract satisfied → alignment passes."""
    state = robot_state()
    trace = TaskTrace(
        entity_mentions=[
            TaskTraceReference(contract_id="E1", output_excerpt="机械臂"),
            TaskTraceReference(contract_id="E2", output_excerpt="sim-to-real"),
        ],
        requirement_mentions=[
            TaskTraceReference(
                contract_id="R1",
                output_excerpt="机械臂 sim-to-real 迁移实验",
            ),
        ],
    )
    client = ConsistentSemanticClient()
    assessment = assess_task_alignment(
        state,
        "机械臂 sim-to-real 迁移实验",
        subject_text="机械臂 sim-to-real 迁移实验",
        semantic_client=client,
        trace=trace,
    )
    assert assessment.passed


def test_semantic_client_inconsistency_is_hard_fail() -> None:
    """Semantic judge returns no → hard gate fail."""
    state = robot_state()
    trace = TaskTrace(
        entity_mentions=[
            TaskTraceReference(contract_id="E1", output_excerpt="leg"),
            TaskTraceReference(contract_id="E2", output_excerpt="design"),
        ],
        requirement_mentions=[
            TaskTraceReference(
                contract_id="R1",
                output_excerpt="leg design for walking robots",
            ),
        ],
    )
    client = InconsistentSemanticClient()
    assessment = assess_task_alignment(
        state,
        "leg design for walking robots",
        subject_text="leg design for walking robots",
        semantic_client=client,
        trace=trace,
    )
    assert not assessment.passed
    assert len(assessment.semantic_conflicts) > 0


def test_semantic_client_failure_propagates() -> None:
    """Semantic judge raises → RuntimeError, not silent pass."""
    state = robot_state()
    trace = TaskTrace(
        entity_mentions=[
            TaskTraceReference(contract_id="E1", output_excerpt="test"),
            TaskTraceReference(contract_id="E2", output_excerpt="candidate"),
        ],
        requirement_mentions=[
            TaskTraceReference(
                contract_id="R1",
                output_excerpt="test candidate",
            ),
        ],
    )
    client = FailingSemanticClient()
    with pytest.raises(RuntimeError, match="Semantic object-consistency"):
        assess_task_alignment(
            state,
            "test candidate",
            subject_text="test candidate",
            semantic_client=client,
            trace=trace,
        )


# --- merged from test_graph_corrections.py ---

from hypoforge.modules.m3_evidence_graph import M3EvidenceGraph
from hypoforge.modules.m6_review_iteration import M6ReviewIteration
from hypoforge.state import (
    EvidenceEdge,
    EvidenceEdgeRelation,
    EvidenceGraph,
    EvidenceNode,
    EvidenceNodeType,
    GraphCorrectionRequest,
    ReviewResult,
    ReviewerDimension,
)


def graph() -> EvidenceGraph:
    return EvidenceGraph(
        nodes=[
            EvidenceNode(
                id="N1", type=EvidenceNodeType.CLAIM, label="claim one",
                metadata={"evidence_ids": ["ev-1"]},
            ),
            EvidenceNode(
                id="N2", type=EvidenceNodeType.CLAIM, label="claim two",
            ),
        ],
        edges=[EvidenceEdge(
            source="N1", target="N2",
            relation=EvidenceEdgeRelation.SUPPORTS,
            evidence_ids=["ev-1"],
        )],
    )


def correction(**updates) -> GraphCorrectionRequest:
    payload = {
        "request_id": "GCR_1",
        "operation": "reclassify_edge",
        "source_node_id": "N1",
        "target_node_id": "N2",
        "current_relation": "supports",
        "proposed_relation": "contradicts",
        "evidence_ids": ["ev-1"],
        "reason": "The cited result has the opposite direction.",
    }
    payload.update(updates)
    return GraphCorrectionRequest(**payload)


def test_graph_correction_reclassifies_and_appends_versioned_audit() -> None:
    updated, requests = M3EvidenceGraph._apply_graph_corrections(
        graph(), [correction()], iteration=2,
    )

    assert updated.edges[0].relation is EvidenceEdgeRelation.CONTRADICTS
    assert updated.version == 2
    assert requests[0].status == "applied"
    assert updated.audit_log[0].graph_version_before == 1
    assert updated.audit_log[0].graph_version_after == 2
    assert updated.audit_log[0].before["relation"] == "supports"
    assert updated.audit_log[0].after["relation"] == "contradicts"


def test_graph_correction_rejects_unknown_node_without_mutation() -> None:
    updated, requests = M3EvidenceGraph._apply_graph_corrections(
        graph(),
        [correction(target_node_id="missing")],
        iteration=2,
    )

    assert updated.edges[0].relation is EvidenceEdgeRelation.SUPPORTS
    assert updated.version == 1
    assert requests[0].status == "rejected"
    assert "unknown graph node" in requests[0].rejection_reason
    assert updated.audit_log[0].status == "rejected"


def test_graph_correction_rejects_request_without_evidence() -> None:
    updated, requests = M3EvidenceGraph._apply_graph_corrections(
        graph(), [correction(evidence_ids=[])], iteration=2,
    )

    assert updated.version == 1
    assert requests[0].status == "rejected"
    assert "no canonical supporting evidence" in requests[0].rejection_reason


def test_m6_normalizes_request_id_and_filters_unknown_evidence() -> None:
    review = ReviewResult(
        dimension=ReviewerDimension("objective_evidence_consistency"),
        score=2,
        graph_correction_requests=[correction(
            request_id="", evidence_ids=["ev-1", "invented"],
        )],
    )

    normalized = M6ReviewIteration._normalise_graph_corrections(
        review, valid_evidence_ids={"ev-1"}, version=3,
    )

    assert normalized[0].request_id.startswith("GCR_")
    assert normalized[0].evidence_ids == ["ev-1"]
    assert normalized[0].iteration == 3
    assert normalized[0].status == "pending"


# ==================== test_strict_contracts ====================

import json

import pytest

from hypoforge.graph_context import GraphContext
from hypoforge.config import PipelineConfig, SearchConfig
from hypoforge.registry import ModuleRegistry
from hypoforge.literature.models import (
    EvidenceChunk,
    EvidenceLinkedKnowledge,
    PaperReadingResult,
    PaperRecord,
)
from hypoforge.memory import PaperStore
from hypoforge.modules.m3_grounding.models import EvidenceRecord
from hypoforge.state import (
    ConfidenceLevel,
    EvidenceGap,
    EvidenceGraph,
    EvidenceNode,
    EvidenceNodeType,
    HypothesisCard,
    KnowledgeEntry,
    KnowledgeEntryType,
    LiteratureResult,
    M2EvidenceExport,
    M2KnowledgeExport,
    M2KnowledgeRun,
    M2PaperExport,
    PipelineState,
    ResearchPlan,
    ResearchPlanEvidenceLink,
    SearchLedger,
)
from hypoforge.strict_contracts import (
    StrictAgenticM2Adapter,
    StrictAgenticM2Module,
    StrictEntityNormalizationService,
    StrictM3EvidenceGraph,
    StrictM4HypothesisGeneration,
    StrictM5ResearchPlan,
    StrictM6ReviewIteration,
)


class EqualEmbeddings:
    async def aembed_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]


class PairDecisionClient:
    async def structured_chat(self, **kwargs):
        requests = json.loads(kwargs["user_prompt"])
        request = requests[0]
        decisions = []
        for candidate in request["candidates"]:
            decisions.append({
                "surface": request["surface"],
                "canonical_name": candidate,
                "same_concept": "regional" in candidate.casefold(),
                "rationale": "pair-level test decision",
            })
        return {"decisions": decisions}


@pytest.mark.asyncio
async def test_entity_pair_decisions_do_not_overwrite_each_other(tmp_path) -> None:
    service = StrictEntityNormalizationService(
        cache_dir=tmp_path,
        client=PairDecisionClient(),
        embedding_backend=EqualEmbeddings(),
    )
    service.register_alias_group("Regional Climate Model", [])
    service.register_alias_group("Global Climate Model", [])

    result = await service.resolve_batch(["RCM"])

    assert result["RCM"].canonical_name == "Regional Climate Model"
    positive = service.pair_decisions[
        service._pair_key("RCM", "Regional Climate Model")
    ]
    negative = service.pair_decisions[
        service._pair_key("RCM", "Global Climate Model")
    ]
    assert positive.same_concept is True
    assert negative.same_concept is False


class PartialDecisionClient:
    async def structured_chat(self, **kwargs):
        request = json.loads(kwargs["user_prompt"])[0]
        global_candidate = next(
            candidate for candidate in request["candidates"]
            if "global" in candidate.casefold()
        )
        return {"decisions": [{
            "surface": request["surface"],
            "canonical_name": global_candidate,
            "same_concept": False,
            "rationale": "global model is explicitly different",
        }]}


@pytest.mark.asyncio
async def test_unresolved_entity_is_not_persisted_as_confirmed_new_entity(tmp_path) -> None:
    first = StrictEntityNormalizationService(
        cache_dir=tmp_path,
        client=PartialDecisionClient(),
        embedding_backend=EqualEmbeddings(),
        run_stamp="strict-run",
    )
    first.register_alias_group("Regional Climate Model", [])
    first.register_alias_group("Global Climate Model", [])

    unresolved = await first.resolve_batch(["RCM"])

    assert unresolved["RCM"].decision_source == "unresolved_identity"
    assert "rcm" not in first.alias_to_id

    second = StrictEntityNormalizationService(
        cache_dir=tmp_path,
        client=PairDecisionClient(),
        embedding_backend=EqualEmbeddings(),
        run_stamp="strict-run",
    )
    resolved = await second.resolve_batch(["RCM"])

    assert resolved["RCM"].canonical_name == "Regional Climate Model"
class ProgrammableIdentityClient:
    """Judge with explicit per-pair rules; unknown pairs are negative."""

    def __init__(self, rules: dict[tuple[str, str], bool]) -> None:
        # Identity is symmetric, so rules are keyed by the sorted pair.
        self.rules = {
            tuple(sorted((left.casefold().strip(), right.casefold().strip()))): same
            for (left, right), same in rules.items()
        }
        self.calls = 0

    async def structured_chat(self, **kwargs):
        self.calls += 1
        requests = json.loads(kwargs["user_prompt"])
        decisions = []
        for request in requests:
            surface = request["surface"].casefold().strip()
            for candidate in request["candidates"]:
                key = tuple(sorted((surface, candidate.casefold().strip())))
                decisions.append({
                    "surface": request["surface"],
                    "canonical_name": candidate,
                    "same_concept": self.rules.get(key, False),
                    "rationale": "programmable test decision",
                })
        return {"decisions": decisions}


class LowSimilarityEmbeddings:
    async def aembed_documents(self, texts):
        # surface [0.6, 0.8] vs canonical [1.0, 0.0] → cosine 0.6:
        # below the 0.82 similarity gate but above the 0.5 low-score gate.
        return [[0.6, 0.8]] + [[1.0, 0.0] for _ in texts[1:]]


@pytest.mark.asyncio
class FailingRelationClient:
    async def structured_chat(self, **kwargs):
        raise ConnectionError("relation extractor unavailable")


@pytest.mark.asyncio
async def test_m3_all_relation_jobs_failed_is_hard_failure() -> None:
    module = StrictM3EvidenceGraph(
        mode="llm",
        relation_batch_size=1,
        enable_cross_batch=False,
    )
    module.client = FailingRelationClient()
    entries = [
        KnowledgeEntry(
            id="k1",
            type=KnowledgeEntryType.ESTABLISHED_FACT,
            content="fact one",
            source_paper_id="p1",
        ),
        KnowledgeEntry(
            id="k2",
            type=KnowledgeEntryType.ESTABLISHED_FACT,
            content="fact two",
            source_paper_id="p2",
        ),
    ]
    graph = module._build_rule_graph(entries)

    with pytest.raises(RuntimeError, match="Every configured M3 relation-extraction"):
        await module._enhance_with_llm_batched(graph, entries)


class RecordingGrounder:
    def __init__(self) -> None:
        self.calls = 0

    async def run(self, state):
        self.calls += 1
        evidence = state.m2_knowledge_export.runs[0].evidence[0]
        return {
            "evidence_records": [EvidenceRecord(
                id="rec-1",
                evidence_id=evidence.evidence_id,
                paper_id=evidence.paper_id,
                chunk_id=evidence.chunk_id,
                quote=evidence.quote,
                normalized_claim=evidence.normalized_claim,
                summary=evidence.normalized_claim,
                relevance_score=9.0,
            )],
            "claims": [],
            "relations": [],
            "report": None,
        }


@pytest.mark.asyncio
async def test_m3_resume_historical_graph_still_runs_grounding() -> None:
    module = StrictM3EvidenceGraph(grounding_enabled=False)
    module.grounding_enabled = True
    grounder = RecordingGrounder()
    module._grounder = grounder
    state = PipelineState(
        input_question="Q",
        literature_results=[],
        evidence_graph=EvidenceGraph(nodes=[
            EvidenceNode(
                id="N_old",
                type=EvidenceNodeType.EVIDENCE,
                label="historical evidence",
            )
        ]),
        m2_knowledge_export=M2KnowledgeExport(runs=[
            M2KnowledgeRun(
                sub_question="SQ",
                papers=[M2PaperExport(paper_id="p1", title="Paper")],
                evidence=[M2EvidenceExport(
                    evidence_id="e1",
                    paper_id="p1",
                    chunk_id="c1",
                    quote="evidence",
                    normalized_claim="claim",
                    relevance_score=0.9,
                )],
            )
        ]),
    )

    result = await module(state)

    assert result["evidence_graph"].nodes
    assert grounder.calls == 1


@pytest.mark.asyncio
async def test_m3_does_not_reground_already_grounded_evidence() -> None:
    module = StrictM3EvidenceGraph(grounding_enabled=False)
    module.grounding_enabled = True
    grounder = RecordingGrounder()
    module._grounder = grounder
    state = PipelineState(
        input_question="Q",
        literature_results=[],
        evidence_graph=EvidenceGraph(nodes=[
            EvidenceNode(
                id="GEV_e1",
                type=EvidenceNodeType.EVIDENCE,
                label="already grounded",
                metadata={
                    "evidence_id": "e1",
                    "provenance_level": "m2_fulltext_grounded",
                },
            )
        ]),
        m2_knowledge_export=M2KnowledgeExport(runs=[
            M2KnowledgeRun(
                sub_question="SQ",
                papers=[M2PaperExport(paper_id="p1", title="Paper")],
                evidence=[M2EvidenceExport(
                    evidence_id="e1",
                    paper_id="p1",
                    chunk_id="c1",
                    quote="evidence",
                    normalized_claim="claim",
                    relevance_score=0.9,
                )],
            )
        ]),
    )

    result = await module(state)

    assert result["evidence_graph"].nodes
    assert grounder.calls == 0


@pytest.mark.asyncio
async def test_m3_grounding_enabled_without_citable_export_fails() -> None:
    module = StrictM3EvidenceGraph(grounding_enabled=False)
    module.grounding_enabled = True
    module._grounder = RecordingGrounder()
    state = PipelineState(
        input_question="Q",
        literature_results=[LiteratureResult(
            sub_question="SQ",
            papers_retrieved=1,
            knowledge_entries=[KnowledgeEntry(
                id="k1",
                type=KnowledgeEntryType.ESTABLISHED_FACT,
                content="fact",
                source_paper_id="p1",
            )],
        )],
    )

    with pytest.raises(RuntimeError, match="no citable evidence"):
        await module(state)


def test_m5_requires_every_evidence_item_to_match_declared_paper() -> None:
    context = GraphContext(
        available_evidence_ids=["e1", "e2"],
        available_paper_ids=["p1", "p2", "p3"],
        evidence_to_paper={"e1": "p1", "e2": "p2"},
    )
    plan = ResearchPlan(
        hypothesis_id="H1",
        evidence_links=[ResearchPlanEvidenceLink(
            plan_element="procedure:1",
            supporting_evidence_ids=["e1", "e2"],
            source_paper_ids=["p1", "p3"],
            support_status="supported",
        )],
    )

    cleaned = StrictM5ResearchPlan._sanitize_evidence_links(plan, context)

    assert cleaned.evidence_links[0].support_status == "hypothesis_to_validate"
    assert cleaned.supporting_evidence_ids == []
    assert cleaned.source_paper_ids == []


class CacheReadingWorkflow:
    def __init__(self) -> None:
        self.calls = 0

    async def run(self, sub_question, papers, search_context=None):
        self.calls += 1
        output = []
        for paper in papers:
            output.append(PaperReadingResult(
                paper_id=paper.paper_id,
                evidence=[EvidenceChunk(
                    evidence_id=f"ev-{paper.paper_id}",
                    paper_id=paper.paper_id,
                    chunk_id=f"chunk-{paper.paper_id}",
                    quote="cached abstract evidence",
                    normalized_claim="cached candidate yields evidence",
                    relevance_score=0.9,
                )],
                knowledge_entries=[EvidenceLinkedKnowledge(
                    entry_id=f"ke-{paper.paper_id}",
                    entry_type=KnowledgeEntryType.ESTABLISHED_FACT,
                    content="cached candidate yields evidence",
                    confidence=ConfidenceLevel.HIGH,
                    evidence_ids=[f"ev-{paper.paper_id}"],
                )],
            ))
        return output


class MustNotSearch:
    async def run(self, *args, **kwargs):
        raise AssertionError("live search should not run after a usable cache hit")


@pytest.mark.asyncio
async def test_agentic_cache_hit_is_read_before_pending_grounding(tmp_path) -> None:
    store = PaperStore(tmp_path)
    cached = M2PaperExport(
        paper_id="cached-1",
        title="Cached paper",
        abstract="A real cached abstract.",
        sources=["paper_store"],
    )
    keys = store.upsert_papers([cached], run_id="old", round=0)
    store.record_query("gap query", keys, run_id="old", round=0)
    gap = EvidenceGap(
        description="missing evidence",
        suggested_queries=["gap query"],
        target_sub_question="SQ",
    )
    state = PipelineState(
        input_question="Q",
        search_round=1,
        run_id="run",
        memory_cache_dir=str(tmp_path),
        literature_results=[LiteratureResult(sub_question="SQ")],
        evidence_gaps=[gap],
        search_ledger=SearchLedger(),
    )
    reader = CacheReadingWorkflow()
    adapter = StrictAgenticM2Adapter(
        search_agent=MustNotSearch(),
        reading_workflow=reader,
    )

    result = await adapter(state)

    assert reader.calls == 1
    assert result["evidence_gaps"][0].status == "pending_grounding"
    assert result["m2_knowledge_export"].runs[0].evidence
    assert result["m2_knowledge_export"].runs[0].knowledge_entries


def test_registry_selects_strict_builtin_contracts() -> None:
    config = PipelineConfig(
        search=SearchConfig(implementation="agentic"),
        entity_embedding_model="text-embedding-v3",
    )
    instances = ModuleRegistry.build_all(config)

    assert isinstance(instances["m2"], StrictAgenticM2Module)
    assert isinstance(instances["m3"], StrictM3EvidenceGraph)
    assert isinstance(instances["m4"], StrictM4HypothesisGeneration)
    assert isinstance(instances["m5"], StrictM5ResearchPlan)
    assert isinstance(instances["m6"], StrictM6ReviewIteration)
    assert instances["m2"].adapter.entity_embedding_model == "text-embedding-v3"
    assert instances["m2"].adapter.entity_embedding_base_url == (
        config.evaluation.embedding.base_url
    )
    assert instances["m2"].adapter.entity_embedding_key_env == (
        config.evaluation.embedding.api_key_env_var
    )
    assert instances["m3"].entity_embedding_base_url == (
        config.evaluation.embedding.base_url
    )
    assert instances["m3"].entity_embedding_key_env == (
        config.evaluation.embedding.api_key_env_var
    )

    # Default config (no entity_embedding_model) → empty string, no embedding.
    config_nonembed = PipelineConfig(search=SearchConfig(implementation="agentic"))
    instances_nonembed = ModuleRegistry.build_all(config_nonembed)
    assert instances_nonembed["m2"].adapter.entity_embedding_model == ""


def test_registry_respects_explicit_empty_entity_embedding_model() -> None:
    config = PipelineConfig(
        search=SearchConfig(implementation="agentic"),
        entity_embedding_model="",
    )
    instances = ModuleRegistry.build_all(config)

    assert instances["m2"].adapter.entity_embedding_model == ""
    assert instances["m3"].entity_embedding_model == ""


class EmptyGrounder:
    async def run(self, state):
        return {
            "evidence_records": [],
            "claims": [],
            "relations": [],
            "report": None,
        }


@pytest.mark.asyncio
async def test_m3_grounding_success_without_current_provenance_fails() -> None:
    module = StrictM3EvidenceGraph(grounding_enabled=False)
    module.grounding_enabled = True
    module._grounder = EmptyGrounder()
    state = PipelineState(
        input_question="Q",
        literature_results=[],
        evidence_graph=EvidenceGraph(nodes=[
            EvidenceNode(
                id="N_old",
                type=EvidenceNodeType.EVIDENCE,
                label="historical evidence",
            )
        ]),
        m2_knowledge_export=M2KnowledgeExport(runs=[
            M2KnowledgeRun(
                sub_question="SQ",
                papers=[M2PaperExport(paper_id="p1", title="Paper")],
                evidence=[M2EvidenceExport(
                    evidence_id="e1",
                    paper_id="p1",
                    chunk_id="c1",
                    quote="evidence",
                    normalized_claim="claim",
                    relevance_score=0.9,
                )],
            )
        ]),
    )

    with pytest.raises(RuntimeError, match="grounding failed"):
        await module(state)


def test_registry_preserves_custom_registered_m3() -> None:
    from hypoforge.modules.m3_evidence_graph import M3EvidenceGraph

    class CustomM3(M3EvidenceGraph):
        pass

    original = ModuleRegistry._modules.get("m3")
    ModuleRegistry._modules["m3"] = CustomM3
    try:
        config = PipelineConfig(search=SearchConfig(implementation="agentic"))
        instances = ModuleRegistry.build_all(config)
        assert isinstance(instances["m3"], CustomM3)
    finally:
        if original is None:
            ModuleRegistry._modules.pop("m3", None)
        else:
            ModuleRegistry._modules["m3"] = original


def test_semantic_alignment_chat_uses_user_prompt_for_qwen_style_client() -> None:
    from hypoforge import task_alignment

    class QwenStyleClient:
        def __init__(self) -> None:
            self.system_prompt = None
            self.user_prompt = None

        async def chat(self, system_prompt="", user_prompt=""):
            self.system_prompt = system_prompt
            self.user_prompt = user_prompt
            return "yes — aligned"

    client = QwenStyleClient()
    response = task_alignment._call_chat(client, "alignment question")

    assert response.startswith("yes")
    assert client.system_prompt == ""
    assert client.user_prompt == "alignment question"


class FailingGroundingSynthesisClient:
    model = "fake-model"

    async def structured_chat(self, **kwargs):
        raise ConnectionError("synthetic synthesis outage")


@pytest.mark.asyncio
async def test_grounding_llm_synthesis_failure_cannot_return_fallback(tmp_path) -> None:
    import hypoforge.strict_contracts as strict
    from hypoforge.modules.m3_grounding.workflow import GroundingWorkflow

    workflow = GroundingWorkflow(
        client=FailingGroundingSynthesisClient(),
        mode="llm",
        cache_dir=str(tmp_path),
    )
    evidence = M2EvidenceExport(
        evidence_id="e-synth",
        paper_id="p1",
        chunk_id="c1",
        quote="quoted evidence",
        normalized_claim="atomic claim",
        relevance_score=0.9,
    )
    item = {"query": "Q", "score": 0.9, "evidence_item": evidence}

    with pytest.raises(RuntimeError, match="Grounding evidence synthesis failed"):
        await strict._strict_grounding_synthesize_one(workflow, item)


class EmptyRelationDecisionClient:
    model = "fake-model"

    async def structured_chat(self, **kwargs):
        return {"relations": []}


@pytest.mark.asyncio
async def test_grounding_relation_batch_requires_one_decision_per_pair(tmp_path) -> None:
    import hypoforge.strict_contracts as strict
    from hypoforge.modules.m3_grounding.models import RelationPair
    from hypoforge.modules.m3_grounding.workflow import GroundingWorkflow
    from hypoforge.state import AtomicClaim

    workflow = GroundingWorkflow(
        client=EmptyRelationDecisionClient(),
        mode="llm",
        cache_dir=str(tmp_path),
    )
    record = EvidenceRecord(
        id="er1",
        evidence_id="e1",
        paper_id="p1",
        quote="evidence",
        normalized_claim="claim",
        summary="evidence summary",
        relevance_score=9.0,
    )
    claim = AtomicClaim(
        id="c1",
        statement="claim",
        evidence_ids=["e1"],
        paper_ids=["p1"],
    )
    pair = RelationPair(
        id="pair1",
        source="er1",
        target="c1",
        source_type="evidence_record",
        target_type="claim",
        source_evidence_ids=["e1"],
        target_evidence_ids=["e1"],
        source_paper_ids=["p1"],
        target_paper_ids=["p1"],
    )

    with pytest.raises(RuntimeError, match="Partial relation judgment"):
        await strict._strict_grounding_judge_relation_batch(
            workflow,
            [pair],
            {claim.id: claim},
            {record.id: record},
        )


@pytest.mark.asyncio
async def test_configured_grounding_embedding_empty_result_fails(monkeypatch) -> None:
    import hypoforge.strict_contracts as strict
    from hypoforge.modules.m3_grounding.workflow import GroundingWorkflow

    async def empty_embedding(self, texts):
        return []

    monkeypatch.setattr(strict, "_ORIGINAL_GROUNDING_EMBED", empty_embedding)
    workflow = GroundingWorkflow(embedding_model="configured-embedding")

    with pytest.raises(RuntimeError, match="returned no vectors"):
        await strict._strict_grounding_embed(workflow, ["document", "query"])


@pytest.mark.asyncio
async def test_configured_grounding_embedding_vector_count_must_match(monkeypatch) -> None:
    import hypoforge.strict_contracts as strict
    from hypoforge.modules.m3_grounding.workflow import GroundingWorkflow

    async def short_embedding(self, texts):
        return [[1.0, 0.0]]

    monkeypatch.setattr(strict, "_ORIGINAL_GROUNDING_EMBED", short_embedding)
    workflow = GroundingWorkflow(embedding_model="configured-embedding")

    with pytest.raises(RuntimeError, match="1 vector"):
        await strict._strict_grounding_embed(workflow, ["document", "query"])


class MalformedScoutClient:
    def __init__(self) -> None:
        self.calls = 0

    async def structured_chat(self, **kwargs):
        self.calls += 1
        return {"notes": []}


@pytest.mark.asyncio
async def test_configured_scout_retries_once_then_skips_malformed_paper() -> None:
    import hypoforge.strict_contracts as strict
    from hypoforge.literature.search.scout import ScoutReader

    client = MalformedScoutClient()
    reader = ScoutReader(client=client)
    paper = PaperRecord(
        paper_id="p-scout",
        title="Relevant paper",
        abstract="Relevant evidence about the research question.",
        sources=["test"],
    )

    notes = await strict._strict_scout_read(reader, "research question", [paper])

    assert notes == []
    assert client.calls == 2


class RetryableScoutClient:
    def __init__(self) -> None:
        self.calls = 0

    async def structured_chat(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return {"notes": []}
        return {"notes": [{
            "paper_id": "p-scout-retry",
            "relevance": 0.7,
            "directness": 0.4,
            "relation": "insufficient",
            "supporting_sentence_ids": [],
            "contradicting_sentence_ids": [],
            "study_type": "method",
        }]}


@pytest.mark.asyncio
async def test_configured_scout_keeps_paper_when_single_paper_retry_recovers() -> None:
    import hypoforge.strict_contracts as strict
    from hypoforge.literature.search.scout import ScoutReader

    client = RetryableScoutClient()
    reader = ScoutReader(client=client)
    paper = PaperRecord(
        paper_id="p-scout-retry",
        title="Relevant method paper",
        abstract="Relevant evidence about the research question.",
        sources=["test"],
    )

    notes = await strict._strict_scout_read(reader, "research question", [paper])

    assert len(notes) == 1
    assert notes[0].paper_id == "p-scout-retry"
    assert client.calls == 2


@pytest.mark.asyncio
async def test_unconfigured_scout_may_use_explicit_conservative_mode() -> None:
    import hypoforge.strict_contracts as strict
    from hypoforge.literature.search.scout import ScoutReader

    reader = ScoutReader(client=None)
    paper = PaperRecord(
        paper_id="p-scout-rule",
        title="Rule screened paper",
        abstract="Some abstract text.",
        sources=["test"],
    )

    notes = await strict._strict_scout_read(reader, "question", [paper])
    assert len(notes) == 1
    assert notes[0].paper_id == paper.paper_id


@pytest.mark.asyncio
async def test_m3_llm_mode_without_client_fails_closed() -> None:
    module = StrictM3EvidenceGraph(mode="llm")
    module.client = None
    state = PipelineState(
        input_question="Q",
        literature_results=[LiteratureResult(
            sub_question="SQ",
            papers_retrieved=1,
            knowledge_entries=[KnowledgeEntry(
                id="k1",
                type=KnowledgeEntryType.ESTABLISHED_FACT,
                content="fact",
                source_paper_id="p1",
            )],
        )],
    )

    with pytest.raises(RuntimeError, match="explicitly requires an LLM client"):
        await module(state)


@pytest.mark.asyncio
async def test_agentic_stop_reason_error_is_raised(monkeypatch) -> None:
    import hypoforge.strict_contracts as strict
    from hypoforge.literature.models import SearchRunResult, StopReason

    async def errored_run(self, *args, **kwargs):
        return SearchRunResult(
            sub_question="SQ",
            stop_reason=StopReason.ERROR,
            errors=["scout_reader: simulated failure"],
        )

    monkeypatch.setattr(strict, "_ORIGINAL_SEARCH_AGENT_RUN", errored_run)

    with pytest.raises(RuntimeError, match="StopReason.ERROR"):
        await strict._strict_search_agent_run(object())


@pytest.mark.asyncio
async def test_agentic_non_error_stop_reason_remains_explicit_recovery(monkeypatch) -> None:
    import hypoforge.strict_contracts as strict
    from hypoforge.literature.models import SearchRunResult, StopReason

    async def bounded_run(self, *args, **kwargs):
        return SearchRunResult(
            sub_question="SQ",
            stop_reason=StopReason.TIME_BUDGET,
        )

    monkeypatch.setattr(strict, "_ORIGINAL_SEARCH_AGENT_RUN", bounded_run)
    result = await strict._strict_search_agent_run(object())
    assert result.stop_reason is StopReason.TIME_BUDGET


class M4VerdictClient:
    def __init__(self, payload):
        self.payload = payload

    async def structured_chat(self, **kwargs):
        return self.payload


@pytest.mark.asyncio
async def test_m4_critic_cannot_readmit_all_rejected_candidates() -> None:
    module = StrictM4HypothesisGeneration(mode="multi_agent")
    candidates = [
        HypothesisCard(hypothesis_id="H1", statement="hypothesis one"),
        HypothesisCard(hypothesis_id="H2", statement="hypothesis two"),
    ]
    module.client = M4VerdictClient([
        {"hypothesis_id": "H1", "pass": False, "critique": "bad", "issues": []},
        {"hypothesis_id": "H2", "pass": False, "critique": "bad", "issues": []},
    ])

    with pytest.raises(RuntimeError, match="rejected every candidate"):
        await module._run_critic(PipelineState(input_question="Q"), candidates)


@pytest.mark.asyncio
async def test_m4_falsifiability_requires_verdict_for_every_candidate() -> None:
    module = StrictM4HypothesisGeneration(mode="multi_agent")
    candidates = [
        HypothesisCard(hypothesis_id="H1", statement="hypothesis one"),
        HypothesisCard(hypothesis_id="H2", statement="hypothesis two"),
    ]
    module.client = M4VerdictClient([
        {
            "hypothesis_id": "H1",
            "is_falsifiable": True,
            "specificity_score": 0.8,
            "assessment": "testable",
        }
    ])

    with pytest.raises(RuntimeError, match="candidate IDs"):
        await module._run_falsifiability(
            PipelineState(input_question="Q"), candidates
        )


def test_m4_ranker_incomplete_top_k_cannot_fallback() -> None:
    module = StrictM4HypothesisGeneration(top_k=2, mode="multi_agent")
    candidates = [
        HypothesisCard(hypothesis_id="H1", statement="hypothesis one"),
        HypothesisCard(hypothesis_id="H2", statement="hypothesis two"),
    ]
    dimensions = {
        "novelty": 0.8,
        "scientific_soundness": 0.8,
        "testability": 0.8,
        "evidence_consistency": 0.8,
    }

    with pytest.raises(RuntimeError, match="incomplete valid ranking"):
        module._attach_rankings(
            candidates,
            [{
                "hypothesis_id": "H1",
                "ranking_rationale": "only one result",
                "scores": dimensions,
            }],
        )


@pytest.mark.asyncio
async def test_m5_empty_top_hypotheses_is_not_success() -> None:
    module = StrictM5ResearchPlan()
    state = PipelineState(input_question="Q", top_hypotheses=[])

    with pytest.raises(RuntimeError, match="at least one top hypothesis"):
        await module(state)


# --- merged from test_entity_normalization.py ---

import pytest

from hypoforge.entity_normalization import EntityNormalizationService
from hypoforge.state import TaskContract, TaskEntity


class SameConceptClient:
    def __init__(self, same: bool = True) -> None:
        self.same = same
        self.calls = 0

    async def structured_chat(self, **kwargs):
        self.calls += 1
        request = __import__("json").loads(kwargs["user_prompt"])[0]
        return {"decisions": [{
            "surface": request["surface"],
            "canonical_name": request["candidates"][0],
            "same_concept": self.same,
            "rationale": "identity judgment",
        }]}


class FixedEmbeddings:
    async def aembed_documents(self, texts):
        # Every surface is recalled as a candidate. The LLM, not this score,
        # remains responsible for the merge decision.
        return [[1.0, 0.0] for _ in texts]


def contract() -> TaskContract:
    return TaskContract(entities=[TaskEntity(
        entity_id="E1",
        name="区域气候模型",
        aliases=["regional climate model"],
        role="primary_object",
        required=True,
    )])


@pytest.mark.asyncio
async def test_task_aliases_have_stable_cross_run_canonical_id(tmp_path) -> None:
    first = EntityNormalizationService.from_task(
        contract(), domains=["climate science"], cache_dir=tmp_path,
    )
    first_result = await first.resolve_batch([
        "区域气候模型", "regional climate model",
    ])
    second = EntityNormalizationService.from_task(
        contract(), domains=["climate science"], cache_dir=tmp_path,
    )
    second_result = await second.resolve_batch(["regional climate model"])

    assert first_result["区域气候模型"].canonical_id == (
        first_result["regional climate model"].canonical_id
    )
    assert second_result["regional climate model"].canonical_id == (
        first_result["regional climate model"].canonical_id
    )


@pytest.mark.asyncio
async def test_embedding_similarity_never_merges_without_identity_judgment(tmp_path) -> None:
    service = EntityNormalizationService.from_task(
        contract(),
        domains=["climate science"],
        cache_dir=tmp_path,
        embedding_backend=FixedEmbeddings(),
    )

    result = await service.resolve_batch(["global climate model"])

    assert result["global climate model"].canonical_name == "global climate model"
    assert result["global climate model"].canonical_name != "区域气候模型"


@pytest.mark.asyncio
async def test_llm_pair_decision_merges_and_is_cached(tmp_path) -> None:
    client = SameConceptClient(same=True)
    first = EntityNormalizationService.from_task(
        contract(),
        domains=["climate science"],
        cache_dir=tmp_path,
        client=client,
        embedding_backend=FixedEmbeddings(),
        run_stamp="run-B",
    )
    first_result = await first.resolve_batch(["RCM climate model"])
    assert first_result["RCM climate model"].canonical_name == "区域气候模型"
    assert client.calls == 1

    second_client = SameConceptClient(same=True)
    second = EntityNormalizationService.from_task(
        contract(),
        domains=["climate science"],
        cache_dir=tmp_path,
        client=second_client,
        embedding_backend=FixedEmbeddings(),
        run_stamp="run-B",
    )
    second_result = await second.resolve_batch(["RCM climate model"])

    assert second_result["RCM climate model"].canonical_id == (
        first_result["RCM climate model"].canonical_id
    )
    assert second_client.calls == 0


@pytest.mark.asyncio
async def test_negative_identity_decision_keeps_related_entities_separate(tmp_path) -> None:
    client = SameConceptClient(same=False)
    service = EntityNormalizationService.from_task(
        contract(),
        domains=["climate science"],
        cache_dir=tmp_path,
        client=client,
        embedding_backend=FixedEmbeddings(),
    )

    result = await service.resolve_batch(["regional rainfall bias"])

    assert result["regional rainfall bias"].canonical_name == "regional rainfall bias"
    assert client.calls == 1


# ---- fail-closed regression tests ----
class FailingEmbeddings:
    async def aembed_documents(self, texts):
        raise ConnectionError("simulated embedding API outage")


@pytest.mark.asyncio
async def test_configured_embedding_failure_propagates(tmp_path) -> None:
    """Embedding explicitly configured → failure must raise, not silently lexical."""
    service = EntityNormalizationService(
        cache_dir=tmp_path,
        embedding_backend=FailingEmbeddings(),
    )
    service.register_alias_group("known-entity", [])
    with pytest.raises(RuntimeError, match="Entity embedding failed"):
        await service.resolve_batch(["test surface"])


@pytest.mark.asyncio
async def test_embedding_not_configured_is_lexical_only(tmp_path) -> None:
    """No embedding configured → lexical-only is a legitimate mode."""
    service = EntityNormalizationService(cache_dir=tmp_path)
    result = await service.resolve_batch(["test surface"])
    assert "test surface" in result


@pytest.mark.asyncio
async def test_embedding_malformed_vector_count_propagates(tmp_path) -> None:
    """Embedding returns wrong count → must raise."""
    class WrongCountEmbeddings:
        async def aembed_documents(self, texts):
            return [[0.1, 0.2]]  # 1 vector for 2+ texts

    service = EntityNormalizationService(
        cache_dir=tmp_path,
        embedding_backend=WrongCountEmbeddings(),
    )
    service.register_alias_group("alpha", [])
    with pytest.raises(RuntimeError, match="Embedding backend returned"):
        await service.resolve_batch(["beta"])


class FailingJudgeClient:
    def __init__(self) -> None:
        self.calls = 0

    async def structured_chat(self, **kwargs):
        self.calls += 1
        raise ConnectionError("simulated judge API outage")


@pytest.mark.asyncio
async def test_judge_failure_propagates(tmp_path) -> None:
    """Configured judge fails → must raise."""
    service = EntityNormalizationService(
        cache_dir=tmp_path,
        client=FailingJudgeClient(),
        embedding_backend=FixedEmbeddings(),
    )
    service.register_alias_group("alpha", [])
    with pytest.raises(RuntimeError, match="Entity identity judge"):
        await service.resolve_batch(["beta"])


@pytest.mark.asyncio
async def test_judge_failure_writes_no_false_negative_cache(tmp_path) -> None:
    """Judge unavailable → must NOT write same_concept=False decisions."""
    service = EntityNormalizationService(
        cache_dir=tmp_path,
        client=None,
    )
    service.register_alias_group("robotic arm", ["robot manipulator"])
    await service.resolve_batch(["mechanical arm", "robot gripper"])
    # No client → no pair decisions should have been written
    # because the for loop skips when selected is None
    assert len(service.pair_decisions) == 0


# ==================== test_iteration_routing ====================

import sys
from pathlib import Path

import pytest

# Ensure HypoForge is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hypoforge.config import PipelineConfig
from hypoforge.pipeline import (
    PipelineRunner,
    _route_after_m1,
    _route_after_m6,
    _should_continue_iterating,
    build_followup_seed,
)
from hypoforge.state import (
    EvidenceGap,
    EvidenceGraph,
    EvidenceNode,
    EvidenceNodeType,
    EvidenceSufficiencyVerdict,
    FollowupRequest,
    GraphCorrectionRequest,
    PipelineState,
    ReviewResult,
    ReviewerDimension,
    RoutingDecision,
    make_gap_id,
)


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #

def _state_after_review(
    score: float = 3.0,
    iteration_count: int = 1,
    max_iterations: int = 3,
    threshold: float = 4.0,
    **extra,
) -> PipelineState:
    """A state as it looks right after M6 finished a review round."""
    return PipelineState(
        iteration_count=iteration_count,
        max_iterations=max_iterations,
        review_score_threshold=threshold,
        reviews=[
            ReviewResult(
                dimension=ReviewerDimension("overall"),
                score=score,
                version=iteration_count,
            )
        ],
        **extra,
    )


def _open_gap(
    sub_question: str = "Hsp70 如何识别底物？",
    gap_type: str = "mechanism",
    entities=("Hsp70",),
    description: str = "缺少 Hsp70 底物谱的直接蛋白质组证据",
) -> EvidenceGap:
    return EvidenceGap(
        description=description,
        target_sub_question=sub_question,
        gap_type=gap_type,
        canonical_entities=list(entities),
    )


def _insufficient_state(score: float = 3.0, search_round: int = 0, **extra) -> PipelineState:
    gap = _open_gap()
    return _state_after_review(
        score=score,
        search_round=search_round,
        evidence_verdict=EvidenceSufficiencyVerdict(sufficient=False, gaps=[gap]),
        evidence_gaps=[gap],
        **extra,
    )


def _nonempty_graph() -> EvidenceGraph:
    return EvidenceGraph(
        nodes=[EvidenceNode(id="n1", type=EvidenceNodeType.CLAIM, label="claim")],
        established_facts=["k1"],
    )


def _revisit_config(**overrides) -> PipelineConfig:
    return PipelineConfig(
        verbose=False,
        followup_routing=True,
        m6_evidence_revisit=True,
        **overrides,
    )


def _runner(switches_on: bool = True) -> PipelineRunner:
    config = (
        PipelineConfig(verbose=False, followup_routing=True, m6_evidence_revisit=True)
        if switches_on
        else PipelineConfig(verbose=False)
    )
    return PipelineRunner(config)


# --------------------------------------------------------------------------- #
# _route_after_m6 — strict four-priority branches × boundaries
# --------------------------------------------------------------------------- #

def test_route_after_m6_priority1_budget_exhausted_wins_over_everything():
    """Budget exhausted → end even when a supplement would otherwise qualify."""
    state = _insufficient_state(score=1.0, search_round=0)
    state.iteration_count = 3
    state.max_iterations = 3
    assert _route_after_m6(state, _revisit_config()) == "end"


def test_route_after_m6_priority2_supplement_beats_score_threshold():
    """Score already ≥ threshold, but evidence insufficient + open gap + search
    budget left → supplement (priority 2 fires BEFORE the threshold rule)."""
    state = _insufficient_state(score=4.9, search_round=1)
    assert _route_after_m6(state, _revisit_config(max_search_rounds=2)) == "supplement_m2"


def test_route_after_m6_priority2_requires_all_conditions():
    # no open gap
    gap = _open_gap()
    gap.status = "closed"
    state = _state_after_review(
        score=3.0,
        evidence_verdict=EvidenceSufficiencyVerdict(sufficient=False, gaps=[gap]),
        evidence_gaps=[gap],
        search_round=0,
    )
    assert _route_after_m6(state, _revisit_config()) == "revise_m4"

    # search-round budget exhausted
    state = _insufficient_state(score=3.0, search_round=2)
    assert _route_after_m6(state, _revisit_config(max_search_rounds=2)) == "revise_m4"

    # empty gap ledger
    state = _state_after_review(
        score=3.0,
        evidence_verdict=EvidenceSufficiencyVerdict(sufficient=False),
        search_round=0,
    )
    assert _route_after_m6(state, _revisit_config()) == "revise_m4"


def test_route_after_m6_priority3_sufficient_and_threshold_met_ends():
    state = _state_after_review(
        score=4.5,
        evidence_verdict=EvidenceSufficiencyVerdict(sufficient=True),
    )
    assert _route_after_m6(state, _revisit_config()) == "end"

    # verdict None while the revisit switch is off counts as sufficient
    state = _state_after_review(score=4.5)
    assert _route_after_m6(state, _revisit_config()) == "end"


def test_route_after_m6_priority4_insufficient_but_stuck_revises_m4():
    """Insufficient verdict but no supplement possible → revise (not end),
    even when the score clears the threshold."""
    state = _insufficient_state(score=4.9, search_round=2)
    assert _route_after_m6(state, _revisit_config(max_search_rounds=2)) == "revise_m4"


def test_route_after_m6_pending_graph_correction_revises_m3():
    state = _state_after_review(
        score=2.0,
        graph_correction_requests=[GraphCorrectionRequest(
            request_id="GCR_1",
            operation="remove_edge",
            source_node_id="N1",
            target_node_id="N2",
            evidence_ids=["EV1"],
            reason="The cited evidence does not support this relation.",
        )],
    )

    assert _route_after_m6(state, PipelineConfig(verbose=False)) == "revise_m3"


def test_route_after_m6_low_score_revises_m4():
    assert _route_after_m6(_state_after_review(score=3.0), _revisit_config()) == "revise_m4"


def test_route_after_m6_switch_off_verdict_is_ignored():
    """Switch off → the verdict never influences routing (legacy behaviour)."""
    state = _insufficient_state(score=3.0)
    # m6_evidence_revisit now defaults True — the test pins the switch OFF
    # explicitly to exercise the legacy verdict-ignored behaviour.
    switch_off = PipelineConfig(verbose=False, m6_evidence_revisit=False)
    assert _route_after_m6(state, switch_off) == "revise_m4"
    assert _route_after_m6(state, None) == "revise_m4"
    # high score + hostile verdict still ends when the switch is off
    high = _insufficient_state(score=4.9)
    assert _route_after_m6(high, switch_off) == "end"


# --------------------------------------------------------------------------- #
# _route_after_m1 — followup search-free triage
# --------------------------------------------------------------------------- #

def _followup_state(skip_search, graph=True) -> PipelineState:
    return PipelineState(
        input_question="追问",
        followup=FollowupRequest(text="追问", skip_search=skip_search),
        evidence_graph=_nonempty_graph() if graph else None,
    )


def test_route_after_m1_switch_off_always_searches():
    config = PipelineConfig(verbose=False)  # followup_routing=False
    state = _followup_state(skip_search=True)
    assert _route_after_m1(state, config) == "search_m2"
    assert _route_after_m1(state, None) == "search_m2"


def test_route_after_m1_skip_requires_all_conditions():
    assert _route_after_m1(_followup_state(True), _revisit_config()) == "direct_m4"


def test_route_after_m1_no_followup_searches():
    state = PipelineState(evidence_graph=_nonempty_graph())
    assert _route_after_m1(state, _revisit_config()) == "search_m2"


@pytest.mark.parametrize("skip_search", [False, None])
def test_route_after_m1_skip_search_not_true_searches(skip_search):
    assert _route_after_m1(_followup_state(skip_search), _revisit_config()) == "search_m2"


def test_route_after_m1_missing_or_empty_graph_searches():
    assert _route_after_m1(_followup_state(True, graph=False), _revisit_config()) == "search_m2"
    empty_graph_state = _followup_state(True)
    empty_graph_state.evidence_graph = EvidenceGraph()
    assert _route_after_m1(empty_graph_state, _revisit_config()) == "search_m2"


# --------------------------------------------------------------------------- #
# make_gap_id — (sub_question, gap_type, canonical_entities) contract
# --------------------------------------------------------------------------- #

def test_make_gap_id_same_triple_different_wording_same_id():
    a = make_gap_id("Hsp70 如何识别底物？", "mechanism", ["Hsp70"])
    b = make_gap_id(
        "Hsp70 如何识别底物？",
        "mechanism",
        ["Hsp70"],
    )
    # identical triple, totally different free-text descriptions upstream
    assert a == b
    assert len(a) == 12


def test_make_gap_id_entity_order_irrelevant():
    a = make_gap_id("q", "conflict", ["Hsp70", "BAG3"])
    b = make_gap_id("q", "conflict", ["BAG3", "Hsp70"])
    assert a == b


def test_make_gap_id_normalises_case_and_whitespace():
    a = make_gap_id("  How does HSP70  bind substrates? ", "MECHANISM", ["Hsp70 "])
    b = make_gap_id("how does hsp70 bind substrates?", "mechanism", ["hsp70"])
    assert a == b


def test_make_gap_id_comma_joined_composite_key_pinned():
    """The composite key is pinned to sha1(sub_q | type | comma-joined sorted
    entities)[:12] so no future refactor can silently change gap identity."""
    import hashlib

    raw = "hsp70 如何识别底物？|mechanism|bag3,hsp70"
    expected = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    assert make_gap_id("Hsp70 如何识别底物？", "mechanism", ["Hsp70", "BAG3"]) == expected


def test_make_gap_id_changes_with_any_component():
    base = make_gap_id("q", "mechanism", ["Hsp70"])
    assert base != make_gap_id("q2", "mechanism", ["Hsp70"])
    assert base != make_gap_id("q", "dosage", ["Hsp70"])
    assert base != make_gap_id("q", "mechanism", ["Hsp70", "BAG3"])


def test_evidence_gap_auto_id_uses_sub_question_or_description():
    gap = _open_gap()
    assert gap.gap_id == make_gap_id(gap.target_sub_question, gap.gap_type, gap.canonical_entities)
    assert gap.status == "open"
    assert gap.attempts == 0

    # missing target_sub_question → description anchors the hash, no crash
    no_sub = EvidenceGap(description="某种缺口", gap_type="coverage")
    assert no_sub.gap_id == make_gap_id("某种缺口", "coverage", [])
    assert no_sub.gap_type == "coverage"
    assert no_sub.canonical_entities == []


def test_evidence_gap_pending_grounding_status():
    gap = _open_gap()
    gap.status = "pending_grounding"
    assert gap.status == "pending_grounding"
    with pytest.raises(ValueError):
        EvidenceGap(description="x", status="mitigated_by_cache")  # removed enum value


# --------------------------------------------------------------------------- #
# Legacy compatibility
# --------------------------------------------------------------------------- #

def test_should_continue_iterating_wrapper_keeps_legacy_vocabulary():
    exhausted = _state_after_review(iteration_count=3, max_iterations=3)
    assert _should_continue_iterating(exhausted) == "end"

    high_score = _state_after_review(score=4.5, threshold=4.0)
    assert _should_continue_iterating(high_score) == "end"

    low_score = _state_after_review(score=3.0, threshold=4.0)
    assert _should_continue_iterating(low_score) == "iterate"


def test_switches_off_config_routing_is_equivalent_to_legacy():
    """With both switches explicitly off, the new routers must reproduce the
    legacy m6→{iterate,end} behaviour and the legacy linear m1→m2 edge.
    (m6_evidence_revisit now defaults True; the switch-off contract is
    tested explicitly.)"""
    config = PipelineConfig(verbose=False, m6_evidence_revisit=False)
    assert config.followup_routing is False
    assert config.m6_evidence_revisit is False

    legacy = {
        "exhausted": _state_after_review(iteration_count=3, max_iterations=3),
        "threshold": _state_after_review(score=4.9, threshold=4.5),
        "iterate": _state_after_review(score=2.5),
        # even hostile iteration-core state must not change behaviour:
        "hostile": _insufficient_state(score=2.5),
    }
    for name, state in legacy.items():
        expected = _should_continue_iterating(state)
        routed = _route_after_m6(state, config)
        assert routed == ("end" if expected == "end" else "revise_m4"), name
        # m1 always takes the search path with default config
        assert _route_after_m1(state, config) == "search_m2"


def test_legacy_checkpoint_without_new_fields_deserializes():
    """An old checkpoint/output JSON (no iteration-core fields) must load."""
    legacy_json = {
        "input_question": "蛋白质如何折叠？",
        "problem_card": {
            "original_question": "蛋白质如何折叠？",
            "domain": ["structural biology"],
            "sub_questions": ["q1", "q2"],
            "key_entities": ["Hsp70"],
            "question_type": "mechanism_explanation",
        },
        "reviews": [
            {"dimension": "overall", "score": 3.5, "version": 1}
        ],
        "iteration_count": 1,
        "max_iterations": 3,
        "review_score_threshold": 4.0,
        "run_id": "legacy-run",
    }
    state = PipelineState(**legacy_json)
    assert state.evidence_verdict is None
    assert state.evidence_gaps == []
    assert state.followup is None
    assert state.parent_run_id == ""
    assert state.search_round == 0
    assert state.revision_count == 0
    assert state.search_ledger.queries_issued == []
    assert state.routing_history == []
    assert state.iteration_count == 1
    assert state.reviews[0].score == 3.5


# --------------------------------------------------------------------------- #
# Graph topology — switch gating × routing_targets_available × module combos
# --------------------------------------------------------------------------- #

def _edges_of(config: PipelineConfig):
    runner = PipelineRunner(config)
    graph = runner._build_graph()
    return {(e.source, e.target) for e in graph.get_graph().edges}


def test_topology_switches_off_is_exactly_legacy_wiring():
    """HARD ACCEPTANCE: followup_routing=false and m6_evidence_revisit=false
    → conditional edges NOT installed; plain linear spine + legacy two-way
    m6 iteration edge (byte-for-byte the pre-refactor topology)."""
    edges = _edges_of(PipelineConfig(
        verbose=False, followup_routing=False, m6_evidence_revisit=False,
    ))

    for src, dst in [("m1", "m2"), ("m2", "m3"), ("m3", "m4"), ("m4", "m5"), ("m5", "m6")]:
        assert (src, dst) in edges, f"missing linear edge {src}→{dst}"
    assert ("m1", "m4") not in edges  # no followup shortcut
    assert ("m6", "m2") not in edges  # no supplement re-hop
    assert ("m6", "m4") in edges       # legacy two-way iteration edge
    assert ("m6", "__end__") in edges


def test_topology_switches_on_full_modules_has_routing_edges():
    config = PipelineConfig(
        verbose=False, followup_routing=True, m6_evidence_revisit=True
    )
    edges = _edges_of(config)

    # linear spine (m1→m2 replaced by the conditional edge)
    for src, dst in [("m2", "m3"), ("m3", "m4"), ("m4", "m5"), ("m5", "m6")]:
        assert (src, dst) in edges, f"missing linear edge {src}→{dst}"

    # m1 routing-aware edge: search_m2→m2, direct_m4→m4 (replaces the linear edge)
    assert ("m1", "m2") in edges
    assert ("m1", "m4") in edges

    # m6 three-way edge (replaces the legacy two-way edge)
    assert ("m6", "__end__") in edges
    assert ("m6", "m4") in edges
    assert ("m6", "m2") in edges


def test_topology_without_m2_falls_back_to_legacy_wiring():
    config = PipelineConfig(
        verbose=False,
        enabled_modules=["m1", "m3", "m4", "m5", "m6"],
        followup_routing=True,
        m6_evidence_revisit=True,
    )
    edges = _edges_of(config)

    # routing targets unavailable → linear m1→m3 kept, no skip_search shortcut
    assert ("m1", "m3") in edges
    assert ("m1", "m2") not in edges
    assert ("m1", "m4") not in edges

    # m6 keeps the legacy two-way conditional edge (iterate→m4 / end)
    assert ("m6", "m4") in edges
    assert ("m6", "__end__") in edges
    assert ("m6", "m2") not in edges


def test_topology_without_m4_falls_back_to_legacy_wiring():
    config = PipelineConfig(
        verbose=False,
        enabled_modules=["m1", "m2", "m3", "m5", "m6"],
        followup_routing=True,
        m6_evidence_revisit=True,
        # m4 is absent, so the legacy iteration edge must target an enabled
        # module explicitly (default iteration_module_target="m4" would be
        # invalid with or without the iteration-core switches).
        iteration_module_target="m3",
    )
    edges = _edges_of(config)

    assert ("m1", "m2") in edges   # plain linear edge retained
    assert ("m1", "m4") not in edges
    assert ("m6", "m2") not in edges
    # legacy two-way iteration edge: iterate→m3 / end
    assert ("m6", "m3") in edges
    assert ("m6", "__end__") in edges


def test_topology_without_m1_keeps_m6_routing_edge():
    """m1 absent → entry point is m2; the m6 three-way edge still applies."""
    config = PipelineConfig(
        verbose=False,
        enabled_modules=["m2", "m3", "m4", "m5", "m6"],
        followup_routing=True,
        m6_evidence_revisit=True,
    )
    edges = _edges_of(config)

    assert not any(src == "m1" for src, _ in edges)
    for src, dst in [("m2", "m3"), ("m3", "m4"), ("m4", "m5"), ("m5", "m6")]:
        assert (src, dst) in edges
    assert ("m6", "m4") in edges
    assert ("m6", "m2") in edges
    assert ("m6", "__end__") in edges


def test_topology_switches_on_but_targets_missing_is_legacy():
    """Only one of m2/m4 present + switches on → legacy wiring everywhere."""
    config = PipelineConfig(
        verbose=False,
        enabled_modules=["m1", "m2", "m3"],
        followup_routing=True,
        m6_evidence_revisit=True,
        enable_iteration=False,
    )
    edges = _edges_of(config)
    assert ("m1", "m2") in edges
    assert ("m1", "m4") not in edges
    assert ("m3", "__end__") in edges


def test_edge_callbacks_match_pure_routers():
    runner = _runner(switches_on=True)
    state = _insufficient_state(score=3.0, search_round=1)
    assert runner._decide_after_m6(state) == _route_after_m6(state, runner.config)
    followup = _followup_state(skip_search=True)
    assert runner._decide_after_m1(followup) == _route_after_m1(followup, runner.config)


# --------------------------------------------------------------------------- #
# Routing bookkeeping — round counter + revision_count
# --------------------------------------------------------------------------- #

def test_routing_decision_round_is_history_length_plus_one():
    runner = _runner()
    state = _state_after_review(score=3.0)
    assert runner._routing_decision("m6", state).round == 1

    state.routing_history.append(
        RoutingDecision(round=1, from_module="m1", to_module="search_m2", decided_by="policy")
    )
    assert runner._routing_decision("m6", state).round == 2


def test_routing_decision_fields_for_m6_supplement():
    runner = _runner()
    state = _insufficient_state(score=3.0, search_round=1)
    decision = runner._routing_decision("m6", state)
    assert decision.from_module == "m6"
    assert decision.to_module == "supplement_m2"
    assert decision.decided_by == "m6"
    assert decision.gap_ids == [g.gap_id for g in state.evidence_gaps if g.status == "open"]


def test_routing_decision_fields_for_m1():
    runner = _runner()
    decision = runner._routing_decision("m1", _followup_state(skip_search=True))
    assert decision.from_module == "m1"
    assert decision.to_module == "direct_m4"
    assert decision.decided_by == "m1"

    decision = runner._routing_decision("m1", PipelineState())
    assert decision.to_module == "search_m2"
    assert decision.decided_by == "policy"


def test_append_routing_extends_history_and_bumps_revision_count():
    runner = _runner()
    state = _state_after_review(score=3.0)  # low score → revise_m4
    patch: dict = {}
    runner._append_routing("m6", state, patch)
    assert len(patch["routing_history"]) == 1
    assert patch["routing_history"][0].to_module == "revise_m4"
    assert patch["revision_count"] == 1  # revise_m4 bumps the audit counter

    # routing to end does NOT bump revision_count
    end_state = _state_after_review(score=4.9, iteration_count=3, max_iterations=3)
    patch2: dict = {}
    runner._append_routing("m6", end_state, patch2)
    assert patch2["routing_history"][0].to_module == "end"
    assert "revision_count" not in patch2

    # non-routing modules never touch the patch
    patch3: dict = {}
    runner._append_routing("m3", state, patch3)
    assert patch3 == {}


def test_append_routing_direct_m4_emits_module_skipped_for_m2_m3():
    """The conditional edge bypasses M2/M3, so the bookkeeping step must emit
    the explicit module_skipped events for both (when enabled)."""

    class _StubRecorder:
        def __init__(self):
            self.events = []

        def emit(self, event_type, **kwargs):
            self.events.append((event_type, kwargs))

    recorder = _StubRecorder()
    runner = _runner()
    runner.event_recorder = recorder
    patch: dict = {}
    runner._append_routing("m1", _followup_state(skip_search=True), patch)

    assert patch["routing_history"][0].to_module == "direct_m4"
    skipped = [
        kwargs["module"]
        for event_type, kwargs in recorder.events
        if event_type == "module_skipped"
    ]
    assert skipped == ["m2", "m3"]

    # the normal search path never emits module_skipped
    recorder.events.clear()
    patch2: dict = {}
    runner._append_routing("m1", PipelineState(), patch2)
    assert patch2["routing_history"][0].to_module == "search_m2"
    assert not [e for e in recorder.events if e[0] == "module_skipped"]


# --------------------------------------------------------------------------- #
# Routing-aware skip rules
# --------------------------------------------------------------------------- #

def test_skip_module_default_semantics_unchanged():
    runner = _runner(switches_on=False)
    done = PipelineState(literature_results=[{"sub_question": "q"}])
    # m2 done, no followup, no supplement route → skip (legacy semantics)
    assert runner._should_skip_module("m2", {"literature_results"}, done) is True
    # not done → run
    assert runner._should_skip_module("m2", {"literature_results"}, PipelineState()) is False
    # m4 inside iteration loop with budget left → never skip
    m4_done = PipelineState(top_hypotheses=[{"hypothesis_id": "h", "statement": "s"}])
    assert runner._should_skip_module("m4", {"top_hypotheses"}, m4_done) is False
    m4_done.iteration_count = m4_done.max_iterations
    assert runner._should_skip_module("m4", {"top_hypotheses"}, m4_done) is True


def test_skip_module_supplement_round_forces_m2_m3():
    runner = _runner()
    state = PipelineState(
        literature_results=[{"sub_question": "q"}],
        evidence_graph=_nonempty_graph(),
        routing_history=[
            RoutingDecision(round=1, from_module="m6", to_module="supplement_m2", decided_by="m6")
        ],
    )
    assert runner._should_skip_module("m2", {"literature_results"}, state) is False
    assert runner._should_skip_module("m3", {"evidence_graph"}, state) is False

    # a later m6→revise decision clears the supplement context
    state.routing_history.append(
        RoutingDecision(round=2, from_module="m6", to_module="revise_m4", decided_by="policy")
    )
    assert runner._should_skip_module("m2", {"literature_results"}, state) is True


def test_skip_module_followup_skip_search_skips_m2_m3():
    runner = _runner()
    state = PipelineState(
        literature_results=[{"sub_question": "q"}],
        evidence_graph=_nonempty_graph(),
        followup=FollowupRequest(text="追问", skip_search=True),
    )
    assert runner._should_skip_module("m2", {"literature_results"}, state) is True
    assert runner._should_skip_module("m3", {"evidence_graph"}, state) is True


def test_skip_module_followup_search_reruns_m2_m3():
    runner = _runner()
    for skip in (False, None):
        state = PipelineState(
            literature_results=[{"sub_question": "q"}],
            evidence_graph=_nonempty_graph(),
            followup=FollowupRequest(text="追问", skip_search=skip),
        )
        assert runner._should_skip_module("m2", {"literature_results"}, state) is False
        assert runner._should_skip_module("m3", {"evidence_graph"}, state) is False


# --- M1 followup exemption (task #16 fix 1) ---

def _m1_done_state(**extra) -> PipelineState:
    """A state whose M1 output field (problem_card) is already populated —
    exactly what ``build_followup_seed`` produces (whitelist inheritance)."""
    return PipelineState(
        problem_card={
            "original_question": "原始问题",
            "domain": ["proteostasis"],
            "sub_questions": ["q1"],
            "key_entities": ["Hsp70"],
            "question_type": "mechanism_explanation",
        },
        **extra,
    )


def test_skip_module_followup_forces_m1_rerun_when_routing_enabled():
    """followup + followup_routing on ⇒ M1 must re-run its triage even though
    the seed inherited a completed problem_card."""
    runner = _runner()  # followup_routing=True
    state = _m1_done_state(
        followup=FollowupRequest(text="给我中文方案", parent_run_id="p1"),
        evidence_graph=_nonempty_graph(),
    )
    assert runner._should_skip_module("m1", {"problem_card"}, state) is False


def test_skip_module_followup_m1_resume_after_triage_skips():
    """Checkpoint resume AFTER triage already ran this run (routing_history
    carries an m1 decision) ⇒ skip again, no duplicate triage."""
    runner = _runner()
    state = _m1_done_state(
        followup=FollowupRequest(text="给我中文方案", skip_search=True, parent_run_id="p1"),
        evidence_graph=_nonempty_graph(),
        routing_history=[
            RoutingDecision(round=1, from_module="m1", to_module="direct_m4", decided_by="m1")
        ],
    )
    assert runner._should_skip_module("m1", {"problem_card"}, state) is True


def test_skip_module_m1_unchanged_without_followup():
    """Non-followup runs keep the legacy done ⇒ skip semantics (switches on
    or off), and an incomplete M1 still runs."""
    for switches in (True, False):
        runner = _runner(switches_on=switches)
        assert runner._should_skip_module("m1", {"problem_card"}, _m1_done_state()) is True
    assert runner._should_skip_module("m1", {"problem_card"}, PipelineState()) is False


def test_skip_module_followup_m1_skips_when_routing_disabled():
    """followup_routing off ⇒ the exemption never fires (legacy behaviour)."""
    runner = _runner(switches_on=False)
    state = _m1_done_state(
        followup=FollowupRequest(text="给我中文方案", parent_run_id="p1"),
        evidence_graph=_nonempty_graph(),
    )
    assert runner._should_skip_module("m1", {"problem_card"}, state) is True


# --------------------------------------------------------------------------- #
# M6 gap merge semantics
# --------------------------------------------------------------------------- #

def test_m6_merge_gaps_hit_inherits_status_and_counts_attempt():
    from hypoforge.modules.m6_review_iteration import M6ReviewIteration

    existing = _open_gap(description="第一版描述")
    existing.source_review_version = 1
    existing.status = "pending_grounding"

    # same identity triple, different wording → same gap_id
    again = _open_gap(description="完全换一种说法")
    assert again.gap_id == existing.gap_id

    merged = M6ReviewIteration._merge_evidence_gaps([existing], [again], version=2)
    assert len(merged) == 1
    assert merged[0].status == "pending_grounding"  # inherited, not reset
    assert merged[0].attempts == 1  # survived one more review round
    assert merged[0].source_review_version == 2


def test_m6_merge_gaps_new_gap_starts_open_and_closed_never_reopens():
    from hypoforge.modules.m6_review_iteration import M6ReviewIteration

    closed_gap = _open_gap(description="已解决的缺口")
    closed_gap.status = "closed"
    same_id_new = _open_gap(description="已解决的缺口!!!")
    assert same_id_new.gap_id == closed_gap.gap_id

    fresh = _open_gap(sub_question="别的子问题", entities=("BAG3",), description="新缺口")
    merged = M6ReviewIteration._merge_evidence_gaps(
        [closed_gap], [same_id_new, fresh], version=2
    )
    by_id = {g.gap_id: g for g in merged}
    assert len(merged) == 2
    assert by_id[closed_gap.gap_id].status == "closed"  # not re-opened
    assert by_id[fresh.gap_id].status == "open"
    assert by_id[fresh.gap_id].attempts == 0


def test_m6_merge_gaps_disappeared_open_gap_closed_when_sufficient():
    from hypoforge.modules.m6_review_iteration import M6ReviewIteration

    vanished = _open_gap(description="本轮没再出现的缺口")
    still_there = _open_gap(sub_question="另一个子问题", entities=("X",), description="仍在")

    # verdict sufficient, only `still_there` re-reported → `vanished` closes
    merged = M6ReviewIteration._merge_evidence_gaps(
        [vanished, still_there], [still_there], version=3, sufficient=True
    )
    by_id = {g.gap_id: g for g in merged}
    assert by_id[vanished.gap_id].status == "closed"
    assert by_id[still_there.gap_id].status == "open"

    # verdict NOT sufficient → disappeared gap stays open (maybe transient)
    merged = M6ReviewIteration._merge_evidence_gaps(
        [vanished, still_there], [still_there], version=3, sufficient=False
    )
    by_id = {g.gap_id: g for g in merged}
    assert by_id[vanished.gap_id].status == "open"


# --------------------------------------------------------------------------- #
# build_followup_seed — whitelist inheritance / reset semantics
# --------------------------------------------------------------------------- #

def _dirty_parent_final_state() -> dict:
    """A realistic final-state JSON dump: whitelist fields + junk keys."""
    parent = PipelineState(
        input_question="原始问题",
        run_id="parent-1",
        iteration_count=3,
        max_iterations=3,
        revision_count=2,
        search_round=2,
        problem_card={
            "original_question": "原始问题",
            "domain": ["proteostasis"],
            "sub_questions": ["q1"],
            "key_entities": ["Hsp70"],
            "question_type": "mechanism_explanation",
        },
        literature_results=[{"sub_question": "q1", "papers_retrieved": 5}],
        evidence_graph=_nonempty_graph(),
        best_hypotheses=[{"hypothesis_id": "best-1", "statement": "保留"}],
        top_hypotheses=[{"hypothesis_id": "top-1", "statement": "应被重置"}],
        candidate_hypotheses=[{"hypothesis_id": "c1", "statement": "应被重置"}],
        research_plans=[{"hypothesis_id": "top-1", "study_subjects": "mice"}],
        reviews=[ReviewResult(dimension=ReviewerDimension("overall"), score=3.0, version=1)],
        routing_history=[
            RoutingDecision(round=1, from_module="m6", to_module="revise_m4", decided_by="policy")
        ],
        evidence_gaps=[_open_gap()],
        evidence_verdict=EvidenceSufficiencyVerdict(sufficient=False),
        errors=["some old error"],
        metrics={"novelty": 0.4},
        total_input_tokens=1234,
        total_output_tokens=567,
        memory_cache_dir="cache-dir",
    )
    dump = parent.model_dump(mode="json")
    # checkpoints carry junk keys that must NOT leak into the seed state
    dump["_last_module"] = "m6"
    dump["some_future_unknown_key"] = {"weird": True}
    return dump


def test_build_followup_seed_inherits_only_whitelist_fields():
    config = PipelineConfig(verbose=False, memory_cache_dir="default-cache")
    state = build_followup_seed(
        _dirty_parent_final_state(), followup_text="追问？", run_id="follow-1", config=config
    )

    # --- newly set ---
    assert state.input_question == "追问？"
    assert state.run_id == "follow-1"
    assert state.parent_run_id == "parent-1"
    assert state.followup == FollowupRequest(text="追问？", parent_run_id="parent-1")
    assert state.max_iterations == config.max_iterations  # fresh budget
    assert state.max_evidence_gap_rounds == config.max_evidence_gap_rounds

    # --- inherited (whitelist) ---
    assert state.problem_card is not None and state.problem_card.key_entities == ["Hsp70"]
    assert state.literature_results[0].sub_question == "q1"
    assert state.evidence_graph is not None
    assert state.best_hypotheses and state.best_hypotheses[0].hypothesis_id == "best-1"
    assert state.memory_cache_dir == "cache-dir"

    # --- reset to defaults ---
    assert state.iteration_count == 0
    assert state.revision_count == 0
    assert state.search_round == 0
    assert state.reviews == []
    assert state.candidate_hypotheses == []
    assert state.top_hypotheses == []
    assert state.research_plans == []
    assert state.errors == []
    assert state.metrics == {}
    assert state.total_input_tokens == 0
    assert state.total_output_tokens == 0
    assert state.routing_history == []
    assert state.evidence_verdict is None
    assert state.evidence_gaps == []


def test_build_followup_seed_memory_cache_falls_back_to_config():
    config = PipelineConfig(verbose=False, memory_cache_dir="default-cache")
    seed = {"run_id": "p", "memory_cache_dir": ""}
    state = build_followup_seed(seed, followup_text="追问", run_id="f", config=config)
    assert state.memory_cache_dir == "default-cache"
    assert state.parent_run_id == "p"


# --------------------------------------------------------------------------- #
# Config switches
# --------------------------------------------------------------------------- #

def test_iteration_core_switches_defaults():
    config = PipelineConfig.from_defaults()
    assert config.followup_routing is False
    # flipped to True by ba60c21 (prevent futile M4 iterations)
    assert config.m6_evidence_revisit is True
    assert config.max_search_rounds == 2
    assert config.supplement_paper_budget == 6
    assert config.gap_no_improvement_limit == 3
    assert config.gap_no_gain_limit == 3
    assert config.max_plan_revisions == 2


def test_iteration_core_switches_load_from_yaml(tmp_path):
    yaml_path = tmp_path / "iteration_core.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                "followup_routing: true",
                "m6_evidence_revisit: true",
                "max_search_rounds: 3",
                "supplement_paper_budget: 8",
                "gap_no_improvement_limit: 2",
            ]
        ),
        encoding="utf-8",
    )
    config = PipelineConfig.from_yaml(str(yaml_path))
    assert config.followup_routing is True
    assert config.m6_evidence_revisit is True
    assert config.max_search_rounds == 3
    assert config.supplement_paper_budget == 8
    assert config.gap_no_improvement_limit == 2


# ==================== test_p2_supplement_loop ====================

import pytest

from hypoforge.config import PipelineConfig
from hypoforge.memory import PaperStore
from hypoforge.memory.paper_store import query_hash
from hypoforge.modules.m6_review_iteration import M6ReviewIteration
from hypoforge.observability import RunEventRecorder, bind_recorder
from hypoforge.paper_sources import (
    dedupe_papers_by_key,
    dedupe_queries,
    gap_candidate_queries,
    merge_literature_increment,
    open_paper_store,
    persist_search_results,
    resolve_gap_sub_question,
)
from hypoforge.pipeline import PipelineRunner
from hypoforge.state import (
    EvidenceGap,
    EvidenceGraph,
    EvidenceNode,
    EvidenceNodeType,
    EvidenceSufficiencyVerdict,
    HypothesisCard,
    KnowledgeEntry,
    KnowledgeEntryType,
    LiteratureResult,
    PipelineState,
    ProblemCard,
    ResearchPlan,
    RoutingDecision,
    SearchLedger,
    make_gap_id,
)


SUB_QUESTION = "What is the Hsp70 mechanism?"


# ---------------------------------------------------------------------------
# Local fakes
# ---------------------------------------------------------------------------


class FakeM6Client:
    """Replays scripted ``structured_chat`` payloads, then raises."""

    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = 0

    async def structured_chat(self, **kwargs):
        self.calls += 1
        if not self.payloads:
            raise RuntimeError("no more scripted payloads")
        payload = self.payloads.pop(0)
        if isinstance(payload, Exception):
            raise payload
        return payload


def _review_payload() -> dict:
    return {
        "reasoning": "sound",
        "score": 4.0,
        "comments": "ok",
        "suggestions": "none",
    }


def make_m6_state(**overrides) -> PipelineState:
    base = dict(
        input_question="Q",
        problem_card=ProblemCard(original_question="Q", sub_questions=[SUB_QUESTION], key_entities=[], domain=[]),
        top_hypotheses=[HypothesisCard(hypothesis_id="h1", statement="s")],
        research_plans=[ResearchPlan(hypothesis_id="h1")],
    )
    base.update(overrides)
    return PipelineState(**base)


def make_m6_module(client, revisit: bool = True, limit: int = 3) -> M6ReviewIteration:
    module = M6ReviewIteration(
        mode="llm", llm_config=None,
        m6_evidence_revisit=revisit, gap_no_gain_limit=limit,
    )
    module.client = client
    return module


# ---------------------------------------------------------------------------
# M6 verdict: switch off = zero behaviour change (hard acceptance)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_m6_switch_off_never_calls_verdict() -> None:
    # Exactly two reviewer payloads; a 3rd call would raise — proving the
    # verdict call is never issued when the switch is off.
    client = FakeM6Client([_review_payload()] * 2)
    module = make_m6_module(client, revisit=False)
    state = make_m6_state()

    patch = await module(state)

    assert client.calls == 2
    assert set(patch) == {
        "reviews", "iteration_count", "graph_correction_requests",
    }  # no evidence-verdict keys
    assert patch["iteration_count"] == 1
    # alignment gate + 2 specialists + 3 objective gates + 3 metric dims + overall
    assert len(patch["reviews"]) == 10
    assert {r.dimension.value for r in patch["reviews"]} == {
        "task_alignment", "scientific_logic", "method_feasibility",
        "evidence_coverage_gate", "answer_completeness_gate",
        "source_quality_gate", "testability_metric", "novelty_metric",
        "objective_evidence_consistency", "overall",
    }


# ---------------------------------------------------------------------------
# M6 verdict: happy path + gap merge + fail-closed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_m6_verdict_matches_existing_gap_inherits_attempts() -> None:
    gap_id = make_gap_id(SUB_QUESTION, "mechanism", ["hsp70"])
    existing = EvidenceGap(
        description="Missing co-chaperone data",
        gap_type="mechanism",
        canonical_entities=["hsp70"],
        target_sub_question=SUB_QUESTION,
        status="pending_grounding",
        attempts=2,
    )
    assert existing.gap_id == gap_id
    verdict_payload = {
        "sufficient": False,
        "gaps": [{
            "description": "Still missing co-chaperone data",
            "gap_type": "mechanism",
            "canonical_entities": ["Hsp70"],
            "target_sub_question": SUB_QUESTION,
            "suggested_queries": ["Hsp70 co-chaperone binding assay"],
        }],
    }
    client = FakeM6Client([_review_payload()] * 2 + [verdict_payload])
    module = make_m6_module(client)
    state = make_m6_state(evidence_gaps=[existing])

    patch = await module(state)

    assert client.calls == 3  # 2 reviewers + 1 verdict
    assert patch["evidence_verdict"].sufficient is False
    gaps = patch["evidence_gaps"]
    assert len(gaps) == 1
    assert gaps[0].gap_id == gap_id
    assert gaps[0].status == "pending_grounding"  # inherited
    assert gaps[0].attempts == 3  # survived one more review round
    assert gaps[0].suggested_queries == ["Hsp70 co-chaperone binding assay"]
    assert patch["metrics"]["m6_zero_gain_streak"] == 0  # no gain map → reset


@pytest.mark.asyncio
async def test_m6_verdict_fail_closed() -> None:
    client = FakeM6Client([_review_payload()] * 2 + [RuntimeError("boom")])
    module = make_m6_module(client)
    state = make_m6_state()

    patch = await module(state)

    assert client.calls == 3  # the verdict call was attempted
    verdict = patch["evidence_verdict"]
    assert verdict.sufficient is False
    assert len(verdict.gaps) == 1
    assert verdict.gaps[0].gap_type == "coverage"
    assert patch["evidence_gaps"][0].status == "open"


# ---------------------------------------------------------------------------
# _merge_evidence_gaps semantics
# ---------------------------------------------------------------------------


def test_merge_gaps_disappeared_open_gap_closes_on_sufficient_or_gain() -> None:
    gap = EvidenceGap(
        description="gone this round",
        target_sub_question=SUB_QUESTION,
        status="open",
    )

    # sufficient verdict closes it
    merged = M6ReviewIteration._merge_evidence_gaps([gap], [], 2, sufficient=True)
    assert merged[0].status == "closed"

    # insufficient verdict but M3 gain > 0 also closes it
    merged = M6ReviewIteration._merge_evidence_gaps(
        [gap], [], 2, sufficient=False, gap_gain={gap.gap_id: 2}
    )
    assert merged[0].status == "closed"

    # neither sufficient nor gained → stays open
    merged = M6ReviewIteration._merge_evidence_gaps(
        [gap], [], 2, sufficient=False, gap_gain={gap.gap_id: 0}
    )
    assert merged[0].status == "open"


def test_merge_gaps_resolved_statuses_never_reopen() -> None:
    closed = EvidenceGap(
        description="done", target_sub_question=SUB_QUESTION, status="closed"
    )
    unimprovable = EvidenceGap(
        description="stuck",
        target_sub_question="other question",
        status="unimprovable",
    )
    re_reported = [
        closed.model_copy(update={"status": "open"}),
        unimprovable.model_copy(update={"status": "open"}),
    ]
    merged = M6ReviewIteration._merge_evidence_gaps(
        [closed, unimprovable], re_reported, 3, sufficient=False
    )
    by_id = {g.gap_id: g for g in merged}
    assert by_id[closed.gap_id].status == "closed"
    assert by_id[unimprovable.gap_id].status == "unimprovable"


def test_merge_gaps_new_gap_recorded_open() -> None:
    incoming = EvidenceGap(description="fresh", target_sub_question=SUB_QUESTION)
    merged = M6ReviewIteration._merge_evidence_gaps([], [incoming], 1)
    assert len(merged) == 1
    assert merged[0].status == "open"
    assert merged[0].attempts == 0


# ---------------------------------------------------------------------------
# Zero-gain convergence protection
# ---------------------------------------------------------------------------


def test_zero_gain_streak_accumulates_and_marks_unimprovable() -> None:
    module = M6ReviewIteration(mode="llm", llm_config=None, gap_no_gain_limit=2)
    gap = EvidenceGap(
        description="stuck", target_sub_question=SUB_QUESTION, status="open"
    )

    # round 1: all-zero gain → streak 1 (below limit)
    state = make_m6_state(
        metrics={"m3_gap_gain": {gap.gap_id: 0}, "m6_zero_gain_streak": 0}
    )
    gaps, metrics = module._apply_zero_gain_convergence(state, [gap])
    assert metrics["m6_zero_gain_streak"] == 1
    assert gaps[0].status == "open"

    # round 2: still zero → streak 2 == limit → unimprovable
    state2 = make_m6_state(
        metrics={"m3_gap_gain": {gap.gap_id: 0}, "m6_zero_gain_streak": 1}
    )
    gaps2, metrics2 = module._apply_zero_gain_convergence(state2, [gap])
    assert metrics2["m6_zero_gain_streak"] == 2
    assert gaps2[0].status == "unimprovable"


def test_zero_gain_streak_resets_on_positive_gain() -> None:
    module = M6ReviewIteration(mode="llm", llm_config=None, gap_no_gain_limit=2)
    gap = EvidenceGap(
        description="stuck", target_sub_question=SUB_QUESTION, status="open"
    )
    state = make_m6_state(
        metrics={"m3_gap_gain": {gap.gap_id: 1}, "m6_zero_gain_streak": 1}
    )
    gaps, metrics = module._apply_zero_gain_convergence(state, [gap])
    assert metrics["m6_zero_gain_streak"] == 0
    assert gaps[0].status == "open"


# ---------------------------------------------------------------------------
# PaperStore contract aliases + provenance
# ---------------------------------------------------------------------------


def test_paper_store_contract_aliases(tmp_path) -> None:
    store = PaperStore(tmp_path)
    keys = store.upsert_papers([{"doi": "10.1/x", "title": "T"}])
    digest = store.record_query("Hsp70 chaperone", keys)
    assert digest == query_hash("Hsp70 chaperone")

    # lookup_by_query accepts raw text...
    assert store.lookup_by_query("hsp70  chaperone!!") == keys
    # ...and a precomputed 12-hex hash
    assert store.lookup_by_query(digest) == keys
    # unknown text → []
    assert store.lookup_by_query("totally different") == []

    # papers_by_ids is the lookup_by_keys alias
    assert store.papers_by_ids(keys) == store.lookup_by_keys(keys)
    assert store.papers_by_ids(keys)[0]["doi"] == "10.1/x"


def test_paper_store_source_provenance(tmp_path) -> None:
    store = PaperStore(tmp_path)
    store.upsert_papers([{"doi": "10.1/x"}], source="agentic_m2")
    store.upsert_papers([{"doi": "10.1/x"}], source="agentic_m2")
    store._ensure_loaded()
    record = store._papers["doi:10.1/x"]
    assert record["sources_seen"] == ["agentic_m2"]


# ---------------------------------------------------------------------------
# paper_sources junction helpers
# ---------------------------------------------------------------------------


def test_open_paper_store_degrades_gracefully() -> None:
    assert open_paper_store("") is None


def test_persist_search_results_never_raises_and_records() -> None:
    assert persist_search_results(None, [{"doi": "10.1/x"}], ["q"]) == []
    import tempfile

    with tempfile.TemporaryDirectory() as cache_dir:
        store = PaperStore(cache_dir)
        keys = persist_search_results(
            store, [{"doi": "10.1/x"}], ["hsp70 query"], source="agentic_m2"
        )
        assert keys == ["doi:10.1/x"]
        assert store.lookup_query("hsp70 query") == keys


def test_dedupe_papers_by_key_preserves_order_and_mutates_known() -> None:
    known = {"doi:10.1/old"}
    papers = [{"doi": "10.1/a"}, {"doi": "10.1/old"}, {"doi": "10.1/a"}]
    fresh, fresh_keys = dedupe_papers_by_key(papers, known)
    assert fresh_keys == ["doi:10.1/a"]
    assert fresh == [{"doi": "10.1/a"}]
    assert "doi:10.1/a" in known


def test_dedupe_queries_normalised_and_mutates_ledger_set() -> None:
    issued = {"hsp70 chaperone"}
    fresh = dedupe_queries(["Hsp70  chaperone!!", "new query text"], issued)
    assert fresh == ["new query text"]
    assert "new query text" in issued


def test_gap_candidate_queries_prefers_suggested() -> None:
    gap = EvidenceGap(
        description="some long description text",
        suggested_queries=["q1", "  q2  ", "", "q3", "q4"],
    )
    assert gap_candidate_queries(gap) == ["q1", "q2", "q3"]

    no_suggested = EvidenceGap(
        description="  Missing   co-chaperone data ",
        canonical_entities=["Hsp70", "Bag1"],
    )
    assert gap_candidate_queries(no_suggested) == [
        "Missing co-chaperone data",
        "Hsp70 Bag1",
    ]


def test_merge_literature_increment_extends_and_never_clears() -> None:
    old_entry = KnowledgeEntry(
        id="KE_old",
        type=KnowledgeEntryType.ESTABLISHED_FACT,
        content="old",
        source_paper_id="p0",
    )
    results = [
        LiteratureResult(
            sub_question=SUB_QUESTION,
            papers_retrieved=1,
            knowledge_entries=[old_entry],
        )
    ]
    new_entry = KnowledgeEntry(
        id="KE_new",
        type=KnowledgeEntryType.ESTABLISHED_FACT,
        content="new",
        source_paper_id="p1",
    )
    merge_literature_increment(results, SUB_QUESTION, 2, [old_entry, new_entry])
    assert results[0].papers_retrieved == 3
    assert [e.id for e in results[0].knowledge_entries] == ["KE_old", "KE_new"]

    # unseen sub-question appends a fresh result
    merge_literature_increment(results, "brand new question", 1, [])
    assert len(results) == 2
    assert results[1].sub_question == "brand new question"


def test_resolve_gap_sub_question() -> None:
    explicit = EvidenceGap(
        description="x", target_sub_question=" explicit target "
    )
    assert resolve_gap_sub_question(explicit, ["other"]) == "explicit target"

    fuzzy = EvidenceGap(description="Hsp70 binding affinity data")
    assert (
        resolve_gap_sub_question(
            fuzzy, ["unrelated topic", "Hsp70 binding mechanism"]
        )
        == "Hsp70 binding mechanism"
    )


# ---------------------------------------------------------------------------
# Legacy M2 supplement flow
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Pipeline supplement-round skip-suppression event
# ---------------------------------------------------------------------------


def test_pipeline_supplement_round_emits_skip_suppressed(tmp_path) -> None:
    runner = PipelineRunner(PipelineConfig())
    recorder = RunEventRecorder(tmp_path / "run", "run-1")
    runner.event_recorder = recorder

    state = PipelineState(
        literature_results=[{"sub_question": "q"}],
        evidence_graph=EvidenceGraph(
            nodes=[EvidenceNode(id="n1", type=EvidenceNodeType.CLAIM, label="c")],
            established_facts=["k1"],
        ),
        search_round=2,
        routing_history=[
            RoutingDecision(
                round=1, from_module="m6", to_module="supplement_m2",
                decided_by="m6",
            )
        ],
    )
    assert runner._should_skip_module("m2", {"literature_results"}, state) is False
    assert runner._should_skip_module("m3", {"evidence_graph"}, state) is False

    suppressed = [
        event
        for event in recorder.read_events()
        if event["event_type"] == "module_skip_suppressed"
    ]
    assert {e["module"] for e in suppressed} == {"m2", "m3"}
    for event in suppressed:
        assert event["details"]["reason"] == "supplement_m2 round"
        assert event["details"]["search_round"] == 2


# ==================== test_run_cancel ====================

import asyncio
import http.client
import json
import threading
from pathlib import Path

import pytest

from hypoforge.config import PipelineConfig
from hypoforge.observability import RunEventRecorder
from hypoforge.pipeline import PipelineRunner, build_followup_seed
from hypoforge.protocol import ModuleProtocol
from hypoforge.state import (
    HypothesisCard,
    M2KnowledgeExport,
    M2KnowledgeRun,
    PipelineState,
    ProblemCard,
)
from hypoforge.webapp import RunManager, build_server


# --------------------------------------------------------------------------- #
# Fake modules for the graph-level cancellation test
# --------------------------------------------------------------------------- #

def _make_fake_module(name: str, patch: dict, hook=None):
    class FakeModule(ModuleProtocol):
        module_name = name
        module_version = "test"
        description = f"fake {name}"

        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def __call__(self, state, config=None):
            if hook is not None:
                hook()
            return dict(patch)

        @classmethod
        def get_input_fields(cls):
            return []

        @classmethod
        def get_output_fields(cls):
            return list(patch)

    return FakeModule()


def _runner_with_fakes(tmp_path: Path, cancel_event, m1_hook=None, m4_hook=None):
    """A PipelineRunner wired with fake m1/m4 modules (no LLM / network).

    Returns ``(runner, recorder, modules)``; callers must monkeypatch
    ``ModuleRegistry.build_all`` to return *modules*.
    """
    problem_card = ProblemCard(
        original_question="问题", sub_questions=["子问题 A"]
    )
    hypothesis = HypothesisCard(hypothesis_id="H1", statement="假设陈述")
    modules = {
        "m1": _make_fake_module(
            "m1", {"problem_card": problem_card}, hook=m1_hook
        ),
        "m4": _make_fake_module(
            "m4", {"top_hypotheses": [hypothesis]}, hook=m4_hook
        ),
    }
    config = PipelineConfig(verbose=False, enabled_modules=["m1", "m4"])
    config.output_dir = str(tmp_path)
    recorder = RunEventRecorder(tmp_path, "run-cancel")
    runner = PipelineRunner(
        config, event_recorder=recorder, cancel_event=cancel_event
    )
    return runner, recorder, modules


@pytest.mark.asyncio
async def test_cancel_flag_stops_after_current_module_and_persists_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cancel set during m1 → m4 must never run; state file keeps m1 output."""
    from hypoforge.registry import ModuleRegistry

    cancel_event = threading.Event()
    m4_calls = {"count": 0}

    def cancel_after_m1():
        cancel_event.set()  # "module completes then stop" (cooperative)

    def count_m4():
        m4_calls["count"] += 1

    runner, recorder, modules = _runner_with_fakes(
        tmp_path, cancel_event, m1_hook=cancel_after_m1, m4_hook=count_m4
    )
    monkeypatch.setattr(
        ModuleRegistry, "build_all", staticmethod(lambda cfg: modules)
    )
    state = await runner.run("问题", run_id="run-cancel")

    assert runner.cancelled is True
    assert m4_calls["count"] == 0
    # m1 artifacts retained, m4 never produced anything
    assert state.problem_card is not None
    assert state.problem_card.sub_questions == ["子问题 A"]
    assert state.top_hypotheses == []

    # Final state persisted → usable as a followup parent
    state_path = tmp_path / "run-cancel.json"
    assert state_path.exists()
    persisted = json.loads(state_path.read_text(encoding="utf-8"))
    assert persisted["problem_card"]["sub_questions"] == ["子问题 A"]

    events = recorder.read_events()
    types = [event["event_type"] for event in events]
    assert "module_cancelled" in types  # m4 refused to start
    assert "run_cancelled" in types
    assert "run_completed" not in types
    cancelled_event = next(
        event for event in events if event["event_type"] == "run_cancelled"
    )
    assert cancelled_event["status"] == "cancelled"


@pytest.mark.asyncio
async def test_cancel_interrupts_a_module_that_is_currently_waiting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A web stop request must cancel active async M4 work promptly."""
    from hypoforge.registry import ModuleRegistry

    started = asyncio.Event()
    cancellation_observed = asyncio.Event()

    class BlockingM4(ModuleProtocol):
        module_name = "m4"
        module_version = "test"
        description = "blocking m4"

        async def __call__(self, state, config=None):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancellation_observed.set()
                raise

        @classmethod
        def get_input_fields(cls):
            return []

        @classmethod
        def get_output_fields(cls):
            return ["top_hypotheses"]

    problem_card = ProblemCard(
        original_question="问题", sub_questions=["子问题 A"]
    )
    modules = {
        "m1": _make_fake_module("m1", {"problem_card": problem_card}),
        "m4": BlockingM4(),
    }
    cancel_event = threading.Event()
    config = PipelineConfig(verbose=False, enabled_modules=["m1", "m4"])
    config.output_dir = str(tmp_path)
    recorder = RunEventRecorder(tmp_path, "run-active-cancel")
    runner = PipelineRunner(
        config, event_recorder=recorder, cancel_event=cancel_event
    )
    monkeypatch.setattr(
        ModuleRegistry, "build_all", staticmethod(lambda cfg: modules)
    )

    run_task = asyncio.create_task(
        runner.run("问题", run_id="run-active-cancel")
    )
    await asyncio.wait_for(started.wait(), timeout=2.0)
    cancel_event.set()
    state = await asyncio.wait_for(run_task, timeout=0.8)

    assert cancellation_observed.is_set()
    assert runner.cancelled is True
    assert state.problem_card is not None
    assert state.problem_card.sub_questions == ["子问题 A"]
    assert state.top_hypotheses == []
    cancelled = [
        event for event in recorder.read_events()
        if event["event_type"] == "module_cancelled"
    ]
    assert cancelled[-1]["module"] == "m4"
    assert cancelled[-1]["details"]["during_module"] is True


@pytest.mark.asyncio
async def test_no_cancel_runs_to_completion(tmp_path: Path, monkeypatch):
    """Control: without a cancel request the same fake graph completes."""
    from hypoforge.registry import ModuleRegistry

    runner, recorder, modules = _runner_with_fakes(
        tmp_path, threading.Event()
    )
    monkeypatch.setattr(
        ModuleRegistry, "build_all", staticmethod(lambda cfg: modules)
    )
    state = await runner.run("问题", run_id="run-full")

    assert runner.cancelled is False
    assert state.problem_card is not None
    assert [h.hypothesis_id for h in state.top_hypotheses] == ["H1"]
    types = [event["event_type"] for event in recorder.read_events()]
    assert "run_completed" in types
    assert "run_cancelled" not in types


# --------------------------------------------------------------------------- #
# Knowledge retention: cancelled parent → followup seed
# --------------------------------------------------------------------------- #

def test_build_followup_seed_from_cancelled_parent_state() -> None:
    """A cancelled run's persisted state must feed build_followup_seed with
    the problem card and the M2 literature knowledge intact."""
    seed_state = PipelineState(
        input_question="原问题",
        run_id="ui-cancelled-parent",
        problem_card=ProblemCard(
            original_question="原问题", sub_questions=["子问题 A"]
        ),
        m2_knowledge_export=M2KnowledgeExport(
            runs=[M2KnowledgeRun(sub_question="子问题 A")]
        ),
    ).model_dump(mode="json")

    seed = build_followup_seed(
        seed_state=seed_state,
        followup_text="细化对照组设计",
        run_id="ui-child",
        config=PipelineConfig(verbose=False),
    )

    assert seed.parent_run_id == "ui-cancelled-parent"
    assert seed.problem_card is not None
    assert seed.problem_card.sub_questions == ["子问题 A"]
    assert seed.m2_knowledge_export is not None
    assert seed.m2_knowledge_export.runs[0].sub_question == "子问题 A"
    assert seed.input_question == "细化对照组设计"


# --------------------------------------------------------------------------- #
# RunManager: request_cancel semantics + worker cancelled status
# --------------------------------------------------------------------------- #

def _write_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
enabled_modules: [m1]
qwen:
  base: {model: default-model}
  max: {model: default-model}
  plus: {model: default-model}
  turbo: {model: default-model}
module_overrides:
  m2:
    kwargs: {}
""",
        encoding="utf-8",
    )
    return config_path


class DeferredThread:
    """threading.Thread stand-in that never starts the worker."""

    def __init__(self, *args, **kwargs):
        pass

    def start(self) -> None:
        pass


class CancellingRunner:
    """FakeRunner honouring the cooperative cancel flag like the real one."""

    def __init__(self, config, event_recorder=None):
        self.config = config
        self.cancel_event = None
        self.cancelled = False

    async def run(self, question: str, run_id: str, **kwargs):
        self.cancelled = bool(
            self.cancel_event is not None and self.cancel_event.is_set()
        )
        state = PipelineState(
            input_question=question,
            run_id=run_id,
            problem_card=ProblemCard(
                original_question=question,
                sub_questions=["已保留的问题卡"],
            ),
        )
        if self.cancelled:
            # Mimic PipelineRunner._save_output on a graceful stop: the
            # accumulated state must land on disk for followup seeding.
            out_dir = Path(self.config.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"{run_id}.json").write_text(
                json.dumps(state.model_dump(mode="json"), ensure_ascii=False),
                encoding="utf-8",
            )
        return state


def test_request_cancel_marks_cancelled_and_retains_knowledge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = RunManager(
        config_path=_write_config(tmp_path), output_root=tmp_path / "runs"
    )
    monkeypatch.setattr(threading, "Thread", DeferredThread)
    monkeypatch.setattr("hypoforge.webapp.PipelineRunner", CancellingRunner)

    run = manager.start("原问题")
    run_id = run["run_id"]

    info = manager.request_cancel(run_id)
    assert info == {"run_id": run_id, "status": "cancelling"}
    assert manager._cancel_events[run_id].is_set()
    assert manager.get(run_id)["cancel_requested"] is True
    # idempotent while cancelling
    assert manager.request_cancel(run_id)["status"] == "cancelling"

    # The worker observes the flag and the run ends as cancelled.
    manager._run_pipeline(run_id, "原问题")
    assert manager.get(run_id)["status"] == "cancelled"

    manifest = json.loads(
        (tmp_path / "runs" / run_id / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["status"] == "cancelled"
    assert "停止" in manifest["cancel_message"]

    events = RunEventRecorder(
        tmp_path / "runs" / run_id, run_id
    ).read_events()
    types = [event["event_type"] for event in events]
    # run_cancelled itself is emitted by the real pipeline runner (covered
    # by the graph-level test above); the web layer records the request.
    assert "cancel_requested" in types

    # Knowledge retention: the final state keeps the problem card …
    state = json.loads(
        (tmp_path / "runs" / run_id / f"{run_id}.json").read_text(
            encoding="utf-8"
        )
    )
    assert state["problem_card"]["sub_questions"] == ["已保留的问题卡"]

    # … and the cancelled run is accepted as a followup parent.
    followup = manager.start(
        "细化对照组", parent_run_id=run_id, followup="细化对照组"
    )
    assert followup["parent_run_id"] == run_id


def test_request_cancel_rejects_finished_unknown_and_malformed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = RunManager(
        config_path=_write_config(tmp_path), output_root=tmp_path / "runs"
    )
    monkeypatch.setattr(threading, "Thread", DeferredThread)
    monkeypatch.setattr("hypoforge.webapp.PipelineRunner", CancellingRunner)

    run = manager.start("原问题")
    run_id = run["run_id"]
    manager._run_pipeline(run_id, "原问题")  # finishes (no cancel requested)
    assert manager.get(run_id)["status"] == "completed"

    with pytest.raises(RuntimeError, match="已结束"):
        manager.request_cancel(run_id)
    with pytest.raises(LookupError, match="不存在"):
        manager.request_cancel("ui-ghost-run")
    with pytest.raises(ValueError, match="非法字符"):
        manager.request_cancel("../escape")

    # Persisted-only run (server restarted) → "已结束", not 404.
    finished_dir = tmp_path / "runs" / "ui-old-finished"
    finished_dir.mkdir(parents=True)
    (finished_dir / "manifest.json").write_text(
        json.dumps({"run_id": "ui-old-finished", "status": "completed"}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="已结束"):
        manager.request_cancel("ui-old-finished")


# --------------------------------------------------------------------------- #
# HTTP endpoint contract
# --------------------------------------------------------------------------- #

def test_http_cancel_endpoint_statuses(tmp_path: Path) -> None:
    _write_config(tmp_path)
    server = build_server(
        host="127.0.0.1",
        port=0,
        config_path=tmp_path / "config.yaml",
        output_root=tmp_path / "runs",
        static_dir=tmp_path,
    )
    # A live in-memory running run.
    server.manager._runs["ui-live"] = {
        "run_id": "ui-live",
        "status": "running",
        "cancel_requested": False,
    }
    server.manager._cancel_events["ui-live"] = threading.Event()
    # An in-memory finished run.
    server.manager._runs["ui-done"] = {
        "run_id": "ui-done",
        "status": "completed",
    }
    # A persisted-only finished run.
    done_dir = tmp_path / "runs" / "ui-disk-done"
    done_dir.mkdir(parents=True)
    (done_dir / "manifest.json").write_text(
        json.dumps({"run_id": "ui-disk-done", "status": "completed"}),
        encoding="utf-8",
    )

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)

        # 运行中 → 202 cancelling，标志置位 + cancel_requested 事件落盘。
        conn.request("POST", "/api/runs/ui-live/cancel")
        response = conn.getresponse()
        assert response.status == 202
        assert json.loads(response.read()) == {
            "run_id": "ui-live",
            "status": "cancelling",
        }
        assert server.manager._cancel_events["ui-live"].is_set()
        events = RunEventRecorder(
            tmp_path / "runs" / "ui-live", "ui-live"
        ).read_events()
        assert any(
            event["event_type"] == "cancel_requested" for event in events
        )

        # 幂等：再次请求仍是 cancelling。
        conn.request("POST", "/api/runs/ui-live/cancel")
        response = conn.getresponse()
        assert response.status == 202
        assert json.loads(response.read())["status"] == "cancelling"

        # 已结束（内存记录 / 仅落盘）→ 409。
        conn.request("POST", "/api/runs/ui-done/cancel")
        response = conn.getresponse()
        assert response.status == 409
        assert "已结束" in json.loads(response.read())["error"]

        conn.request("POST", "/api/runs/ui-disk-done/cancel")
        response = conn.getresponse()
        assert response.status == 409
        assert "已结束" in json.loads(response.read())["error"]

        # 不存在 → 404。
        conn.request("POST", "/api/runs/ui-ghost/cancel")
        response = conn.getresponse()
        assert response.status == 404
        assert "不存在" in json.loads(response.read())["error"]

        # 非法 run_id → 400。
        conn.request("POST", "/api/runs/..%2Fxxx/cancel")
        response = conn.getresponse()
        assert response.status == 400
    finally:
        server.shutdown()
        server.server_close()


# ==================== test_observability_webapp ====================

import http.client
import json
import re
import threading
from pathlib import Path

import pytest

from hypoforge.observability import RunEventRecorder, bind_recorder, emit_event
from hypoforge.state import PipelineState
from hypoforge.webapp import RunManager, build_server


def test_recorder_persists_events_and_snapshots(tmp_path: Path) -> None:
    recorder = RunEventRecorder(tmp_path / "run-1", "run-1")
    recorder.emit(
        "run_started",
        status="running",
        message="start",
        details={"question": "q"},
    )
    with bind_recorder(recorder):
        emit_event(
            "tool_completed",
            module="m2",
            tool="pubmed",
            status="completed",
            details={"papers": 3},
        )
    snapshot = recorder.save_snapshot("m2-iteration-0", {"papers": [1, 2, 3]})
    recorder.write_manifest({"run_id": "run-1", "status": "running"})

    events = recorder.read_events()
    assert [event["sequence"] for event in events] == [1, 2]
    assert events[1]["tool"] == "pubmed"
    assert json.loads(snapshot.read_text(encoding="utf-8"))["papers"] == [1, 2, 3]
    assert json.loads(recorder.manifest_path.read_text(encoding="utf-8"))[
        "run_id"
    ] == "run-1"


def test_run_manager_reads_persisted_result_and_artifacts(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("enabled_modules: [m1]\n", encoding="utf-8")
    manager = RunManager(config_path=config_path, output_root=tmp_path / "runs")
    run_id = "ui-test"
    run_dir = tmp_path / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"run_id": run_id, "status": "completed"}),
        encoding="utf-8",
    )
    (run_dir / f"{run_id}.json").write_text(
        json.dumps(
            {
                "input_question": "q",
                "top_hypotheses": [{"hypothesis_id": "H1", "statement": "s"}],
                "research_plans": [],
                "reviews": [],
                "errors": [],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / f"{run_id}_scores.json").write_text(
        json.dumps({"aggregate": {"top1_composite": 0.8}}),
        encoding="utf-8",
    )

    assert manager.get(run_id)["status"] == "completed"
    assert manager.result(run_id)["scores"]["aggregate"]["top1_composite"] == 0.8
    names = {item["name"] for item in manager.artifacts(run_id)}
    assert f"{run_id}.json" in names
    assert f"{run_id}_scores.json" in names


def test_run_credentials_are_memory_only_and_applied_per_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
enabled_modules: [m1]
qwen:
  base: {model: default-model}
  max: {model: default-model}
  plus: {model: default-model}
  turbo: {model: default-model}
module_overrides:
  m2:
    kwargs: {}
""",
        encoding="utf-8",
    )
    manager = RunManager(config_path=config_path, output_root=tmp_path / "runs")
    captured = {}

    class DeferredThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self) -> None:
            pass

    class FakeRunner:
        def __init__(self, config, event_recorder=None):
            captured["config"] = config

        async def run(self, question: str, run_id: str):
            return PipelineState(input_question=question, run_id=run_id)

    monkeypatch.setattr(threading, "Thread", DeferredThread)
    monkeypatch.setattr("hypoforge.webapp.PipelineRunner", FakeRunner)
    run = manager.start(
        "q",
        model_name="qwen-custom",
        qwen_api_key="qwen-secret-for-test",
        semantic_scholar_api_key="s2-secret-for-test",
        openalex_api_key="oa-secret-for-test",
        openalex_mailto="lab@example.org",
    )
    assert "secret" not in json.dumps(run)

    manager._run_pipeline(run["run_id"], "q")
    config = captured["config"]
    assert config.qwen.plus.model == "qwen-custom"
    assert config.qwen.plus.api_key == "qwen-secret-for-test"
    assert (
        config.module_overrides["m2"].kwargs["semantic_scholar_api_key"]
        == "s2-secret-for-test"
    )
    assert (
        config.module_overrides["m2"].kwargs["openalex_api_key"]
        == "oa-secret-for-test"
    )
    assert (
        config.module_overrides["m2"].kwargs["openalex_mailto"]
        == "lab@example.org"
    )
    persisted = (
        tmp_path / "runs" / run["run_id"] / "manifest.json"
    ).read_text(encoding="utf-8")
    assert "qwen-secret-for-test" not in persisted
    assert "s2-secret-for-test" not in persisted
    assert "oa-secret-for-test" not in persisted


def test_run_manager_rejects_malformed_openalex_credentials(
    tmp_path: Path,
) -> None:
    manager = RunManager(
        config_path=_write_config(tmp_path), output_root=tmp_path / "runs"
    )
    with pytest.raises(ValueError, match="API Key 长度异常"):
        manager.start("q", openalex_api_key="x" * 4097)
    with pytest.raises(ValueError, match="Mailto 长度异常"):
        manager.start("q", openalex_mailto="a" * 321)


def test_web_ui_preserves_open_event_details_and_has_ephemeral_key_fields() -> None:
    html_path = (
        Path(__file__).resolve().parent.parent
        / "hypoforge"
        / "web"
        / "index.html"
    )
    html = html_path.read_text(encoding="utf-8")

    assert 'id="modelName"' in html
    assert 'id="qwenApiKey"' in html
    assert 'id="semanticApiKey"' in html
    assert 'id="openalexApiKey"' in html
    assert 'id="openalexMailto"' in html
    # Ephemeral keys: never echoed back, persisted, or sent in plaintext.
    # localStorage is allowed only for non-sensitive UI state (history order);
    # credential fields must never appear in any storage read/write path.
    storage_keys = re.findall(
        r'localStorage\.(?:setItem|getItem|removeItem)\(\s*["\']([^"\']+)["\']',
        html,
    )
    credential_ids = {
        "modelName", "qwenApiKey", "semanticApiKey",
        "openalexApiKey", "openalexMailto",
    }
    assert not (set(storage_keys) & credential_ids)
    assert "hypoforge_order" in storage_keys  # the only non-sensitive persistence
    assert 'id="openalexApiKey" type="password"' in html
    assert "openSequences" in html
    assert "data-event-sequence" in html


class DeferredThread:
    def __init__(self, *args, **kwargs):
        pass

    def start(self) -> None:
        pass


def _make_parent_run(tmp_path: Path, parent_id: str = "ui-parent") -> Path:
    parent_dir = tmp_path / "runs" / parent_id
    parent_dir.mkdir(parents=True)
    (parent_dir / f"{parent_id}.json").write_text(
        json.dumps(
            {
                "run_id": parent_id,
                "input_question": "parent question",
                "problem_card": {"original_question": "parent question"},
                "top_hypotheses": [{"statement": "parent H1"}],
                "errors": [],
            }
        ),
        encoding="utf-8",
    )
    (parent_dir / "manifest.json").write_text(
        json.dumps({"run_id": parent_id, "status": "completed"}),
        encoding="utf-8",
    )
    return parent_dir


def test_followup_defaults_keep_behavior_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = RunManager(
        config_path=_write_config(tmp_path), output_root=tmp_path / "runs"
    )
    captured: dict = {}

    class FakeRunner:
        def __init__(self, config, event_recorder=None):
            pass

        async def run(self, question: str, run_id: str, **kwargs):
            captured["kwargs"] = kwargs
            return PipelineState(input_question=question, run_id=run_id)

    monkeypatch.setattr(threading, "Thread", DeferredThread)
    monkeypatch.setattr("hypoforge.webapp.PipelineRunner", FakeRunner)
    run = manager.start("q")
    assert "parent_run_id" not in run
    manager._run_pipeline(run["run_id"], "q")
    # No followup -> run() called without seed_state/followup_text kwargs.
    assert captured["kwargs"] == {}
    manifest = json.loads(
        (tmp_path / "runs" / run["run_id"] / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert "parent_run_id" not in manifest


def test_followup_missing_parent_returns_error(tmp_path: Path) -> None:
    manager = RunManager(
        config_path=_write_config(tmp_path), output_root=tmp_path / "runs"
    )
    with pytest.raises(ValueError, match="不存在"):
        manager.start("追问", parent_run_id="ghost-run", followup="再细化一下")
    with pytest.raises(ValueError):
        manager.start("追问", followup="只有追问没有父运行")
    with pytest.raises(ValueError, match="非法字符"):
        manager.start("追问", parent_run_id="../escape", followup="x")


def test_followup_corrupt_parent_state_returns_error(tmp_path: Path) -> None:
    manager = RunManager(
        config_path=_write_config(tmp_path), output_root=tmp_path / "runs"
    )
    parent_dir = tmp_path / "runs" / "ui-corrupt"
    parent_dir.mkdir(parents=True)
    (parent_dir / "ui-corrupt.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="损坏"):
        manager.start("追问", parent_run_id="ui-corrupt", followup="x")


def test_followup_passes_seed_state_and_marks_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = RunManager(
        config_path=_write_config(tmp_path), output_root=tmp_path / "runs"
    )
    _make_parent_run(tmp_path)
    captured: dict = {}

    class FakeRunner:
        def __init__(self, config, event_recorder=None):
            pass

        async def run(self, question: str, run_id: str, **kwargs):
            captured["kwargs"] = kwargs
            return PipelineState(input_question=question, run_id=run_id)

    monkeypatch.setattr(threading, "Thread", DeferredThread)
    monkeypatch.setattr("hypoforge.webapp.PipelineRunner", FakeRunner)
    run = manager.start(
        "请细化对照组设计",
        parent_run_id="ui-parent",
        followup="请细化对照组设计",
    )
    assert run["parent_run_id"] == "ui-parent"
    manager._run_pipeline(run["run_id"], "请细化对照组设计")
    kwargs = captured["kwargs"]
    assert kwargs["followup_text"] == "请细化对照组设计"
    assert kwargs["seed_state"]["run_id"] == "ui-parent"
    manifest = json.loads(
        (tmp_path / "runs" / run["run_id"] / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["parent_run_id"] == "ui-parent"
    assert manifest["followup"] == "请细化对照组设计"


def test_resume_reads_parent_checkpoint_without_followup_semantics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = RunManager(
        config_path=_write_config(tmp_path), output_root=tmp_path / "runs"
    )
    parent_dir = tmp_path / "runs" / "ui-parent"
    parent_dir.mkdir(parents=True)
    (parent_dir / "ui-parent_checkpoint.json").write_text(
        json.dumps({
            "run_id": "ui-parent",
            "input_question": "蛋白质错误折叠如何导致神经退行性疾病？",
            "_last_module": "m1",
        }),
        encoding="utf-8",
    )
    captured: dict = {}

    class FakeRunner:
        def __init__(self, config, event_recorder=None):
            pass

        async def run(self, question: str, run_id: str, **kwargs):
            captured["kwargs"] = kwargs
            return PipelineState(input_question=question, run_id=run_id)

    monkeypatch.setattr(threading, "Thread", DeferredThread)
    monkeypatch.setattr("hypoforge.webapp.PipelineRunner", FakeRunner)
    run = manager.start("ignored question", resume_of="ui-parent")
    assert run["resume_of"] == "ui-parent"
    # 问题文本以后端 checkpoint 为准，调用方无法漂移。
    assert run["question"] == "蛋白质错误折叠如何导致神经退行性疾病？"
    manager._run_pipeline(run["run_id"], run["question"])
    kwargs = captured["kwargs"]
    # 纯续传：无 followup_text，不触发 followup 路由/预算重置。
    assert "followup_text" not in kwargs
    assert kwargs["seed_state"]["_last_module"] == "m1"
    assert kwargs["seed_state"]["input_question"] == (
        "蛋白质错误折叠如何导致神经退行性疾病？"
    )
    manifest = json.loads(
        (tmp_path / "runs" / run["run_id"] / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["resume_of"] == "ui-parent"
    assert "parent_run_id" not in manifest


def test_resume_rejects_running_parent(tmp_path: Path) -> None:
    manager = RunManager(
        config_path=_write_config(tmp_path), output_root=tmp_path / "runs"
    )
    manager._runs["ui-live"] = {"run_id": "ui-live", "status": "running"}
    with pytest.raises(ValueError, match="仍在运行"):
        manager.start("q", resume_of="ui-live")


def test_resume_missing_checkpoint_returns_error(tmp_path: Path) -> None:
    manager = RunManager(
        config_path=_write_config(tmp_path), output_root=tmp_path / "runs"
    )
    (tmp_path / "runs" / "ui-ghost").mkdir(parents=True)
    with pytest.raises(ValueError, match="没有可用的断点"):
        manager.start("q", resume_of="ui-ghost")


def test_resume_rejects_illegal_parent_id_and_followup_conflict(
    tmp_path: Path,
) -> None:
    manager = RunManager(
        config_path=_write_config(tmp_path), output_root=tmp_path / "runs"
    )
    with pytest.raises(ValueError, match="非法字符"):
        manager.start("q", resume_of="../escape")
    with pytest.raises(ValueError, match="不能与"):
        manager.start(
            "q",
            parent_run_id="ui-parent",
            followup="x",
            resume_of="ui-parent",
        )


def test_rename_run_writes_display_name_and_keeps_question(
    tmp_path: Path,
) -> None:
    manager = RunManager(
        config_path=_write_config(tmp_path), output_root=tmp_path / "runs"
    )
    run_dir = tmp_path / "runs" / "ui-rename"
    run_dir.mkdir(parents=True)
    manifest = {"run_id": "ui-rename", "question": "原始问题", "status": "completed"}
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    updated = manager.rename("ui-rename", "  我的标注名称  ")

    assert updated["display_name"] == "我的标注名称"
    assert updated["question"] == "原始问题"
    assert json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))[
        "display_name"
    ] == "我的标注名称"
    # 侧栏列表直接带出 display_name。
    listed = next(r for r in manager.list_runs() if r["run_id"] == "ui-rename")
    assert listed["display_name"] == "我的标注名称"


def test_rename_run_validations(tmp_path: Path) -> None:
    manager = RunManager(
        config_path=_write_config(tmp_path), output_root=tmp_path / "runs"
    )
    with pytest.raises(ValueError, match="非法字符"):
        manager.rename("../escape", "x")
    with pytest.raises(ValueError, match="不能为空"):
        manager.rename("ui-x", "   ")
    with pytest.raises(ValueError, match="过长"):
        manager.rename("ui-x", "长" * 201)
    with pytest.raises(LookupError, match="不存在"):
        manager.rename("ui-ghost", "新名字")


def test_single_run_mutex_error_message_mentions_waiting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = RunManager(
        config_path=_write_config(tmp_path), output_root=tmp_path / "runs"
    )
    monkeypatch.setattr(threading, "Thread", DeferredThread)
    manager.start("第一条")
    with pytest.raises(RuntimeError, match="当前运行结束后才能提交"):
        manager.start("第二条")


def test_followup_rejected_with_409_while_run_in_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Submitting a followup while another run is executing must be rejected
    with the mutex error (HTTP layer maps RuntimeError → 409); no queueing."""
    manager = RunManager(
        config_path=_write_config(tmp_path), output_root=tmp_path / "runs"
    )
    _make_parent_run(tmp_path)
    monkeypatch.setattr(threading, "Thread", DeferredThread)
    manager.start("第一条")
    with pytest.raises(RuntimeError, match="当前运行结束后才能提交"):
        manager.start("追问", parent_run_id="ui-parent", followup="细化对照组")


def _write_routing_fixture(tmp_path: Path, run_id: str) -> None:
    run_dir = tmp_path / "runs" / run_id
    recorder = RunEventRecorder(run_dir, run_id)
    recorder.emit(
        "module_started", module="m1", status="running",
        message="M1 开始", details={"iteration_count": 0},
    )
    recorder.emit(
        "routing_decision", module="m1", status="completed", message="route",
        details={"from": "m1", "to": "search_m2", "decided_by": "policy",
                 "reason": "standard path", "gap_ids": [], "round": 1},
    )
    for name in ("m2", "m3", "m4", "m5", "m6"):
        recorder.emit(
            "module_started", module=name, status="running",
            message=f"{name} 开始", details={"iteration_count": 0},
        )
    recorder.emit(
        "routing_decision", module="m6", status="completed", message="route",
        details={"from": "m6", "to": "revise_m4", "decided_by": "policy",
                 "reason": "revise hypotheses", "gap_ids": ["G-1"], "round": 2},
    )
    for name in ("m4", "m5", "m6"):
        recorder.emit(
            "module_started", module=name, status="running",
            message=f"{name} 再来一轮", details={"iteration_count": 1},
        )
    recorder.emit(
        "routing_decision", module="m6", status="completed", message="route",
        details={"from": "m6", "to": "end", "decided_by": "policy",
                 "reason": "threshold met", "gap_ids": [], "round": 3},
    )
    (run_dir / f"{run_id}.json").write_text(
        json.dumps(
            {
                "input_question": "q",
                "reviews": [
                    {"dimension": "overall", "score": 3.5, "version": 1},
                    {"dimension": "overall", "score": 4.5, "version": 2},
                ],
                "evidence_gaps": [{"gap_id": "G-1", "status": "closed"}],
                "top_hypotheses": [{"statement": "final H1"}],
                "research_plans": [{"hypothesis_id": "H1"}],
                "iteration_count": 2,
                "search_round": 1,
                "routing_history": [
                    {"round": 1, "from_module": "m1", "to_module": "search_m2",
                     "decided_by": "policy", "reason": "standard path", "gap_ids": []},
                    {"round": 2, "from_module": "m6", "to_module": "revise_m4",
                     "decided_by": "policy", "reason": "revise hypotheses",
                     "gap_ids": ["G-1"]},
                    {"round": 3, "from_module": "m6", "to_module": "end",
                     "decided_by": "policy", "reason": "threshold met", "gap_ids": []},
                ],
                "errors": [],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "snapshots").mkdir(exist_ok=True)
    (run_dir / "snapshots" / "001-m4-r0-iter0.json").write_text(
        json.dumps({"top_hypotheses": [{"statement": "v1 H1"}], "research_plans": [{}]}),
        encoding="utf-8",
    )
    (run_dir / "snapshots" / "002-m6-r0-iter1.json").write_text(
        json.dumps(
            {
                "top_hypotheses": [{"statement": "v2 H1"}],
                "research_plans": [{}, {}],
                "reviews": [{"dimension": "overall", "score": 4.5, "version": 2}],
            }
        ),
        encoding="utf-8",
    )


def test_rounds_endpoint_contract_shape_from_state_json(tmp_path: Path) -> None:
    manager = RunManager(
        config_path=_write_config(tmp_path), output_root=tmp_path / "runs"
    )
    _write_routing_fixture(tmp_path, "ui-rounds")
    payload = manager.rounds("ui-rounds")
    # Contract shape: {"routing_history": [...], "search_round": n}
    assert set(payload) == {"routing_history", "search_round"}
    assert payload["search_round"] == 1
    assert [d["to_module"] for d in payload["routing_history"]] == [
        "search_m2", "revise_m4", "end",
    ]
    assert payload["routing_history"][1]["gap_ids"] == ["G-1"]
    # Missing run → empty default shape (never a crash, never a raw list).
    assert manager.rounds("missing-run") == {"routing_history": [], "search_round": 0}


def test_versions_endpoint_groups_reviews_and_snapshots(tmp_path: Path) -> None:
    manager = RunManager(
        config_path=_write_config(tmp_path), output_root=tmp_path / "runs"
    )
    _write_routing_fixture(tmp_path, "ui-vers")
    versions = manager.versions("ui-vers")
    assert manager.versions("missing-run") == []
    assert [v["version"] for v in versions] == [1, 2]
    # Contract shape: each version entry carries version/overall/timestamp.
    for entry in versions:
        assert {"version", "overall", "timestamp"} <= set(entry)
    v1, v2 = versions
    assert v1["overall"] == 3.5
    assert v1["top_hypothesis_titles"] == ["v1 H1"]
    assert v1["research_plans_count"] == 1
    assert v1["research_plans"] == [{}]
    assert v1["snapshots"] == ["snapshots/001-m4-r0-iter0.json"]
    assert v2["overall"] == 4.5
    # Final state is authoritative for the latest version.
    assert v2["top_hypothesis_titles"] == ["final H1"]
    assert v2["research_plans_count"] == 1
    assert v2["research_plans"] == [{"hypothesis_id": "H1"}]
    assert "snapshots/002-m6-r0-iter1.json" in v2["snapshots"]


def test_result_exposes_routing_fields(tmp_path: Path) -> None:
    manager = RunManager(
        config_path=_write_config(tmp_path), output_root=tmp_path / "runs"
    )
    _write_routing_fixture(tmp_path, "ui-res")
    result = manager.result("ui-res")
    assert result["search_round"] == 1
    assert result["revision_count"] == 0
    assert [d["to_module"] for d in result["routing_history"]] == [
        "search_m2", "revise_m4", "end",
    ]
    assert result["evidence_gaps"][0]["status"] == "closed"


def _make_run(
    tmp_path: Path, run_id: str, parent: str = "", status: str = "completed"
) -> Path:
    run_dir = tmp_path / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"run_id": run_id, "status": status}
    if parent:
        manifest["parent_run_id"] = parent
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return run_dir


def test_delete_run_removes_directory_and_returns_list(tmp_path: Path) -> None:
    manager = RunManager(
        config_path=_write_config(tmp_path), output_root=tmp_path / "runs"
    )
    _make_run(tmp_path, "ui-del-a")
    deleted = manager.delete("ui-del-a")
    assert deleted == ["ui-del-a"]
    assert not (tmp_path / "runs" / "ui-del-a").exists()


def test_delete_chain_root_cascades_to_descendants(tmp_path: Path) -> None:
    manager = RunManager(
        config_path=_write_config(tmp_path), output_root=tmp_path / "runs"
    )
    _make_run(tmp_path, "ui-root")
    _make_run(tmp_path, "ui-child", parent="ui-root")
    _make_run(tmp_path, "ui-grandchild", parent="ui-child")
    _make_run(tmp_path, "ui-unrelated")
    deleted = manager.delete("ui-root")
    assert set(deleted) == {"ui-root", "ui-child", "ui-grandchild"}
    for run_id in ("ui-root", "ui-child", "ui-grandchild"):
        assert not (tmp_path / "runs" / run_id).exists()
    assert (tmp_path / "runs" / "ui-unrelated").exists()


def test_delete_middle_run_cascades_only_its_subtree(tmp_path: Path) -> None:
    manager = RunManager(
        config_path=_write_config(tmp_path), output_root=tmp_path / "runs"
    )
    _make_run(tmp_path, "ui-root")
    _make_run(tmp_path, "ui-child", parent="ui-root")
    _make_run(tmp_path, "ui-grandchild", parent="ui-child")
    deleted = manager.delete("ui-child")
    assert set(deleted) == {"ui-child", "ui-grandchild"}
    assert (tmp_path / "runs" / "ui-root").exists()


def test_delete_missing_run_raises_lookup_error(tmp_path: Path) -> None:
    manager = RunManager(
        config_path=_write_config(tmp_path), output_root=tmp_path / "runs"
    )
    with pytest.raises(LookupError, match="不存在"):
        manager.delete("ui-ghost")


def test_delete_illegal_run_id_rejected_and_nothing_deleted(
    tmp_path: Path,
) -> None:
    manager = RunManager(
        config_path=_write_config(tmp_path), output_root=tmp_path / "runs"
    )
    kept = _make_run(tmp_path, "ui-keep")
    outside = tmp_path / "outside"
    outside.mkdir()
    for bad_id in ("../outside", "ui-../outside", "../xxx", "plain-name", ""):
        with pytest.raises(ValueError):
            manager.delete(bad_id)
    assert kept.exists()
    assert outside.exists()


def test_delete_running_run_rejected_with_conflict(tmp_path: Path) -> None:
    """Running state is injected into the manager's in-memory registry (the
    same mechanism start() uses) since a live pipeline run cannot be built
    in a unit test."""
    manager = RunManager(
        config_path=_write_config(tmp_path), output_root=tmp_path / "runs"
    )
    run_dir = _make_run(tmp_path, "ui-busy", status="running")
    manager._runs["ui-busy"] = {"run_id": "ui-busy", "status": "running"}
    with pytest.raises(RuntimeError, match="正在执行中"):
        manager.delete("ui-busy")
    assert run_dir.exists()


def test_delete_rejected_when_descendant_is_running(tmp_path: Path) -> None:
    manager = RunManager(
        config_path=_write_config(tmp_path), output_root=tmp_path / "runs"
    )
    root_dir = _make_run(tmp_path, "ui-root")
    child_dir = _make_run(tmp_path, "ui-child", parent="ui-root", status="running")
    manager._runs["ui-child"] = {"run_id": "ui-child", "status": "running"}
    with pytest.raises(RuntimeError, match="正在执行中"):
        manager.delete("ui-root")
    assert root_dir.exists()
    assert child_dir.exists()


def test_http_delete_endpoint_statuses(tmp_path: Path) -> None:
    _write_config(tmp_path)
    server = build_server(
        host="127.0.0.1",
        port=0,
        config_path=tmp_path / "config.yaml",
        output_root=tmp_path / "runs",
        static_dir=tmp_path,
    )
    _make_run(tmp_path, "ui-http-del")
    _make_run(tmp_path, "ui-http-root")
    _make_run(tmp_path, "ui-http-child", parent="ui-http-root")
    _make_run(tmp_path, "ui-http-busy", status="running")
    server.manager._runs["ui-http-busy"] = {
        "run_id": "ui-http-busy",
        "status": "running",
    }
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)

        # 删除存在的运行 → 200 + deleted 列表，目录消失。
        conn.request("DELETE", "/api/runs/ui-http-del")
        response = conn.getresponse()
        assert response.status == 200
        assert json.loads(response.read())["deleted"] == ["ui-http-del"]
        assert not (tmp_path / "runs" / "ui-http-del").exists()

        # 删除链根 → 后代一并删除。
        conn.request("DELETE", "/api/runs/ui-http-root")
        response = conn.getresponse()
        assert response.status == 200
        assert set(json.loads(response.read())["deleted"]) == {
            "ui-http-root",
            "ui-http-child",
        }

        # 不存在 → 404。
        conn.request("DELETE", "/api/runs/ui-ghost")
        response = conn.getresponse()
        assert response.status == 404
        assert "不存在" in json.loads(response.read())["error"]

        # 非法 run_id → 400，且不删除任何目录。
        conn.request("DELETE", "/api/runs/..%2Fxxx")
        response = conn.getresponse()
        assert response.status == 400
        assert (tmp_path / "runs" / "ui-http-busy").exists()

        # 运行中 → 409 中文提示。
        conn.request("DELETE", "/api/runs/ui-http-busy")
        response = conn.getresponse()
        assert response.status == 409
        assert "正在执行中" in json.loads(response.read())["error"]
    finally:
        server.shutdown()
        server.server_close()


def test_http_endpoints_rounds_versions_and_followup_error(
    tmp_path: Path,
) -> None:
    _write_config(tmp_path)
    server = build_server(
        host="127.0.0.1",
        port=0,
        config_path=tmp_path / "config.yaml",
        output_root=tmp_path / "runs",
        static_dir=tmp_path,
    )
    _write_routing_fixture(tmp_path, "ui-http")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)

        conn.request("GET", "/api/runs/ui-http/rounds")
        response = conn.getresponse()
        assert response.status == 200
        rounds_payload = json.loads(response.read())
        # Contract shape at the HTTP layer too.
        assert set(rounds_payload) == {"routing_history", "search_round"}
        assert len(rounds_payload["routing_history"]) == 3
        assert rounds_payload["search_round"] == 1

        conn.request("GET", "/api/runs/ui-http/versions")
        response = conn.getresponse()
        assert response.status == 200
        assert len(json.loads(response.read())["versions"]) == 2

        conn.request(
            "POST",
            "/api/runs",
            body=json.dumps(
                {"question": "q", "parent_run_id": "ghost", "followup": "x"}
            ),
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        assert response.status == 400
        assert "不存在" in json.loads(response.read())["error"]

        # Followup submitted while a run is in progress → 409 (mutex, no queue).
        server.manager._runs["busy-run"] = {"run_id": "busy-run", "status": "running"}
        _make_parent_run(tmp_path)
        conn.request(
            "POST",
            "/api/runs",
            body=json.dumps(
                {
                    "question": "追问",
                    "parent_run_id": "ui-parent",
                    "followup": "细化对照组",
                }
            ),
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        assert response.status == 409
        assert "当前运行结束后才能提交" in json.loads(response.read())["error"]
    finally:
        server.shutdown()
        server.server_close()

def _read_index_html() -> str:
    html_path = (
        Path(__file__).resolve().parent.parent
        / "hypoforge"
        / "web"
        / "index.html"
    )
    return html_path.read_text(encoding="utf-8")


# --- merged from webapp_literal_ids_test.py ---

def test_preserves_six_legacy_literals() -> None:
    html = _read_index_html()

    # Ephemeral credential inputs (never persisted).
    assert 'id="modelName"' in html
    assert 'id="qwenApiKey"' in html
    assert 'id="semanticApiKey"' in html
    # Expanded event <details> survive incremental re-renders.
    assert "openSequences" in html
    assert "data-event-sequence" in html


def test_iteration_visualisation_element_ids_present() -> None:
    html = _read_index_html()

    # Round timeline (consumes GET /api/runs/{id}/rounds routing_history).
    assert 'id="roundsCard"' in html
    assert 'id="roundsList"' in html
    # Iteration state panel: three counters + evidence gaps.
    assert 'id="counterIteration"' in html
    assert 'id="counterRevision"' in html
    assert 'id="counterSearchRound"' in html
    assert 'id="gapList"' in html
    # Plan version switching (consumes GET /api/runs/{id}/versions).
    assert 'id="planCard"' in html
    assert 'id="planVersions"' in html
    # Followup entry (POST /api/runs with parent_run_id + followup).
    assert 'id="followupBox"' in html
    assert 'id="followupInput"' in html
    assert 'id="followupSubmit"' in html
    assert 'id="followupHint"' in html
    # Preserved capabilities: error state / artifacts / event stream.
    assert 'id="runErrorBanner"' in html
    assert 'id="artifactsList"' in html
    assert 'id="eventsList"' in html


# ==================== test_pipeline ====================

import asyncio
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

# Ensure HypoForge is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hypoforge.config import PipelineConfig
from hypoforge.pipeline import PipelineRunner, _should_continue_iterating
from hypoforge.protocol import ModuleProtocol
from hypoforge.state import (
    HypothesisCard,
    PipelineState,
    ResearchPlan,
    ReviewResult,
    ReviewerDimension,
)


# --------------------------------------------------------------------------- #
# Config loading
# --------------------------------------------------------------------------- #

def test_default_config_loads():
    """Default config should load without errors."""
    config = PipelineConfig.from_defaults()
    assert config.enabled_modules == ["m1", "m2", "m3", "m4", "m5", "m6"]
    assert config.max_iterations == 3
    assert config.search.implementation == "agentic"
    assert config.evaluation.embedding.model_name == "text-embedding-v3"
    assert config.evaluation.consistency.similarity_threshold == 0.5


def test_pipeline_evaluation_config_is_validated() -> None:
    config = PipelineConfig(evaluation={
        "embedding": {"model_name": "custom-embedding"},
        "consistency": {"similarity_threshold": 0.42},
    })

    assert config.evaluation.embedding.model_name == "custom-embedding"
    assert config.evaluation.consistency.similarity_threshold == 0.42
    with pytest.raises(ValidationError):
        PipelineConfig(evaluation={
            "consistency": {"similarity_threshold": 1.1},
        })


def test_agentic_search_implementation_is_explicitly_supported() -> None:
    config = PipelineConfig(search={"implementation": "agentic"})

    assert config.search.implementation == "agentic"
    assert config.get_module_kwargs("m2")["implementation"] == "agentic"
    assert config.get_module_kwargs("m2")["budget"]["max_rounds"] == 3
    assert config.get_module_kwargs("m2")["enabled_sources"] == [
        "semantic_scholar",
        "pubmed",
    ]


def test_agentic_search_config_is_forwarded_to_integrated_adapter() -> None:
    config = PipelineConfig(
        search={
            "implementation": "agentic",
            "tools": ["pubmed", "arxiv"],
            "papers_per_sub_question": 7,
            "max_papers_total": 23,
            "max_rounds": 2,
            "max_queries": 5,
            "max_tokens": 4567,
            "max_seconds": 89,
        }
    )

    kwargs = config.get_module_kwargs("m2")
    assert kwargs["enabled_sources"] == ["pubmed", "arxiv"]
    assert kwargs["per_query_limit"] == 7
    assert kwargs["budget"] == {
        "max_rounds": 2,
        "max_queries": 5,
        "max_papers": 23,
        "max_tokens": 4567,
        "max_seconds": 89,
    }


def test_module_override_cannot_reintroduce_removed_legacy_implementation() -> None:
    config = PipelineConfig(
        search={"implementation": "agentic"},
        module_overrides={"m2": {"kwargs": {"implementation": "agentic"}}},
    )

    assert config.get_module_kwargs("m2")["implementation"] == "agentic"


def test_unknown_search_implementation_is_rejected() -> None:
    with pytest.raises(ValidationError):
        PipelineConfig(search={"implementation": "automatic"})


def test_yaml_config_loads():
    """Every PipelineConfig YAML file should load.

    Skips ``evaluation.yaml`` — it uses MasterEvaluationConfig, not
    PipelineConfig (by design; see TODO Task C.5).
    """
    config_dir = Path(__file__).resolve().parent.parent / "configs"
    skip = {"evaluation.yaml"}
    for yaml_file in config_dir.glob("*.yaml"):
        if yaml_file.name in skip:
            continue
        config = PipelineConfig.from_yaml(str(yaml_file))
        assert config.enabled_modules, f"{yaml_file.name}: no modules enabled"


# --------------------------------------------------------------------------- #
# Rubric — single source of truth for weights / composite
# --------------------------------------------------------------------------- #

def test_rubric_composite_and_weights():
    from hypoforge.evaluation.rubric import composite_score, normalise_weights

    # default weights sum to 1.0
    assert abs(sum(normalise_weights(None).values()) - 1.0) < 1e-9

    # all-equal dimensions → composite equals that value
    scores = {
        "novelty": 0.8, "scientific_soundness": 0.8,
        "testability": 0.8, "evidence_consistency": 0.8,
    }
    assert abs(composite_score(scores) - 0.8) < 1e-6

    # missing dimensions renormalise over what's present (no zero-drag)
    assert abs(composite_score({"novelty": 0.6, "scientific_soundness": 0.6}) - 0.6) < 1e-6


# --------------------------------------------------------------------------- #
# Scorer — tested against a hand-built state (no pipeline run, no LLM)
# --------------------------------------------------------------------------- #

def _make_scored_state() -> PipelineState:
    """A completed-looking PipelineState fixture for scorer tests."""
    h1 = HypothesisCard(
        hypothesis_id="H1",
        statement="NAD+/NADH ratio regulates Hsp70 ATPase cycling in aged cells.",
        observable_predictions=["NMN restores folding activity in aged cells"],
        falsification_conditions=["no change in Hsp70 activity after NMN"],
        scores={"novelty": 0.8, "scientific_soundness": 0.8, "testability": 0.8,
                "evidence_consistency": 0.8, "composite": 0.8},
    )
    h2 = HypothesisCard(
        hypothesis_id="H2",
        statement="Stress granules buffer proteotoxicity by enriching Hsp70.",
        scores={"novelty": 0.6, "scientific_soundness": 0.6, "testability": 0.6,
                "evidence_consistency": 0.6, "composite": 0.6},
    )
    p1 = ResearchPlan(
        hypothesis_id="H1",
        study_subjects="C57BL/6 mice, 8 weeks old",
        timeline="12 months across 3 phases",
    )
    reviews = [ReviewResult(dimension=ReviewerDimension("overall"), score=4.1, version=1)]
    return PipelineState(
        run_id="fixture-001", input_question="q",
        top_hypotheses=[h1, h2], research_plans=[p1], reviews=reviews, iteration_count=1,
    )


@pytest.mark.asyncio
async def test_scorer_report_structure_and_not_circular():
    """Report must separate self-reported from independent, and independent
    metrics must NOT echo the generator's self-reported scores."""
    from hypoforge.evaluation.scorer import score_pipeline_state_async

    report = await score_pipeline_state_async(_make_scored_state())
    assert report["run_id"] == "fixture-001"
    for key in ("hypothesis_scores", "plan_scores", "aggregate", "rubric"):
        assert key in report

    h0 = report["hypothesis_scores"][0]
    assert {"self_reported", "independent", "composite"} <= set(h0)
    # The metrics are now implemented, so they appear in independent.
    # Verify they don't echo the self-reported 0.8 score.
    assert "novelty" in h0["independent"]
    assert h0["independent"]["novelty"] != 0.8
    assert "evidence_consistency" in h0["independent"]
    assert h0["independent"]["evidence_consistency"] != 0.8
    assert "testability" in h0["independent"]              # objectively recomputed
    assert abs(sum(report["rubric"]["weights"].values()) - 1.0) < 1e-6


@pytest.mark.asyncio
async def test_save_scoring_report_writes_file():
    import json
    from hypoforge.evaluation.scorer import save_scoring_report_async

    path = await save_scoring_report_async(_make_scored_state(), "./output")
    assert Path(path).exists(), "scores report was not written"
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    assert data["run_id"] == "fixture-001"
    assert "aggregate" in data


def test_plan_completeness_placeholder_aware():
    """Placeholders / sub-3-char stubs do not count as filled."""
    from hypoforge.evaluation.scorer import score_plan_completeness

    assert score_plan_completeness(ResearchPlan()) == 0.0
    placeholder = ResearchPlan(study_subjects="TBD", timeline="待定", expected_results_if_supported="x")
    assert score_plan_completeness(placeholder) == 0.0
    real = ResearchPlan(study_subjects="C57BL/6 mice, 8 weeks old", timeline="12 months across 3 phases")
    assert 0.0 < score_plan_completeness(real) < 1.0


# --------------------------------------------------------------------------- #
# M4 feedback loop + reason-before-score schema
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_m4_feedback_loop_and_no_hang():
    """Revision rounds build a feedback block from reviews + guidance; interactive
    is off by default so nothing blocks on stdin."""
    from hypoforge.modules.m4_hypothesis_generation import M4HypothesisGeneration

    m4 = M4HypothesisGeneration()  # no client — only helper methods are exercised
    assert m4.interactive is False

    # first round → no feedback block
    assert m4._build_feedback_context(PipelineState(input_question="q"), []) == ""

    # revision round → block cites reviewer suggestions + user guidance
    state = PipelineState(
        input_question="q",
        iteration_count=1,
        reviews=[
            ReviewResult(
                dimension=ReviewerDimension("scientific_logic"),
                attribution="hypothesis",
                score=3.0,
                suggestions="clarify the causal mechanism",
                version=1,
            ),
            ReviewResult(
                dimension=ReviewerDimension("method_feasibility"),
                attribution="plan",
                score=3.0,
                suggestions="add a power analysis",
                version=1,
            ),
        ],
    )
    block = m4._build_feedback_context(state, ["prefer in-vivo models"])
    assert "causal mechanism" in block
    assert "power analysis" not in block
    assert "prefer in-vivo models" in block

    # prompting is a no-op when non-interactive (must not read stdin / hang)
    assert await m4._prompt_user_guidance(state) == ""


def test_review_result_reason_before_score():
    """The reasoning field must precede score so the judge reasons first."""
    fields = list(ReviewResult.model_fields.keys())
    assert "reasoning" in fields
    assert fields.index("reasoning") < fields.index("score")


class _EmptyStandardM4(ModuleProtocol):
    module_name = "m4"
    module_version = "test"
    description = "empty standard M4"

    async def __call__(self, state, config=None):
        return {"candidate_hypotheses": [], "top_hypotheses": []}

    @classmethod
    def get_input_fields(cls):
        return []

    @classmethod
    def get_output_fields(cls):
        return ["candidate_hypotheses", "top_hypotheses"]


class _FailingStandardM5(ModuleProtocol):
    module_name = "m5"
    module_version = "test"
    description = "failing standard M5"

    async def __call__(self, state, config=None):
        raise ValueError("plan generation failed")

    @classmethod
    def get_input_fields(cls):
        return []

    @classmethod
    def get_output_fields(cls):
        return ["research_plans"]


@pytest.mark.asyncio
async def test_standard_core_module_empty_output_fails_closed(tmp_path) -> None:
    runner = PipelineRunner(PipelineConfig(verbose=False, output_dir=str(tmp_path)))
    runner._skills = []
    runner._current_run_id = "core-empty"

    with pytest.raises(RuntimeError, match="top_hypotheses"):
        await runner._make_node_wrapper("m4", _EmptyStandardM4())(
            PipelineState(input_question="q")
        )


@pytest.mark.asyncio
async def test_standard_core_module_exception_is_not_converted_to_soft_error(
    tmp_path,
) -> None:
    runner = PipelineRunner(PipelineConfig(verbose=False, output_dir=str(tmp_path)))
    runner._skills = []
    runner._current_run_id = "core-error"

    with pytest.raises(ValueError, match="plan generation failed"):
        await runner._make_node_wrapper("m5", _FailingStandardM5())(
            PipelineState(input_question="q")
        )


def test_legacy_core_error_cannot_trigger_another_iteration() -> None:
    state = PipelineState(
        input_question="q",
        iteration_count=1,
        max_iterations=3,
        errors=["[m4] contract validation failed"],
    )

    assert _should_continue_iterating(state) == "end"


def test_checkpoint_is_atomically_published_without_temp_residue(tmp_path) -> None:
    import json

    runner = PipelineRunner(PipelineConfig(verbose=False, output_dir=str(tmp_path)))
    runner._current_run_id = "atomic-checkpoint"
    state = PipelineState(input_question="q")

    nested_plan = ResearchPlan(
        hypothesis_id="H1",
        study_subjects="A recursively serialized subject",
    )
    runner._save_checkpoint("m5", state, {
        "metrics": {"complete": True},
        "research_plan_history": {1: [nested_plan]},
    })

    checkpoint = tmp_path / "atomic-checkpoint_checkpoint.json"
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert payload["_last_module"] == "m5"
    assert payload["metrics"] == {"complete": True}
    assert payload["research_plan_history"]["1"][0]["hypothesis_id"] == "H1"
    assert list(tmp_path.glob("*.tmp")) == []


@pytest.mark.asyncio
async def test_pure_resume_seed_skips_completed_modules(tmp_path) -> None:
    """seed_state without followup_text is a pure resume: the completed
    module is skipped (checkpoint semantics) and run_id is rebound."""
    config = PipelineConfig(
        verbose=False,
        output_dir=str(tmp_path),
        enabled_modules=["m1"],
    )
    seed = {
        "run_id": "parent-run",
        "input_question": "蛋白质错误折叠如何导致神经退行性疾病？",
        "problem_card": {
            "original_question": "蛋白质错误折叠如何导致神经退行性疾病？",
        },
        "_last_module": "m1",
    }
    runner = PipelineRunner(config)
    final = await runner.run(
        question="ignored",
        run_id="child-run",
        seed_state=seed,
    )

    assert final.run_id == "child-run"
    # 问题卡来自断点，未被 followup 重置；M1 因输出齐全而跳过。
    assert final.input_question == "蛋白质错误折叠如何导致神经退行性疾病？"
    assert final.problem_card.original_question == (
        "蛋白质错误折叠如何导致神经退行性疾病？"
    )
    assert runner._resume_last_module is None  # 光标已消费（M1 跳过后清空）


def test_run_lock_rejects_a_second_writer_for_the_same_run(tmp_path) -> None:
    runner = PipelineRunner(PipelineConfig(verbose=False, output_dir=str(tmp_path)))
    runner._current_run_id = "single-writer"
    first = runner._acquire_run_lock()
    try:
        with pytest.raises(RuntimeError, match="active writer"):
            runner._acquire_run_lock()
    finally:
        runner._release_run_lock(first)

    second = runner._acquire_run_lock()
    runner._release_run_lock(second)


def test_resume_cursor_skips_last_completed_module_once() -> None:
    card = HypothesisCard(
        hypothesis_id="H1",
        statement="A testable system-level hypothesis.",
    )
    state = PipelineState(
        input_question="q",
        candidate_hypotheses=[card],
        top_hypotheses=[card],
        iteration_count=0,
        max_iterations=2,
    )
    runner = PipelineRunner(PipelineConfig(verbose=False))
    runner._resume_last_module = "m4"
    fields = {"candidate_hypotheses", "top_hypotheses"}

    assert runner._should_skip_module("m4", fields, state) is True
    assert runner._resume_last_module is None
    assert runner._should_skip_module("m4", fields, state) is False


def test_resume_cursor_cannot_skip_incomplete_declared_output() -> None:
    runner = PipelineRunner(PipelineConfig(verbose=False))
    runner._resume_last_module = "m4"

    assert runner._should_skip_module(
        "m4",
        {"candidate_hypotheses", "top_hypotheses"},
        PipelineState(input_question="q"),
    ) is False
    assert runner._resume_last_module is None


def test_domain_router_selects_only_relevant_specialist_sources() -> None:
    from hypoforge.literature.search.domain_routing import route_specialist_sources

    assert route_specialist_sources(
        ["Mathematical Sciences"], "Will Navier-Stokes solutions remain smooth?"
    ) == ["zbmath", "arxiv", "crossref"]
    assert route_specialist_sources(
        ["Medicine & Health"], "What causes autism?"
    ) == ["europe_pmc"]
    assert route_specialist_sources(
        ["Astronomy"], "What is the universe made of?"
    ) == ["ads", "arxiv", "inspire"]
    assert "inspire" not in route_specialist_sources(
        ["Biophysics", "Structural Biology"], "How do proteins fold?"
    )


@pytest.mark.asyncio
async def test_domain_routed_planner_enforces_serper_and_filters_irrelevant_sources() -> None:
    from hypoforge.literature.models import SearchQuery
    from hypoforge.literature.search.domain_routing import DomainRoutedQueryPlanner

    class Delegate:
        async def plan(self, *args, **kwargs):
            return [
                SearchQuery(
                    query_id="irrelevant",
                    text="clinical query",
                    target_source="europe_pmc",
                    purpose="llm_choice",
                    relation_to_question="test",
                )
            ]

    planner = DomainRoutedQueryPlanner(
        Delegate(), ["serper_openalex", "zbmath", "arxiv", "crossref", "europe_pmc"]
    )
    queries = await planner.plan(
        "How are prime numbers distributed?",
        key_entities=["prime numbers"],
        domains=["Mathematical Sciences"],
    )
    targets = [query.target_source for query in queries]
    assert targets[:2] == ["serper_openalex", "serper_openalex"]
    assert {"zbmath", "arxiv", "crossref"}.issubset(targets)
    assert "europe_pmc" not in targets


@pytest.mark.asyncio
async def test_domain_routed_planner_matches_compact_benchmark_call_matrix() -> None:
    from collections import Counter
    from hypoforge.literature.models import SearchQuery
    from hypoforge.literature.search.domain_routing import DomainRoutedQueryPlanner

    class Delegate:
        async def plan(self, *args, **kwargs):
            return [
                SearchQuery(
                    query_id="mechanism", text="Navier Stokes regularity",
                    target_source="arxiv", purpose="core_mechanism",
                    relation_to_question="test",
                ),
                SearchQuery(
                    query_id="review", text="Navier Stokes foundational review",
                    target_source="crossref", purpose="review",
                    relation_to_question="test",
                ),
                SearchQuery(
                    query_id="recent", text="Navier Stokes recent evidence",
                    target_source="arxiv", purpose="recent_research",
                    relation_to_question="test",
                ),
            ]

    planner = DomainRoutedQueryPlanner(
        Delegate(),
        ["serper_openalex", "openalex_oa", "zbmath", "arxiv", "crossref"],
    )
    queries = await planner.plan(
        "Will Navier-Stokes solutions remain smooth?",
        domains=["Mathematical Sciences"],
    )
    counts = Counter(query.target_source for query in queries)
    assert counts == {
        "serper_openalex": 2,
        "zbmath": 2,
        "arxiv": 2,
        "crossref": 2,
    }
    assert "openalex_oa" not in counts


@pytest.mark.asyncio
async def test_domain_routed_expansion_adds_citation_sorted_oa_search() -> None:
    from collections import Counter
    from hypoforge.literature.models import SearchQuery
    from hypoforge.literature.search.domain_routing import DomainRoutedQueryPlanner

    class Delegate:
        async def plan(self, *args, **kwargs):
            return [
                SearchQuery(
                    query_id="evidence", text="protein folding experiments",
                    target_source="europe_pmc", purpose="supporting_evidence",
                    relation_to_question="test",
                ),
                SearchQuery(
                    query_id="open", text="protein folding open preprint",
                    target_source="arxiv", purpose="open_preprint",
                    relation_to_question="test",
                ),
            ]

    planner = DomainRoutedQueryPlanner(
        Delegate(), ["serper_openalex", "openalex_oa", "europe_pmc"],
    )
    queries = await planner.plan(
        "How do proteins fold?",
        domains=["Biology"],
        question_type="open_access_expansion",
    )
    counts = Counter(query.target_source for query in queries)
    assert counts == {
        "serper_openalex": 2,
        "openalex_oa": 2,
        "europe_pmc": 2,
    }
    assert [query.target_source for query in queries[:4]] == [
        "serper_openalex", "serper_openalex", "openalex_oa", "openalex_oa",
    ]


@pytest.mark.asyncio
async def test_domain_routed_planner_timeout_uses_tested_fallback_portfolio() -> None:
    import asyncio
    from collections import Counter
    from hypoforge.literature.search.domain_routing import DomainRoutedQueryPlanner

    class SlowDelegate:
        async def plan(self, *args, **kwargs):
            await asyncio.sleep(1)
            return []

    planner = DomainRoutedQueryPlanner(
        SlowDelegate(),
        ["serper_openalex", "crossref"],
        planner_timeout_seconds=0.01,
    )
    queries = await planner.plan(
        "How do proteins fold?", key_entities=["protein folding"],
        domains=["Chemistry"],
    )
    assert Counter(query.target_source for query in queries) == {
        "serper_openalex": 2,
        "crossref": 2,
    }


def test_openalex_oa_search_uses_filter_and_citation_sort(monkeypatch) -> None:
    from hypoforge.tools import semantic_scholar

    captured = {}

    def fake_get(url, deadline=None):
        captured["url"] = url
        return {"results": [], "meta": {"count": 0}}

    monkeypatch.setattr(semantic_scholar, "_http_get_json", fake_get)
    semantic_scholar._oa_search(
        "protein folding", 5, open_access_only=True,
    )
    assert "filter=is_oa%3Atrue" in captured["url"]
    assert "sort=cited_by_count%3Adesc" in captured["url"]


def test_m1_unknown_candidate_role_degrades_to_other() -> None:
    from hypoforge.modules.m1_problem_understanding import _CandidateEntity

    candidate = _CandidateEntity.model_validate({
        "name": "protein folding",
        "source_mention": "蛋白质是如何折叠的",
        "role": "process",
    })
    assert candidate.role == "other"


@pytest.mark.asyncio
async def test_fulltext_probe_stops_before_reader_for_abstract_only() -> None:
    from hypoforge.literature.models import ContentLevel, DocumentRecord, PaperRecord
    from hypoforge.literature.reading.store import InMemoryChunkStore
    from hypoforge.literature.reading.workflow import FullTextReadingWorkflow

    paper = PaperRecord(paper_id="p", title="paper", sources=["crossref"])

    class Resolver:
        async def resolve(self, value):
            return DocumentRecord(
                document_id="d", paper_id=value.paper_id,
                content_level=ContentLevel.ABSTRACT,
                local_path="/tmp/abstract.json",
            )

    class MustNotRun:
        async def parse(self, document):
            raise AssertionError("abstract must not be parsed as full text")

        async def retrieve(self, *args, **kwargs):
            raise AssertionError("retrieval must not run")

        async def read(self, *args, **kwargs):
            raise AssertionError("reader must not run")

    workflow = FullTextReadingWorkflow(
        resolver=Resolver(), parser=MustNotRun(), retriever=MustNotRun(),
        reader=MustNotRun(), store=InMemoryChunkStore(),
    )
    assert await workflow.probe_parsed_fulltext(paper) is False


@pytest.mark.asyncio
async def test_serper_openalex_failure_keeps_scholar_discovery() -> None:
    from hypoforge.literature.models import PaperRecord, SearchQuery
    from hypoforge.literature.sources.openalex_enriched_scholar import (
        OpenAlexEnrichedScholarSource,
    )

    paper = PaperRecord(paper_id="S:1", title="Protein folding", sources=["serper_scholar"])

    class Scholar:
        async def search(self, query, limit=20):
            return [paper]

    class BrokenOpenAlex:
        async def search(self, query, limit=20):
            raise TimeoutError("OpenAlex unavailable")

    source = OpenAlexEnrichedScholarSource(
        scholar_source=Scholar(), openalex_source=BrokenOpenAlex(),
        enrichment_timeout_seconds=0.1,
    )
    result = await source.search(SearchQuery(
        query_id="q", text="protein folding", target_source="serper_openalex",
        purpose="test", relation_to_question="test",
    ))
    assert result == [paper]


@pytest.mark.asyncio
async def test_enriching_resolver_tries_multiple_oa_locations() -> None:
    from hypoforge.literature.models import ContentLevel, DocumentRecord, PaperRecord
    from hypoforge.literature.reading.access import (
        AccessEnrichment, EnrichingFulltextResolver,
    )

    paper = PaperRecord(paper_id="DOI:x", title="x", doi="10.1/x", sources=["crossref"])

    class Enricher:
        async def enrich(self, value):
            return AccessEnrichment(value, ("https://bad.test/x.pdf", "https://good.test/x.pdf"))

    class Resolver:
        def __init__(self):
            self.urls = []

        async def resolve(self, value):
            url = value.external_ids.get("oa_pdf_url", "")
            self.urls.append(url)
            if "good.test" in url:
                return DocumentRecord(
                    document_id="d:pdf", paper_id=value.paper_id,
                    content_level=ContentLevel.PDF, source_uri=url,
                    local_path="/tmp/paper.pdf",
                )
            return DocumentRecord(
                document_id="d:abstract", paper_id=value.paper_id,
                content_level=ContentLevel.ABSTRACT, local_path="/tmp/abstract.json",
                retrieval_error="invalid PDF",
            )

        async def resolve_abstract(self, value):
            return await self.resolve(value)

    inner = Resolver()
    document = await EnrichingFulltextResolver(inner, Enricher()).resolve(paper)
    assert document.content_level is ContentLevel.PDF
    assert inner.urls == ["https://bad.test/x.pdf", "https://good.test/x.pdf"]


@pytest.mark.asyncio
async def test_pmc_resolver_falls_back_to_europe_pmc_xml(tmp_path) -> None:
    from hypoforge.literature.models import ContentLevel, PaperRecord
    from hypoforge.literature.reading.resolver import PMCFulltextResolver

    async def pmc_backend(url, timeout):
        return b"[Error] no open-access BioC record"

    async def europe_backend(url, timeout):
        return b"<article><body><sec><title>Results</title><p>Readable full text.</p></sec></body></article>"

    resolver = PMCFulltextResolver(
        tmp_path, backend=pmc_backend, europe_pmc_backend=europe_backend,
    )
    document = await resolver.resolve(PaperRecord(
        paper_id="PMC:1", title="x", pmcid="PMC1", sources=["europe_pmc"],
    ))
    assert document.content_level is ContentLevel.STRUCTURED_FULLTEXT
    assert "europepmc" in document.source_uri


@pytest.mark.asyncio
async def test_ntrs_source_preserves_open_pdf_identifier(monkeypatch) -> None:
    from hypoforge.literature.models import FulltextStatus, SearchQuery
    from hypoforge.literature.sources import specialist_sources

    async def fake_get(url, timeout, headers=None):
        assert "/api/citations/search?" in url
        assert "page%5Bsize%5D=2" in url
        return {
            "results": [{
                "id": "20250000001",
                "title": "Mars manufacturing systems",
                "abstract": "An openly available NASA report.",
                "distributionDate": "2025-04-10",
                "authorAffiliations": [{
                    "meta": {"author": {"name": "Ada Researcher"}},
                }],
                "downloads": [{
                    "mimetype": "application/pdf",
                    "links": {"pdf": "/api/citations/20250000001/downloads/report.pdf"},
                }],
            }],
        }

    monkeypatch.setattr(specialist_sources, "_get", fake_get)
    source = specialist_sources.NtrsSource()
    records = await source.search(SearchQuery(
        query_id="q", text="Mars manufacturing", target_source="ntrs",
        purpose="test", relation_to_question="test",
    ), limit=2)

    assert len(records) == 1
    assert records[0].authors == ["Ada Researcher"]
    assert records[0].external_ids["oa_pdf_url"] == (
        "https://ntrs.nasa.gov/api/citations/20250000001/downloads/report.pdf"
    )
    assert records[0].fulltext_status is FulltextStatus.PDF_AVAILABLE


@pytest.mark.asyncio
async def test_high_citation_ranker_prioritizes_absolute_impact() -> None:
    from hypoforge.literature.models import PaperRecord
    from hypoforge.literature.search.ranking import PaperRanker

    papers = [
        PaperRecord(
            paper_id="classic", title="protein folding mechanism", year=1990,
            citation_count=5000, sources=["serper_scholar"],
        ),
        PaperRecord(
            paper_id="recent", title="protein folding mechanism", year=2026,
            citation_count=20, sources=["crossref"],
        ),
    ]
    ranked = await PaperRanker(
        current_year=2026, prefer_high_citation=True
    ).rank("protein folding mechanism", papers, limit=2)
    assert ranked[0].paper_id == "classic"


if __name__ == "__main__":
    # Allow running directly: python tests/test_pipeline.py
    async def _run_all():
        test_default_config_loads()
        print("[PASS] Default config loads")
        test_yaml_config_loads()
        print("[PASS] YAML configs load")
        test_rubric_composite_and_weights()
        print("[PASS] Rubric composite/weights")
        await test_scorer_report_structure_and_not_circular()
        print("[PASS] Scorer report structure / non-circular")
        await test_save_scoring_report_writes_file()
        print("[PASS] scores.json persistence")
        test_plan_completeness_placeholder_aware()
        print("[PASS] Plan completeness placeholder-aware")
        await test_m4_feedback_loop_and_no_hang()
        print("[PASS] M4 feedback loop / no-hang")
        test_review_result_reason_before_score()
        print("[PASS] Reason-before-score field order")
        print("\nAll offline unit tests passed!")
    asyncio.run(_run_all())
