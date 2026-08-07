from __future__ import annotations

import pytest
from pathlib import Path

from hypoforge.graph_context import build_graph_context
from hypoforge.modules.m1_problem_understanding import M1ProblemUnderstanding
from hypoforge.modules.m4_hypothesis_generation import M4HypothesisGeneration
from hypoforge.modules.m5_research_plan import M5ResearchPlan
from hypoforge.modules.m6_review_iteration import M6ReviewIteration
from hypoforge.modules.m3_evidence_graph import M3EvidenceGraph
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
    client = HypothesisRepairClient([[invalid], [repaired]])
    module = M4HypothesisGeneration(num_candidates=1, top_k=1, mode="direct")
    module.client = client

    result = await module._run_llm(state)

    assert len(client.calls) == 2
    assert client.calls[1]["temperature"] == 0.0
    assert "missing required task entities" in client.calls[1]["user_prompt"]
    assert result["top_hypotheses"][0].statement == valid_statement


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
        "机制是什么；环境应激又如何影响它？？",
        "域随机化如何影响机械臂迁移成功率？",
    ])

    assert len(violations) == 1
    assert "multiple question marks" in violations[0]


class PlanClient:
    def __init__(self) -> None:
        self.calls = []

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
    evidence_review = next(
        review for review in reviewed.reviews
        if review.dimension.value == "evidence_consistency"
    )
    overall = next(
        review for review in reviewed.reviews
        if review.dimension.value == "overall"
    )

    assert evidence_review.score == 2.0
    assert evidence_review.hard_gate_passed is False
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
    module.client = ReviewClient()

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
