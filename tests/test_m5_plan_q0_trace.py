"""Regression tests for M5 whole-question Q0 trace generation."""

from hypoforge.modules.m5_research_plan import M5ResearchPlan
from hypoforge.synthesis_contract import SYNTHESIS_REQUIREMENT_ID, synthesis_contract_for_state
from hypoforge.state import (
    PipelineState,
    ProblemCard,
    ResearchPlan,
    TaskContract,
    TaskEntity,
    TaskRequirement,
    TaskTrace,
    TaskTraceReference,
)
from hypoforge.task_alignment import assess_task_alignment


def _state() -> PipelineState:
    entities = [
        TaskEntity(
            entity_id="E1",
            name="common cold",
            source_mention="the common cold",
            role="primary_object",
            required=True,
        ),
        TaskEntity(
            entity_id="E2",
            name="rhinovirus",
            source_mention="common cold viruses",
            role="context",
            required=False,
        ),
    ]
    contract = TaskContract(
        source="m1",
        entities=entities,
        requirements=[
            TaskRequirement(
                requirement_id="R1",
                sub_question="Will we ever find a cure for the common cold?",
                primary_entity_id="E1",
                related_entity_ids=["E2"],
                relation="find a cure",
                required=True,
            ),
        ],
    )
    card = ProblemCard(
        original_question="Will we ever find a cure for the common cold?",
        domain=["Medicine & Health"],
        sub_questions=["Will we ever find a cure for the common cold?"],
        key_entities=["common cold", "rhinovirus"],
        task_contract=contract,
    )
    return PipelineState(input_question=card.original_question, problem_card=card)


def _plan() -> ResearchPlan:
    return ResearchPlan(
        hypothesis_id="H1",
        study_subjects="普通感冒患者和鼻病毒感染的呼吸道细胞模型",
        independent_variables=["候选药物干预"],
        dependent_variables=["病毒载量、症状持续时间"],
        control_groups=["安慰剂对照"],
        procedures=["招募患者，进行药物干预，测量病毒载量和症状变化。"],
        measurement_metrics=["病毒载量、症状评分"],
        analysis_methods=["统计检验"],
        expected_results_if_supported="药物显著降低病毒载量和症状持续时间。",
        expected_results_if_refuted="药物无显著效果。",
        timeline="Months 1-3: 招募和干预。",
        risks_and_alternatives="Risk: 样本不足。 Alternative: 扩大样本。",
        task_trace=TaskTrace(
            entity_mentions=[
                TaskTraceReference(
                    contract_id="E1",
                    output_excerpt="普通感冒患者",
                ),
            ],
            requirement_mentions=[],
        ),
    )


def test_m5_canonicalize_always_carries_q0_whole_question_trace():
    state = _state()
    plan = _plan()

    plan = M5ResearchPlan._canonicalize_task_trace(
        state,
        plan,
        requirement_ids={SYNTHESIS_REQUIREMENT_ID},
    )

    req_ids = [r.contract_id for r in plan.task_trace.requirement_mentions]
    assert SYNTHESIS_REQUIREMENT_ID in req_ids

    alignment = assess_task_alignment(
        state,
        M5ResearchPlan._plan_alignment_text(plan),
        subject_text=plan.study_subjects,
        trace=plan.task_trace,
        semantic_client=None,
        required_requirement_ids={SYNTHESIS_REQUIREMENT_ID},
        contract_override=synthesis_contract_for_state(state),
    )
    assert alignment.passed is True
    assert SYNTHESIS_REQUIREMENT_ID not in alignment.missing_requirement_ids


def test_fast_repair_injects_primary_object_when_plan_uses_proxy_language():
    """Fast mode's lightweight repair must keep the primary object visible."""
    state = _state()
    plan = _plan()

    repaired = M5ResearchPlan._fast_repair_plan_alignment(state, plan)

    assert "common cold" in repaired.study_subjects
    assert "Original research target" in repaired.study_subjects


def test_fast_repair_preserves_existing_primary_mention():
    """If the plan already names the primary object, repair should not strip it."""
    state = _state()
    plan = _plan().model_copy(update={
        "study_subjects": "common cold patients and rhinovirus-infected airway cells",
    })

    repaired = M5ResearchPlan._fast_repair_plan_alignment(state, plan)

    assert "common cold" in repaired.study_subjects
    assert "Proposed study system" in repaired.study_subjects


def test_m5_semantic_audit_accepts_models_as_means_when_primary_object_is_anchored():
    import asyncio
    from hypoforge.modules.m5_research_plan import M5ResearchPlan

    class FakeChatClient:
        def __init__(self):
            self.prompt = ""
        async def chat(self, user_prompt, **kwargs):
            self.prompt = user_prompt
            return "yes"

    module = M5ResearchPlan()
    module.client = FakeChatClient()
    state = _state()
    plan = _plan()
    # Ensure plan is anchored by fast repair
    plan = M5ResearchPlan._fast_repair_plan_alignment(state, plan)

    result = asyncio.run(module._audit_plan_semantics(state, plan))
    assert result[0] is True
    assert "means to study it" in module.client.prompt
