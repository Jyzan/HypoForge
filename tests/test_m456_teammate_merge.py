from __future__ import annotations

from hypoforge.modules.m4_hypothesis_generation import M4HypothesisGeneration
from hypoforge.prompts.m4_prompts import M4_GENERATOR_USER_TEMPLATE
from hypoforge.prompts.m5_prompts import M5_SYSTEM_PROMPT
from hypoforge.state import (
    PipelineState,
    ProblemCard,
    TaskContract,
    TaskEntity,
    TaskRequirement,
)


def _contract_state() -> PipelineState:
    question = "How can offline reinforcement learning transfer robot-arm policies from simulation to reality?"
    return PipelineState(
        input_question=question,
        problem_card=ProblemCard(
            original_question=question,
            task_contract=TaskContract(
                entities=[
                    TaskEntity(
                        entity_id="E1",
                        name="robot arm",
                        aliases=["robotic manipulator"],
                        role="primary_object",
                        required=True,
                    ),
                    TaskEntity(
                        entity_id="E2",
                        name="offline reinforcement learning",
                        role="method",
                        required=True,
                    ),
                ],
                requirements=[
                    TaskRequirement(
                        requirement_id="R1",
                        sub_question="How can the policy cross the sim-to-real gap?",
                        primary_entity_id="E1",
                        related_entity_ids=["E2"],
                        relation="transfer from simulation to reality",
                    )
                ],
            ),
        ),
    )


def test_m4_renders_exact_task_contract_for_the_generator() -> None:
    state = _contract_state()

    block = M4HypothesisGeneration._render_task_contract_block(state)
    prompt = M4_GENERATOR_USER_TEMPLATE.format(
        graph_context="graph",
        knowledge_gaps="gaps",
        established_facts="facts",
        conflicts="conflicts",
        original_question=state.input_question,
        feedback_context="",
        task_contract_block=block,
        num_candidates=3,
    )

    assert "Binding task contract" in prompt
    assert "E1: robot arm" in prompt
    assert "robotic manipulator" in prompt
    assert "R1: transfer from simulation to reality" in prompt


def test_m5_prompt_allows_only_matching_language_contract_aliases() -> None:
    normalized_prompt = " ".join(M5_SYSTEM_PROMPT.split())

    assert "same language as the original question" in M5_SYSTEM_PROMPT
    assert "alias that matches the plan's language" in M5_SYSTEM_PROMPT
    assert "Never embed a Chinese contract name inside an English sentence" in normalized_prompt
