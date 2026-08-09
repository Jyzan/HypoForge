from __future__ import annotations

from typing import Any

import pytest

from hypoforge.modules.m4_hypothesis_generation import M4HypothesisGeneration
from hypoforge.modules.m5_research_plan import M5ResearchPlan
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

    result = await module._run_llm(PipelineState(input_question="q", problem_card=ProblemCard(original_question="q", sub_questions=["q"], key_entities=[], domain=[])), "revision")

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
                score=3.0,
                suggestions="clarify the causal mechanism",
                version=1,
            ),
            ReviewResult(
                dimension=ReviewerDimension("method_feasibility"),
                score=3.0,
                suggestions="add a power analysis",
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
