"""Regression tests for M4 clarification/refinement flow."""

from hypoforge.modules.m4_hypothesis_generation import M4RefinementRequired
from hypoforge.pipeline import _route_after_m4
from hypoforge.state import (
    ClarificationRequest,
    HypothesisCard,
    PipelineState,
)


def test_m4_refinement_exception_carries_candidates():
    candidates = [
        HypothesisCard(
            hypothesis_id="H1",
            statement="A testable narrow direction.",
        ),
    ]
    clarification = ClarificationRequest(
        original_question="What is gravity?",
        message="Please refine the question.",
        suggested_directions=["A testable narrow direction."],
        rejected_hypothesis_ids=["H1"],
    )
    exc = M4RefinementRequired(clarification, candidates)
    assert exc.clarification is clarification
    assert exc.candidates == candidates
    assert "refine" in str(exc)


def test_route_after_m4_returns_clarify_when_clarification_request_present():
    state = PipelineState(
        input_question="What is gravity?",
        clarification_request=ClarificationRequest(
            original_question="What is gravity?",
            message="Please refine.",
        ),
    )
    assert _route_after_m4(state) == "clarify"


def test_core_validation_allows_m4_clarification_without_top_hypotheses():
    from hypoforge.pipeline import PipelineRunner
    from hypoforge.config import PipelineConfig

    result = {
        "candidate_hypotheses": [
            HypothesisCard(hypothesis_id="H1", statement="Narrow direction"),
        ],
        "top_hypotheses": [],
        "clarification_request": ClarificationRequest(
            original_question="What is gravity?",
            message="Please refine.",
        ),
    }
    # Should not raise.
    PipelineRunner._validate_core_result(
        "m4",
        {"candidate_hypotheses", "top_hypotheses"},
        result,
    )


def test_m4_first_gate_rejection_requests_user_refinement():
    import pytest
    from hypoforge.modules.m4_hypothesis_generation import M4RefinementRequired
    from hypoforge.strict_contracts import StrictM4HypothesisGeneration

    state = PipelineState(input_question="What is gravity?")
    candidates = [
        HypothesisCard(
            hypothesis_id="H1",
            statement="Loop quantum gravity is the correct theory.",
        ),
        HypothesisCard(
            hypothesis_id="H2",
            statement="Quantum conformal gravity is the correct theory.",
        ),
    ]
    module = StrictM4HypothesisGeneration(mode="multi_agent", fast_mode=False)
    with pytest.raises(M4RefinementRequired) as exc_info:
        module._raise_refinement_required(
            state,
            candidates,
            gate_name="M4 Critic",
            excerpts="candidate only covers one theory.",
        )
    clarification = exc_info.value.clarification
    assert clarification.original_question == "What is gravity?"
    assert len(clarification.suggested_directions) == 2
    assert clarification.rejected_hypothesis_ids == ["H1", "H2"]


def test_should_request_refinement_for_broad_definitional_question():
    from hypoforge.strict_contracts import StrictM4HypothesisGeneration
    assert StrictM4HypothesisGeneration._should_request_refinement(
        "What is gravity?",
        "The candidate is logically inconsistent.",
    ) is True
    assert StrictM4HypothesisGeneration._should_request_refinement(
        "什么是宇宙？",
        "The candidate only discusses one component.",
    ) is True


def test_should_not_request_refinement_for_specific_question_with_logic_critique():
    from hypoforge.strict_contracts import StrictM4HypothesisGeneration
    assert StrictM4HypothesisGeneration._should_request_refinement(
        "Does inhibiting enzyme X reduce tumor growth in mice?",
        "The causal chain is incomplete and the prediction is not specific.",
    ) is False


def test_should_request_refinement_when_critic_reasons_indicate_whole_question_missing():
    from hypoforge.strict_contracts import StrictM4HypothesisGeneration
    assert StrictM4HypothesisGeneration._should_request_refinement(
        "A specific but broad-sounding topic?",
        "The hypothesis only addresses one component and fails to answer the whole question.",
    ) is True
