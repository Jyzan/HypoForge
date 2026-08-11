from __future__ import annotations

import json

import pytest

from hypoforge.modules.m4_hypothesis_generation import M4HypothesisGeneration
from hypoforge.state import PipelineState


def _card(hypothesis_id: str, statement: str) -> dict:
    return {
        "hypothesis_id": hypothesis_id,
        "statement": statement,
        "mechanism": "The intervention changes the measured outcome.",
        "observable_predictions": ["The measured outcome changes."],
        "falsification_conditions": ["The measured outcome does not change."],
    }


class GeneratorRetryClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def structured_chat(self, **kwargs):
        self.calls.append(kwargs)
        is_generator = "statement" in json.dumps(kwargs["output_schema"])
        if is_generator:
            # The prompt itself is intentionally not used for routing; the
            # sequence models a short first pass followed by one retry.
            generator_calls = [
                call for call in self.calls
                if "statement" in json.dumps(call["output_schema"])
            ]
            if len(generator_calls) == 1:
                return [_card("H1", "The intervention improves the measured outcome.")]
            return [_card("H1", "The intervention improves the measured outcome under stress.")]
        return []


@pytest.mark.asyncio
async def test_m4_retries_once_when_generator_returns_too_few_candidates() -> None:
    client = GeneratorRetryClient()
    module = M4HypothesisGeneration(num_candidates=3, top_k=2, mode="direct")
    module.client = client
    module._check_context_contract = lambda state, context, cards, **kwargs: (cards, [])
    module._audit_context_contract_semantics = (
        lambda state, cards: _identity_audit(cards)
    )

    result = await module._run_llm(PipelineState(input_question="How does it work?"))

    generator_calls = [
        call for call in client.calls
        if "statement" in json.dumps(call["output_schema"])
    ]
    assert len(generator_calls) == 2
    assert len(result["candidate_hypotheses"]) == 2
    assert len(result["top_hypotheses"]) == 2


class DegradedGateM4(M4HypothesisGeneration):
    async def _run_critic(self, state, candidates):
        raise RuntimeError("all candidates rejected")

    async def _run_falsifiability(self, state, candidates):
        raise RuntimeError("falsifiability unavailable")

    def _attach_rankings(self, candidates, payload, context=None):
        raise RuntimeError("ranker returned too few rows")


@pytest.mark.asyncio
async def test_m4_continues_with_short_retry_result_when_quality_gates_fail() -> None:
    client = GeneratorRetryClient()
    module = DegradedGateM4(num_candidates=3, top_k=1, mode="multi_agent")
    module.client = client
    module.ranker_client = client
    module._check_context_contract = lambda state, context, cards, **kwargs: (cards, [])
    module._audit_context_contract_semantics = (
        lambda state, cards: _identity_audit(cards)
    )

    result = await module._run_llm(PipelineState(input_question="How does it work?"))

    assert len(result["candidate_hypotheses"]) == 2
    assert len(result["top_hypotheses"]) == 1


async def _identity_audit(cards):
    return cards, []
