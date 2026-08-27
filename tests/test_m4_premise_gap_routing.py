import pytest

from hypoforge.evidence_audit import PremiseAuditResult
from hypoforge.graph_context import GraphContext
from hypoforge.modules.m4_hypothesis_generation import M4HypothesisGeneration
from hypoforge.state import HypothesisCard, HypothesisPremise, PipelineState


class PremiseAuditor:
    def __init__(self, results):
        self.results = results
        self.calls = []

    async def audit_premises(self, premises, graph, **kwargs):
        self.calls.append(list(premises))
        return self.results


def card_with_premise():
    return HypothesisCard(
        hypothesis_id="H1",
        statement="A may influence B under C.",
        mechanism="A activates X, which changes B.",
        factual_premises=[HypothesisPremise(
            premise_id="P1",
            claim="A is present in the system.",
            kind="evidence_backed",
            supporting_evidence_ids=["E1"],
        )],
        research_gap="Whether X mediates the change in B.",
    )


@pytest.mark.asyncio
async def test_m4_routes_only_unsupported_required_factual_premises():
    module = M4HypothesisGeneration(fast_mode=False)
    module.evidence_auditor = PremiseAuditor([PremiseAuditResult(
        premise_id="P1",
        claim="A is present in the system.",
        verdict="unsupported",
        rationale="Citation is related but does not entail the premise.",
    )])
    state = PipelineState(input_question="A affects B")
    card = card_with_premise()
    gaps = await module._audit_hypotheses(state, [card], GraphContext())
    assert len(gaps) == 1
    assert gaps[0].premise_id == "P1"
    assert gaps[0].audit_claim == "A is present in the system."
    assert gaps[0].status == "pending"
    assert card.factual_premises[0].audit_verdict == "unsupported"


@pytest.mark.asyncio
async def test_m4_does_not_route_mechanism_or_predictions_as_fact_gaps():
    module = M4HypothesisGeneration(fast_mode=False)
    auditor = PremiseAuditor([])
    module.evidence_auditor = auditor
    card = card_with_premise().model_copy(update={
        "factual_premises": [],
        "observable_predictions": ["B increases when A increases."],
    })
    gaps = await module._audit_hypotheses(
        PipelineState(input_question="A affects B"), [card], GraphContext()
    )
    assert gaps == []
    assert auditor.calls == []
