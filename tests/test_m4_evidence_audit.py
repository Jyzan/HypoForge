from unittest.mock import AsyncMock

import pytest

from hypoforge.graph_context import GraphContext
from hypoforge.evidence_audit import ClaimEvidenceVerdict, PremiseAuditResult
from hypoforge.modules.m4_hypothesis_generation import M4HypothesisGeneration
from hypoforge.state import (
    EvidenceGapRequest,
    HypothesisCard,
    HypothesisPremise,
    PipelineState,
)


class StubAuditor:
    def __init__(self, verdicts):
        self.verdicts = verdicts

    @staticmethod
    def decompose_hypothesis(hypothesis):
        return [hypothesis.mechanism or hypothesis.statement]

    async def audit_claims(self, claims, graph, **kwargs):
        return self.verdicts


def hypothesis():
    return HypothesisCard(
        hypothesis_id="H1",
        statement="A causes B.",
        mechanism="A causes B through C.",
    )


class StubPremiseAuditor:
    def __init__(self, verdict):
        self.verdict = verdict

    async def audit_premises(self, premises, graph, **kwargs):
        return [self.verdict]


@pytest.mark.asyncio
async def test_valid_id_without_entailment_creates_gap():
    module = M4HypothesisGeneration(fast_mode=False)
    module.evidence_auditor = StubAuditor([
        ClaimEvidenceVerdict(
            claim="A causes B through C.",
            support_status="related_only",
            evidence_ids=["EVID_1"],
            rationale="Related topic only",
        )
    ])
    state = PipelineState(input_question="A causes B")
    gaps = await module._audit_hypotheses(state, [hypothesis()], GraphContext())
    gap = gaps[0]
    assert gap.audit_claim == "A causes B through C."
    assert gap.status == "pending"


@pytest.mark.asyncio
async def test_direct_support_resolves_existing_gap():
    module = M4HypothesisGeneration(fast_mode=False)
    module.evidence_auditor = StubAuditor([
        ClaimEvidenceVerdict(
            claim="A causes B through C.",
            support_status="direct_support",
            evidence_ids=["EVID_2"],
            rationale="Direct causal support",
        )
    ])
    state = PipelineState(
        input_question="A causes B",
        evidence_gap_requests=[EvidenceGapRequest(
            gap_id="GAP_OLD",
            hypothesis_id="H1",
            sub_question="What evidence tests A causes B through C.?",
            audit_claim="A causes B through C.",
            status="searched",
            attempts=1,
        )],
    )
    gaps = await module._audit_hypotheses(state, [hypothesis()], GraphContext())
    gap = gaps[0]
    assert gap.status == "resolved"
    assert gap.resolution_evidence_ids == ["EVID_2"]


@pytest.mark.asyncio
async def test_hypothesized_gap_is_not_reopened():
    module = M4HypothesisGeneration(fast_mode=False)
    module.evidence_auditor = StubAuditor([
        ClaimEvidenceVerdict(
            claim="A causes B through C.",
            support_status="unsupported",
            evidence_ids=[],
            rationale="No suitable evidence",
        )
    ])
    state = PipelineState(
        input_question="A causes B",
        evidence_gap_requests=[EvidenceGapRequest(
            gap_id="GAP_OLD",
            hypothesis_id="H1",
            sub_question="What evidence tests A causes B through C.?",
            audit_claim="A causes B through C.",
            status="hypothesized",
            bridge_hypothesis_node_id="HYP_OLD",
            attempts=1,
        )],
    )
    gaps = await module._audit_hypotheses(state, [hypothesis()], GraphContext())
    assert not [gap for gap in gaps if gap.status == "pending"]
    assert gaps[0].status == "hypothesized"


@pytest.mark.asyncio
async def test_unsupported_final_audit_removes_stale_supporting_evidence_ids():
    module = M4HypothesisGeneration(fast_mode=False)
    module.evidence_auditor = StubPremiseAuditor(PremiseAuditResult(
        premise_id="P1",
        claim="Amyloid formation is linked to approximately 50 human diseases.",
        verdict="unsupported",
        evidence_ids=["E1"],
        rationale="The selected snippet does not entail the numerical claim.",
    ))
    card = HypothesisCard(
        hypothesis_id="H1",
        statement="Test an amyloid mechanism.",
        factual_premises=[HypothesisPremise(
            premise_id="P1",
            claim="Amyloid formation is linked to approximately 50 human diseases.",
            kind="evidence_backed",
            supporting_evidence_ids=["E1"],
            audit_verdict="supported",
        )],
    )

    await module._audit_hypotheses(
        PipelineState(input_question="How does protein misfolding cause disease?"),
        [card],
        GraphContext(available_evidence_ids=["E1"]),
    )

    premise = card.factual_premises[0]
    assert premise.audit_verdict == "unsupported"
    assert premise.supporting_evidence_ids == []


def test_final_audited_top_card_replaces_stale_candidate_copy():
    candidate = HypothesisCard(
        hypothesis_id="H1",
        statement="Candidate",
        factual_premises=[HypothesisPremise(
            premise_id="P1",
            claim="Claim",
            kind="evidence_backed",
            supporting_evidence_ids=["E1"],
            audit_verdict="supported",
        )],
    )
    audited_top = candidate.model_copy(deep=True)
    audited_top.factual_premises[0].audit_verdict = "unsupported"
    audited_top.factual_premises[0].supporting_evidence_ids = []
    untouched = HypothesisCard(hypothesis_id="H2", statement="Untouched")

    synchronized = M4HypothesisGeneration._sync_audited_top_to_candidates(
        [candidate, untouched], [audited_top],
    )

    by_id = {card.hypothesis_id: card for card in synchronized}
    assert by_id["H1"].factual_premises[0].audit_verdict == "unsupported"
    assert by_id["H1"].factual_premises[0].supporting_evidence_ids == []
    assert by_id["H2"] is untouched
