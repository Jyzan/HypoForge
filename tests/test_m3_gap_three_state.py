import pytest

from hypoforge.evidence_audit import PremiseAuditResult
from hypoforge.modules.m3_evidence_graph import M3EvidenceGraph
from hypoforge.state import EvidenceGapRequest, EvidenceGraph, PipelineState


class PremiseAuditor:
    def __init__(self, result):
        self.result = result
        self.calls = []

    async def audit_premises(self, premises, graph, **kwargs):
        self.calls.append((list(premises), kwargs))
        return [self.result]


class BridgeClient:
    async def structured_chat(self, **kwargs):
        return {
            "statement": "A may influence B through C.",
            "falsifiable_prediction": "Blocking C removes the effect.",
            "involved_entity_ids": [],
            "rationale": "The relation remains unverified.",
        }


def state_for(result_status="searched"):
    gap = EvidenceGapRequest(
        gap_id="GAP_1",
        premise_id="P1",
        hypothesis_id="H1",
        sub_question="What evidence tests A is present?",
        audit_claim="A is present.",
        status=result_status,
        attempts=1,
        search_completed=True,
    )
    return PipelineState(
        input_question="A affects B",
        evidence_graph=EvidenceGraph(),
        evidence_gap_requests=[gap],
        max_evidence_gap_rounds=2,
    )


@pytest.mark.asyncio
async def test_supported_gap_closes_with_scientific_resolution():
    module = M3EvidenceGraph(mode="rule")
    module.evidence_auditor = PremiseAuditor(PremiseAuditResult(
        premise_id="P1", claim="A is present.", verdict="supported",
        evidence_ids=["E1"], rationale="Canonical evidence entails the premise.",
    ))
    state = state_for()
    _, gaps = await module._resolve_m4_gap_requests(
        state, state.evidence_graph, state.evidence_gap_requests,
    )
    assert gaps[0].status == "resolved"
    assert gaps[0].scientific_resolution == "supported"
    assert gaps[0].resolution_evidence_ids == ["E1"]


@pytest.mark.asyncio
async def test_contradicted_gap_closes_without_creating_bridge():
    module = M3EvidenceGraph(mode="rule")
    module.evidence_auditor = PremiseAuditor(PremiseAuditResult(
        premise_id="P1", claim="A is present.", verdict="contradicted",
        evidence_ids=["E2"], rationale="Canonical evidence contradicts it.",
    ))
    state = state_for()
    graph, gaps = await module._resolve_m4_gap_requests(
        state, state.evidence_graph, state.evidence_gap_requests,
    )
    assert gaps[0].status == "resolved"
    assert gaps[0].scientific_resolution == "contradicted"
    assert gaps[0].contradicting_evidence_ids == ["E2"]
    assert graph.nodes == []


@pytest.mark.asyncio
async def test_unresolved_completed_search_creates_unverified_bridge_immediately():
    module = M3EvidenceGraph(mode="rule")
    module.client = BridgeClient()
    module.evidence_auditor = PremiseAuditor(PremiseAuditResult(
        premise_id="P1", claim="A is present.", verdict="unsupported",
        evidence_ids=[], rationale="Search completed but no entailment was found.",
    ))
    state = state_for()
    graph, gaps = await module._resolve_m4_gap_requests(
        state, state.evidence_graph, state.evidence_gap_requests,
    )
    assert gaps[0].status == "hypothesized"
    assert gaps[0].scientific_resolution == "unresolved"
    assert gaps[0].bridge_hypothesis_node_id == "HYP_1"
    assert graph.nodes[0].metadata["verification_status"] == "unverified"
