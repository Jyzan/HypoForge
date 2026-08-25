import pytest

from hypoforge.evidence_audit import ClaimEvidenceVerdict, EvidenceAuditService
from hypoforge.modules.m3_evidence_graph import M3EvidenceGraph
from hypoforge.state import (
    EvidenceGapRequest,
    EvidenceGraph,
    EvidenceNode,
    EvidenceNodeType,
    PipelineState,
)


class StubAuditor:
    def __init__(self, verdict):
        self.verdict = verdict

    async def audit_claims(self, claims, graph, **kwargs):
        return [self.verdict.model_copy(update={"claim": claims[0]})]


class StubBridgeClient:
    async def structured_chat(self, **kwargs):
        return {
            "statement": "A may influence B through C.",
            "falsifiable_prediction": "Blocking C removes the effect.",
            "involved_entity_ids": [],
            "rationale": "A bridge remains unverified after bounded search.",
        }


def searched_gap_state(*, attempts: int, max_rounds: int) -> PipelineState:
    gap = EvidenceGapRequest(
        gap_id="GAP_1",
        hypothesis_id="H1",
        sub_question="What evidence tests A causes B through C?",
        audit_claim="A causes B through C.",
        status="searched",
        attempts=attempts,
    )
    return PipelineState(
        input_question="A causes B",
        evidence_graph=EvidenceGraph(),
        evidence_gap_requests=[gap],
        max_evidence_gap_rounds=max_rounds,
    )


@pytest.mark.asyncio
async def test_related_new_paper_does_not_resolve_gap():
    module = M3EvidenceGraph(mode="rule")
    module.evidence_auditor = StubAuditor(ClaimEvidenceVerdict(
        claim="A causes B through C.",
        support_status="related_only",
        evidence_ids=["EVID_1"],
        rationale="Related topic only",
    ))
    state = searched_gap_state(attempts=0, max_rounds=2)
    graph, gaps = await module._resolve_m4_gap_requests(
        state, state.evidence_graph, state.evidence_gap_requests,
    )
    assert gaps[0].status == "pending"
    assert gaps[0].resolution_evidence_ids == []
    assert not graph.nodes


@pytest.mark.asyncio
async def test_direct_support_resolves_gap():
    module = M3EvidenceGraph(mode="rule")
    module.evidence_auditor = StubAuditor(ClaimEvidenceVerdict(
        claim="A causes B through C.",
        support_status="direct_support",
        evidence_ids=["EVID_2"],
        rationale="Direct support",
    ))
    state = searched_gap_state(attempts=1, max_rounds=1)
    graph, gaps = await module._resolve_m4_gap_requests(
        state, state.evidence_graph, state.evidence_gap_requests,
    )
    assert gaps[0].status == "resolved"
    assert gaps[0].resolution_evidence_ids == ["EVID_2"]
    assert not graph.nodes


@pytest.mark.asyncio
async def test_no_support_at_budget_creates_one_unverified_bridge():
    module = M3EvidenceGraph(mode="rule")
    module.client = StubBridgeClient()
    module.evidence_auditor = StubAuditor(ClaimEvidenceVerdict(
        claim="A causes B through C.",
        support_status="unsupported",
        evidence_ids=[],
        rationale="No suitable evidence",
    ))
    state = searched_gap_state(attempts=1, max_rounds=1)
    graph, gaps = await module._resolve_m4_gap_requests(
        state, state.evidence_graph, state.evidence_gap_requests,
    )
    nodes = [node for node in graph.nodes if node.id == "HYP_1"]
    assert len(nodes) == 1
    assert gaps[0].status == "hypothesized"
    assert gaps[0].bridge_hypothesis_node_id == "HYP_1"
    assert nodes[0].metadata["verification_status"] == "unverified"
    assert nodes[0].metadata["supporting_evidence_ids"] == []


@pytest.mark.asyncio
async def test_unrelated_historical_evidence_cannot_resolve_gap():
    class ExplodingClient:
        async def structured_chat(self, **kwargs):
            raise AssertionError("no historical evidence should be considered for this gap")

    module = M3EvidenceGraph(mode="rule")
    module.evidence_auditor = EvidenceAuditService(ExplodingClient())
    state = searched_gap_state(attempts=0, max_rounds=2)
    state.evidence_graph.nodes.append(EvidenceNode(
        # This is canonical graph evidence, but it did not originate from GAP_1.
        # The gap-specific candidate whitelist must exclude it.
        id="GCLM_OLD",
        type=EvidenceNodeType.CLAIM,
        label="A causes B through C.",
        metadata={"evidence_ids": ["EVID_OLD"]},
    ))
    _, gaps = await module._resolve_m4_gap_requests(
        state, state.evidence_graph, state.evidence_gap_requests,
    )
    assert gaps[0].status == "pending"
