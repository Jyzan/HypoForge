import pytest

from hypoforge.evidence_audit import EvidenceAuditService
from hypoforge.state import EvidenceGraph, EvidenceNode, EvidenceNodeType, HypothesisCard


class ExplodingClient:
    async def structured_chat(self, **kwargs):
        raise AssertionError("the LLM should not be called when no evidence anchor exists")


class StubStructuredClient:
    def __init__(self, payload):
        self.payload = payload

    async def structured_chat(self, **kwargs):
        return self.payload


def graph_with_canonical_evidence(evidence_id: str, label: str) -> EvidenceGraph:
    return EvidenceGraph(nodes=[EvidenceNode(
        id="GCLM_1",
        type=EvidenceNodeType.CLAIM,
        label=label,
        metadata={"evidence_ids": [evidence_id]},
    )])


@pytest.mark.asyncio
async def test_no_anchor_fails_closed_without_llm_call():
    auditor = EvidenceAuditService(client=ExplodingClient())
    verdicts = await auditor.audit_claims(["A causes B"], EvidenceGraph())
    assert verdicts[0].support_status == "unsupported"
    assert verdicts[0].evidence_ids == []


@pytest.mark.asyncio
async def test_topic_similarity_does_not_become_support():
    client = StubStructuredClient({"verdicts": [{
        "claim_index": 0,
        "support_status": "related_only",
        "evidence_ids": ["EVID_1"],
        "rationale": "Same entities, no causal entailment.",
    }]})
    graph = graph_with_canonical_evidence("EVID_1", "A and B were measured together")
    verdict = (await EvidenceAuditService(client).audit_claims(["A causes B"], graph))[0]
    assert verdict.support_status == "related_only"
    assert verdict.evidence_ids == ["EVID_1"]


@pytest.mark.asyncio
async def test_invalid_returned_evidence_ids_are_removed():
    client = StubStructuredClient({"verdicts": [{
        "claim_index": 0,
        "support_status": "direct_support",
        "evidence_ids": ["MADE_UP"],
        "rationale": "claimed support",
    }]})
    verdict = (await EvidenceAuditService(client).audit_claims(
        ["A causes B"], graph_with_canonical_evidence("EVID_1", "A causes B")
    ))[0]
    assert verdict.support_status == "unsupported"
    assert verdict.evidence_ids == []


def test_decompose_hypothesis_keeps_mechanism_as_an_atomic_claim():
    hypothesis = HypothesisCard(
        hypothesis_id="H1",
        statement="A causes B.",
        mechanism="A activates C, which causes B.",
        observable_predictions=["B increases when A increases."],
    )
    claims = EvidenceAuditService(None).decompose_hypothesis(hypothesis)
    assert claims == [
        "A causes B.",
        "A activates C, which causes B.",
        "B increases when A increases.",
    ]
