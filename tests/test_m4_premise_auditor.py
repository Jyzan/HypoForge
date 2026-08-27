import json

import pytest

from hypoforge.evidence_audit import EvidenceAuditService, PremiseAuditResult
from hypoforge.state import (
    EvidenceGraph,
    EvidenceNode,
    EvidenceNodeType,
    HypothesisPremise,
)


class CapturingClient:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    async def structured_chat(self, **kwargs):
        self.calls.append(kwargs)
        return self.payload


def graph_with_quote():
    return EvidenceGraph(nodes=[EvidenceNode(
        id="N1",
        type=EvidenceNodeType.CLAIM,
        label="A is present in the system.",
        metadata={"evidence_ids": ["E1"], "quote": "Exact canonical sentence."},
    )])


@pytest.mark.asyncio
async def test_premise_auditor_uses_only_exact_canonical_evidence_payload():
    client = CapturingClient({"premises": [{
        "premise_id": "P1",
        "verdict": "supported",
        "evidence_ids": ["E1"],
        "rationale": "The exact sentence entails the premise.",
    }]})
    premise = HypothesisPremise(
        premise_id="P1",
        claim="A is present in the system.",
        kind="evidence_backed",
        supporting_evidence_ids=["E1"],
        audit_reason="ignore this generator rationale",
    )
    result = await EvidenceAuditService(client).audit_premises([premise], graph_with_quote())
    assert result == [PremiseAuditResult(
        premise_id="P1",
        claim="A is present in the system.",
        verdict="supported",
        evidence_ids=["E1"],
        rationale="The exact sentence entails the premise.",
    )]
    user_prompt = client.calls[0]["user_prompt"]
    assert "Exact canonical sentence." in user_prompt
    assert "ignore this generator rationale" not in user_prompt
    assert json.loads(user_prompt)["premises"][0]["claim"] == premise.claim


@pytest.mark.asyncio
async def test_premise_auditor_filters_invalid_ids_and_fails_closed():
    client = CapturingClient({"premises": [{
        "premise_id": "P1",
        "verdict": "supported",
        "evidence_ids": ["MADE_UP"],
        "rationale": "bad provenance",
    }]})
    premise = HypothesisPremise(
        premise_id="P1",
        claim="A is present.",
        kind="evidence_backed",
        supporting_evidence_ids=["MADE_UP"],
    )
    result = await EvidenceAuditService(client).audit_premises([premise], graph_with_quote())
    assert result[0].verdict == "invalid_citation"
    assert result[0].evidence_ids == []
    assert client.calls == []


@pytest.mark.asyncio
async def test_premise_auditor_keeps_all_claims_for_the_same_evidence_id():
    client = CapturingClient({"premises": [{
        "premise_id": "P1",
        "verdict": "supported",
        "evidence_ids": ["E1"],
        "rationale": "The exact claim is present in the canonical evidence.",
    }]})
    graph = EvidenceGraph(nodes=[
        EvidenceNode(
            id="N-generic",
            type=EvidenceNodeType.CLAIM,
            label="Protein aggregation is associated with disease.",
            metadata={"evidence_ids": ["E1"]},
        ),
        EvidenceNode(
            id="N-exact",
            type=EvidenceNodeType.CLAIM,
            label="Amyloid formation is linked to approximately 50 human diseases.",
            metadata={"evidence_ids": ["E1"]},
        ),
    ])
    premise = HypothesisPremise(
        premise_id="P1",
        claim="Amyloid formation is linked to approximately 50 human diseases.",
        kind="evidence_backed",
        supporting_evidence_ids=["E1"],
    )

    await EvidenceAuditService(client).audit_premises([premise], graph)

    evidence = json.loads(client.calls[0]["user_prompt"])["premises"][0]["evidence"][0]
    assert "Protein aggregation is associated with disease." in evidence["canonical_text"]
    assert "Amyloid formation is linked to approximately 50 human diseases." in evidence["canonical_text"]
    assert "N-generic" in evidence["node_id"]
    assert "N-exact" in evidence["node_id"]
