"""Integration tests for M3 grounding: direct vs Evidence-GAMS selection modes."""

import pytest

from hypoforge.modules.m3_grounding.workflow import GroundingWorkflow
from hypoforge.modules.m3_grounding.models import (
    AtomicClaim,
    EvidenceRecord,
    GroundingReport,
    RelationCandidate,
)
from hypoforge.state import EvidenceEdgeRelation, EvidenceEdge, EvidenceGraph, EvidenceNode, EvidenceNodeType


def _evidence(record_id: str, paper_id: str) -> EvidenceRecord:
    return EvidenceRecord(
        id="ER_" + record_id,
        evidence_id=record_id,
        paper_id=paper_id,
        query="test query",
        summary=f"summary {record_id}",
        excerpt=f"excerpt {record_id}",
        relevance_score=8.0,
        retrieval_score=0.8,
        claims=[f"Scientific claim from {record_id} with sufficient length for testing purposes."],
    )


@pytest.mark.asyncio
async def test_direct_and_gams_select_differently(tmp_path) -> None:
    """GAMS should filter out low-confidence edges that direct mode accepts."""
    records = [_evidence("E1", "P1"), _evidence("E2", "P2")]
    claims = [
        AtomicClaim(
            id="C1",
            statement="Claim one has enough scientific content for testing.",
            evidence_ids=["E1"],
        ),
        AtomicClaim(
            id="C2",
            statement="Claim two has enough scientific content for testing.",
            evidence_ids=["E2"],
        ),
    ]
    candidates = [
        RelationCandidate(
            id="GOOD",
            source="C1",
            target="C2",
            relation="supports",
            confidence=0.9,
            rationale="Independent evidence agrees.",
            evidence_ids=["E1", "E2"],
            source_paper_ids=["P1"],
            target_paper_ids=["P2"],
            retrieval_score=0.8,
            condition_comparability=0.9,
            candidate_origin=["shared_entity"],
        ),
        RelationCandidate(
            id="BAD",
            source="C1",
            target="C2",
            relation="contradicts",
            confidence=0.2,
            rationale="Unsupported conflict.",
            evidence_ids=[],
            source_paper_ids=["P1"],
            target_paper_ids=["P2"],
            retrieval_score=0.2,
            condition_comparability=0.2,
            candidate_origin=[],
        ),
    ]
    state = {
        "claims": claims,
        "evidence_records": records,
        "relation_candidates": candidates,
        "report": GroundingReport(),
    }

    direct = GroundingWorkflow(
        mode="rule",
        cache_dir=str(tmp_path / "direct"),
        relation_selection_mode="direct",
        relation_min_confidence=0.0,
    )
    gams = GroundingWorkflow(
        mode="rule",
        cache_dir=str(tmp_path / "gams"),
        relation_selection_mode="evidence_gams",
        relation_min_confidence=0.35,
        evidence_gams_iterations=64,
    )

    import asyncio
    direct_output = await direct._select_relations(state)  # type: ignore[arg-type]
    gams_output = await gams._select_relations(state)  # type: ignore[arg-type]

    direct_semantic_ids = {
        r.id for r in direct_output["relations"] if r.id in {"GOOD", "BAD"}
    }
    gams_semantic_ids = {
        r.id for r in gams_output["relations"] if r.id in {"GOOD", "BAD"}
    }

    # Direct mode accepts both (BAD has confidence 0.2 which is >= 0.0)
    assert direct_semantic_ids == {"GOOD", "BAD"}
    # GAMS should filter out BAD (low confidence, no evidence, low retrieval)
    assert gams_semantic_ids == {"GOOD"}
    assert gams_output["report"].relation_selection_mode == "evidence_gams"


def test_merge_grounding_creates_claim_nodes():
    """Verify that _merge_grounding adds CLAIM and EVIDENCE nodes to graph."""
    from hypoforge.modules.m3_evidence_graph import M3EvidenceGraph

    records = [
        EvidenceRecord(
            id="ER_E1",
            evidence_id="E1",
            paper_id="P1",
            query="q",
            quote="Test quote.",
            normalized_claim="Test claim.",
            summary="Summary of test.",
            excerpt="Test quote.",
            relevance_score=7.0,
        ),
    ]
    claims = [
        AtomicClaim(
            id="CLM_test",
            statement="A test scientific claim with sufficient detail for the evidence graph.",
            evidence_ids=["E1"],
            paper_ids=["P1"],
            entities=["Protein A", "pathway B"],
            confidence=0.8,
        ),
    ]

    graph = EvidenceGraph()
    result = M3EvidenceGraph._merge_grounding(
        graph, records, claims, [], GroundingReport()
    )

    # Should have 1 EVIDENCE node and 1 CLAIM node
    evidence_nodes = [n for n in result.nodes if n.type == EvidenceNodeType.EVIDENCE]
    claim_nodes = [n for n in result.nodes if n.type == EvidenceNodeType.CLAIM]
    assert len(evidence_nodes) == 1
    assert len(claim_nodes) == 1
    assert evidence_nodes[0].metadata["evidence_id"] == "E1"
    assert claim_nodes[0].metadata["claim_id"] == "CLM_test"
    assert claim_nodes[0].metadata["confidence"] == 0.8

    # Should have 1 provenance edge (evidence → claim)
    prov_edges = [
        e for e in result.edges
        if e.relation == EvidenceEdgeRelation.SUPPORTS
    ]
    assert len(prov_edges) == 1
