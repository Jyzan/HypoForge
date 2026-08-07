from __future__ import annotations

from hypoforge.modules.m3_evidence_graph import M3EvidenceGraph
from hypoforge.modules.m6_review_iteration import M6ReviewIteration
from hypoforge.state import (
    EvidenceEdge,
    EvidenceEdgeRelation,
    EvidenceGraph,
    EvidenceNode,
    EvidenceNodeType,
    GraphCorrectionRequest,
    ReviewResult,
    ReviewerDimension,
)


def graph() -> EvidenceGraph:
    return EvidenceGraph(
        nodes=[
            EvidenceNode(
                id="N1", type=EvidenceNodeType.CLAIM, label="claim one",
                metadata={"evidence_ids": ["ev-1"]},
            ),
            EvidenceNode(
                id="N2", type=EvidenceNodeType.CLAIM, label="claim two",
            ),
        ],
        edges=[EvidenceEdge(
            source="N1", target="N2",
            relation=EvidenceEdgeRelation.SUPPORTS,
            evidence_ids=["ev-1"],
        )],
    )


def correction(**updates) -> GraphCorrectionRequest:
    payload = {
        "request_id": "GCR_1",
        "operation": "reclassify_edge",
        "source_node_id": "N1",
        "target_node_id": "N2",
        "current_relation": "supports",
        "proposed_relation": "contradicts",
        "evidence_ids": ["ev-1"],
        "reason": "The cited result has the opposite direction.",
    }
    payload.update(updates)
    return GraphCorrectionRequest(**payload)


def test_graph_correction_reclassifies_and_appends_versioned_audit() -> None:
    updated, requests = M3EvidenceGraph._apply_graph_corrections(
        graph(), [correction()], iteration=2,
    )

    assert updated.edges[0].relation is EvidenceEdgeRelation.CONTRADICTS
    assert updated.version == 2
    assert requests[0].status == "applied"
    assert updated.audit_log[0].graph_version_before == 1
    assert updated.audit_log[0].graph_version_after == 2
    assert updated.audit_log[0].before["relation"] == "supports"
    assert updated.audit_log[0].after["relation"] == "contradicts"


def test_graph_correction_rejects_unknown_node_without_mutation() -> None:
    updated, requests = M3EvidenceGraph._apply_graph_corrections(
        graph(),
        [correction(target_node_id="missing")],
        iteration=2,
    )

    assert updated.edges[0].relation is EvidenceEdgeRelation.SUPPORTS
    assert updated.version == 1
    assert requests[0].status == "rejected"
    assert "unknown graph node" in requests[0].rejection_reason
    assert updated.audit_log[0].status == "rejected"


def test_graph_correction_rejects_request_without_evidence() -> None:
    updated, requests = M3EvidenceGraph._apply_graph_corrections(
        graph(), [correction(evidence_ids=[])], iteration=2,
    )

    assert updated.version == 1
    assert requests[0].status == "rejected"
    assert "no canonical supporting evidence" in requests[0].rejection_reason


def test_m6_normalizes_request_id_and_filters_unknown_evidence() -> None:
    review = ReviewResult(
        dimension=ReviewerDimension.EVIDENCE_CONSISTENCY,
        score=2,
        graph_correction_requests=[correction(
            request_id="", evidence_ids=["ev-1", "invented"],
        )],
    )

    normalized = M6ReviewIteration._normalise_graph_corrections(
        review, valid_evidence_ids={"ev-1"}, version=3,
    )

    assert normalized[0].request_id.startswith("GCR_")
    assert normalized[0].evidence_ids == ["ev-1"]
    assert normalized[0].iteration == 3
    assert normalized[0].status == "pending"
