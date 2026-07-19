import pytest

from hypoforge.modules.m3_grounding.models import (
    AtomicClaim,
    EvidenceRecord,
    GroundingReport,
    RelationCandidate,
)
from hypoforge.modules.m3_grounding.workflow import FullTextEvidenceGrounding
from hypoforge.memory.schema import evidence_graph_to_knowledge_graph, knowledge_graph_to_evidence_graph
from hypoforge.state import EvidenceEdge, EvidenceEdgeRelation, EvidenceGraph, EvidenceNode, EvidenceNodeType


def evidence(record_id: str, paper_id: str) -> EvidenceRecord:
    return EvidenceRecord(
        id=record_id,
        chunk_id=f"CHK_{record_id}",
        paper_id=paper_id,
        query="test query",
        summary=f"summary {record_id}",
        excerpt=f"excerpt {record_id}",
        relevance_score=8.0,
        retrieval_score=0.8,
        claims=[f"Scientific claim from {record_id} with sufficient length."],
    )


@pytest.mark.asyncio
async def test_direct_and_gams_share_provenance_but_select_semantic_edges_differently(tmp_path) -> None:
    records = [evidence("E1", "P1"), evidence("E2", "P2")]
    claims = [
        AtomicClaim(id="C1", statement="Claim one has enough scientific content.", evidence_record_ids=["E1"]),
        AtomicClaim(id="C2", statement="Claim two has enough scientific content.", evidence_record_ids=["E2"]),
    ]
    candidates = [
        RelationCandidate(
            id="GOOD", source="C1", target="C2", relation="supports", confidence=0.9,
            rationale="Independent evidence agrees.", evidence_ids=["E1", "E2"],
            source_paper_ids=["P1"], target_paper_ids=["P2"], retrieval_score=0.8,
            condition_comparability=0.9, candidate_origin=["shared_entity"],
        ),
        RelationCandidate(
            id="BAD", source="C1", target="C2", relation="contradicts", confidence=0.2,
            rationale="Unsupported conflict.", evidence_ids=[], source_paper_ids=["P1"],
            target_paper_ids=["P2"], retrieval_score=0.2, condition_comparability=0.2,
        ),
    ]
    state = {
        "claims": claims,
        "evidence_records": records,
        "relation_candidates": candidates,
        "report": GroundingReport(),
    }

    direct = FullTextEvidenceGrounding(
        mode="rule", cache_dir=str(tmp_path / "direct"), relation_selection_mode="direct",
        relation_min_confidence=0.0,
    )
    gams = FullTextEvidenceGrounding(
        mode="rule", cache_dir=str(tmp_path / "gams"), relation_selection_mode="evidence_gams",
        relation_min_confidence=0.35, evidence_gams_iterations=64,
    )
    direct_output = await direct._select_relations(state)
    gams_output = await gams._select_relations(state)

    direct_semantic = {relation.id for relation in direct_output["relations"] if relation.id in {"GOOD", "BAD"}}
    gams_semantic = {relation.id for relation in gams_output["relations"] if relation.id in {"GOOD", "BAD"}}
    direct_provenance = {relation.id for relation in direct_output["relations"] if "provenance" in relation.candidate_origin}
    gams_provenance = {relation.id for relation in gams_output["relations"] if "provenance" in relation.candidate_origin}

    assert direct_semantic == {"GOOD", "BAD"}
    assert gams_semantic == {"GOOD"}
    assert direct_provenance == gams_provenance
    assert len(gams_provenance) == 2
    assert gams_output["report"].relation_search["mode"] == "evidence_gams"


def test_evidence_edge_search_metadata_survives_persistent_roundtrip() -> None:
    graph = EvidenceGraph(
        nodes=[
            EvidenceNode(id="E1", type=EvidenceNodeType.EVIDENCE_RECORD, label="Evidence"),
            EvidenceNode(id="C1", type=EvidenceNodeType.CLAIM, label="Claim"),
        ],
        edges=[EvidenceEdge(
            source="E1", target="C1", relation=EvidenceEdgeRelation.SUPPORTS,
            confidence=0.88, rationale="Exact excerpt supports the claim.",
            evidence_ids=["E1"], metadata={"condition_comparability": 0.9},
        )],
        grounding_report={"relation_selection_mode": "evidence_gams"},
    )

    restored = knowledge_graph_to_evidence_graph(evidence_graph_to_knowledge_graph(graph))
    edge = restored.edges[0]
    assert edge.confidence == 0.88
    assert edge.rationale == "Exact excerpt supports the claim."
    assert edge.evidence_ids == ["E1"]
    assert edge.metadata["condition_comparability"] == 0.9
    assert restored.grounding_report["relation_selection_mode"] == "evidence_gams"
