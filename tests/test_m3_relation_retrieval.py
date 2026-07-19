from hypoforge.modules.m3_grounding.models import AtomicClaim, EvidenceRecord
from hypoforge.modules.m3_grounding.relation_retrieval import RelationCandidateRetriever


def record(record_id: str, paper_id: str, claim: str, entities: list[str], query: str) -> EvidenceRecord:
    return EvidenceRecord(
        id=record_id,
        chunk_id=f"CHK_{record_id}",
        paper_id=paper_id,
        query=query,
        summary=claim,
        excerpt=claim,
        relevance_score=8.0,
        retrieval_score=0.8,
        claims=[claim],
        entities=entities,
        context={"population": "human cells", "outcome": "Hsp70 activity"},
    )


def test_relation_retrieval_prioritises_shared_entity_cross_paper_pairs() -> None:
    records = [
        record("ER1", "P1", "NAD+ increases Hsp70 ATPase activity.", ["NAD+", "Hsp70"], "Hsp70 mechanism"),
        record("ER2", "P2", "NAD+ supplementation does not change Hsp70 activity.", ["NAD+", "Hsp70"], "Hsp70 mechanism"),
        record("ER3", "P3", "Kinase A phosphorylates substrate B.", ["Kinase A", "substrate B"], "kinase signalling"),
    ]
    claims = [
        AtomicClaim(id="C1", statement=records[0].claims[0], evidence_record_ids=["ER1"], entities=records[0].entities),
        AtomicClaim(id="C2", statement=records[1].claims[0], evidence_record_ids=["ER2"], entities=records[1].entities),
        AtomicClaim(id="C3", statement=records[2].claims[0], evidence_record_ids=["ER3"], entities=records[2].entities),
    ]

    pairs = RelationCandidateRetriever(max_candidates_per_claim=2).recall(claims, records)
    target = next(pair for pair in pairs if {pair.source, pair.target} == {"C1", "C2"})

    assert "shared_entity" in target.candidate_origin
    assert "cross_source" in target.candidate_origin
    assert "same_query" in target.candidate_origin
    assert {entity.lower() for entity in target.entity_overlap} == {"nad+", "hsp70"}
    assert target.retrieval_score > 0.4


def test_relation_retrieval_avoids_unrelated_low_signal_pairs() -> None:
    records = [
        record("ER1", "P1", "NAD+ increases Hsp70 ATPase activity.", ["NAD+", "Hsp70"], "Hsp70 mechanism"),
        record("ER2", "P2", "Kinase A phosphorylates substrate B.", ["Kinase A", "substrate B"], "kinase signalling"),
    ]
    records[1].context = {"population": "bacteria", "outcome": "substrate phosphorylation"}
    claims = [
        AtomicClaim(id="C1", statement=records[0].claims[0], evidence_record_ids=["ER1"], entities=records[0].entities),
        AtomicClaim(id="C2", statement=records[1].claims[0], evidence_record_ids=["ER2"], entities=records[1].entities),
    ]

    pairs = RelationCandidateRetriever(max_candidates_per_claim=2).recall(claims, records)
    assert pairs == []


def test_relation_retrieval_reserves_global_capacity_for_cross_source_pairs() -> None:
    records = [
        record("ER1", "P1", "Hsp70 assists folding of client protein A.", ["Hsp70"], "protein folding"),
        record("ER2", "P1", "Hsp70 assists folding of client protein B.", ["Hsp70"], "protein folding"),
        record("ER3", "P1", "Hsp70 assists folding of client protein C.", ["Hsp70"], "protein folding"),
        record("ER4", "P2", "Hsp70 limits aggregation of damaged proteins.", ["Hsp70"], "protein folding"),
    ]
    claims = [
        AtomicClaim(id=f"C{index}", statement=item.claims[0], evidence_record_ids=[item.id], entities=item.entities)
        for index, item in enumerate(records, start=1)
    ]

    pairs = RelationCandidateRetriever(
        max_candidates_per_claim=4, max_pairs_total=2, cross_source_ratio=0.5
    ).recall(claims, records)

    assert len(pairs) == 2
    assert any(set(pair.source_paper_ids).isdisjoint(pair.target_paper_ids) for pair in pairs)
