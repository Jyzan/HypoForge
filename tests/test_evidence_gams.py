"""Tests for EvidenceGraphGAMS — Monte Carlo Tree Search over relation candidates."""

from hypoforge.modules.m3_grounding.evidence_gams import EvidenceGraphGAMS
from hypoforge.modules.m3_grounding.models import RelationCandidate


def candidate(
    candidate_id: str,
    source: str,
    target: str,
    relation: str,
    confidence: float,
    evidence_ids: list[str],
    source_paper: str,
    target_paper: str,
    comparability: float = 0.9,
) -> RelationCandidate:
    return RelationCandidate(
        id=candidate_id,
        source=source,
        target=target,
        relation=relation,
        confidence=confidence,
        rationale=f"{source} {relation} {target}",
        evidence_ids=evidence_ids,
        source_paper_ids=[source_paper],
        target_paper_ids=[target_paper],
        retrieval_score=0.8,
        condition_comparability=comparability,
        candidate_origin=["shared_entity", "cross_source"],
    )


def test_evidence_gams_removes_low_quality_edges() -> None:
    candidates = [
        candidate("R1", "C1", "C4", "supports", 0.92, ["E1"], "P1", "P4"),
        candidate("R2", "C2", "C4", "supports", 0.88, ["E2"], "P2", "P4"),
        candidate("R3", "C3", "C4", "contradicts", 0.82, ["E3"], "P3", "P4"),
        candidate("R4", "C2", "C4", "extends", 0.72, ["E4"], "P2", "P4"),
        candidate("R5", "C3", "C4", "limits", 0.76, ["E5"], "P3", "P4"),
        candidate("BAD", "C1", "C4", "contradicts", 0.18, [], "P1", "P4"),
        candidate("UNSOURCED", "C1", "C2", "supports", 0.25, [], "P1", "P2"),
    ]
    search = EvidenceGraphGAMS(
        minimum_confidence=0.35,
        exploration_weight=0.35,
        random_seed=42,
    )

    selected, trace = search.search(candidates, iterations=128)
    selected_ids = {item.id for item in selected}

    assert set(trace["operator_counts"]) == set(EvidenceGraphGAMS.OPERATORS)
    assert all(count > 0 for count in trace["operator_counts"].values())
    assert trace["states_explored"] > 6
    assert trace["best_reward"] >= trace["direct_accept_all_reward"]
    assert "BAD" not in selected_ids
    assert "UNSOURCED" not in selected_ids
    assert {"R1", "R2", "R3", "R5"}.issubset(selected_ids)
    assert all(
        item.evidence_ids and item.confidence >= 0.35 for item in selected
    )


def test_evidence_gams_reproducible_with_fixed_seed() -> None:
    candidates = [
        candidate("R1", "C1", "C2", "supports", 0.9, ["E1"], "P1", "P2"),
        candidate("R2", "C2", "C3", "extends", 0.8, ["E2"], "P2", "P3"),
        candidate("R3", "C1", "C3", "limits", 0.7, ["E3"], "P1", "P3"),
    ]
    first, first_trace = EvidenceGraphGAMS(random_seed=7).search(
        candidates, iterations=64
    )
    second, second_trace = EvidenceGraphGAMS(random_seed=7).search(
        candidates, iterations=64
    )

    assert [item.id for item in first] == [item.id for item in second]
    assert first_trace["best_reward"] == second_trace["best_reward"]
