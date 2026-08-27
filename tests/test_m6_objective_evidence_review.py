from hypoforge.modules.m6_review_iteration import M6ReviewIteration
from hypoforge.state import (
    EvidenceSufficiencyVerdict,
    FactualPremiseAudit,
    GraphCorrectionRequest,
)


def _verdict(audit_verdict: str) -> EvidenceSufficiencyVerdict:
    return EvidenceSufficiencyVerdict(
        sufficient=audit_verdict in {"supported", "partially_supported", "not_applicable"},
        premise_audits=[FactualPremiseAudit(
            premise_id="P1",
            claim="A is associated with B.",
            verdict=audit_verdict,
            evidence_ids=["E1"] if audit_verdict != "unsupported" else [],
            rationale=f"P1 is {audit_verdict}.",
        )],
    )


def test_supported_premises_are_the_authoritative_evidence_score():
    review = M6ReviewIteration._objective_evidence_review(
        _verdict("supported"), version=2,
    )
    assert review.score == 5.0
    assert review.hard_gate_passed is True
    assert review.evidence_ids == ["E1"]


def test_partial_premise_support_passes_with_visible_score_reduction():
    review = M6ReviewIteration._objective_evidence_review(
        _verdict("partially_supported"), version=2,
    )
    assert review.score == 3.5
    assert review.hard_gate_passed is True


def test_unsupported_and_contradicted_premises_fail_the_evidence_gate():
    unsupported = M6ReviewIteration._objective_evidence_review(
        _verdict("unsupported"), version=2,
    )
    contradicted = M6ReviewIteration._objective_evidence_review(
        _verdict("contradicted"), version=2,
    )
    assert unsupported.score == 2.0
    assert unsupported.hard_gate_passed is False
    assert contradicted.score == 1.0
    assert contradicted.hard_gate_passed is False


def test_no_auditable_factual_premises_is_neutral_not_a_mechanism_penalty():
    review = M6ReviewIteration._objective_evidence_review(
        EvidenceSufficiencyVerdict(sufficient=True), version=2,
    )
    assert review.score == 5.0
    assert review.hard_gate_passed is True


def test_sufficient_authoritative_audit_discards_stale_graph_corrections():
    stale = GraphCorrectionRequest(
        request_id="gc_stale",
        operation="remove_edge",
        source_node_id="E1",
        target_node_id="H1",
        current_relation="supports",
        evidence_ids=["E1"],
        reason="The raw reviewer disputed this citation.",
    )

    reconciled = M6ReviewIteration._reconcile_graph_corrections(
        _verdict("supported"),
        [stale],
    )

    assert reconciled == []


def test_insufficient_authoritative_audit_keeps_actionable_graph_corrections():
    correction = GraphCorrectionRequest(
        request_id="gc_valid",
        operation="remove_edge",
        source_node_id="E1",
        target_node_id="H1",
        current_relation="supports",
        evidence_ids=["E1"],
        reason="The authoritative premise audit found a contradiction.",
    )

    reconciled = M6ReviewIteration._reconcile_graph_corrections(
        _verdict("contradicted"),
        [correction],
    )

    assert reconciled == [correction]
