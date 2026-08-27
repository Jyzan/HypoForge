import pytest
from pydantic import ValidationError

from hypoforge.state import EvidenceGapRequest, HypothesisCard, HypothesisPremise, ResearchPlan


def test_evidence_backed_premise_cannot_reference_bridge_node():
    with pytest.raises(ValidationError):
        HypothesisPremise(
            premise_id="P1",
            claim="A causes B in cultured cells.",
            kind="evidence_backed",
            supporting_evidence_ids=["E1"],
            bridge_hypothesis_node_id="HYP_G1",
        )


def test_working_assumption_cannot_claim_canonical_evidence():
    with pytest.raises(ValidationError):
        HypothesisPremise(
            premise_id="U1",
            claim="A may cause C through B.",
            kind="unverified_bridge",
            supporting_evidence_ids=["E1"],
            bridge_hypothesis_node_id="HYP_G1",
        )


def test_old_hypothesis_checkpoint_loads_with_empty_new_fields():
    card = HypothesisCard(
        hypothesis_id="H1",
        statement="A may influence B.",
    )
    assert card.factual_premises == []
    assert card.working_assumptions == []
    assert card.research_gap == ""


def test_old_research_plan_loads_with_empty_bridge_validations():
    plan = ResearchPlan(hypothesis_id="H1")
    assert plan.bridge_validations == []


def test_gap_defaults_to_unreviewed_and_keeps_technical_failures_separate():
    gap = EvidenceGapRequest(gap_id="G1", sub_question="Does A affect B?")
    assert gap.scientific_resolution == "unreviewed"
    assert gap.search_completed is False
    assert gap.technical_errors == []
