from hypoforge.graph_context import GraphContext
from hypoforge.modules.m5_research_plan import M5ResearchPlan
from hypoforge.state import HypothesisCard, HypothesisPremise, ResearchPlan


def test_new_plan_inherits_only_audited_factual_premise_evidence():
    hypothesis = HypothesisCard(
        hypothesis_id="H1",
        statement="A may cause B through a novel mechanism.",
        factual_premises=[
            HypothesisPremise(
                premise_id="P1",
                claim="A is associated with B.",
                kind="evidence_backed",
                supporting_evidence_ids=["E1", "NOT_CANONICAL"],
                source_paper_ids=["PAPER1"],
                audit_verdict="supported",
            ),
            HypothesisPremise(
                premise_id="P2",
                claim="C may modulate B.",
                kind="evidence_backed",
                supporting_evidence_ids=["E2"],
                source_paper_ids=["PAPER2"],
                audit_verdict="partially_supported",
            ),
            HypothesisPremise(
                premise_id="P3",
                claim="D always prevents B.",
                kind="evidence_backed",
                supporting_evidence_ids=["E3"],
                source_paper_ids=["PAPER3"],
                audit_verdict="unsupported",
            ),
        ],
    )
    context = GraphContext(
        available_evidence_ids=["E1", "E2", "E3"],
        available_paper_ids=["PAPER1", "PAPER2", "PAPER3"],
        evidence_to_paper={"E1": "PAPER1", "E2": "PAPER2", "E3": "PAPER3"},
    )

    plan = M5ResearchPlan._inherit_audited_premise_evidence(
        ResearchPlan(hypothesis_id="H1"),
        hypothesis,
        context,
    )

    assert plan.supporting_evidence_ids == ["E1", "E2"]
    assert plan.source_paper_ids == ["PAPER1", "PAPER2"]
    assert [(link.claim, link.support_status) for link in plan.evidence_links] == [
        ("A is associated with B.", "supported"),
        ("C may modulate B.", "supported"),
    ]
    assert all("novel mechanism" not in link.claim for link in plan.evidence_links)


def test_premise_inheritance_does_not_upgrade_existing_unverified_claim():
    hypothesis = HypothesisCard(
        hypothesis_id="H1",
        statement="A may cause B.",
        factual_premises=[HypothesisPremise(
            premise_id="P1",
            claim="A is associated with B.",
            kind="evidence_backed",
            supporting_evidence_ids=["E1"],
            source_paper_ids=["PAPER1"],
            audit_verdict="supported",
        )],
    )
    context = GraphContext(
        available_evidence_ids=["E1"],
        available_paper_ids=["PAPER1"],
        evidence_to_paper={"E1": "PAPER1"},
    )
    original = ResearchPlan.model_validate({
        "hypothesis_id": "H1",
        "evidence_links": [{
            "plan_element": "Test the novel mechanism",
            "claim": "A may cause B.",
            "support_status": "hypothesis_to_validate",
        }],
    })

    plan = M5ResearchPlan._inherit_audited_premise_evidence(
        original, hypothesis, context,
    )

    novel_link = next(link for link in plan.evidence_links if link.claim == "A may cause B.")
    assert novel_link.support_status == "hypothesis_to_validate"
    assert novel_link.supporting_evidence_ids == []
