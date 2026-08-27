from hypoforge.graph_context import GraphContext, GraphContextItem
from hypoforge.modules.m5_research_plan import M5ResearchPlan
from hypoforge.state import ResearchPlan


def bridge_context():
    return GraphContext(bridge_hypotheses=[GraphContextItem(
        entry_id="HYP_GAP_1",
        kind="bridge_hypothesis",
        text="A may influence B through C.",
    )])


def test_m5_marks_bridge_as_hypothesis_to_validate():
    plan = M5ResearchPlan._sanitize_evidence_links(
        ResearchPlan(
            hypothesis_id="H1",
            study_subjects="A and B",
            procedures=["Measure the outcome"],
        ),
        bridge_context(),
    )
    bridge_links = [
        link for link in plan.evidence_links
        if "HYP_GAP_1" in link.claim
    ]
    assert len(bridge_links) == 1
    assert bridge_links[0].support_status == "hypothesis_to_validate"
    assert "HYP_GAP_1" not in plan.supporting_evidence_ids


def test_m6_cannot_cite_bridge_as_canonical_evidence():
    context = bridge_context().model_copy(update={"available_evidence_ids": ["EVID_1"]})
    plan = ResearchPlan(supporting_evidence_ids=["HYP_GAP_1", "EVID_1"])
    sanitized = M5ResearchPlan._sanitize_evidence_links(plan, context)
    assert sanitized.supporting_evidence_ids == ["EVID_1"]
