from types import SimpleNamespace

from hypoforge.evaluation import scorer
from hypoforge.state import (
    EvidenceGraph,
    EvidenceNode,
    EvidenceNodeType,
    HypothesisCard,
    HypothesisPremise,
    PipelineState,
    ResearchPlan,
    WorkingAssumptionValidation,
)


def _bridge_state(*, validated: bool) -> PipelineState:
    hypothesis = HypothesisCard(
        hypothesis_id="H1",
        statement="A may influence B through C.",
        working_assumptions=[HypothesisPremise(
            premise_id="U1",
            claim="A may influence B through C.",
            kind="unverified_bridge",
            bridge_hypothesis_node_id="HYP_1",
            required=True,
        )],
        research_gap="Whether the bridge is valid.",
        grounding_status="bridge_only",
    )
    plan = ResearchPlan(
        hypothesis_id="H1",
        study_subjects="A and B",
        bridge_validations=(
            [WorkingAssumptionValidation(
                bridge_hypothesis_node_id="HYP_1",
                procedure="Run intervention and matched control.",
                measurement="Measure B.",
                falsification_condition="B does not change.",
            )] if validated else []
        ),
    )
    return PipelineState(
        input_question="Does A influence B?",
        top_hypotheses=[hypothesis],
        research_plans=[plan],
        evidence_graph=EvidenceGraph(nodes=[EvidenceNode(
            id="HYP_1",
            type=EvidenceNodeType.HYPOTHESIS,
            label="A may influence B through C.",
            metadata={"verification_status": "unverified"},
        )]),
    )


def test_bridge_only_hypothesis_uses_m5_validation_not_paper_citations(monkeypatch):
    monkeypatch.setattr(
        scorer,
        "assess_task_alignment",
        lambda *args, **kwargs: SimpleNamespace(
            passed=True, score=5.0, matched_anchors=[], conflicts=[], rationale="ok",
        ),
    )
    gates = scorer._quality_gates(_bridge_state(validated=True))
    assert gates["evidence_coverage_hypothesis"] == 1.0


def test_bridge_only_hypothesis_without_m5_validation_does_not_receive_coverage_credit(monkeypatch):
    monkeypatch.setattr(
        scorer,
        "assess_task_alignment",
        lambda *args, **kwargs: SimpleNamespace(
            passed=True, score=5.0, matched_anchors=[], conflicts=[], rationale="ok",
        ),
    )
    gates = scorer._quality_gates(_bridge_state(validated=False))
    assert gates["evidence_coverage_hypothesis"] == 0.0
