from hypoforge.graph_context import build_graph_context
from hypoforge.state import (
    EvidenceGraph,
    EvidenceNode,
    EvidenceNodeType,
    EvidenceGapRequest,
    PipelineState,
)


def test_bridge_hypothesis_has_explicit_non_evidence_type():
    node = EvidenceNode(
        id="HYP_GAP_1",
        type=EvidenceNodeType.HYPOTHESIS,
        label="A may influence B through C.",
        metadata={"verification_status": "unverified"},
    )
    assert node.type is EvidenceNodeType.HYPOTHESIS


def test_gap_can_record_hypothesized_terminal_state():
    gap = EvidenceGapRequest(
        gap_id="GAP_1",
        sub_question="Does A cause B through C?",
        audit_claim="A causes B through C.",
        status="hypothesized",
        bridge_hypothesis_node_id="HYP_GAP_1",
    )
    assert gap.status == "hypothesized"


def test_graph_context_never_exposes_bridge_hypothesis_as_evidence():
    state = PipelineState(
        input_question="question",
        evidence_gap_requests=[EvidenceGapRequest(
            gap_id="GAP_1",
            sub_question="Does A cause B?",
            status="hypothesized",
            scientific_resolution="unresolved",
            bridge_hypothesis_node_id="HYP_GAP_1",
        )],
        evidence_graph=EvidenceGraph(
            nodes=[
                EvidenceNode(
                    id="HYP_GAP_1",
                    type=EvidenceNodeType.HYPOTHESIS,
                    label="A may influence B.",
                    metadata={"verification_status": "unverified"},
                )
            ]
        ),
    )
    context = build_graph_context(state)
    assert "HYP_GAP_1" not in context.available_evidence_ids
    assert [item.entry_id for item in context.bridge_hypotheses] == ["HYP_GAP_1"]
