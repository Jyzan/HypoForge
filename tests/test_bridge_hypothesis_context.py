from hypoforge.context import ContextPlanner, ContextRequest
from hypoforge.graph_context import GraphContext, GraphContextItem, build_graph_context
from hypoforge.state import (
    EvidenceGapRequest,
    EvidenceGraph,
    EvidenceNode,
    EvidenceNodeType,
    PipelineState,
)


def test_context_planner_renders_unverified_bridge_assumptions():
    context = GraphContext(bridge_hypotheses=[GraphContextItem(
        entry_id="HYP_GAP_1",
        kind="bridge_hypothesis",
        text="A may influence B through C.",
    )])
    pack = ContextPlanner().plan(context, ContextRequest(purpose="m5_plan"))
    assert "UNVERIFIED BRIDGE HYPOTHESES" in pack.rendered
    assert "HYP_GAP_1" in pack.rendered


def test_graph_context_excludes_historical_bridge_from_an_unrelated_run():
    graph = EvidenceGraph(nodes=[
        EvidenceNode(
            id="HYP_AGING",
            type=EvidenceNodeType.HYPOTHESIS,
            label="Aging therapy may improve mouse motor endurance.",
            metadata={
                "verification_status": "unverified",
                "origin_gap_id": "GAP_AGING",
            },
        ),
        EvidenceNode(
            id="HYP_YOLO",
            type=EvidenceNodeType.HYPOTHESIS,
            label="A differentiable renderer may improve YOLO pose estimation.",
            metadata={
                "verification_status": "unverified",
                "origin_gap_id": "GAP_YOLO",
            },
        ),
    ])
    state = PipelineState(
        input_question="How can YOLO estimate 3D rotation?",
        evidence_graph=graph,
        evidence_gap_requests=[EvidenceGapRequest(
            gap_id="GAP_YOLO",
            hypothesis_id="H1",
            sub_question="Can differentiable rendering propagate pose gradients?",
            bridge_hypothesis_node_id="HYP_YOLO",
            status="hypothesized",
            scientific_resolution="unresolved",
        )],
    )

    context = build_graph_context(state)

    assert [item.entry_id for item in context.bridge_hypotheses] == ["HYP_YOLO"]
