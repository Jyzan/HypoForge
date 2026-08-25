from hypoforge.context import ContextPlanner, ContextRequest
from hypoforge.graph_context import GraphContext, GraphContextItem


def test_context_planner_renders_unverified_bridge_assumptions():
    context = GraphContext(bridge_hypotheses=[GraphContextItem(
        entry_id="HYP_GAP_1",
        kind="bridge_hypothesis",
        text="A may influence B through C.",
    )])
    pack = ContextPlanner().plan(context, ContextRequest(purpose="m5_plan"))
    assert "UNVERIFIED BRIDGE HYPOTHESES" in pack.rendered
    assert "HYP_GAP_1" in pack.rendered
