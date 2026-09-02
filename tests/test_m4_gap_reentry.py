from hypoforge.graph_context import GraphContext, GraphContextItem
from hypoforge.modules.m4_hypothesis_generation import M4HypothesisGeneration
from hypoforge.state import EvidenceGapRequest, HypothesisCard, PipelineState
import pytest


def test_m4_reentry_attaches_only_explicit_m3_bridge_nodes_as_working_assumptions():
    module = object.__new__(M4HypothesisGeneration)
    state = PipelineState(
        input_question="A affects B",
        evidence_gap_requests=[EvidenceGapRequest(
            gap_id="G1",
            hypothesis_id="H1",
            sub_question="Does A affect B through C?",
            status="hypothesized",
            scientific_resolution="unresolved",
            bridge_hypothesis_node_id="HYP_G1",
        )],
    )
    context = GraphContext(bridge_hypotheses=[GraphContextItem(
        entry_id="HYP_G1",
        kind="bridge_hypothesis",
        text="A may affect B through C.",
    )])
    cards = [HypothesisCard(
        hypothesis_id="H1",
        statement="A may affect B through C.",
    )]
    attached = module._attach_bridge_assumptions(state, cards, context)
    assert attached[0].working_assumptions[0].bridge_hypothesis_node_id == "HYP_G1"
    assert attached[0].working_assumptions[0].supporting_evidence_ids == []


def test_m4_reentry_does_not_attach_a_bridge_without_a_current_gap():
    module = object.__new__(M4HypothesisGeneration)
    state = PipelineState(input_question="How can YOLO estimate 3D rotation?")
    context = GraphContext(bridge_hypotheses=[GraphContextItem(
        entry_id="HYP_AGING",
        kind="bridge_hypothesis",
        text="Aging therapy may improve mouse motor endurance.",
    )])
    cards = [HypothesisCard(
        hypothesis_id="H1",
        statement="A renderer may improve YOLO pose estimation.",
    )]

    attached = module._attach_bridge_assumptions(state, cards, context)

    assert attached[0].working_assumptions == []


@pytest.mark.asyncio
async def test_hypothesized_gap_is_not_reopened_by_m4_reentry():
    module = M4HypothesisGeneration(fast_mode=False)
    state = PipelineState(
        input_question="A affects B",
        evidence_gap_requests=[EvidenceGapRequest(
            gap_id="G1",
            premise_id="P1",
            hypothesis_id="H1",
            sub_question="Does A exist?",
            audit_claim="A exists",
            status="hypothesized",
            scientific_resolution="unresolved",
            bridge_hypothesis_node_id="HYP_G1",
        )],
    )
    card = HypothesisCard(
        hypothesis_id="H1",
        statement="A may affect B.",
        working_assumptions=[],
    )
    gaps = await module._audit_hypotheses(state, [card], GraphContext())
    assert not [gap for gap in gaps if gap.status == "pending"]
    assert gaps[0].status == "hypothesized"
