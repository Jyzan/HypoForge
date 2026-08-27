from __future__ import annotations

import pytest

import hypoforge.pipeline as pipeline
from hypoforge.evidence_audit import PremiseAuditResult
from hypoforge.modules.m3_evidence_graph import M3EvidenceGraph
from hypoforge.modules.m4_hypothesis_generation import M4HypothesisGeneration
from hypoforge.modules.m5_research_plan import M5ResearchPlan
from hypoforge.state import (
    EvidenceGap,
    EvidenceGapRequest,
    EvidenceGraph,
    EvidenceNode,
    EvidenceNodeType,
    HypothesisCard,
    HypothesisPremise,
    PipelineState,
    ResearchPlan,
    RoutingDecision,
)


def _set_extra(model, name, value):
    # Keeps RED tests loadable against the old Pydantic model, while requiring
    # the implementation to add the real persisted fields before GREEN.
    object.__setattr__(model, name, value)
    return model


def _m6_return_state(resolution: str) -> PipelineState:
    gap = EvidenceGap(
        gap_id="M6-GAP-1",
        description="The intervention improves the target outcome.",
        target_sub_question="Does the intervention improve the target outcome?",
        status="closed",
    )
    _set_extra(gap, "hypothesis_ids", ["H1"])
    _set_extra(gap, "scientific_resolution", resolution)
    _set_extra(gap, "search_completed", True)
    return PipelineState(
        input_question="Does the intervention work?",
        evidence_gaps=[gap],
        routing_history=[RoutingDecision(
            round=1,
            from_module="m6",
            to_module="supplement_m2",
            decided_by="m6",
            gap_ids=[gap.gap_id],
        )],
    )


@pytest.mark.parametrize("resolution", ["supported", "unresolved"])
def test_m6_supported_or_unresolved_supplement_skips_m4(resolution):
    route = getattr(pipeline, "_route_after_m3", None)
    assert route is not None
    assert route(_m6_return_state(resolution)) == "m5"


def test_m6_contradiction_returns_only_to_m4():
    route = getattr(pipeline, "_route_after_m3", None)
    assert route is not None
    assert route(_m6_return_state("contradicted")) == "m4"


def test_m4_originated_supplement_always_returns_to_m4():
    route = getattr(pipeline, "_route_after_m3", None)
    assert route is not None
    state = PipelineState(
        input_question="Question",
        routing_history=[RoutingDecision(
            round=0,
            from_module="m4",
            to_module="supplement_m2",
            decided_by="m4",
            gap_ids=["GAP-1"],
        )],
    )
    assert route(state) == "m4"


def test_m4_reentry_preserves_unaffected_cards_and_demotes_unresolved_premise():
    p1 = HypothesisPremise(
        premise_id="P1", claim="Fact one", kind="evidence_backed",
        supporting_evidence_ids=["E-old"],
    )
    p2 = HypothesisPremise(
        premise_id="P2", claim="Fact two", kind="evidence_backed",
        supporting_evidence_ids=["E2"],
    )
    h1 = HypothesisCard(hypothesis_id="H1", statement="Hypothesis one", factual_premises=[p1])
    h2 = HypothesisCard(hypothesis_id="H2", statement="Hypothesis two", factual_premises=[p2])
    gap = EvidenceGapRequest(
        gap_id="GAP-1", hypothesis_id="H1", premise_id="P1",
        sub_question="Is fact one supported?", audit_claim="Fact one",
        status="hypothesized", scientific_resolution="unresolved",
        search_completed=True, bridge_hypothesis_node_id="HYP_1",
    )
    state = PipelineState(
        input_question="Question",
        candidate_hypotheses=[h1, h2], top_hypotheses=[h1, h2],
        evidence_gap_requests=[gap],
        evidence_graph=EvidenceGraph(nodes=[EvidenceNode(
            id="HYP_1", type=EvidenceNodeType.HYPOTHESIS,
            label="It may be the case that fact one.",
            metadata={"verification_status": "unverified"},
        )]),
        routing_history=[RoutingDecision(
            round=0, from_module="m4", to_module="supplement_m2",
            decided_by="m4", gap_ids=["GAP-1"],
        )],
    )

    reconcile = getattr(M4HypothesisGeneration, "_reconcile_reentry_portfolio", None)
    assert reconcile is not None
    candidates, top, contradicted = reconcile(state)

    assert contradicted == set()
    assert [card.hypothesis_id for card in top] == ["H1", "H2"]
    assert top[1] == h2
    assert top[0].factual_premises == []
    assert top[0].working_assumptions[0].bridge_hypothesis_node_id == "HYP_1"
    assert top[0].grounding_status == "bridge_only"
    assert top[0].supporting_evidence == []
    assert top[0].source_paper_ids == []
    assert [card.hypothesis_id for card in candidates] == ["H1", "H2"]


def test_m4_reentry_marks_only_contradicted_hypothesis_for_replacement():
    h1 = HypothesisCard(
        hypothesis_id="H1", statement="Affected",
        factual_premises=[HypothesisPremise(
            premise_id="P1", claim="False premise", kind="evidence_backed",
        )],
    )
    h2 = HypothesisCard(hypothesis_id="H2", statement="Unaffected")
    state = PipelineState(
        input_question="Question", candidate_hypotheses=[h1, h2], top_hypotheses=[h1, h2],
        evidence_gap_requests=[EvidenceGapRequest(
            gap_id="GAP-1", hypothesis_id="H1", premise_id="P1",
            sub_question="Is it false?", status="resolved",
            scientific_resolution="contradicted", search_completed=True,
            contradicting_evidence_ids=["E-no"],
        )],
    )

    reconcile = getattr(M4HypothesisGeneration, "_reconcile_reentry_portfolio", None)
    assert reconcile is not None
    candidates, top, contradicted = reconcile(state)

    assert contradicted == {"H1"}
    assert [card.hypothesis_id for card in candidates] == ["H2"]
    assert [card.hypothesis_id for card in top] == ["H2"]


def test_m5_reuses_plan_for_unchanged_hypothesis_only():
    h1 = HypothesisCard(hypothesis_id="H1", statement="Unchanged")
    h2 = HypothesisCard(hypothesis_id="H2-revision-1", statement="Replacement")
    old_h1 = ResearchPlan(hypothesis_id="H1", procedures=["keep me"])
    old_h2 = ResearchPlan(hypothesis_id="H2", procedures=["obsolete"])
    state = PipelineState(
        input_question="Question", top_hypotheses=[h1, h2],
        research_plans=[old_h1, old_h2],
    )

    reusable = getattr(M5ResearchPlan, "_reusable_plans", None)
    assert reusable is not None
    reusable = reusable(state)

    assert list(reusable) == ["H1"]
    assert reusable["H1"].procedures == ["keep me"]


def test_old_supplement_does_not_activate_after_m6_revision():
    state = PipelineState(
        input_question="Question",
        routing_history=[
            RoutingDecision(
                round=1, from_module="m4", to_module="supplement_m2",
                decided_by="m4", gap_ids=["G1"],
            ),
            RoutingDecision(
                round=2, from_module="m3", to_module="m4",
                decided_by="policy", gap_ids=["G1"],
            ),
            RoutingDecision(
                round=3, from_module="m6", to_module="revise_m4",
                decided_by="policy",
            ),
        ],
    )
    assert pipeline.active_supplement_origin(state) is None
    assert M4HypothesisGeneration._is_supplement_reentry(state) is False


def test_only_contiguous_supplement_return_activates_reentry():
    state = PipelineState(
        input_question="Question",
        routing_history=[
            RoutingDecision(
                round=1, from_module="m6", to_module="supplement_m2",
                decided_by="m6", gap_ids=["G1"],
            ),
            RoutingDecision(
                round=2, from_module="m3", to_module="m4",
                decided_by="policy", gap_ids=["G1"],
            ),
        ],
    )
    assert pipeline.active_supplement_origin(state).from_module == "m6"
    assert M4HypothesisGeneration._is_supplement_reentry(state) is True
    state.routing_history.append(RoutingDecision(
        round=3, from_module="m6", to_module="revise_m4", decided_by="policy",
    ))
    assert pipeline.active_supplement_origin(state) is None
    assert M4HypothesisGeneration._is_supplement_reentry(state) is False


def test_m5_does_not_reuse_plan_after_direct_m6_plan_revision():
    state = PipelineState(
        input_question="Question",
        top_hypotheses=[HypothesisCard(hypothesis_id="H1", statement="Updated")],
        research_plans=[ResearchPlan(hypothesis_id="H1", procedures=["old"])],
        routing_history=[RoutingDecision(
            round=1, from_module="m6", to_module="revise_m5", decided_by="policy",
        )],
    )
    assert M5ResearchPlan._is_supplement_reentry(state) is False


class _PremiseAuditor:
    def __init__(self, result: PremiseAuditResult):
        self.result = result

    async def audit_premises(self, premises, graph, **kwargs):
        return [self.result]


@pytest.mark.asyncio
async def test_m3_semantically_resolves_legacy_m6_gap_instead_of_counting_entries():
    module = M3EvidenceGraph(mode="rule")
    module.evidence_auditor = _PremiseAuditor(PremiseAuditResult(
        premise_id="M6-GAP-1",
        claim="The intervention improves the target outcome.",
        verdict="contradicted",
        evidence_ids=["E-no"],
        rationale="The supplement directly reports no benefit.",
    ))
    gap = EvidenceGap(
        gap_id="M6-GAP-1",
        description="The intervention improves the target outcome.",
        target_sub_question="Does the intervention improve the target outcome?",
        status="pending_grounding",
    )
    _set_extra(gap, "search_completed", True)
    _set_extra(gap, "scientific_resolution", "unreviewed")
    state = PipelineState(input_question="Question", evidence_gaps=[gap])

    resolver = getattr(module, "_resolve_m6_gap_requests", None)
    assert resolver is not None
    _, gaps = await resolver(state, EvidenceGraph(), state.evidence_gaps)

    assert gaps[0].scientific_resolution == "contradicted"
    assert gaps[0].contradicting_evidence_ids == ["E-no"]
