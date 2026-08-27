from hypoforge.pipeline import _route_after_m6
from hypoforge.pipeline import PipelineRunner
from hypoforge.config import PipelineConfig
from hypoforge.state import (
    EvidenceGap,
    EvidenceSufficiencyVerdict,
    PipelineState,
    ReviewResult,
    ReviewerDimension,
)


def _config():
    return PipelineConfig(
        verbose=False,
        m6_evidence_revisit=True,
        max_search_rounds=2,
    )


def _review(dimension, *, passed=True, attribution="both", score=4.5):
    return ReviewResult(
        dimension=ReviewerDimension(dimension),
        score=score,
        hard_gate_passed=passed,
        attribution=attribution,
        version=1,
    )


def _base(**kwargs):
    reviews = kwargs.pop("reviews", [_review("overall")])
    iteration_count = kwargs.pop("iteration_count", 1)
    max_iterations = kwargs.pop("max_iterations", 3)
    return PipelineState(
        iteration_count=iteration_count,
        max_iterations=max_iterations,
        reviews=reviews,
        **kwargs,
    )


def test_contradicted_fact_routes_to_m4_without_spending_search_budget():
    gap = EvidenceGap(
        description="Fact contradicted",
        source_claim_type="factual_premise",
        source_claim_id="P1",
        status="open",
        scientific_resolution="contradicted",
        contradicting_evidence_ids=["E9"],
    )
    state = _base(
        evidence_verdict=EvidenceSufficiencyVerdict(sufficient=False, gaps=[gap]),
        evidence_gaps=[gap],
        search_round=0,
    )
    assert _route_after_m6(state, _config()) == "revise_m4"


def test_unsupported_fact_with_budget_routes_to_m2():
    gap = EvidenceGap(
        description="Fact unsupported",
        source_claim_type="factual_premise",
        source_claim_id="P1",
        status="open",
        scientific_resolution="unreviewed",
    )
    state = _base(
        evidence_verdict=EvidenceSufficiencyVerdict(sufficient=False, gaps=[gap]),
        evidence_gaps=[gap],
        search_round=0,
    )
    assert _route_after_m6(state, _config()) == "supplement_m2"


def test_validation_and_method_failures_route_to_m5():
    validation = _review("experimental_validation_coverage", passed=False, attribution="plan", score=2.5)
    method = _review("method_feasibility", passed=False, attribution="plan", score=2.5)
    state = _base(reviews=[validation, method, _review("overall", score=2.5)])
    assert _route_after_m6(state, _config()) == "revise_m5"


def test_plan_only_evidence_coverage_failure_routes_to_m5():
    coverage = _review(
        "evidence_coverage_gate",
        passed=False,
        attribution="plan",
        score=0.0,
    )
    state = _base(reviews=[coverage, _review("overall", passed=False, score=2.9)])
    assert _route_after_m6(state, _config()) == "revise_m5"


def test_hypothesis_evidence_coverage_failure_routes_to_m4():
    coverage = _review(
        "evidence_coverage_gate",
        passed=False,
        attribution="hypothesis",
        score=0.0,
    )
    state = _base(reviews=[coverage, _review("overall", passed=False, score=2.9)])
    assert _route_after_m6(state, _config()) == "revise_m4"


def test_joint_evidence_coverage_failure_routes_to_m4_first():
    coverage = _review(
        "evidence_coverage_gate",
        passed=False,
        attribution="both",
        score=0.0,
    )
    state = _base(reviews=[coverage, _review("overall", passed=False, score=2.9)])
    assert _route_after_m6(state, _config()) == "revise_m4"


def test_logic_failure_routes_to_m4():
    logic = _review("scientific_logic", passed=False, attribution="hypothesis", score=2.0)
    state = _base(reviews=[logic, _review("overall", score=2.0)])
    assert _route_after_m6(state, _config()) == "revise_m4"


def test_all_layers_pass_route_to_end():
    state = _base(
        evidence_verdict=EvidenceSufficiencyVerdict(sufficient=True),
        reviews=[
            _review("scientific_logic"),
            _review("objective_evidence_consistency"),
            _review("method_feasibility"),
            _review("experimental_validation_coverage", attribution="plan"),
            _review("overall", score=4.5),
        ],
    )
    assert _route_after_m6(state, _config()) == "end"


def test_iteration_budget_exhaustion_is_a_hard_stop():
    state = _base(iteration_count=3, max_iterations=3)
    assert _route_after_m6(state, _config()) == "end"


def test_dynamic_revise_m4_edge_ignores_legacy_iteration_target():
    config = _config()
    config.iteration_module_target = "m3"
    edges = {
        (edge.source, edge.target)
        for edge in PipelineRunner(config)._build_graph().get_graph().edges
    }
    assert ("m6", "m4") in edges
    # The m6→m3 edge remains valid for the separate revise_m3 correction
    # route; revise_m4 is nevertheless directly available and selected by the
    # pure router for hypothesis feedback.
