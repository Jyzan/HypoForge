"""
Offline unit tests for the iteration core (task #3, v2 contract).

Covers:
* ``_route_after_m6`` — strict four-priority branches × budget boundaries;
* ``_route_after_m1`` — followup search-free triage;
* ``make_gap_id`` — (sub_question, gap_type, entities) stability;
* ``build_followup_seed`` — whitelist inheritance / reset semantics;
* legacy checkpoint compatibility (old JSON without new fields);
* default-config equivalence with the pre-refactor routing;
* routing-aware ``_should_skip_module``;
* routing bookkeeping (round counter, revision_count) and M6 gap merge.

Run::

    python -m pytest tests/test_iteration_routing.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure HypoForge is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hypoforge.config import PipelineConfig
from hypoforge.pipeline import (
    PipelineRunner,
    _route_after_m1,
    _route_after_m6,
    _should_continue_iterating,
    build_followup_seed,
)
from hypoforge.state import (
    EvidenceGap,
    EvidenceGraph,
    EvidenceNode,
    EvidenceNodeType,
    EvidenceSufficiencyVerdict,
    FollowupRequest,
    GraphCorrectionRequest,
    PipelineState,
    ReviewResult,
    ReviewerDimension,
    RoutingDecision,
    make_gap_id,
)


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #

def _state_after_review(
    score: float = 3.0,
    iteration_count: int = 1,
    max_iterations: int = 3,
    threshold: float = 4.0,
    **extra,
) -> PipelineState:
    """A state as it looks right after M6 finished a review round."""
    return PipelineState(
        iteration_count=iteration_count,
        max_iterations=max_iterations,
        review_score_threshold=threshold,
        reviews=[
            ReviewResult(
                dimension=ReviewerDimension("overall"),
                score=score,
                version=iteration_count,
            )
        ],
        **extra,
    )


def _open_gap(
    sub_question: str = "Hsp70 如何识别底物？",
    gap_type: str = "mechanism",
    entities=("Hsp70",),
    description: str = "缺少 Hsp70 底物谱的直接蛋白质组证据",
) -> EvidenceGap:
    return EvidenceGap(
        description=description,
        target_sub_question=sub_question,
        gap_type=gap_type,
        canonical_entities=list(entities),
    )


def _insufficient_state(score: float = 3.0, search_round: int = 0, **extra) -> PipelineState:
    gap = _open_gap()
    return _state_after_review(
        score=score,
        search_round=search_round,
        evidence_verdict=EvidenceSufficiencyVerdict(sufficient=False, gaps=[gap]),
        evidence_gaps=[gap],
        **extra,
    )


def _nonempty_graph() -> EvidenceGraph:
    return EvidenceGraph(
        nodes=[EvidenceNode(id="n1", type=EvidenceNodeType.CLAIM, label="claim")],
        established_facts=["k1"],
    )


def _revisit_config(**overrides) -> PipelineConfig:
    return PipelineConfig(
        verbose=False,
        followup_routing=True,
        m6_evidence_revisit=True,
        **overrides,
    )


def _runner(switches_on: bool = True) -> PipelineRunner:
    config = (
        PipelineConfig(verbose=False, followup_routing=True, m6_evidence_revisit=True)
        if switches_on
        else PipelineConfig(verbose=False)
    )
    return PipelineRunner(config)


# --------------------------------------------------------------------------- #
# _route_after_m6 — strict four-priority branches × boundaries
# --------------------------------------------------------------------------- #

def test_route_after_m6_priority1_budget_exhausted_wins_over_everything():
    """Budget exhausted → end even when a supplement would otherwise qualify."""
    state = _insufficient_state(score=1.0, search_round=0)
    state.iteration_count = 3
    state.max_iterations = 3
    assert _route_after_m6(state, _revisit_config()) == "end"


def test_route_after_m6_priority2_supplement_beats_score_threshold():
    """Score already ≥ threshold, but evidence insufficient + open gap + search
    budget left → supplement (priority 2 fires BEFORE the threshold rule)."""
    state = _insufficient_state(score=4.9, search_round=1)
    assert _route_after_m6(state, _revisit_config(max_search_rounds=2)) == "supplement_m2"


def test_route_after_m6_priority2_requires_all_conditions():
    # no open gap
    gap = _open_gap()
    gap.status = "closed"
    state = _state_after_review(
        score=3.0,
        evidence_verdict=EvidenceSufficiencyVerdict(sufficient=False, gaps=[gap]),
        evidence_gaps=[gap],
        search_round=0,
    )
    assert _route_after_m6(state, _revisit_config()) == "revise_m4"

    # search-round budget exhausted
    state = _insufficient_state(score=3.0, search_round=2)
    assert _route_after_m6(state, _revisit_config(max_search_rounds=2)) == "revise_m4"

    # empty gap ledger
    state = _state_after_review(
        score=3.0,
        evidence_verdict=EvidenceSufficiencyVerdict(sufficient=False),
        search_round=0,
    )
    assert _route_after_m6(state, _revisit_config()) == "revise_m4"


def test_route_after_m6_priority3_sufficient_and_threshold_met_ends():
    state = _state_after_review(
        score=4.5,
        evidence_verdict=EvidenceSufficiencyVerdict(sufficient=True),
    )
    assert _route_after_m6(state, _revisit_config()) == "end"

    # verdict None while the revisit switch is off counts as sufficient
    state = _state_after_review(score=4.5)
    assert _route_after_m6(state, _revisit_config()) == "end"


def test_route_after_m6_priority4_insufficient_but_stuck_revises_m4():
    """Insufficient verdict but no supplement possible → revise (not end),
    even when the score clears the threshold."""
    state = _insufficient_state(score=4.9, search_round=2)
    assert _route_after_m6(state, _revisit_config(max_search_rounds=2)) == "revise_m4"


def test_route_after_m6_pending_graph_correction_revises_m3():
    state = _state_after_review(
        score=2.0,
        graph_correction_requests=[GraphCorrectionRequest(
            request_id="GCR_1",
            operation="remove_edge",
            source_node_id="N1",
            target_node_id="N2",
            evidence_ids=["EV1"],
            reason="The cited evidence does not support this relation.",
        )],
    )

    assert _route_after_m6(state, PipelineConfig(verbose=False)) == "revise_m3"


def test_route_after_m6_low_score_revises_m4():
    assert _route_after_m6(_state_after_review(score=3.0), _revisit_config()) == "revise_m4"


def test_route_after_m6_switch_off_verdict_is_ignored():
    """Switch off → the verdict never influences routing (legacy behaviour)."""
    state = _insufficient_state(score=3.0)
    assert _route_after_m6(state, PipelineConfig(verbose=False)) == "revise_m4"
    assert _route_after_m6(state, None) == "revise_m4"
    # high score + hostile verdict still ends when the switch is off
    high = _insufficient_state(score=4.9)
    assert _route_after_m6(high, PipelineConfig(verbose=False)) == "end"


# --------------------------------------------------------------------------- #
# _route_after_m1 — followup search-free triage
# --------------------------------------------------------------------------- #

def _followup_state(skip_search, graph=True) -> PipelineState:
    return PipelineState(
        input_question="追问",
        followup=FollowupRequest(text="追问", skip_search=skip_search),
        evidence_graph=_nonempty_graph() if graph else None,
    )


def test_route_after_m1_switch_off_always_searches():
    config = PipelineConfig(verbose=False)  # followup_routing=False
    state = _followup_state(skip_search=True)
    assert _route_after_m1(state, config) == "search_m2"
    assert _route_after_m1(state, None) == "search_m2"


def test_route_after_m1_skip_requires_all_conditions():
    assert _route_after_m1(_followup_state(True), _revisit_config()) == "direct_m4"


def test_route_after_m1_no_followup_searches():
    state = PipelineState(evidence_graph=_nonempty_graph())
    assert _route_after_m1(state, _revisit_config()) == "search_m2"


@pytest.mark.parametrize("skip_search", [False, None])
def test_route_after_m1_skip_search_not_true_searches(skip_search):
    assert _route_after_m1(_followup_state(skip_search), _revisit_config()) == "search_m2"


def test_route_after_m1_missing_or_empty_graph_searches():
    assert _route_after_m1(_followup_state(True, graph=False), _revisit_config()) == "search_m2"
    empty_graph_state = _followup_state(True)
    empty_graph_state.evidence_graph = EvidenceGraph()
    assert _route_after_m1(empty_graph_state, _revisit_config()) == "search_m2"


# --------------------------------------------------------------------------- #
# make_gap_id — (sub_question, gap_type, canonical_entities) contract
# --------------------------------------------------------------------------- #

def test_make_gap_id_same_triple_different_wording_same_id():
    a = make_gap_id("Hsp70 如何识别底物？", "mechanism", ["Hsp70"])
    b = make_gap_id(
        "Hsp70 如何识别底物？",
        "mechanism",
        ["Hsp70"],
    )
    # identical triple, totally different free-text descriptions upstream
    assert a == b
    assert len(a) == 12


def test_make_gap_id_entity_order_irrelevant():
    a = make_gap_id("q", "conflict", ["Hsp70", "BAG3"])
    b = make_gap_id("q", "conflict", ["BAG3", "Hsp70"])
    assert a == b


def test_make_gap_id_normalises_case_and_whitespace():
    a = make_gap_id("  How does HSP70  bind substrates? ", "MECHANISM", ["Hsp70 "])
    b = make_gap_id("how does hsp70 bind substrates?", "mechanism", ["hsp70"])
    assert a == b


def test_make_gap_id_comma_joined_composite_key_pinned():
    """The composite key is pinned to sha1(sub_q | type | comma-joined sorted
    entities)[:12] so no future refactor can silently change gap identity."""
    import hashlib

    raw = "hsp70 如何识别底物？|mechanism|bag3,hsp70"
    expected = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    assert make_gap_id("Hsp70 如何识别底物？", "mechanism", ["Hsp70", "BAG3"]) == expected


def test_make_gap_id_changes_with_any_component():
    base = make_gap_id("q", "mechanism", ["Hsp70"])
    assert base != make_gap_id("q2", "mechanism", ["Hsp70"])
    assert base != make_gap_id("q", "dosage", ["Hsp70"])
    assert base != make_gap_id("q", "mechanism", ["Hsp70", "BAG3"])


def test_evidence_gap_auto_id_uses_sub_question_or_description():
    gap = _open_gap()
    assert gap.gap_id == make_gap_id(gap.target_sub_question, gap.gap_type, gap.canonical_entities)
    assert gap.status == "open"
    assert gap.attempts == 0

    # missing target_sub_question → description anchors the hash, no crash
    no_sub = EvidenceGap(description="某种缺口", gap_type="coverage")
    assert no_sub.gap_id == make_gap_id("某种缺口", "coverage", [])
    assert no_sub.gap_type == "coverage"
    assert no_sub.canonical_entities == []


def test_evidence_gap_pending_grounding_status():
    gap = _open_gap()
    gap.status = "pending_grounding"
    assert gap.status == "pending_grounding"
    with pytest.raises(ValueError):
        EvidenceGap(description="x", status="mitigated_by_cache")  # removed enum value


# --------------------------------------------------------------------------- #
# Legacy compatibility
# --------------------------------------------------------------------------- #

def test_should_continue_iterating_wrapper_keeps_legacy_vocabulary():
    exhausted = _state_after_review(iteration_count=3, max_iterations=3)
    assert _should_continue_iterating(exhausted) == "end"

    high_score = _state_after_review(score=4.5, threshold=4.0)
    assert _should_continue_iterating(high_score) == "end"

    low_score = _state_after_review(score=3.0, threshold=4.0)
    assert _should_continue_iterating(low_score) == "iterate"


def test_default_config_routing_is_equivalent_to_legacy():
    """With both switches off, the new routers must reproduce the legacy
    m6→{iterate,end} behaviour and the legacy linear m1→m2 edge."""
    config = PipelineConfig(verbose=False)
    assert config.followup_routing is False
    assert config.m6_evidence_revisit is False

    legacy = {
        "exhausted": _state_after_review(iteration_count=3, max_iterations=3),
        "threshold": _state_after_review(score=4.9, threshold=4.5),
        "iterate": _state_after_review(score=2.5),
        # even hostile iteration-core state must not change behaviour:
        "hostile": _insufficient_state(score=2.5),
    }
    for name, state in legacy.items():
        expected = _should_continue_iterating(state)
        routed = _route_after_m6(state, config)
        assert routed == ("end" if expected == "end" else "revise_m4"), name
        # m1 always takes the search path with default config
        assert _route_after_m1(state, config) == "search_m2"


def test_legacy_checkpoint_without_new_fields_deserializes():
    """An old checkpoint/output JSON (no iteration-core fields) must load."""
    legacy_json = {
        "input_question": "蛋白质如何折叠？",
        "problem_card": {
            "original_question": "蛋白质如何折叠？",
            "domain": ["structural biology"],
            "sub_questions": ["q1", "q2"],
            "key_entities": ["Hsp70"],
            "question_type": "mechanism_explanation",
        },
        "reviews": [
            {"dimension": "overall", "score": 3.5, "version": 1}
        ],
        "iteration_count": 1,
        "max_iterations": 3,
        "review_score_threshold": 4.0,
        "run_id": "legacy-run",
    }
    state = PipelineState(**legacy_json)
    assert state.evidence_verdict is None
    assert state.evidence_gaps == []
    assert state.followup is None
    assert state.parent_run_id == ""
    assert state.search_round == 0
    assert state.revision_count == 0
    assert state.search_ledger.queries_issued == []
    assert state.routing_history == []
    assert state.iteration_count == 1
    assert state.reviews[0].score == 3.5


# --------------------------------------------------------------------------- #
# Graph topology — switch gating × routing_targets_available × module combos
# --------------------------------------------------------------------------- #

def _edges_of(config: PipelineConfig):
    runner = PipelineRunner(config)
    graph = runner._build_graph()
    return {(e.source, e.target) for e in graph.get_graph().edges}


def test_topology_switches_off_is_exactly_legacy_wiring():
    """HARD ACCEPTANCE: followup_routing=false and m6_evidence_revisit=false
    → conditional edges NOT installed; plain linear spine + legacy two-way
    m6 iteration edge (byte-for-byte the pre-refactor topology)."""
    edges = _edges_of(PipelineConfig(verbose=False))

    for src, dst in [("m1", "m2"), ("m2", "m3"), ("m3", "m4"), ("m4", "m5"), ("m5", "m6")]:
        assert (src, dst) in edges, f"missing linear edge {src}→{dst}"
    assert ("m1", "m4") not in edges  # no followup shortcut
    assert ("m6", "m2") not in edges  # no supplement re-hop
    assert ("m6", "m4") in edges       # legacy two-way iteration edge
    assert ("m6", "__end__") in edges


def test_topology_switches_on_full_modules_has_routing_edges():
    config = PipelineConfig(
        verbose=False, followup_routing=True, m6_evidence_revisit=True
    )
    edges = _edges_of(config)

    # linear spine (m1→m2 replaced by the conditional edge)
    for src, dst in [("m2", "m3"), ("m3", "m4"), ("m4", "m5"), ("m5", "m6")]:
        assert (src, dst) in edges, f"missing linear edge {src}→{dst}"

    # m1 routing-aware edge: search_m2→m2, direct_m4→m4 (replaces the linear edge)
    assert ("m1", "m2") in edges
    assert ("m1", "m4") in edges

    # m6 three-way edge (replaces the legacy two-way edge)
    assert ("m6", "__end__") in edges
    assert ("m6", "m4") in edges
    assert ("m6", "m2") in edges


def test_topology_without_m2_falls_back_to_legacy_wiring():
    config = PipelineConfig(
        verbose=False,
        enabled_modules=["m1", "m3", "m4", "m5", "m6"],
        followup_routing=True,
        m6_evidence_revisit=True,
    )
    edges = _edges_of(config)

    # routing targets unavailable → linear m1→m3 kept, no skip_search shortcut
    assert ("m1", "m3") in edges
    assert ("m1", "m2") not in edges
    assert ("m1", "m4") not in edges

    # m6 keeps the legacy two-way conditional edge (iterate→m4 / end)
    assert ("m6", "m4") in edges
    assert ("m6", "__end__") in edges
    assert ("m6", "m2") not in edges


def test_topology_without_m4_falls_back_to_legacy_wiring():
    config = PipelineConfig(
        verbose=False,
        enabled_modules=["m1", "m2", "m3", "m5", "m6"],
        followup_routing=True,
        m6_evidence_revisit=True,
        # m4 is absent, so the legacy iteration edge must target an enabled
        # module explicitly (default iteration_module_target="m4" would be
        # invalid with or without the iteration-core switches).
        iteration_module_target="m3",
    )
    edges = _edges_of(config)

    assert ("m1", "m2") in edges   # plain linear edge retained
    assert ("m1", "m4") not in edges
    assert ("m6", "m2") not in edges
    # legacy two-way iteration edge: iterate→m3 / end
    assert ("m6", "m3") in edges
    assert ("m6", "__end__") in edges


def test_topology_without_m1_keeps_m6_routing_edge():
    """m1 absent → entry point is m2; the m6 three-way edge still applies."""
    config = PipelineConfig(
        verbose=False,
        enabled_modules=["m2", "m3", "m4", "m5", "m6"],
        followup_routing=True,
        m6_evidence_revisit=True,
    )
    edges = _edges_of(config)

    assert not any(src == "m1" for src, _ in edges)
    for src, dst in [("m2", "m3"), ("m3", "m4"), ("m4", "m5"), ("m5", "m6")]:
        assert (src, dst) in edges
    assert ("m6", "m4") in edges
    assert ("m6", "m2") in edges
    assert ("m6", "__end__") in edges


def test_topology_switches_on_but_targets_missing_is_legacy():
    """Only one of m2/m4 present + switches on → legacy wiring everywhere."""
    config = PipelineConfig(
        verbose=False,
        enabled_modules=["m1", "m2", "m3"],
        followup_routing=True,
        m6_evidence_revisit=True,
        enable_iteration=False,
    )
    edges = _edges_of(config)
    assert ("m1", "m2") in edges
    assert ("m1", "m4") not in edges
    assert ("m3", "__end__") in edges


def test_edge_callbacks_match_pure_routers():
    runner = _runner(switches_on=True)
    state = _insufficient_state(score=3.0, search_round=1)
    assert runner._decide_after_m6(state) == _route_after_m6(state, runner.config)
    followup = _followup_state(skip_search=True)
    assert runner._decide_after_m1(followup) == _route_after_m1(followup, runner.config)


# --------------------------------------------------------------------------- #
# Routing bookkeeping — round counter + revision_count
# --------------------------------------------------------------------------- #

def test_routing_decision_round_is_history_length_plus_one():
    runner = _runner()
    state = _state_after_review(score=3.0)
    assert runner._routing_decision("m6", state).round == 1

    state.routing_history.append(
        RoutingDecision(round=1, from_module="m1", to_module="search_m2", decided_by="policy")
    )
    assert runner._routing_decision("m6", state).round == 2


def test_routing_decision_fields_for_m6_supplement():
    runner = _runner()
    state = _insufficient_state(score=3.0, search_round=1)
    decision = runner._routing_decision("m6", state)
    assert decision.from_module == "m6"
    assert decision.to_module == "supplement_m2"
    assert decision.decided_by == "m6"
    assert decision.gap_ids == [g.gap_id for g in state.evidence_gaps if g.status == "open"]


def test_routing_decision_fields_for_m1():
    runner = _runner()
    decision = runner._routing_decision("m1", _followup_state(skip_search=True))
    assert decision.from_module == "m1"
    assert decision.to_module == "direct_m4"
    assert decision.decided_by == "m1"

    decision = runner._routing_decision("m1", PipelineState())
    assert decision.to_module == "search_m2"
    assert decision.decided_by == "policy"


def test_append_routing_extends_history_and_bumps_revision_count():
    runner = _runner()
    state = _state_after_review(score=3.0)  # low score → revise_m4
    patch: dict = {}
    runner._append_routing("m6", state, patch)
    assert len(patch["routing_history"]) == 1
    assert patch["routing_history"][0].to_module == "revise_m4"
    assert patch["revision_count"] == 1  # revise_m4 bumps the audit counter

    # routing to end does NOT bump revision_count
    end_state = _state_after_review(score=4.9, iteration_count=3, max_iterations=3)
    patch2: dict = {}
    runner._append_routing("m6", end_state, patch2)
    assert patch2["routing_history"][0].to_module == "end"
    assert "revision_count" not in patch2

    # non-routing modules never touch the patch
    patch3: dict = {}
    runner._append_routing("m3", state, patch3)
    assert patch3 == {}


def test_append_routing_direct_m4_emits_module_skipped_for_m2_m3():
    """The conditional edge bypasses M2/M3, so the bookkeeping step must emit
    the explicit module_skipped events for both (when enabled)."""

    class _StubRecorder:
        def __init__(self):
            self.events = []

        def emit(self, event_type, **kwargs):
            self.events.append((event_type, kwargs))

    recorder = _StubRecorder()
    runner = _runner()
    runner.event_recorder = recorder
    patch: dict = {}
    runner._append_routing("m1", _followup_state(skip_search=True), patch)

    assert patch["routing_history"][0].to_module == "direct_m4"
    skipped = [
        kwargs["module"]
        for event_type, kwargs in recorder.events
        if event_type == "module_skipped"
    ]
    assert skipped == ["m2", "m3"]

    # the normal search path never emits module_skipped
    recorder.events.clear()
    patch2: dict = {}
    runner._append_routing("m1", PipelineState(), patch2)
    assert patch2["routing_history"][0].to_module == "search_m2"
    assert not [e for e in recorder.events if e[0] == "module_skipped"]


# --------------------------------------------------------------------------- #
# Routing-aware skip rules
# --------------------------------------------------------------------------- #

def test_skip_module_default_semantics_unchanged():
    runner = _runner(switches_on=False)
    done = PipelineState(literature_results=[{"sub_question": "q"}])
    # m2 done, no followup, no supplement route → skip (legacy semantics)
    assert runner._should_skip_module("m2", {"literature_results"}, done) is True
    # not done → run
    assert runner._should_skip_module("m2", {"literature_results"}, PipelineState()) is False
    # m4 inside iteration loop with budget left → never skip
    m4_done = PipelineState(top_hypotheses=[{"hypothesis_id": "h", "statement": "s"}])
    assert runner._should_skip_module("m4", {"top_hypotheses"}, m4_done) is False
    m4_done.iteration_count = m4_done.max_iterations
    assert runner._should_skip_module("m4", {"top_hypotheses"}, m4_done) is True


def test_skip_module_supplement_round_forces_m2_m3():
    runner = _runner()
    state = PipelineState(
        literature_results=[{"sub_question": "q"}],
        evidence_graph=_nonempty_graph(),
        routing_history=[
            RoutingDecision(round=1, from_module="m6", to_module="supplement_m2", decided_by="m6")
        ],
    )
    assert runner._should_skip_module("m2", {"literature_results"}, state) is False
    assert runner._should_skip_module("m3", {"evidence_graph"}, state) is False

    # a later m6→revise decision clears the supplement context
    state.routing_history.append(
        RoutingDecision(round=2, from_module="m6", to_module="revise_m4", decided_by="policy")
    )
    assert runner._should_skip_module("m2", {"literature_results"}, state) is True


def test_skip_module_followup_skip_search_skips_m2_m3():
    runner = _runner()
    state = PipelineState(
        literature_results=[{"sub_question": "q"}],
        evidence_graph=_nonempty_graph(),
        followup=FollowupRequest(text="追问", skip_search=True),
    )
    assert runner._should_skip_module("m2", {"literature_results"}, state) is True
    assert runner._should_skip_module("m3", {"evidence_graph"}, state) is True


def test_skip_module_followup_search_reruns_m2_m3():
    runner = _runner()
    for skip in (False, None):
        state = PipelineState(
            literature_results=[{"sub_question": "q"}],
            evidence_graph=_nonempty_graph(),
            followup=FollowupRequest(text="追问", skip_search=skip),
        )
        assert runner._should_skip_module("m2", {"literature_results"}, state) is False
        assert runner._should_skip_module("m3", {"evidence_graph"}, state) is False


# --- M1 followup exemption (task #16 fix 1) ---

def _m1_done_state(**extra) -> PipelineState:
    """A state whose M1 output field (problem_card) is already populated —
    exactly what ``build_followup_seed`` produces (whitelist inheritance)."""
    return PipelineState(
        problem_card={
            "original_question": "原始问题",
            "domain": ["proteostasis"],
            "sub_questions": ["q1"],
            "key_entities": ["Hsp70"],
            "question_type": "mechanism_explanation",
        },
        **extra,
    )


def test_skip_module_followup_forces_m1_rerun_when_routing_enabled():
    """followup + followup_routing on ⇒ M1 must re-run its triage even though
    the seed inherited a completed problem_card."""
    runner = _runner()  # followup_routing=True
    state = _m1_done_state(
        followup=FollowupRequest(text="给我中文方案", parent_run_id="p1"),
        evidence_graph=_nonempty_graph(),
    )
    assert runner._should_skip_module("m1", {"problem_card"}, state) is False


def test_skip_module_followup_m1_resume_after_triage_skips():
    """Checkpoint resume AFTER triage already ran this run (routing_history
    carries an m1 decision) ⇒ skip again, no duplicate triage."""
    runner = _runner()
    state = _m1_done_state(
        followup=FollowupRequest(text="给我中文方案", skip_search=True, parent_run_id="p1"),
        evidence_graph=_nonempty_graph(),
        routing_history=[
            RoutingDecision(round=1, from_module="m1", to_module="direct_m4", decided_by="m1")
        ],
    )
    assert runner._should_skip_module("m1", {"problem_card"}, state) is True


def test_skip_module_m1_unchanged_without_followup():
    """Non-followup runs keep the legacy done ⇒ skip semantics (switches on
    or off), and an incomplete M1 still runs."""
    for switches in (True, False):
        runner = _runner(switches_on=switches)
        assert runner._should_skip_module("m1", {"problem_card"}, _m1_done_state()) is True
    assert runner._should_skip_module("m1", {"problem_card"}, PipelineState()) is False


def test_skip_module_followup_m1_skips_when_routing_disabled():
    """followup_routing off ⇒ the exemption never fires (legacy behaviour)."""
    runner = _runner(switches_on=False)
    state = _m1_done_state(
        followup=FollowupRequest(text="给我中文方案", parent_run_id="p1"),
        evidence_graph=_nonempty_graph(),
    )
    assert runner._should_skip_module("m1", {"problem_card"}, state) is True


# --------------------------------------------------------------------------- #
# M6 gap merge semantics
# --------------------------------------------------------------------------- #

def test_m6_merge_gaps_hit_inherits_status_and_counts_attempt():
    from hypoforge.modules.m6_review_iteration import M6ReviewIteration

    existing = _open_gap(description="第一版描述")
    existing.source_review_version = 1
    existing.status = "pending_grounding"

    # same identity triple, different wording → same gap_id
    again = _open_gap(description="完全换一种说法")
    assert again.gap_id == existing.gap_id

    merged = M6ReviewIteration._merge_evidence_gaps([existing], [again], version=2)
    assert len(merged) == 1
    assert merged[0].status == "pending_grounding"  # inherited, not reset
    assert merged[0].attempts == 1  # survived one more review round
    assert merged[0].source_review_version == 2


def test_m6_merge_gaps_new_gap_starts_open_and_closed_never_reopens():
    from hypoforge.modules.m6_review_iteration import M6ReviewIteration

    closed_gap = _open_gap(description="已解决的缺口")
    closed_gap.status = "closed"
    same_id_new = _open_gap(description="已解决的缺口!!!")
    assert same_id_new.gap_id == closed_gap.gap_id

    fresh = _open_gap(sub_question="别的子问题", entities=("BAG3",), description="新缺口")
    merged = M6ReviewIteration._merge_evidence_gaps(
        [closed_gap], [same_id_new, fresh], version=2
    )
    by_id = {g.gap_id: g for g in merged}
    assert len(merged) == 2
    assert by_id[closed_gap.gap_id].status == "closed"  # not re-opened
    assert by_id[fresh.gap_id].status == "open"
    assert by_id[fresh.gap_id].attempts == 0


def test_m6_merge_gaps_disappeared_open_gap_closed_when_sufficient():
    from hypoforge.modules.m6_review_iteration import M6ReviewIteration

    vanished = _open_gap(description="本轮没再出现的缺口")
    still_there = _open_gap(sub_question="另一个子问题", entities=("X",), description="仍在")

    # verdict sufficient, only `still_there` re-reported → `vanished` closes
    merged = M6ReviewIteration._merge_evidence_gaps(
        [vanished, still_there], [still_there], version=3, sufficient=True
    )
    by_id = {g.gap_id: g for g in merged}
    assert by_id[vanished.gap_id].status == "closed"
    assert by_id[still_there.gap_id].status == "open"

    # verdict NOT sufficient → disappeared gap stays open (maybe transient)
    merged = M6ReviewIteration._merge_evidence_gaps(
        [vanished, still_there], [still_there], version=3, sufficient=False
    )
    by_id = {g.gap_id: g for g in merged}
    assert by_id[vanished.gap_id].status == "open"


# --------------------------------------------------------------------------- #
# build_followup_seed — whitelist inheritance / reset semantics
# --------------------------------------------------------------------------- #

def _dirty_parent_final_state() -> dict:
    """A realistic final-state JSON dump: whitelist fields + junk keys."""
    parent = PipelineState(
        input_question="原始问题",
        run_id="parent-1",
        iteration_count=3,
        max_iterations=3,
        revision_count=2,
        search_round=2,
        problem_card={
            "original_question": "原始问题",
            "domain": ["proteostasis"],
            "sub_questions": ["q1"],
            "key_entities": ["Hsp70"],
            "question_type": "mechanism_explanation",
        },
        literature_results=[{"sub_question": "q1", "papers_retrieved": 5}],
        evidence_graph=_nonempty_graph(),
        best_hypotheses=[{"hypothesis_id": "best-1", "statement": "保留"}],
        top_hypotheses=[{"hypothesis_id": "top-1", "statement": "应被重置"}],
        candidate_hypotheses=[{"hypothesis_id": "c1", "statement": "应被重置"}],
        research_plans=[{"hypothesis_id": "top-1", "study_subjects": "mice"}],
        reviews=[ReviewResult(dimension=ReviewerDimension("overall"), score=3.0, version=1)],
        routing_history=[
            RoutingDecision(round=1, from_module="m6", to_module="revise_m4", decided_by="policy")
        ],
        evidence_gaps=[_open_gap()],
        evidence_verdict=EvidenceSufficiencyVerdict(sufficient=False),
        errors=["some old error"],
        metrics={"novelty": 0.4},
        total_input_tokens=1234,
        total_output_tokens=567,
        memory_cache_dir="cache-dir",
    )
    dump = parent.model_dump(mode="json")
    # checkpoints carry junk keys that must NOT leak into the seed state
    dump["_last_module"] = "m6"
    dump["some_future_unknown_key"] = {"weird": True}
    return dump


def test_build_followup_seed_inherits_only_whitelist_fields():
    config = PipelineConfig(verbose=False, memory_cache_dir="default-cache")
    state = build_followup_seed(
        _dirty_parent_final_state(), followup_text="追问？", run_id="follow-1", config=config
    )

    # --- newly set ---
    assert state.input_question == "追问？"
    assert state.run_id == "follow-1"
    assert state.parent_run_id == "parent-1"
    assert state.followup == FollowupRequest(text="追问？", parent_run_id="parent-1")
    assert state.max_iterations == config.max_iterations  # fresh budget
    assert state.max_evidence_gap_rounds == config.max_evidence_gap_rounds

    # --- inherited (whitelist) ---
    assert state.problem_card is not None and state.problem_card.key_entities == ["Hsp70"]
    assert state.literature_results[0].sub_question == "q1"
    assert state.evidence_graph is not None
    assert state.best_hypotheses and state.best_hypotheses[0].hypothesis_id == "best-1"
    assert state.memory_cache_dir == "cache-dir"

    # --- reset to defaults ---
    assert state.iteration_count == 0
    assert state.revision_count == 0
    assert state.search_round == 0
    assert state.reviews == []
    assert state.candidate_hypotheses == []
    assert state.top_hypotheses == []
    assert state.research_plans == []
    assert state.errors == []
    assert state.metrics == {}
    assert state.total_input_tokens == 0
    assert state.total_output_tokens == 0
    assert state.routing_history == []
    assert state.evidence_verdict is None
    assert state.evidence_gaps == []


def test_build_followup_seed_memory_cache_falls_back_to_config():
    config = PipelineConfig(verbose=False, memory_cache_dir="default-cache")
    seed = {"run_id": "p", "memory_cache_dir": ""}
    state = build_followup_seed(seed, followup_text="追问", run_id="f", config=config)
    assert state.memory_cache_dir == "default-cache"
    assert state.parent_run_id == "p"


# --------------------------------------------------------------------------- #
# Config switches
# --------------------------------------------------------------------------- #

def test_iteration_core_switches_default_off():
    config = PipelineConfig.from_defaults()
    assert config.followup_routing is False
    assert config.m6_evidence_revisit is False
    assert config.max_search_rounds == 2
    assert config.supplement_paper_budget == 6
    assert config.gap_no_improvement_limit == 3
    assert config.gap_no_gain_limit == 3


def test_iteration_core_switches_load_from_yaml(tmp_path):
    yaml_path = tmp_path / "iteration_core.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                "followup_routing: true",
                "m6_evidence_revisit: true",
                "max_search_rounds: 3",
                "supplement_paper_budget: 8",
                "gap_no_improvement_limit: 2",
            ]
        ),
        encoding="utf-8",
    )
    config = PipelineConfig.from_yaml(str(yaml_path))
    assert config.followup_routing is True
    assert config.m6_evidence_revisit is True
    assert config.max_search_rounds == 3
    assert config.supplement_paper_budget == 8
    assert config.gap_no_improvement_limit == 2
