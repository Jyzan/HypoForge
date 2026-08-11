"""Tests for the P2 supplement closure loop (task #11 contract).

Covers:

* M6 structured evidence-sufficiency verdict — fail-closed on error, switch-off
  produces zero extra calls and no verdict keys (hard acceptance);
* gap merge semantics — gap_id matching, attempts inheritance, close-on
  ``sufficient`` **or** M3 gain > 0, resolved gaps stay resolved;
* zero-gain convergence protection (``m6_zero_gain_streak`` → unimprovable);
* ``PaperStore`` contract aliases (``lookup_by_query`` text/hash,
  ``papers_by_ids``) and ``source`` provenance;
* ``paper_sources`` junction helpers;
* pipeline supplement-round skip-suppression event.
"""

from __future__ import annotations

import pytest

from hypoforge.config import PipelineConfig
from hypoforge.memory import PaperStore
from hypoforge.memory.paper_store import query_hash
from hypoforge.modules.m6_review_iteration import M6ReviewIteration
from hypoforge.observability import RunEventRecorder, bind_recorder
from hypoforge.paper_sources import (
    dedupe_papers_by_key,
    dedupe_queries,
    gap_candidate_queries,
    merge_literature_increment,
    open_paper_store,
    persist_search_results,
    resolve_gap_sub_question,
)
from hypoforge.pipeline import PipelineRunner
from hypoforge.state import (
    EvidenceGap,
    EvidenceGraph,
    EvidenceNode,
    EvidenceNodeType,
    EvidenceSufficiencyVerdict,
    HypothesisCard,
    KnowledgeEntry,
    KnowledgeEntryType,
    LiteratureResult,
    PipelineState,
    ProblemCard,
    ResearchPlan,
    ReviewResult,
    RoutingDecision,
    SearchLedger,
    make_gap_id,
)


SUB_QUESTION = "What is the Hsp70 mechanism?"


def test_review_result_accepts_multiple_suggestion_strings() -> None:
    review = ReviewResult.model_validate({
        "dimension": "scientific_logic",
        "score": 3.0,
        "suggestions": ["Clarify the strategy.", "Add an error analysis."],
    })

    assert review.suggestions == "Clarify the strategy.\nAdd an error analysis."


def test_review_result_accepts_multiple_reasoning_strings() -> None:
    review = ReviewResult.model_validate({
        "dimension": "scientific_logic",
        "score": 4.0,
        "reasoning": [
            "- **Task fit**: The hypothesis addresses the stated object.",
            "- **Testability**: The proposed claims can be measured.",
        ],
    })

    assert review.reasoning == (
        "- **Task fit**: The hypothesis addresses the stated object.\n"
        "- **Testability**: The proposed claims can be measured."
    )


# ---------------------------------------------------------------------------
# Local fakes
# ---------------------------------------------------------------------------


class FakeM6Client:
    """Replays scripted ``structured_chat`` payloads, then raises."""

    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = 0

    async def structured_chat(self, **kwargs):
        self.calls += 1
        if not self.payloads:
            raise RuntimeError("no more scripted payloads")
        payload = self.payloads.pop(0)
        if isinstance(payload, Exception):
            raise payload
        return payload


def _review_payload() -> dict:
    return {
        "reasoning": "sound",
        "score": 4.0,
        "comments": "ok",
        "suggestions": "none",
    }


def make_m6_state(**overrides) -> PipelineState:
    base = dict(
        input_question="Q",
        problem_card=ProblemCard(original_question="Q", sub_questions=[SUB_QUESTION], key_entities=[], domain=[]),
        top_hypotheses=[HypothesisCard(hypothesis_id="h1", statement="s")],
        research_plans=[ResearchPlan(hypothesis_id="h1")],
    )
    base.update(overrides)
    return PipelineState(**base)


def make_m6_module(client, revisit: bool = True, limit: int = 3) -> M6ReviewIteration:
    module = M6ReviewIteration(
        mode="llm", llm_config=None,
        m6_evidence_revisit=revisit, gap_no_gain_limit=limit,
    )
    module.client = client
    return module


# ---------------------------------------------------------------------------
# M6 verdict: switch off = zero behaviour change (hard acceptance)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_m6_switch_off_never_calls_verdict() -> None:
    # Exactly three reviewer payloads; a 4th call would raise — proving the
    # verdict call is never issued when the switch is off.
    client = FakeM6Client([_review_payload()] * 3)
    module = make_m6_module(client, revisit=False)
    state = make_m6_state()

    patch = await module(state)

    assert client.calls == 3
    assert set(patch) == {
        "reviews", "iteration_count", "graph_correction_requests",
    }  # no evidence-verdict keys
    assert patch["iteration_count"] == 1
    assert len(patch["reviews"]) == 5  # alignment gate + 3 specialists + overall


# ---------------------------------------------------------------------------
# M6 verdict: happy path + gap merge + fail-closed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_m6_verdict_matches_existing_gap_inherits_attempts() -> None:
    gap_id = make_gap_id(SUB_QUESTION, "mechanism", ["hsp70"])
    existing = EvidenceGap(
        description="Missing co-chaperone data",
        gap_type="mechanism",
        canonical_entities=["hsp70"],
        target_sub_question=SUB_QUESTION,
        status="pending_grounding",
        attempts=2,
    )
    assert existing.gap_id == gap_id
    verdict_payload = {
        "sufficient": False,
        "gaps": [{
            "description": "Still missing co-chaperone data",
            "gap_type": "mechanism",
            "canonical_entities": ["Hsp70"],
            "target_sub_question": SUB_QUESTION,
            "suggested_queries": ["Hsp70 co-chaperone binding assay"],
        }],
    }
    client = FakeM6Client([_review_payload()] * 3 + [verdict_payload])
    module = make_m6_module(client)
    state = make_m6_state(evidence_gaps=[existing])

    patch = await module(state)

    assert client.calls == 4  # 3 reviewers + 1 verdict
    assert patch["evidence_verdict"].sufficient is False
    gaps = patch["evidence_gaps"]
    assert len(gaps) == 1
    assert gaps[0].gap_id == gap_id
    assert gaps[0].status == "pending_grounding"  # inherited
    assert gaps[0].attempts == 3  # survived one more review round
    assert gaps[0].suggested_queries == ["Hsp70 co-chaperone binding assay"]
    assert patch["metrics"]["m6_zero_gain_streak"] == 0  # no gain map → reset


@pytest.mark.asyncio
async def test_m6_verdict_fail_closed() -> None:
    client = FakeM6Client([_review_payload()] * 3 + [RuntimeError("boom")])
    module = make_m6_module(client)
    state = make_m6_state()

    patch = await module(state)

    assert client.calls == 4  # the verdict call was attempted
    verdict = patch["evidence_verdict"]
    assert verdict.sufficient is False
    assert len(verdict.gaps) == 1
    assert verdict.gaps[0].gap_type == "coverage"
    assert patch["evidence_gaps"][0].status == "open"


# ---------------------------------------------------------------------------
# _merge_evidence_gaps semantics
# ---------------------------------------------------------------------------


def test_merge_gaps_disappeared_open_gap_closes_on_sufficient_or_gain() -> None:
    gap = EvidenceGap(
        description="gone this round",
        target_sub_question=SUB_QUESTION,
        status="open",
    )

    # sufficient verdict closes it
    merged = M6ReviewIteration._merge_evidence_gaps([gap], [], 2, sufficient=True)
    assert merged[0].status == "closed"

    # insufficient verdict but M3 gain > 0 also closes it
    merged = M6ReviewIteration._merge_evidence_gaps(
        [gap], [], 2, sufficient=False, gap_gain={gap.gap_id: 2}
    )
    assert merged[0].status == "closed"

    # neither sufficient nor gained → stays open
    merged = M6ReviewIteration._merge_evidence_gaps(
        [gap], [], 2, sufficient=False, gap_gain={gap.gap_id: 0}
    )
    assert merged[0].status == "open"


def test_merge_gaps_resolved_statuses_never_reopen() -> None:
    closed = EvidenceGap(
        description="done", target_sub_question=SUB_QUESTION, status="closed"
    )
    unimprovable = EvidenceGap(
        description="stuck",
        target_sub_question="other question",
        status="unimprovable",
    )
    re_reported = [
        closed.model_copy(update={"status": "open"}),
        unimprovable.model_copy(update={"status": "open"}),
    ]
    merged = M6ReviewIteration._merge_evidence_gaps(
        [closed, unimprovable], re_reported, 3, sufficient=False
    )
    by_id = {g.gap_id: g for g in merged}
    assert by_id[closed.gap_id].status == "closed"
    assert by_id[unimprovable.gap_id].status == "unimprovable"


def test_merge_gaps_new_gap_recorded_open() -> None:
    incoming = EvidenceGap(description="fresh", target_sub_question=SUB_QUESTION)
    merged = M6ReviewIteration._merge_evidence_gaps([], [incoming], 1)
    assert len(merged) == 1
    assert merged[0].status == "open"
    assert merged[0].attempts == 0


# ---------------------------------------------------------------------------
# Zero-gain convergence protection
# ---------------------------------------------------------------------------


def test_zero_gain_streak_accumulates_and_marks_unimprovable() -> None:
    module = M6ReviewIteration(mode="llm", llm_config=None, gap_no_gain_limit=2)
    gap = EvidenceGap(
        description="stuck", target_sub_question=SUB_QUESTION, status="open"
    )

    # round 1: all-zero gain → streak 1 (below limit)
    state = make_m6_state(
        metrics={"m3_gap_gain": {gap.gap_id: 0}, "m6_zero_gain_streak": 0}
    )
    gaps, metrics = module._apply_zero_gain_convergence(state, [gap])
    assert metrics["m6_zero_gain_streak"] == 1
    assert gaps[0].status == "open"

    # round 2: still zero → streak 2 == limit → unimprovable
    state2 = make_m6_state(
        metrics={"m3_gap_gain": {gap.gap_id: 0}, "m6_zero_gain_streak": 1}
    )
    gaps2, metrics2 = module._apply_zero_gain_convergence(state2, [gap])
    assert metrics2["m6_zero_gain_streak"] == 2
    assert gaps2[0].status == "unimprovable"


def test_zero_gain_streak_resets_on_positive_gain() -> None:
    module = M6ReviewIteration(mode="llm", llm_config=None, gap_no_gain_limit=2)
    gap = EvidenceGap(
        description="stuck", target_sub_question=SUB_QUESTION, status="open"
    )
    state = make_m6_state(
        metrics={"m3_gap_gain": {gap.gap_id: 1}, "m6_zero_gain_streak": 1}
    )
    gaps, metrics = module._apply_zero_gain_convergence(state, [gap])
    assert metrics["m6_zero_gain_streak"] == 0
    assert gaps[0].status == "open"


# ---------------------------------------------------------------------------
# PaperStore contract aliases + provenance
# ---------------------------------------------------------------------------


def test_paper_store_contract_aliases(tmp_path) -> None:
    store = PaperStore(tmp_path)
    keys = store.upsert_papers([{"doi": "10.1/x", "title": "T"}])
    digest = store.record_query("Hsp70 chaperone", keys)
    assert digest == query_hash("Hsp70 chaperone")

    # lookup_by_query accepts raw text...
    assert store.lookup_by_query("hsp70  chaperone!!") == keys
    # ...and a precomputed 12-hex hash
    assert store.lookup_by_query(digest) == keys
    # unknown text → []
    assert store.lookup_by_query("totally different") == []

    # papers_by_ids is the lookup_by_keys alias
    assert store.papers_by_ids(keys) == store.lookup_by_keys(keys)
    assert store.papers_by_ids(keys)[0]["doi"] == "10.1/x"


def test_paper_store_source_provenance(tmp_path) -> None:
    store = PaperStore(tmp_path)
    store.upsert_papers([{"doi": "10.1/x"}], source="agentic_m2")
    store.upsert_papers([{"doi": "10.1/x"}], source="agentic_m2")
    store._ensure_loaded()
    record = store._papers["doi:10.1/x"]
    assert record["sources_seen"] == ["agentic_m2"]


# ---------------------------------------------------------------------------
# paper_sources junction helpers
# ---------------------------------------------------------------------------


def test_open_paper_store_degrades_gracefully() -> None:
    assert open_paper_store("") is None


def test_persist_search_results_never_raises_and_records() -> None:
    assert persist_search_results(None, [{"doi": "10.1/x"}], ["q"]) == []
    import tempfile

    with tempfile.TemporaryDirectory() as cache_dir:
        store = PaperStore(cache_dir)
        keys = persist_search_results(
            store, [{"doi": "10.1/x"}], ["hsp70 query"], source="agentic_m2"
        )
        assert keys == ["doi:10.1/x"]
        assert store.lookup_query("hsp70 query") == keys


def test_dedupe_papers_by_key_preserves_order_and_mutates_known() -> None:
    known = {"doi:10.1/old"}
    papers = [{"doi": "10.1/a"}, {"doi": "10.1/old"}, {"doi": "10.1/a"}]
    fresh, fresh_keys = dedupe_papers_by_key(papers, known)
    assert fresh_keys == ["doi:10.1/a"]
    assert fresh == [{"doi": "10.1/a"}]
    assert "doi:10.1/a" in known


def test_dedupe_queries_normalised_and_mutates_ledger_set() -> None:
    issued = {"hsp70 chaperone"}
    fresh = dedupe_queries(["Hsp70  chaperone!!", "new query text"], issued)
    assert fresh == ["new query text"]
    assert "new query text" in issued


def test_gap_candidate_queries_prefers_suggested() -> None:
    gap = EvidenceGap(
        description="some long description text",
        suggested_queries=["q1", "  q2  ", "", "q3", "q4"],
    )
    assert gap_candidate_queries(gap) == ["q1", "q2", "q3"]

    no_suggested = EvidenceGap(
        description="  Missing   co-chaperone data ",
        canonical_entities=["Hsp70", "Bag1"],
    )
    assert gap_candidate_queries(no_suggested) == [
        "Missing co-chaperone data",
        "Hsp70 Bag1",
    ]


def test_merge_literature_increment_extends_and_never_clears() -> None:
    old_entry = KnowledgeEntry(
        id="KE_old",
        type=KnowledgeEntryType.ESTABLISHED_FACT,
        content="old",
        source_paper_id="p0",
    )
    results = [
        LiteratureResult(
            sub_question=SUB_QUESTION,
            papers_retrieved=1,
            knowledge_entries=[old_entry],
        )
    ]
    new_entry = KnowledgeEntry(
        id="KE_new",
        type=KnowledgeEntryType.ESTABLISHED_FACT,
        content="new",
        source_paper_id="p1",
    )
    merge_literature_increment(results, SUB_QUESTION, 2, [old_entry, new_entry])
    assert results[0].papers_retrieved == 3
    assert [e.id for e in results[0].knowledge_entries] == ["KE_old", "KE_new"]

    # unseen sub-question appends a fresh result
    merge_literature_increment(results, "brand new question", 1, [])
    assert len(results) == 2
    assert results[1].sub_question == "brand new question"


def test_resolve_gap_sub_question() -> None:
    explicit = EvidenceGap(
        description="x", target_sub_question=" explicit target "
    )
    assert resolve_gap_sub_question(explicit, ["other"]) == "explicit target"

    fuzzy = EvidenceGap(description="Hsp70 binding affinity data")
    assert (
        resolve_gap_sub_question(
            fuzzy, ["unrelated topic", "Hsp70 binding mechanism"]
        )
        == "Hsp70 binding mechanism"
    )


# ---------------------------------------------------------------------------
# Legacy M2 supplement flow
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Pipeline supplement-round skip-suppression event
# ---------------------------------------------------------------------------


def test_pipeline_supplement_round_emits_skip_suppressed(tmp_path) -> None:
    runner = PipelineRunner(PipelineConfig())
    recorder = RunEventRecorder(tmp_path / "run", "run-1")
    runner.event_recorder = recorder

    state = PipelineState(
        literature_results=[{"sub_question": "q"}],
        evidence_graph=EvidenceGraph(
            nodes=[EvidenceNode(id="n1", type=EvidenceNodeType.CLAIM, label="c")],
            established_facts=["k1"],
        ),
        search_round=2,
        routing_history=[
            RoutingDecision(
                round=1, from_module="m6", to_module="supplement_m2",
                decided_by="m6",
            )
        ],
    )
    assert runner._should_skip_module("m2", {"literature_results"}, state) is False
    assert runner._should_skip_module("m3", {"evidence_graph"}, state) is False

    suppressed = [
        event
        for event in recorder.read_events()
        if event["event_type"] == "module_skip_suppressed"
    ]
    assert {e["module"] for e in suppressed} == {"m2", "m3"}
    for event in suppressed:
        assert event["details"]["reason"] == "supplement_m2 round"
        assert event["details"]["search_round"] == 2
