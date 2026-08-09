from __future__ import annotations

import asyncio
import math
import time

import pytest

from hypoforge.literature.models import (
    CoverageReport,
    PaperRecord,
    PaperRetentionDecision,
    QueryIntent,
    ScoutNote,
    SearchBudget,
    SearchQuery,
    StopReason,
)
from hypoforge.literature.search.agent import IterativeSearchAgent
from hypoforge.literature.sources.pubmed_source import PubMedSource

from .fakes import (
    FakeCoverageEvaluator,
    FakeDeduplicator,
    FakePlanner,
    FakeRanker,
    FakeScoutReader,
    FakeSource,
)


def query(query_id: str, text: str, source: str) -> SearchQuery:
    return SearchQuery(
        query_id=query_id,
        text=text,
        intent=QueryIntent.CORE,
        target_source=source,
        purpose="Find direct evidence",
        relation_to_question="Directly addresses the sub-question",
    )


def paper(paper_id: str, source: str) -> PaperRecord:
    return PaperRecord(
        paper_id=paper_id,
        title=f"Paper {paper_id}",
        abstract=f"Abstract for {paper_id}",
        sources=[source],
    )


@pytest.mark.asyncio
async def test_retention_judge_events_are_readable_for_web_and_terminal(
    monkeypatch,
) -> None:
    """Mojibake in emitted messages is user-visible in the shared event stream."""

    class Judge:
        async def structured_chat(self, **kwargs):
            return {"rulings": [{
                "paper_id": "paper-1",
                "decision": "retain",
                "roles": ["core_evidence"],
                "rationale": "direct evidence",
            }]}

    events = []
    monkeypatch.setattr(
        "hypoforge.literature.search.agent.emit_event",
        lambda event_type, **kwargs: events.append({
            "event_type": event_type,
            **kwargs,
        }),
    )
    agent = IterativeSearchAgent(
        query_planner=FakePlanner([]),
        sources=[FakeSource("pubmed", {})],
        deduplicator=FakeDeduplicator(),
        ranker=FakeRanker(),
        scout_reader=FakeScoutReader(),
        coverage_evaluator=FakeCoverageEvaluator([]),
        retention_judge_client=Judge(),
        final_k=1,
    )
    candidate = paper("paper-1", "pubmed")
    candidate.rank_scores["post_scout_total"] = 0.8
    decision = PaperRetentionDecision(
        paper_id="paper-1",
        decision="retain",
        roles=["core_evidence"],
        reason="primary set",
        rank_position=1,
    )

    await agent._review_retention_boundary(
        [candidate], [decision], [], "question", ["entity"]
    )

    messages = [str(event.get("message", "")) for event in events]
    assert messages == ["M2 开始进行 LLM 论文保留裁决", "LLM 论文保留裁决完成"]


@pytest.mark.asyncio
async def test_agent_uses_deterministic_four_to_three_term_fallback() -> None:
    planner = FakePlanner([[query("q-strict", "alpha beta gamma delta", "pubmed")]])
    source = FakeSource("pubmed", {
        "alpha beta gamma delta": [],
        "alpha beta gamma": [paper("three-term-hit", "pubmed")],
    })
    agent = IterativeSearchAgent(
        query_planner=planner,
        sources=[source],
        deduplicator=FakeDeduplicator(),
        ranker=FakeRanker(),
        scout_reader=FakeScoutReader(),
        coverage_evaluator=FakeCoverageEvaluator([
            CoverageReport(sufficient=True, rationale="hit")
        ]),
        final_k=1,
    )

    result = await agent.run(
        "alpha relationship",
        key_entities=["alpha"],
        budget=SearchBudget(max_queries=4),
    )

    assert [call.text for call in source.calls] == [
        "alpha beta gamma delta",
        "alpha beta gamma",
    ]
    assert [item.paper_id for item in result.final_papers] == ["three-term-hit"]
    assert result.queries[-1].purpose.startswith("deterministic fallback")


@pytest.mark.asyncio
async def test_agent_reuses_normalized_search_cache_across_runs() -> None:
    planner = FakePlanner([[query("q-cache", "Alpha   Beta", "pubmed")]])
    source = FakeSource("pubmed", {
        "Alpha   Beta": [paper("cached-paper", "pubmed")],
    })
    agent = IterativeSearchAgent(
        query_planner=planner,
        sources=[source],
        deduplicator=FakeDeduplicator(),
        ranker=FakeRanker(),
        scout_reader=FakeScoutReader(),
        coverage_evaluator=FakeCoverageEvaluator([
            CoverageReport(sufficient=True, rationale="done")
        ]),
        final_k=1,
    )

    first = await agent.run("Alpha Beta")
    second = await agent.run("Alpha Beta")

    assert len(source.calls) == 1
    assert first.final_papers[0].paper_id == "cached-paper"
    assert second.final_papers[0].paper_id == "cached-paper"


@pytest.mark.asyncio
async def test_agent_returns_complete_result_when_first_round_is_sufficient() -> None:
    planner = FakePlanner(
        [[query("q-1", "query one", "pubmed"), query("q-2", "query two", "openalex")]]
    )
    pubmed = FakeSource(
        "pubmed",
        {"query one": [paper("paper-1", "pubmed"), paper("shared", "pubmed")]},
    )
    openalex = FakeSource(
        "openalex",
        {"query two": [paper("shared", "openalex"), paper("paper-3", "openalex")]},
    )
    scout = FakeScoutReader()
    agent = IterativeSearchAgent(
        query_planner=planner,
        sources=[pubmed, openalex],
        deduplicator=FakeDeduplicator(),
        ranker=FakeRanker(),
        scout_reader=scout,
        coverage_evaluator=FakeCoverageEvaluator(
            [CoverageReport(sufficient=True, rationale="Coverage complete")]
        ),
        final_k=2,
    )

    result = await agent.run(
        "Does the mechanism hold?",
        key_entities=["mechanism"],
        domains=[],
        question_type="mechanism",
        budget=SearchBudget(max_rounds=3),
    )

    assert result.stop_reason is StopReason.COVERAGE_SATISFIED
    assert [item.paper_id for item in result.candidates] == [
        "paper-1",
        "shared",
        "paper-3",
    ]
    assert [item.paper_id for item in result.final_papers] == ["paper-1", "shared"]
    assert result.papers_found == 4
    assert result.papers_after_dedup == 3
    assert result.source_result_counts == {"pubmed": 2, "openalex": 2}
    assert [item.round_index for item in result.queries] == [1, 1]
    assert [note.paper_id for note in result.scout_notes] == [
        "paper-1",
        "shared",
        "paper-3",
    ]
    assert result.final_state is not None
    assert result.final_state.round_index == 1
    assert len(pubmed.calls) == 1
    assert len(openalex.calls) == 1


@pytest.mark.asyncio
async def test_agent_uses_coverage_gap_to_plan_a_second_round() -> None:
    planner = FakePlanner(
        [
            [query("q-1", "core query", "pubmed")],
            [query("q-2", "negative evidence", "pubmed")],
        ]
    )
    source = FakeSource(
        "pubmed",
        {
            "core query": [paper("paper-1", "pubmed")],
            "negative evidence": [paper("paper-2", "pubmed")],
        },
    )
    agent = IterativeSearchAgent(
        query_planner=planner,
        sources=[source],
        deduplicator=FakeDeduplicator(),
        ranker=FakeRanker(),
        scout_reader=FakeScoutReader(),
        coverage_evaluator=FakeCoverageEvaluator(
            [
                CoverageReport(
                    sufficient=False,
                    missing_topics=["negative evidence"],
                    rationale="Contradictory evidence is missing",
                ),
                CoverageReport(sufficient=True, rationale="Gap filled"),
            ]
        ),
    )

    result = await agent.run(
        "Does the mechanism hold?",
        budget=SearchBudget(max_rounds=3),
    )

    assert len(planner.states) == 2
    assert planner.states[1].missing_topics == {"negative evidence"}
    assert [item.round_index for item in result.queries] == [1, 2]
    assert [item.paper_id for item in result.candidates] == ["paper-1", "paper-2"]
    assert result.iterations == 2
    assert result.stop_reason is StopReason.COVERAGE_SATISFIED


@pytest.mark.asyncio
async def test_agent_scouts_only_new_papers_and_reuses_cached_notes() -> None:
    planner = FakePlanner(
        [
            [query("q-1", "first", "pubmed")],
            [query("q-2", "second", "pubmed")],
        ]
    )
    source = FakeSource(
        "pubmed",
        {
            "first": [paper("paper-1", "pubmed")],
            "second": [paper("paper-2", "pubmed")],
        },
    )
    scout = FakeScoutReader()
    coverage = FakeCoverageEvaluator(
        [CoverageReport(sufficient=False), CoverageReport(sufficient=True)]
    )
    agent = IterativeSearchAgent(
        query_planner=planner,
        sources=[source],
        deduplicator=FakeDeduplicator(),
        ranker=FakeRanker(),
        scout_reader=scout,
        coverage_evaluator=coverage,
    )

    result = await agent.run("question")

    assert result.stop_reason is StopReason.COVERAGE_SATISFIED
    assert scout.calls == [["paper-1"], ["paper-2"]]
    assert coverage.note_calls == [["paper-1"], ["paper-1", "paper-2"]]


@pytest.mark.asyncio
async def test_agent_reranks_candidates_with_scout_relevance_before_final_k() -> None:
    class RelevanceScout(FakeScoutReader):
        async def read(self, sub_question, papers):
            self.calls.append([item.paper_id for item in papers])
            relevance = {"metadata-favorite": 0.1, "scout-favorite": 0.95}
            return [
                ScoutNote(
                    paper_id=item.paper_id,
                    relevance_to_question=relevance[item.paper_id],
                )
                for item in papers
            ]

    metadata_favorite = paper("metadata-favorite", "pubmed").model_copy(
        update={"rank_scores": {"total": 0.9}}
    )
    scout_favorite = paper("scout-favorite", "pubmed").model_copy(
        update={"rank_scores": {"total": 0.1}}
    )
    agent = IterativeSearchAgent(
        query_planner=FakePlanner([[query("q-1", "core", "pubmed")]]),
        sources=[
            FakeSource(
                "pubmed",
                {"core": [metadata_favorite, scout_favorite]},
            )
        ],
        deduplicator=FakeDeduplicator(),
        ranker=FakeRanker(),
        scout_reader=RelevanceScout(),
        coverage_evaluator=FakeCoverageEvaluator([CoverageReport(sufficient=True)]),
        final_k=1,
    )

    result = await agent.run("Which paper directly addresses the mechanism?")

    assert [item.paper_id for item in result.final_papers] == ["scout-favorite"]
    assert result.final_papers[0].rank_scores["scout_relevance"] == 0.95
    assert "post_scout_total" in result.final_papers[0].rank_scores


@pytest.mark.asyncio
async def test_agent_excludes_not_applicable_papers_from_final_k_when_possible() -> None:
    class SemanticScout(FakeScoutReader):
        async def read(self, sub_question, papers):
            return [
                ScoutNote(
                    paper_id=item.paper_id,
                    relevance_to_question=0.9 if item.paper_id == "relevant" else 0.1,
                    directness_to_question=0.8 if item.paper_id == "relevant" else 0.1,
                )
                for item in papers
            ]

    agent = IterativeSearchAgent(
        query_planner=FakePlanner([[query("q-1", "core", "pubmed")]]),
        sources=[
            FakeSource(
                "pubmed",
                {
                    "core": [
                        paper("relevant", "pubmed"),
                        paper("irrelevant-1", "pubmed"),
                        paper("irrelevant-2", "pubmed"),
                    ]
                },
            )
        ],
        deduplicator=FakeDeduplicator(),
        ranker=FakeRanker(),
        scout_reader=SemanticScout(),
        coverage_evaluator=FakeCoverageEvaluator(
            [CoverageReport(sufficient=True)]
        ),
        final_k=3,
    )

    result = await agent.run("question")

    assert [item.paper_id for item in result.final_papers] == ["relevant"]


def build_agent(
    *,
    planner: FakePlanner | None = None,
    sources: list[FakeSource] | None = None,
    deduplicator: FakeDeduplicator | None = None,
    ranker: FakeRanker | None = None,
    scout: FakeScoutReader | None = None,
    coverage: FakeCoverageEvaluator | None = None,
    **kwargs,
) -> IterativeSearchAgent:
    return IterativeSearchAgent(
        query_planner=planner
        or FakePlanner([[query("q-1", "core query", "pubmed")]]),
        sources=sources
        or [FakeSource("pubmed", {"core query": [paper("paper-1", "pubmed")]})],
        deduplicator=deduplicator or FakeDeduplicator(),
        ranker=ranker or FakeRanker(),
        scout_reader=scout or FakeScoutReader(),
        coverage_evaluator=coverage
        or FakeCoverageEvaluator([CoverageReport(sufficient=True)]),
        **kwargs,
    )


@pytest.mark.asyncio
async def test_one_source_failure_keeps_other_source_results() -> None:
    planner = FakePlanner(
        [[query("q-1", "bad", "pubmed"), query("q-2", "good", "openalex")]]
    )
    agent = build_agent(
        planner=planner,
        sources=[
            FakeSource("pubmed", errors={"bad": RuntimeError("service unavailable")}),
            FakeSource("openalex", {"good": [paper("paper-2", "openalex")]}),
        ],
    )

    result = await agent.run("question")

    assert [item.paper_id for item in result.final_papers] == ["paper-2"]
    assert result.failed_sources == ["pubmed"]
    assert result.source_result_counts == {"openalex": 1}
    assert result.stop_reason is StopReason.COVERAGE_SATISFIED


@pytest.mark.asyncio
async def test_failed_source_circuit_opens_for_later_rounds() -> None:
    planner = FakePlanner(
        [
            [
                query("q-1", "bad one", "semantic_scholar"),
                query("q-2", "good one", "pubmed"),
            ],
            [
                query("q-3", "bad two", "semantic_scholar"),
                query("q-4", "good two", "pubmed"),
            ],
        ]
    )
    academic = FakeSource(
        "semantic_scholar",
        errors={
            "bad one": RuntimeError("rate limited"),
            "bad two": RuntimeError("must not be called"),
        },
    )
    pubmed = FakeSource(
        "pubmed",
        {
            "good one": [paper("paper-1", "pubmed")],
            "good two": [paper("paper-2", "pubmed")],
        },
    )
    agent = build_agent(
        planner=planner,
        sources=[academic, pubmed],
        coverage=FakeCoverageEvaluator(
            [CoverageReport(sufficient=False), CoverageReport(sufficient=True)]
        ),
    )

    result = await agent.run("question")

    assert [call.text for call in academic.calls] == ["bad one"]
    assert [call.text for call in pubmed.calls] == ["good one", "good two"]
    assert planner.states[1].unavailable_sources == {"semantic_scholar"}
    assert result.failed_sources == ["semantic_scholar"]


@pytest.mark.asyncio
async def test_all_sources_failed_returns_structured_error() -> None:
    agent = build_agent(
        sources=[
            FakeSource(
                "pubmed",
                errors={"core query": RuntimeError("service unavailable")},
            )
        ],
        coverage=FakeCoverageEvaluator([CoverageReport(sufficient=False)]),
    )

    result = await agent.run("question")

    assert result.stop_reason is StopReason.ERROR
    assert result.final_papers == []
    assert result.failed_sources == ["pubmed"]
    assert "service unavailable" in result.errors[0]


@pytest.mark.asyncio
async def test_agent_records_failure_from_strict_teammate_source() -> None:
    class StrictFailingPubMedTool:
        async def search_strict(self, query: str, limit: int = 20):
            raise OSError("real backend unavailable")

        async def search(self, query: str, limit: int = 20):
            return []

    agent = build_agent(
        sources=[PubMedSource(tool=StrictFailingPubMedTool())],
        coverage=FakeCoverageEvaluator([CoverageReport(sufficient=False)]),
    )

    result = await agent.run("question")

    assert result.stop_reason is StopReason.ERROR
    assert result.failed_sources == ["pubmed"]
    assert any("real backend unavailable" in error for error in result.errors)


@pytest.mark.asyncio
async def test_unknown_source_is_reported_without_guessing_a_fallback() -> None:
    agent = build_agent(
        planner=FakePlanner([[query("q-1", "core query", "missing")]]),
        coverage=FakeCoverageEvaluator([CoverageReport(sufficient=False)]),
    )

    result = await agent.run("question")

    assert result.stop_reason is StopReason.ERROR
    assert result.failed_sources == ["missing"]
    assert "unknown literature source" in result.errors[0]


@pytest.mark.asyncio
async def test_source_timeout_is_isolated_as_a_source_failure() -> None:
    agent = build_agent(
        sources=[FakeSource("pubmed", delays={"core query": 0.05})],
        coverage=FakeCoverageEvaluator([CoverageReport(sufficient=False)]),
        source_timeout_seconds=0.001,
    )

    result = await agent.run("question")

    assert result.stop_reason is StopReason.ERROR
    assert result.failed_sources == ["pubmed"]


@pytest.mark.asyncio
async def test_error_messages_redact_credentials() -> None:
    agent = build_agent(
        sources=[
            FakeSource(
                "pubmed",
                errors={
                    "core query": RuntimeError(
                        "request failed api_key=TOPSECRET token=ABC123"
                    )
                },
            )
        ],
        coverage=FakeCoverageEvaluator([CoverageReport(sufficient=False)]),
    )

    result = await agent.run("question")

    assert "TOPSECRET" not in result.errors[0]
    assert "ABC123" not in result.errors[0]
    assert "<redacted>" in result.errors[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["planner", "deduplicator", "ranker", "scout", "coverage"])
async def test_core_tool_failure_returns_error_result(stage: str) -> None:
    failure = RuntimeError(f"{stage} failed")
    agent = build_agent(
        planner=FakePlanner(
            [[query("q-1", "core query", "pubmed")]],
            error=failure if stage == "planner" else None,
        ),
        deduplicator=FakeDeduplicator(failure if stage == "deduplicator" else None),
        ranker=FakeRanker(failure if stage == "ranker" else None),
        scout=FakeScoutReader(failure if stage == "scout" else None),
        coverage=FakeCoverageEvaluator(
            [CoverageReport(sufficient=False)],
            failure if stage == "coverage" else None,
        ),
    )

    result = await agent.run("question")

    assert result.stop_reason is StopReason.ERROR
    assert any(f"{stage} failed" in error for error in result.errors)
    assert result.final_state is not None


@pytest.mark.asyncio
async def test_query_budget_truncates_planned_queries_before_dispatch() -> None:
    planner = FakePlanner(
        [[query(f"q-{index}", f"query {index}", "pubmed") for index in range(3)]]
    )
    source = FakeSource(
        "pubmed",
        {f"query {index}": [paper(f"paper-{index}", "pubmed")] for index in range(3)},
    )
    agent = build_agent(
        planner=planner,
        sources=[source],
        coverage=FakeCoverageEvaluator([CoverageReport(sufficient=False)]),
    )

    result = await agent.run("question", budget=SearchBudget(max_queries=2))

    assert len(source.calls) == 2
    assert len(result.queries) == 2
    assert result.stop_reason is StopReason.QUERY_BUDGET


@pytest.mark.asyncio
async def test_max_rounds_stops_an_unsatisfied_search() -> None:
    agent = build_agent(
        coverage=FakeCoverageEvaluator([CoverageReport(sufficient=False)])
    )

    result = await agent.run("question", budget=SearchBudget(max_rounds=1))

    assert result.iterations == 1
    assert result.stop_reason is StopReason.MAX_ROUNDS


@pytest.mark.asyncio
async def test_unique_paper_budget_keeps_ranked_prefix() -> None:
    source = FakeSource(
        "pubmed",
        {
            "core query": [
                paper("paper-1", "pubmed"),
                paper("paper-2", "pubmed"),
                paper("paper-3", "pubmed"),
            ]
        },
    )
    agent = build_agent(
        sources=[source],
        coverage=FakeCoverageEvaluator([CoverageReport(sufficient=False)]),
    )

    result = await agent.run("question", budget=SearchBudget(max_papers=2))

    assert [item.paper_id for item in result.candidates] == ["paper-1", "paper-2"]
    assert result.stop_reason is StopReason.PAPER_BUDGET


@pytest.mark.asyncio
async def test_paper_budget_counts_unique_hits_before_candidate_pruning() -> None:
    source = FakeSource(
        "pubmed",
        {
            "core query": [
                paper("paper-1", "pubmed"),
                paper("paper-2", "pubmed"),
            ]
        },
    )
    agent = build_agent(
        sources=[source],
        coverage=FakeCoverageEvaluator([CoverageReport(sufficient=False)]),
        candidate_limit=1,
    )

    result = await agent.run("question", budget=SearchBudget(max_papers=2))

    assert [item.paper_id for item in result.candidates] == ["paper-1"]
    assert result.final_state is not None
    assert result.final_state.unique_papers_seen == 2
    assert result.stop_reason is StopReason.PAPER_BUDGET


@pytest.mark.asyncio
async def test_estimated_token_budget_stops_search() -> None:
    agent = build_agent(
        coverage=FakeCoverageEvaluator([CoverageReport(sufficient=False)]),
        token_estimator=lambda _: 10,
    )

    result = await agent.run("question", budget=SearchBudget(max_tokens=5))

    assert result.final_state is not None
    assert result.final_state.estimated_tokens_used == 10
    assert result.stop_reason is StopReason.TOKEN_BUDGET


@pytest.mark.asyncio
async def test_wall_clock_budget_uses_injected_monotonic_clock() -> None:
    ticks = iter([0.0, 10.0])
    agent = build_agent(
        coverage=FakeCoverageEvaluator([CoverageReport(sufficient=False)]),
        clock=lambda: next(ticks),
    )

    result = await agent.run("question", budget=SearchBudget(max_seconds=5))

    assert result.stop_reason is StopReason.TIME_BUDGET


@pytest.mark.asyncio
async def test_global_time_budget_cancels_a_slow_planner() -> None:
    class SlowPlanner:
        async def plan(self, *args, **kwargs):
            await asyncio.sleep(1.5)
            return [query("q-1", "core query", "pubmed")]

    agent = build_agent(planner=SlowPlanner())
    started = time.perf_counter()

    result = await agent.run("question", budget=SearchBudget(max_seconds=1))

    elapsed = time.perf_counter() - started
    assert elapsed < 1.3
    assert result.stop_reason is StopReason.TIME_BUDGET
    assert any("time budget" in error for error in result.errors)


@pytest.mark.asyncio
async def test_agent_reports_cumulative_elapsed_time_for_each_tool_stage() -> None:
    class IncrementingClock:
        def __init__(self) -> None:
            self.value = -0.25

        def __call__(self) -> float:
            self.value += 0.25
            return self.value

    agent = build_agent(stage_clock=IncrementingClock())

    result = await agent.run("question", budget=SearchBudget(max_seconds=1000))

    assert set(result.stage_elapsed_seconds) == {
        "query_planner",
        "source_search",
        "paper_deduplicator",
        "paper_ranker",
        "scout_reader",
        "coverage_evaluator",
    }
    assert all(
        math.isfinite(seconds) and seconds >= 0
        for seconds in result.stage_elapsed_seconds.values()
    )


@pytest.mark.asyncio
async def test_two_successful_but_empty_rounds_stop_with_no_results() -> None:
    planner = FakePlanner(
        [
            [query("q-1", "empty one", "pubmed")],
            [query("q-2", "empty two", "pubmed")],
        ]
    )
    agent = build_agent(
        planner=planner,
        sources=[FakeSource("pubmed")],
        coverage=FakeCoverageEvaluator(
            [CoverageReport(sufficient=False), CoverageReport(sufficient=False)]
        ),
    )

    result = await agent.run("question", budget=SearchBudget(max_rounds=3))

    assert result.iterations == 2
    assert result.stop_reason is StopReason.NO_RESULTS


@pytest.mark.asyncio
async def test_two_low_gain_rounds_stop_search() -> None:
    planner = FakePlanner(
        [
            [query("q-1", "one", "pubmed")],
            [query("q-2", "two", "pubmed")],
        ]
    )
    source = FakeSource(
        "pubmed",
        {
            "one": [paper("paper-1", "pubmed")],
            "two": [paper("paper-2", "pubmed")],
        },
    )
    agent = build_agent(
        planner=planner,
        sources=[source],
        coverage=FakeCoverageEvaluator(
            [CoverageReport(sufficient=False), CoverageReport(sufficient=False)]
        ),
        min_new_papers=2,
    )

    result = await agent.run("question", budget=SearchBudget(max_rounds=3))

    assert result.iterations == 2
    assert result.stop_reason is StopReason.LOW_MARGINAL_GAIN


@pytest.mark.asyncio
async def test_duplicate_queries_are_not_dispatched_again() -> None:
    duplicate = query("q-duplicate", "  CORE   QUERY ", "PUBMED")
    planner = FakePlanner(
        [
            [query("q-1", "core query", "pubmed")],
            [duplicate, query("q-2", "new query", "pubmed")],
        ]
    )
    source = FakeSource(
        "pubmed",
        {
            "core query": [paper("paper-1", "pubmed")],
            "new query": [paper("paper-2", "pubmed")],
        },
    )
    agent = build_agent(
        planner=planner,
        sources=[source],
        coverage=FakeCoverageEvaluator(
            [CoverageReport(sufficient=False), CoverageReport(sufficient=True)]
        ),
    )

    result = await agent.run("question")

    assert [call.text for call in source.calls] == ["core query", "new query"]
    assert [item.query_id for item in result.queries] == ["q-1", "q-2"]


@pytest.mark.asyncio
async def test_papers_are_deduplicated_across_rounds() -> None:
    planner = FakePlanner(
        [
            [query("q-1", "one", "pubmed")],
            [query("q-2", "two", "pubmed")],
        ]
    )
    source = FakeSource(
        "pubmed",
        {
            "one": [paper("shared", "pubmed")],
            "two": [paper("shared", "pubmed")],
        },
    )
    agent = build_agent(
        planner=planner,
        sources=[source],
        coverage=FakeCoverageEvaluator(
            [CoverageReport(sufficient=False), CoverageReport(sufficient=True)]
        ),
    )

    result = await agent.run("question")

    assert [item.paper_id for item in result.candidates] == ["shared"]
    assert result.papers_found == 2
    assert result.papers_after_dedup == 1


@pytest.mark.asyncio
async def test_existing_catalog_record_is_reused_as_canonical_paper() -> None:
    catalog_paper = PaperRecord(
        paper_id="paper-1",
        title="Canonical catalog title",
        sources=["catalog"],
    )
    source = FakeSource(
        "pubmed",
        {"core query": [paper("paper-1", "pubmed")]},
    )
    agent = build_agent(sources=[source])

    result = await agent.run("question", existing_papers=[catalog_paper])

    assert result.final_papers[0].title == "Canonical catalog title"
    assert result.reused_paper_ids == ["paper-1"]
