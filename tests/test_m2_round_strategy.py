"""Tests for the entity must/unmust grouped round strategy (task #39).

Covers:
- entity classification normal path + LLM failure degradation;
- Case A / B1 / B2 round planning (grouping correctness, ≤4 entities);
- must soft constraint (>3 must entities never errors);
- zero-result rescue (≥2 zero sources or total ≤ threshold → one extra
  must-only round, at most once per sub-question, total ≤3 rounds);
- planner receives the round's focus entities;
- downstream dedup/rank/scout/retention are still invoked;
- supplement (gap) path and minimal path stay on the legacy flow.
"""

from __future__ import annotations

from typing import Any, List

import pytest

from hypoforge.literature.models import (
    CoverageReport,
    PaperRecord,
    QueryIntent,
    SearchBudget,
    SearchQuery,
    StopReason,
)
from hypoforge.literature.search.agent import IterativeSearchAgent
from hypoforge.literature.search.round_plan import (
    DEFAULT_POOR_RESULT_THRESHOLD,
    EntityClassification,
    classify_entities,
    deterministic_classification,
    plan_entity_rounds,
    poor_round_reason,
)

from tests.literature.fakes import (
    FakeCoverageEvaluator,
    FakeDeduplicator,
    FakePlanner,
    FakeRanker,
    FakeScoutReader,
    FakeSource,
)

SUB_QUESTION = "Hsp70 如何协助蛋白质折叠？"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def query(query_id: str, text: str, source: str = "pubmed") -> SearchQuery:
    return SearchQuery(
        query_id=query_id,
        text=text,
        intent=QueryIntent.CORE,
        target_source=source,
        purpose="Find direct evidence",
        relation_to_question="Directly addresses the sub-question",
    )


def paper(paper_id: str, source: str = "pubmed") -> PaperRecord:
    return PaperRecord(
        paper_id=paper_id,
        title=f"Paper {paper_id}",
        abstract=f"Abstract for {paper_id}",
        sources=[source],
    )


class ScriptedClient:
    """Returns scripted structured_chat payloads in order."""

    def __init__(self, payloads: List[Any]) -> None:
        self.payloads = list(payloads)
        self.prompts: List[str] = []

    async def structured_chat(self, **kwargs):
        self.prompts.append(str(kwargs.get("user_prompt", "")))
        if not self.payloads:
            raise RuntimeError("no scripted payload left")
        payload = self.payloads.pop(0)
        if isinstance(payload, BaseException):
            raise payload
        return payload


def build_agent(
    *,
    planner: FakePlanner,
    sources: list[FakeSource],
    classifier: Any = None,
    deduplicator: FakeDeduplicator | None = None,
    ranker: FakeRanker | None = None,
    scout: FakeScoutReader | None = None,
    **kwargs,
) -> IterativeSearchAgent:
    return IterativeSearchAgent(
        query_planner=planner,
        sources=sources,
        deduplicator=deduplicator or FakeDeduplicator(),
        ranker=ranker or FakeRanker(),
        scout_reader=scout or FakeScoutReader(),
        coverage_evaluator=FakeCoverageEvaluator(
            [CoverageReport(sufficient=False)]
        ),
        entity_classifier=classifier,
        **kwargs,
    )


async def run_entity_group(
    agent: IterativeSearchAgent,
    key_entities: List[str],
    budget: SearchBudget | None = None,
    **kwargs,
):
    return await agent.run(
        SUB_QUESTION,
        key_entities=key_entities,
        budget=budget or SearchBudget(max_rounds=3, max_queries=24),
        round_strategy="entity_group",
        **kwargs,
    )


# ---------------------------------------------------------------------------
# entity classification
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_classify_entities_normal_path() -> None:
    client = ScriptedClient([
        {"must_entities": ["Hsp70", "protein folding"], "unmapped_entities": ["ATP"]},
    ])

    result = await classify_entities(
        client, SUB_QUESTION, ["Hsp70", "protein folding", "ATP"], ["biology"]
    )

    assert result.must_entities == ["Hsp70", "protein folding"]
    assert result.unmapped_entities == ["ATP"]
    assert result.degraded is False


@pytest.mark.asyncio
async def test_classify_entities_llm_failure_degrades_to_deterministic() -> None:
    client = ScriptedClient([RuntimeError("llm down")])

    result = await classify_entities(
        client, SUB_QUESTION, ["Hsp70", "protein folding", "ATP", "aging"]
    )

    assert result.degraded is True
    assert result.must_entities == ["Hsp70", "protein folding"]
    assert result.unmapped_entities == ["ATP", "aging"]


@pytest.mark.asyncio
async def test_classify_entities_missing_entities_fall_into_unmapped() -> None:
    # The LLM forgot one entity entirely; it must not vanish.
    client = ScriptedClient([{"must_entities": ["Hsp70"], "unmapped_entities": []}])

    result = await classify_entities(
        client, SUB_QUESTION, ["Hsp70", "protein folding", "ATP"]
    )

    assert result.must_entities == ["Hsp70"]
    assert set(result.unmapped_entities) == {"protein folding", "ATP"}


@pytest.mark.asyncio
async def test_classify_entities_soft_limit_allows_more_than_three_must() -> None:
    client = ScriptedClient([
        {
            "must_entities": ["a", "b", "c", "d", "e"],
            "unmapped_entities": ["f", "g"],
        },
    ])

    result = await classify_entities(
        client, SUB_QUESTION, ["a", "b", "c", "d", "e", "f", "g"]
    )

    assert len(result.must_entities) == 5  # tolerated, never raises
    assert result.unmapped_entities == ["f", "g"]


def test_deterministic_classification_shapes() -> None:
    empty = deterministic_classification([])
    assert empty.must_entities == []
    assert empty.unmapped_entities == []
    single = deterministic_classification(["only"])
    assert single.must_entities == ["only"]
    assert single.unmapped_entities == []


# ---------------------------------------------------------------------------
# round planning: Case A / B1 / B2
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_plan_case_a_single_round_with_all_entities() -> None:
    classification = EntityClassification(
        must_entities=["Hsp70"], unmapped_entities=["ATP", "folding"]
    )

    plan = await plan_entity_rounds(None, SUB_QUESTION, classification)

    assert plan.case == "A"
    assert len(plan.rounds) == 1
    assert plan.rounds[0].focus_entities == ["Hsp70", "ATP", "folding"]
    assert plan.rescue_focus == ["Hsp70"]


@pytest.mark.asyncio
async def test_plan_case_b1_every_round_keeps_all_must() -> None:
    classification = EntityClassification(
        must_entities=["Hsp70", "folding"],
        unmapped_entities=["ATP", "aging", "proteostasis"],
    )

    plan = await plan_entity_rounds(None, SUB_QUESTION, classification)

    assert plan.case == "B1"
    assert len(plan.rounds) == 2
    for spec in plan.rounds:
        assert spec.focus_entities[:2] == ["Hsp70", "folding"]
        assert len(spec.focus_entities) <= 4
    assigned = [
        entity
        for spec in plan.rounds
        for entity in spec.focus_entities[2:]
    ]
    # Deterministic chunking (chunk_size=2) distributes all optional
    # entities across the two rounds.
    assert assigned == ["ATP", "aging", "proteostasis"]
    assert all(1 <= len(spec.focus_entities[2:]) <= 2 for spec in plan.rounds)


@pytest.mark.asyncio
async def test_plan_case_b2_first_two_must_group_one_rest_group_two() -> None:
    classification = EntityClassification(
        must_entities=["m1", "m2", "m3"],
        unmapped_entities=["u1", "u2", "u3", "u4"],
    )

    plan = await plan_entity_rounds(None, SUB_QUESTION, classification)

    assert plan.case == "B2"
    assert len(plan.rounds) == 2
    assert plan.rounds[0].focus_entities[:2] == ["m1", "m2"]
    assert plan.rounds[1].focus_entities[0] == "m3"
    for spec in plan.rounds:
        assert len(spec.focus_entities) <= 4
    # Unmust split into two groups, both non-empty when possible.
    first_unmapped = plan.rounds[0].focus_entities[2:]
    second_unmapped = plan.rounds[1].focus_entities[1:]
    assert first_unmapped and second_unmapped
    assert set(first_unmapped) | set(second_unmapped) == {"u1", "u2", "u3", "u4"}


@pytest.mark.asyncio
async def test_plan_case_b2_rebalances_empty_second_group() -> None:
    classification = EntityClassification(
        must_entities=["m1", "m2", "m3"],
        unmapped_entities=["u1", "u2"],
    )

    plan = await plan_entity_rounds(None, SUB_QUESTION, classification)

    assert plan.case == "B2"
    assert plan.rounds[0].focus_entities == ["m1", "m2", "u1"]
    assert plan.rounds[1].focus_entities == ["m3", "u2"]


@pytest.mark.asyncio
async def test_plan_case_b1_uses_llm_grouping_when_available() -> None:
    client = ScriptedClient([
        {"rounds": [["proteostasis"], ["ATP"]], "reasoning": "balanced"},
    ])
    classification = EntityClassification(
        must_entities=["Hsp70"],
        unmapped_entities=["ATP", "aging", "proteostasis", "chaperone"],
    )

    plan = await plan_entity_rounds(client, SUB_QUESTION, classification)

    assert plan.case == "B1"
    assert plan.rounds[0].focus_entities == ["Hsp70", "proteostasis"]
    assert plan.rounds[1].focus_entities == ["Hsp70", "ATP"]


@pytest.mark.asyncio
async def test_plan_case_b1_llm_grouping_failure_falls_back() -> None:
    client = ScriptedClient([RuntimeError("grouping unavailable")])
    classification = EntityClassification(
        must_entities=["Hsp70"],
        unmapped_entities=["ATP", "aging", "proteostasis", "chaperone"],
    )

    plan = await plan_entity_rounds(client, SUB_QUESTION, classification)

    assert plan.case == "B1"
    assert [spec.focus_entities for spec in plan.rounds] == [
        ["Hsp70", "ATP", "aging"],
        ["Hsp70", "proteostasis", "chaperone"],
    ]


@pytest.mark.asyncio
async def test_plan_with_more_than_three_must_never_raises() -> None:
    # Soft constraint: 4 must entities are accepted and grouped B2-style.
    classification = EntityClassification(
        must_entities=["m1", "m2", "m3", "m4"],
        unmapped_entities=["u1", "u2"],
    )

    plan = await plan_entity_rounds(None, SUB_QUESTION, classification)

    assert plan.case == "B2"
    assert plan.rounds[0].focus_entities[:2] == ["m1", "m2"]
    assert plan.rounds[1].focus_entities[:2] == ["m3", "m4"]


# ---------------------------------------------------------------------------
# poor-round detection
# ---------------------------------------------------------------------------


def test_poor_round_reason_thresholds() -> None:
    # Two zero-result sources trigger regardless of total.
    assert poor_round_reason({"pubmed": 0, "openalex": 0, "arxiv": 9}, 9)
    # A single zero source with plenty of results is fine.
    assert not poor_round_reason({"pubmed": 0, "openalex": 5}, 5)
    # Low totals are poor even with no zero source.
    assert poor_round_reason(
        {"pubmed": 1, "openalex": 1}, DEFAULT_POOR_RESULT_THRESHOLD
    )
    assert not poor_round_reason(
        {"pubmed": 2, "openalex": 2}, DEFAULT_POOR_RESULT_THRESHOLD + 2
    )


# ---------------------------------------------------------------------------
# agent integration: focus entities reach the planner
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_planner_receives_round_focus_entities() -> None:
    planner = FakePlanner([[query("q-1", "Hsp70 folding", "pubmed")]])
    source = FakeSource(
        "pubmed", {"Hsp70 folding": [paper(f"p-{i}") for i in range(4)]}
    )
    client = ScriptedClient([
        {"must_entities": ["Hsp70"], "unmapped_entities": ["ATP"]},
    ])
    agent = build_agent(planner=planner, sources=[source], classifier=client)

    await run_entity_group(agent, key_entities=["Hsp70", "ATP"])

    # Case A: one base round focused on all entities.
    assert planner.focus_calls == [["Hsp70", "ATP"]]
    assert planner.supplement_calls == [[]]


@pytest.mark.asyncio
async def test_planner_receives_focus_entities_per_round_in_b1() -> None:
    planner = FakePlanner(
        [
            [query("q-1", "round one", "pubmed")],
            [query("q-2", "round two", "pubmed")],
        ]
    )
    source = FakeSource(
        "pubmed",
        {
            "round one": [paper(f"a-{i}") for i in range(3)],
            "round two": [paper(f"b-{i}") for i in range(3)],
        },
    )
    client = ScriptedClient([
        {
            "must_entities": ["Hsp70"],
            "unmapped_entities": ["ATP", "aging", "proteostasis", "chaperone"],
        },
    ])
    agent = build_agent(planner=planner, sources=[source], classifier=client)

    result = await run_entity_group(
        agent,
        key_entities=["Hsp70", "ATP", "aging", "proteostasis", "chaperone"],
    )

    assert planner.focus_calls == [
        ["Hsp70", "ATP", "aging"],
        ["Hsp70", "proteostasis", "chaperone"],
    ]
    assert result.iterations == 2
    assert result.stop_reason is not StopReason.ERROR


# ---------------------------------------------------------------------------
# zero-result rescue
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rescue_triggered_when_two_sources_return_zero() -> None:
    planner = FakePlanner(
        [
            [query("q-1", "narrow round", "pubmed")],
            [query("q-2", "narrow round two", "openalex")],
            [query("q-3", "must only rescue", "pubmed")],
        ]
    )
    pubmed = FakeSource(
        "pubmed",
        {
            "narrow round": [],
            "must only rescue": [paper(f"r-{i}") for i in range(3)],
        },
    )
    openalex = FakeSource("openalex", {"narrow round two": []})
    client = ScriptedClient([
        {"must_entities": ["Hsp70"], "unmapped_entities": ["ATP"]},
    ])
    agent = build_agent(
        planner=planner, sources=[pubmed, openalex], classifier=client
    )

    result = await run_entity_group(agent, key_entities=["Hsp70", "ATP"])

    # Round 1 (all entities) is poor → Case A must-only follow-up is also
    # poor → rescue round must NOT fire a second time.  ≤3 rounds total.
    assert result.iterations <= 3
    # The rescue/follow-up round queries were built around must entities only.
    rescue_calls = planner.focus_calls[1:]
    assert rescue_calls, "expected a follow-up/rescue round"
    for call in rescue_calls:
        assert call == ["Hsp70"]


@pytest.mark.asyncio
async def test_rescue_triggers_at_most_once_per_sub_question() -> None:
    planner = FakePlanner(
        [
            [query(f"q-{index}", f"empty {index}", "pubmed")]
            for index in range(6)
        ]
    )
    source = FakeSource("pubmed")  # always empty
    client = ScriptedClient([
        {
            "must_entities": ["Hsp70"],
            "unmapped_entities": ["ATP", "aging", "proteostasis", "chaperone"],
        },
    ])
    agent = build_agent(planner=planner, sources=[source], classifier=client)

    result = await run_entity_group(
        agent,
        key_entities=["Hsp70", "ATP", "aging", "proteostasis", "chaperone"],
    )

    # B1: 2 base rounds + at most 1 rescue = hard cap of 3 rounds.
    assert result.iterations == 3
    rescue_rounds = [
        call for call in planner.focus_calls if call == ["Hsp70"]
    ]
    assert len(rescue_rounds) == 1
    assert result.stop_reason is not StopReason.ERROR


@pytest.mark.asyncio
async def test_no_rescue_when_results_are_good() -> None:
    planner = FakePlanner(
        [
            [query("q-1", "round one", "pubmed")],
            [query("q-2", "round two", "pubmed")],
        ]
    )
    source = FakeSource(
        "pubmed",
        {
            "round one": [paper(f"a-{i}") for i in range(4)],
            "round two": [paper(f"b-{i}") for i in range(4)],
        },
    )
    client = ScriptedClient([
        {
            "must_entities": ["Hsp70"],
            "unmapped_entities": ["ATP", "aging", "proteostasis", "chaperone"],
        },
    ])
    agent = build_agent(planner=planner, sources=[source], classifier=client)

    result = await run_entity_group(
        agent,
        key_entities=["Hsp70", "ATP", "aging", "proteostasis", "chaperone"],
    )

    assert result.iterations == 2
    assert all(call != ["Hsp70"] for call in planner.focus_calls)


@pytest.mark.asyncio
async def test_low_total_results_trigger_rescue() -> None:
    planner = FakePlanner(
        [
            [query("q-1", "sparse round", "pubmed")],
            [query("q-2", "must rescue", "pubmed")],
        ]
    )
    source = FakeSource(
        "pubmed",
        {
            # Only 1 paper in round 1 → poor by total threshold.
            "sparse round": [paper("only-one")],
            "must rescue": [paper(f"r-{i}") for i in range(3)],
        },
    )
    client = ScriptedClient([
        {
            "must_entities": ["Hsp70"],
            "unmapped_entities": ["ATP", "aging", "proteostasis", "chaperone"],
        },
    ])
    agent = build_agent(planner=planner, sources=[source], classifier=client)

    result = await run_entity_group(
        agent,
        key_entities=["Hsp70", "ATP", "aging", "proteostasis", "chaperone"],
    )

    assert planner.focus_calls[1] == ["Hsp70"]
    assert result.iterations >= 2


# ---------------------------------------------------------------------------
# downstream stages still run + round cap
# ---------------------------------------------------------------------------


class CountingDeduplicator(FakeDeduplicator):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    async def deduplicate(self, papers, existing_papers=()):
        self.calls += 1
        return await super().deduplicate(papers, existing_papers)


class CountingRanker(FakeRanker):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    async def rank(self, sub_question, papers, limit):
        self.calls += 1
        return await super().rank(sub_question, papers, limit)


@pytest.mark.asyncio
async def test_downstream_dedup_rank_scout_still_invoked() -> None:
    planner = FakePlanner([[query("q-1", "core", "pubmed")]])
    source = FakeSource("pubmed", {"core": [paper(f"p-{i}") for i in range(3)]})
    dedup = CountingDeduplicator()
    ranker = CountingRanker()
    scout = FakeScoutReader()
    client = ScriptedClient([
        {"must_entities": ["Hsp70"], "unmapped_entities": ["ATP"]},
    ])
    agent = build_agent(
        planner=planner,
        sources=[source],
        classifier=client,
        deduplicator=dedup,
        ranker=ranker,
        scout=scout,
    )

    result = await run_entity_group(agent, key_entities=["Hsp70", "ATP"])

    assert dedup.calls == result.iterations
    assert ranker.calls == result.iterations
    assert scout.calls == [["p-0", "p-1", "p-2"]]
    assert [item.paper_id for item in result.final_papers] == [
        "p-0",
        "p-1",
        "p-2",
    ]
    assert result.retention_decisions, "retention decisions must be produced"


@pytest.mark.asyncio
async def test_round_cap_respects_budget_max_rounds() -> None:
    planner = FakePlanner(
        [[query(f"q-{index}", f"query {index}", "pubmed")] for index in range(5)]
    )
    source = FakeSource("pubmed")  # empty → every round poor
    client = ScriptedClient([
        {
            "must_entities": ["Hsp70"],
            "unmapped_entities": ["ATP", "aging", "proteostasis", "chaperone"],
        },
    ])
    agent = build_agent(planner=planner, sources=[source], classifier=client)

    result = await agent.run(
        SUB_QUESTION,
        key_entities=["Hsp70", "ATP", "aging", "proteostasis", "chaperone"],
        budget=SearchBudget(max_rounds=3, max_queries=24),
        round_strategy="entity_group",
    )

    assert result.iterations <= 3


# ---------------------------------------------------------------------------
# classification failure degrades but the search proceeds
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_classification_failure_degrades_and_search_continues() -> None:
    planner = FakePlanner([[query("q-1", "core", "pubmed")]])
    source = FakeSource("pubmed", {"core": [paper(f"p-{i}") for i in range(3)]})
    client = ScriptedClient([RuntimeError("classifier offline")])
    agent = build_agent(planner=planner, sources=[source], classifier=client)

    result = await run_entity_group(
        agent, key_entities=["Hsp70", "ATP", "folding"]
    )

    # Deterministic fallback: first 2 entities are must → Case A single round
    # with all entities; search completed instead of erroring.
    assert planner.focus_calls == [["Hsp70", "ATP", "folding"]]
    assert result.stop_reason is not StopReason.ERROR
    assert result.final_papers


# ---------------------------------------------------------------------------
# regression: legacy paths unaffected
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_default_strategy_stays_coverage_driven() -> None:
    planner = FakePlanner(
        [
            [query("q-1", "core query", "pubmed")],
            [query("q-2", "gap query", "pubmed")],
        ]
    )
    source = FakeSource(
        "pubmed",
        {
            "core query": [paper("paper-1")],
            "gap query": [paper("paper-2")],
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
                CoverageReport(sufficient=False, missing_topics=["gap"]),
                CoverageReport(sufficient=True),
            ]
        ),
    )

    result = await agent.run(
        SUB_QUESTION,
        key_entities=["Hsp70"],
        budget=SearchBudget(max_rounds=3),
    )

    # Coverage gap still triggers a second round on the default path.
    assert result.iterations == 2
    assert result.stop_reason is StopReason.COVERAGE_SATISFIED
    assert planner.focus_calls == [[], []]


@pytest.mark.asyncio
async def test_supplement_gap_planner_ignores_focus_entities_kwarg() -> None:
    from hypoforge.literature.adapter import _GapQueryPlanner

    gap_query = query("sup-1", "Hsp70 co-chaperone", "pubmed")
    planner = _GapQueryPlanner([gap_query])

    served = await planner.plan(
        SUB_QUESTION,
        key_entities=["Hsp70"],
        focus_entities=["Hsp70"],
        supplement_entities=["ATP"],
    )
    second = await planner.plan(SUB_QUESTION)

    assert served == [gap_query]
    assert second == []
