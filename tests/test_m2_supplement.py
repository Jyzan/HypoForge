"""Tests for the M2 cache-first supplement ("gap-filling") flow.

Covers:

* trigger semantics — fresh round / no open gaps keep the full flow;
* cache-hit path — ``PaperStore`` hits become paper-level increments,
  gaps move to ``pending_grounding`` and a ``memory_hit`` event fires;
* query dedup against ``search_ledger``;
* monotonic merge of ``literature_results`` / appended export runs that
  still satisfy ``M2KnowledgeRun.validate_provenance``;
* ``supplement_paper_budget`` enforcement;
* ``PaperStore`` unit behaviour (key priority, idempotent upsert,
  query-cache round-trip, missing-file tolerance);
* legacy M2 short-circuit.
"""

from __future__ import annotations

import pytest

from hypoforge.literature.adapter import AgenticM2Adapter
from hypoforge.literature.models import (
    CoverageReport,
    EvidenceChunk,
    EvidenceLinkedKnowledge,
    PaperReadingResult,
    PaperRecord,
    SearchRunResult,
    StopReason,
)
from hypoforge.literature.search import IterativeSearchAgent
from hypoforge.memory import PaperStore, paper_key
from hypoforge.observability import RunEventRecorder, bind_recorder
from hypoforge.state import (
    ConfidenceLevel,
    EvidenceGap,
    KnowledgeEntry,
    KnowledgeEntryType,
    LiteratureResult,
    M2KnowledgeExport,
    M2KnowledgeRun,
    M2PaperExport,
    PipelineState,
    ProblemCard,
    SearchLedger,
)

from tests.literature.fakes import (
    FakeCoverageEvaluator,
    FakeDeduplicator,
    FakePlanner,
    FakeRanker,
    FakeReadingWorkflow,
    FakeScoutReader,
    FakeSource,
)


SUB_QUESTION = "What is the Hsp70 mechanism?"


# ---------------------------------------------------------------------------
# Local fakes / helpers
# ---------------------------------------------------------------------------


class RecordingSearchAgent:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    async def run(self, sub_question, **kwargs):
        self.calls.append((sub_question, kwargs))
        return self.results[len(self.calls) - 1].model_copy(deep=True)


class ExplodingSearchAgent:
    async def run(self, *args, **kwargs):
        raise AssertionError("search agent must not run on a cache hit")


class ExplodingReadingWorkflow:
    async def run(self, *args, **kwargs):
        raise AssertionError("reading workflow must not run on a cache hit")


class EchoReadingWorkflow:
    """Returns one reading per input paper (optionally with evidence)."""

    def __init__(self, with_evidence: bool = False) -> None:
        self.with_evidence = with_evidence
        self.calls = []

    async def run(self, sub_question, papers, search_context=None):
        self.calls.append([paper.paper_id for paper in papers])
        results = []
        for paper in papers:
            if not self.with_evidence:
                results.append(PaperReadingResult(paper_id=paper.paper_id))
                continue
            results.append(
                PaperReadingResult(
                    paper_id=paper.paper_id,
                    evidence=[
                        EvidenceChunk(
                            evidence_id=f"ev-{paper.paper_id}",
                            paper_id=paper.paper_id,
                            chunk_id=f"chunk-{paper.paper_id}",
                            quote="observed binding",
                            normalized_claim="observed binding",
                            relevance_score=0.9,
                        )
                    ],
                    knowledge_entries=[
                        EvidenceLinkedKnowledge(
                            entry_id=f"ke-{paper.paper_id}",
                            entry_type=KnowledgeEntryType.MECHANISTIC_CONCLUSION,
                            content=f"Finding from {paper.paper_id}",
                            confidence=ConfidenceLevel.HIGH,
                            evidence_ids=[f"ev-{paper.paper_id}"],
                        )
                    ],
                )
            )
        return results


def make_problem_card() -> ProblemCard:
    return ProblemCard(
        original_question="Q",
        sub_questions=[SUB_QUESTION],
        key_entities=["Hsp70"],
        domain=["biology"],
    )


def make_prior_run() -> M2KnowledgeRun:
    return M2KnowledgeRun(
        sub_question=SUB_QUESTION,
        papers=[M2PaperExport(paper_id="old-1", title="Old paper", doi="10.1000/old")],
    )


def make_prior_entry() -> KnowledgeEntry:
    return KnowledgeEntry(
        id="KE_old",
        type=KnowledgeEntryType.ESTABLISHED_FACT,
        content="old fact",
        source_paper_id="old-1",
        evidence_ids=["ev-old"],
    )


def make_supplement_state(**overrides) -> PipelineState:
    base = dict(
        input_question="Q",
        problem_card=make_problem_card(),
        search_round=1,
        run_id="run-1",
        literature_results=[
            LiteratureResult(
                sub_question=SUB_QUESTION,
                papers_retrieved=1,
                knowledge_entries=[make_prior_entry()],
            )
        ],
        m2_knowledge_export=M2KnowledgeExport(runs=[make_prior_run()]),
        search_ledger=SearchLedger(
            queries_issued=["initial query"], paper_keys=["doi:10.1000/old"]
        ),
    )
    base.update(overrides)
    return PipelineState(**base)


def make_base_agent(source_results):
    source = FakeSource("pubmed", results=source_results)
    agent = IterativeSearchAgent(
        query_planner=FakePlanner([[]]),
        sources=[source],
        deduplicator=FakeDeduplicator(),
        ranker=FakeRanker(),
        scout_reader=FakeScoutReader(),
        coverage_evaluator=FakeCoverageEvaluator(
            [CoverageReport(sufficient=True)]
        ),
    )
    return agent, source


def make_gap(**overrides) -> EvidenceGap:
    base = dict(
        description="Missing Hsp70 co-chaperone data",
        suggested_queries=["Hsp70 co-chaperone binding"],
        target_sub_question=SUB_QUESTION,
    )
    base.update(overrides)
    return EvidenceGap(**base)


# ---------------------------------------------------------------------------
# Trigger semantics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_first_round_runs_full_flow() -> None:
    paper = PaperRecord(paper_id="p1", title="T", sources=["pubmed"])
    agent = RecordingSearchAgent(
        [
            SearchRunResult(
                sub_question=SUB_QUESTION,
                final_papers=[paper],
                stop_reason=StopReason.COVERAGE_SATISFIED,
            )
        ]
    )
    adapter = AgenticM2Adapter(
        search_agent=agent,
        reading_workflow=FakeReadingWorkflow([PaperReadingResult(
            paper_id="p1",
            evidence=[EvidenceChunk(
                evidence_id="ev-p1", paper_id="p1", chunk_id="chunk-p1",
                quote="Observed binding.", normalized_claim="Observed binding.", relevance_score=0.9,
            )],
            knowledge_entries=[EvidenceLinkedKnowledge(
                entry_id="ke-p1",
                entry_type=KnowledgeEntryType.MECHANISTIC_CONCLUSION,
                content="Observed binding.",
                confidence=ConfidenceLevel.HIGH,
                evidence_ids=["ev-p1"],
            )],
        )]),
    )
    state = PipelineState(input_question="Q", problem_card=make_problem_card())

    output = await adapter(state)

    assert len(agent.calls) == 1
    # fresh flow returns the Agentic literature and grounded export fields
    assert set(output) == {"literature_results", "m2_knowledge_export"}


@pytest.mark.asyncio
async def test_no_open_gap_runs_full_flow_even_on_later_round() -> None:
    paper = PaperRecord(paper_id="p1", title="T", sources=["pubmed"])
    agent = RecordingSearchAgent(
        [
            SearchRunResult(
                sub_question=SUB_QUESTION,
                final_papers=[paper],
                stop_reason=StopReason.COVERAGE_SATISFIED,
            )
        ]
    )
    adapter = AgenticM2Adapter(
        search_agent=agent,
        reading_workflow=FakeReadingWorkflow([PaperReadingResult(
            paper_id="p1",
            evidence=[EvidenceChunk(
                evidence_id="ev-p1", paper_id="p1", chunk_id="chunk-p1",
                quote="Observed binding.", normalized_claim="Observed binding.", relevance_score=0.9,
            )],
            knowledge_entries=[EvidenceLinkedKnowledge(
                entry_id="ke-p1",
                entry_type=KnowledgeEntryType.MECHANISTIC_CONCLUSION,
                content="Observed binding.",
                confidence=ConfidenceLevel.HIGH,
                evidence_ids=["ev-p1"],
            )],
        )]),
    )
    state = make_supplement_state(
        evidence_gaps=[make_gap(status="closed")],
    )

    output = await adapter(state)

    assert len(agent.calls) == 1
    assert set(output) == {"literature_results", "m2_knowledge_export"}


# ---------------------------------------------------------------------------
# Cache-hit path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_supplement_cache_hit_rebuilds_paper_level_increment(
    tmp_path,
) -> None:
    cached = M2PaperExport(
        paper_id="cached-1", title="Cached paper", doi="10.1000/cached"
    )
    store = PaperStore(tmp_path)
    keys = store.upsert_papers([cached], run_id="run-0", round=1)
    store.record_query("hsp70 chaperone mechanism", keys, run_id="run-0", round=1)

    state = make_supplement_state(
        memory_cache_dir=str(tmp_path),
        evidence_gaps=[
            make_gap(suggested_queries=["Hsp70 chaperone mechanism"])
        ],
    )
    adapter = AgenticM2Adapter(
        search_agent=ExplodingSearchAgent(),
        reading_workflow=ExplodingReadingWorkflow(),
    )
    recorder = RunEventRecorder(tmp_path / "run", "run-1")

    with bind_recorder(recorder):
        output = await adapter(state)

    # gap attempted → pending_grounding
    gaps = output["evidence_gaps"]
    assert len(gaps) == 1
    assert gaps[0].status == "open"

    # literature_results merged monotonically (old entry preserved)
    results = output["literature_results"]
    assert len(results) == 1
    assert results[0].papers_retrieved == 2
    assert [entry.id for entry in results[0].knowledge_entries] == ["KE_old"]

    # export appends a new, provenance-valid paper-level run
    export = output["m2_knowledge_export"]
    assert len(export.runs) == 2
    new_run = export.runs[1]
    assert new_run.sub_question == SUB_QUESTION
    assert [paper.paper_id for paper in new_run.papers] == ["cached-1"]
    assert new_run.evidence == []
    assert new_run.knowledge_entries == []  # metadata cache hits are not grounding
    assert new_run.search_provenance.stop_reason == "cache_hit"
    assert (
        new_run.search_provenance.queries[0].target_gap
        == "Missing Hsp70 co-chaperone data"
    )

    # ledger gains the cached paper key but no new live query
    ledger = output["search_ledger"]
    assert ledger.queries_issued == ["initial query"]
    assert "doi:10.1000/cached" in ledger.paper_keys

    # memory_hit event carries gap_id + hit count
    hits = [
        event
        for event in recorder.read_events()
        if event["event_type"] == "memory_hit"
    ]
    assert len(hits) == 1
    assert hits[0]["details"]["gap_id"] == gaps[0].gap_id
    assert hits[0]["details"]["hits"] == 1


# ---------------------------------------------------------------------------
# Live gap search path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_supplement_search_merges_and_updates_ledger() -> None:
    new_paper = PaperRecord(
        paper_id="new-1", title="New paper", doi="10.1000/new", sources=["pubmed"]
    )
    agent, source = make_base_agent({"Hsp70 co-chaperone binding": [new_paper]})
    adapter = AgenticM2Adapter(
        search_agent=agent, reading_workflow=EchoReadingWorkflow(with_evidence=True)
    )
    state = make_supplement_state(evidence_gaps=[make_gap()])

    output = await adapter(state)

    # exactly the suggested query hit the source (no cache dir → no lookup)
    assert [query.text for query in source.calls] == ["Hsp70 co-chaperone binding"]

    results = output["literature_results"]
    assert len(results) == 1
    assert results[0].papers_retrieved == 2
    entry_ids = [entry.id for entry in results[0].knowledge_entries]
    assert "KE_old" in entry_ids and "ke-new-1" in entry_ids

    export = output["m2_knowledge_export"]
    assert len(export.runs) == 2
    new_run = export.runs[1]
    assert [paper.paper_id for paper in new_run.papers] == ["new-1"]
    # target_gap activated on the issued queries
    assert new_run.search_provenance.queries
    assert all(
        query.target_gap == "Missing Hsp70 co-chaperone data"
        for query in new_run.search_provenance.queries
    )
    # validate_provenance holds (evidence/knowledge reference the new paper)
    assert new_run.evidence[0].paper_id == "new-1"
    assert new_run.knowledge_entries[0].source_paper_id == "new-1"

    ledger = output["search_ledger"]
    assert ledger.queries_issued == ["initial query", "Hsp70 co-chaperone binding"]
    assert ledger.paper_keys == ["doi:10.1000/old", "doi:10.1000/new"]

    assert output["evidence_gaps"][0].status == "pending_grounding"


@pytest.mark.asyncio
async def test_supplement_dedups_queries_already_in_ledger() -> None:
    new_paper = PaperRecord(
        paper_id="new-1", title="New paper", doi="10.1000/new", sources=["pubmed"]
    )
    agent, source = make_base_agent({"Hsp70 co-chaperone binding": [new_paper]})
    adapter = AgenticM2Adapter(
        search_agent=agent, reading_workflow=EchoReadingWorkflow(with_evidence=True)
    )
    state = make_supplement_state(
        evidence_gaps=[make_gap()],
        search_ledger=SearchLedger(
            queries_issued=["hsp70   co-chaperone binding!!"],
            paper_keys=["doi:10.1000/old"],
        ),
    )

    output = await adapter(state)

    assert source.calls == []  # duplicate query dropped before any search
    assert len(output["m2_knowledge_export"].runs) == 1  # no new run
    assert output["literature_results"][0].papers_retrieved == 1
    assert output["search_ledger"].queries_issued == [
        "hsp70   co-chaperone binding!!"
    ]
    assert output["evidence_gaps"][0].status == "open"


@pytest.mark.asyncio
async def test_supplement_skips_papers_already_in_prior_export() -> None:
    duplicate = PaperRecord(
        paper_id="dup-1", title="Old paper", doi="10.1000/old", sources=["pubmed"]
    )
    agent, source = make_base_agent({"Hsp70 co-chaperone binding": [duplicate]})
    adapter = AgenticM2Adapter(
        search_agent=agent, reading_workflow=EchoReadingWorkflow(with_evidence=True)
    )
    state = make_supplement_state(evidence_gaps=[make_gap()])

    output = await adapter(state)

    assert len(source.calls) == 1
    assert len(output["m2_knowledge_export"].runs) == 1  # duplicate discarded
    assert output["literature_results"][0].papers_retrieved == 1
    assert output["search_ledger"].paper_keys == ["doi:10.1000/old"]
    assert output["evidence_gaps"][0].status == "open"


@pytest.mark.asyncio
async def test_supplement_paper_budget_caps_new_papers() -> None:
    papers = [
        PaperRecord(
            paper_id=f"new-{index}",
            title=f"New paper {index}",
            doi=f"10.1000/new{index}",
            sources=["pubmed"],
        )
        for index in range(3)
    ]
    agent, source = make_base_agent({"Hsp70 co-chaperone binding": papers})
    adapter = AgenticM2Adapter(
        search_agent=agent,
        reading_workflow=EchoReadingWorkflow(with_evidence=True),
        supplement_paper_budget=1,
    )
    state = make_supplement_state(evidence_gaps=[make_gap()])

    output = await adapter(state)

    new_keys = [
        key
        for key in output["search_ledger"].paper_keys
        if key != "doi:10.1000/old"
    ]
    assert len(new_keys) == 1
    assert output["literature_results"][0].papers_retrieved == 2
    assert len(output["m2_knowledge_export"].runs[1].papers) == 1


@pytest.mark.asyncio
async def test_supplement_without_cache_dir_still_searches(tmp_path) -> None:
    new_paper = PaperRecord(
        paper_id="new-1", title="New paper", doi="10.1000/new", sources=["pubmed"]
    )
    agent, source = make_base_agent({"Hsp70 co-chaperone binding": [new_paper]})
    adapter = AgenticM2Adapter(
        search_agent=agent, reading_workflow=EchoReadingWorkflow(with_evidence=True)
    )
    # memory_cache_dir left empty → degraded but functional
    state = make_supplement_state(evidence_gaps=[make_gap()])

    output = await adapter(state)

    assert len(source.calls) == 1
    assert output["literature_results"][0].papers_retrieved == 2


# ---------------------------------------------------------------------------
# PaperStore unit tests
# ---------------------------------------------------------------------------


def test_paper_key_priority_and_normalisation() -> None:
    rich = {
        "doi": "HTTPS://DOI.ORG/10.1000/ABC",
        "pmid": "123",
        "external_ids": {"arxiv": "2301.00001"},
        "title": "Some title",
    }
    assert paper_key(rich) == "doi:10.1000/abc"
    assert paper_key({"pmid": "123", "title": "Some title"}) == "pmid:123"
    assert (
        paper_key({"external_ids": {"arXiv": "arXiv:2301.00001"}, "title": "T"})
        == "arxiv:2301.00001"
    )
    title_key = paper_key({"title": "  A Study   of Chaperones "})
    assert title_key == paper_key({"title": "a study of chaperones"})
    assert title_key.startswith("title:")


def test_paper_store_upsert_idempotent_and_first_seen(tmp_path) -> None:
    store = PaperStore(tmp_path / "cache")
    paper = {"doi": "10.1/x", "title": "T", "abstract": "old"}
    keys_first = store.upsert_papers([paper], run_id="run-1", round=1)
    keys_second = store.upsert_papers(
        [{"doi": "10.1/x", "title": "T", "abstract": "new"}],
        run_id="run-2",
        round=2,
    )
    assert keys_first == keys_second
    assert len(store.lookup_by_keys(keys_first)) == 1

    # reload from disk: one logical record, first_seen preserved
    reloaded = PaperStore(tmp_path / "cache")
    papers = reloaded.lookup_by_keys(keys_first)
    assert len(papers) == 1
    assert papers[0]["abstract"] == "new"
    reloaded._ensure_loaded()
    record = reloaded._papers[keys_first[0]]
    assert record["first_seen_run"] == "run-1"
    assert record["first_seen_round"] == 1
    assert record["last_seen_run"] == "run-2"


def test_paper_store_query_cache_roundtrip_and_normalisation(tmp_path) -> None:
    store = PaperStore(tmp_path / "cache")
    store.record_query("Hsp70  chaperone mechanism!", ["doi:10.1/x"])

    assert store.lookup_query("hsp70 chaperone mechanism") == ["doi:10.1/x"]
    assert store.lookup_query("  HSP70   chaperone, mechanism? ") == [
        "doi:10.1/x"
    ]
    assert store.lookup_query("something else") == []

    # re-record wins (append-only, last write per hash)
    store.record_query("hsp70 chaperone mechanism", ["doi:10.1/y"])
    reloaded = PaperStore(tmp_path / "cache")
    assert reloaded.lookup_query("hsp70 chaperone mechanism") == ["doi:10.1/y"]


def test_paper_store_missing_files_are_graceful(tmp_path) -> None:
    store = PaperStore(tmp_path / "fresh-cache")
    assert store.lookup_query("anything") == []
    assert store.lookup_by_keys(["doi:10.1/x", "pmid:9"]) == []
