from __future__ import annotations

import pytest

from hypoforge.literature.minimal import (
    AbstractReadingWorkflow,
    AbstractScoutReader,
    ExactPaperDeduplicator,
    MetadataPaperRanker,
    RuleBasedQueryPlanner,
    SingleSourceCoverageEvaluator,
    build_minimal_pubmed_adapter,
)
from hypoforge.literature.models import (
    CoverageReport,
    EvidenceBucket,
    PaperRecord,
    SearchState,
)
from hypoforge.state import PipelineState


def paper(paper_id: str, *, title: str, abstract: str = "", year: int | None = None,
          pmid: str = "", doi: str = "") -> PaperRecord:
    return PaperRecord(
        paper_id=paper_id,
        title=title,
        abstract=abstract,
        year=year,
        pmid=pmid,
        doi=doi,
        sources=["pubmed"],
    )


@pytest.mark.asyncio
async def test_rule_planner_builds_one_entity_query_then_stops() -> None:
    planner = RuleBasedQueryPlanner()
    first = await planner.plan(
        "Hippo–YAP/TAZ如何限制器官大小？",
        key_entities=["Hippo", "YAP", "TAZ"],
        state=SearchState(),
    )
    second = await planner.plan(
        "Hippo–YAP/TAZ如何限制器官大小？",
        state=SearchState(queries_used=first),
    )

    assert [item.text for item in first] == ["Hippo AND YAP AND TAZ"]
    assert first[0].target_source == "pubmed"
    assert second == []


@pytest.mark.asyncio
async def test_exact_deduplicator_reuses_catalog_by_pmid_doi_and_title() -> None:
    canonical = paper("catalog", title="Hippo Signaling", pmid="123")
    incoming = [
        paper("new-pmid", title="Other", pmid="123"),
        paper("new-doi", title="DOI copy", doi="10.1/same"),
        paper("new-doi-2", title="Different", doi="10.1/same"),
        paper("new-title", title="  Hippo signaling!  "),
    ]

    result = await ExactPaperDeduplicator().deduplicate(
        incoming,
        existing_papers=[canonical],
    )

    assert result[0] is canonical
    assert [item.paper_id for item in result] == ["catalog", "new-doi"]


@pytest.mark.asyncio
async def test_metadata_ranker_prefers_abstract_then_recent_year() -> None:
    ranked = await MetadataPaperRanker().rank(
        "question",
        [
            paper("old", title="Old", abstract="Evidence.", year=2010),
            paper("none", title="No abstract", year=2025),
            paper("new", title="New", abstract="Evidence.", year=2024),
        ],
        limit=2,
    )

    assert [item.paper_id for item in ranked] == ["new", "old"]
    assert ranked[0].rank_scores["has_abstract"] == 1.0


@pytest.mark.asyncio
async def test_scout_and_coverage_return_bounded_deterministic_output() -> None:
    papers = [paper(
        "p1",
        title="Hippo YAP signaling",
        abstract="TAZ controls mechanotransduction and organ size.",
    )]
    notes = await AbstractScoutReader().read("question", papers)
    report = await SingleSourceCoverageEvaluator().evaluate(
        "question", papers, notes, SearchState()
    )

    assert notes[0].paper_id == "p1"
    assert {"Hippo", "YAP", "TAZ"}.issubset(set(notes[0].key_terms))
    assert report.sufficient is True
    assert EvidenceBucket.SUPPORTING in report.covered_buckets


@pytest.mark.asyncio
async def test_coverage_is_insufficient_without_papers() -> None:
    report = await SingleSourceCoverageEvaluator().evaluate(
        "question", [], [], SearchState()
    )
    assert report == CoverageReport(
        missing_buckets={EvidenceBucket.SUPPORTING},
        missing_topics=["PubMed evidence"],
        sufficient=False,
        rationale="PubMed returned no valid candidate papers.",
    )


@pytest.mark.asyncio
async def test_abstract_reader_links_one_real_abstract_sentence() -> None:
    source = paper(
        "PMID:123",
        title="Hippo signaling",
        abstract="Hippo signaling restrains YAP activity. A second sentence.",
        pmid="123",
    )

    result = (await AbstractReadingWorkflow().run("question", [source]))[0]

    assert result.degraded_to_abstract is True
    assert result.evidence[0].quote == "Hippo signaling restrains YAP activity."
    assert result.knowledge_entries[0].evidence_ids == [result.evidence[0].evidence_id]
    assert "PubMed abstract reports" in result.knowledge_entries[0].content


@pytest.mark.asyncio
async def test_abstract_reader_does_not_invent_content_when_abstract_missing() -> None:
    source = paper("PMID:123", title="No abstract", pmid="123")

    result = (await AbstractReadingWorkflow().run("question", [source]))[0]

    assert result.degraded_to_abstract is True
    assert result.evidence == []
    assert result.knowledge_entries == []
    assert result.errors == ["PubMed abstract unavailable"]


@pytest.mark.asyncio
async def test_factory_runs_m2_directly_with_real_shaped_backend() -> None:
    async def backend(text: str, limit: int):
        return [{
            "pmid": "123",
            "title": "Hippo signaling",
            "abstract": "Hippo signaling restrains YAP activity.",
            "year": 2024,
        }]

    module = build_minimal_pubmed_adapter(backend=backend, final_k=3)

    output = await module(PipelineState(input_question="Hippo YAP TAZ organ size"))

    result = output["literature_results"][0]
    assert result.sub_question == "Hippo YAP TAZ organ size"
    assert result.papers_retrieved == 1
    assert result.knowledge_entries[0].source_paper_id == "PMID:123"


@pytest.mark.asyncio
async def test_factory_returns_empty_result_when_pubmed_returns_none() -> None:
    async def backend(text: str, limit: int):
        return []

    module = build_minimal_pubmed_adapter(backend=backend)

    output = await module(PipelineState(input_question="no matching topic"))

    assert output["literature_results"][0].papers_retrieved == 0
    assert output["literature_results"][0].knowledge_entries == []
    assert output["m2_knowledge_export"].runs[0].papers == []


@pytest.mark.asyncio
async def test_factory_surfaces_pubmed_network_failure() -> None:
    async def backend(text: str, limit: int):
        raise OSError("network unavailable")

    module = build_minimal_pubmed_adapter(backend=backend)

    with pytest.raises(RuntimeError, match="network unavailable"):
        await module(PipelineState(input_question="question"))


def test_factory_passes_source_timeout_to_real_pubmed_source() -> None:
    adapter = build_minimal_pubmed_adapter(source_timeout_seconds=0.75)

    source = adapter.search_agent.sources["pubmed"]
    assert source.timeout_seconds == 0.75
