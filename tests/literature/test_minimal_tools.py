from __future__ import annotations

import pytest

from hypoforge.literature.minimal import (
    AbstractScoutReader,
    ExactPaperDeduplicator,
    MetadataPaperRanker,
    RuleBasedQueryPlanner,
    SingleSourceCoverageEvaluator,
)
from hypoforge.literature.models import (
    CoverageReport,
    EvidenceBucket,
    PaperRecord,
    SearchState,
)


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
