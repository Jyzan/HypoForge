from __future__ import annotations

import pytest

from hypoforge.literature.models import FulltextStatus, PaperRecord
from hypoforge.literature.search.ranking import PaperRanker


def paper(paper_id: str, title: str, **kwargs: object) -> PaperRecord:
    return PaperRecord(
        paper_id=paper_id,
        title=title,
        sources=["test"],
        **kwargs,
    )


@pytest.mark.asyncio
async def test_relevant_paper_can_outrank_an_earlier_irrelevant_result() -> None:
    irrelevant = paper("irrelevant", "Unrelated ecology observations")
    relevant = paper(
        "relevant",
        "Hsp70 ATPase controls protein folding",
        abstract="Hsp70 regulates ATP-dependent protein folding.",
    )

    ranked = await PaperRanker(current_year=2026).rank(
        "Hsp70 protein folding", [irrelevant, relevant], limit=2
    )

    assert ranked[0].paper_id == "relevant"
    assert ranked[0].rank_scores["query_relevance"] > 0


@pytest.mark.asyncio
async def test_ranker_populates_all_scores_without_mutating_inputs() -> None:
    source = paper(
        "paper",
        "Protein folding",
        abstract="Mechanisms of protein folding.",
        authors=["Author"],
        year=2025,
        journal="Journal",
        doi="10.1/test",
        citation_count=12,
        is_open_access=True,
        fulltext_status=FulltextStatus.XML_AVAILABLE,
    )
    before = source.model_dump()

    ranked = await PaperRanker(current_year=2026).rank(
        "protein folding", [source], limit=1
    )

    assert source.model_dump() == before
    assert set(ranked[0].rank_scores) >= {
        "query_relevance",
        "retrieval_prior",
        "citation_impact",
        "recency",
        "metadata_quality",
        "access_quality",
        "total",
    }
    assert 0 <= ranked[0].rank_scores["total"] <= 1


@pytest.mark.asyncio
async def test_citation_impact_is_age_adjusted() -> None:
    older = paper("old", "Same topic", year=2006, citation_count=100)
    newer = paper("new", "Same topic", year=2025, citation_count=100)

    ranked = await PaperRanker(current_year=2026).rank(
        "same topic", [older, newer], limit=2
    )
    scores = {item.paper_id: item.rank_scores for item in ranked}

    assert scores["new"]["citation_impact"] > scores["old"]["citation_impact"]


@pytest.mark.asyncio
async def test_unknown_citation_metadata_is_not_treated_as_known_zero() -> None:
    unknown = paper(
        "unknown",
        "Same topic",
        citation_count=None,
        rank_scores={"source_relevance": 0.5},
    )
    known_zero = paper(
        "zero",
        "Same topic",
        citation_count=0,
        rank_scores={"source_relevance": 0.5},
    )

    ranked = await PaperRanker(current_year=2026).rank(
        "same topic", [unknown, known_zero], limit=2
    )
    scores = {item.paper_id: item.rank_scores for item in ranked}

    assert scores["unknown"]["total"] > scores["zero"]["total"]


@pytest.mark.asyncio
async def test_equal_scores_keep_input_order_and_limit() -> None:
    first = paper("z", "Identical", rank_scores={"source_relevance": 0.5})
    second = paper("a", "Identical", rank_scores={"source_relevance": 0.5})

    ranked = await PaperRanker(current_year=2026).rank(
        "absent query", [first, second], limit=1
    )

    assert [item.paper_id for item in ranked] == ["z"]


@pytest.mark.asyncio
async def test_ranker_supports_cjk_query_tokens_and_nonpositive_limit() -> None:
    item = paper("paper", "蛋白质折叠与衰老")
    ranker = PaperRanker(current_year=2026)

    ranked = await ranker.rank("蛋白质折叠", [item], limit=1)

    assert ranked[0].rank_scores["query_relevance"] > 0
    assert await ranker.rank("蛋白质折叠", [item], limit=0) == []
