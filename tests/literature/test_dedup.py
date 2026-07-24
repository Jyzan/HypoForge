from __future__ import annotations

import pytest

from hypoforge.literature.models import FulltextStatus, PaperRecord
from hypoforge.literature.search.dedup import PaperDeduplicator


def paper(paper_id: str, title: str, **kwargs: object) -> PaperRecord:
    return PaperRecord(
        paper_id=paper_id,
        title=title,
        sources=kwargs.pop("sources", ["test"]),
        **kwargs,
    )


@pytest.mark.asyncio
async def test_deduplicates_normalized_doi_and_merges_richer_metadata() -> None:
    first = paper(
        "PMID:1",
        "A canonical title",
        doi="10.1000/ABC",
        pmid="1",
        abstract="Short abstract.",
        authors=["First Author"],
        citation_count=4,
        sources=["pubmed"],
    )
    second = paper(
        "S2:2",
        "A title returned by another source",
        doi="https://doi.org/10.1000/abc",
        abstract="A substantially longer abstract with additional metadata.",
        authors=["First Author", "Second Author"],
        journal="Journal",
        citation_count=10,
        is_open_access=True,
        fulltext_status=FulltextStatus.PDF_AVAILABLE,
        sources=["semantic_scholar"],
    )

    result = await PaperDeduplicator().deduplicate([first, second])

    assert len(result) == 1
    merged = result[0]
    assert merged.paper_id == "PMID:1"
    assert merged.title == "A canonical title"
    assert merged.abstract == second.abstract
    assert merged.authors == second.authors
    assert merged.journal == "Journal"
    assert merged.citation_count == 10
    assert merged.sources == ["pubmed", "semantic_scholar"]
    assert merged.is_open_access is True
    assert merged.fulltext_status is FulltextStatus.PDF_AVAILABLE


@pytest.mark.asyncio
async def test_external_identifier_matches_the_corresponding_primary_identifier() -> None:
    from_field = paper("PMID:123", "Field record", pmid="123")
    from_external = paper(
        "S2:abc",
        "External record",
        external_ids={"PMID": "123"},
    )

    result = await PaperDeduplicator().deduplicate([from_field, from_external])

    assert [item.paper_id for item in result] == ["PMID:123"]


@pytest.mark.asyncio
async def test_normalized_and_conservatively_fuzzy_titles_are_deduplicated() -> None:
    exact_left = paper("one", "NAD+, Hsp70: regulation in ageing")
    exact_right = paper("two", "nad hsp70 regulation in ageing")
    fuzzy_left = paper(
        "three",
        "Molecular mechanisms regulating protein quality control during cellular ageing in mammalian tissues and organs",
        year=2025,
    )
    fuzzy_right = paper(
        "four",
        "Molecular mechanism regulating protein quality control during cellular ageing in mammalian tissues and organs",
        year=2024,
    )

    result = await PaperDeduplicator().deduplicate(
        [exact_left, exact_right, fuzzy_left, fuzzy_right]
    )

    assert [item.paper_id for item in result] == ["one", "three"]


@pytest.mark.asyncio
async def test_similar_titles_with_distant_years_are_not_fuzzy_merged() -> None:
    title = (
        "Molecular mechanisms regulating protein quality control during "
        "cellular ageing in mammalian tissues and organs"
    )
    first = paper("one", title, year=2000)
    second = paper("two", title.replace("mechanisms", "mechanism"), year=2025)

    result = await PaperDeduplicator().deduplicate([first, second])

    assert [item.paper_id for item in result] == ["one", "two"]


@pytest.mark.asyncio
async def test_existing_catalog_record_remains_canonical_and_order_is_stable() -> None:
    catalog = paper(
        "catalog-id",
        "Catalog title",
        doi="10.1/shared",
        sources=["catalog"],
    )
    incoming_duplicate = paper(
        "incoming-id",
        "Incoming title",
        doi="10.1/shared",
        abstract="New abstract",
        sources=["pubmed"],
    )
    new_paper = paper("new-id", "A new paper")

    result = await PaperDeduplicator().deduplicate(
        [incoming_duplicate, new_paper], existing_papers=[catalog]
    )

    assert [item.paper_id for item in result] == ["catalog-id", "new-id"]
    assert result[0].title == "Catalog title"
    assert result[0].abstract == "New abstract"


@pytest.mark.asyncio
async def test_deduplicator_does_not_mutate_input_models() -> None:
    first = paper("one", "Original", doi="10.1/x", sources=["one"])
    second = paper("two", "Duplicate", doi="10.1/x", sources=["two"])
    before = first.model_dump()

    await PaperDeduplicator().deduplicate([first, second])

    assert first.model_dump() == before
