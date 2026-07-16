from __future__ import annotations

import pytest

from hypoforge.literature.models import FulltextStatus, QueryIntent, SearchQuery
from hypoforge.literature.sources.pubmed import PubMedLiteratureSource


def query() -> SearchQuery:
    return SearchQuery(
        query_id="q-1",
        text="Hippo AND YAP AND TAZ",
        intent=QueryIntent.CORE,
        target_source="pubmed",
        purpose="Find direct PubMed evidence",
        relation_to_question="Uses named pathway entities",
    )


@pytest.mark.asyncio
async def test_pubmed_source_maps_real_shaped_metadata() -> None:
    calls = []

    async def backend(text: str, limit: int):
        calls.append((text, limit))
        return [{
            "pmid": "123",
            "title": "Hippo signaling controls organ size",
            "abstract": "Hippo signaling regulates YAP activity.",
            "authors": ["Example A"],
            "year": 2024,
            "journal": "Example Journal",
            "doi": "https://doi.org/10.1/example",
        }]

    result = await PubMedLiteratureSource(backend=backend).search(query(), limit=5)

    assert calls == [("Hippo AND YAP AND TAZ", 5)]
    assert len(result) == 1
    assert result[0].paper_id == "PMID:123"
    assert result[0].pmid == "123"
    assert result[0].doi == "10.1/example"
    assert result[0].sources == ["pubmed"]
    assert result[0].fulltext_status is FulltextStatus.ABSTRACT_ONLY


@pytest.mark.asyncio
async def test_pubmed_source_propagates_backend_failure() -> None:
    async def backend(text: str, limit: int):
        raise OSError("network unavailable")

    with pytest.raises(OSError, match="network unavailable"):
        await PubMedLiteratureSource(backend=backend).search(query())


@pytest.mark.asyncio
async def test_pubmed_source_skips_only_unidentifiable_records() -> None:
    async def backend(text: str, limit: int):
        return [
            {"pmid": "", "doi": "", "title": ""},
            {"pmid": "", "doi": "10.2/doi-only", "title": "DOI paper"},
            {"pmid": "", "doi": "", "title": "Title only paper"},
        ]

    result = await PubMedLiteratureSource(backend=backend).search(query())

    assert len(result) == 2
    assert result[0].paper_id == "DOI:10.2/doi-only"
    assert result[1].paper_id.startswith("TITLE:")


@pytest.mark.asyncio
async def test_pubmed_source_uses_normalized_doi_for_stable_identity() -> None:
    async def backend(text: str, limit: int):
        return [
            {"doi": "https://doi.org/10.1000/ABC"},
            {"doi": "10.1000/abc"},
        ]

    result = await PubMedLiteratureSource(backend=backend).search(query())

    assert [paper.doi for paper in result] == ["10.1000/abc", "10.1000/abc"]
    assert [paper.paper_id for paper in result] == [
        "DOI:10.1000/abc",
        "DOI:10.1000/abc",
    ]
