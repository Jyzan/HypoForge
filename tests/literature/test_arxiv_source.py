from __future__ import annotations

import asyncio
import threading

import pytest

from hypoforge.literature.models import FulltextStatus, SearchQuery
from hypoforge.literature.sources.arxiv_source import (
    ArxivBackendError,
    ArxivSource,
    canonical_arxiv_id,
)
from hypoforge.tools import arxiv_search
from hypoforge.tools.arxiv_search import ArxivTool


def query(text: str = "graph neural networks") -> SearchQuery:
    return SearchQuery(
        query_id="q-arxiv",
        text=text,
        target_source="arxiv",
        purpose="recent technical evidence",
        relation_to_question="Searches recent technical preprints.",
    )


class FakeTool:
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows
        self.calls: list[tuple[str, int]] = []

    async def search_strict(self, text: str, limit: int = 20) -> list[dict]:
        self.calls.append((text, limit))
        return list(self.rows)


def test_canonical_arxiv_id_strips_urls_and_versions() -> None:
    assert canonical_arxiv_id("2401.12345v2") == "2401.12345"
    assert canonical_arxiv_id("https://arxiv.org/abs/hep-th/9901001v3") == (
        "hep-th/9901001"
    )


@pytest.mark.asyncio
async def test_arxiv_source_maps_versioned_result_to_canonical_record() -> None:
    source = ArxivSource(
        tool=FakeTool(
            [
                {
                    "arxiv_id": "2401.12345v2",
                    "entry_url": "https://arxiv.org/abs/2401.12345v2",
                    "pdf_url": "https://arxiv.org/pdf/2401.12345v2",
                    "title": "  A useful preprint  ",
                    "abstract": "Evidence.",
                    "authors": ["Ada Lovelace", "Alan Turing"],
                    "year": 2025,
                    "doi": "https://doi.org/10.1000/ABC",
                    "primary_category": "cs.AI",
                    "categories": ["cs.AI", "cs.LG"],
                }
            ]
        )
    )

    papers = await source.search(query(), limit=3)

    assert len(papers) == 1
    paper = papers[0]
    assert paper.paper_id == "ARXIV:2401.12345"
    assert paper.title == "A useful preprint"
    assert paper.external_ids == {
        "arxiv": "2401.12345v2",
        "entry_url": "https://arxiv.org/abs/2401.12345v2",
        "pdf_url": "https://arxiv.org/pdf/2401.12345v2",
        "primary_category": "cs.AI",
        "categories": "cs.AI,cs.LG",
    }
    assert paper.doi == "10.1000/abc"
    assert paper.sources == ["arxiv"]
    assert paper.publication_type == "preprint"
    assert paper.citation_count is None
    assert paper.is_open_access is True
    assert paper.fulltext_status is FulltextStatus.PDF_AVAILABLE


@pytest.mark.asyncio
async def test_arxiv_source_uses_abstract_status_without_pdf_and_skips_empty_rows() -> None:
    source = ArxivSource(
        tool=FakeTool(
            [
                {},
                {
                    "arxiv_id": "2402.00001",
                    "title": "Abstract only",
                    "abstract": "Useful abstract.",
                },
            ]
        )
    )

    papers = await source.search(query())

    assert [paper.paper_id for paper in papers] == ["ARXIV:2402.00001"]
    assert papers[0].fulltext_status is FulltextStatus.ABSTRACT_ONLY
    assert papers[0].is_open_access is None


@pytest.mark.asyncio
async def test_arxiv_source_propagates_backend_failure_with_provider_name() -> None:
    class FailingTool:
        async def search_strict(self, text: str, limit: int = 20) -> list[dict]:
            raise OSError("offline")

    with pytest.raises(ArxivBackendError, match="arxiv backend failed.*offline"):
        await ArxivSource(tool=FailingTool()).search(query())


@pytest.mark.asyncio
async def test_arxiv_source_propagates_cancellation() -> None:
    class CancelledTool:
        async def search_strict(self, text: str, limit: int = 20) -> list[dict]:
            raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await ArxivSource(tool=CancelledTool()).search(query())


@pytest.mark.asyncio
async def test_arxiv_tool_offloads_blocking_client(monkeypatch) -> None:
    entered = threading.Event()
    release = threading.Event()
    worker_threads: list[int] = []

    def blocking_search(text: str, limit: int) -> list[dict]:
        worker_threads.append(threading.get_ident())
        entered.set()
        release.wait(timeout=1)
        return []

    monkeypatch.setattr(arxiv_search, "_search_arxiv_sync", blocking_search)
    main_thread = threading.get_ident()
    try:
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                ArxivTool().search_strict("quantum", limit=2), timeout=0.02
            )
    finally:
        release.set()

    assert entered.is_set()
    assert worker_threads != [main_thread]


@pytest.mark.asyncio
async def test_arxiv_tool_rejects_blank_query_and_nonpositive_limit() -> None:
    tool = ArxivTool()
    with pytest.raises(ValueError, match="query must not be blank"):
        await tool.search_strict("   ")
    assert await tool.search_strict("quantum", limit=0) == []
