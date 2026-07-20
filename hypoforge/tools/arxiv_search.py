"""Async wrapper around the blocking third-party arXiv API client."""

from __future__ import annotations

import asyncio
from typing import Any


def _clean_text(value: object) -> str:
    return " ".join(str(value or "").split())


def _result_to_dict(result: Any) -> dict:
    published = getattr(result, "published", None)
    authors = [
        _clean_text(getattr(author, "name", author))
        for author in (getattr(result, "authors", None) or [])
    ]
    categories = [
        _clean_text(item)
        for item in (getattr(result, "categories", None) or [])
        if _clean_text(item)
    ]
    get_short_id = getattr(result, "get_short_id", None)
    arxiv_id = _clean_text(get_short_id() if callable(get_short_id) else "")
    return {
        "arxiv_id": arxiv_id,
        "entry_url": _clean_text(getattr(result, "entry_id", "")),
        "pdf_url": _clean_text(getattr(result, "pdf_url", "")),
        "title": _clean_text(getattr(result, "title", "")),
        "abstract": _clean_text(getattr(result, "summary", "")),
        "authors": [author for author in authors if author],
        "year": getattr(published, "year", None),
        "doi": _clean_text(getattr(result, "doi", "")),
        "primary_category": _clean_text(
            getattr(result, "primary_category", "")
        ),
        "categories": categories,
        "journal": _clean_text(getattr(result, "journal_ref", "")),
        "comment": _clean_text(getattr(result, "comment", "")),
    }


def _search_arxiv_sync(query: str, limit: int) -> list[dict]:
    import arxiv

    client = arxiv.Client(
        page_size=min(limit, 100),
        delay_seconds=3.0,
        num_retries=3,
    )
    search = arxiv.Search(
        query=query,
        max_results=limit,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    return [_result_to_dict(item) for item in client.results(search)]


class ArxivTool:
    """Search real arXiv metadata without blocking the asyncio event loop."""

    tool_name = "arxiv_search"

    async def search_strict(self, query: str, limit: int = 20) -> list[dict]:
        clean_query = " ".join(str(query or "").split())
        if not clean_query:
            raise ValueError("query must not be blank")
        if limit <= 0:
            return []
        return await asyncio.to_thread(_search_arxiv_sync, clean_query, limit)

    async def search(self, query: str, limit: int = 20) -> list[dict]:
        return await self.search_strict(query, limit=limit)
