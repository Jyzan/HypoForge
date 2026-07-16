from __future__ import annotations

import pytest

from hypoforge.tools import pubmed_search


@pytest.mark.asyncio
async def test_strict_pubmed_search_fetches_metadata_in_a_thread(monkeypatch) -> None:
    monkeypatch.setattr(pubmed_search, "_esearch", lambda query, limit: ["123"])
    monkeypatch.setattr(
        pubmed_search,
        "_efetch_batch",
        lambda pmids: [{"pmid": "123", "title": "A paper"}],
    )

    result = await pubmed_search.search_pubmed_strict("Hippo YAP", limit=3)

    assert result == [{"pmid": "123", "title": "A paper"}]


@pytest.mark.asyncio
async def test_strict_pubmed_search_propagates_network_failure(monkeypatch) -> None:
    def fail(query: str, limit: int):
        raise OSError("network unavailable")

    monkeypatch.setattr(pubmed_search, "_esearch", fail)

    with pytest.raises(OSError, match="network unavailable"):
        await pubmed_search.search_pubmed_strict("Hippo YAP")


@pytest.mark.asyncio
async def test_legacy_pubmed_tool_still_returns_empty_on_failure(monkeypatch) -> None:
    def fail(query: str, limit: int):
        raise OSError("network unavailable")

    monkeypatch.setattr(pubmed_search, "_esearch", fail)

    assert await pubmed_search.PubMedTool().search("Hippo YAP") == []
