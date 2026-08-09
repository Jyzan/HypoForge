"""Contract tests for OpenAlex as an independent M2 source."""

from __future__ import annotations

import urllib.parse

import pytest

from hypoforge.literature.models import FulltextStatus, SearchQuery
from hypoforge.literature.search.search_tool import LiteratureSearchTool
from hypoforge.literature.sources.openalex_source import (
    OpenAlexBackendError,
    OpenAlexSource,
)
from hypoforge.tools import semantic_scholar as ss


def make_query(text: str = "protein misfolding") -> SearchQuery:
    return SearchQuery(
        query_id="q1",
        text=text,
        target_source="openalex",
        purpose="test",
        relation_to_question="test",
    )


@pytest.mark.asyncio
async def test_openalex_source_maps_backend_rows_to_records() -> None:
    async def fake_backend(query: str, limit: int):
        assert query == "protein misfolding"
        assert limit == 5
        return [
            {
                "paper_id": "https://openalex.org/W123",
                "source": "openalex",
                "title": "OpenAlex study",
                "abstract": "Real abstract.",
                "doi": "https://doi.org/10.5000/ABC",
                "year": 2021,
            },
            {"doi": "https://doi.org/10.6000/xyz", "title": "DOI-only work"},
            {},
        ]

    papers = await OpenAlexSource(backend=fake_backend).search(
        make_query(), limit=5
    )

    assert [paper.paper_id for paper in papers] == [
        "OA:W123",
        "DOI:10.6000/xyz",
    ]
    assert all(paper.sources == ["openalex"] for paper in papers)
    assert papers[0].fulltext_status is FulltextStatus.ABSTRACT_ONLY


@pytest.mark.asyncio
async def test_openalex_source_propagates_backend_failure() -> None:
    async def failing_backend(query: str, limit: int):
        raise OSError("provider unavailable")

    with pytest.raises(OpenAlexBackendError, match="openalex backend failed"):
        await OpenAlexSource(backend=failing_backend).search(make_query())


@pytest.mark.asyncio
async def test_search_tool_registers_and_dispatches_openalex() -> None:
    calls: list[tuple[str, int]] = []

    async def fake_backend(query: str, limit: int):
        calls.append((query, limit))
        return [{"paper_id": "W9", "source": "openalex", "title": "T"}]

    tool = LiteratureSearchTool(
        openalex_source=OpenAlexSource(backend=fake_backend),
        enabled_sources=["openalex", "pubmed"],
    )

    assert {item["name"] for item in tool.tool_definitions} == {
        "openalex",
        "pubmed",
    }
    assert {source.source_name for source in tool.as_source_list()} == {
        "openalex",
        "pubmed",
    }
    papers = await tool.search("robot grasping", backend="openalex", limit=4)
    assert [paper.paper_id for paper in papers] == ["OA:W9"]
    assert calls == [("robot grasping", 4)]


def test_search_tool_default_definitions_cover_four_engines() -> None:
    assert [item["name"] for item in LiteratureSearchTool.TOOL_DEFINITIONS] == [
        "pubmed",
        "semantic_scholar",
        "openalex",
        "arxiv",
    ]


def test_openalex_transport_normalizes_boolean_query_and_forwards_credentials(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_http_get(url: str, *, s2_api_key: str = "", deadline=None) -> dict:
        captured["url"] = url
        captured["deadline"] = deadline
        return {"results": [], "meta": {"count": 0}}

    monkeypatch.setattr(ss, "_http_get_json", fake_http_get)

    ss._oa_search(
        '"视觉模型" AND ("relative depth" OR "ordinal depth" prediction) [tiab]',
        3,
        api_key="oa-test-key",
        mailto="team@example.org",
    )

    query = urllib.parse.parse_qs(
        urllib.parse.urlsplit(str(captured["url"])).query
    )
    assert query["search"] == ["视觉模型 relative depth ordinal depth prediction"]
    assert query["api_key"] == ["oa-test-key"]
    assert query["mailto"] == ["team@example.org"]


@pytest.mark.asyncio
async def test_openalex_strict_search_forwards_explicit_credentials(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_search(
        query: str,
        limit: int,
        api_key: str = "",
        mailto: str = "",
        deadline=None,
    ):
        captured.update(
            query=query,
            limit=limit,
            api_key=api_key,
            mailto=mailto,
            deadline=deadline,
        )
        return []

    monkeypatch.setattr(ss, "_oa_search", fake_search)

    await ss.search_openalex_strict(
        "robot grasping",
        limit=4,
        api_key="run-key",
        mailto="run@example.org",
    )

    assert captured["query"] == "robot grasping"
    assert captured["limit"] == 4
    assert captured["api_key"] == "run-key"
    assert captured["mailto"] == "run@example.org"


def test_s2_fallback_gives_openalex_a_fresh_deadline(monkeypatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(ss.time, "monotonic", lambda: 100.0)

    def fake_openalex(
        query: str,
        limit: int,
        api_key: str = "",
        mailto: str = "",
        deadline=None,
    ):
        captured["deadline"] = deadline
        return []

    monkeypatch.setattr(ss, "_oa_search", fake_openalex)

    ss._s2_with_oa_fallback(
        "robot grasping",
        3,
        "s2-key",
        "oa-key",
        s2_error=TimeoutError("S2 expired"),
        deadline=99.0,
    )

    assert float(captured["deadline"]) > 100.0


def test_zero_result_s2_is_not_repeated_before_openalex(monkeypatch) -> None:
    calls = {"s2": 0, "openalex": 0}

    def fake_s2(query: str, limit: int, api_key: str = "", deadline=None):
        calls["s2"] += 1
        return []

    def fake_openalex(
        query: str,
        limit: int,
        api_key: str = "",
        mailto: str = "",
        deadline=None,
    ):
        calls["openalex"] += 1
        return []

    monkeypatch.setattr(ss, "_BACKEND", "semantic_scholar")
    monkeypatch.setattr(ss, "_S2_API_KEY", "s2-key")
    monkeypatch.setattr(ss, "_s2_circuit_open", lambda: False)
    monkeypatch.setattr(ss, "_s2_search", fake_s2)
    monkeypatch.setattr(ss, "_oa_search", fake_openalex)

    ss._search("robot grasping", 3)

    assert calls == {"s2": 1, "openalex": 1}
