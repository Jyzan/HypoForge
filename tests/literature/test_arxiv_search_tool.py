from __future__ import annotations

import pytest

from hypoforge.literature.models import FulltextStatus, PaperRecord, SearchQuery
from hypoforge.literature.models import SearchState
from hypoforge.literature.search.query_planner import QueryPlanner
from hypoforge.literature.search.search_tool import LiteratureSearchTool


class FakeSource:
    def __init__(self, source_name: str) -> None:
        self.source_name = source_name
        self.calls = []

    async def search(self, query, limit: int = 20):
        self.calls.append((query, limit))
        return [
            PaperRecord(
                paper_id=f"{self.source_name.upper()}:1",
                title=f"{self.source_name} result",
                sources=[self.source_name],
                fulltext_status=FulltextStatus.UNKNOWN,
            )
        ]


class FakePlannerClient:
    def __init__(self, response: dict) -> None:
        self.response = response
        self.calls: list[dict] = []

    async def structured_chat(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def make_tool() -> tuple[LiteratureSearchTool, FakeSource, FakeSource, FakeSource]:
    pubmed = FakeSource("pubmed")
    academic = FakeSource("semantic_scholar")
    arxiv = FakeSource("arxiv")
    return (
        LiteratureSearchTool(
            pubmed=pubmed,
            academic=academic,
            arxiv_source=arxiv,
        ),
        pubmed,
        academic,
        arxiv,
    )


def test_search_tool_exposes_three_independent_sources() -> None:
    tool, _, _, _ = make_tool()

    assert [item["name"] for item in tool.tool_definitions] == [
        "pubmed",
        "semantic_scholar",
        "arxiv",
    ]
    assert [item.source_name for item in tool.as_source_list()] == [
        "pubmed",
        "semantic_scholar",
        "arxiv",
    ]
    assert tool.is_valid_backend("arxiv") is True


@pytest.mark.asyncio
async def test_direct_search_dispatches_arxiv_query() -> None:
    tool, pubmed, academic, arxiv = make_tool()

    records = await tool.search(
        "graph neural networks", backend="arxiv", limit=4
    )

    assert [record.paper_id for record in records] == ["ARXIV:1"]
    assert pubmed.calls == []
    assert academic.calls == []
    assert len(arxiv.calls) == 1
    search_query, limit = arxiv.calls[0]
    assert search_query.text == "graph neural networks"
    assert search_query.target_source == "arxiv"
    assert limit == 4


@pytest.mark.asyncio
async def test_multi_search_keeps_arxiv_results() -> None:
    tool, _, _, _ = make_tool()
    query = SearchQuery(
        query_id="q-arxiv",
        text="quantum error correction",
        target_source="arxiv",
        purpose="technical evidence",
        relation_to_question="Direct evidence.",
    )

    output = await tool.search_multi([query], limit=2)

    assert [paper.paper_id for paper in output["q-arxiv"]] == ["ARXIV:1"]


@pytest.mark.asyncio
async def test_query_planner_sanitizes_pubmed_tags_for_arxiv() -> None:
    client = FakePlannerClient(
        {
            "queries": [
                {
                    "text": '"interpretability"[tiab] AND sparse autoencoder[MeSH Terms]',
                    "tool": "arxiv",
                    "purpose": "recent_research",
                    "reasoning": "Recent technical preprints.",
                }
            ]
        }
    )
    planner = QueryPlanner(client, LiteratureSearchTool.TOOL_DEFINITIONS)

    queries = await planner.plan("interpretability", state=SearchState())

    assert queries[0].target_source == "arxiv"
    assert queries[0].text == '"interpretability" AND sparse autoencoder'


@pytest.mark.asyncio
async def test_query_planner_prompt_explicitly_teaches_arxiv_selection() -> None:
    client = FakePlannerClient(
        {
            "queries": [
                {
                    "text": "sparse autoencoder interpretability",
                    "tool": "arxiv",
                    "purpose": "recent_research",
                    "reasoning": "Recent technical preprints.",
                }
            ]
        }
    )
    planner = QueryPlanner(client, LiteratureSearchTool.TOOL_DEFINITIONS)

    await planner.plan("interpretability", state=SearchState())

    call = client.calls[0]
    assert "### For arXiv queries" in call["system_prompt"]
    assert "one of the available search tools" in call["user_prompt"]
