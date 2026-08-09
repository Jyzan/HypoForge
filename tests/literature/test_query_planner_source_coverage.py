from __future__ import annotations

import pytest

from hypoforge.literature.models import SearchState
from hypoforge.literature.search.query_planner import QueryPlanner
from hypoforge.literature.search.search_tool import LiteratureSearchTool


class FakeClient:
    def __init__(self, response: dict) -> None:
        self.response = response

    async def structured_chat(self, **kwargs):
        return self.response


def response_for(source: str, text: str = "cancer immunotherapy") -> dict:
    return {
        "queries": [
            {
                "text": text,
                "tool": source,
                "purpose": "core_mechanism",
                "reasoning": "Direct evidence.",
            }
        ]
    }


@pytest.mark.asyncio
async def test_first_round_adds_every_missing_configured_source() -> None:
    planner = QueryPlanner(
        FakeClient(response_for("pubmed")),
        LiteratureSearchTool.TOOL_DEFINITIONS,
    )

    queries = await planner.plan(
        "How does cancer immunotherapy work?",
        state=SearchState(),
    )

    assert {item.target_source for item in queries} == {
        "pubmed",
        "semantic_scholar",
        "openalex",
        "arxiv",
    }
    added = [
        item for item in queries if item.purpose == "cross_source_coverage"
    ]
    assert {item.target_source for item in added} == {
        "semantic_scholar",
        "openalex",
        "arxiv",
    }
    assert all(
        item.text == "How does cancer immunotherapy work?" for item in added
    )
    assert len({item.query_id for item in queries}) == 4


@pytest.mark.asyncio
async def test_first_round_does_not_duplicate_model_selected_sources() -> None:
    planner = QueryPlanner(
        FakeClient(response_for("arxiv")),
        LiteratureSearchTool.TOOL_DEFINITIONS,
    )

    queries = await planner.plan("question", state=SearchState())

    assert [item.target_source for item in queries].count("arxiv") == 1
    assert len(queries) == 4


@pytest.mark.asyncio
async def test_first_round_coverage_precedes_repeated_source_queries() -> None:
    response = {
        "queries": [
            {
                "text": f"pubmed query {index}",
                "tool": "pubmed",
                "purpose": "core_mechanism",
                "reasoning": "Direct evidence.",
            }
            for index in range(4)
        ]
    }
    planner = QueryPlanner(
        FakeClient(response),
        LiteratureSearchTool.TOOL_DEFINITIONS,
    )

    queries = await planner.plan("question", state=SearchState())

    assert {item.target_source for item in queries[:4]} == {
        "pubmed",
        "semantic_scholar",
        "openalex",
        "arxiv",
    }


@pytest.mark.asyncio
async def test_later_round_does_not_force_all_sources() -> None:
    planner = QueryPlanner(
        FakeClient(response_for("arxiv", "targeted gap query")),
        LiteratureSearchTool.TOOL_DEFINITIONS,
    )
    state = SearchState(
        round_index=1,
        missing_topics={"review evidence"},
    )

    queries = await planner.plan("question", state=state)

    assert [item.target_source for item in queries] == ["arxiv"]
    assert all(item.purpose != "cross_source_coverage" for item in queries)


@pytest.mark.asyncio
async def test_unavailable_source_is_not_reintroduced_on_first_round() -> None:
    planner = QueryPlanner(
        FakeClient(response_for("pubmed")),
        LiteratureSearchTool.TOOL_DEFINITIONS,
    )
    state = SearchState(unavailable_sources={"semantic_scholar"})

    queries = await planner.plan("question", state=state)

    assert {item.target_source for item in queries} == {
        "pubmed",
        "openalex",
        "arxiv",
    }


@pytest.mark.asyncio
async def test_fallback_does_not_reintroduce_unavailable_source() -> None:
    class FailingClient:
        async def structured_chat(self, **kwargs):
            raise OSError("model unavailable")

    planner = QueryPlanner(
        FailingClient(),
        LiteratureSearchTool.TOOL_DEFINITIONS,
    )
    state = SearchState(
        round_index=1,
        unavailable_sources={"semantic_scholar"},
    )

    queries = await planner.plan("question", state=state)

    assert {item.target_source for item in queries} == {
        "pubmed",
        "openalex",
        "arxiv",
    }


@pytest.mark.asyncio
async def test_missing_core_entity_is_added_as_grouped_conjunct() -> None:
    planner = QueryPlanner(
        FakeClient(response_for("arxiv", "method testing")),
        LiteratureSearchTool.TOOL_DEFINITIONS,
    )

    queries = await planner.plan(
        "How are robot arms tested?",
        key_entities=["robot arm"],
        state=SearchState(round_index=1),
    )

    assert queries[0].text == '"robot arm" AND (method testing)'
    assert not queries[0].text.endswith("robot arm")


@pytest.mark.asyncio
async def test_core_entity_tokens_already_present_are_not_duplicated() -> None:
    planner = QueryPlanner(
        FakeClient(response_for("arxiv", "robot safety arm")),
        LiteratureSearchTool.TOOL_DEFINITIONS,
    )

    queries = await planner.plan(
        "How are robot arms tested?",
        key_entities=["robot arm"],
        state=SearchState(round_index=1),
    )

    assert queries[0].text == "robot safety arm"
