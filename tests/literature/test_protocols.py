from __future__ import annotations

import inspect

import pytest

from hypoforge.literature.models import (
    PaperReadingResult,
    PaperRecord,
    QueryIntent,
    SearchQuery,
    SearchState,
)
from hypoforge.literature.protocols import (
    CoverageEvaluatorProtocol,
    DocumentParserProtocol,
    EvidenceRetrieverProtocol,
    FulltextResolverProtocol,
    LiteratureSourceProtocol,
    PaperDeduplicatorProtocol,
    PaperRankerProtocol,
    PaperReaderProtocol,
    QueryPlannerProtocol,
    ReadingExtractionWorkflowProtocol,
    ScoutReaderProtocol,
)


@pytest.mark.asyncio
async def test_literature_source_can_be_implemented_without_framework_coupling() -> None:
    class FakeSource(LiteratureSourceProtocol):
        source_name = "fake"

        async def search(self, query: SearchQuery, limit: int = 20) -> list[PaperRecord]:
            return [
                PaperRecord(
                    paper_id="paper-1",
                    title=f"Result for {query.text}",
                    sources=[self.source_name],
                )
            ][:limit]

    query = SearchQuery(
        query_id="q-1",
        text="Hsp70 proteostasis",
        intent=QueryIntent.CORE,
        target_source="fake",
        purpose="Find evidence",
        relation_to_question="Direct",
    )

    results = await FakeSource().search(query, limit=1)

    assert results[0].sources == ["fake"]


def test_incomplete_literature_source_cannot_be_instantiated() -> None:
    class MissingSearch(LiteratureSourceProtocol):
        source_name = "missing"

    with pytest.raises(TypeError):
        MissingSearch()


def test_all_public_tool_protocols_are_abstract_contracts() -> None:
    protocol_types = [
        QueryPlannerProtocol,
        LiteratureSourceProtocol,
        PaperDeduplicatorProtocol,
        PaperRankerProtocol,
        ScoutReaderProtocol,
        CoverageEvaluatorProtocol,
        FulltextResolverProtocol,
        DocumentParserProtocol,
        EvidenceRetrieverProtocol,
        PaperReaderProtocol,
        ReadingExtractionWorkflowProtocol,
    ]

    assert all(inspect.isabstract(protocol) for protocol in protocol_types)


@pytest.mark.asyncio
async def test_query_planner_receives_iterative_search_state() -> None:
    class StatefulPlanner(QueryPlannerProtocol):
        async def plan(
            self,
            sub_question: str,
            key_entities=(),
            domains=(),
            question_type: str = "",
            state: SearchState | None = None,
        ) -> list[SearchQuery]:
            assert state is not None
            return []

    assert await StatefulPlanner().plan("question", state=SearchState()) == []


@pytest.mark.asyncio
async def test_reading_workflow_can_be_implemented_as_a_public_contract() -> None:
    class FakeReadingWorkflow(ReadingExtractionWorkflowProtocol):
        async def run(
            self,
            sub_question: str,
            papers: list[PaperRecord],
        ) -> list[PaperReadingResult]:
            return [PaperReadingResult(paper_id=paper.paper_id) for paper in papers]

    paper = PaperRecord(paper_id="paper-1", title="Paper", sources=["fake"])
    results = await FakeReadingWorkflow().run("question", [paper])

    assert results[0].paper_id == "paper-1"
