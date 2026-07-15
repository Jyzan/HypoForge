from __future__ import annotations

import inspect

import pytest

from hypoforge.literature.models import PaperRecord, QueryIntent, SearchQuery
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
    ]

    assert all(inspect.isabstract(protocol) for protocol in protocol_types)
