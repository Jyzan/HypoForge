from __future__ import annotations

import asyncio
from collections.abc import Sequence

from hypoforge.literature.models import (
    CoverageReport,
    PaperReadingResult,
    PaperRecord,
    ScoutNote,
    SearchQuery,
    SearchState,
)
from hypoforge.literature.protocols import (
    CoverageEvaluatorProtocol,
    LiteratureSourceProtocol,
    PaperDeduplicatorProtocol,
    PaperRankerProtocol,
    QueryPlannerProtocol,
    ReadingExtractionWorkflowProtocol,
    ScoutReaderProtocol,
)


class FakePlanner(QueryPlannerProtocol):
    def __init__(
        self,
        plans: Sequence[Sequence[SearchQuery]],
        error: Exception | None = None,
    ) -> None:
        self.plans = [list(plan) for plan in plans]
        self.error = error
        self.states: list[SearchState] = []

    async def plan(
        self,
        sub_question: str,
        key_entities: Sequence[str] = (),
        domains: Sequence[str] = (),
        question_type: str = "",
        state: SearchState | None = None,
    ) -> list[SearchQuery]:
        if self.error:
            raise self.error
        self.states.append((state or SearchState()).model_copy(deep=True))
        index = min(len(self.states) - 1, len(self.plans) - 1)
        return [query.model_copy(deep=True) for query in self.plans[index]]


class FakeSource(LiteratureSourceProtocol):
    def __init__(
        self,
        source_name: str,
        results: dict[str, Sequence[PaperRecord]] | None = None,
        errors: dict[str, Exception] | None = None,
        delays: dict[str, float] | None = None,
    ) -> None:
        self.source_name = source_name
        self.results = {key: list(value) for key, value in (results or {}).items()}
        self.errors = errors or {}
        self.delays = delays or {}
        self.calls: list[SearchQuery] = []

    async def search(self, query: SearchQuery, limit: int = 20) -> list[PaperRecord]:
        self.calls.append(query.model_copy(deep=True))
        if query.text in self.delays:
            await asyncio.sleep(self.delays[query.text])
        if query.text in self.errors:
            raise self.errors[query.text]
        return [paper.model_copy(deep=True) for paper in self.results.get(query.text, [])][
            :limit
        ]


class FakeDeduplicator(PaperDeduplicatorProtocol):
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error

    async def deduplicate(
        self,
        papers: Sequence[PaperRecord],
        existing_papers: Sequence[PaperRecord] = (),
    ) -> list[PaperRecord]:
        if self.error:
            raise self.error
        canonical = {paper.paper_id: paper for paper in existing_papers}
        unique: dict[str, PaperRecord] = {}
        for paper in papers:
            unique.setdefault(paper.paper_id, canonical.get(paper.paper_id, paper))
        return list(unique.values())


class FakeRanker(PaperRankerProtocol):
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error

    async def rank(
        self,
        sub_question: str,
        papers: Sequence[PaperRecord],
        limit: int,
    ) -> list[PaperRecord]:
        if self.error:
            raise self.error
        return list(papers)[:limit]


class FakeScoutReader(ScoutReaderProtocol):
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[list[str]] = []

    async def read(
        self,
        sub_question: str,
        papers: Sequence[PaperRecord],
    ) -> list[ScoutNote]:
        if self.error:
            raise self.error
        self.calls.append([paper.paper_id for paper in papers])
        return [
            ScoutNote(
                paper_id=paper.paper_id,
                key_terms=[f"term-{paper.paper_id}"],
                relevance_to_question=0.8,
            )
            for paper in papers
        ]


class FakeCoverageEvaluator(CoverageEvaluatorProtocol):
    def __init__(
        self,
        reports: Sequence[CoverageReport],
        error: Exception | None = None,
    ) -> None:
        self.reports = list(reports)
        self.error = error
        self.states: list[SearchState] = []

    async def evaluate(
        self,
        sub_question: str,
        papers: Sequence[PaperRecord],
        scout_notes: Sequence[ScoutNote],
        state: SearchState,
    ) -> CoverageReport:
        if self.error:
            raise self.error
        self.states.append(state.model_copy(deep=True))
        index = min(len(self.states) - 1, len(self.reports) - 1)
        return self.reports[index].model_copy(deep=True)


class FakeReadingWorkflow(ReadingExtractionWorkflowProtocol):
    def __init__(self, results: Sequence[PaperReadingResult]) -> None:
        self.results = list(results)
        self.calls: list[list[str]] = []

    async def run(
        self,
        sub_question: str,
        papers: Sequence[PaperRecord],
    ) -> list[PaperReadingResult]:
        self.calls.append([paper.paper_id for paper in papers])
        return [result.model_copy(deep=True) for result in self.results]
