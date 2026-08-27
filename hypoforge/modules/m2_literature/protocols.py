"""Asynchronous Tool contracts for the agentic M2 literature pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence

from .models import (
    CoverageReport,
    DocumentChunk,
    DocumentRecord,
    EvidenceChunk,
    PaperReadingResult,
    PaperRecord,
    ScoutNote,
    SearchQuery,
    SearchRunResult,
    SearchState,
)


class QueryPlannerProtocol(ABC):
    tool_name = "query_planner"

    @abstractmethod
    async def plan(
        self,
        sub_question: str,
        key_entities: Sequence[str] = (),
        domains: Sequence[str] = (),
        question_type: str = "",
        state: SearchState | None = None,
        supplement_entities: Sequence[str] = (),
        focus_entities: Sequence[str] = (),
    ) -> List[SearchQuery]:
        """Create source-aware queries with explicit search intents.

        ``supplement_entities`` carries optional M2-local search concepts
        generated for this sub-question; they must only influence query
        construction and never leave the M2 search data flow.

        ``focus_entities`` carries the current round's mandatory concepts
        assigned by the entity-group round strategy; queries for this round
        must be built around all of them.  Planners that predate this
        parameter may ignore it.
        """
        ...


class LiteratureSourceProtocol(ABC):
    source_name: str

    @abstractmethod
    async def search(
        self,
        query: SearchQuery,
        limit: int = 20,
    ) -> List[PaperRecord]:
        """Search one literature source and return normalized paper metadata."""
        ...


class PaperDeduplicatorProtocol(ABC):
    tool_name = "paper_deduplicator"

    @abstractmethod
    async def deduplicate(
        self,
        papers: Sequence[PaperRecord],
        existing_papers: Sequence[PaperRecord] = (),
    ) -> List[PaperRecord]:
        """Return canonical records for incoming hits, reusing catalog matches.

        ``existing_papers`` is a lookup catalog, not an instruction to return
        every catalog record.  When an incoming hit matches that catalog, the
        returned item should be the canonical existing record.
        """
        ...


class PaperRankerProtocol(ABC):
    tool_name = "paper_ranker"

    @abstractmethod
    async def rank(
        self,
        sub_question: str,
        papers: Sequence[PaperRecord],
        limit: int,
    ) -> List[PaperRecord]:
        """Rank papers without changing their stable identity fields."""
        ...


class ScoutReaderProtocol(ABC):
    tool_name = "scout_reader"

    @abstractmethod
    async def read(
        self,
        sub_question: str,
        papers: Sequence[PaperRecord],
    ) -> List[ScoutNote]:
        """Perform lightweight title/abstract/citation reading for search refinement."""
        ...


class CoverageEvaluatorProtocol(ABC):
    tool_name = "coverage_evaluator"

    @abstractmethod
    async def evaluate(
        self,
        sub_question: str,
        papers: Sequence[PaperRecord],
        scout_notes: Sequence[ScoutNote],
        state: SearchState,
    ) -> CoverageReport:
        """Report evidence coverage and explicit gaps for the next search round."""
        ...


class FulltextResolverProtocol(ABC):
    tool_name = "fulltext_resolver"

    @abstractmethod
    async def resolve(self, paper: PaperRecord) -> DocumentRecord:
        """Resolve the best legally available content representation for a paper."""
        ...


class DocumentParserProtocol(ABC):
    tool_name = "document_parser"

    @abstractmethod
    async def parse(self, document: DocumentRecord) -> List[DocumentChunk]:
        """Parse a stored document into source-addressable, section-aware chunks."""
        ...


class EvidenceRetrieverProtocol(ABC):
    tool_name = "evidence_retriever"

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        paper_ids: Sequence[str],
        top_k: int = 10,
    ) -> List[EvidenceChunk]:
        """Retrieve a small set of attributable evidence passages."""
        ...


class PaperReaderProtocol(ABC):
    tool_name = "paper_reader"

    @abstractmethod
    async def read(
        self,
        sub_question: str,
        paper: PaperRecord,
        evidence: Sequence[EvidenceChunk],
    ) -> PaperReadingResult:
        """Summarize one paper and extract evidence-linked knowledge entries."""
        ...


class ReadingExtractionWorkflowProtocol(ABC):
    tool_name = "reading_extraction_workflow"

    @abstractmethod
    async def run(
        self,
        sub_question: str,
        papers: Sequence[PaperRecord],
        search_context: SearchRunResult | None = None,
    ) -> List[PaperReadingResult]:
        """Read Final-K papers and return evidence-linked paper results."""
        ...
