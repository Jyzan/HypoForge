from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..tools.qwen_client import QwenClient
from .adapter import AgenticM2Adapter
from .models import SearchBudget
from .reading import (
    ArxivPDFFetchBackend,
    ArxivPDFResolver,
    BioCDocumentParser,
    FetchBackend,
    FullTextReadingWorkflow,
    HybridEvidenceRetriever,
    InMemoryChunkStore,
    PDFDocumentParser,
    PMCFulltextResolver,
    QwenPaperReader,
    RoutingDocumentParser,
    RoutingFulltextResolver,
)
from .search import (
    CoverageEvaluator,
    IterativeSearchAgent,
    PaperDeduplicator,
    PaperRanker,
    ScoutReader,
)
from .search.query_planner import QueryPlanner
from .search.search_tool import LiteratureSearchTool


def build_integrated_search_adapter(
    *,
    client: QwenClient,
    search_tool: LiteratureSearchTool | None = None,
    final_k: int = 5,
    per_query_limit: int | None = None,
    enabled_sources: Sequence[str] | None = None,
    semantic_scholar_api_key: str = "",
    source_timeout_seconds: float = 30.0,
    budget: SearchBudget | Mapping[str, Any] | None = None,
    reading_cache_dir: str | Path = ".cache/hypoforge/literature/documents",
    pmc_backend: FetchBackend | None = None,
    arxiv_pdf_backend: ArxivPDFFetchBackend | None = None,
    arxiv_max_pdf_bytes: int = 52_428_800,
    arxiv_download_timeout_seconds: float = 180.0,
    resolver_timeout_seconds: float = 210.0,
    reader_timeout_seconds: float = 120.0,
    reading_workflow_timeout_seconds: float = 600.0,
) -> AgenticM2Adapter:
    if final_k <= 0:
        raise ValueError("final_k must be positive")
    if source_timeout_seconds <= 0:
        raise ValueError("source_timeout_seconds must be positive")
    if per_query_limit is not None and per_query_limit <= 0:
        raise ValueError("per_query_limit must be positive")
    if arxiv_max_pdf_bytes <= 0:
        raise ValueError("arxiv_max_pdf_bytes must be positive")
    if arxiv_download_timeout_seconds <= 0 or resolver_timeout_seconds <= 0:
        raise ValueError("reading timeouts must be positive")

    tool = search_tool or LiteratureSearchTool(
        enabled_sources=enabled_sources,
        semantic_scholar_api_key=semantic_scholar_api_key,
    )
    if isinstance(budget, Mapping):
        resolved_budget = SearchBudget.model_validate(dict(budget))
    else:
        resolved_budget = budget or SearchBudget(
            max_rounds=3,
            max_queries=12,
            max_papers=max(100, final_k),
        )
    planner = QueryPlanner(
        client=client,
        tool_definitions=tool.tool_definitions,
        max_rounds=resolved_budget.max_rounds,
        strict=True,
    )
    agent = IterativeSearchAgent(
        query_planner=planner,
        sources=tool.as_source_list(),
        deduplicator=PaperDeduplicator(),
        ranker=PaperRanker(),
        scout_reader=ScoutReader(client),
        # Coverage must be satisfied by the same leading papers that the
        # agent will hand to the reading workflow, not by discarded tail
        # candidates. This remains internal wiring; Protocol signatures stay
        # unchanged.
        coverage_evaluator=CoverageEvaluator(client, selection_limit=final_k),
        final_k=final_k,
        candidate_limit=max(20, final_k),
        per_query_limit=per_query_limit or final_k,
        source_timeout_seconds=source_timeout_seconds,
    )
    store = InMemoryChunkStore()
    pmc_resolver = PMCFulltextResolver(
        cache_dir=reading_cache_dir,
        timeout_seconds=source_timeout_seconds,
        backend=pmc_backend,
    )
    arxiv_resolver = ArxivPDFResolver(
        cache_dir=reading_cache_dir,
        timeout_seconds=source_timeout_seconds,
        max_pdf_bytes=arxiv_max_pdf_bytes,
        download_timeout_seconds=arxiv_download_timeout_seconds,
        backend=arxiv_pdf_backend,
    )
    reading_workflow = FullTextReadingWorkflow(
        resolver=RoutingFulltextResolver(
            pmc_resolver,
            arxiv_resolver,
        ),
        parser=RoutingDocumentParser(
            BioCDocumentParser(),
            PDFDocumentParser(),
        ),
        retriever=HybridEvidenceRetriever(store),
        reader=QwenPaperReader(client),
        store=store,
        resolver_timeout_seconds=resolver_timeout_seconds,
        reader_timeout_seconds=reader_timeout_seconds,
        workflow_timeout_seconds=reading_workflow_timeout_seconds,
    )
    return AgenticM2Adapter(
        search_agent=agent,
        reading_workflow=reading_workflow,
        budget=resolved_budget,
    )
