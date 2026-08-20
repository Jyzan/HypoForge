from __future__ import annotations

from collections.abc import Mapping, Sequence
import os
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
    EnrichingFulltextResolver,
    OpenAccessEnricher,
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
from .search.domain_routing import DomainRoutedQueryPlanner
from .search.search_tool import LiteratureSearchTool


def build_integrated_search_adapter(
    *,
    client: QwenClient,
    query_client: QwenClient | None = None,
    search_tool: LiteratureSearchTool | None = None,
    final_k: int = 5,
    per_query_limit: int | None = None,
    enabled_sources: Sequence[str] | None = None,
    semantic_scholar_api_key: str = "",
    openalex_api_key: str = "",
    openalex_mailto: str = "",
    serper_api_key: str = "",
    ads_api_token: str = "",
    crossref_mailto: str = "",
    unpaywall_email: str = "",
    domain_routing_enabled: bool = False,
    access_enrichment_timeout_seconds: float = 12.0,
    fulltext_target_per_subquestion: int = 3,
    fulltext_backfill_max_attempts: int = 8,
    zero_result_relaxation: bool = True,
    source_timeout_seconds: float = 30.0,
    scout_timeout_seconds: float = 90.0,
    scout_candidate_limit: int = 24,
    budget: SearchBudget | Mapping[str, Any] | None = None,
    reading_cache_dir: str | Path = ".cache/hypoforge/literature/documents",
    pmc_backend: FetchBackend | None = None,
    arxiv_pdf_backend: ArxivPDFFetchBackend | None = None,
    arxiv_max_pdf_bytes: int = 52_428_800,
    arxiv_download_timeout_seconds: float = 180.0,
    resolver_timeout_seconds: float = 210.0,
    reader_timeout_seconds: float = 120.0,
    reading_workflow_timeout_seconds: float = 600.0,
    entity_embedding_model: str = "",
    round_strategy: str = "entity_group",
    subquestion_concurrency: int = 2,
    source_concurrency_limit: int = 2,
    fresh_run_timeout_seconds: float = 840.0,
) -> AgenticM2Adapter:
    if final_k <= 0:
        raise ValueError("final_k must be positive")
    if source_timeout_seconds <= 0:
        raise ValueError("source_timeout_seconds must be positive")
    if scout_timeout_seconds <= 0 or scout_candidate_limit <= 0:
        raise ValueError("Scout limits must be positive")
    if per_query_limit is not None and per_query_limit <= 0:
        raise ValueError("per_query_limit must be positive")
    if arxiv_max_pdf_bytes <= 0:
        raise ValueError("arxiv_max_pdf_bytes must be positive")
    if arxiv_download_timeout_seconds <= 0 or resolver_timeout_seconds <= 0:
        raise ValueError("reading timeouts must be positive")
    if access_enrichment_timeout_seconds <= 0:
        raise ValueError("access_enrichment_timeout_seconds must be positive")
    if fulltext_backfill_max_attempts <= 0:
        raise ValueError("fulltext_backfill_max_attempts must be positive")
    if fulltext_target_per_subquestion <= 0:
        raise ValueError("fulltext_target_per_subquestion must be positive")
    if fulltext_target_per_subquestion > final_k:
        raise ValueError(
            "fulltext_target_per_subquestion cannot exceed final_k; otherwise "
            "every sub-question is forced into backfill/expansion"
        )
    if subquestion_concurrency <= 0 or source_concurrency_limit <= 0:
        raise ValueError("M2 concurrency limits must be positive")
    if fresh_run_timeout_seconds <= 0:
        raise ValueError("fresh_run_timeout_seconds must be positive")

    tool = search_tool or LiteratureSearchTool(
        enabled_sources=enabled_sources,
        semantic_scholar_api_key=semantic_scholar_api_key,
        openalex_api_key=openalex_api_key,
        openalex_mailto=openalex_mailto,
        serper_api_key=serper_api_key,
        ads_api_token=ads_api_token,
        crossref_mailto=crossref_mailto,
        zero_result_relaxation=zero_result_relaxation,
    )
    if isinstance(budget, Mapping):
        resolved_budget = SearchBudget.model_validate(dict(budget))
    else:
        resolved_budget = budget or SearchBudget(
            max_rounds=3,
            max_queries=12,
            max_papers=max(100, final_k),
        )
    query_client = query_client or client
    base_planner = QueryPlanner(
        client=query_client,
        tool_definitions=tool.tool_definitions,
        max_rounds=resolved_budget.max_rounds,
        strict=True,
    )
    planner = (
        DomainRoutedQueryPlanner(
            base_planner,
            [definition["name"] for definition in tool.tool_definitions],
        )
        if domain_routing_enabled else base_planner
    )
    agent = IterativeSearchAgent(
        query_planner=planner,
        sources=tool.as_source_list(),
        deduplicator=PaperDeduplicator(),
        ranker=PaperRanker(prefer_high_citation=domain_routing_enabled),
        scout_reader=ScoutReader(client),
        # Coverage must be satisfied by the same leading papers that the
        # agent will hand to the reading workflow, not by discarded tail
        # candidates. This remains internal wiring; Protocol signatures stay
        # unchanged.
        coverage_evaluator=CoverageEvaluator(client, selection_limit=final_k),
        final_k=final_k,
        candidate_limit=max(48, final_k),
        per_query_limit=per_query_limit or final_k,
        scout_candidate_limit=scout_candidate_limit,
        scout_timeout_seconds=scout_timeout_seconds,
        source_timeout_seconds=source_timeout_seconds,
        retention_judge_client=client,
        # Must/unmust entity classification for the entity-group round
        # strategy (reuses the shared Qwen client; failures degrade to a
        # deterministic fallback inside the search agent).
        entity_classifier=client,
        source_concurrency_limit=source_concurrency_limit,
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
        resolver=EnrichingFulltextResolver(
            RoutingFulltextResolver(pmc_resolver, arxiv_resolver),
            OpenAccessEnricher(
                email=unpaywall_email or openalex_mailto,
                openalex_api_key=openalex_api_key,
                semantic_scholar_api_key=semantic_scholar_api_key,
                timeout_seconds=access_enrichment_timeout_seconds,
            ),
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
        entity_judge_client=client,
        entity_embedding_model=entity_embedding_model,
        subquestion_entity_client=query_client,
        round_strategy=round_strategy,
        fulltext_backfill_enabled=domain_routing_enabled,
        fulltext_backfill_target=fulltext_target_per_subquestion,
        fulltext_backfill_max_attempts=fulltext_backfill_max_attempts,
        subquestion_concurrency=subquestion_concurrency,
        fresh_run_timeout_seconds=fresh_run_timeout_seconds,
    )
