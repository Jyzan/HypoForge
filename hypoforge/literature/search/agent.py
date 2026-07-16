"""Dependency-injected iterative literature search orchestration."""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Callable, Sequence

from ..models import (
    CoverageReport,
    PaperRecord,
    ScoutNote,
    SearchBudget,
    SearchQuery,
    SearchRunResult,
    SearchState,
    StopReason,
)
from ..protocols import (
    CoverageEvaluatorProtocol,
    LiteratureSourceProtocol,
    PaperDeduplicatorProtocol,
    PaperRankerProtocol,
    QueryPlannerProtocol,
    ScoutReaderProtocol,
)
from .budget import calculate_remaining, choose_stop_reason, estimate_tokens


class IterativeSearchAgent:
    """Coordinate search Tools while keeping documents outside Agent state."""

    _SECRET_PATTERN = re.compile(
        r"(?i)\b(api[_-]?key|access[_-]?token|token|authorization|password)"
        r"\b(\s*[=:]\s*)(?:\"[^\"]*\"|'[^']*'|[^\s&]+)"
    )

    def __init__(
        self,
        *,
        query_planner: QueryPlannerProtocol,
        sources: Sequence[LiteratureSourceProtocol],
        deduplicator: PaperDeduplicatorProtocol,
        ranker: PaperRankerProtocol,
        scout_reader: ScoutReaderProtocol,
        coverage_evaluator: CoverageEvaluatorProtocol,
        final_k: int = 10,
        candidate_limit: int = 30,
        per_query_limit: int = 20,
        source_timeout_seconds: float = 30.0,
        min_new_papers: int = 1,
        no_result_round_limit: int = 2,
        low_gain_round_limit: int = 2,
        clock: Callable[[], float] = time.monotonic,
        token_estimator: Callable[[str], int] = estimate_tokens,
    ) -> None:
        for name, value in (
            ("final_k", final_k),
            ("candidate_limit", candidate_limit),
            ("per_query_limit", per_query_limit),
            ("source_timeout_seconds", source_timeout_seconds),
            ("min_new_papers", min_new_papers),
            ("no_result_round_limit", no_result_round_limit),
            ("low_gain_round_limit", low_gain_round_limit),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")

        source_map: dict[str, LiteratureSourceProtocol] = {}
        for source in sources:
            source_name = source.source_name.strip()
            if not source_name:
                raise ValueError("source_name must not be empty")
            key = source_name.casefold()
            if key in source_map:
                raise ValueError(f"duplicate literature source: {source_name}")
            source_map[key] = source

        self.query_planner = query_planner
        self.sources = source_map
        self.deduplicator = deduplicator
        self.ranker = ranker
        self.scout_reader = scout_reader
        self.coverage_evaluator = coverage_evaluator
        self.final_k = final_k
        self.candidate_limit = candidate_limit
        self.per_query_limit = per_query_limit
        self.source_timeout_seconds = source_timeout_seconds
        self.min_new_papers = min_new_papers
        self.no_result_round_limit = no_result_round_limit
        self.low_gain_round_limit = low_gain_round_limit
        self.clock = clock
        self.token_estimator = token_estimator

    async def run(
        self,
        sub_question: str,
        key_entities: Sequence[str] = (),
        domains: Sequence[str] = (),
        question_type: str = "",
        budget: SearchBudget | None = None,
        existing_papers: Sequence[PaperRecord] = (),
    ) -> SearchRunResult:
        limits = budget or SearchBudget()
        started_at = self.clock()
        state = SearchState()
        state.remaining_budget = calculate_remaining(limits, state)
        all_queries: list[SearchQuery] = []
        query_history: set[tuple[str, str]] = set()
        canonical_history: dict[str, PaperRecord] = {}
        candidate_pool: dict[str, PaperRecord] = {}
        scout_by_paper: dict[str, ScoutNote] = {}
        coverage = CoverageReport()
        ranked: list[PaperRecord] = []
        papers_found = 0
        failed_sources: list[str] = []
        errors: list[str] = []
        source_result_counts: dict[str, int] = {}
        reused_ids: set[str] = set()
        existing_ids = {paper.paper_id for paper in existing_papers}
        stop_reason = None

        while stop_reason is None:
            try:
                planned = await self.query_planner.plan(
                    sub_question,
                    key_entities=key_entities,
                    domains=domains,
                    question_type=question_type,
                    state=state.model_copy(deep=True),
                )
            except Exception as exc:
                errors.append(self._format_error("query_planner", exc))
                stop_reason = StopReason.ERROR
                break
            round_index = state.round_index + 1
            queries: list[SearchQuery] = []
            for planned_query in planned:
                key = self._query_key(planned_query)
                if key in query_history:
                    continue
                queries.append(
                    planned_query.model_copy(update={"round_index": round_index})
                )
                query_history.add(key)
                if len(queries) >= state.remaining_budget.max_queries:
                    break

            if not queries:
                errors.append("query planner produced no new executable queries")
                stop_reason = StopReason.NO_RESULTS
                break

            search_results = await asyncio.gather(
                *(self._search(query) for query in queries),
                return_exceptions=True,
            )
            raw_papers: list[PaperRecord] = []
            successful_queries = 0
            for query, result in zip(queries, search_results):
                source_name = query.target_source
                if isinstance(result, BaseException):
                    if source_name not in failed_sources:
                        failed_sources.append(source_name)
                    errors.append(self._format_error(source_name, result))
                    continue
                successful_queries += 1
                source_result_counts[source_name] = (
                    source_result_counts.get(source_name, 0) + len(result)
                )
                raw_papers.extend(result)

            papers_found += len(raw_papers)
            all_queries.extend(queries)
            state.queries_used = list(all_queries)
            state.queries_executed += len(queries)
            previous_ids = set(canonical_history)
            try:
                canonical_batch = await self.deduplicator.deduplicate(
                    raw_papers,
                    existing_papers=[*existing_papers, *canonical_history.values()],
                )
            except Exception as exc:
                errors.append(self._format_error("paper_deduplicator", exc))
                stop_reason = StopReason.ERROR
                break
            for paper in canonical_batch:
                if (
                    paper.paper_id not in canonical_history
                    and len(canonical_history) >= limits.max_papers
                ):
                    continue
                canonical_history[paper.paper_id] = paper
                candidate_pool[paper.paper_id] = paper
                if paper.paper_id in existing_ids:
                    reused_ids.add(paper.paper_id)

            try:
                ranked = await self.ranker.rank(
                    sub_question,
                    list(candidate_pool.values()),
                    limit=min(self.candidate_limit, limits.max_papers),
                )
            except Exception as exc:
                errors.append(self._format_error("paper_ranker", exc))
                stop_reason = StopReason.ERROR
                break
            candidate_pool = {paper.paper_id: paper for paper in ranked}
            new_papers = len(set(canonical_history) - previous_ids)
            try:
                scout_notes = await self.scout_reader.read(sub_question, ranked)
            except Exception as exc:
                errors.append(self._format_error("scout_reader", exc))
                stop_reason = StopReason.ERROR
                break
            scout_by_paper.update({note.paper_id: note for note in scout_notes})

            state.round_index = round_index
            state.candidate_paper_ids = list(candidate_pool)
            state.unique_papers_seen = len(canonical_history)
            if new_papers == 0:
                state.consecutive_no_result_rounds += 1
            else:
                state.consecutive_no_result_rounds = 0
            if new_papers < self.min_new_papers:
                state.consecutive_low_gain_rounds += 1
            else:
                state.consecutive_low_gain_rounds = 0
            state.known_terms.update(
                term
                for note in scout_notes
                for term in (*note.key_terms, *note.entities)
            )
            state.estimated_tokens_used += self.token_estimator(
                "\n".join(
                    [
                        sub_question,
                        *(paper.abstract for paper in ranked),
                        state.model_dump_json(),
                    ]
                )
            )
            state.elapsed_seconds = max(0.0, self.clock() - started_at)

            try:
                coverage = await self.coverage_evaluator.evaluate(
                    sub_question,
                    ranked,
                    scout_notes,
                    state.model_copy(deep=True),
                )
            except Exception as exc:
                errors.append(self._format_error("coverage_evaluator", exc))
                stop_reason = StopReason.ERROR
                break
            self._apply_coverage(state, coverage)
            state.remaining_budget = calculate_remaining(limits, state)
            stop_reason = choose_stop_reason(
                coverage=coverage,
                budget=limits,
                state=state,
                all_queries_failed=bool(queries) and successful_queries == 0,
                has_candidates=bool(ranked),
                no_result_round_limit=self.no_result_round_limit,
                low_gain_round_limit=self.low_gain_round_limit,
            )

        return SearchRunResult(
            sub_question=sub_question,
            queries=all_queries,
            papers_found=papers_found,
            papers_after_dedup=len(canonical_history),
            candidates=ranked,
            final_papers=ranked[: self.final_k],
            coverage=coverage,
            failed_sources=failed_sources,
            iterations=state.round_index,
            stop_reason=stop_reason,
            errors=errors,
            source_result_counts=source_result_counts,
            scout_notes=list(scout_by_paper.values()),
            reused_paper_ids=[
                paper.paper_id for paper in ranked if paper.paper_id in reused_ids
            ],
            final_state=state,
        )

    async def _search(self, query: SearchQuery) -> list[PaperRecord]:
        source = self.sources.get(query.target_source.casefold())
        if source is None:
            raise LookupError(f"unknown literature source: {query.target_source}")
        return await asyncio.wait_for(
            source.search(query, limit=self.per_query_limit),
            timeout=self.source_timeout_seconds,
        )

    @staticmethod
    def _apply_coverage(state: SearchState, coverage: CoverageReport) -> None:
        state.covered_topics = set(coverage.covered_topics)
        state.missing_topics = set(coverage.missing_topics)
        for bucket in coverage.covered_buckets:
            state.bucket_counts[bucket] = max(1, state.bucket_counts.get(bucket, 0))

    @staticmethod
    def _query_key(query: SearchQuery) -> tuple[str, str]:
        normalized_text = " ".join(query.text.casefold().split())
        return query.target_source.casefold(), normalized_text

    @classmethod
    def _format_error(cls, stage: str, exc: BaseException) -> str:
        message = " ".join(str(exc).split())
        message = cls._SECRET_PATTERN.sub(r"\1\2<redacted>", message)
        return f"{stage}: {type(exc).__name__}: {message[:300]}"
