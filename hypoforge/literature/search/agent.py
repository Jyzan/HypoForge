"""Dependency-injected iterative literature search orchestration."""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Callable, Sequence

from ...observability import emit_event
from ..models import (
    CoverageReport,
    PaperRecord,
    PaperRetentionDecision,
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
from .ranking import domain_token_overlap, rerank_with_scout, select_retained_papers


class _SearchTimeBudgetExpired(TimeoutError):
    def __init__(self, stage: str) -> None:
        super().__init__(f"search time budget exhausted during {stage}")
        self.stage = stage


class IterativeSearchAgent:
    """Coordinate search Tools while keeping documents outside Agent state."""

    _SECRET_PATTERN = re.compile(
        r"(?i)\b(api[_-]?key|access[_-]?token|token|authorization|password)"
        r"\b(\s*[=:]\s*)(?:\"[^\"]*\"|'[^']*'|[^\s&]+)"
    )

    @staticmethod
    def _deterministic_relaxations(
        query: SearchQuery,
        key_entities: Sequence[str],
    ) -> list[SearchQuery]:
        """Build a stable strict→unfielded→unquoted→keyword fallback chain."""

        original = " ".join(query.text.split())
        candidates: list[str] = []
        unfielded = re.sub(r"\[[^\]]+\]", "", original)
        unfielded = re.sub(
            r"\b(?:title|abstract|author|keyword):", "", unfielded, flags=re.I
        )
        candidates.append(" ".join(unfielded.split()))
        unquoted = re.sub(r'["“”]', "", unfielded)
        candidates.append(" ".join(unquoted.split()))

        anchors = [" ".join(str(value).split()) for value in key_entities if str(value).strip()]
        anchor_words = {
            token.casefold()
            for anchor in anchors
            for token in re.findall(r"[\w\-]+", anchor, flags=re.UNICODE)
        }
        terms = re.findall(r"[\w\-]+", unquoted, flags=re.UNICODE)
        removable = [
            index for index, term in enumerate(terms)
            if term.casefold() not in anchor_words
        ]
        for index in reversed(removable):
            if len(terms) <= max(3, len(anchor_words)):
                break
            terms = terms[:index] + terms[index + 1:]
            anchored = list(terms)
            for anchor in anchors:
                if anchor.casefold() not in " ".join(anchored).casefold():
                    anchored.append(anchor)
            candidates.append(" ".join(anchored))

        output: list[SearchQuery] = []
        seen = {original.casefold()}
        for index, text in enumerate(candidates, start=1):
            key = text.casefold()
            if not text or key in seen:
                continue
            seen.add(key)
            output.append(query.model_copy(update={
                "query_id": f"{query.query_id}:relax{index}",
                "text": text,
                "purpose": f"deterministic fallback: {query.purpose}",
                "relation_to_question": (
                    query.relation_to_question
                    + " Deterministic relaxation with core entities preserved."
                ),
            }))
        return output

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
        stage_clock: Callable[[], float] = time.perf_counter,
        token_estimator: Callable[[str], int] = estimate_tokens,
        retention_judge_client: object | None = None,
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
        self.retention_judge_client = retention_judge_client
        self.final_k = final_k
        self.candidate_limit = candidate_limit
        self.per_query_limit = per_query_limit
        self.source_timeout_seconds = source_timeout_seconds
        self.min_new_papers = min_new_papers
        self.no_result_round_limit = no_result_round_limit
        self.low_gain_round_limit = low_gain_round_limit
        self.clock = clock
        self.stage_clock = stage_clock
        self.token_estimator = token_estimator
        self._search_cache: dict[tuple[str, str, int], list[PaperRecord]] = {}

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
        deadline_started_at = time.monotonic()
        state = SearchState(
            question_type=question_type,
            key_entities=set(key_entities),
            domains=set(domains),
        )
        alignment_question = "\n".join(filter(None, [
            sub_question,
            "Task entities for this atomic sub-question: " + ", ".join(key_entities)
            if key_entities else "",
            "Task domains (required subject areas — papers outside these domains are not applicable): " + ", ".join(domains)
            if domains else "",
        ]))
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
        stage_elapsed_seconds = {
            "query_planner": 0.0,
            "source_search": 0.0,
            "paper_deduplicator": 0.0,
            "paper_ranker": 0.0,
            "scout_reader": 0.0,
            "coverage_evaluator": 0.0,
        }

        async def measure(stage: str, operation):
            stage_started_at = self.stage_clock()
            emit_event(
                "tool_started",
                module="m2",
                tool=stage,
                status="running",
                message=f"M2 Tool 开始：{stage}",
                details={"round": state.round_index + 1},
            )
            try:
                remaining_seconds = limits.max_seconds - (
                    time.monotonic() - deadline_started_at
                )
                if remaining_seconds <= 0:
                    close = getattr(operation, "close", None)
                    if callable(close):
                        close()
                    cancel = getattr(operation, "cancel", None)
                    if callable(cancel):
                        cancel()
                    raise _SearchTimeBudgetExpired(stage)
                task = asyncio.ensure_future(operation)
                try:
                    result = await asyncio.wait_for(task, timeout=remaining_seconds)
                    emit_event(
                        "tool_completed",
                        module="m2",
                        tool=stage,
                        status="completed",
                        message=f"M2 Tool 完成：{stage}",
                        elapsed_seconds=max(
                            0.0, self.stage_clock() - stage_started_at
                        ),
                        details={"round": state.round_index + 1},
                    )
                    return result
                except asyncio.TimeoutError as exc:
                    if task.done() and not task.cancelled():
                        raise
                    raise _SearchTimeBudgetExpired(stage) from exc
            except BaseException as exc:
                emit_event(
                    "tool_failed",
                    module="m2",
                    tool=stage,
                    status="failed",
                    message=f"M2 Tool 失败：{stage}：{type(exc).__name__}: {exc}",
                    elapsed_seconds=max(0.0, self.stage_clock() - stage_started_at),
                    details={"round": state.round_index + 1},
                )
                raise
            finally:
                elapsed = max(0.0, self.stage_clock() - stage_started_at)
                stage_elapsed_seconds[stage] += elapsed

        def mark_time_budget_expired(exc: _SearchTimeBudgetExpired) -> StopReason:
            errors.append(self._format_error(exc.stage, exc))
            state.elapsed_seconds = max(
                state.elapsed_seconds, float(limits.max_seconds)
            )
            state.remaining_budget = calculate_remaining(limits, state)
            return StopReason.TIME_BUDGET

        while stop_reason is None:
            try:
                planned = await measure(
                    "query_planner",
                    self.query_planner.plan(
                        sub_question,
                        key_entities=key_entities,
                        domains=domains,
                        question_type=question_type,
                        state=state.model_copy(deep=True),
                    ),
                )
            except _SearchTimeBudgetExpired as exc:
                stop_reason = mark_time_budget_expired(exc)
                break
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
                query_history.add(key)
                if planned_query.target_source in state.unavailable_sources:
                    continue
                queries.append(
                    planned_query.model_copy(update={"round_index": round_index})
                )
                if len(queries) >= state.remaining_budget.max_queries:
                    break

            if not queries:
                errors.append("query planner produced no new executable queries")
                stop_reason = StopReason.NO_RESULTS
                break
            emit_event(
                "tool_result",
                module="m2",
                tool="query_planner",
                status="completed",
                message=f"第 {round_index} 轮生成 {len(queries)} 条检索式",
                details={
                    "round": round_index,
                    "queries": [
                        {
                            "text": query.text,
                            "source": query.target_source,
                            "purpose": query.purpose,
                        }
                        for query in queries
                    ],
                },
            )

            try:
                search_results = await measure(
                    "source_search",
                    asyncio.gather(
                        *(self._search(query) for query in queries),
                        return_exceptions=True,
                    ),
                )
            except _SearchTimeBudgetExpired as exc:
                stop_reason = mark_time_budget_expired(exc)
                break
            # Empty results use a deterministic relaxation chain before asking
            # the planner for another stochastic round. Every fallback is
            # recorded and consumes the same query budget.
            for original_query, original_result in list(zip(queries, search_results)):
                if isinstance(original_result, BaseException) or original_result:
                    continue
                for relaxed in self._deterministic_relaxations(
                    original_query, key_entities
                ):
                    if state.queries_executed + len(queries) >= limits.max_queries:
                        break
                    key = self._query_key(relaxed)
                    if key in query_history:
                        continue
                    query_history.add(key)
                    queries.append(relaxed)
                    try:
                        relaxed_result = await self._search(relaxed)
                    except BaseException as exc:
                        search_results.append(exc)
                        break
                    search_results.append(relaxed_result)
                    if relaxed_result:
                        break
            raw_papers: list[PaperRecord] = []
            successful_queries = 0
            for query, result in zip(queries, search_results):
                source_name = query.target_source
                if isinstance(result, BaseException):
                    if source_name not in failed_sources:
                        failed_sources.append(source_name)
                    state.unavailable_sources.add(source_name)
                    errors.append(self._format_error(source_name, result))
                    continue
                successful_queries += 1
                source_result_counts[source_name] = (
                    source_result_counts.get(source_name, 0) + len(result)
                )
                for paper in result:
                    providers = {
                        provider.casefold() for provider in paper.sources
                    }
                    if providers and source_name.casefold() not in providers:
                        for provider in sorted(providers):
                            fallback_key = f"{source_name}->{provider}"
                            source_result_counts[fallback_key] = (
                                source_result_counts.get(fallback_key, 0) + 1
                            )
                raw_papers.extend(result)

            papers_found += len(raw_papers)
            emit_event(
                "tool_result",
                module="m2",
                tool="source_search",
                status="completed",
                message=f"第 {round_index} 轮检索获得 {len(raw_papers)} 篇记录",
                details={
                    "round": round_index,
                    "source_result_counts": dict(source_result_counts),
                    "failed_sources": list(failed_sources),
                },
            )
            all_queries.extend(queries)
            state.queries_used = list(all_queries)
            state.queries_executed += len(queries)
            previous_ids = set(canonical_history)
            try:
                canonical_batch = await measure(
                    "paper_deduplicator",
                    self.deduplicator.deduplicate(
                        raw_papers,
                        existing_papers=[
                            *existing_papers,
                            *canonical_history.values(),
                        ],
                    ),
                )
            except _SearchTimeBudgetExpired as exc:
                stop_reason = mark_time_budget_expired(exc)
                break
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
            emit_event(
                "tool_result",
                module="m2",
                tool="paper_deduplicator",
                status="completed",
                message=f"去重后累计 {len(canonical_history)} 篇论文",
                details={"round": round_index, "papers": len(canonical_history)},
            )

            try:
                ranked = await measure(
                    "paper_ranker",
                    self.ranker.rank(
                        alignment_question,
                        list(candidate_pool.values()),
                        limit=min(self.candidate_limit, limits.max_papers),
                    ),
                )
            except _SearchTimeBudgetExpired as exc:
                stop_reason = mark_time_budget_expired(exc)
                break
            except Exception as exc:
                errors.append(self._format_error("paper_ranker", exc))
                stop_reason = StopReason.ERROR
                break
            candidate_pool = {paper.paper_id: paper for paper in ranked}
            emit_event(
                "tool_result",
                module="m2",
                tool="paper_ranker",
                status="completed",
                message=f"排序后保留 {len(ranked)} 篇候选论文",
                details={
                    "round": round_index,
                    "top_titles": [paper.title for paper in ranked[:5]],
                },
            )
            new_papers = len(set(canonical_history) - previous_ids)
            papers_needing_scout = [
                paper for paper in ranked if paper.paper_id not in scout_by_paper
            ]
            try:
                new_scout_notes = await measure(
                    "scout_reader",
                    self.scout_reader.read(alignment_question, papers_needing_scout),
                )
            except _SearchTimeBudgetExpired as exc:
                stop_reason = mark_time_budget_expired(exc)
                break
            except Exception as exc:
                errors.append(self._format_error("scout_reader", exc))
                stop_reason = StopReason.ERROR
                break
            scout_by_paper.update(
                {note.paper_id: note for note in new_scout_notes}
            )
            emit_event(
                "tool_result",
                module="m2",
                tool="scout_reader",
                status="completed",
                message=f"快速阅读新增 {len(new_scout_notes)} 篇论文",
                details={"round": round_index, "notes": len(new_scout_notes)},
            )
            scout_notes = [
                scout_by_paper[paper.paper_id]
                for paper in ranked
                if paper.paper_id in scout_by_paper
            ]
            ranked = rerank_with_scout(
                ranked,
                scout_notes,
                selection_limit=self.final_k,
            )
            candidate_pool = {paper.paper_id: paper for paper in ranked}
            scout_notes = [
                scout_by_paper[paper.paper_id]
                for paper in ranked
                if paper.paper_id in scout_by_paper
            ]

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
                coverage = await measure(
                    "coverage_evaluator",
                    self.coverage_evaluator.evaluate(
                        sub_question,
                        ranked,
                        scout_notes,
                        state.model_copy(deep=True),
                    ),
                )
            except _SearchTimeBudgetExpired as exc:
                stop_reason = mark_time_budget_expired(exc)
                break
            except Exception as exc:
                errors.append(self._format_error("coverage_evaluator", exc))
                stop_reason = StopReason.ERROR
                break
            self._apply_coverage(state, coverage)
            emit_event(
                "tool_result",
                module="m2",
                tool="coverage_evaluator",
                status="completed",
                message=(
                    "证据覆盖充分"
                    if coverage.sufficient
                    else "证据覆盖仍有缺口，将按预算决定是否迭代"
                ),
                details={
                    "round": round_index,
                    "sufficient": coverage.sufficient,
                    "covered_topics": list(coverage.covered_topics),
                    "missing_topics": list(coverage.missing_topics),
                    "rationale": coverage.rationale,
                },
            )
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

        applicable_finalists = [
            paper
            for paper in ranked
            if paper.rank_scores.get("semantic_not_applicable", 0.0) < 0.5
        ]
        # Domain token overlap is a weak prior, NOT a hard reject.  Literal
        # token mismatch can produce false negatives (e.g. "robotics" vs
        # "robotic", "machine learning" vs "neural network").  The primary
        # hard gate remains Scout `semantic_not_applicable`.
        if state.domains:
            for paper in applicable_finalists:
                if domain_token_overlap(paper, state.domains) == 0.0:
                    paper.rank_scores["domain_token_mismatch"] = True
        # Note: we do NOT re-filter based on domain_token_mismatch alone.
        # Scout's semantic applicability judgment is the definitive gate.
        final_papers, applicable_decisions = select_retained_papers(
            applicable_finalists,
            list(scout_by_paper.values()),
            final_k=self.final_k,
        )
        if self.retention_judge_client is not None:
            final_papers, applicable_decisions = (
                await self._review_retention_boundary(
                    applicable_finalists,
                    applicable_decisions,
                    list(scout_by_paper.values()),
                    sub_question,
                    key_entities,
                )
            )
        decisions_by_id = {
            decision.paper_id: decision for decision in applicable_decisions
        }
        retention_decisions: list[PaperRetentionDecision] = []
        for rank_position, paper in enumerate(ranked, start=1):
            decision = decisions_by_id.get(paper.paper_id)
            if decision is None:
                decision = PaperRetentionDecision(
                    paper_id=paper.paper_id,
                    decision="reject",
                    roles=[],
                    reason="Scout marked the paper not applicable to the atomic task",
                    rank_position=rank_position,
                )
            retention_decisions.append(decision)
        if ranked and not final_papers:
            errors.append(
                "domain guard rejected all ranked papers as not applicable; "
                "no unrelated fallback papers were retained"
            )
        return SearchRunResult(
            sub_question=sub_question,
            queries=all_queries,
            papers_found=papers_found,
            papers_after_dedup=len(canonical_history),
            candidates=ranked,
            final_papers=final_papers,
            coverage=coverage,
            failed_sources=failed_sources,
            iterations=state.round_index,
            stop_reason=stop_reason,
            errors=errors,
            source_result_counts=source_result_counts,
            stage_elapsed_seconds=stage_elapsed_seconds,
            scout_notes=list(scout_by_paper.values()),
            reused_paper_ids=[
                paper.paper_id for paper in ranked if paper.paper_id in reused_ids
            ],
            retention_decisions=retention_decisions,
            final_state=state,
        )

    async def _review_retention_boundary(
        self,
        ranked: list[PaperRecord],
        decisions: list[PaperRetentionDecision],
        notes: list[ScoutNote],
        sub_question: str,
        key_entities: list[str],
    ) -> tuple[list[PaperRecord], list[PaperRetentionDecision]]:
        """Apply LLM boundary review to rule-based retention decisions.

        This is called from the async ``run()`` context, so the QwenClient
        async ``chat`` method is properly awaited.  Failure propagates as a
        ``RuntimeError`` — the retention judge, once enabled, is required.
        """
        if self.retention_judge_client is None:
            return ranked, decisions

        note_by_id = {note.paper_id: note for note in notes}
        from .ranking import paper_retention_roles, _safe_score

        window_size = min(len(ranked), max(self.final_k * 2, self.final_k + 3))
        retained_ids = {dec.paper_id for dec in decisions if dec.decision == "retain"}
        boundary_ids: set[str] = set()

        # Rejected papers inside the window
        for dec in decisions:
            if dec.decision == "reject" and dec.rank_position <= window_size:
                boundary_ids.add(dec.paper_id)

        # Lowest-scoring retained papers (up to 2)
        retained_by_score = sorted(
            [dec for dec in decisions
             if dec.decision == "retain" and dec.paper_id not in boundary_ids],
            key=lambda d: _safe_score(
                ranked[d.rank_position - 1].rank_scores.get("post_scout_total"), 0.0,
            ),
        )
        for dec in retained_by_score[:2]:
            boundary_ids.add(dec.paper_id)

        if not boundary_ids:
            return ranked, decisions

        boundary_papers = [p for p in ranked if p.paper_id in boundary_ids]
        candidates_text = "\n".join(
            f"- {p.paper_id} | {p.title or '?'} | {p.year or '?'} | "
            f"rel={p.rank_scores.get('scout_relevance', 0.0):.2f} | "
            f"dir={p.rank_scores.get('scout_directness', 0.0):.2f} | "
            f"roles={','.join(paper_retention_roles(p, note_by_id.get(p.paper_id)))} | "
            f"current={'retain' if p.paper_id in retained_ids else 'reject'}"
            for p in boundary_papers
        )
        prompt = (
            f"Sub-question: {sub_question}\n"
            f"Key entities: {', '.join(key_entities) if key_entities else ''}\n\n"
            f"Candidates (title | year | relevance | directness | evidence_roles | current_decision):\n"
            f"{candidates_text}\n\n"
            f"For each candidate above, output your retain/reject decision "
            f"as JSON: {{\"paper_id\": \"...\", \"decision\": \"retain\"|\"reject\", "
            f"\"rationale\": \"...\"}}.  Only re-judge boundary papers."
        )
        import json as _json
        try:
            raw = await self.retention_judge_client.chat(
                system_prompt=(
                    "You audit a paper retention decision for a scientific "
                    "literature search. Decide whether each boundary candidate "
                    "should be retained or rejected based on its unique "
                    "contribution, directness, methodological complement, and "
                    "recency.  Output JSON only."
                ),
                user_prompt=prompt,
            )
        except Exception as exc:
            raise RuntimeError(
                "Retention judge LLM call failed — the judge is enabled and "
                "required. Check the retention judge client configuration."
            ) from exc
        try:
            rulings = _json.loads(str(raw)) if isinstance(raw, str) else raw
        except Exception as exc:
            raise RuntimeError(
                "Retention judge returned invalid JSON — structured output "
                "is required when the retention judge is enabled."
            ) from exc
        if isinstance(rulings, dict):
            rulings = [rulings]
        if not isinstance(rulings, list):
            raise RuntimeError(
                "Retention judge returned unexpected output type "
                f"{type(rulings).__name__}; expected a JSON array."
            )
        for ruling in rulings:
            pid = str(ruling.get("paper_id", ""))
            new_decision = str(ruling.get("decision", "")).lower()
            rationale = str(ruling.get("rationale", ""))
            if new_decision not in ("retain", "reject"):
                continue
            for dec in decisions:
                if dec.paper_id == pid:
                    if new_decision != dec.decision:
                        dec.decision = new_decision
                        dec.llm_override = True
                        dec.llm_rationale = rationale
                        if new_decision == "retain":
                            dec.reason = f"LLM override: {rationale}"
                        else:
                            dec.reason = f"LLM rejected: {rationale}"
                    break
        retained = [
            ranked[dec.rank_position - 1]
            for dec in decisions if dec.decision == "retain"
        ]
        return retained, decisions

    async def _search(self, query: SearchQuery) -> list[PaperRecord]:
        source = self.sources.get(query.target_source.casefold())
        if source is None:
            raise LookupError(f"unknown literature source: {query.target_source}")
        started_at = self.stage_clock()
        cache_key = (
            query.target_source.casefold(),
            " ".join(query.text.casefold().split()),
            self.per_query_limit,
        )
        cached = self._search_cache.get(cache_key)
        if cached is not None:
            emit_event(
                "tool_result",
                module="m2",
                tool=f"source:{query.target_source}",
                status="completed",
                message=f"复用 {query.target_source} 检索缓存 ({len(cached)} 篇)",
                details={"query": query.text, "cache_hit": True},
            )
            return [paper.model_copy(deep=True) for paper in cached]
        tool = f"source:{query.target_source}"
        emit_event(
            "tool_started",
            module="m2",
            tool=tool,
            status="running",
            message=f"检索 {query.target_source}",
            details={
                "query": query.text,
                "round": query.round_index,
                "limit": self.per_query_limit,
            },
        )
        try:
            result = await asyncio.wait_for(
                source.search(query, limit=self.per_query_limit),
                timeout=self.source_timeout_seconds,
            )
        except BaseException as exc:
            emit_event(
                "tool_failed",
                module="m2",
                tool=tool,
                status="failed",
                message=f"{query.target_source} 检索失败：{type(exc).__name__}: {exc}",
                elapsed_seconds=max(0.0, self.stage_clock() - started_at),
                details={"query": query.text, "round": query.round_index},
            )
            raise
        emit_event(
            "tool_completed",
            module="m2",
            tool=tool,
            status="completed",
            message=f"{query.target_source} 返回 {len(result)} 篇记录",
            elapsed_seconds=max(0.0, self.stage_clock() - started_at),
            details={
                "query": query.text,
                "round": query.round_index,
                "papers": len(result),
            },
        )
        self._search_cache[cache_key] = [
            paper.model_copy(deep=True) for paper in result
        ]
        return result

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
