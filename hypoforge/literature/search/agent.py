"""Dependency-injected iterative literature search orchestration."""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

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
from .round_plan import (
    RoundSpec,
    classify_entities,
    plan_entity_rounds,
    poor_round_reason,
)


class _SearchTimeBudgetExpired(TimeoutError):
    def __init__(self, stage: str) -> None:
        super().__init__(f"search time budget exhausted during {stage}")
        self.stage = stage


class _RoundHalt(Exception):
    """Internal control flow: a search round ended the run early."""

    def __init__(self, stop_reason: StopReason) -> None:
        super().__init__(stop_reason.value)
        self.stop_reason = stop_reason


@dataclass
class _RoundContext:
    """Mutable state shared between search rounds inside one ``run()`` call."""

    state: SearchState
    limits: SearchBudget
    alignment_question: str
    stage_elapsed_seconds: Dict[str, float]
    existing_papers: Tuple[PaperRecord, ...] = ()
    existing_ids: Set[str] = field(default_factory=set)
    run_started_at: float = 0.0
    all_queries: List[SearchQuery] = field(default_factory=list)
    query_history: Set[Tuple[str, str]] = field(default_factory=set)
    canonical_history: Dict[str, PaperRecord] = field(default_factory=dict)
    candidate_pool: Dict[str, PaperRecord] = field(default_factory=dict)
    scout_by_paper: Dict[str, ScoutNote] = field(default_factory=dict)
    coverage: CoverageReport = field(default_factory=CoverageReport)
    ranked: List[PaperRecord] = field(default_factory=list)
    papers_found: int = 0
    failed_sources: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    source_result_counts: Dict[str, int] = field(default_factory=dict)
    reused_ids: Set[str] = field(default_factory=set)
    # Snapshots taken right before each round, used for per-round deltas.
    round_start_counts: Dict[str, int] = field(default_factory=dict)
    round_start_papers_found: int = 0


class IterativeSearchAgent:
    """Coordinate search Tools while keeping documents outside Agent state."""

    _SECRET_PATTERN = re.compile(
        r"(?i)\b(api[_-]?key|access[_-]?token|token|authorization|password)"
        r"\b(\s*[=:]\s*)(?:\"[^\"]*\"|'[^']*'|[^\s&]+)"
    )


    @staticmethod
    def _finite_float(value: object) -> bool:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return False
        return number == number and number not in (float("inf"), float("-inf"))

    @staticmethod
    def _deterministic_relaxations(
        query: SearchQuery,
        key_entities: Sequence[str],
    ) -> list[SearchQuery]:
        """Build a deterministic, semantic-importance-aware fallback ladder.

        LLM-provided ``term_importance`` is the primary ordering signal.  The
        local heuristic is only a conservative fallback for offline planners.
        Anchors are protected by phrase match, and no missing anchor is ever
        appended to the end of a query (which used to create broken syntax).
        """

        original = " ".join(query.text.split())
        candidates: list[str] = []
        fieldless = re.sub(r"\[[^\]]+\]", "", original)
        fieldless = re.sub(
            r"\b(?:title|abstract|author|keyword):", "", fieldless, flags=re.I
        )
        fieldless = " ".join(fieldless.split())
        candidates.append(fieldless)
        unquoted = re.sub(r'["\u201c\u201d\u2018\u2019]', "", fieldless)
        candidates.append(" ".join(unquoted.split()))

        anchors = [
            " ".join(str(value).split()).casefold()
            for value in key_entities
            if str(value).strip()
        ]
        importance: dict[str, float] = {}
        for key, value in getattr(query, "term_importance", {}).items():
            if not str(key).strip() or not IterativeSearchAgent._finite_float(value):
                continue
            importance[" ".join(str(key).split()).casefold()] = max(
                0.0, min(1.0, float(value))
            )

        stop_modifiers = {
            "a", "an", "the", "role", "study", "using", "use", "used",
            "novel", "new", "review", "analysis", "effect", "impact",
            "association", "approach", "method", "methods", "evidence",
            "investigation", "investigating", "based", "related",
        }
        boolean = {"and", "or", "not"}

        def units(text: str) -> list[str]:
            return re.findall(
                r'"[^"\r\n]+"|\u201c[^\u201d\r\n]+\u201d|\(|\)|\bAND\b|\bOR\b|\bNOT\b|[\w][\w\-]*',
                text,
                flags=re.I | re.UNICODE,
            )

        def semantic_unit(token: str) -> str:
            return re.sub(
                r'^["\u201c\u201d\u2018\u2019]|["\u201c\u201d\u2018\u2019]$',
                "",
                token,
            ).casefold().strip()

        def clean(values: Sequence[str]) -> str:
            kept = list(values)
            # Drop unmatched parentheses and empty groups.
            while True:
                changed = False
                depth = 0
                filtered: list[str] = []
                for token in kept:
                    if token == "(":
                        depth += 1
                        filtered.append(token)
                    elif token == ")":
                        if depth:
                            depth -= 1
                            filtered.append(token)
                        else:
                            changed = True
                    else:
                        filtered.append(token)
                if depth:
                    filtered = [token for token in filtered if token != "("]
                    changed = True
                kept = filtered
                for i in range(len(kept) - 1):
                    if kept[i] == "(" and kept[i + 1] == ")":
                        del kept[i:i + 2]
                        changed = True
                        break
                if not changed:
                    break
            # Remove boolean operators that no longer have operands.
            output: list[str] = []
            for token in kept:
                low = token.casefold()
                previous = output[-1].casefold() if output else ""
                if low in boolean and (not output or previous in boolean or previous == "("):
                    continue
                if low == ")" and (not output or previous in boolean or previous == "("):
                    continue
                output.append(token)
            while output and output[-1].casefold() in boolean | {"("}:
                output.pop()
            while output and output[0] == ")":
                output.pop(0)
            return " ".join(output).strip()

        token_units = units(unquoted)
        if not token_units:
            return []

        def protected_indices(values: Sequence[str]) -> set[int]:
            semantic = [semantic_unit(token) for token in values]
            protected: set[int] = set()
            for anchor in anchors:
                parts = anchor.split()
                if not parts:
                    continue
                for index, token in enumerate(semantic):
                    if token == anchor:
                        protected.add(index)
                if len(parts) > 1:
                    for index in range(len(semantic) - len(parts) + 1):
                        if semantic[index:index + len(parts)] == parts:
                            protected.update(range(index, index + len(parts)))
            return protected

        while True:
            substantive = [
                index for index, token in enumerate(token_units)
                if semantic_unit(token) not in boolean | {"(", ")"}
            ]
            protected = protected_indices(token_units)
            removable: list[tuple[float, int]] = []
            for index in substantive:
                if index in protected:
                    continue
                term = semantic_unit(token_units[index])
                score = importance.get(term)
                if score is None:
                    words = term.split()
                    known_scores = [importance[word] for word in words if word in importance]
                    score = max(known_scores) if known_scores else (
                        0.15 if term in stop_modifiers else 0.45
                    )
                removable.append((score, index))
            if not removable:
                break
            # Lowest semantic importance first; rightmost wins ties only for
            # deterministic output, not as the semantic decision itself.
            _, remove_index = min(removable, key=lambda item: (item[0], -item[1]))
            remaining_count = len(substantive) - 1
            if remaining_count < max(2, len(protected)):
                break
            del token_units[remove_index]
            text = clean(token_units)
            if text:
                candidates.append(text)

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
                "purpose": f"deterministic fallback (importance-aware): {query.purpose}",
                "relation_to_question": (
                    query.relation_to_question
                    + " Deterministic relaxation preserves semantic anchors and removes low-importance terms first."
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
        scout_candidate_limit: int | None = None,
        scout_timeout_seconds: float = 90.0,
        source_timeout_seconds: float = 30.0,
        min_new_papers: int = 1,
        no_result_round_limit: int = 2,
        low_gain_round_limit: int = 2,
        clock: Callable[[], float] = time.monotonic,
        stage_clock: Callable[[], float] = time.perf_counter,
        token_estimator: Callable[[str], int] = estimate_tokens,
        retention_judge_client: object | None = None,
        entity_classifier: object | None = None,
    ) -> None:
        for name, value in (
            ("final_k", final_k),
            ("candidate_limit", candidate_limit),
            ("per_query_limit", per_query_limit),
            ("source_timeout_seconds", source_timeout_seconds),
            ("scout_timeout_seconds", scout_timeout_seconds),
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
        self.scout_candidate_limit = int(
            scout_candidate_limit or candidate_limit
        )
        if self.scout_candidate_limit <= 0:
            raise ValueError("scout_candidate_limit must be positive")
        self.scout_timeout_seconds = float(scout_timeout_seconds)
        self.source_timeout_seconds = source_timeout_seconds
        self.min_new_papers = min_new_papers
        self.no_result_round_limit = no_result_round_limit
        self.low_gain_round_limit = low_gain_round_limit
        self.clock = clock
        self.stage_clock = stage_clock
        self.token_estimator = token_estimator
        # LLM client for the must/unmust entity classification used by the
        # entity-group round strategy (None → deterministic fallback).
        self.entity_classifier = entity_classifier
        self._search_cache: dict[tuple[str, str, int], list[PaperRecord]] = {}

    async def run(
        self,
        sub_question: str,
        key_entities: Sequence[str] = (),
        domains: Sequence[str] = (),
        question_type: str = "",
        budget: SearchBudget | None = None,
        existing_papers: Sequence[PaperRecord] = (),
        supplement_entities: Sequence[str] = (),
        round_strategy: str = "coverage",
    ) -> SearchRunResult:
        """Run the iterative search for one atomic sub-question.

        ``round_strategy`` selects how rounds are organised:

        - ``"coverage"`` (default, legacy): the coverage evaluator decides
          whether another planner round runs.  Used by the supplement
          (gap-search) path, the minimal path and DI-fake callers.
        - ``"entity_group"``: rounds follow the deterministic must/unmust
          entity grouping plan plus a single zero-result rescue round;
          the coverage evaluator never decides rounds there.
        """

        limits = budget or SearchBudget()
        started_at = self.clock()
        deadline_started_at = time.monotonic()
        state = SearchState(
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
        existing_ids = {paper.paper_id for paper in existing_papers}
        ctx = _RoundContext(
            state=state,
            limits=limits,
            alignment_question=alignment_question,
            stage_elapsed_seconds={
                "query_planner": 0.0,
                "source_search": 0.0,
                "paper_deduplicator": 0.0,
                "paper_ranker": 0.0,
                "scout_reader": 0.0,
                "coverage_evaluator": 0.0,
            },
            existing_papers=tuple(existing_papers),
            existing_ids=existing_ids,
            run_started_at=started_at,
        )

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
                ctx.stage_elapsed_seconds[stage] += elapsed

        def mark_time_budget_expired(exc: _SearchTimeBudgetExpired) -> StopReason:
            ctx.errors.append(self._format_error(exc.stage, exc))
            state.elapsed_seconds = max(
                state.elapsed_seconds, float(limits.max_seconds)
            )
            state.remaining_budget = calculate_remaining(limits, state)
            return StopReason.TIME_BUDGET

        if round_strategy == "entity_group":
            stop_reason = await self._entity_group_rounds(
                ctx,
                measure,
                mark_time_budget_expired,
                deadline_started_at,
                sub_question,
                key_entities,
                domains,
                supplement_entities,
                question_type,
            )
        else:
            stop_reason = await self._legacy_coverage_loop(
                ctx,
                measure,
                mark_time_budget_expired,
                deadline_started_at,
                sub_question,
                key_entities,
                domains,
                supplement_entities,
                question_type,
            )
        return await self._finalize_search(
            ctx,
            sub_question,
            key_entities,
            existing_ids,
            started_at,
            stop_reason,
        )

    async def _legacy_coverage_loop(
        self,
        ctx: _RoundContext,
        measure,
        mark_time_budget_expired,
        deadline_started_at: float,
        sub_question: str,
        key_entities: Sequence[str],
        domains: Sequence[str],
        supplement_entities: Sequence[str],
        question_type: str = "",
    ) -> StopReason:
        """Legacy round organisation: the coverage evaluator decides whether
        another planner round runs.  Kept verbatim for the supplement (gap)
        path, the minimal path and DI-fake callers."""

        state = ctx.state
        limits = ctx.limits
        stop_reason: StopReason | None = None
        while stop_reason is None:
            # M2-local supplementary concepts only reach the query
            # planner (and only when present, so narrower planner
            # implementations keep working unchanged).
            planner_kwargs: dict = {}
            if question_type:
                planner_kwargs["question_type"] = question_type
            if supplement_entities:
                planner_kwargs["supplement_entities"] = list(supplement_entities)
            try:
                successful_queries = await self._execute_round(
                    ctx,
                    measure,
                    mark_time_budget_expired,
                    deadline_started_at,
                    sub_question,
                    key_entities,
                    domains,
                    planner_kwargs,
                )
            except _RoundHalt as halt:
                stop_reason = halt.stop_reason
                break

            try:
                ctx.coverage = await measure(
                    "coverage_evaluator",
                    self.coverage_evaluator.evaluate(
                        sub_question,
                        ctx.ranked,
                        [
                            ctx.scout_by_paper[paper.paper_id]
                            for paper in ctx.ranked
                            if paper.paper_id in ctx.scout_by_paper
                        ],
                        state.model_copy(deep=True),
                    ),
                )
            except _SearchTimeBudgetExpired as exc:
                stop_reason = mark_time_budget_expired(exc)
                break
            except Exception as exc:
                ctx.errors.append(self._format_error("coverage_evaluator", exc))
                stop_reason = StopReason.ERROR
                break
            self._apply_coverage(state, ctx.coverage)
            emit_event(
                "tool_result",
                module="m2",
                tool="coverage_evaluator",
                status="completed",
                message=(
                    "证据覆盖充分"
                    if ctx.coverage.sufficient
                    else "证据覆盖仍有缺口，将按预算决定是否迭代"
                ),
                details={
                    "round": state.round_index,
                    "sufficient": ctx.coverage.sufficient,
                    "covered_topics": list(ctx.coverage.covered_topics),
                    "missing_topics": list(ctx.coverage.missing_topics),
                    "rationale": ctx.coverage.rationale,
                },
            )
            state.remaining_budget = calculate_remaining(limits, state)
            stop_reason = choose_stop_reason(
                coverage=ctx.coverage,
                budget=limits,
                state=state,
                all_queries_failed=successful_queries == 0,
                has_candidates=bool(ctx.ranked),
                no_result_round_limit=self.no_result_round_limit,
                low_gain_round_limit=self.low_gain_round_limit,
            )
        return stop_reason

    async def _entity_group_rounds(
        self,
        ctx: _RoundContext,
        measure,
        mark_time_budget_expired,
        deadline_started_at: float,
        sub_question: str,
        key_entities: Sequence[str],
        domains: Sequence[str],
        supplement_entities: Sequence[str],
        question_type: str = "",
    ) -> StopReason:
        """User-designed round organisation: must/unmust entity grouping
        plus a single zero-result rescue round.

        The coverage evaluator never decides rounds here; continuation is
        driven entirely by the deterministic :class:`RoundPlan` and the
        poor-result check.
        """

        state = ctx.state
        limits = ctx.limits

        all_entities = [*key_entities, *supplement_entities]
        classification = await classify_entities(
            self.entity_classifier, sub_question, all_entities, list(domains)
        )
        if classification.total == 0:
            # Nothing to group: degrade to one plain round (legacy shape).
            emit_event(
                "tool_result",
                module="m2",
                tool="m2_round_plan",
                status="warning",
                message="子问题没有可用实体，实体分组轮次策略降级为单轮普通检索",
                details={"sub_question": sub_question},
            )
            planner_kwargs: dict = {}
            if question_type:
                planner_kwargs["question_type"] = question_type
            if supplement_entities:
                planner_kwargs["supplement_entities"] = list(supplement_entities)
            try:
                successful = await self._execute_round(
                    ctx,
                    measure,
                    mark_time_budget_expired,
                    deadline_started_at,
                    sub_question,
                    key_entities,
                    domains,
                    planner_kwargs,
                )
            except _RoundHalt as halt:
                return halt.stop_reason
            if successful == 0 and not ctx.ranked:
                return StopReason.ERROR
            if state.round_index >= limits.max_rounds:
                return StopReason.MAX_ROUNDS
            if ctx.coverage.sufficient:
                return StopReason.COVERAGE_SATISFIED
            return StopReason.PLAN_COMPLETE

        plan = await plan_entity_rounds(
            self.entity_classifier, sub_question, classification
        )

        # Assemble the executable round queue: base rounds (Case A keeps its
        # must-only follow-up in reserve and only enters it on poor results).
        pending: List[tuple[RoundSpec, str]] = []
        if plan.case == "A":
            pending.append((plan.rounds[0], "base"))
            must_only = list(plan.rescue_focus)
            if must_only and must_only != list(plan.rounds[0].focus_entities):
                pending.append(
                    (RoundSpec(focus_entities=must_only, label="must_only"), "case_a_followup")
                )
        else:
            pending.extend((spec, "base") for spec in plan.rounds)
        rescue = (
            RoundSpec(focus_entities=list(plan.rescue_focus), label="rescue_must_only"),
            "rescue",
        )
        max_rounds = max(1, limits.max_rounds)

        executed = 0
        rescue_used = False
        last_poor = ""
        prev_poor = False
        all_failed = False
        while pending and executed < max_rounds:
            spec, kind = pending.pop(0)
            if kind == "case_a_followup" and (not prev_poor or rescue_used):
                # Case A's must-only round is reserved for poor results, and
                # it never duplicates an already-executed rescue round.
                continue
            state.remaining_budget = calculate_remaining(limits, state)
            if state.remaining_budget.max_rounds <= 0:
                return StopReason.MAX_ROUNDS
            planner_kwargs = {"focus_entities": list(spec.focus_entities)}
            if question_type:
                planner_kwargs["question_type"] = question_type
            if supplement_entities:
                planner_kwargs["supplement_entities"] = list(supplement_entities)
            try:
                successful = await self._execute_round(
                    ctx,
                    measure,
                    mark_time_budget_expired,
                    deadline_started_at,
                    sub_question,
                    key_entities,
                    domains,
                    planner_kwargs,
                )
            except _RoundHalt as halt:
                return halt.stop_reason
            executed += 1
            all_failed = successful == 0
            current_must_only = list(spec.focus_entities) == list(
                plan.rescue_focus
            )

            round_counts: Dict[str, int] = {}
            for query in state.queries_used:
                if query.round_index != state.round_index:
                    continue
                if query.target_source in ctx.failed_sources:
                    # Failed sources did not return zero results — skip them.
                    continue
                round_counts.setdefault(query.target_source.casefold(), 0)
            for source, count in ctx.source_result_counts.items():
                if "->" in source:
                    continue
                key = source.casefold()
                if key in round_counts:
                    round_counts[key] += count - ctx.round_start_counts.get(source, 0)
            round_total = ctx.papers_found - ctx.round_start_papers_found
            poor = poor_round_reason(round_counts, round_total)
            last_poor = poor
            prev_poor = bool(poor)
            if poor and kind != "rescue" and not rescue_used:
                if current_must_only:
                    # This round already searched the must entities only;
                    # a rescue would repeat it verbatim (at-most-once rule).
                    continue
                if plan.rescue_focus:
                    rescue_used = True
                    pending.insert(0, rescue)
                    emit_event(
                        "tool_result",
                        module="m2",
                        tool="m2_zero_result_rescue",
                        status="warning",
                        message=(
                            f"第 {state.round_index} 轮检索结果不佳（{poor}），"
                            "触发零结果补救：追加仅必须实体轮"
                        ),
                        details={
                            "sub_question": sub_question,
                            "round": state.round_index,
                            "reason": poor,
                            "rescue_focus": list(plan.rescue_focus),
                            "round_source_counts": dict(round_counts),
                        },
                    )
                else:
                    emit_event(
                        "tool_result",
                        module="m2",
                        tool="m2_zero_result_rescue",
                        status="warning",
                        message=(
                            f"第 {state.round_index} 轮检索结果不佳（{poor}），"
                            "但无必须实体可用于补救，跳过补救轮"
                        ),
                        details={
                            "sub_question": sub_question,
                            "round": state.round_index,
                            "reason": poor,
                        },
                    )

        if all_failed and not ctx.ranked:
            return StopReason.ERROR
        state.remaining_budget = calculate_remaining(limits, state)
        if state.queries_executed >= limits.max_queries:
            return StopReason.QUERY_BUDGET
        if state.unique_papers_seen >= limits.max_papers:
            return StopReason.PAPER_BUDGET
        if state.estimated_tokens_used >= limits.max_tokens:
            return StopReason.TOKEN_BUDGET
        if state.elapsed_seconds >= limits.max_seconds:
            return StopReason.TIME_BUDGET
        if state.round_index >= limits.max_rounds:
            return StopReason.MAX_ROUNDS
        if last_poor and not ctx.ranked:
            return StopReason.NO_RESULTS
        if ctx.coverage.sufficient:
            return StopReason.COVERAGE_SATISFIED
        return StopReason.PLAN_COMPLETE

    async def _execute_round(
        self,
        ctx: _RoundContext,
        measure,
        mark_time_budget_expired,
        deadline_started_at: float,
        sub_question: str,
        key_entities: Sequence[str],
        domains: Sequence[str],
        planner_kwargs: dict,
    ) -> int:
        """Execute one planner→search→dedup→rank→scout round.

        Returns the number of successfully executed queries.  Raises
        :class:`_RoundHalt` when a stage failure or budget expiry must end
        the whole run (the stop reason is carried on the exception).
        """

        state = ctx.state
        limits = ctx.limits
        ctx.round_start_counts = dict(ctx.source_result_counts)
        ctx.round_start_papers_found = ctx.papers_found
        try:
            planned = await measure(
                "query_planner",
                self.query_planner.plan(
                    sub_question,
                    key_entities=key_entities,
                    domains=domains,
                    state=state.model_copy(deep=True),
                    **planner_kwargs,
                ),
            )
        except _SearchTimeBudgetExpired as exc:
            raise _RoundHalt(mark_time_budget_expired(exc)) from exc
        except Exception as exc:
            ctx.errors.append(self._format_error("query_planner", exc))
            raise _RoundHalt(StopReason.ERROR) from exc

        round_index = state.round_index + 1
        queries: list[SearchQuery] = []
        for planned_query in planned:
            key = self._query_key(planned_query)
            if key in ctx.query_history:
                continue
            ctx.query_history.add(key)
            if planned_query.target_source in state.unavailable_sources:
                continue
            queries.append(
                planned_query.model_copy(update={"round_index": round_index})
            )
            if len(queries) >= state.remaining_budget.max_queries:
                break

        if not queries:
            ctx.errors.append("query planner produced no new executable queries")
            raise _RoundHalt(StopReason.NO_RESULTS)
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
            raise _RoundHalt(mark_time_budget_expired(exc)) from exc
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
                if key in ctx.query_history:
                    continue
                ctx.query_history.add(key)
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
                if source_name not in ctx.failed_sources:
                    ctx.failed_sources.append(source_name)
                state.unavailable_sources.add(source_name)
                ctx.errors.append(self._format_error(source_name, result))
                continue
            successful_queries += 1
            ctx.source_result_counts[source_name] = (
                ctx.source_result_counts.get(source_name, 0) + len(result)
            )
            for paper in result:
                providers = {
                    provider.casefold() for provider in paper.sources
                }
                if providers and source_name.casefold() not in providers:
                    for provider in sorted(providers):
                        fallback_key = f"{source_name}->{provider}"
                        ctx.source_result_counts[fallback_key] = (
                            ctx.source_result_counts.get(fallback_key, 0) + 1
                        )
            raw_papers.extend(result)

        ctx.papers_found += len(raw_papers)
        emit_event(
            "tool_result",
            module="m2",
            tool="source_search",
            status="completed",
            message=f"第 {round_index} 轮检索获得 {len(raw_papers)} 篇记录",
            details={
                "round": round_index,
                "source_result_counts": dict(ctx.source_result_counts),
                "failed_sources": list(ctx.failed_sources),
            },
        )
        ctx.all_queries.extend(queries)
        state.queries_used = list(ctx.all_queries)
        state.queries_executed += len(queries)
        previous_ids = set(ctx.canonical_history)
        try:
            canonical_batch = await measure(
                "paper_deduplicator",
                self.deduplicator.deduplicate(
                    raw_papers,
                    existing_papers=[
                        *ctx.existing_papers,
                        *ctx.canonical_history.values(),
                    ],
                ),
            )
        except _SearchTimeBudgetExpired as exc:
            raise _RoundHalt(mark_time_budget_expired(exc)) from exc
        except Exception as exc:
            ctx.errors.append(self._format_error("paper_deduplicator", exc))
            raise _RoundHalt(StopReason.ERROR) from exc
        for paper in canonical_batch:
            if (
                paper.paper_id not in ctx.canonical_history
                and len(ctx.canonical_history) >= limits.max_papers
            ):
                continue
            ctx.canonical_history[paper.paper_id] = paper
            ctx.candidate_pool[paper.paper_id] = paper
            if paper.paper_id in ctx.existing_ids:
                ctx.reused_ids.add(paper.paper_id)
        emit_event(
            "tool_result",
            module="m2",
            tool="paper_deduplicator",
            status="completed",
            message=f"去重后累计 {len(ctx.canonical_history)} 篇论文",
            details={"round": round_index, "papers": len(ctx.canonical_history)},
        )

        try:
            ranked = await measure(
                "paper_ranker",
                self.ranker.rank(
                    ctx.alignment_question,
                    list(ctx.candidate_pool.values()),
                    limit=min(self.candidate_limit, limits.max_papers),
                ),
            )
        except _SearchTimeBudgetExpired as exc:
            raise _RoundHalt(mark_time_budget_expired(exc)) from exc
        except Exception as exc:
            ctx.errors.append(self._format_error("paper_ranker", exc))
            raise _RoundHalt(StopReason.ERROR) from exc
        ctx.candidate_pool = {paper.paper_id: paper for paper in ranked}
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
        new_papers = len(set(ctx.canonical_history) - previous_ids)
        papers_needing_scout = [
            paper for paper in ranked if paper.paper_id not in ctx.scout_by_paper
        ]
        semantic_pool = papers_needing_scout[: self.scout_candidate_limit]
        lexical_tail = papers_needing_scout[self.scout_candidate_limit :]
        try:
            new_scout_notes = await asyncio.wait_for(measure(
                "scout_reader",
                self.scout_reader.read(ctx.alignment_question, semantic_pool),
            ), timeout=self.scout_timeout_seconds)
        except asyncio.TimeoutError:
            from .scout import ScoutReader

            ctx.errors.append(
                f"scout_reader timed out after {self.scout_timeout_seconds}s; "
                "used lexical fallback"
            )
            new_scout_notes = await ScoutReader(None).read(
                ctx.alignment_question, semantic_pool
            )
        except _SearchTimeBudgetExpired as exc:
            raise _RoundHalt(mark_time_budget_expired(exc)) from exc
        except Exception as exc:
            ctx.errors.append(self._format_error("scout_reader", exc))
            raise _RoundHalt(StopReason.ERROR) from exc
        if lexical_tail:
            from .scout import ScoutReader

            new_scout_notes = [
                *new_scout_notes,
                *await ScoutReader(None).read(ctx.alignment_question, lexical_tail),
            ]
        ctx.scout_by_paper.update(
            {note.paper_id: note for note in new_scout_notes}
        )
        skipped_scout_ids = [
            paper.paper_id
            for paper in papers_needing_scout
            if paper.paper_id not in ctx.scout_by_paper
        ]
        if skipped_scout_ids:
            emit_event(
                "tool_result",
                module="m2",
                tool="scout_reader",
                status="warning",
                message=f"快速阅读失败，已跳过 {len(skipped_scout_ids)} 篇论文",
                details={"paper_ids": skipped_scout_ids, "round": round_index},
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
            ctx.scout_by_paper[paper.paper_id]
            for paper in ranked
            if paper.paper_id in ctx.scout_by_paper
        ]
        ranked = [
            paper for paper in ranked if paper.paper_id in ctx.scout_by_paper
        ]
        ranked = rerank_with_scout(
            ranked,
            scout_notes,
            selection_limit=self.final_k,
        )
        ctx.ranked = ranked
        ctx.candidate_pool = {paper.paper_id: paper for paper in ranked}
        scout_notes = [
            ctx.scout_by_paper[paper.paper_id]
            for paper in ranked
            if paper.paper_id in ctx.scout_by_paper
        ]

        state.round_index = round_index
        state.candidate_paper_ids = list(ctx.candidate_pool)
        state.unique_papers_seen = len(ctx.canonical_history)
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
        state.elapsed_seconds = max(0.0, self.clock() - ctx.run_started_at)
        return successful_queries

    async def _finalize_search(
        self,
        ctx: _RoundContext,
        sub_question: str,
        key_entities: Sequence[str],
        existing_ids: set,
        started_at: float,
        stop_reason: StopReason,
    ) -> SearchRunResult:
        """Retention judging and result packaging shared by all strategies."""

        state = ctx.state
        ranked = ctx.ranked
        applicable_finalists = [
            paper
            for paper in ranked
            if paper.rank_scores.get("semantic_not_applicable", 0.0) < 0.5
        ]
        contextual_fallback = not applicable_finalists and bool(ranked)
        if contextual_fallback:
            # A frontier question often has no paper that answers the complete
            # relation.  In that case preserve the best adjacent papers as
            # contextual/method evidence instead of collapsing the search to
            # zero.  Scout scores still determine their order and provenance.
            applicable_finalists = list(ranked)

        # Domain token overlap is a weak diagnostic prior, NOT a hard reject.
        # Literal token mismatch can produce false negatives (e.g. "robotics"
        # vs "robotic", "machine learning" vs "neural network").
        if state.domains:
            for paper in applicable_finalists:
                if domain_token_overlap(paper, state.domains) == 0.0:
                    paper.rank_scores["domain_token_mismatch"] = True
        # Note: we do NOT re-filter based on domain_token_mismatch alone.
        # Scout's semantic applicability judgment is the definitive gate.
        final_papers, applicable_decisions = select_retained_papers(
            applicable_finalists,
            list(ctx.scout_by_paper.values()),
            final_k=self.final_k,
        )
        if self.retention_judge_client is not None:
            final_papers, applicable_decisions = (
                await self._review_retention_boundary(
                    applicable_finalists,
                    applicable_decisions,
                    list(ctx.scout_by_paper.values()),
                    sub_question,
                    key_entities,
                )
            )
        if not final_papers and applicable_finalists:
            # The boundary judge may legitimately find no direct evidence, but
            # M2 must still expose the strongest adjacent literature to M3.
            # Directness remains visible in Scout scores; it is not an
            # all-or-nothing retention gate.
            final_papers = list(applicable_finalists[: self.final_k])
            fallback_ids = {paper.paper_id for paper in final_papers}
            for decision in applicable_decisions:
                if decision.paper_id not in fallback_ids:
                    continue
                decision.decision = "retain"
                if "context_evidence" not in decision.roles:
                    decision.roles.append("context_evidence")
                decision.reason = (
                    "contextual fallback: retained the strongest adjacent "
                    "paper because no direct paper survived the boundary review"
                )
            emit_event(
                "tool_result",
                module="m2",
                tool="retention_fallback",
                status="warning",
                message=(
                    f"未找到直接命中文献，保留 {len(final_papers)} 篇最相关的"
                    "方法/背景论文"
                ),
                details={
                    "papers": len(final_papers),
                    "all_scout_not_applicable": contextual_fallback,
                    "paper_ids": [paper.paper_id for paper in final_papers],
                },
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
            ctx.errors.append(
                "no ranked paper could be retained, including contextual fallback"
            )
        return SearchRunResult(
            sub_question=sub_question,
            queries=ctx.all_queries,
            papers_found=ctx.papers_found,
            papers_after_dedup=len(ctx.canonical_history),
            candidates=ranked,
            final_papers=final_papers,
            coverage=ctx.coverage,
            failed_sources=ctx.failed_sources,
            iterations=state.round_index,
            stop_reason=stop_reason,
            errors=ctx.errors,
            source_result_counts=ctx.source_result_counts,
            stage_elapsed_seconds=ctx.stage_elapsed_seconds,
            scout_notes=list(ctx.scout_by_paper.values()),
            reused_paper_ids=[
                paper.paper_id for paper in ranked if paper.paper_id in ctx.reused_ids
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
            "Return one ruling for every candidate. Distinguish the portfolio "
            "role of conventional/core evidence, methodological evidence, "
            "contradicting evidence, and recent/time-sensitive evidence. "
            "Retain a paper when it contributes a role or direct finding that "
            "the current retained set would otherwise miss."
        )
        import json as _json
        judge_system = (
            "You are the final LLM retention judge for a scientific literature "
            "search. Audit only the supplied boundary candidates. Use the title, "
            "abstract-derived Scout signals, directness, and explicit portfolio "
            "roles. Frontier questions often have no paper that answers the full "
            "question. Low directness alone is not a reason to reject a paper: "
            "retain useful methodological, review, background, component-level, "
            "or adjacent evidence and label its role honestly. Reject papers only "
            "when they concern a clearly different object/domain or add no useful "
            "evidence beyond generic vocabulary or duplication. "
            "Output JSON matching the supplied schema only."
        )
        output_schema = {
            "type": "object",
            "properties": {
                "rulings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "paper_id": {"type": "string"},
                            "decision": {"type": "string", "enum": ["retain", "reject"]},
                            "roles": {"type": "array", "items": {"type": "string"}},
                            "rationale": {"type": "string"},
                        },
                        "required": ["paper_id", "decision", "rationale"],
                    },
                },
            },
            "required": ["rulings"],
        }
        emit_event(
            "tool_started",
            module="m2",
            tool="retention_judge",
            status="running",
            message="M2 开始进行 LLM 论文保留裁决",
            details={"candidates": len(boundary_papers), "window_size": window_size},
        )
        try:
            structured_chat = getattr(self.retention_judge_client, "structured_chat", None)
            if callable(structured_chat):
                raw = await structured_chat(
                    system_prompt=judge_system,
                    user_prompt=prompt,
                    output_schema=output_schema,
                    max_tokens=4096,
                    temperature=0.0,
                    disable_thinking=True,
                )
            else:
                raw = await self.retention_judge_client.chat(
                    system_prompt=judge_system,
                    user_prompt=prompt,
                    max_tokens=4096,
                    temperature=0.0,
                    disable_thinking=True,
                )
        except Exception as exc:
            emit_event(
                "tool_failed",
                module="m2",
                tool="retention_judge",
                status="failed",
                message=f"LLM 论文保留裁决失败: {type(exc).__name__}: {exc}",
                details={"candidates": len(boundary_papers)},
            )
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
            rulings = rulings.get("rulings", rulings.get("decisions", [rulings]))
        if not isinstance(rulings, list):
            raise RuntimeError(
                "Retention judge returned unexpected output type "
                f"{type(rulings).__name__}; expected a JSON array."
            )
        for ruling in rulings:
            if not isinstance(ruling, dict):
                continue
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
        emit_event(
            "tool_completed",
            module="m2",
            tool="retention_judge",
            status="completed",
            message="LLM 论文保留裁决完成",
            details={
                "candidates": len(boundary_papers),
                "retained": len(retained),
                "overrides": sum(bool(dec.llm_override) for dec in decisions),
            },
        )
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
