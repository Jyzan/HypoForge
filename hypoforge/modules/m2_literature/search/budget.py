"""Deterministic budget accounting and stopping rules for literature search."""

from __future__ import annotations

import math
from typing import Optional

from ..models import (
    CoverageReport,
    RemainingSearchBudget,
    SearchBudget,
    SearchState,
    StopReason,
)


def estimate_tokens(text: str) -> int:
    """Estimate context size deterministically without a model tokenizer."""
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def calculate_remaining(
    budget: SearchBudget,
    state: SearchState,
) -> RemainingSearchBudget:
    """Return a non-negative snapshot of every remaining budget dimension."""
    return RemainingSearchBudget(
        max_rounds=max(0, budget.max_rounds - state.round_index),
        max_queries=max(0, budget.max_queries - state.queries_executed),
        max_papers=max(0, budget.max_papers - state.unique_papers_seen),
        max_tokens=max(0, budget.max_tokens - state.estimated_tokens_used),
        max_seconds=max(0.0, budget.max_seconds - state.elapsed_seconds),
    )


def choose_stop_reason(
    *,
    coverage: CoverageReport,
    budget: SearchBudget,
    state: SearchState,
    all_queries_failed: bool,
    has_candidates: bool,
    no_result_round_limit: int = 2,
    low_gain_round_limit: int = 2,
) -> Optional[StopReason]:
    """Choose the single highest-priority reason to end the search."""
    if coverage.sufficient:
        return StopReason.COVERAGE_SATISFIED
    if all_queries_failed and not has_candidates:
        return StopReason.ERROR
    if state.round_index >= budget.max_rounds:
        return StopReason.MAX_ROUNDS
    if state.queries_executed >= budget.max_queries:
        return StopReason.QUERY_BUDGET
    if state.unique_papers_seen >= budget.max_papers:
        return StopReason.PAPER_BUDGET
    if state.estimated_tokens_used >= budget.max_tokens:
        return StopReason.TOKEN_BUDGET
    if state.elapsed_seconds >= budget.max_seconds:
        return StopReason.TIME_BUDGET
    if state.consecutive_no_result_rounds >= no_result_round_limit:
        return StopReason.NO_RESULTS
    if state.consecutive_low_gain_rounds >= low_gain_round_limit:
        return StopReason.LOW_MARGINAL_GAIN
    return None
