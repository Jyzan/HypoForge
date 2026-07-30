from __future__ import annotations

import pytest

from hypoforge.literature.models import (
    CoverageReport,
    SearchBudget,
    SearchState,
    StopReason,
)
from hypoforge.literature.search.budget import (
    calculate_remaining,
    choose_stop_reason,
    estimate_tokens,
)


def _choose(
    *,
    budget: SearchBudget | None = None,
    state: SearchState | None = None,
    coverage: CoverageReport | None = None,
    all_queries_failed: bool = False,
    has_candidates: bool = True,
) -> StopReason | None:
    return choose_stop_reason(
        coverage=coverage or CoverageReport(),
        budget=budget or SearchBudget(),
        state=state or SearchState(),
        all_queries_failed=all_queries_failed,
        has_candidates=has_candidates,
    )


def test_estimate_tokens_is_deterministic_and_handles_empty_text() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("12345") == 2


def test_calculate_remaining_reaches_zero_without_validation_error() -> None:
    budget = SearchBudget(
        max_rounds=2,
        max_queries=3,
        max_papers=4,
        max_tokens=5,
        max_seconds=6,
    )
    state = SearchState(
        round_index=2,
        queries_executed=3,
        unique_papers_seen=4,
        estimated_tokens_used=5,
        elapsed_seconds=6,
    )

    assert calculate_remaining(budget, state).model_dump() == {
        "max_rounds": 0,
        "max_queries": 0,
        "max_papers": 0,
        "max_tokens": 0,
        "max_seconds": 0.0,
    }


def test_coverage_has_highest_stop_priority() -> None:
    reason = _choose(
        budget=SearchBudget(max_rounds=1),
        state=SearchState(round_index=1),
        coverage=CoverageReport(sufficient=True),
        all_queries_failed=True,
        has_candidates=False,
    )

    assert reason is StopReason.COVERAGE_SATISFIED


def test_all_failed_without_candidates_is_error() -> None:
    reason = _choose(
        budget=SearchBudget(max_rounds=1),
        state=SearchState(round_index=1),
        all_queries_failed=True,
        has_candidates=False,
    )

    assert reason is StopReason.ERROR


@pytest.mark.parametrize(
    ("budget", "state", "expected"),
    [
        (
            SearchBudget(max_rounds=2),
            SearchState(round_index=2),
            StopReason.MAX_ROUNDS,
        ),
        (
            SearchBudget(max_rounds=3, max_queries=2),
            SearchState(round_index=1, queries_executed=2),
            StopReason.QUERY_BUDGET,
        ),
        (
            SearchBudget(max_rounds=3, max_papers=2),
            SearchState(round_index=1, unique_papers_seen=2),
            StopReason.PAPER_BUDGET,
        ),
        (
            SearchBudget(max_rounds=3, max_tokens=10),
            SearchState(round_index=1, estimated_tokens_used=10),
            StopReason.TOKEN_BUDGET,
        ),
        (
            SearchBudget(max_rounds=3, max_seconds=5),
            SearchState(round_index=1, elapsed_seconds=5),
            StopReason.TIME_BUDGET,
        ),
    ],
)
def test_hard_limits_choose_their_structured_reason(
    budget: SearchBudget,
    state: SearchState,
    expected: StopReason,
) -> None:
    assert _choose(budget=budget, state=state) is expected


def test_two_empty_rounds_stop_with_no_results() -> None:
    state = SearchState(round_index=2, consecutive_no_result_rounds=2)

    assert _choose(state=state, has_candidates=False) is StopReason.NO_RESULTS


def test_two_low_gain_rounds_stop_with_low_marginal_gain() -> None:
    state = SearchState(round_index=2, consecutive_low_gain_rounds=2)

    assert _choose(state=state) is StopReason.LOW_MARGINAL_GAIN


def test_no_stop_reason_allows_another_round() -> None:
    state = SearchState(round_index=1, unique_papers_seen=3)

    assert _choose(state=state) is None
