import pytest

from hypoforge.evaluation.m6_scoring import (
    M6ScoreConditions,
    aggregate_m6_scoring,
)
from hypoforge.state import ScoreDimensionDetail


WEIGHTS = {
    "task_coverage": 0.10,
    "novelty": 0.15,
    "scientific_logic": 0.15,
    "evidence_reliability": 0.15,
    "testability": 0.10,
    "experimental_rigor": 0.15,
    "statistics_reproducibility": 0.10,
    "technical_feasibility": 0.10,
}


def _detail(name: str, score: float) -> ScoreDimensionDetail:
    weight = WEIGHTS[name]
    return ScoreDimensionDetail(
        dimension=name,
        score=score,
        weight=weight,
        weighted_contribution=round(score * weight, 4),
        source="deterministic",
        confidence=1.0,
    )


def test_weighted_score_uses_all_eight_dimensions():
    rows = [
        _detail(name, score)
        for name, score in {
            "task_coverage": 4.0,
            "novelty": 2.0,
            "scientific_logic": 4.0,
            "evidence_reliability": 4.0,
            "testability": 4.0,
            "experimental_rigor": 3.0,
            "statistics_reproducibility": 3.0,
            "technical_feasibility": 4.0,
        }.items()
    ]

    summary = aggregate_m6_scoring(rows, M6ScoreConditions())

    assert summary.raw_score == 3.45
    assert summary.final_score == 3.6
    assert len(summary.dimensions) == 8


@pytest.mark.parametrize(
    ("raw_dimension_score", "expected_display_score"),
    [
        (2.0, 1.9),
        (3.0, 3.0),
        (3.6, 3.8),
        (3.7, 4.0),
        (3.8, 4.2),
        (4.0, 4.4),
        (5.0, 5.0),
    ],
)
def test_display_score_uses_historical_anchor_calibration(
    raw_dimension_score,
    expected_display_score,
):
    rows = [_detail(name, raw_dimension_score) for name in WEIGHTS]

    summary = aggregate_m6_scoring(rows, M6ScoreConditions())

    assert summary.raw_score == raw_dimension_score
    assert summary.final_score == expected_display_score


@pytest.mark.parametrize(
    ("precise_raw_score", "expected_display_score"),
    [
        (3.55, 3.7),
        (3.65, 3.9),
        (3.75, 4.1),
    ],
)
def test_precise_weighted_score_keeps_intermediate_bands_reachable(
    precise_raw_score,
    expected_display_score,
):
    rows = [_detail(name, precise_raw_score) for name in WEIGHTS]

    summary = aggregate_m6_scoring(rows, M6ScoreConditions())

    assert summary.raw_score == precise_raw_score
    assert summary.final_score == expected_display_score


def test_low_novelty_caps_an_otherwise_perfect_score_at_3_9():
    rows = [
        _detail(name, 1.9 if name == "novelty" else 5.0)
        for name in WEIGHTS
    ]

    summary = aggregate_m6_scoring(rows, M6ScoreConditions())

    assert summary.raw_score > 4.0
    assert summary.final_score == 3.9
    assert [cap.rule_id for cap in summary.applied_caps] == ["low_novelty"]


def test_multiple_caps_use_the_strictest_limit_and_preserve_reasons():
    rows = [_detail(name, 4.0) for name in WEIGHTS]
    conditions = M6ScoreConditions(
        task_misaligned=True,
        core_fact_contradicted=True,
        experimental_validation_missing=True,
    )

    summary = aggregate_m6_scoring(rows, conditions)

    assert summary.final_score == 1.9
    assert {cap.rule_id for cap in summary.applied_caps} == {
        "task_misaligned",
        "core_fact_contradicted",
        "experimental_validation_missing",
    }


def test_invalid_fallback_output_caps_an_otherwise_high_score_at_1_9():
    rows = [_detail(name, 4.0) for name in WEIGHTS]

    summary = aggregate_m6_scoring(
        rows,
        M6ScoreConditions(invalid_hypothesis_output=True),
    )

    assert summary.raw_score == 4.0
    assert summary.final_score == 1.9
    assert [cap.rule_id for cap in summary.applied_caps] == [
        "invalid_hypothesis_output"
    ]


def test_aggregator_rejects_missing_or_duplicate_dimensions():
    rows = [_detail(name, 4.0) for name in WEIGHTS if name != "novelty"]

    with pytest.raises(ValueError, match="exactly one"):
        aggregate_m6_scoring(rows, M6ScoreConditions())

    duplicate_rows = rows + [_detail("task_coverage", 4.0)]
    with pytest.raises(ValueError, match="exactly one"):
        aggregate_m6_scoring(duplicate_rows, M6ScoreConditions())
