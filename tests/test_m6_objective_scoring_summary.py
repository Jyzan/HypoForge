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

    assert summary.raw_score == 3.5
    assert summary.final_score == 3.5
    assert len(summary.dimensions) == 8


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


def test_aggregator_rejects_missing_or_duplicate_dimensions():
    rows = [_detail(name, 4.0) for name in WEIGHTS if name != "novelty"]

    with pytest.raises(ValueError, match="exactly one"):
        aggregate_m6_scoring(rows, M6ScoreConditions())

    duplicate_rows = rows + [_detail("task_coverage", 4.0)]
    with pytest.raises(ValueError, match="exactly one"):
        aggregate_m6_scoring(duplicate_rows, M6ScoreConditions())
