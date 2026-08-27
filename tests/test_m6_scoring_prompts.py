import pytest

from hypoforge.evaluation.m6_scoring import (
    PlanQualityAssessment,
    SemanticScoreAssessment,
)
from hypoforge.prompts.m6_prompts import (
    M6_NOVELTY_REVIEW_SYSTEM,
    M6_PLAN_QUALITY_SYSTEM,
)


def test_novelty_prompt_requires_corpus_comparison_and_coverage_caution():
    prompt = M6_NOVELTY_REVIEW_SYSTEM.lower()
    assert "retrieved corpus" in prompt
    assert "insufficient coverage" in prompt
    assert "must not" in prompt and "novel" in prompt
    assert all(anchor in prompt for anchor in ["1.0", "3.0", "5.0"])


def test_plan_quality_schema_covers_all_prompt_scored_dimensions():
    assert set(PlanQualityAssessment.model_fields) == {
        "task_coverage",
        "evidence_reliability",
        "testability",
        "experimental_rigor",
        "statistics_reproducibility",
        "technical_feasibility",
    }


def test_semantic_score_payload_rejects_out_of_range_scores():
    with pytest.raises(ValueError):
        SemanticScoreAssessment(score=5.1, confidence=1.0)


def test_plan_quality_schema_rejects_missing_dimension():
    item = SemanticScoreAssessment(score=3.0, confidence=0.5)
    with pytest.raises(ValueError):
        PlanQualityAssessment(
            task_coverage=item,
            evidence_reliability=item,
            testability=item,
            experimental_rigor=item,
            statistics_reproducibility=item,
        )


def test_plan_quality_prompt_prevents_field_presence_from_earning_full_marks():
    prompt = " ".join(M6_PLAN_QUALITY_SYSTEM.lower().split())
    assert "original question" in prompt
    assert "full task breadth" in prompt
    assert "source independence" in prompt
    assert "alternative explanations" in prompt
    assert "non-empty" in prompt and "5.0" in prompt


def test_plan_quality_prompt_treats_2026_as_the_current_cutoff_year():
    prompt = " ".join(M6_PLAN_QUALITY_SYSTEM.lower().split())
    assert "completion cutoff year is 2026" in prompt
    assert "must not treat a 2026 publication year as future-dated" in prompt
