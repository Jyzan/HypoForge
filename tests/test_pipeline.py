"""
Offline unit tests for HypoForge (no LLM / no network required).

The stub module modes were removed, so end-to-end integration is now exercised
by ``scripts/smoke_pipeline.py`` (which runs the real M1→M6 pipeline and needs
an API key).  These tests cover pure logic + the scorer against a hand-built
``PipelineState`` fixture.

Run::

    python -m pytest tests/test_pipeline.py -q
    python tests/test_pipeline.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

# Ensure HypoForge is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hypoforge.config import PipelineConfig
from hypoforge.state import (
    HypothesisCard,
    PipelineState,
    ResearchPlan,
    ReviewResult,
    ReviewerDimension,
)


# --------------------------------------------------------------------------- #
# Config loading
# --------------------------------------------------------------------------- #

def test_default_config_loads():
    """Default config should load without errors."""
    config = PipelineConfig.from_defaults()
    assert config.enabled_modules == ["m1", "m2", "m3", "m4", "m5", "m6"]
    assert config.max_iterations == 3


def test_yaml_config_loads():
    """Every YAML config file should load."""
    config_dir = Path(__file__).resolve().parent.parent / "configs"
    for yaml_file in config_dir.glob("*.yaml"):
        config = PipelineConfig.from_yaml(str(yaml_file))
        assert config.enabled_modules, f"{yaml_file.name}: no modules enabled"


# --------------------------------------------------------------------------- #
# Rubric — single source of truth for weights / composite
# --------------------------------------------------------------------------- #

def test_rubric_composite_and_weights():
    from hypoforge.evaluation.rubric import composite_score, normalise_weights

    # default weights sum to 1.0
    assert abs(sum(normalise_weights(None).values()) - 1.0) < 1e-9

    # all-equal dimensions → composite equals that value
    scores = {
        "novelty": 0.8, "scientific_soundness": 0.8,
        "testability": 0.8, "evidence_consistency": 0.8,
    }
    assert abs(composite_score(scores) - 0.8) < 1e-6

    # missing dimensions renormalise over what's present (no zero-drag)
    assert abs(composite_score({"novelty": 0.6, "scientific_soundness": 0.6}) - 0.6) < 1e-6


# --------------------------------------------------------------------------- #
# Scorer — tested against a hand-built state (no pipeline run, no LLM)
# --------------------------------------------------------------------------- #

def _make_scored_state() -> PipelineState:
    """A completed-looking PipelineState fixture for scorer tests."""
    h1 = HypothesisCard(
        hypothesis_id="H1",
        statement="NAD+/NADH ratio regulates Hsp70 ATPase cycling in aged cells.",
        observable_predictions=["NMN restores folding activity in aged cells"],
        falsification_conditions=["no change in Hsp70 activity after NMN"],
        scores={"novelty": 0.8, "scientific_soundness": 0.8, "testability": 0.8,
                "evidence_consistency": 0.8, "composite": 0.8},
    )
    h2 = HypothesisCard(
        hypothesis_id="H2",
        statement="Stress granules buffer proteotoxicity by enriching Hsp70.",
        scores={"novelty": 0.6, "scientific_soundness": 0.6, "testability": 0.6,
                "evidence_consistency": 0.6, "composite": 0.6},
    )
    p1 = ResearchPlan(
        hypothesis_id="H1",
        study_subjects="C57BL/6 mice, 8 weeks old",
        timeline="12 months across 3 phases",
    )
    reviews = [ReviewResult(dimension=ReviewerDimension("overall"), score=4.1, version=1)]
    return PipelineState(
        run_id="fixture-001", input_question="q",
        top_hypotheses=[h1, h2], research_plans=[p1], reviews=reviews, iteration_count=1,
    )


@pytest.mark.asyncio
async def test_scorer_report_structure_and_not_circular():
    """Report must separate self-reported from independent, and independent
    metrics must NOT echo the generator's self-reported scores."""
    from hypoforge.evaluation.scorer import score_pipeline_state_async

    report = await score_pipeline_state_async(_make_scored_state())
    assert report["run_id"] == "fixture-001"
    for key in ("hypothesis_scores", "plan_scores", "aggregate", "rubric"):
        assert key in report

    h0 = report["hypothesis_scores"][0]
    assert {"self_reported", "independent", "composite"} <= set(h0)
    # The metrics are now implemented, so they appear in independent.
    # Verify they don't echo the self-reported 0.8 score.
    assert "novelty" in h0["independent"]
    assert h0["independent"]["novelty"] != 0.8
    assert "evidence_consistency" in h0["independent"]
    assert h0["independent"]["evidence_consistency"] != 0.8
    assert "testability" in h0["independent"]              # objectively recomputed
    assert abs(sum(report["rubric"]["weights"].values()) - 1.0) < 1e-6


@pytest.mark.asyncio
async def test_save_scoring_report_writes_file():
    import json
    from hypoforge.evaluation.scorer import save_scoring_report_async

    path = await save_scoring_report_async(_make_scored_state(), "./output")
    assert Path(path).exists(), "scores report was not written"
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    assert data["run_id"] == "fixture-001"
    assert "aggregate" in data


def test_plan_completeness_placeholder_aware():
    """Placeholders / sub-3-char stubs do not count as filled."""
    from hypoforge.evaluation.scorer import score_plan_completeness

    assert score_plan_completeness(ResearchPlan()) == 0.0
    placeholder = ResearchPlan(study_subjects="TBD", timeline="待定", expected_results_if_supported="x")
    assert score_plan_completeness(placeholder) == 0.0
    real = ResearchPlan(study_subjects="C57BL/6 mice, 8 weeks old", timeline="12 months across 3 phases")
    assert 0.0 < score_plan_completeness(real) < 1.0


# --------------------------------------------------------------------------- #
# M4 feedback loop + reason-before-score schema
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_m4_feedback_loop_and_no_hang():
    """Revision rounds build a feedback block from reviews + guidance; interactive
    is off by default so nothing blocks on stdin."""
    from hypoforge.modules.m4_hypothesis_generation import M4HypothesisGeneration

    m4 = M4HypothesisGeneration()  # no client — only helper methods are exercised
    assert m4.interactive is False

    # first round → no feedback block
    assert m4._build_feedback_context(PipelineState(input_question="q"), []) == ""

    # revision round → block cites reviewer suggestions + user guidance
    state = PipelineState(
        input_question="q",
        iteration_count=1,
        reviews=[ReviewResult(
            dimension=ReviewerDimension("method_feasibility"),
            score=3.0, suggestions="add a power analysis", version=1,
        )],
    )
    block = m4._build_feedback_context(state, ["prefer in-vivo models"])
    assert "power analysis" in block
    assert "prefer in-vivo models" in block

    # prompting is a no-op when non-interactive (must not read stdin / hang)
    assert await m4._prompt_user_guidance(state) == ""


def test_review_result_reason_before_score():
    """The reasoning field must precede score so the judge reasons first."""
    fields = list(ReviewResult.model_fields.keys())
    assert "reasoning" in fields
    assert fields.index("reasoning") < fields.index("score")


if __name__ == "__main__":
    # Allow running directly: python tests/test_pipeline.py
    async def _run_all():
        test_default_config_loads()
        print("[PASS] Default config loads")
        test_yaml_config_loads()
        print("[PASS] YAML configs load")
        test_rubric_composite_and_weights()
        print("[PASS] Rubric composite/weights")
        await test_scorer_report_structure_and_not_circular()
        print("[PASS] Scorer report structure / non-circular")
        await test_save_scoring_report_writes_file()
        print("[PASS] scores.json persistence")
        test_plan_completeness_placeholder_aware()
        print("[PASS] Plan completeness placeholder-aware")
        await test_m4_feedback_loop_and_no_hang()
        print("[PASS] M4 feedback loop / no-hang")
        test_review_result_reason_before_score()
        print("[PASS] Reason-before-score field order")
        print("\nAll offline unit tests passed!")
    asyncio.run(_run_all())
