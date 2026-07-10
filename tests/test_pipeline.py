"""
Smoke tests for the HypoForge pipeline.

Run with::

    python -m pytest tests/test_pipeline.py -v
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

# Ensure HypoForge is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hypoforge.config import PipelineConfig
from hypoforge.pipeline import PipelineRunner


@pytest.mark.asyncio
async def test_default_config_loads():
    """Default config should load without errors."""
    config = PipelineConfig.from_defaults()
    assert config.enabled_modules == ["m1", "m2", "m3", "m4", "m5", "m6"]
    assert config.max_iterations == 3


@pytest.mark.asyncio
async def test_yaml_config_loads():
    """YAML config files should all load."""
    config_dir = Path(__file__).resolve().parent.parent / "configs"
    for yaml_file in config_dir.glob("*.yaml"):
        config = PipelineConfig.from_yaml(str(yaml_file))
        assert config.enabled_modules, f"{yaml_file.name}: no modules enabled"


@pytest.mark.asyncio
async def test_pipeline_runs_with_default_config():
    """Full pipeline (stub mode) should run end-to-end without errors."""
    config = PipelineConfig.from_defaults()
    config.verbose = False  # keep test output clean

    runner = PipelineRunner(config)
    state = await runner.run(
        question="蛋白质如何折叠及错误折叠导致疾病的机制？",
        run_id="test-001",
    )

    # Check that all expected fields are populated
    assert state.input_question != ""
    assert state.problem_card is not None, "M1 did not produce a problem card"
    assert len(state.literature_results) > 0, "M2 did not produce literature results"
    assert state.evidence_graph is not None, "M3 did not produce an evidence graph"
    assert len(state.top_hypotheses) > 0, "M4 did not produce top hypotheses"
    assert len(state.research_plans) > 0, "M5 did not produce research plans"
    assert len(state.reviews) > 0, "M6 did not produce reviews"


@pytest.mark.asyncio
async def test_baseline_b0_runs():
    """B0 baseline (M1 → M4 → M5) should run."""
    config_path = Path(__file__).resolve().parent.parent / "configs" / "baseline_b0.yaml"
    config = PipelineConfig.from_yaml(str(config_path))
    config.verbose = False

    runner = PipelineRunner(config)
    state = await runner.run(
        question="衰老的生物学基础是什么？",
        run_id="test-b0",
    )

    assert state.problem_card is not None
    assert len(state.top_hypotheses) > 0


@pytest.mark.asyncio
async def test_iteration_converges():
    """The iteration loop should terminate within max_iterations."""
    config = PipelineConfig.from_defaults()
    config.verbose = False
    config.max_iterations = 2
    config.enable_iteration = True

    runner = PipelineRunner(config)
    state = await runner.run(
        question="Test iteration convergence",
        run_id="test-iter",
    )

    assert state.iteration_count <= config.max_iterations


@pytest.mark.asyncio
async def test_scoring_produces_report():
    """The scoring pipeline should produce valid JSON."""
    from hypoforge.evaluation.scorer import score_pipeline_state_async

    config = PipelineConfig.from_defaults()
    config.verbose = False

    runner = PipelineRunner(config)
    state = await runner.run(question="Test scoring", run_id="test-score")

    report = await score_pipeline_state_async(state)
    assert report["run_id"] == "test-score"
    assert "hypothesis_scores" in report
    assert "plan_scores" in report


if __name__ == "__main__":
    # Allow running directly: python tests/test_pipeline.py
    async def _run_all():
        await test_pipeline_runs_with_default_config()
        print("[PASS] Default pipeline test")
        await test_baseline_b0_runs()
        print("[PASS] B0 baseline test")
        await test_iteration_converges()
        print("[PASS] Iteration convergence test")
        await test_scoring_produces_report()
        print("[PASS] Scoring test")
        print("\nAll 4 tests passed!")
    asyncio.run(_run_all())
