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
from pydantic import ValidationError

# Ensure HypoForge is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hypoforge.config import PipelineConfig
from hypoforge.pipeline import PipelineRunner, _should_continue_iterating
from hypoforge.protocol import ModuleProtocol
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
    assert config.search.implementation == "agentic"
    assert config.evaluation.embedding.model_name == "text-embedding-v3"
    assert config.evaluation.consistency.similarity_threshold == 0.5


def test_pipeline_evaluation_config_is_validated() -> None:
    config = PipelineConfig(evaluation={
        "embedding": {"model_name": "custom-embedding"},
        "consistency": {"similarity_threshold": 0.42},
    })

    assert config.evaluation.embedding.model_name == "custom-embedding"
    assert config.evaluation.consistency.similarity_threshold == 0.42
    with pytest.raises(ValidationError):
        PipelineConfig(evaluation={
            "consistency": {"similarity_threshold": 1.1},
        })


def test_agentic_search_implementation_is_explicitly_supported() -> None:
    config = PipelineConfig(search={"implementation": "agentic"})

    assert config.search.implementation == "agentic"
    assert config.get_module_kwargs("m2")["implementation"] == "agentic"
    assert config.get_module_kwargs("m2")["budget"]["max_rounds"] == 3
    assert config.get_module_kwargs("m2")["enabled_sources"] == [
        "semantic_scholar",
        "pubmed",
    ]


def test_agentic_search_config_is_forwarded_to_integrated_adapter() -> None:
    config = PipelineConfig(
        search={
            "implementation": "agentic",
            "tools": ["pubmed", "arxiv"],
            "papers_per_sub_question": 7,
            "max_papers_total": 23,
            "max_rounds": 2,
            "max_queries": 5,
            "max_tokens": 4567,
            "max_seconds": 89,
        }
    )

    kwargs = config.get_module_kwargs("m2")
    assert kwargs["enabled_sources"] == ["pubmed", "arxiv"]
    assert kwargs["per_query_limit"] == 7
    assert kwargs["budget"] == {
        "max_rounds": 2,
        "max_queries": 5,
        "max_papers": 23,
        "max_tokens": 4567,
        "max_seconds": 89,
    }


def test_module_override_cannot_reintroduce_removed_legacy_implementation() -> None:
    config = PipelineConfig(
        search={"implementation": "agentic"},
        module_overrides={"m2": {"kwargs": {"implementation": "agentic"}}},
    )

    assert config.get_module_kwargs("m2")["implementation"] == "agentic"


def test_unknown_search_implementation_is_rejected() -> None:
    with pytest.raises(ValidationError):
        PipelineConfig(search={"implementation": "automatic"})


def test_yaml_config_loads():
    """Every PipelineConfig YAML file should load.

    Skips ``evaluation.yaml`` — it uses MasterEvaluationConfig, not
    PipelineConfig (by design; see TODO Task C.5).
    """
    config_dir = Path(__file__).resolve().parent.parent / "configs"
    skip = {"evaluation.yaml"}
    for yaml_file in config_dir.glob("*.yaml"):
        if yaml_file.name in skip:
            continue
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
        reviews=[
            ReviewResult(
                dimension=ReviewerDimension("scientific_logic"),
                score=3.0,
                suggestions="clarify the causal mechanism",
                version=1,
            ),
            ReviewResult(
                dimension=ReviewerDimension("method_feasibility"),
                score=3.0,
                suggestions="add a power analysis",
                version=1,
            ),
        ],
    )
    block = m4._build_feedback_context(state, ["prefer in-vivo models"])
    assert "causal mechanism" in block
    assert "power analysis" not in block
    assert "prefer in-vivo models" in block

    # prompting is a no-op when non-interactive (must not read stdin / hang)
    assert await m4._prompt_user_guidance(state) == ""


def test_review_result_reason_before_score():
    """The reasoning field must precede score so the judge reasons first."""
    fields = list(ReviewResult.model_fields.keys())
    assert "reasoning" in fields
    assert fields.index("reasoning") < fields.index("score")


class _EmptyStandardM4(ModuleProtocol):
    module_name = "m4"
    module_version = "test"
    description = "empty standard M4"

    async def __call__(self, state, config=None):
        return {"candidate_hypotheses": [], "top_hypotheses": []}

    @classmethod
    def get_input_fields(cls):
        return []

    @classmethod
    def get_output_fields(cls):
        return ["candidate_hypotheses", "top_hypotheses"]


class _FailingStandardM5(ModuleProtocol):
    module_name = "m5"
    module_version = "test"
    description = "failing standard M5"

    async def __call__(self, state, config=None):
        raise ValueError("plan generation failed")

    @classmethod
    def get_input_fields(cls):
        return []

    @classmethod
    def get_output_fields(cls):
        return ["research_plans"]


@pytest.mark.asyncio
async def test_standard_core_module_empty_output_fails_closed(tmp_path) -> None:
    runner = PipelineRunner(PipelineConfig(verbose=False, output_dir=str(tmp_path)))
    runner._skills = []
    runner._current_run_id = "core-empty"

    with pytest.raises(RuntimeError, match="top_hypotheses"):
        await runner._make_node_wrapper("m4", _EmptyStandardM4())(
            PipelineState(input_question="q")
        )


@pytest.mark.asyncio
async def test_standard_core_module_exception_is_not_converted_to_soft_error(
    tmp_path,
) -> None:
    runner = PipelineRunner(PipelineConfig(verbose=False, output_dir=str(tmp_path)))
    runner._skills = []
    runner._current_run_id = "core-error"

    with pytest.raises(ValueError, match="plan generation failed"):
        await runner._make_node_wrapper("m5", _FailingStandardM5())(
            PipelineState(input_question="q")
        )


def test_legacy_core_error_cannot_trigger_another_iteration() -> None:
    state = PipelineState(
        input_question="q",
        iteration_count=1,
        max_iterations=3,
        errors=["[m4] contract validation failed"],
    )

    assert _should_continue_iterating(state) == "end"


def test_checkpoint_is_atomically_published_without_temp_residue(tmp_path) -> None:
    import json

    runner = PipelineRunner(PipelineConfig(verbose=False, output_dir=str(tmp_path)))
    runner._current_run_id = "atomic-checkpoint"
    state = PipelineState(input_question="q")

    nested_plan = ResearchPlan(
        hypothesis_id="H1",
        study_subjects="A recursively serialized subject",
    )
    runner._save_checkpoint("m5", state, {
        "metrics": {"complete": True},
        "research_plan_history": {1: [nested_plan]},
    })

    checkpoint = tmp_path / "atomic-checkpoint_checkpoint.json"
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert payload["_last_module"] == "m5"
    assert payload["metrics"] == {"complete": True}
    assert payload["research_plan_history"]["1"][0]["hypothesis_id"] == "H1"
    assert list(tmp_path.glob("*.tmp")) == []


@pytest.mark.asyncio
async def test_pure_resume_seed_skips_completed_modules(tmp_path) -> None:
    """seed_state without followup_text is a pure resume: the completed
    module is skipped (checkpoint semantics) and run_id is rebound."""
    config = PipelineConfig(
        verbose=False,
        output_dir=str(tmp_path),
        enabled_modules=["m1"],
    )
    seed = {
        "run_id": "parent-run",
        "input_question": "蛋白质错误折叠如何导致神经退行性疾病？",
        "problem_card": {
            "original_question": "蛋白质错误折叠如何导致神经退行性疾病？",
        },
        "_last_module": "m1",
    }
    runner = PipelineRunner(config)
    final = await runner.run(
        question="ignored",
        run_id="child-run",
        seed_state=seed,
    )

    assert final.run_id == "child-run"
    # 问题卡来自断点，未被 followup 重置；M1 因输出齐全而跳过。
    assert final.input_question == "蛋白质错误折叠如何导致神经退行性疾病？"
    assert final.problem_card.original_question == (
        "蛋白质错误折叠如何导致神经退行性疾病？"
    )
    assert runner._resume_last_module is None  # 光标已消费（M1 跳过后清空）


def test_run_lock_rejects_a_second_writer_for_the_same_run(tmp_path) -> None:
    runner = PipelineRunner(PipelineConfig(verbose=False, output_dir=str(tmp_path)))
    runner._current_run_id = "single-writer"
    first = runner._acquire_run_lock()
    try:
        with pytest.raises(RuntimeError, match="active writer"):
            runner._acquire_run_lock()
    finally:
        runner._release_run_lock(first)

    second = runner._acquire_run_lock()
    runner._release_run_lock(second)


def test_resume_cursor_skips_last_completed_module_once() -> None:
    card = HypothesisCard(
        hypothesis_id="H1",
        statement="A testable system-level hypothesis.",
    )
    state = PipelineState(
        input_question="q",
        candidate_hypotheses=[card],
        top_hypotheses=[card],
        iteration_count=0,
        max_iterations=2,
    )
    runner = PipelineRunner(PipelineConfig(verbose=False))
    runner._resume_last_module = "m4"
    fields = {"candidate_hypotheses", "top_hypotheses"}

    assert runner._should_skip_module("m4", fields, state) is True
    assert runner._resume_last_module is None
    assert runner._should_skip_module("m4", fields, state) is False


def test_resume_cursor_cannot_skip_incomplete_declared_output() -> None:
    runner = PipelineRunner(PipelineConfig(verbose=False))
    runner._resume_last_module = "m4"

    assert runner._should_skip_module(
        "m4",
        {"candidate_hypotheses", "top_hypotheses"},
        PipelineState(input_question="q"),
    ) is False
    assert runner._resume_last_module is None


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
