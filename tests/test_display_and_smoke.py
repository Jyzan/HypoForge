from io import StringIO

import pytest
from rich.console import Console
from rich.padding import Padding

import hypoforge.pipeline as pipeline_module
import hypoforge.display.panels as panels_module
import hypoforge.modules.m4_hypothesis_generation as m4_module
from hypoforge.config import PipelineConfig
from hypoforge.display.panels import (
    _labelled_wrapped_table,
    _numbered_wrapped_table,
    _split_risks_and_alternatives,
)
from hypoforge.pipeline import PipelineRunner, _should_continue_iterating
from hypoforge.protocol import ModuleProtocol
from hypoforge.modules.m4_hypothesis_generation import M4HypothesisGeneration
from hypoforge.state import PipelineState, ReviewResult, ReviewerDimension
from scripts.smoke_pipeline import build_fast_config


def test_risk_alternative_parser_handles_single_initial_number():
    value = (
        "1. Risk: Protein may aggregate during purification. "
        "Alternative: Use fresh protein with SEC validation. "
        "Risk: FRAP may be confounded by phototoxicity. "
        "Alternative: Reduce laser power and use an orthogonal assay."
    )

    risks, alternatives = _split_risks_and_alternatives(value)

    assert risks == [
        "Protein may aggregate during purification.",
        "FRAP may be confounded by phototoxicity.",
    ]
    assert alternatives == [
        "Use fresh protein with SEC validation.",
        "Reduce laser power and use an orthogonal assay.",
    ]


def test_risk_parser_falls_back_to_numbered_items():
    risks, alternatives = _split_risks_and_alternatives(
        "1. Low sample yield. 2. High inter-animal variability."
    )
    assert risks == ["Low sample yield.", "High inter-animal variability."]
    assert alternatives == []


def test_key_entities_wrapped_lines_align_with_first_entity():
    output = StringIO()
    test_console = Console(file=output, width=58, color_system=None)
    test_console.print(Padding(
        _labelled_wrapped_table(
            "Key entities:",
            "CRISPR-Cas9, CAR-T cells, tumor microenvironment, "
            "antibody-drug conjugates, nanoparticle drug delivery, genes",
        ),
        (0, 0, 0, 2),
    ))

    lines = [line for line in output.getvalue().splitlines() if line.strip()]
    first_entity_column = lines[0].index("CRISPR-Cas9")
    wrapped_lines = lines[1:]
    assert wrapped_lines
    assert all(len(line) - len(line.lstrip()) == first_entity_column for line in wrapped_lines)


def test_risk_wrapped_lines_align_with_item_body():
    output = StringIO()
    test_console = Console(file=output, width=62, color_system=None)
    test_console.print(Padding(
        _numbered_wrapped_table(
            [
                "If precursor frequencies are too low, enrich the starting "
                "B-cell population and validate in a second model."
            ],
            marker_style="bold yellow",
        ),
        (0, 0, 0, 2),
    ))

    lines = [line for line in output.getvalue().splitlines() if line.strip()]
    first_body_column = lines[0].index("If")
    assert len(lines) > 1
    assert all(len(line) - len(line.lstrip()) == first_body_column for line in lines[1:])


def test_smoke_config_runs_one_real_feedback_round_by_default():
    config = build_fast_config("configs/default.yaml")
    assert config.enable_iteration is True
    assert config.max_iterations == 2
    assert config.interactive is True
    assert config.iteration_module_target == "m4"
    assert config.scoring.review_threshold > 5.0

    state = PipelineState(
        iteration_count=1,
        max_iterations=config.max_iterations,
        review_score_threshold=config.scoring.review_threshold,
        reviews=[
            ReviewResult(
                dimension=ReviewerDimension("overall"),
                score=5.0,
                version=1,
            )
        ],
    )
    assert _should_continue_iterating(state) == "iterate"


def test_smoke_feedback_round_can_be_disabled():
    config = build_fast_config("configs/default.yaml", feedback_round=False)
    assert config.enable_iteration is False
    assert config.max_iterations == 1
    assert config.interactive is False


@pytest.mark.asyncio
async def test_guidance_is_collected_before_m4_header(monkeypatch, tmp_path):
    events = []

    class FakeM4(ModuleProtocol):
        module_name = "m4"
        module_version = "test"
        description = "test revision"

        async def collect_user_guidance(self, state):
            events.append("guidance")
            return {"user_guidance": ["[iter 1] focus on mechanism"]}

        async def __call__(self, state, config=None):
            events.append("module")
            assert state.user_guidance == ["[iter 1] focus on mechanism"]
            return {"metrics": {"revised": True}}

        @classmethod
        def get_input_fields(cls):
            return []

        @classmethod
        def get_output_fields(cls):
            return ["metrics"]

    monkeypatch.setattr(
        pipeline_module,
        "render_phase_header",
        lambda *args, **kwargs: events.append("header"),
    )
    monkeypatch.setattr(pipeline_module, "render_module_result", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline_module, "render_phase_done", lambda *args, **kwargs: None)

    runner = PipelineRunner(PipelineConfig(verbose=True, output_dir=str(tmp_path)))
    runner._skills = []
    runner._current_run_id = "guidance-order"
    result = await runner._make_node_wrapper("m4", FakeM4())(
        PipelineState(iteration_count=1)
    )

    assert events == ["guidance", "header", "module"]
    assert result["user_guidance"] == ["[iter 1] focus on mechanism"]


@pytest.mark.asyncio
async def test_typed_guidance_shows_captured_confirmation(monkeypatch):
    events = []

    class FakeStdin:
        @staticmethod
        def isatty():
            return True

    async def fake_to_thread(function, *args, **kwargs):
        events.append("input")
        return "Focus on the mechanism"

    monkeypatch.setattr(m4_module.sys, "stdin", FakeStdin())
    monkeypatch.setattr(m4_module.asyncio, "to_thread", fake_to_thread)
    monkeypatch.setattr(
        panels_module,
        "render_guidance_prompt",
        lambda iteration: events.append("prompt"),
    )
    monkeypatch.setattr(
        panels_module,
        "render_guidance_result",
        lambda guidance: events.append(("captured", guidance)),
    )

    module = M4HypothesisGeneration(interactive=True)
    guidance = await module._prompt_user_guidance(PipelineState(iteration_count=1))

    assert guidance == "Focus on the mechanism"
    assert events == [
        "prompt",
        "input",
        ("captured", "Focus on the mechanism"),
    ]
