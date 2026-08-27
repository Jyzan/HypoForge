from __future__ import annotations

import pytest

import hypoforge.modules.m5_research_plan as m5_module
from hypoforge.modules.m5_research_plan import M5ResearchPlan
from hypoforge.state import (
    ExperimentalValidationVerdict,
    HypothesisCard,
    PipelineState,
    ResearchPlan,
    ValidationCoverageItem,
)
from hypoforge.experimental_validation import ExperimentalValidationAuditOutcome


def _plan(*, subject="previous plan"):
    return ResearchPlan(
        hypothesis_id="H1",
        study_subjects=subject,
        procedures=["Apply intervention."],
        measurement_metrics=["Measure outcome."],
        analysis_methods=["Compare groups."],
    )


def test_coverage_retry_context_lists_every_unresolved_target_and_previous_plan():
    verdict = ExperimentalValidationVerdict(
        sufficient=False,
        items=[
            ValidationCoverageItem(
                target_id="H1:mechanism",
                target_kind="mechanism",
                target_text="A activates pathway B.",
                verdict="partial",
                rationale="No direct pathway measurement.",
            ),
            ValidationCoverageItem(
                target_id="H1:prediction:1",
                target_kind="prediction",
                target_text="The human marker declines after intervention.",
                verdict="missing",
                rationale="No human cohort or longitudinal analysis.",
            ),
        ],
    )

    text = M5ResearchPlan._build_coverage_retry_context(_plan(), verdict)

    assert "H1:mechanism" in text
    assert "No direct pathway measurement" in text
    assert "H1:prediction:1" in text
    assert "No human cohort" in text
    assert "Previous ResearchPlan JSON" in text
    assert "only allowed coverage rewrite" in text


def _state(*, iteration_count=0):
    return PipelineState(
        input_question="Can A improve Y?",
        iteration_count=iteration_count,
        top_hypotheses=[HypothesisCard(
            hypothesis_id="H1",
            statement="A improves Y.",
            mechanism="A changes pathway B.",
            observable_predictions=["Y increases."],
            falsification_conditions=["Y does not increase."],
        )],
    )


def _covered_verdict():
    return ExperimentalValidationVerdict(sufficient=True, rationale="covered")


def _missing_verdict():
    return ExperimentalValidationVerdict(
        sufficient=False,
        items=[ValidationCoverageItem(
            target_id="H1:mechanism",
            target_kind="mechanism",
            target_text="A changes pathway B.",
            verdict="missing",
            rationale="No direct pathway measurement.",
        )],
        rationale="incomplete",
    )


class _FakeAuditor:
    outcomes = []
    calls = []

    def __init__(self, **kwargs):
        self.__class__.calls = []

    async def audit(self, hypothesis, plan, version):
        self.__class__.calls.append((hypothesis.hypothesis_id, plan.study_subjects, version))
        return self.__class__.outcomes.pop(0)


class _FakeContextPack:
    rendered = "graph context"


class _FakePlanner:
    def plan(self, *args, **kwargs):
        return _FakeContextPack()


def _patch_pipeline_surface(monkeypatch):
    monkeypatch.setattr(m5_module, "ContextPlanner", _FakePlanner)
    monkeypatch.setattr(m5_module, "emit_context_built", lambda *args, **kwargs: None)
    monkeypatch.setattr(m5_module, "build_graph_context", lambda state: object())


@pytest.mark.asyncio
async def test_m5_rewrites_once_with_structured_coverage_feedback(monkeypatch):
    generated = []

    async def fake_generate(self, **kwargs):
        generated.append(kwargs["extra_feedback"])
        return _plan(subject="first plan" if len(generated) == 1 else "repaired plan")

    _FakeAuditor.outcomes = [
        ExperimentalValidationAuditOutcome(_missing_verdict()),
        ExperimentalValidationAuditOutcome(_covered_verdict()),
    ]
    monkeypatch.setattr(m5_module, "ExperimentalValidationAuditor", _FakeAuditor)
    monkeypatch.setattr(M5ResearchPlan, "_generate_plan_candidate", fake_generate, raising=False)
    _patch_pipeline_surface(monkeypatch)

    module = object.__new__(M5ResearchPlan)
    module.client = object()
    module.fast_mode = False
    module.llm_config = None
    module.validation_timeout_seconds = 1.0

    patch = await module(_state(iteration_count=0))

    assert len(generated) == 2
    assert "H1:mechanism" in generated[1]
    assert "No direct pathway measurement" in generated[1]
    assert patch["research_plans"][0].study_subjects == "repaired plan"
    assert patch["experimental_validation_verdict"].sufficient is True


@pytest.mark.asyncio
async def test_m5_stops_after_one_coverage_rewrite(monkeypatch):
    generated = []

    async def fake_generate(self, **kwargs):
        generated.append(kwargs["extra_feedback"])
        return _plan(subject=f"plan {len(generated)}")

    _FakeAuditor.outcomes = [
        ExperimentalValidationAuditOutcome(_missing_verdict()),
        ExperimentalValidationAuditOutcome(_missing_verdict()),
    ]
    monkeypatch.setattr(m5_module, "ExperimentalValidationAuditor", _FakeAuditor)
    monkeypatch.setattr(M5ResearchPlan, "_generate_plan_candidate", fake_generate, raising=False)
    _patch_pipeline_surface(monkeypatch)

    module = object.__new__(M5ResearchPlan)
    module.client = object()
    module.fast_mode = False
    module.llm_config = None
    module.validation_timeout_seconds = 1.0

    patch = await module(_state())

    assert len(generated) == 2
    assert patch["research_plans"][0].study_subjects == "plan 2"


@pytest.mark.asyncio
async def test_fast_mode_skips_internal_validation(monkeypatch):
    generated = []

    async def fake_generate(self, **kwargs):
        generated.append(kwargs["extra_feedback"])
        return _plan(subject="fast plan")

    _FakeAuditor.outcomes = [ExperimentalValidationAuditOutcome(_missing_verdict())]
    monkeypatch.setattr(m5_module, "ExperimentalValidationAuditor", _FakeAuditor)
    monkeypatch.setattr(M5ResearchPlan, "_generate_plan_candidate", fake_generate, raising=False)
    _patch_pipeline_surface(monkeypatch)

    module = object.__new__(M5ResearchPlan)
    module.client = object()
    module.fast_mode = True
    module.llm_config = None
    module.validation_timeout_seconds = 1.0

    patch = await module(_state())

    assert len(generated) == 1
    assert _FakeAuditor.outcomes
    assert patch["research_plans"][0].study_subjects == "fast plan"


@pytest.mark.asyncio
async def test_m5_preserves_plan_when_precheck_has_technical_error(monkeypatch):
    generated = []

    async def fake_generate(self, **kwargs):
        generated.append(kwargs["extra_feedback"])
        return _plan(subject="preserved plan")

    _FakeAuditor.outcomes = [ExperimentalValidationAuditOutcome(
        _missing_verdict(), error="TimeoutError: network"
    )]
    monkeypatch.setattr(m5_module, "ExperimentalValidationAuditor", _FakeAuditor)
    monkeypatch.setattr(M5ResearchPlan, "_generate_plan_candidate", fake_generate, raising=False)
    _patch_pipeline_surface(monkeypatch)

    module = object.__new__(M5ResearchPlan)
    module.client = object()
    module.fast_mode = False
    module.llm_config = None
    module.validation_timeout_seconds = 1.0

    patch = await module(_state())

    assert len(generated) == 1
    assert patch["research_plans"][0].study_subjects == "preserved plan"
