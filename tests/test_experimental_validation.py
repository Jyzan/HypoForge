"""Coverage decisions must survive batching, timeouts, and M5/M6 handoff."""

import asyncio
import json

import pytest

from hypoforge.experimental_validation import ExperimentalValidationAuditor
from hypoforge.modules.m5_research_plan import M5ResearchPlan
from hypoforge.modules.m6_review_iteration import M6ReviewIteration
from hypoforge.state import (
    ExperimentalValidationVerdict, HypothesisCard, PipelineState, ProblemCard,
    ResearchPlan, ValidationCoverageItem,
)
from hypoforge.strict_contracts import StrictM6ReviewIteration


def card(identifier="H1"):
    return HypothesisCard(
        hypothesis_id=identifier, statement="Treatment changes the measured outcome.",
        mechanism="Treatment alters the mediator and the outcome.",
        observable_predictions=[f"Outcome {index} decreases." for index in range(4)],
        falsification_conditions=["The outcome does not decrease."],
    )


def plan(identifier="H1"):
    return ResearchPlan(
        hypothesis_id=identifier, procedures=["Compare treated and untreated samples."],
        measurement_metrics=["Measure the outcome and mediator."],
        control_groups=["Untreated samples."], analysis_methods=["Estimate the treatment effect."],
    )


def payload(kwargs):
    prompt = kwargs["user_prompt"]
    return json.JSONDecoder().raw_decode(prompt[prompt.index("{"):])[0]


def decision(target):
    return {
        "target_id": target["target_id"], "verdict": "covered",
        "procedure_refs": ["procedure:0"], "measurement_refs": ["metric:0"],
        "control_refs": ["control:0"], "analysis_refs": ["analysis:0"],
        "falsification_text": "No outcome decrease falsifies the prediction.",
    }


@pytest.mark.asyncio
async def test_batched_audit_checks_all_targets_with_canonical_text_and_bounded_concurrency():
    active = peak = 0
    seen = []

    class Client:
        async def structured_chat(self, **kwargs):
            nonlocal active, peak
            assert kwargs["max_tokens"] == 2048
            assert kwargs["disable_thinking"] is True
            row_schema = kwargs["output_schema"]["$defs"]["_ValidationAuditRow"]["properties"]
            assert "target_text" not in row_schema and "target_kind" not in row_schema
            targets = payload(kwargs)["validation_targets"]
            seen.extend(t["target_id"] for t in targets)
            active += 1
            peak = max(peak, active)
            try:
                await asyncio.sleep(.02 if targets[0]["target_kind"] == "statement" else .005)
                return {"sufficient": True, "items": [
                    {**decision(t), "target_text": "Model rewrote the claim."} for t in targets
                ]}
            finally:
                active -= 1

    hypothesis = card()
    targets = ExperimentalValidationAuditor.validation_targets(hypothesis)
    outcome = await ExperimentalValidationAuditor(
        client=Client(), llm_config=None, timeout_seconds=1,
        target_batch_size=3, concurrency=2, max_tokens=2048,
    ).audit(hypothesis, plan(), 1)
    assert not outcome.error
    assert outcome.verdict.sufficient
    assert peak == 2
    assert sorted(seen) == sorted(t.target_id for t in targets)
    assert [(i.target_id, i.target_text) for i in outcome.verdict.items] == [
        (t.target_id, t.target_text) for t in targets
    ]


@pytest.mark.asyncio
async def test_invalid_references_cannot_be_approved_by_model_boolean():
    class Client:
        async def structured_chat(self, **kwargs):
            targets = payload(kwargs)["validation_targets"]
            return {"sufficient": True, "items": [
                {**decision(t), "measurement_refs": ["invented:0"]} for t in targets
            ]}

    outcome = await ExperimentalValidationAuditor(
        client=Client(), llm_config=None, timeout_seconds=1,
    ).audit(card(), plan(), 1)
    assert not outcome.error
    assert not outcome.verdict.sufficient
    assert len(outcome.verdict.items) == 7
    assert outcome.verdict.items[0].verdict == "partial"
    assert outcome.verdict.items[0].measurement_refs == []
    assert all(item.verdict == "partial" for item in outcome.verdict.items)


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", ["missing", "duplicate", "foreign"])
async def test_incomplete_or_misidentified_rows_are_audit_errors_not_quality_decisions(shape):
    class Client:
        async def structured_chat(self, **kwargs):
            targets = payload(kwargs)["validation_targets"]
            rows = [decision(t) for t in targets]
            if shape == "missing":
                rows.pop()
            elif shape == "duplicate":
                rows[-1] = rows[0]
            else:
                rows[-1]["target_id"] = "H999:statement"
            return {"sufficient": True, "items": rows}

    outcome = await ExperimentalValidationAuditor(
        client=Client(), llm_config=None, timeout_seconds=1,
    ).audit(card(), plan(), 1)
    assert "exactly one row per requested target" in outcome.error
    assert outcome.verdict.audit_errors == [outcome.error]
    assert len(outcome.verdict.items) == 7
    assert not outcome.verdict.sufficient


@pytest.mark.asyncio
async def test_audit_timeout_cancels_active_batches_and_preserves_every_target_as_failed():
    cancelled = []

    class Client:
        async def structured_chat(self, **kwargs):
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.append(payload(kwargs)["validation_targets"][0]["target_id"])
                raise

    outcome = await ExperimentalValidationAuditor(
        client=Client(), llm_config=None, timeout_seconds=.025,
        target_batch_size=3, concurrency=2,
    ).audit(card(), plan(), 1)
    assert len(cancelled) == 2
    assert "TimeoutError" in outcome.error
    assert outcome.verdict.audit_errors == [outcome.error]
    assert not outcome.verdict.sufficient
    assert len(outcome.verdict.items) == 7
    assert all(item.verdict == "missing" for item in outcome.verdict.items)


@pytest.mark.asyncio
async def test_m5_preserves_failed_audit_alongside_successful_other_hypothesis():
    class Client:
        async def structured_chat(self, **kwargs):
            targets = payload(kwargs)["validation_targets"]
            if targets[0]["hypothesis_id"] == "H2":
                raise TimeoutError("provider did not finish")
            return {"sufficient": True, "items": [decision(t) for t in targets]}

    state = PipelineState(
        input_question="Propose a testable hypothesis.", top_hypotheses=[card(), card("H2")],
        problem_card=ProblemCard(original_question="Propose a testable hypothesis."),
    )
    module = M5ResearchPlan()
    module.client = Client()

    async def generate(**kwargs):
        return plan(kwargs["hypothesis"].hypothesis_id)

    module._generate_plan_candidate = generate
    result = await module(state)
    verdict = result["experimental_validation_verdict"]
    assert len(result["research_plans"]) == 2
    assert len(verdict.items) == 14
    assert not verdict.sufficient
    assert verdict.audit_errors == ["TimeoutError: provider did not finish"]
    assert all(i.verdict == "covered" for i in verdict.items if i.target_id.startswith("H1:"))
    assert all(i.verdict == "missing" for i in verdict.items if i.target_id.startswith("H2:"))


@pytest.mark.asyncio
async def test_second_attempt_transport_failure_cannot_leave_first_attempt_verdict():
    class Client:
        calls = 0

        async def structured_chat(self, **kwargs):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("connection lost after plan rewrite")
            return {"sufficient": False, "items": [
                {**decision(t), "verdict": "partial"} for t in payload(kwargs)["validation_targets"]
            ]}

    state = PipelineState(input_question="Test treatment.", top_hypotheses=[card()])
    module = M5ResearchPlan()
    module.client = Client()
    attempts = []

    async def generate(**kwargs):
        attempts.append(kwargs["coverage_attempt"])
        return plan()

    module._generate_plan_candidate = generate
    result = await module(state)
    verdict = result["experimental_validation_verdict"]
    assert attempts == [1, 2]
    assert verdict.audit_errors == ["RuntimeError: connection lost after plan rewrite"]
    assert all(item.verdict == "missing" for item in verdict.items)


def completed_verdict(hypothesis):
    return ExperimentalValidationVerdict(
        sufficient=False,
        items=[ValidationCoverageItem(
            target_id=t.target_id, target_kind=t.target_kind, target_text=t.target_text,
            verdict="partial",
        ) for t in ExperimentalValidationAuditor.validation_targets(hypothesis)],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["absent", "transport", "subset", "stale", "duplicate"])
async def test_standard_m6_cannot_skip_unavailable_or_stale_validation_audit(failure):
    hypothesis = card()
    verdict = completed_verdict(hypothesis)
    if failure == "absent":
        verdict = None
    elif failure == "transport":
        verdict.audit_errors = ["TimeoutError: provider"]
    elif failure == "subset":
        verdict.items.pop()
    elif failure == "stale":
        verdict.items[0].target_text = "A different hypothesis."
    else:
        verdict.items.append(verdict.items[0])
    state = PipelineState(input_question="Test treatment.", top_hypotheses=[hypothesis],
                          experimental_validation_verdict=verdict)
    with pytest.raises(RuntimeError, match="completed M5 experimental validation audit"):
        await StrictM6ReviewIteration()(state)


@pytest.mark.asyncio
async def test_completed_negative_audit_reaches_m6_for_normal_plan_revision(monkeypatch):
    seen = []

    async def review(self, state, config=None):
        seen.append(state.experimental_validation_verdict)
        return {"experimental_validation_verdict": state.experimental_validation_verdict}

    monkeypatch.setattr(M6ReviewIteration, "__call__", review)
    hypothesis = card()
    state = PipelineState(input_question="Test treatment.", top_hypotheses=[hypothesis],
                          experimental_validation_verdict=completed_verdict(hypothesis))
    await StrictM6ReviewIteration()(state)
    assert len(seen) == 1
    assert not seen[0].sufficient


def test_benchmark_export_requires_completed_audit_but_does_not_filter_negative_decisions():
    from hypoforge.benchmarks.agentideabench import select_submission
    from hypoforge.state import ReviewResult, ReviewerDimension
    hypothesis = card()
    state = PipelineState(
        input_question="Test treatment.", top_hypotheses=[hypothesis], research_plans=[plan()],
        reviews=[ReviewResult(dimension=ReviewerDimension.OVERALL, score=2, reasoning="Low quality.")],
    )
    with pytest.raises(ValueError, match="no completed M5 validation audit"):
        select_submission(state)
    state.experimental_validation_verdict = completed_verdict(hypothesis)
    assert select_submission(state)["hypothesis"]["hypothesis_id"] == "H1"


@pytest.mark.asyncio
async def test_m5_mixed_reuse_and_new_plan_preserves_audits_for_both(monkeypatch):
    first, second = card(), card("H2")
    prior = completed_verdict(first)
    for item in prior.items:
        item.verdict = "covered"
    # The aggregate could have failed for a previously selected different card.
    prior.sufficient = False
    state = PipelineState(
        input_question="Test treatment.", top_hypotheses=[first, second],
        experimental_validation_verdict=prior,
    )
    module = M5ResearchPlan()
    monkeypatch.setattr(module, "_is_supplement_reentry", lambda state: True)
    monkeypatch.setattr(module, "_reusable_plans", lambda state: {"H1": plan()})

    class Client:
        async def structured_chat(self, **kwargs):
            targets = payload(kwargs)["validation_targets"]
            return {"sufficient": True, "items": [decision(t) for t in targets]}

    module.client = Client()

    async def generate(**kwargs):
        assert kwargs["hypothesis"].hypothesis_id == "H2"
        return plan("H2")

    module._generate_plan_candidate = generate
    result = await module(state)
    verdict = result["experimental_validation_verdict"]
    assert verdict.sufficient
    assert len(verdict.items) == 14
    assert {i.target_id.split(":")[0] for i in verdict.items} == {"H1", "H2"}
