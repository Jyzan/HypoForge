import pytest

from hypoforge.modules.m6_review_iteration import M6ReviewIteration
from hypoforge.state import (
    ExperimentalValidationVerdict,
    HypothesisCard,
    HypothesisPremise,
    ResearchPlan,
)
from hypoforge.experimental_validation import ExperimentalValidationAuditOutcome


class FakeValidationClient:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    async def structured_chat(self, **kwargs):
        self.calls.append(kwargs)
        return self.payload


def _hypothesis():
    return HypothesisCard(
        hypothesis_id="H1",
        statement="A and B act synergistically to improve Y.",
        mechanism="A and B jointly alter pathway C.",
        observable_predictions=["The combination has a positive interaction effect."],
        falsification_conditions=["The interaction term is absent or negative."],
        working_assumptions=[HypothesisPremise(
            premise_id="BRIDGE_HYP_G1",
            claim="The bridge from C to Y is active.",
            kind="unverified_bridge",
            bridge_hypothesis_node_id="HYP_G1",
        )],
    )


def _plan(*, include_bridge=True, analyses=None):
    return ResearchPlan(
        hypothesis_id="H1",
        study_subjects="Target system",
        independent_variables=["A", "B", "A+B"],
        dependent_variables=["Y"],
        control_groups=["vehicle", "A only", "B only", "A+B"],
        procedures=["Apply treatments and collect outcomes."],
        measurement_metrics=["Measure Y and pathway C."],
        analysis_methods=analyses or ["Two-way ANOVA with an interaction term."],
        expected_results_if_supported="The interaction term is positive.",
        expected_results_if_refuted="The interaction term is absent or negative.",
        bridge_validations=([{
            "bridge_hypothesis_node_id": "HYP_G1",
            "procedure": "Measure the bridge activity after treatment.",
            "measurement": "Bridge activity assay.",
            "falsification_condition": "Bridge activity is absent.",
        }] if include_bridge else []),
    )


def _module(payload):
    module = object.__new__(M6ReviewIteration)
    module.client = FakeValidationClient(payload)
    module.reviewer_timeout_seconds = 1.0
    module.llm_config = None
    return module


@pytest.mark.asyncio
async def test_m6_formal_review_delegates_to_shared_auditor(monkeypatch):
    import hypoforge.modules.m6_review_iteration as m6_module

    calls = []

    class SharedAuditor:
        def __init__(self, **kwargs):
            pass

        async def audit(self, hypothesis, plan, version):
            calls.append((hypothesis.hypothesis_id, plan.hypothesis_id, version))
            return ExperimentalValidationAuditOutcome(
                verdict=ExperimentalValidationVerdict(sufficient=True),
            )

    monkeypatch.setattr(m6_module, "ExperimentalValidationAuditor", SharedAuditor)
    module = object.__new__(M6ReviewIteration)
    module.client = object()
    module.llm_config = None
    module.reviewer_timeout_seconds = 1.0

    verdict = await module._judge_experimental_validation(
        _hypothesis(), _plan(), version=2,
    )

    assert verdict.sufficient is True
    assert calls == [("H1", "H1", 2)]


def test_validation_targets_cover_all_m4_sections():
    targets = M6ReviewIteration._validation_targets(_hypothesis())
    assert [item.target_id for item in targets] == [
        "H1:statement",
        "H1:mechanism",
        "H1:prediction:0",
        "H1:falsification:0",
        "H1:working_assumption:BRIDGE_HYP_G1",
    ]


@pytest.mark.asyncio
async def test_complete_m5_plan_covers_every_target():
    payload = {"sufficient": True, "items": [
        {
            "target_id": target_id,
            "target_kind": "statement" if target_id.endswith(":statement") else "prediction",
            "target_text": "",
            "verdict": "covered",
            "procedure_refs": ["procedure:0"],
            "measurement_refs": ["metric:0"],
            "control_refs": ["control:0"],
            "analysis_refs": ["analysis:0"],
            "bridge_validation_refs": (["bridge_validation:HYP_G1"] if "working_assumption" in target_id else []),
            "falsification_text": "The effect is absent.",
        }
        for target_id in [
            "H1:statement", "H1:mechanism", "H1:prediction:0",
            "H1:falsification:0", "H1:working_assumption:BRIDGE_HYP_G1",
        ]
    ]}
    module = _module(payload)
    verdict = await module._judge_experimental_validation(_hypothesis(), _plan(), 1)
    assert verdict.sufficient is True
    assert all(item.verdict == "covered" for item in verdict.items)
    assert "validation_targets" in module.client.calls[0]["user_prompt"]


@pytest.mark.asyncio
async def test_missing_bridge_validation_is_a_plan_failure_not_an_evidence_gap():
    payload = {"sufficient": True, "items": [{
        "target_id": "H1:working_assumption:BRIDGE_HYP_G1",
        "target_kind": "working_assumption",
        "target_text": "The bridge is active.",
        "verdict": "covered",
        "procedure_refs": ["procedure:0"],
        "measurement_refs": ["metric:0"],
        "analysis_refs": ["analysis:0"],
        "falsification_text": "The bridge is absent.",
    }]}
    module = _module(payload)
    verdict = await module._judge_experimental_validation(
        _hypothesis(), _plan(include_bridge=False), 1,
    )
    item = next(item for item in verdict.items if item.target_kind == "working_assumption")
    assert verdict.sufficient is False
    assert item.verdict == "missing"
    assert "bridge_validation" in item.rationale


@pytest.mark.asyncio
async def test_complete_bridge_validation_covers_working_assumption_without_duplicate_generic_refs():
    bridge_only_hypothesis = HypothesisCard(
        hypothesis_id="H1",
        statement="",
        working_assumptions=[HypothesisPremise(
            premise_id="BRIDGE_HYP_G1",
            claim="The bridge from C to Y is active.",
            kind="unverified_bridge",
            bridge_hypothesis_node_id="HYP_G1",
        )],
    )
    payload = {"sufficient": True, "items": [{
        "target_id": "H1:working_assumption:BRIDGE_HYP_G1",
        "target_kind": "working_assumption",
        "target_text": "The bridge is active.",
        "verdict": "covered",
        "procedure_refs": [],
        "measurement_refs": [],
        "analysis_refs": [],
        "bridge_validation_refs": ["bridge_validation:HYP_G1"],
        "falsification_text": "The bridge is absent.",
    }]}
    module = _module(payload)

    verdict = await module._judge_experimental_validation(
        bridge_only_hypothesis, _plan(include_bridge=True), 1,
    )

    item = next(item for item in verdict.items if item.target_kind == "working_assumption")
    assert item.verdict == "covered"
    assert verdict.sufficient is True


@pytest.mark.asyncio
async def test_one_way_anova_does_not_cover_synergy_target():
    payload = {"sufficient": True, "items": [{
        "target_id": "H1:statement",
        "target_kind": "statement",
        "target_text": "A and B act synergistically.",
        "verdict": "partial",
        "procedure_refs": ["procedure:0"],
        "measurement_refs": ["metric:0"],
        "control_refs": ["control:0"],
        "analysis_refs": ["analysis:0"],
        "falsification_text": "",
        "rationale": "One-way ANOVA cannot estimate an interaction term.",
    }]}
    module = _module(payload)
    verdict = await module._judge_experimental_validation(
        _hypothesis(), _plan(analyses=["One-way ANOVA"]), 1,
    )
    item = next(item for item in verdict.items if item.target_id == "H1:statement")
    assert verdict.sufficient is False
    assert item.verdict == "partial"
