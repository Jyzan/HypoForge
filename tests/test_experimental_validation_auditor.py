from __future__ import annotations

import pytest

from hypoforge.experimental_validation import ExperimentalValidationAuditor
from hypoforge.state import (
    ExperimentalValidationVerdict,
    HypothesisCard,
    HypothesisPremise,
    ResearchPlan,
)


class FakeValidationClient:
    def __init__(self, payload=None, error=None):
        self.payload = payload
        self.error = error
        self.calls = []

    async def structured_chat(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
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


def _plan():
    return ResearchPlan(
        hypothesis_id="H1",
        study_subjects="Target system",
        independent_variables=["A", "B", "A+B"],
        dependent_variables=["Y"],
        control_groups=["vehicle", "A only", "B only", "A+B"],
        procedures=["Apply treatments and collect outcomes."],
        measurement_metrics=["Measure Y and pathway C."],
        analysis_methods=["Two-way ANOVA with an interaction term."],
        expected_results_if_supported="The interaction term is positive.",
        expected_results_if_refuted="The interaction term is absent or negative.",
        bridge_validations=[{
            "bridge_hypothesis_node_id": "HYP_G1",
            "procedure": "Measure the bridge activity after treatment.",
            "measurement": "Bridge activity assay.",
            "falsification_condition": "Bridge activity is absent.",
        }],
    )


def _covered_payload():
    target_ids = [
        "H1:statement",
        "H1:mechanism",
        "H1:prediction:0",
        "H1:falsification:0",
        "H1:working_assumption:BRIDGE_HYP_G1",
    ]
    return {
        "sufficient": True,
        "items": [
            {
                "target_id": target_id,
                "target_kind": "working_assumption" if "working_assumption" in target_id else "prediction",
                "target_text": "",
                "verdict": "covered",
                "procedure_refs": ["procedure:0"],
                "measurement_refs": ["metric:0"],
                "control_refs": ["control:0"],
                "analysis_refs": ["analysis:0"],
                "bridge_validation_refs": (
                    ["bridge_validation:HYP_G1"]
                    if "working_assumption" in target_id else []
                ),
                "falsification_text": "The effect is absent.",
            }
            for target_id in target_ids
        ],
    }


@pytest.mark.asyncio
async def test_shared_auditor_returns_covered_verdict_and_no_error():
    client = FakeValidationClient(_covered_payload())
    auditor = ExperimentalValidationAuditor(
        client=client,
        llm_config=None,
        timeout_seconds=1.0,
    )

    outcome = await auditor.audit(_hypothesis(), _plan(), version=1)

    assert outcome.error == ""
    assert outcome.verdict.sufficient is True
    assert all(item.verdict == "covered" for item in outcome.verdict.items)
    assert "validation_targets" in client.calls[0]["user_prompt"]


@pytest.mark.asyncio
async def test_shared_auditor_reports_transport_error_separately():
    auditor = ExperimentalValidationAuditor(
        client=FakeValidationClient(error=TimeoutError("network")),
        llm_config=None,
        timeout_seconds=1.0,
    )

    outcome = await auditor.audit(_hypothesis(), _plan(), version=1)

    assert outcome.error.startswith("TimeoutError:")
    assert outcome.verdict.sufficient is False
    assert all(item.verdict == "missing" for item in outcome.verdict.items)
