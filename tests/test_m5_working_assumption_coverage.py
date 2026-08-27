from hypoforge.modules.m5_research_plan import M5ResearchPlan
from hypoforge.prompts.m5_prompts import M5_SYSTEM_PROMPT
from hypoforge.state import (
    HypothesisCard,
    HypothesisPremise,
    ResearchPlan,
)


def test_m5_plan_contains_an_explicit_validation_for_every_bridge_assumption():
    module = object.__new__(M5ResearchPlan)
    hypothesis = HypothesisCard(
        hypothesis_id="H1",
        statement="A may influence B.",
        working_assumptions=[HypothesisPremise(
            premise_id="BRIDGE_HYP_G1",
            claim="A may influence B through C.",
            kind="unverified_bridge",
            bridge_hypothesis_node_id="HYP_G1",
        )],
        falsification_conditions=["No effect is observed after blocking C."],
    )
    plan = ResearchPlan(
        hypothesis_id="H1",
        procedures=["Apply the intervention and block C in a matched control."],
        measurement_metrics=["Measure the downstream B outcome."],
        expected_results_if_refuted="The intervention has no reliable effect.",
    )
    normalized = module._ensure_bridge_validations(plan, hypothesis)
    assert len(normalized.bridge_validations) == 1
    validation = normalized.bridge_validations[0]
    assert validation.bridge_hypothesis_node_id == "HYP_G1"
    assert validation.procedure
    assert validation.measurement
    assert validation.falsification_condition


def test_m5_prompt_keeps_conjecture_out_of_literature_evidence_scope():
    prompt = M5_SYSTEM_PROMPT.lower()
    assert "statement" in prompt and "mechanism" in prompt
    assert "proposed contributions" in prompt
    assert "bridge_validations" in prompt
