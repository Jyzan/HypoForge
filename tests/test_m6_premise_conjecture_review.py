import pytest

from hypoforge.evaluation.metrics import EvidenceConsistencyMetric
from hypoforge.prompts.m6_prompts import M6_EVIDENCE_VERDICT_SYSTEM, M6_REVIEWER_PROMPTS
from hypoforge.state import (
    EvidenceGraph,
    EvidenceNode,
    EvidenceNodeType,
    HypothesisCard,
    HypothesisPremise,
    PipelineState,
    ResearchPlan,
)


def test_m6_evidence_review_separates_facts_from_conjectures_and_bridges():
    prompt = M6_REVIEWER_PROMPTS["objective_evidence_consistency"].lower()
    assert "factual_premises" in prompt
    assert "working_assumptions" in prompt
    assert "not" in prompt and "established" in prompt
    assert "exact canonical evidence" in prompt


def test_m6_logic_review_does_not_equate_novelty_with_missing_evidence():
    prompt = M6_REVIEWER_PROMPTS["scientific_logic"].lower()
    assert "factual_premises" in prompt
    assert "working_assumptions" in prompt
    assert "absence of direct literature support" in prompt
    assert "must not by itself" in prompt
    assert "internally inconsistent" in prompt
    assert "falsifiable" in prompt


def test_m6_evidence_sufficiency_scope_excludes_m4_conjecture_fields():
    prompt = M6_EVIDENCE_VERDICT_SYSTEM.lower()
    assert "factual_premises" in prompt
    assert "working_assumptions" in prompt
    assert "not literature claims" in prompt


@pytest.mark.asyncio
async def test_m6_objective_gate_is_neutral_for_bridge_only_hypothesis(monkeypatch):
    from hypoforge.modules.m6_review_iteration import M6ReviewIteration
    import hypoforge.modules.m6_review_iteration as m6_module
    from types import SimpleNamespace

    monkeypatch.setattr(
        m6_module,
        "assess_task_alignment",
        lambda *args, **kwargs: SimpleNamespace(
            passed=True, score=5.0, rationale="ok",
        ),
    )

    class Client:
        async def structured_chat(self, **kwargs):
            return {
                "score": 1.0,
                "reasoning": "No paper IDs were cited.",
                "comments": "",
                "suggestions": "",
                "evidence_ids": [],
            }

    hypothesis = HypothesisCard(
        hypothesis_id="H1",
        statement="A may influence B through C.",
        working_assumptions=[HypothesisPremise(
            premise_id="U1", claim="A may influence B through C.",
            kind="unverified_bridge", bridge_hypothesis_node_id="HYP_1",
        )],
        research_gap="Whether the bridge is valid.",
        grounding_status="bridge_only",
    )
    module = M6ReviewIteration(
        reviewers=["objective_evidence_consistency"], fast_mode=True,
    )
    module.client = Client()
    result = await module(
        PipelineState(
            input_question="Does A influence B?",
            top_hypotheses=[hypothesis],
            research_plans=[ResearchPlan(
                hypothesis_id="H1", study_subjects="A and B",
            )],
        )
    )
    review = next(
        item for item in result["reviews"]
        if item.dimension.value == "objective_evidence_consistency"
    )
    assert review.dimension.value == "objective_evidence_consistency"
    assert review.score == 5.0
    assert review.hard_gate_passed is True


@pytest.mark.asyncio
async def test_independent_evidence_metric_audits_factual_premises_only():
    class Client:
        async def structured_chat(self, **kwargs):
            return {"premises": [{
                "premise_id": "P1",
                "verdict": "supported",
                "evidence_ids": ["E1"],
                "rationale": "Exact evidence support.",
            }]}

    metric = object.__new__(EvidenceConsistencyMetric)
    metric._client = Client()
    hypothesis = HypothesisCard(
        hypothesis_id="H1",
        statement="A may influence B.",
        mechanism="A activates an unproven bridge X.",
        factual_premises=[HypothesisPremise(
            premise_id="P1",
            claim="A is present.",
            kind="evidence_backed",
            supporting_evidence_ids=["E1"],
        )],
    )
    graph = EvidenceGraph(nodes=[EvidenceNode(
        id="N1",
        type=EvidenceNodeType.CLAIM,
        label="A is present.",
        metadata={"evidence_ids": ["E1"], "quote": "A is present."},
    )])
    score, trace = await metric.compute(hypothesis, [], evidence_graph=graph)
    assert score == 1.0
    assert trace["atomic_claims"][0]["claim"] == "A is present."
    assert trace["atomic_claims"][0]["premise_id"] == "P1"
