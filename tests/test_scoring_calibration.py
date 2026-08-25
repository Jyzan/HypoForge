from unittest.mock import AsyncMock

import pytest

from hypoforge.evaluation import scorer
from hypoforge.evaluation.metrics import EvidenceConsistencyMetric, NoveltyMetric
from hypoforge.modules import m6_review_iteration
from hypoforge.modules.m6_review_iteration import M6ReviewIteration
from hypoforge.state import (
    EvidenceGraph,
    EvidenceNode,
    EvidenceNodeType,
    HypothesisCard,
    ReviewResult,
    ReviewerDimension,
)


def _hypothesis(**updates):
    values = {
        "hypothesis_id": "H1",
        "statement": "A activates B.",
        "mechanism": "A -> B",
        "observable_predictions": ["B increases after A treatment."],
        "falsification_conditions": ["B does not increase after A treatment."],
        "scores": {
            "novelty": 0.9,
            "scientific_soundness": 0.9,
            "testability": 0.9,
            "evidence_consistency": 0.9,
        },
    }
    values.update(updates)
    return HypothesisCard(**values)


@pytest.mark.asyncio
async def test_missing_graph_entity_is_unknown_not_maximally_novel(monkeypatch):
    """Catch incomplete graph coverage being rewarded as scientific novelty."""

    metric = object.__new__(NoveltyMetric)
    metric._client = object()
    monkeypatch.setattr(
        metric,
        "_decompose_hypothesis",
        AsyncMock(return_value=[{
            "claim": "A activates B.",
            "subject": "A",
            "subject_components": [["A"]],
            "object": "B",
            "object_components": [["B"]],
        }]),
    )
    graph = EvidenceGraph(nodes=[
        EvidenceNode(id="ENT_A", type=EvidenceNodeType.ENTITY, label="A")
    ])

    score, trace = await metric.compute(_hypothesis(), [], evidence_graph=graph)

    assert score == 0.0
    assert trace["claims_novelty"][0]["assessment"] == "insufficient_graph_coverage"


@pytest.mark.asyncio
async def test_claim_without_any_evidence_anchor_receives_no_consistency_credit(
    monkeypatch,
):
    """Catch 'not contradicted' being mistaken for 'supported by evidence'."""

    metric = object.__new__(EvidenceConsistencyMetric)
    metric._client = object()
    metric.embeddings = None
    metric.similarity_threshold = 0.5
    monkeypatch.setattr(
        metric,
        "_decompose_hypothesis",
        AsyncMock(return_value=[{
            "claim": "A activates B.",
            "subject": "A",
            "subject_components": [["A"]],
            "object": "B",
            "object_components": [["B"]],
        }]),
    )
    graph = EvidenceGraph(nodes=[
        EvidenceNode(
            id="N_unrelated",
            type=EvidenceNodeType.CLAIM,
            label="C is associated with D.",
        )
    ])

    score, trace = await metric.compute(_hypothesis(), [], evidence_graph=graph)

    assert score == 0.0
    assert trace["atomic_claims"][0]["support_status"] == "unsupported"


@pytest.mark.asyncio
async def test_report_composite_uses_independent_metrics_not_m4_self_scores(
    monkeypatch,
):
    """Catch M4's self-assessment being labelled as the final composite."""

    values = {"novelty": 0.2, "testability": 0.8, "evidence_consistency": 0.3}

    def metric_class(name):
        class FakeMetric:
            independent = True
            implemented = True

            def __init__(self, **kwargs):
                pass

            async def compute(self, hypothesis, entries, **kwargs):
                return values[name]

        return FakeMetric

    monkeypatch.setattr(scorer.MetricRegistry, "list_all", lambda: list(values))
    monkeypatch.setattr(
        scorer.MetricRegistry,
        "get",
        lambda name: metric_class(name),
    )

    result = await scorer.score_hypothesis_async(_hypothesis(), [])

    assert result["self_reported_composite"] == 0.9
    assert result["independent_composite"] == 0.4267
    assert result["composite"] == result["independent_composite"]


def _review(dimension, score, hard_gate_passed=None):
    return ReviewResult(
        dimension=ReviewerDimension(dimension),
        score=score,
        hard_gate_passed=hard_gate_passed,
    )


def test_m6_default_reviewers_include_evidence_entailment():
    """Catch standard M6 runs silently omitting the evidence specialist."""

    assert "objective_evidence_consistency" in M6ReviewIteration().reviewer_dims


def test_overall_uses_core_science_reviews_and_caps_unsupported_evidence():
    """Catch structural/full-text gates diluting a failed evidence review."""

    aggregate = getattr(m6_review_iteration, "_aggregate_overall_reviews", None)
    assert callable(aggregate)
    reviews = [
        _review("task_alignment", 5.0, True),
        _review("scientific_logic", 5.0),
        _review("objective_evidence_consistency", 2.0, False),
        _review("method_feasibility", 5.0),
        _review("evidence_coverage_gate", 5.0, True),
        _review("answer_completeness_gate", 5.0, True),
        _review("source_quality_gate", 5.0, True),
    ]

    score, failed_gates = aggregate(reviews)

    assert score == 2.9
    assert failed_gates == ["objective_evidence_consistency"]


def test_auxiliary_gate_scores_do_not_change_a_scientific_overall():
    """Catch source availability and field presence entering the mean score."""

    aggregate = getattr(m6_review_iteration, "_aggregate_overall_reviews", None)
    assert callable(aggregate)
    reviews = [
        _review("task_alignment", 5.0, True),
        _review("scientific_logic", 4.0),
        _review("objective_evidence_consistency", 4.0, True),
        _review("method_feasibility", 4.0),
        _review("evidence_coverage_gate", 1.0, True),
        _review("answer_completeness_gate", 1.0, True),
        _review("source_quality_gate", 1.0, True),
    ]

    score, failed_gates = aggregate(reviews)

    assert score == 4.0
    assert failed_gates == []
