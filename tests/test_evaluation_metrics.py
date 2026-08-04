from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from hypoforge.evaluation.metrics import EvidenceConsistencyMetric, NoveltyMetric
from hypoforge.state import EvidenceGraph, EvidenceNode, EvidenceNodeType, HypothesisCard


def _hypothesis(identifier: str = "H1") -> HypothesisCard:
    return HypothesisCard(
        hypothesis_id=identifier,
        statement="Protein A regulates Protein B.",
    )


def _claim() -> dict[str, object]:
    return {
        "subject": "Protein A",
        "subject_synonyms": [],
        "relation": "regulates",
        "object": "Protein B",
        "object_synonyms": [],
        "claim": "Protein A regulates Protein B.",
    }


@pytest.mark.asyncio
async def test_novelty_trace_for_disconnected_entities_is_strict_json() -> None:
    metric = NoveltyMetric()
    metric._client = object()
    metric._decompose_hypothesis = AsyncMock(return_value=[_claim()])
    graph = EvidenceGraph(
        nodes=[
            EvidenceNode(id="A", type=EvidenceNodeType.ENTITY, label="Protein A"),
            EvidenceNode(id="B", type=EvidenceNodeType.ENTITY, label="Protein B"),
        ]
    )

    result = await metric.compute(_hypothesis(), [], evidence_graph=graph)

    assert isinstance(result, tuple)
    score, trace = result
    assert score == 0.5
    assert trace["claims_novelty"][0]["distance"] is None
    json.dumps(trace, allow_nan=False)


@pytest.mark.asyncio
async def test_novelty_trace_preserves_matches_when_one_concept_is_missing() -> None:
    metric = NoveltyMetric()
    metric._client = object()
    metric._decompose_hypothesis = AsyncMock(return_value=[_claim()])
    graph = EvidenceGraph(
        nodes=[
            EvidenceNode(id="A", type=EvidenceNodeType.ENTITY, label="Protein A"),
        ]
    )

    result = await metric.compute(_hypothesis(), [], evidence_graph=graph)

    assert isinstance(result, tuple)
    score, trace = result
    claim_trace = trace["claims_novelty"][0]
    assert score == 1.0
    assert claim_trace["start_set"] == [{"id": "A", "label": "Protein A"}]
    assert claim_trace["end_set"] == []
    assert claim_trace["distance"] is None
    json.dumps(trace, allow_nan=False)


@pytest.mark.asyncio
@pytest.mark.parametrize("metric_class", [NoveltyMetric, EvidenceConsistencyMetric])
async def test_graph_metric_batch_compute_keeps_float_contract(metric_class: type) -> None:
    metric = object.__new__(metric_class)
    metric.compute = AsyncMock(
        side_effect=[
            (0.25, {"trace": "first"}),
            (0.75, {"trace": "second"}),
        ]
    )

    scores = await metric.batch_compute([_hypothesis("H1"), _hypothesis("H2")], [])

    assert scores == [0.25, 0.75]
    assert all(isinstance(score, float) for score in scores)
