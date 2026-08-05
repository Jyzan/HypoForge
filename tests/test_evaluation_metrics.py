from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from hypoforge.evaluation.metrics import (
    AsyncMaaSEmbeddings,
    EvidenceConsistencyMetric,
    NoveltyMetric,
)
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
    assert score == 1.0
    assert trace["claims_novelty"][0]["distance"] is None
    json.dumps(trace, allow_nan=False)


def test_keyword_search_can_disable_metadata_matching() -> None:
    metric = NoveltyMetric()
    node = EvidenceNode(
        id="A",
        type=EvidenceNodeType.ENTITY,
        label="Unrelated entity",
        metadata={"searchable_text": "Protein A regulates survival"},
    )

    assert metric._keyword_search(["Protein A"], [node], search_metadata=False) == []
    assert metric._keyword_search(["Protein A"], [node], search_metadata=True) == [node]


@pytest.mark.asyncio
async def test_novelty_requires_every_composite_entity_component() -> None:
    metric = NoveltyMetric()
    metric._client = object()
    claim = _claim()
    claim["subject_components"] = [["Protein A"], ["Kinase X"]]
    claim["object_components"] = [["Protein B"]]
    metric._decompose_hypothesis = AsyncMock(return_value=[claim])
    graph = EvidenceGraph(
        nodes=[
            EvidenceNode(id="A", type=EvidenceNodeType.ENTITY, label="Protein A"),
            EvidenceNode(id="B", type=EvidenceNodeType.ENTITY, label="Protein B"),
        ]
    )

    score, trace = await metric.compute(_hypothesis(), [], evidence_graph=graph)

    assert score == 1.0
    assert trace["claims_novelty"][0]["distance"] is None
    assert trace["claims_novelty"][0]["start_components"][1]["matched_nodes"] == []
    json.dumps(trace, allow_nan=False)


def test_component_queries_fall_back_to_legacy_subject_object_fields() -> None:
    assert EvidenceConsistencyMetric._claim_component_queries(_claim()) == [
        "Protein A",
        "Protein B",
    ]


def test_full_evaluation_config_controls_embedding_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_EMBEDDING_API_KEY", "test-key")
    metric = EvidenceConsistencyMetric(embed_config={
        "embedding": {
            "model_name": "embedding-model",
            "api_key_env_var": "TEST_EMBEDDING_API_KEY",
            "base_url": "https://embedding.example/v1/",
        },
        "consistency": {"similarity_threshold": 0.42},
    })

    assert metric.similarity_threshold == 0.42
    assert isinstance(metric.embeddings, AsyncMaaSEmbeddings)
    assert metric.embeddings.model == "embedding-model"
    assert metric.embeddings.base_url == "https://embedding.example/v1"


@pytest.mark.asyncio
async def test_embedding_client_replaces_blank_inputs_and_sorts_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import httpx

    calls: list[dict[str, object]] = []

    class _Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                "data": [
                    {"index": 1, "embedding": [2.0]},
                    {"index": 0, "embedding": [1.0]},
                ]
            }

    class _AsyncClient:
        async def __aenter__(self) -> "_AsyncClient":
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        async def post(self, url: str, **kwargs: object) -> _Response:
            calls.append({"url": url, **kwargs})
            return _Response()

    monkeypatch.setattr(httpx, "AsyncClient", _AsyncClient)
    client = AsyncMaaSEmbeddings("model", "key", "https://example.test/v1/")

    embeddings = await client._call_api(["", "Protein A"])

    assert embeddings == [[1.0], [2.0]]
    assert calls[0]["url"] == "https://example.test/v1/embeddings"
    assert calls[0]["json"] == {"model": "model", "input": [" ", "Protein A"]}


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
