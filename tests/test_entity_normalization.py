from __future__ import annotations

import pytest

from hypoforge.entity_normalization import EntityNormalizationService
from hypoforge.state import TaskContract, TaskEntity


class SameConceptClient:
    def __init__(self, same: bool = True) -> None:
        self.same = same
        self.calls = 0

    async def structured_chat(self, **kwargs):
        self.calls += 1
        request = __import__("json").loads(kwargs["user_prompt"])[0]
        return {"decisions": [{
            "surface": request["surface"],
            "canonical_name": request["candidates"][0],
            "same_concept": self.same,
            "rationale": "identity judgment",
        }]}


class FixedEmbeddings:
    async def aembed_documents(self, texts):
        # Every surface is recalled as a candidate. The LLM, not this score,
        # remains responsible for the merge decision.
        return [[1.0, 0.0] for _ in texts]


def contract() -> TaskContract:
    return TaskContract(entities=[TaskEntity(
        entity_id="E1",
        name="区域气候模型",
        aliases=["regional climate model"],
        role="primary_object",
        required=True,
    )])


@pytest.mark.asyncio
async def test_task_aliases_have_stable_cross_run_canonical_id(tmp_path) -> None:
    first = EntityNormalizationService.from_task(
        contract(), domains=["climate science"], cache_dir=tmp_path,
    )
    first_result = await first.resolve_batch([
        "区域气候模型", "regional climate model",
    ])
    second = EntityNormalizationService.from_task(
        contract(), domains=["climate science"], cache_dir=tmp_path,
    )
    second_result = await second.resolve_batch(["regional climate model"])

    assert first_result["区域气候模型"].canonical_id == (
        first_result["regional climate model"].canonical_id
    )
    assert second_result["regional climate model"].canonical_id == (
        first_result["regional climate model"].canonical_id
    )


@pytest.mark.asyncio
async def test_embedding_similarity_never_merges_without_identity_judgment(tmp_path) -> None:
    service = EntityNormalizationService.from_task(
        contract(),
        domains=["climate science"],
        cache_dir=tmp_path,
        embedding_backend=FixedEmbeddings(),
    )

    result = await service.resolve_batch(["global climate model"])

    assert result["global climate model"].canonical_name == "global climate model"
    assert result["global climate model"].canonical_name != "区域气候模型"


@pytest.mark.asyncio
async def test_llm_pair_decision_merges_and_is_cached(tmp_path) -> None:
    client = SameConceptClient(same=True)
    first = EntityNormalizationService.from_task(
        contract(),
        domains=["climate science"],
        cache_dir=tmp_path,
        client=client,
        embedding_backend=FixedEmbeddings(),
    )
    first_result = await first.resolve_batch(["RCM climate model"])
    assert first_result["RCM climate model"].canonical_name == "区域气候模型"
    assert client.calls == 1

    second_client = SameConceptClient(same=True)
    second = EntityNormalizationService.from_task(
        contract(),
        domains=["climate science"],
        cache_dir=tmp_path,
        client=second_client,
        embedding_backend=FixedEmbeddings(),
    )
    second_result = await second.resolve_batch(["RCM climate model"])

    assert second_result["RCM climate model"].canonical_id == (
        first_result["RCM climate model"].canonical_id
    )
    assert second_client.calls == 0


@pytest.mark.asyncio
async def test_negative_identity_decision_keeps_related_entities_separate(tmp_path) -> None:
    client = SameConceptClient(same=False)
    service = EntityNormalizationService.from_task(
        contract(),
        domains=["climate science"],
        cache_dir=tmp_path,
        client=client,
        embedding_backend=FixedEmbeddings(),
    )

    result = await service.resolve_batch(["regional rainfall bias"])

    assert result["regional rainfall bias"].canonical_name == "regional rainfall bias"
    assert client.calls == 1
