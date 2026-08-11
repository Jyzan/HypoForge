from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from hypoforge.entity_normalization import (
    EntityEmbeddingTransportError,
    EntityNormalizationService,
    _embed_raw,
)
from hypoforge.observability import RunEventRecorder, bind_recorder
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


@pytest.mark.asyncio
async def test_embedding_transport_retries_without_environment_proxy(monkeypatch) -> None:
    attempted_routes: list[bool] = []

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"data": [{"embedding": [1.0, 0.0]}]}

    class FakeAsyncClient:
        def __init__(self, *, timeout: float, trust_env: bool = True) -> None:
            self.trust_env = trust_env
            attempted_routes.append(trust_env)

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback) -> None:
            return None

        async def post(self, endpoint, *, headers, json):
            if self.trust_env:
                raise httpx.ConnectError("broken local proxy")
            return FakeResponse()

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    vectors = await _embed_raw(
        model="text-embedding-v3",
        inputs=["YOLO model"],
        base_url="https://example.invalid/v1",
        api_key="test-key",
    )

    assert vectors == [[1.0, 0.0]]
    assert attempted_routes == [True, False]


@pytest.mark.asyncio
async def test_auto_embedding_transport_outage_degrades_to_lexical_matching(
    monkeypatch,
    tmp_path,
) -> None:
    async def fail_transport(**kwargs):
        raise EntityEmbeddingTransportError("both network routes failed")

    monkeypatch.setattr(
        "hypoforge.entity_normalization._embed_raw",
        fail_transport,
    )
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "secret-test-token")
    recorder = RunEventRecorder(tmp_path / "run", "run-embedding-degraded")
    service = EntityNormalizationService(
        cache_dir=tmp_path / "cache",
        embedding_model="text-embedding-v3",
    )
    service.register_alias_group("known entity", [])

    with bind_recorder(recorder):
        result = await service.resolve_batch(["new entity"])

    assert result["new entity"].canonical_name == "new entity"
    events = recorder.read_events()
    degraded = [
        event for event in events
        if event["event_type"] == "entity_embedding_degraded"
    ]
    assert len(degraded) == 1
    assert degraded[0]["status"] == "warning"
    assert "secret-test-token" not in str(degraded[0])


def test_entity_cache_retries_transient_windows_replace_error(
    monkeypatch,
    tmp_path,
) -> None:
    original_replace = Path.replace
    replace_calls = 0

    def flaky_replace(path: Path, target: Path):
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls == 1:
            raise PermissionError("simulated transient Windows file lock")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", flaky_replace)
    service = EntityNormalizationService(
        cache_dir=tmp_path,
        namespace="portable-cache",
    )

    service.register_alias_group("YOLO model", [])

    assert replace_calls == 2
    assert service.cache_path is not None
    assert service.cache_path.exists()


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
    # Same explicit stamp = same logical run: both constructions must share
    # one cache file (canonical ids are namespace-derived and stable anyway).
    first = EntityNormalizationService.from_task(
        contract(), domains=["climate science"], cache_dir=tmp_path,
        run_stamp="run-A",
    )
    first_result = await first.resolve_batch([
        "区域气候模型", "regional climate model",
    ])
    second = EntityNormalizationService.from_task(
        contract(), domains=["climate science"], cache_dir=tmp_path,
        run_stamp="run-A",
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
        run_stamp="run-B",
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
        run_stamp="run-B",
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


# ---- fail-closed regression tests ----


class FailingEmbeddings:
    async def aembed_documents(self, texts):
        raise ConnectionError("simulated embedding API outage")


@pytest.mark.asyncio
async def test_configured_embedding_failure_propagates(tmp_path) -> None:
    """Embedding explicitly configured → failure must raise, not silently lexical."""
    service = EntityNormalizationService(
        cache_dir=tmp_path,
        embedding_backend=FailingEmbeddings(),
    )
    service.register_alias_group("known-entity", [])
    with pytest.raises(RuntimeError, match="Entity embedding failed"):
        await service.resolve_batch(["test surface"])


@pytest.mark.asyncio
async def test_embedding_not_configured_is_lexical_only(tmp_path) -> None:
    """No embedding configured → lexical-only is a legitimate mode."""
    service = EntityNormalizationService(cache_dir=tmp_path)
    result = await service.resolve_batch(["test surface"])
    assert "test surface" in result


@pytest.mark.asyncio
async def test_embedding_malformed_vector_count_propagates(tmp_path) -> None:
    """Embedding returns wrong count → must raise."""
    class WrongCountEmbeddings:
        async def aembed_documents(self, texts):
            return [[0.1, 0.2]]  # 1 vector for 2+ texts

    service = EntityNormalizationService(
        cache_dir=tmp_path,
        embedding_backend=WrongCountEmbeddings(),
    )
    service.register_alias_group("alpha", [])
    with pytest.raises(RuntimeError, match="Embedding backend returned"):
        await service.resolve_batch(["beta"])


class FailingJudgeClient:
    def __init__(self) -> None:
        self.calls = 0

    async def structured_chat(self, **kwargs):
        self.calls += 1
        raise ConnectionError("simulated judge API outage")


@pytest.mark.asyncio
async def test_judge_failure_propagates(tmp_path) -> None:
    """Configured judge fails → must raise."""
    service = EntityNormalizationService(
        cache_dir=tmp_path,
        client=FailingJudgeClient(),
        embedding_backend=FixedEmbeddings(),
    )
    service.register_alias_group("alpha", [])
    with pytest.raises(RuntimeError, match="Entity identity judge"):
        await service.resolve_batch(["beta"])


@pytest.mark.asyncio
async def test_judge_failure_writes_no_false_negative_cache(tmp_path) -> None:
    """Judge unavailable → must NOT write same_concept=False decisions."""
    service = EntityNormalizationService(
        cache_dir=tmp_path,
        client=None,
    )
    service.register_alias_group("robotic arm", ["robot manipulator"])
    await service.resolve_batch(["mechanical arm", "robot gripper"])
    # No client → no pair decisions should have been written
    # because the for loop skips when selected is None
    assert len(service.pair_decisions) == 0


# ---- per-run cache isolation (time-stamped cache file names) ----


@pytest.mark.asyncio
async def test_two_runs_same_namespace_get_distinct_cache_files(tmp_path) -> None:
    """Same namespace, two constructions without an explicit stamp →
    different cache files, each embedding the current-time stamp."""
    import re

    first = EntityNormalizationService(
        cache_dir=tmp_path, namespace="robotic arm",
    )
    second = EntityNormalizationService(
        cache_dir=tmp_path, namespace="robotic arm",
    )
    assert first.cache_path is not None and second.cache_path is not None
    assert first.cache_path != second.cache_path
    stamp_pattern = re.compile(r"\d{8}-\d{6}-\d{6}")
    assert stamp_pattern.search(first.cache_path.name)
    assert stamp_pattern.search(second.cache_path.name)
    # Namespace digest prefix stays stable across runs.
    assert first.cache_path.name.split("-")[0] == second.cache_path.name.split("-")[0]


@pytest.mark.asyncio
async def test_explicit_run_stamp_is_consistent_within_one_run(tmp_path) -> None:
    """An explicit stamp (e.g. the pipeline run_id) is embedded verbatim and
    shared by every construction inside the same run."""
    stamp = "ui-20260811-120000-abcd12"
    first = EntityNormalizationService.from_task(
        contract(), domains=["climate science"], cache_dir=tmp_path,
        run_stamp=stamp,
    )
    second = EntityNormalizationService.from_task(
        contract(), domains=["climate science"], cache_dir=tmp_path,
        run_stamp=stamp,
    )
    assert first.cache_path is not None
    assert first.cache_path == second.cache_path
    assert stamp in first.cache_path.name


@pytest.mark.asyncio
async def test_old_unstamped_cache_file_is_not_hit(tmp_path) -> None:
    """A legacy ``{digest}.json`` file written by the old naming scheme must
    no longer be loaded once time-stamped naming is in effect."""
    legacy = EntityNormalizationService(cache_dir=tmp_path, namespace="legacy ns")
    legacy.register_alias_group("legacy concept", [])
    legacy_path = legacy.cache_path.parent / (
        legacy.cache_path.name.split("-")[0] + ".json"
    )
    legacy_path.write_text(
        legacy.cache_path.read_text(encoding="utf-8"), encoding="utf-8"
    )

    fresh = EntityNormalizationService(cache_dir=tmp_path, namespace="legacy ns")
    assert fresh.cache_path is not None
    assert fresh.cache_path != legacy_path
    assert "legacy concept" not in fresh.alias_to_id
