"""Regressions for real pilot failures: access, malformed claims and S2 throttling."""

import asyncio
import json
import time
import urllib.error
from types import SimpleNamespace

import httpx
import pytest
from openai import AuthenticationError, PermissionDeniedError

from hypoforge.benchmarks import agentideabench as bench
from hypoforge.benchmarks import agentideabench_scoring as scoring
from hypoforge.config import PipelineConfig
from hypoforge.evaluation.metrics import NoveltyMetric
from hypoforge.modules.m1_problem_understanding import M1ProblemUnderstanding
from hypoforge.state import EvidenceGraph, EvidenceNode, EvidenceNodeType, HypothesisCard
from hypoforge.tools import semantic_scholar as s2
from hypoforge.tools.qwen_client import QwenClient


def access_error(status=403):
    response = httpx.Response(status, request=httpx.Request("POST", "https://example.test/v1"))
    error_type = PermissionDeniedError if status == 403 else AuthenticationError
    return error_type("Model access denied", response=response, body={"code": "AccessDenied.Unpurchased"})


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 403])
@pytest.mark.parametrize("denied_attempt", [1, 2])
async def test_json_fallback_stops_immediately_on_access_failure(monkeypatch, status, denied_attempt):
    calls = []

    class FakeLLM:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        async def ainvoke(self, messages):
            if len(calls) == denied_attempt:
                raise access_error(status)
            raise ValueError("Unsupported JSON parameter")

    monkeypatch.setattr("hypoforge.tools.qwen_client.ChatOpenAI", FakeLLM)
    client = QwenClient(model="glm-5.1", api_key="test", api_base="https://example.test/v1")
    with pytest.raises((AuthenticationError, PermissionDeniedError)):
        await client.structured_chat(user_prompt="test", disable_thinking=True)
    assert len(calls) == denied_attempt


@pytest.mark.asyncio
async def test_export_text_is_checked_before_other_preflight_calls(monkeypatch):
    calls = []

    class FakeClient:
        async def structured_chat(self, **kwargs):
            calls.append("json")
            return {"ok": True}

        async def chat(self, **kwargs):
            calls.append("text")
            raise access_error()

    async def pdf_check():
        calls.append("pdf")

    monkeypatch.setattr(scoring, "check_pdf_environment", pdf_check)
    monkeypatch.setattr(QwenClient, "from_config", lambda *args: FakeClient())
    monkeypatch.setattr(scoring, "CriticAPI", lambda *args: pytest.fail("Cannot proceed after GLM denial"))
    with pytest.raises(PermissionDeniedError):
        await scoring.preflight(PipelineConfig())
    assert calls == ["pdf", "json", "text"]


@pytest.mark.asyncio
async def test_access_failure_stops_batch_and_preserves_cancelled_checkpoint_paths(tmp_path, monkeypatch):
    from hypoforge import pipeline
    config = PipelineConfig(qwen={"base": {"model": "glm-5.1", "api_key": "test",
                                           "enable_thinking": False, "seed": 42}})
    topic = {"domain": "Biology", "subdomain": "test topic"}
    manifest = {"expected_cells": bench.cells([topic], 4)}
    monkeypatch.setattr(bench, "prepare_manifest", lambda *args: manifest)
    started = []
    both_running = asyncio.Event()

    class FakeRunner:
        def __init__(self, cfg):
            pass

        async def run(self, question, run_id):
            started.append(run_id)
            if len(started) == 1:
                await both_running.wait()
                raise access_error()
            both_running.set()
            await asyncio.Event().wait()

    monkeypatch.setattr(pipeline, "PipelineRunner", FakeRunner)
    with pytest.raises(PermissionDeniedError):
        await bench.generate(tmp_path, tmp_path, config, [topic], 4, workers=2)
    assert len(started) == 2
    records = [bench.read_json(path) for path in (tmp_path / "cells").glob("*/result.json")]
    assert len(records) == 2
    assert any("interrupted" in record["error"] for record in records)
    assert all(record["checkpoint_path"].endswith("_checkpoint.json") for record in records)
    assert bench.read_json(tmp_path / "generation_summary.json")["statuses"] == {
        "pipeline_failed": 2, "pending": 2,
    }
    assert (tmp_path / "submissions.jsonl").read_text() == ""
    assert len((tmp_path / "execution_history.jsonl").read_text().splitlines()) == 1
    with bench.experiment_writer(tmp_path):
        pass


VALID_CLAIM = {"subject": "A", "relation": "affects", "object": "B", "claim": "A affects B."}


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_claims", [
    ["A affects B."],
    [VALID_CLAIM, "malformed second claim"],
    [{**VALID_CLAIM, "subject_components": ["not a component group"]}],
    [{**VALID_CLAIM, "object_synonyms": "B"}],
    [{**VALID_CLAIM, "object": None}],
])
async def test_malformed_claim_batch_requires_repair_without_dropping_rows(bad_claims):
    class FakeClient:
        repairs = 0

        async def structured_chat(self, **kwargs):
            return {"claims": bad_claims}

        async def chat(self, **kwargs):
            self.repairs += 1
            return json.dumps({"claims": [VALID_CLAIM]})

    metric = NoveltyMetric()
    metric._client = FakeClient()
    result = await metric._decompose_hypothesis(HypothesisCard(hypothesis_id="H", statement="A affects B."))
    assert metric._client.repairs == 1
    assert result == [VALID_CLAIM]


@pytest.mark.asyncio
async def test_unrepairable_claims_do_not_crash_or_receive_high_novelty():
    class FakeClient:
        async def structured_chat(self, **kwargs):
            return {"claims": ["salvaged text"]}

        async def chat(self, **kwargs):
            return '{"claims": ["still malformed"]}'

    metric = NoveltyMetric()
    metric._client = FakeClient()
    graph = EvidenceGraph(nodes=[
        EvidenceNode(id="A", type=EvidenceNodeType.ENTITY, label="A"),
        EvidenceNode(id="B", type=EvidenceNodeType.ENTITY, label="B"),
    ])
    score, trace = await metric.compute(
        HypothesisCard(hypothesis_id="H", statement="A affects B."), [], evidence_graph=graph)
    assert score == 0.0
    assert trace["claims_novelty"][0]["assessment"] == "insufficient_graph_coverage"


class Response:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return b'{"data": []}'


def fake_s2_opener(monkeypatch, *, failures=1, retry_after="0"):
    class Opener:
        calls = 0

        def open(self, request, timeout):
            assert "semanticscholar.org" in request.full_url
            self.calls += 1
            if self.calls <= failures:
                raise urllib.error.HTTPError(request.full_url, 429, "Too Many Requests",
                                             {"Retry-After": retry_after}, None)
            return Response()

    opener = Opener()
    waits = []
    monkeypatch.setattr(s2, "_HTTP_OPENER", opener)
    monkeypatch.setattr(s2, "_rate_limit", lambda *args: None)
    monkeypatch.setattr(s2, "_sleep_with_deadline", lambda seconds, deadline: waits.append(seconds))
    monkeypatch.setattr(s2, "_s2_circuit_open_until", 0.0)
    monkeypatch.setattr(s2, "_S2_API_KEY", "test")
    return opener, waits


def test_authenticated_s2_recovers_one_429_without_opening_circuit(monkeypatch):
    opener, waits = fake_s2_opener(monkeypatch)
    assert s2._search("test") == []
    assert opener.calls == 2
    assert waits == [s2._S2_RATE_LIMIT]
    assert not s2._s2_circuit_open()


def test_persistent_s2_429_opens_circuit_after_only_one_retry(monkeypatch):
    opener, waits = fake_s2_opener(monkeypatch, failures=3)
    with pytest.raises(urllib.error.HTTPError) as error:
        s2._search("test")
    assert error.value.code == 429
    assert opener.calls == 2
    assert len(waits) == 1
    assert s2._s2_circuit_open()
    with pytest.raises(RuntimeError, match="circuit is open"):
        s2._search("another test")
    assert opener.calls == 2


@pytest.mark.parametrize("key,retry_after", [("", "0"), ("test", "60")])
def test_anonymous_or_out_of_budget_s2_429_fails_fast(monkeypatch, key, retry_after):
    opener, waits = fake_s2_opener(monkeypatch, retry_after=retry_after)
    with pytest.raises(urllib.error.HTTPError):
        s2._http_get_json("https://api.semanticscholar.org/graph/v1/paper/search",
                          s2_api_key=key, deadline=time.monotonic() + 10)
    assert opener.calls == 1
    assert waits == []


def test_s2_local_timeout_does_not_poison_later_queries(monkeypatch):
    monkeypatch.setattr(s2, "_s2_circuit_open_until", 0.0)

    def timeout(*args, **kwargs):
        raise TimeoutError("Local rate-limit queue exhausted the deadline")

    monkeypatch.setattr(s2, "_s2_search", timeout)
    with pytest.raises(TimeoutError):
        s2._search("test")
    assert not s2._s2_circuit_open()
    monkeypatch.setattr(s2, "_s2_search", lambda *args, **kwargs: [{"paper_id": "test"}])
    assert s2._search("next query") == [{"paper_id": "test"}]


@pytest.mark.asyncio
async def test_m1_still_rejects_fragmentation_and_records_final_reason():
    audit = {"sufficient": True, "core_intent_covered": True, "missing_aspects": [],
             "over_fragmented": True, "merge_instructions": ["Merge parallel outputs"],
             "reason": "The same relation remains split."}
    questions = ["How does A affect B?", "How does A affect C?"]
    payloads = iter([audit, {"sub_questions": questions}, audit])

    class FakeClient:
        async def structured_chat(self, **kwargs):
            return next(payloads)

    module = M1ProblemUnderstanding(mode="llm", coverage_max_rounds=1)
    module.client = FakeClient()
    with pytest.raises(ValueError, match="over-fragmented") as error:
        await module._check_subquestion_coverage("How does A affect B and C?", questions)
    assert audit["reason"] in str(error.value)
    assert questions[0] in str(error.value)
