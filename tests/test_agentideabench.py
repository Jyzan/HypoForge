"""Benchmark protocol, resume isolation, scoring completeness and API parameters."""

import asyncio
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from hypoforge.benchmarks import agentideabench as bench
from hypoforge.benchmarks import agentideabench_scoring as scoring
from hypoforge.config import LLMConfig, PipelineConfig
from hypoforge.tools.credentials import read_api_key_csv
from hypoforge.tools.qwen_client import QwenClient


@pytest.mark.parametrize("content", [
    "id,123\napiKey,test-key\nopenAiCompatible,https://example.test/v1\n",
    "apiKey,openAiCompatible\ntest-key,https://example.test/v1\n",
    "api_key,base_url\ntest-key,https://example.test/v1\n",
])
def test_credentials_export_layouts(tmp_path, content):
    path = tmp_path / "key.csv"
    path.write_text(content, encoding="utf-8-sig")
    assert read_api_key_csv(path) == ("test-key", "https://example.test/v1")


def test_credentials_reject_multiple_records_and_insecure_endpoint(tmp_path):
    path = tmp_path / "key.csv"
    path.write_text("apiKey,openAiCompatible\na,https://example.test/v1\nb,https://example.test/v1\n")
    with pytest.raises(ValueError, match="single"):
        read_api_key_csv(path)
    path.write_text("apiKey,test-key\nopenAiCompatible,http://example.test/v1\n")
    with pytest.raises(ValueError, match="HTTPS"):
        read_api_key_csv(path)


@pytest.mark.asyncio
async def test_explicit_thinking_and_seed_survive_json_fallback(monkeypatch):
    calls = []

    class FakeLLM:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        async def ainvoke(self, messages):
            if len(calls) < 3:
                raise ValueError("JSON mode rejected")
            return SimpleNamespace(content='{"ok": true}', response_metadata={}, usage_metadata={})

    monkeypatch.setattr("hypoforge.tools.qwen_client.ChatOpenAI", FakeLLM)
    client = QwenClient(model="glm-5.1", api_key="test", api_base="https://example.test/v1",
                        enable_thinking=False, seed=42)
    assert await client.structured_chat(user_prompt="test", disable_thinking=True) == {"ok": True}
    assert len(calls) == 3
    assert all(call["extra_body"] == {"enable_thinking": False} and call["seed"] == 42 for call in calls)


def test_model_assignment_carries_experiment_parameters():
    config = PipelineConfig(qwen={"base": {"model": "glm-5.1", "enable_thinking": False, "seed": 42}})
    assert all(tier.model == "glm-5.1" and tier.enable_thinking is False and tier.seed == 42
               for tier in (config.qwen.base, config.qwen.max, config.qwen.plus, config.qwen.turbo))
    assert all("api_key" not in tier for tier in bench.public_config(config)["qwen"].values())


def test_pilot_balances_disciplines(monkeypatch):
    topics = [{"domain": d, "subdomain": f"{d} topic {i}"} for d in bench.DOMAINS for i in range(8)]
    monkeypatch.setattr(bench, "scored_topics", lambda root: topics)
    pilot = bench.select_topics(Path("unused"), "pilot")
    assert [topic["domain"] for topic in pilot] == list(bench.DOMAINS)
    assert len({cell["item_id"] for cell in bench.cells(pilot, 3)}) == 15


@pytest.mark.parametrize("text", ["short", "word " * 151, "word " * 80 + "\nword", "word " * 80 + "中文"])
def test_reject_noncompliant_export(text):
    with pytest.raises(ValueError):
        bench.validate_paragraph(text)


def test_final_choice_uses_internal_ranking_not_external_scores():
    class Card:
        def __init__(self, identifier):
            self.hypothesis_id = identifier

        def model_dump(self, **kwargs):
            return {"hypothesis_id": self.hypothesis_id, "statement": "claim", "scores": {"originality": 10}}

    card1, card2 = Card("h1"), Card("h2")
    state = SimpleNamespace(errors=[], reviews=["review"], top_hypotheses=[card1, card2],
                            research_plans=[card1, card2])
    assert bench.select_submission(state)["hypothesis"]["hypothesis_id"] == "h1"
    assert "scores" not in bench.select_submission(state)["hypothesis"]


@pytest.mark.asyncio
async def test_independent_runs_resume_without_generating_again(tmp_path, monkeypatch):
    from hypoforge import pipeline
    observed = []
    config = PipelineConfig(qwen={"base": {"model": "glm-5.1", "api_key": "test",
                                           "enable_thinking": False, "seed": 42}})
    topic = {"domain": "Biology", "subdomain": "test topic"}
    manifest = {"expected_cells": bench.cells([topic], 3)}
    monkeypatch.setattr(bench, "prepare_manifest", lambda *args: manifest)
    monkeypatch.setattr(bench, "select_submission", lambda state: {"hypothesis": {"statement": "claim"}})

    class FakeRunner:
        def __init__(self, cfg):
            self.config = cfg

        async def run(self, question, run_id):
            observed.append((self.config.memory_cache_dir, self.config.entity_cache_dir, run_id, self.config.qwen.base.seed))
            return SimpleNamespace(total_input_tokens=5, total_output_tokens=7, token_usage_by_module={},
                                   iteration_count=1, search_round=1, literature_results=[], m2_knowledge_export=None,
                                   search_ledger=SimpleNamespace(model_dump=lambda **kw: {}))

    async def fake_export(config, payload):
        return "I will " + "test " * 98, {"input": 1, "output": 2, "calls": 1}

    monkeypatch.setattr(pipeline, "PipelineRunner", FakeRunner)
    monkeypatch.setattr(bench, "export_submission", fake_export)
    assert await bench.generate(tmp_path, tmp_path, config, [topic], 3)
    assert len(observed) == 3
    assert len({row[0] for row in observed}) == 3
    assert len({row[1] for row in observed}) == 3
    assert [row[3] for row in observed] == [42, 43, 44]
    assert await bench.generate(tmp_path, tmp_path, config, [topic], 3)
    assert len(observed) == 3
    assert len((tmp_path / "submissions.jsonl").read_text().splitlines()) == 3


def test_evidence_balances_queries_and_freezes_abstracts(monkeypatch):
    monkeypatch.setattr(scoring, "extract_queries", lambda *args: ["q1", "q2"])
    monkeypatch.setattr(scoring.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(scoring, "search_prior_art", lambda query: [
        {"paperId": f"{query}-{i}", "title": query, "abstract": "a" * 600} for i in range(10)])
    evidence = scoring.build_evidence(None, None, None, "idea")
    assert [paper["paperId"] for paper in evidence["evidence"]] == [
        "q1-0", "q2-0", "q1-1", "q2-1", "q1-2", "q2-2", "q1-3", "q2-3"]
    assert all(len(paper["abstract"]) == 450 for paper in evidence["evidence"])
    assert evidence["cutoff"] == "2026-05-31"


def test_search_failure_is_not_empty_prior_art(monkeypatch):
    import httpx

    class FakeHTTP:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get(self, url, **kwargs):
            return httpx.Response(429, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "Client", FakeHTTP)
    monkeypatch.setattr(scoring.time, "sleep", lambda seconds: None)
    with pytest.raises(httpx.HTTPStatusError):
        scoring.search_prior_art("query")


def scoring_fixture(tmp_path, monkeypatch):
    topic = {"domain": "Biology", "subdomain": "test topic"}
    cell = bench.cells([topic], 1)[0]
    bench.write_json(tmp_path / "manifest.json", {"fingerprint": "test", "protocol": {"repeats": 1},
                                                  "expected_cells": [cell]})
    bench.write_json(tmp_path / "score_manifest.json", {"generation_fingerprint": "test", "base_url": "test"})
    bench.write_json(tmp_path / "cells" / cell["item_id"] / "result.json", {"status": "success", "idea_text": "idea"})
    baseline = {(topic["domain"], topic["subdomain"], i): {d: 5 for d in bench.WEIGHTS} for i in (1, 2, 3)}
    monkeypatch.setattr(bench, "historical_baseline", lambda root: baseline)
    return cell


def test_aggregation_trims_per_dimension(tmp_path, monkeypatch):
    cell = scoring_fixture(tmp_path, monkeypatch)
    for critic, originality, feasibility in zip(bench.CRITICS, (9, 8, 7), (1, 10, 3)):
        ratings = {d: 4 for d in bench.WEIGHTS}
        ratings.update(originality=originality, feasibility=feasibility)
        bench.write_json(tmp_path / "scores" / cell["item_id"] / f"{critic}.json",
                         {"scores": ratings, "idea_sha256": bench.digest("idea")})
    assert bench.analyze(tmp_path, tmp_path)
    report = bench.read_json(tmp_path / "comparison.json")
    assert report["dimensions"]["originality"] == 7.5
    assert report["dimensions"]["feasibility"] == 2
    assert report["weighted_total"] == pytest.approx(27 / 5.5)
    assert report["historical_baseline"] == 5
    assert report["table2_comparable_sample"] is False


def test_missing_critic_prevents_headline_score(tmp_path, monkeypatch):
    cell = scoring_fixture(tmp_path, monkeypatch)
    bench.write_json(tmp_path / "scores" / cell["item_id"] / "glm-5.1.json",
                     {"scores": {d: 5 for d in bench.WEIGHTS}, "idea_sha256": bench.digest("idea")})
    assert not bench.analyze(tmp_path, tmp_path)
    report = bench.read_json(tmp_path / "comparison.json")
    assert "weighted_total" not in report


def test_upstream_table2_baseline_when_release_is_present():
    root = Path(__file__).resolve().parents[2] / "AgentIdeaBench"
    if not (root / "release_data/core/lit8d_scores_3seed.csv.gz").exists():
        pytest.skip("Optional adjacent AgentIdeaBench checkout is absent")
    topics = bench.scored_topics(root)
    assert len(topics) == 40
    scores = bench.historical_baseline(root)
    assert len(scores) == 120
    total = sum(bench.weighted(value) for value in scores.values()) / len(scores)
    assert total == pytest.approx(6.3326, abs=0.0001)


@pytest.mark.asyncio
async def test_parallel_generation_bounds_requests_and_isolates_usage(tmp_path, monkeypatch):
    from hypoforge import pipeline
    from hypoforge.tools.qwen_client import track_token_usage
    config = PipelineConfig(qwen={"base": {"model": "glm-5.1", "api_key": "test",
                                           "enable_thinking": False, "seed": 42}})
    topic = {"domain": "CS", "subdomain": "test topic"}
    manifest = {"expected_cells": bench.cells([topic], 3)}
    monkeypatch.setattr(bench, "prepare_manifest", lambda *args: manifest)
    counters = {"cases": 0, "case_peak": 0, "m2": 0, "m2_peak": 0, "llm": 0, "llm_peak": 0}
    observed = []

    class FakeLLM:
        def __init__(self, **kwargs):
            self.seed = kwargs["seed"]

        async def ainvoke(self, messages):
            counters["llm"] += 1
            counters["llm_peak"] = max(counters["llm_peak"], counters["llm"])
            await asyncio.sleep(0.01)
            counters["llm"] -= 1
            return SimpleNamespace(content="OK", response_metadata={},
                                   usage_metadata={"input_tokens": self.seed, "output_tokens": 1})

    class FakeRunner:
        def __init__(self, cfg):
            self.config = cfg

        def _make_node_wrapper(self, name, module):
            async def node(state):
                counters["m2"] += 1
                counters["m2_peak"] = max(counters["m2_peak"], counters["m2"])
                await asyncio.sleep(0.01)
                counters["m2"] -= 1
                return {}
            return node

        async def run(self, question, run_id):
            counters["cases"] += 1
            counters["case_peak"] = max(counters["case_peak"], counters["cases"])
            observed.append((self.config.memory_cache_dir, self.config.entity_cache_dir, run_id))
            with track_token_usage() as usage:
                await self._make_node_wrapper("m2", None)(None)
                await QwenClient.from_config(self.config.qwen.base).chat(user_prompt="test")
                snapshot = usage.snapshot()
            counters["cases"] -= 1
            return SimpleNamespace(total_input_tokens=snapshot["input"], total_output_tokens=snapshot["output"],
                                   token_usage_by_module={"m1": snapshot}, iteration_count=1, search_round=1,
                                   literature_results=[], m2_knowledge_export=None,
                                   search_ledger=SimpleNamespace(model_dump=lambda **kw: {}))

    async def fake_export(config, payload):
        return "I will " + "test " * 98, {"input": 1, "output": 2, "calls": 1}

    monkeypatch.setattr("hypoforge.tools.qwen_client.ChatOpenAI", FakeLLM)
    monkeypatch.setattr(pipeline, "PipelineRunner", FakeRunner)
    monkeypatch.setattr(bench, "select_submission", lambda state: {"hypothesis": {"statement": "claim"}})
    monkeypatch.setattr(bench, "export_submission", fake_export)
    assert await bench.generate(tmp_path, tmp_path, config, [topic], 3, workers=2, llm_concurrency=1)
    assert counters["case_peak"] == 2
    assert counters["m2_peak"] == counters["llm_peak"] == 1
    rows = [json.loads(line) for line in (tmp_path / "submissions.jsonl").read_text().splitlines()]
    assert [row["pipeline_usage"]["input"] for row in rows] == [42, 43, 44]
    assert all(row["pipeline_usage"]["calls"] == 1 for row in rows)
    assert len({row[0] for row in observed}) == len({row[1] for row in observed}) == 3
    assert await bench.generate(tmp_path, tmp_path, config, [topic], 3, workers=4)
    assert len(observed) == 3
    executions = [json.loads(line) for line in (tmp_path / "execution_history.jsonl").read_text().splitlines()]
    assert [entry["workers"] for entry in executions] == [2, 4]


def test_experiment_rejects_another_writer_and_releases_lock(tmp_path):
    with bench.experiment_writer(tmp_path):
        with pytest.raises(ValueError, match="Another generator/scorer"):
            with bench.experiment_writer(tmp_path):
                pass
    with bench.experiment_writer(tmp_path):
        pass


@pytest.mark.parametrize("failed_critic", [None, bench.CRITICS[1]])
def test_parallel_critics_share_one_evidence_and_resume_only_failures(tmp_path, monkeypatch, failed_critic):
    topic = {"domain": "CS", "subdomain": "test topic"}
    cell = bench.cells([topic], 1)[0]
    bench.write_json(tmp_path / "manifest.json", {"fingerprint": "test", "expected_cells": [cell]})
    bench.write_json(tmp_path / "cells" / cell["item_id"] / "result.json",
                     {"status": "success", "idea_text": "idea"})
    rubric = SimpleNamespace(LIT8D_SYSTEM="rubric", EXTRACT_SYSTEM="extract", EXTRACT_TEMPLATE="queries")
    scorer = SimpleNamespace(USER_TEMPLATE="prompt")
    monkeypatch.setattr(scoring, "load_rubric", lambda root: (rubric, scorer))
    barrier = threading.Barrier(3)
    observations, clients = [], []
    evidence_calls = []
    retrying = False

    class FakeAPI:
        def __init__(self, config):
            self.usage = {"input": 0, "output": 0, "calls": 0}
            self.closed = False
            self.client = SimpleNamespace(close=lambda: setattr(self, "closed", True))
            clients.append(self)

    def fake_evidence(api, rubric, scorer, idea):
        evidence_calls.append(idea)
        api.usage.update(input=7, output=1, calls=1)
        return {"queries": ["q"], "evidence": [], "idea_sha256": bench.digest(idea)}

    def fake_judge(api, rubric, scorer, model, idea, domain, evidence):
        if not retrying:
            barrier.wait(timeout=3)
        assert evidence["idea_sha256"] == bench.digest(idea)
        assert (tmp_path / "scores" / cell["item_id"] / "evidence.json").exists()
        observations.append((model, id(api)))
        api.usage.update(input=10 + bench.CRITICS.index(model), output=2, calls=1)
        if model == failed_critic and not retrying:
            return {"error": "invalid critic response"}
        return {"scores": {dimension: 5 for dimension in bench.WEIGHTS}, "responses": ["raw"]}

    monkeypatch.setattr(scoring, "CriticAPI", FakeAPI)
    monkeypatch.setattr(scoring, "build_evidence", fake_evidence)
    monkeypatch.setattr(scoring, "judge", fake_judge)
    config = SimpleNamespace(api_base="https://example.test/v1")
    assert scoring.score(tmp_path, tmp_path, config, workers=3) is (failed_critic is None)
    assert len({identifier for _, identifier in observations}) == 3
    for index, model in enumerate(bench.CRITICS):
        result = bench.read_json(tmp_path / "scores" / cell["item_id"] / f"{model}.json")
        assert result["usage"] == {"input": 10 + index, "output": 2, "calls": 1}
    retrying = True
    assert scoring.score(tmp_path, tmp_path, config, retry_failed=True, workers=1)
    assert len(observations) == 3 + int(failed_critic is not None)
    assert evidence_calls == ["idea"]
    assert all(client.closed for client in clients)


@pytest.mark.asyncio
async def test_cancelled_parallel_generation_persists_failures_and_unlocks(tmp_path, monkeypatch):
    from hypoforge import pipeline
    config = PipelineConfig(qwen={"base": {"model": "glm-5.1", "api_key": "test",
                                           "enable_thinking": False, "seed": 42}})
    topic = {"domain": "CS", "subdomain": "test topic"}
    manifest = {"expected_cells": bench.cells([topic], 3)}
    monkeypatch.setattr(bench, "prepare_manifest", lambda *args: manifest)
    both_started = asyncio.Event()
    blocked = asyncio.Event()
    active = 0

    class FakeRunner:
        def __init__(self, config):
            pass

        async def run(self, *args, **kwargs):
            nonlocal active
            active += 1
            if active == 2:
                both_started.set()
            await blocked.wait()

    monkeypatch.setattr(pipeline, "PipelineRunner", FakeRunner)
    task = asyncio.create_task(bench.generate(tmp_path, tmp_path, config, [topic], 3, workers=2))
    await asyncio.wait_for(both_started.wait(), timeout=3)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    summary = bench.read_json(tmp_path / "generation_summary.json")
    assert summary["statuses"] == {"pipeline_failed": 2, "pending": 1}
    for cell in manifest["expected_cells"][:2]:
        result = bench.read_json(tmp_path / "cells" / cell["item_id"] / "result.json")
        assert "interrupted" in result["error"]
    with bench.experiment_writer(tmp_path):
        pass
