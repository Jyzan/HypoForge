"""Benchmark protocol, resume isolation, scoring completeness and API parameters."""

import json
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
