"""Regressions for the five-domain SiliconFlow pilot that produced no results."""

import asyncio
import json
import re
import subprocess
import sys
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from hypoforge.benchmarks import agentideabench as bench
from hypoforge.modules.m1_problem_understanding import M1ProblemUnderstanding
from hypoforge.modules.m2_literature.search.query_planner import QueryPlanner, _sanitize_query
from hypoforge.modules.m2_literature.search.round_plan import classify_entities
from hypoforge.modules.m4_hypothesis_generation import M4HypothesisGeneration
from hypoforge.state import HypothesisCard, PipelineState
from hypoforge.tools import semantic_scholar as s2
from hypoforge.tools.qwen_client import QwenClient
from hypoforge.tools import s2_rate_limit as limiter


class SequenceClient:
    def __init__(self, payloads):
        self.payloads = iter(payloads)
        self.calls = []

    async def structured_chat(self, **kwargs):
        self.calls.append(kwargs)
        return next(self.payloads)


PASS = {"sufficient": True, "core_intent_covered": True, "over_fragmented": False}
MERGE = {**PASS, "over_fragmented": True, "merge_instructions": ["Merge related questions"]}
LONG = "How does an intervention influence " + "a precisely measured outcome " * 12 + "?"


@pytest.mark.asyncio
async def test_coverage_merge_overflow_is_semantically_repaired_then_reaudited():
    revised = ["Which CRISPR editing mechanism supports a new therapeutic hypothesis?",
               "Which experiment would falsify that therapeutic hypothesis?"]
    client = SequenceClient([MERGE, {"sub_questions": [LONG]},
                             {"sub_questions": revised}, PASS])
    module = M1ProblemUnderstanding(coverage_max_rounds=1)
    module.client = client
    result = await module._check_subquestion_coverage(
        "Propose a novel, specific, and testable CRISPR therapeutic hypothesis.",
        ["Which CRISPR mechanism could be tested?", "How would it be tested?"],
    )
    assert result == revised
    assert len(client.calls) == 4
    assert LONG in client.calls[2]["user_prompt"]
    assert revised[1] in client.calls[3]["user_prompt"]


@pytest.mark.asyncio
async def test_atomic_repair_still_fails_closed_on_invalid_output():
    module = M1ProblemUnderstanding(coverage_max_rounds=1)
    module.client = SequenceClient([MERGE, {"sub_questions": [LONG]},
                                   {"sub_questions": [LONG]}])
    with pytest.raises(ValueError, match="atomic repair failed.*240"):
        await module._check_subquestion_coverage("Study a mechanism", ["How does A affect B?"])


@pytest.mark.asyncio
async def test_atomic_rewrite_cannot_bypass_core_intent_audit():
    module = M1ProblemUnderstanding(coverage_max_rounds=1)
    module.client = SequenceClient([MERGE, {"sub_questions": [LONG]},
                                   {"sub_questions": ["What is CRISPR?"]},
                                   {**PASS, "core_intent_covered": False}])
    with pytest.raises(ValueError, match="core user intent remains uncovered"):
        await module._check_subquestion_coverage("Propose a CRISPR hypothesis", ["What is CRISPR?"])


@pytest.mark.asyncio
async def test_last_coverage_merge_retains_independently_audited_core_action():
    core = "What novel, specific, and testable hypothesis can be proposed for therapeutic CRISPR base and prime editing?"
    background = "What mechanisms of CRISPR base and prime editing are already characterized?"
    module = M1ProblemUnderstanding(coverage_max_rounds=1)
    module.client = SequenceClient([
        {**MERGE, "core_intent_question_indices": [3]},
        {"sub_questions": [background]},  # The model drops the requested action.
        {**PASS, "core_intent_question_indices": [2]},
    ])
    result = await module._check_subquestion_coverage(
        "Propose a novel, specific, and testable hypothesis for therapeutic CRISPR editing.",
        ["What is known about base editing?", "What is known about prime editing?", core],
    )
    assert result == [background, core]
    assert core in module.client.calls[1]["user_prompt"]
    assert core in module.client.calls[2]["user_prompt"]  # Independent final audit.


@pytest.mark.asyncio
@pytest.mark.parametrize("repair", ["shape", "limit"])
async def test_structural_repair_cannot_discard_audited_core_action(repair):
    core = "What new hypothesis can be proposed for CRISPR therapy?"
    background = "Which CRISPR mechanisms are already characterized?"
    module = M1ProblemUnderstanding()
    module.client = SequenceClient([{"sub_questions": [background]}])
    if repair == "shape":
        result = await module._repair_subquestion_shape(
            "Propose a CRISPR hypothesis", [LONG, core], protected_questions=[core],
        )
    else:
        result = await module._merge_subquestions_to_limit(
            "Propose a CRISPR hypothesis", [f"Background aspect {i}?" for i in range(5)] + [core],
            protected_questions=[core],
        )
    assert result == [background, core]
    assert core in module.client.calls[0]["user_prompt"]


@pytest.mark.asyncio
async def test_protected_action_still_requires_a_passing_final_coverage_audit():
    core = "What new hypothesis can be proposed for CRISPR therapy?"
    module = M1ProblemUnderstanding(coverage_max_rounds=1)
    module.client = SequenceClient([
        {**MERGE, "core_intent_question_indices": [1]},
        {"sub_questions": ["What is CRISPR?"]},
        {**PASS, "core_intent_covered": False},
    ])
    with pytest.raises(ValueError, match="core user intent remains uncovered"):
        await module._check_subquestion_coverage("Propose a CRISPR hypothesis", [core])
    assert core in module.client.calls[-1]["user_prompt"]


@pytest.mark.asyncio
async def test_invalid_core_question_indices_do_not_select_other_questions():
    module = M1ProblemUnderstanding(coverage_max_rounds=1)
    module.client = SequenceClient([
        {**MERGE, "core_intent_question_indices": [0, -1, 99]},
        {"sub_questions": ["What is CRISPR?"]},
        {**PASS, "core_intent_covered": False},
    ])
    with pytest.raises(ValueError, match="core user intent remains uncovered"):
        await module._check_subquestion_coverage(
            "Propose a CRISPR hypothesis", ["How can a unified editing system be designed?"],
        )
    assert "How can a unified editing system be designed?" not in module.client.calls[-1]["user_prompt"]


@pytest.mark.asyncio
async def test_relevance_query_keeps_nerf_alias_without_forced_output_artifacts():
    client = SequenceClient([{"queries": [{"text": 'NeRF AND "scene understanding"',
                                          "tool": "semantic_scholar", "purpose": "methods"}]}])
    planner = QueryPlanner(client, [{"name": "semantic_scholar", "display_name": "S2",
                                      "description": "Paper relevance search"}], strict=True)
    queries = await planner.plan(
        "Propose a new NeRF hypothesis", key_entities=["3D scene understanding neural radiance field"],
        focus_entities=["scientific hypothesis", "mechanistic gap"],
    )
    assert queries[0].text == "NeRF scene understanding"
    assert "scientific hypothesis" not in queries[0].text


def test_relevance_syntax_normalizes_hyphens_and_rejects_unrepresentable_negation():
    assert _sanitize_query('"van-der-Waals" AND graphene[tiab]', "semantic_scholar") == "van der Waals graphene"
    assert _sanitize_query('graphene[tiab] AND cancer', "pubmed") == 'graphene[tiab] AND cancer'
    with pytest.raises(ValueError, match="Boolean NOT"):
        _sanitize_query("cancer NOT melanoma", "semantic_scholar")


@pytest.mark.asyncio
async def test_search_projection_excludes_output_criteria_but_keeps_actual_outcomes():
    client = SequenceClient([{"must_entities": ["hypoxia", "tumor growth"],
                              "unmapped_entities": [],
                              "non_search_entities": ["scientific hypothesis", "testability"]}])
    result = await classify_entities(client, "Propose a testable hypothesis on hypoxia and tumor growth",
                                     ["hypoxia", "tumor growth", "scientific hypothesis", "testability"], [])
    assert result.must_entities == ["hypoxia", "tumor growth"]
    assert result.non_search_entities == ["scientific hypothesis", "testability"]
    assert result.total == 2


def test_shared_limiter_coordinates_separate_processes(tmp_path, monkeypatch):
    monkeypatch.setenv("SEMANTIC_SCHOLAR_RATE_LIMIT_DIR", str(tmp_path))
    code = ("import time; from hypoforge.tools.s2_rate_limit import wait_for_slot; "
            "wait_for_slot('test-key', .08); print(time.monotonic())")
    processes = [subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True) for _ in range(3)]
    values = []
    for process in processes:
        out, err = process.communicate(timeout=10)
        assert process.returncode == 0, err
        values.append(float(out))
    starts = sorted(values)
    assert all(right - left >= .065 for left, right in zip(starts, starts[1:]))
    assert all("test-key" not in p.read_text() for p in tmp_path.iterdir())


def test_queue_timeout_does_not_reserve_or_send_a_request(tmp_path, monkeypatch):
    monkeypatch.setenv("SEMANTIC_SCHOLAR_RATE_LIMIT_DIR", str(tmp_path))
    import time
    limiter.wait_for_slot("key", 5)
    before = limiter._state_path("key").read_text()
    with pytest.raises(TimeoutError, match="shared request queue"):
        limiter.wait_for_slot("key", 5, time.monotonic() + .01)
    assert limiter._state_path("key").read_text() == before


def test_metadata_uses_same_authenticated_transport_without_unlimited_urllib(monkeypatch):
    from hypoforge.modules.m2_literature.reading import access
    calls = []
    monkeypatch.setattr(s2, "_http_get_json", lambda url, **kw: calls.append((url, kw)) or {"paperId": "p"})
    monkeypatch.setattr(access.urllib.request, "urlopen", lambda *a, **kw: pytest.fail("Uncoordinated request"))
    assert access._json_get("https://api.semanticscholar.org/graph/v1/paper/DOI:x",
                           timeout=12, headers={"x-api-key": "test-key"}) == {"paperId": "p"}
    assert calls[0][1]["s2_api_key"] == "test-key"
    assert calls[0][1]["request_timeout"] == 12


@pytest.mark.asyncio
async def test_array_response_prompt_preserves_root_json_schema_references(monkeypatch):
    class FakeLLM:
        def __init__(self, **kwargs):
            pass

        async def ainvoke(self, messages):
            schema, _ = json.JSONDecoder().raw_decode(messages[-1].content.split("Expected JSON schema:\n")[1])
            assert schema["$defs"]["Item"]["properties"]["name"]["type"] == "string"
            assert schema["properties"]["entries"]["items"]["$ref"] == "#/$defs/Item"
            return AIMessage(content='{"entries":[{"name":"valid"}]}')

    monkeypatch.setattr("hypoforge.tools.qwen_client.ChatOpenAI", FakeLLM)
    client = QwenClient(model="glm-5.1", api_key="test")
    result = await client.structured_chat(user_prompt="Return an array", output_schema={
        "$defs": {"Item": {"type": "object", "properties": {"name": {"type": "string"}}}},
        "type": "array", "items": {"$ref": "#/$defs/Item"},
    })
    assert result == [{"name": "valid"}]


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout_second_batch", [False, True])
@pytest.mark.parametrize("reject_second_card", [False, True])
async def test_seven_candidate_portfolio_uses_small_batches_with_unique_ids(monkeypatch, timeout_second_batch, reject_second_card):
    from hypoforge.graph_context import build_graph_context
    module = M4HypothesisGeneration(generator_batch_size=2)
    calls = []

    class Client:
        async def structured_chat(self, **kw):
            count = int(re.findall(r"Generate (\d+) candidate hypotheses", kw["user_prompt"])[-1])
            calls.append(count)
            if timeout_second_batch and len(calls) == 2:
                raise TimeoutError("provider delayed one batch")
            return [{"hypothesis_id": f"H{i}",
                     "statement": f"Treatment {len(calls)}.{i} increases measured protein folding stability."}
                    for i in range(1, count + 1)]

    module.client = Client()
    def check(state, graph, cards, **kw):
        if reject_second_card and len(cards) > 1:
            return cards[:1], [{"hypothesis_id": cards[1].hypothesis_id,
                               "failure_type": "contract_failure"}]
        return cards, []
    monkeypatch.setattr(module, "_check_context_contract", check)
    async def audit(state, cards):
        return cards, []
    monkeypatch.setattr(module, "_audit_context_contract_semantics", audit)
    state = PipelineState(input_question="Propose a protein folding hypothesis")
    _, cards, failures = await module._generate_hypothesis_batch(
        state, question=state.input_question, graph_context=build_graph_context(state),
        feedback_context="", tool_name="generator", attempt=1,
    )
    assert calls == ([2, 2] if timeout_second_batch else [2, 2, 2, 1])
    expected = list(range(1, 3 if timeout_second_batch else 8))
    if reject_second_card:
        expected = [i for i in expected if i % 2]
    assert [card.hypothesis_id for card in cards] == [f"H{i}" for i in expected]
    assert [f["hypothesis_id"] for f in failures if f.get("failure_type") == "contract_failure"] == (
        ["H2"] if timeout_second_batch and reject_second_card
        else ["H2", "H4", "H6"] if reject_second_card else []
    )
    if timeout_second_batch:
        assert failures[-1]["failure_type"] == "portfolio_batch_timeout"


@pytest.mark.asyncio
async def test_benchmark_generation_stops_before_deterministic_continuity_fallback(monkeypatch):
    module = M4HypothesisGeneration(allow_continuity_fallback=False)
    module.client = object()
    async def empty(*args, **kwargs):
        return [], [], []
    async def repair(*args, **kwargs):
        return [], []
    monkeypatch.setattr(module, "_generate_hypothesis_batch", empty)
    monkeypatch.setattr(module, "_repair_context_contract", repair)
    with pytest.raises(RuntimeError, match="continuity fallback is disabled"):
        await module._run_llm(PipelineState(input_question="Propose a protein folding hypothesis"))


def test_benchmark_never_exports_a_deterministic_fallback():
    state = SimpleNamespace(errors=[], top_hypotheses=[HypothesisCard(
        hypothesis_id="FH1", statement="A systematic investigation is needed",
        mechanism="Fast-mode fallback: investigate the problem",
    )], reviews=[object()])
    with pytest.raises(ValueError, match="not a model-generated"):
        bench.select_submission(state)
