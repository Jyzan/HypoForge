"""Behavior tests for mandatory LLM-based M1 follow-up triage."""

from __future__ import annotations

from typing import Any

import pytest

from hypoforge.modules.m1_problem_understanding import M1ProblemUnderstanding
from hypoforge.state import (
    EvidenceGraph,
    EvidenceNode,
    FollowupRequest,
    HypothesisCard,
    PipelineState,
    ProblemCard,
)


class ScriptedClient:
    def __init__(self, *payloads: Any):
        self.payloads = list(payloads)
        self.calls: list[dict[str, Any]] = []

    async def structured_chat(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if not self.payloads:
            raise AssertionError("unexpected extra M1 LLM call")
        payload = self.payloads.pop(0)
        if isinstance(payload, BaseException):
            raise payload
        return payload


def make_module(client: Any, *, threshold: float = 0.75) -> M1ProblemUnderstanding:
    module = M1ProblemUnderstanding(
        mode="llm",
        followup_routing=True,
        followup_triage_confidence_threshold=threshold,
    )
    module.client = client
    return module


def triage_payload(
    category: str = "presentation_adjustment",
    *,
    skip_search: bool = True,
    rebuild_problem_card: bool = False,
    confidence: float = 0.99,
) -> dict[str, Any]:
    return {
        "category": category,
        "skip_search": skip_search,
        "rebuild_problem_card": rebuild_problem_card,
        "rationale": "仅调整语言和呈现方式。",
        "confidence": confidence,
    }


def parent_state(text: str, *, graph: bool = True) -> PipelineState:
    evidence_graph = None
    if graph:
        evidence_graph = EvidenceGraph(nodes=[
            EvidenceNode(id="n1", type="claim", label="NRF2 evidence")
        ])
    return PipelineState(
        input_question=text,
        problem_card=ProblemCard(
            original_question="Nrf2 激动剂在阿尔茨海默病中的神经保护机制",
            domain=["neuroscience"],
            sub_questions=["Nrf2 通路如何调控氧化应激？"],
            key_entities=["Nrf2 激动剂", "阿尔茨海默病"],
        ),
        followup=FollowupRequest(text=text, parent_run_id="parent-run"),
        evidence_graph=evidence_graph,
        best_hypotheses=[HypothesisCard(hypothesis_id="h1", statement="s1")],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    ["给我中文方案", "翻译成英文", "改成表格", "写得更简洁"],
)
async def test_presentation_followup_always_calls_llm_and_reuses_parent_card(
    text: str,
) -> None:
    """Removing the LLM triage call would silently bypass user-intent review."""

    client = ScriptedClient(triage_payload())
    module = make_module(client)
    state = parent_state(text)

    patch = await module(state)

    assert len(client.calls) == 1
    assert patch["followup"].skip_search is True
    assert patch["problem_card"] == state.problem_card
    assert patch["problem_card"] is not state.problem_card


@pytest.mark.asyncio
async def test_presentation_followup_without_reusable_graph_fails_open_to_search() -> None:
    """A formatting decision cannot route into an evidence-free M4 run."""

    client = ScriptedClient(triage_payload())
    module = make_module(client)

    patch = await module(parent_state("给我中文方案", graph=False))

    assert len(client.calls) == 1
    assert patch["followup"].skip_search is False


@pytest.mark.asyncio
async def test_low_confidence_followup_fails_open_to_search() -> None:
    """Uncertain triage must spend search cost rather than reuse wrong evidence."""

    client = ScriptedClient(triage_payload(confidence=0.40))
    module = make_module(client)

    patch = await module(parent_state("帮我调整一下"))

    assert patch["followup"].skip_search is False


@pytest.mark.asyncio
async def test_inconsistent_presentation_flags_fail_open_to_search() -> None:
    """A malformed decision cannot claim presentation-only while rebuilding."""

    client = ScriptedClient(triage_payload(rebuild_problem_card=True))
    module = make_module(client)

    patch = await module(parent_state("给我中文方案"))

    assert patch["followup"].skip_search is False


@pytest.mark.asyncio
async def test_research_change_requests_full_m1_path() -> None:
    client = ScriptedClient(triage_payload(
        "research_change",
        skip_search=False,
        rebuild_problem_card=True,
    ))
    module = make_module(client)
    rebuilt = ProblemCard(
        original_question="combined research task",
        domain=["neuroscience"],
        sub_questions=["new atomic question"],
        key_entities=["new object"],
    )
    received: list[str] = []

    async def rebuild(question: str) -> ProblemCard:
        received.append(question)
        return rebuilt

    module._understand_question = rebuild

    patch = await module(parent_state("加入帕金森病模型重新研究"))

    assert patch["followup"].skip_search is False
    assert patch["problem_card"] is rebuilt
    assert len(received) == 1
    assert "Original research question:" in received[0]
    assert "research-changing follow-up:" in received[0]


@pytest.mark.asyncio
async def test_triage_event_records_llm_rationale(monkeypatch) -> None:
    events: list[dict[str, Any]] = []

    def record(event_type: str, **kwargs: Any) -> None:
        events.append({"event_type": event_type, **kwargs})

    monkeypatch.setattr(
        "hypoforge.modules.m1_problem_understanding.emit_event",
        record,
    )
    module = make_module(ScriptedClient(triage_payload()))

    await module(parent_state("给我中文方案"))

    completed = [
        event for event in events
        if event["event_type"] == "tool_completed"
        and event.get("tool") == "qwen_followup_triage"
    ]
    assert len(completed) == 1
    assert completed[0]["details"]["decision_source"] == "llm_triage"
    assert completed[0]["details"]["rationale"] == "仅调整语言和呈现方式。"


@pytest.mark.asyncio
async def test_triage_receives_parent_artifact_summary() -> None:
    client = ScriptedClient(triage_payload())
    module = make_module(client)

    await module(parent_state("给我中文方案"))

    prompt = client.calls[0]["user_prompt"]
    assert "Parent run artefacts:" in prompt
    assert "best_hypotheses: already produced" in prompt
    assert "给我中文方案" in prompt
