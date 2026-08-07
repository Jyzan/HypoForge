"""Tests for M1 followup triage hardening (task #22).

Covers:
* deterministic format-only shortcut — hit with plans / best_hypotheses,
  miss when the parent run has no downstream artefacts, miss for semantic
  follow-ups;
* prompt rule existence (formatting class, no-new-entity guard, rationale);
* audit trail — rationale / decision_source land in ``tool_completed``
  event details on both the shortcut and the LLM path.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from hypoforge.modules.m1_problem_understanding import M1ProblemUnderstanding
from hypoforge.prompts.m1_prompts import (
    M1_FOLLOWUP_SYSTEM_PROMPT,
    M1_FOLLOWUP_USER_TEMPLATE,
)
from hypoforge.state import (
    FollowupRequest,
    HypothesisCard,
    LiteratureResult,
    PipelineState,
    ProblemCard,
    ResearchPlan,
)


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #

class _FakeClient:
    """Records structured_chat calls and returns a canned payload."""

    def __init__(self, payload: Dict[str, Any]):
        self.payload = payload
        self.calls: List[Dict[str, Any]] = []

    async def structured_chat(self, **kwargs: Any) -> Dict[str, Any]:
        self.calls.append(kwargs)
        return self.payload


class _ExplodingClient:
    """Any LLM call fails loudly — proves the shortcut avoided the model."""

    async def structured_chat(self, **kwargs: Any) -> Dict[str, Any]:
        raise AssertionError("LLM must not be called for format-only followups")


def _module(client: Any) -> M1ProblemUnderstanding:
    mod = M1ProblemUnderstanding(mode="llm", followup_routing=True)
    mod.client = client
    return mod


def _parent_state(
    followup_text: str,
    *,
    plans: bool = True,
    best: bool = False,
    literature: bool = True,
) -> PipelineState:
    return PipelineState(
        input_question=followup_text,
        problem_card=ProblemCard(
            original_question="Nrf2 激动剂在阿尔茨海默病中的神经保护机制",
            domain=["neuroscience"],
            sub_questions=["Nrf2 通路如何调控氧化应激？"],
            key_entities=["NRF2"],
        ),
        followup=FollowupRequest(text=followup_text, parent_run_id="run-parent"),
        literature_results=(
            [LiteratureResult(sub_question="q1", papers_retrieved=10)]
            if literature
            else []
        ),
        research_plans=[ResearchPlan(hypothesis_id="h1")] if plans else [],
        best_hypotheses=(
            [HypothesisCard(hypothesis_id="h1", statement="s1")] if best else []
        ),
    )


def _collect_events(monkeypatch) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []

    def _spy(event_type: str, **kwargs: Any) -> None:
        events.append({"event_type": event_type, **kwargs})

    monkeypatch.setattr(
        "hypoforge.modules.m1_problem_understanding.emit_event", _spy
    )
    return events


def _llm_payload(skip_search: bool = True) -> Dict[str, Any]:
    return {
        "problem_card": {
            "original_question": "Nrf2 激动剂在阿尔茨海默病中的神经保护机制",
            "domain": ["neuroscience"],
            "sub_questions": ["Nrf2 通路如何调控氧化应激？"],
            "key_entities": ["NRF2"],
            "question_type": "mechanism_explanation",
        },
        "skip_search": skip_search,
        "rationale": "格式类追问，沿用父卡，无需检索。",
    }


# --------------------------------------------------------------------------- #
# Deterministic shortcut — hits
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
@pytest.mark.parametrize("followup_text", [
    "请你给我中文方案",
    "给我中文版",
    "翻译成中文",
    "把方案改成英文",
    "重新排版",
    "再给我一遍方案",
])
async def test_format_followup_shortcircuits_with_plans(
    monkeypatch, followup_text
):
    events = _collect_events(monkeypatch)
    mod = _module(_ExplodingClient())  # LLM must never be called
    patch = await mod(_parent_state(followup_text, plans=True))

    assert patch["followup"].skip_search is True
    # ProblemCard must be the parent card verbatim — no invented entities.
    assert patch["problem_card"].key_entities == ["NRF2"]
    assert patch["problem_card"].domain == ["neuroscience"]

    completed = [
        e for e in events
        if e["event_type"] == "tool_completed"
        and e["tool"] == "followup_format_shortcut"
    ]
    assert len(completed) == 1
    details = completed[0]["details"]
    assert details["skip_search"] is True
    assert details["decision_source"] == "deterministic_shortcut"
    assert details["rationale"]
    assert details["matched_pattern"]


@pytest.mark.asyncio
async def test_format_followup_shortcircuits_with_best_hypotheses_only(
    monkeypatch,
):
    _collect_events(monkeypatch)
    mod = _module(_ExplodingClient())
    state = _parent_state("请你给我中文方案", plans=False, best=True)
    patch = await mod(state)
    assert patch["followup"].skip_search is True


# --------------------------------------------------------------------------- #
# Deterministic shortcut — misses
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_format_followup_without_artifacts_falls_back_to_llm(monkeypatch):
    _collect_events(monkeypatch)
    client = _FakeClient(_llm_payload(skip_search=True))
    mod = _module(client)
    state = _parent_state("请你给我中文方案", plans=False, best=False)
    patch = await mod(state)

    assert len(client.calls) == 1  # no shortcut → LLM triage ran
    assert patch["followup"].skip_search is True


@pytest.mark.asyncio
@pytest.mark.parametrize("followup_text", [
    "聚焦阿尔茨海默方向",
    "换成帕金森模型重新分析",
    "用中文总结最新的Nrf2激动剂",
    "补充关于线粒体自噬的证据",
])
async def test_semantic_followup_never_shortcircuits(monkeypatch, followup_text):
    _collect_events(monkeypatch)
    client = _FakeClient(_llm_payload(skip_search=False))
    mod = _module(client)
    patch = await mod(_parent_state(followup_text, plans=True))

    assert len(client.calls) == 1  # LLM triage, not the shortcut
    assert patch["followup"].skip_search is False


# --------------------------------------------------------------------------- #
# Prompt rules
# --------------------------------------------------------------------------- #

def test_followup_prompt_has_formatting_rules():
    prompt = M1_FOLLOWUP_SYSTEM_PROMPT
    assert "formatting/language/presentation" in prompt
    assert "skip_search=true" in prompt
    # format-only follow-ups must never add new entities
    assert "NEVER add new entities" in prompt
    # the search-bias is scoped to genuinely new directions
    assert "search when in doubt" in prompt
    assert "rationale" in prompt
    assert "{parent_artifacts_summary}" in M1_FOLLOWUP_USER_TEMPLATE


def test_parent_artifacts_summary_mentions_completeness():
    state = _parent_state("x", plans=True, best=True)
    summary = M1ProblemUnderstanding._parent_artifacts_summary(state)
    assert "10 paper(s)" in summary
    assert "best_hypotheses: already produced" in summary
    assert "research_plans: already produced" in summary
    assert "ONLY if the follow-up" in summary


# --------------------------------------------------------------------------- #
# Audit trail — rationale in events (both paths)
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_llm_triage_rationale_lands_in_event(monkeypatch):
    events = _collect_events(monkeypatch)
    mod = _module(_FakeClient(_llm_payload(skip_search=True)))
    state = _parent_state("帮我精简一点", plans=True)  # not whitelisted
    await mod(state)

    assert len(events) > 0
    completed = [
        e for e in events
        if e["event_type"] == "tool_completed"
        and e["tool"] == "qwen_followup_triage"
    ]
    assert len(completed) == 1
    details = completed[0]["details"]
    assert details["rationale"] == "格式类追问，沿用父卡，无需检索。"
    assert details["decision_source"] == "llm_triage"


@pytest.mark.asyncio
async def test_llm_triage_receives_parent_artifact_summary(monkeypatch):
    _collect_events(monkeypatch)
    client = _FakeClient(_llm_payload(skip_search=True))
    mod = _module(client)
    await mod(_parent_state("聚焦阿尔茨海默方向", plans=True))

    user_prompt = client.calls[0]["user_prompt"]
    assert "Parent run artefacts:" in user_prompt
    assert "10 paper(s) already retrieved" in user_prompt
