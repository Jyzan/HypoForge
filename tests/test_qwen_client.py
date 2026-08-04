from __future__ import annotations

from typing import Any

import pytest

from hypoforge.tools import qwen_client as qwen_client_module
from hypoforge.tools.qwen_client import QwenClient


class _FakeResponse:
    content = '{"ok": true}'
    response_metadata: dict[str, Any] = {}


class _CapturingChatOpenAI:
    calls: list[dict[str, Any]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)

    async def ainvoke(self, messages: Any) -> _FakeResponse:
        return _FakeResponse()


def test_build_llm_sends_thinking_control_through_extra_body() -> None:
    """Provider-specific thinking control is passed through OpenAI extra_body."""
    client = QwenClient(
        model="qwen3.6-plus",
        api_key="test-key",
        api_base="https://example.test/v1",
    )

    llm = client._build_llm(disable_thinking=True)

    assert llm.extra_body == {"thinking": {"type": "disabled"}}
    assert llm.model_kwargs == {}


def test_build_llm_merges_extra_body_without_mutating_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _CapturingChatOpenAI.calls = []
    monkeypatch.setattr(qwen_client_module, "ChatOpenAI", _CapturingChatOpenAI)
    client = QwenClient(
        model="qwen3.6-plus",
        api_key="test-key",
        api_base="https://example.test/v1",
    )
    model_kwargs = {
        "response_format": {"type": "json_object"},
        "extra_body": {"custom_flag": True},
    }

    client._build_llm(disable_thinking=True, model_kwargs=model_kwargs)

    assert model_kwargs == {
        "response_format": {"type": "json_object"},
        "extra_body": {"custom_flag": True},
    }
    assert _CapturingChatOpenAI.calls[0]["model_kwargs"] == {
        "response_format": {"type": "json_object"}
    }
    assert _CapturingChatOpenAI.calls[0]["extra_body"] == {
        "thinking": {"type": "disabled"},
        "custom_flag": True,
    }


@pytest.mark.asyncio
async def test_structured_chat_passes_thinking_in_extra_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The JSON-mode request sends thinking control through extra_body."""
    _CapturingChatOpenAI.calls = []
    monkeypatch.setattr(qwen_client_module, "ChatOpenAI", _CapturingChatOpenAI)
    client = QwenClient(
        model="qwen3.6-plus",
        api_key="test-key",
        api_base="https://example.test/v1",
    )

    result = await client.structured_chat(
        user_prompt="Return an object.",
        output_schema={
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
        },
        disable_thinking=True,
    )

    assert result == {"ok": True}
    assert len(_CapturingChatOpenAI.calls) == 1
    call = _CapturingChatOpenAI.calls[0]
    assert call["extra_body"]["thinking"] == {"type": "disabled"}
    assert call["model_kwargs"]["response_format"] == {
        "type": "json_object"
    }


@pytest.mark.asyncio
async def test_structured_chat_retries_without_unsupported_thinking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _RejectingThinkingChatOpenAI(_CapturingChatOpenAI):
        async def ainvoke(self, messages: Any) -> _FakeResponse:
            call = self.calls[-1]
            if "thinking" in call.get("extra_body", {}):
                raise RuntimeError("unsupported thinking parameter")
            return _FakeResponse()

    _RejectingThinkingChatOpenAI.calls = []
    monkeypatch.setattr(
        qwen_client_module,
        "ChatOpenAI",
        _RejectingThinkingChatOpenAI,
    )
    client = QwenClient(
        model="qwen3.6-plus",
        api_key="test-key",
        api_base="https://example.test/v1",
    )

    result = await client.structured_chat(
        user_prompt="Return an object.",
        output_schema={"type": "object"},
        disable_thinking=True,
    )

    assert result == {"ok": True}
    assert len(_RejectingThinkingChatOpenAI.calls) == 2
    first, second = _RejectingThinkingChatOpenAI.calls
    assert first["extra_body"]["thinking"] == {"type": "disabled"}
    assert "extra_body" not in second
    assert second["model_kwargs"]["response_format"] == {
        "type": "json_object"
    }
