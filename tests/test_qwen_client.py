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


def test_build_llm_sends_thinking_control_through_model_kwargs() -> None:
    """Track B: thinking control is passed via model_kwargs, not extra_body."""
    client = QwenClient(
        model="qwen3.6-plus",
        api_key="test-key",
        api_base="https://example.test/v1",
    )

    llm = client._build_llm(disable_thinking=True)

    assert llm.model_kwargs == {"thinking": {"type": "disabled"}}


@pytest.mark.asyncio
async def test_structured_chat_passes_thinking_in_model_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Track B: structured_chat uses _build_llm which puts thinking in model_kwargs."""
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
    assert len(_CapturingChatOpenAI.calls) == 2
    # Track B: thinking is now inside model_kwargs, not extra_body
    for call in _CapturingChatOpenAI.calls:
        assert call["model_kwargs"]["thinking"] == {"type": "disabled"}
    assert _CapturingChatOpenAI.calls[1]["model_kwargs"]["response_format"] == {
        "type": "json_object"
    }
