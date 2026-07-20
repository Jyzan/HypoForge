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


def test_build_llm_sends_qwen_thinking_control_through_extra_body() -> None:
    client = QwenClient(
        model="qwen3.6-plus",
        api_key="test-key",
        api_base="https://example.test/v1",
    )

    llm = client._build_llm(disable_thinking=True)

    assert llm.extra_body == {"enable_thinking": False}
    assert "thinking" not in llm.model_kwargs


@pytest.mark.asyncio
async def test_structured_chat_keeps_provider_extras_out_of_model_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    for call in _CapturingChatOpenAI.calls:
        assert call["extra_body"] == {"enable_thinking": False}
        assert "thinking" not in call.get("model_kwargs", {})
    assert _CapturingChatOpenAI.calls[1]["model_kwargs"]["response_format"] == {
        "type": "json_object"
    }
