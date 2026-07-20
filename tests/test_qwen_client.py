import pytest

from hypoforge.tools import qwen_client
from hypoforge.tools.qwen_client import QwenClient


class _FakeResponse:
    content = '{"answer": "ok"}'
    response_metadata = {}


class _FakeChatOpenAI:
    calls = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.__class__.calls.append(kwargs)

    async def ainvoke(self, messages):
        return _FakeResponse()


@pytest.mark.asyncio
async def test_structured_chat_never_sends_thinking_to_compatible_endpoint(monkeypatch):
    monkeypatch.setattr(qwen_client, "ChatOpenAI", _FakeChatOpenAI)
    _FakeChatOpenAI.calls.clear()

    client = QwenClient(
        model="test-model", api_key="test-key", api_base="https://example.invalid/v1"
    )
    result = await client.structured_chat(
        system_prompt="system",
        user_prompt="user",
        output_schema={"type": "object"},
        disable_thinking=True,
    )

    assert result == {"answer": "ok"}
    assert _FakeChatOpenAI.calls
    for kwargs in _FakeChatOpenAI.calls:
        assert "thinking" not in kwargs.get("model_kwargs", {})
