"""
Qwen (千问百炼) API client stub.

Real implementation in Phase 1 will use langchain-openai ChatOpenAI
pointing at the Bailian endpoint.

Usage (real impl)::

    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(
        model="qwen-max",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=os.environ["DASHSCOPE_API_KEY"],
    )
"""

from __future__ import annotations

from typing import Any, Dict


class QwenClient:
    """
    Lightweight wrapper around Qwen's OpenAI-compatible API.

    In stub mode all methods return placeholder strings.
    Replace the body of ``chat`` and ``structured_chat`` with real
    langchain-openai calls during Phase 1.
    """

    def __init__(self, model: str = "qwen-max", api_key: str = "", api_base: str = ""):
        self.model = model
        self.api_key = api_key
        self.api_base = api_base or "https://dashscope.aliyuncs.com/compatible-mode/v1"

    async def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 4096,
        temperature: float = 0.1,
    ) -> str:
        """Send a chat completion request and return the text response.

        Stub implementation — replace with real API call.
        """
        # TODO: Replace with:
        #   from langchain_openai import ChatOpenAI
        #   llm = ChatOpenAI(model=self.model, base_url=self.api_base, api_key=self.api_key)
        #   msg = await llm.ainvoke([SystemMessage(system_prompt), HumanMessage(user_prompt)])
        #   return msg.content
        _ = (system_prompt, user_prompt, max_tokens, temperature)
        return "[STUB] Qwen response — implement real API call."

    async def structured_chat(
        self,
        system_prompt: str,
        user_prompt: str,
        output_schema: Dict[str, Any],
        max_tokens: int = 4096,
    ) -> dict:
        """Send a chat request and return a structured (JSON) response.

        Stub implementation — replace with langchain's .with_structured_output().
        """
        # TODO: Replace with real structured output call
        _ = (system_prompt, user_prompt, output_schema, max_tokens)
        return {"_stub": True, "message": "Implement real structured_chat with Qwen API."}


# Singleton convenience
_default_client: QwenClient | None = None


def get_qwen_client(tier: str = "max") -> QwenClient:
    """Return a cached QwenClient for the given tier."""
    global _default_client
    if _default_client is None:
        _default_client = QwenClient(model=f"qwen-{tier}")
    return _default_client
