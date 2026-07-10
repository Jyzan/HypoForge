"""
Qwen (千问百炼) API client — OpenAI-compatible endpoint via langchain_openai.

This module provides a lightweight async wrapper around the ChatOpenAI client,
configured for the Alibaba Cloud MaaS (Model-as-a-Service) platform.

Supported models include all Qwen series (qwen3.7-max, qwen-plus, qwen-turbo,
qwq-plus, …), DeepSeek series (deepseek-v4-pro, deepseek-v3.2, …), GLM series,
Kimi series, and more — any model listed by the ``/models`` endpoint.

Usage::

    from hypoforge.tools.qwen_client import QwenClient

    client = QwenClient(model="qwen3.7-max")
    reply = await client.chat("You are a scientist.", "Explain protein folding.")
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

# Ensure .env is loaded even when this module is imported standalone
_load_dotenv_done = False


def _ensure_dotenv() -> None:
    global _load_dotenv_done
    if _load_dotenv_done:
        return
    _load_dotenv_done = True
    # Search for .env in common locations
    from pathlib import Path
    candidates = [
        Path(__file__).resolve().parent.parent.parent / ".env",  # hypoforge/tools/ → repo root
        Path.cwd() / ".env",
    ]
    for p in candidates:
        if p.exists():
            load_dotenv(p, override=True)
            return
    load_dotenv(override=True)


_ensure_dotenv()


class QwenClient:
    """
    Lightweight async wrapper around Qwen's OpenAI-compatible API.

    Uses ``langchain_openai.ChatOpenAI`` under the hood, which works with
    any OpenAI-compatible endpoint (Alibaba Bailian / MaaS, DashScope, etc.).

    Parameters
    ----------
    model : str
        Model identifier, e.g. ``"qwen3.7-max"``, ``"deepseek-v4-pro"``.
    api_key : str
        API key.  If empty, reads ``OPENAI_API_KEY`` from environment.
    api_base : str
        Base URL.  If empty, reads ``OPENAI_BASE_URL`` from environment,
        falling back to the public DashScope endpoint.
    """

    def __init__(
        self,
        model: str = "qwen3.7-max",
        api_key: str = "",
        api_base: str = "",
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.api_base = (
            api_base
            or os.environ.get("OPENAI_BASE_URL")
            or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

    @classmethod
    def from_config(cls, llm_config: Any) -> "QwenClient":
        """Build a client from a ``hypoforge.config.LLMConfig``-like object."""
        return cls(
            model=getattr(llm_config, "model", "qwen3.7-max"),
            api_key=getattr(llm_config, "api_key", ""),
            api_base=getattr(llm_config, "api_base", ""),
        )

    def list_models(self) -> list[str]:
        """Return model IDs exposed by the configured OpenAI-compatible API."""
        base = self.api_base.rstrip("/")
        request = urllib.request.Request(
            f"{base}/models",
            headers={"Authorization": f"Bearer {self.api_key}"},
            method="GET",
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))

        data = payload.get("data", [])
        models = []
        for item in data:
            if isinstance(item, dict) and item.get("id"):
                models.append(str(item["id"]))
        return sorted(models)

    # ------------------------------------------------------------------
    # Internal — lazy LLM builder
    # ------------------------------------------------------------------

    def _build_llm(
        self,
        max_tokens: int = 4096,
        temperature: float = 0.1,
    ) -> ChatOpenAI:
        """Create a fresh ChatOpenAI instance with the current settings.

        We create a new instance per call so that different ``max_tokens`` /
        ``temperature`` values don't interfere across concurrent requests.
        """
        return ChatOpenAI(
            model=self.model,
            base_url=self.api_base,
            api_key=self.api_key,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    # ------------------------------------------------------------------
    # chat  — plain text response
    # ------------------------------------------------------------------

    async def chat(
        self,
        system_prompt: str = "",
        user_prompt: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.1,
    ) -> str:
        """Send a chat completion request and return the text response.

        Parameters
        ----------
        system_prompt : str
            System message (can be empty).
        user_prompt : str
            User message.
        max_tokens : int
            Maximum completion tokens.
        temperature : float
            Sampling temperature (0.0 – 2.0).

        Returns
        -------
        str
            The model's text response (reasoning/thinking tokens are stripped).
        """
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=user_prompt))

        llm = self._build_llm(max_tokens=max_tokens, temperature=temperature)

        try:
            response = await llm.ainvoke(messages)
            content = response.content

            # Handle case where content is a list (e.g. multiple content blocks)
            if isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif isinstance(block, str):
                        text_parts.append(block)
                return "".join(text_parts)

            return str(content).strip() if content else ""

        except Exception as exc:
            logger.error(f"QwenClient.chat() failed: {exc}")
            raise

    # ------------------------------------------------------------------
    # structured_chat  — JSON / Pydantic response
    # ------------------------------------------------------------------

    async def structured_chat(
        self,
        system_prompt: str = "",
        user_prompt: str = "",
        output_schema: Optional[Dict[str, Any]] = None,
        max_tokens: int = 4096,
        temperature: float = 0.1,
    ) -> dict:
        """Send a chat request and return a structured (JSON) response.

        Uses LangChain's ``with_structured_output()`` when a schema is
        provided; otherwise falls back to a raw ``chat()`` + JSON parse.

        Parameters
        ----------
        system_prompt : str
        user_prompt : str
        output_schema : dict
            JSON Schema describing the expected output shape, or a Pydantic model.
        max_tokens : int
        temperature : float

        Returns
        -------
        dict
            Parsed JSON response.
        """
        # If a JSON schema dict was provided, prompt the model for JSON.
        # We use the "json_object" response_format for models that support it,
        # and fall back to prompt-based JSON for others.
        json_instruction = (
            "\n\nYou MUST respond with valid JSON only — no markdown fences, "
            "no extra text before or after the JSON object."
        )

        full_user = user_prompt
        if output_schema and isinstance(output_schema, dict):
            schema_str = json.dumps(output_schema, ensure_ascii=False, indent=2)
            full_user += (
                f"\n\nExpected JSON schema:\n{schema_str}"
                + json_instruction
            )

        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=full_user))

        llm = self._build_llm(max_tokens=max_tokens, temperature=temperature)

        try:
            # Try with response_format for models that support it
            llm_with_format = ChatOpenAI(
                model=self.model,
                base_url=self.api_base,
                api_key=self.api_key,
                max_tokens=max_tokens,
                temperature=temperature,
                model_kwargs={"response_format": {"type": "json_object"}},
            )
            response = await llm_with_format.ainvoke(messages)
        except Exception:
            # Fallback: request without response_format
            logger.debug(
                "Model %s may not support response_format; falling back to text parse.",
                self.model,
            )
            response = await llm.ainvoke(messages)

        content = response.content

        # Handle list-type content
        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    text_parts.append(block)
            content = "".join(text_parts)

        raw = str(content).strip() if content else "{}"

        # Strip markdown fences if present
        if raw.startswith("```"):
            lines = raw.split("\n")
            # Remove first line (```json or ```) and last line (```)
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            raw = "\n".join(lines).strip()

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON from model response. Raw: %s", raw[:500])
            return {"_parse_error": True, "raw_response": raw}


# ============================================================================
# Singleton convenience
# ============================================================================

_default_clients: Dict[str, QwenClient] = {}


def get_qwen_client(tier: str = "max", model: str = "") -> QwenClient:
    """Return a cached QwenClient for the given tier.

    Parameters
    ----------
    tier : str
        One of ``"max"``, ``"plus"``, ``"turbo"``.  Used to look up the
        default model for that tier unless ``model`` is given explicitly.
    model : str
        Explicit model name override (bypasses tier lookup).

    Returns
    -------
    QwenClient
        A (cached) client instance.
    """
    from hypoforge.config import DEFAULT_MODEL_MAP

    if not model:
        model = DEFAULT_MODEL_MAP.get(tier, "qwen3.7-max")

    cache_key = f"{tier}:{model}"
    if cache_key not in _default_clients:
        _default_clients[cache_key] = QwenClient(model=model)

    return _default_clients[cache_key]


def clear_client_cache() -> None:
    """Clear the singleton cache (useful for tests)."""
    _default_clients.clear()
