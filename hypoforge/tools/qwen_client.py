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
import re
import urllib.request
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


def _recover_partial_edges(raw: str) -> Optional[dict]:
    """Recover complete edge objects from a response truncated mid-JSON.

    This is intentionally narrow: it only handles the relation schema used by
    M3 and never pretends that an incomplete arbitrary JSON document is valid.
    """
    if '"edges"' not in raw:
        return None
    pattern = re.compile(
        r'\{\s*"source"\s*:\s*"([^"]+)"\s*,\s*'
        r'"target"\s*:\s*"([^"]+)"\s*,\s*'
        r'"relation"\s*:\s*"([^"]+)"\s*\}'
    )
    edges = [
        {"source": source, "target": target, "relation": relation}
        for source, target, relation in pattern.findall(raw)
    ]
    return {"edges": edges, "_partial_json": True} if edges else None


def _salvage_string_arrays(raw: str) -> Optional[dict]:
    """Try to extract string-array values from garbled / truncated JSON.

    Looks for patterns like ``"queries": ["…", "…"]`` or ``"items": ["…"]``
    and returns a dict keyed by the property name.  This is the last-resort
    recovery for structured_chat when a smaller model (e.g. turbo tier)
    produces syntactically broken JSON.
    """
    # Match top-level string-array properties: "key": ["val1", "val2", …]
    pattern = re.compile(
        r'"(\w+)"\s*:\s*\[(.*?)\]',
        re.DOTALL,
    )
    recovered: Dict[str, list] = {}
    for key, body in pattern.findall(raw):
        # Extract individual quoted strings from the array body
        values = re.findall(r'"((?:[^"\\]|\\.)*)"', body)
        values = [v.strip() for v in values if v.strip()]
        if values:
            recovered[key] = values
    return recovered if recovered else None

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

    # ------------------------------------------------------------------
    # Class-level token counters (shared across all instances)
    # ------------------------------------------------------------------

    _total_input_tokens: int = 0
    _total_output_tokens: int = 0

    def __init__(
        self,
        model: str = "qwen3.7-max",
        api_key: str = "",
        api_base: str = "",
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("QWEN_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
        self.api_base = (
            api_base
            or os.environ.get("OPENAI_BASE_URL")
            or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

    @classmethod
    def get_token_totals(cls) -> tuple[int, int]:
        """Return cumulative (input_tokens, output_tokens) across all instances."""
        return cls._total_input_tokens, cls._total_output_tokens

    @classmethod
    def reset_token_totals(cls) -> None:
        """Reset the class-level token counters (useful between runs)."""
        cls._total_input_tokens = 0
        cls._total_output_tokens = 0

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
        disable_thinking: bool = False,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> ChatOpenAI:
        """Create a fresh ChatOpenAI instance with the current settings.

        We create a new instance per call so that different ``max_tokens`` /
        ``temperature`` values don't interfere across concurrent requests.

        Parameters
        ----------
        disable_thinking : bool
            When True, instruct reasoning models (Qwen3, DeepSeek-R1, etc.)
            to skip the thinking phase so that the full ``max_tokens`` budget
            is reserved for the visible output.  This is recommended for
            structured JSON extraction where chain-of-thought is unnecessary.
        model_kwargs : dict | None
            Extra parameters merged into the API request body
            (e.g. ``response_format``, ``thinking`` control).
        """
        # ``thinking`` is provider-specific, so the OpenAI SDK only accepts it
        # through ``extra_body``.  Copy the caller's mappings before merging so
        # a reusable ``model_kwargs`` dictionary is never mutated in-place.
        merged_kwargs: Dict[str, Any] = dict(model_kwargs or {})
        caller_extra_body = merged_kwargs.pop("extra_body", None)
        extra_body: Dict[str, Any] = {}
        if disable_thinking:
            extra_body["thinking"] = {"type": "disabled"}
        if caller_extra_body:
            extra_body.update(dict(caller_extra_body))

        chat_kwargs: Dict[str, Any] = {}
        if merged_kwargs:
            chat_kwargs["model_kwargs"] = merged_kwargs
        if extra_body:
            chat_kwargs["extra_body"] = extra_body

        return ChatOpenAI(
            model=self.model,
            base_url=self.api_base,
            api_key=self.api_key,
            max_tokens=max_tokens,
            temperature=temperature,
            **chat_kwargs,
        )

    @classmethod
    def _record_tokens(cls, response: Any) -> None:
        """Extract token usage from a LangChain AIMessage response and accumulate."""
        try:
            meta = response.response_metadata or {}
            usage = meta.get("token_usage", {})
            if usage:
                cls._total_input_tokens += int(usage.get("prompt_tokens", 0))
                cls._total_output_tokens += int(usage.get("completion_tokens", 0))
        except Exception:
            pass

    # ------------------------------------------------------------------
    # chat  — plain text response
    # ------------------------------------------------------------------

    async def chat(
        self,
        system_prompt: str = "",
        user_prompt: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.1,
        disable_thinking: bool = False,
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
        disable_thinking : bool
            When True, skip reasoning/thinking phase (see :meth:`_build_llm`).

        Returns
        -------
        str
            The model's text response (reasoning/thinking tokens are stripped).
        """
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=user_prompt))

        llm = self._build_llm(
            max_tokens=max_tokens, temperature=temperature,
            disable_thinking=disable_thinking,
        )

        try:
            response = await llm.ainvoke(messages)
            self._record_tokens(response)
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
        disable_thinking: bool = False,
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
            Top-level arrays are automatically wrapped in ``{"entries": …}``
            for compatibility with ``json_object`` response format.
        max_tokens : int
        temperature : float
        disable_thinking : bool
            When True, instruct reasoning models to skip the thinking phase.
            The full ``max_tokens`` budget is reserved for the visible JSON output.
            If the API rejects this parameter, the call is retried without it.

        Returns
        -------
        dict
            Parsed JSON response.  If the original schema was a top-level array,
            the returned dict has an ``"entries"`` key containing the array.
        """
        # ``json_object`` response_format requires the top-level to be an object.
        # If the caller asks for a bare array, wrap it automatically.
        array_wrapped = False
        if output_schema and isinstance(output_schema, dict):
            if output_schema.get("type") == "array":
                array_wrapped = True
                output_schema = {
                    "type": "object",
                    "properties": {
                        "entries": output_schema,
                    },
                    "required": ["entries"],
                }

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

        response_format_kwargs: Dict[str, Any] = {
            "response_format": {"type": "json_object"}
        }

        response = None
        # Attempt 1: request JSON mode and, when requested, disable thinking.
        try:
            llm_with_format = self._build_llm(
                max_tokens=max_tokens,
                temperature=temperature,
                disable_thinking=disable_thinking,
                model_kwargs=response_format_kwargs,
            )
            response = await llm_with_format.ainvoke(messages)
            self._record_tokens(response)
        except Exception:
            if disable_thinking:
                # Attempt 2: retry without thinking disable
                # (the model / proxy may not support the parameter)
                logger.debug(
                    "Model %s may not support thinking disable; retrying without it.",
                    self.model,
                )
                try:
                    llm_rfmt = self._build_llm(
                        max_tokens=max_tokens,
                        temperature=temperature,
                        disable_thinking=False,
                        model_kwargs=response_format_kwargs,
                    )
                    response = await llm_rfmt.ainvoke(messages)
                    self._record_tokens(response)
                except Exception:
                    logger.debug(
                        "Model %s may not support response_format; falling back to text parse.",
                        self.model,
                    )
            else:
                logger.debug(
                    "Model %s may not support response_format; falling back to text parse.",
                    self.model,
                )

        # Attempt 3: plain chat fallback.  Leave provider-specific thinking
        # control off here because a previous failure may have rejected it.
        if response is None:
            llm = self._build_llm(
                max_tokens=max_tokens,
                temperature=temperature,
                disable_thinking=False,
            )
            response = await llm.ainvoke(messages)
            self._record_tokens(response)

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
            result = json.loads(raw)
        except json.JSONDecodeError:
            recovered = _recover_partial_edges(raw)
            if recovered is not None:
                logger.warning(
                    "Recovered %d complete edges from truncated model JSON.",
                    len(recovered["edges"]),
                )
                return recovered
            # General fallback: try to salvage string-array properties from
            # garbled JSON (e.g. "queries", "items", "entries").
            salvaged = _salvage_string_arrays(raw)
            if salvaged is not None:
                logger.warning(
                    "Salvaged %d string-array key(s) from garbled JSON: %s",
                    len(salvaged), list(salvaged.keys()),
                )
                return salvaged
            logger.warning("Failed to parse JSON from model response. Raw: %s", raw[:500])
            return {"_parse_error": True, "raw_response": raw}

        # Unwrap array-wrapper if we added one
        if array_wrapped and isinstance(result, dict) and "entries" in result:
            entries = result["entries"]
            if isinstance(entries, list):
                return entries  # type: ignore[return-value]

        return result


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
