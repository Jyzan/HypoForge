"""
Configuration management for HypoForge.

Inspired by ``open_deep_research``'s ``Configuration`` Pydantic model
with environment-variable override.  All tunable parameters live here;
they can be set via YAML file, environment variable, or runtime override.

Environment variables (loaded from ``.env`` or system env):
  - ``OPENAI_API_KEY`` — API key for the OpenAI-compatible endpoint (required)
  - ``OPENAI_BASE_URL`` — Base URL for the OpenAI-compatible endpoint
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Auto-load .env  (searches from project root upward)
# ---------------------------------------------------------------------------

def _find_and_load_dotenv() -> None:
    """Locate the project-root ``.env`` and load it into ``os.environ``.

    Searches upward from this file's directory; the first ``.env`` found wins.
    Also checks ``Path.cwd()`` as a fallback so that ``python run_hypoforge.py``
    works regardless of the working directory.
    """
    candidates = [
        Path(__file__).resolve().parent.parent / ".env",   # repo root relative to hypoforge/
        Path.cwd() / ".env",                               # current working directory
    ]
    for dotenv_path in candidates:
        if dotenv_path.exists():
            load_dotenv(dotenv_path, override=True)
            return
    # If no .env file is found, try load_dotenv() which searches cwd upward
    load_dotenv(override=True)


_find_and_load_dotenv()


# ============================================================================
# Sub-models
# ============================================================================

# Tier → default model mapping (uses the latest available models from the API)
DEFAULT_MODEL_MAP = {
    "base":  "qwen3.7-max",
    "max":   "qwen3.7-max",
    "plus":  "qwen3.7-plus",
    "turbo": "qwen3.6-flash",
}


class LLMConfig(BaseModel):
    """Configuration for a single LLM endpoint."""

    model: str = "qwen3.7-max"
    api_base: str = ""
    api_key: str = ""
    max_tokens: int = 4096
    temperature: float = 0.1

    @model_validator(mode="after")
    def _resolve_env(self) -> "LLMConfig":
        # ---- api_key ----
        if not self.api_key:
            self.api_key = os.environ.get("OPENAI_API_KEY") or ""
        # ---- api_base ----
        if not self.api_base:
            self.api_base = (
                os.environ.get("OPENAI_BASE_URL")
                or "https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
        return self


class QwenModelsConfig(BaseModel):
    """Three-tier model assignment (matches competition spec).

    The YAML config can override each tier's model / temperature / max_tokens.
    If ``model`` is left empty, defaults from ``DEFAULT_MODEL_MAP`` are used.
    """

    base: LLMConfig = Field(default_factory=lambda: LLMConfig(
        model=DEFAULT_MODEL_MAP["base"], max_tokens=8192,
    ))
    max: LLMConfig = Field(default_factory=lambda: LLMConfig(
        model=DEFAULT_MODEL_MAP["max"], max_tokens=8192,
    ))
    plus: LLMConfig = Field(default_factory=lambda: LLMConfig(
        model=DEFAULT_MODEL_MAP["plus"], max_tokens=4096,
    ))
    turbo: LLMConfig = Field(default_factory=lambda: LLMConfig(
        model=DEFAULT_MODEL_MAP["turbo"], max_tokens=4096, temperature=0.3,
    ))


class SearchConfig(BaseModel):
    """Literature-search related settings."""

    implementation: Literal["legacy", "agentic"] = "legacy"
    tools: List[str] = Field(default_factory=lambda: ["semantic_scholar", "pubmed"])
    papers_per_sub_question: int = 15
    max_papers_total: int = 80


class ModuleOverride(BaseModel):
    """Per-module configuration overrides (passed as ``**kwargs`` to __init__)."""

    class_name: Optional[str] = None  # fully-qualified class name for custom impl
    kwargs: Dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# Top-level Pipeline Config
# ============================================================================

class PipelineConfig(BaseModel):
    """
    Master configuration for a HypoForge pipeline run.

    Load from a YAML file::

        config = PipelineConfig.from_yaml("configs/full_pipeline.yaml")
    """

    # ---- general ----
    run_name: str = "hypoforge-run"
    output_dir: str = "./output"
    verbose: bool = True
    log_level: str = "INFO"  # DEBUG / INFO / WARNING / ERROR

    # ---- pipeline control ----
    enabled_modules: List[str] = Field(
        default_factory=lambda: ["m1", "m2", "m3", "m4", "m5", "m6"]
    )
    max_iterations: int = 3
    enable_iteration: bool = True
    iteration_module_target: str = "m4"  # which module receives feedback

    # ---- LLM assignment ----
    qwen: QwenModelsConfig = Field(default_factory=QwenModelsConfig)

    # ---- search ----
    search: SearchConfig = Field(default_factory=SearchConfig)

    # ---- per-module overrides ----
    module_overrides: Dict[str, ModuleOverride] = Field(default_factory=dict)

    # ---- persistence ----
    memory_cache_dir: str = ""  # if non-empty, M3 persists knowledge graph here

    # ---- skills (middleware) ----
    enabled_skills: List[str] = Field(default_factory=list)

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        """Load configuration from a YAML file.

        Environment variables take precedence over YAML values for any
        field that is also a recognised env var (``OPENAI_API_KEY``, …).
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r", encoding="utf-8") as fh:
            raw: Dict[str, Any] = yaml.safe_load(fh) or {}

        # Flatten nested "pipeline" key if present (for backward compat)
        if "pipeline" in raw and isinstance(raw["pipeline"], dict):
            pipeline_raw = raw.pop("pipeline")
            raw.setdefault("enabled_modules", pipeline_raw.get("enabled_modules", ["m1", "m2", "m3", "m4", "m5", "m6"]))
            raw.setdefault("max_iterations", pipeline_raw.get("max_iterations", 3))
            raw.setdefault("enable_iteration", pipeline_raw.get("enable_iteration", True))
            raw.setdefault("iteration_module_target", pipeline_raw.get("iteration_module_target", "m4"))

        # Resolve env vars for well-known keys
        return cls(**raw)

    @classmethod
    def from_defaults(cls) -> "PipelineConfig":
        """Return a config with all defaults (no YAML file needed)."""
        return cls()

    def get_llm_for_tier(self, tier: str) -> LLMConfig:
        """Convenience: return the LLMConfig for a named model tier."""
        mapping = {
            "base": self.qwen.base,
            "max": self.qwen.max,
            "plus": self.qwen.plus,
            "turbo": self.qwen.turbo,
        }
        if tier not in mapping:
            raise ValueError(f"Unknown LLM tier '{tier}'. Choose from {list(mapping)}.")
        return mapping[tier]

    def get_module_kwargs(self, module_name: str) -> Dict[str, Any]:
        """Return the override kwargs dict for *module_name*, or {}."""
        override = self.module_overrides.get(module_name)
        kwargs = dict(override.kwargs) if override is not None else {}
        if module_name == "m2":
            kwargs.setdefault("implementation", self.search.implementation)
        return kwargs
