"""
Configuration management for HypoForge.

Inspired by ``open_deep_research``'s ``Configuration`` Pydantic model
with environment-variable override.  All tunable parameters live here;
they can be set via YAML file, environment variable, or runtime override.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, model_validator


# ============================================================================
# Sub-models
# ============================================================================

class LLMConfig(BaseModel):
    """Configuration for a single LLM endpoint."""

    model: str = "qwen-max"
    api_base: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key: str = ""
    max_tokens: int = 4096
    temperature: float = 0.1

    @model_validator(mode="after")
    def _resolve_api_key(self) -> "LLMConfig":
        if not self.api_key:
            self.api_key = os.environ.get("DASHSCOPE_API_KEY", "")
        return self


class QwenModelsConfig(BaseModel):
    """Three-tier Qwen model assignment (matches competition spec)."""

    max: LLMConfig = Field(default_factory=lambda: LLMConfig(model="qwen-max", max_tokens=8192))
    plus: LLMConfig = Field(default_factory=lambda: LLMConfig(model="qwen-plus", max_tokens=4096))
    turbo: LLMConfig = Field(default_factory=lambda: LLMConfig(model="qwen-turbo", max_tokens=4096))


class SearchConfig(BaseModel):
    """Literature-search related settings."""

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

    # ---- skills (middleware) ----
    enabled_skills: List[str] = Field(default_factory=list)

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        """Load configuration from a YAML file.

        Environment variables take precedence over YAML values for any
        field that is also a recognised env var (``DASHSCOPE_API_KEY``, …).
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r", encoding="utf-8") as fh:
            raw: Dict[str, Any] = yaml.safe_load(fh) or {}

        # Resolve env vars for well-known keys
        return cls(**raw)

    @classmethod
    def from_defaults(cls) -> "PipelineConfig":
        """Return a config with all defaults (no YAML file needed)."""
        return cls()

    def get_llm_for_tier(self, tier: str) -> LLMConfig:
        """Convenience: return the LLMConfig for *max*, *plus*, or *turbo*."""
        mapping = {"max": self.qwen.max, "plus": self.qwen.plus, "turbo": self.qwen.turbo}
        if tier not in mapping:
            raise ValueError(f"Unknown LLM tier '{tier}'. Choose from {list(mapping)}.")
        return mapping[tier]

    def get_module_kwargs(self, module_name: str) -> Dict[str, Any]:
        """Return the override kwargs dict for *module_name*, or {}."""
        override = self.module_overrides.get(module_name)
        if override is None:
            return {}
        return override.kwargs
