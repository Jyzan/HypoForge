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
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .evaluation.rubric import DEFAULT_HYPOTHESIS_WEIGHTS


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
    "base":  "qwen3.7-max-2026-06-08",
    "max":   "qwen3.7-max-2026-06-08",
    "plus":  "qwen3.7-plus",
    "turbo": "qwen3.6-flash",
}


class LLMConfig(BaseModel):
    """Configuration for a single LLM endpoint."""

    model: str = "qwen3.7-max-2026-06-08"
    api_base: str = ""
    api_key: str = ""
    max_tokens: int = 4096
    temperature: float = 0.1

    @model_validator(mode="after")
    def _resolve_env(self) -> "LLMConfig":
        # ---- api_key ----
        if not self.api_key:
            self.api_key = (
                os.environ.get("QWEN_API_KEY")
                or os.environ.get("OPENAI_API_KEY")
                or ""
            )
        # ---- api_base ----
        if not self.api_base:
            self.api_base = (
                os.environ.get("QWEN_BASE_URL")
                or os.environ.get("OPENAI_BASE_URL")
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

    implementation: Literal["agentic"] = "agentic"
    tools: List[str] = Field(default_factory=lambda: ["semantic_scholar", "pubmed"])
    papers_per_sub_question: int = Field(default=15, ge=0)
    max_papers_total: int = Field(default=80, ge=0)
    max_rounds: int = Field(default=3, ge=1)
    max_queries: int = Field(default=12, ge=1)
    max_tokens: int = Field(default=100_000, ge=1)
    max_seconds: int = Field(default=900, ge=1)
    # Zero-result relaxation ladder inside the PubMed sources (L1 strip
    # field tags → L2 drop AND clauses → L3 unquote core concepts).
    zero_result_relaxation: bool = True

    @model_validator(mode="after")
    def _validate_agentic_budget(self) -> "SearchConfig":
        # M2 applies the paper-budget gate only when it is enabled.
        return self


class ModuleOverride(BaseModel):
    """Per-module configuration overrides (passed as ``**kwargs`` to __init__)."""

    class_name: Optional[str] = None  # fully-qualified class name for custom impl
    kwargs: Dict[str, Any] = Field(default_factory=dict)


class ScoringConfig(BaseModel):
    """Scoring / evaluation settings — the single source of truth for the
    composite weights and iteration threshold.

    ``hypothesis_weights`` feeds *both* M4's composite formula and the
    post-hoc scorer, so the two can never drift apart.  ``review_threshold``
    is the M6 ``overall`` score (1–5) at or above which iteration stops early.
    """

    hypothesis_weights: Dict[str, float] = Field(
        default_factory=lambda: dict(DEFAULT_HYPOTHESIS_WEIGHTS)
    )
    review_threshold: float = 4.0
    auto_score: bool = True  # write {run_id}_scores.json after each run


class EmbeddingConfig(BaseModel):
    """Embedding endpoint used by evidence-consistency evaluation and entity normalization.

    ``base_url`` should point to the embedding endpoint.  When unset, the
    code falls back to ``ENTITY_EMBEDDING_BASE_URL`` → ``OPENAI_BASE_URL``
    environment variables.  ``api_key_env_var`` names the env-var that holds
    the credential (defaults to ``ENTITY_EMBEDDING_API_KEY``, then
    ``OPENAI_API_KEY``).
    """

    model_name: str = "text-embedding-v3"
    api_key_env_var: str = "ENTITY_EMBEDDING_API_KEY"
    base_url: str = ""  # empty → read from ENTITY_EMBEDDING_BASE_URL / OPENAI_BASE_URL


class ConsistencyConfig(BaseModel):
    similarity_threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class EvaluationConfig(BaseModel):
    """Configuration for novelty and evidence-consistency metrics."""

    model_config = ConfigDict(extra="forbid")

    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    consistency: ConsistencyConfig = Field(default_factory=ConsistencyConfig)


class GroundingConfig(BaseModel):
    """Configuration for M3 evidence grounding (full-text → claims → relations).

    .. note::
        ``enable_gams`` defaults to ``False``.  GAMS is experimental and
        should only be enabled after the comparison framework in Track B.8
        validates it against simpler baselines on a human-annotated set.
    """

    enabled: bool = False
    max_papers: int = Field(default=10, ge=1)
    max_evidence_items: int = Field(default=200, ge=1)
    max_claims: int = Field(default=100, ge=1)
    enable_gams: bool = False  # experimental — default OFF
    gams_iterations: int = Field(default=100, ge=1)
    gams_min_confidence: float = Field(default=0.35, ge=0.0, le=1.0)


# ============================================================================
# Top-level Pipeline Config
# ============================================================================

class PipelineConfig(BaseModel):
    """
    Master configuration for a HypoForge pipeline run.

    Load from a YAML file::

        config = PipelineConfig.from_yaml("configs/full_pipeline.yaml")
    """

    model_config = ConfigDict(extra="forbid")

    # ---- general ----
    run_name: str = "hypoforge-run"
    output_dir: str = "./output"
    verbose: bool = True
    log_level: str = "INFO"  # DEBUG / INFO / WARNING / ERROR
    interactive: bool = False  # solicit human guidance between iterations (the CLI enables this)
    advanced_model_tiers: bool = False

    # ---- model tier assignment (used when advanced_model_tiers=False or as
    #      explicit overrides; "base" uses the primary model for everything) ----
    evaluation_model_tier: str = "base"   # post-hoc scorer LLM tier
    secondary_model_tier: str = "base"    # query_llm / ranker_llm / auxiliary LLMs

    # ---- entity normalization ----
    entity_embedding_model: str = ""  # empty = lexical-only; set to model name to enable embedding

    # ---- pipeline control ----
    enabled_modules: List[str] = Field(
        default_factory=lambda: ["m1", "m2", "m3", "m4", "m5", "m6"]
    )
    max_iterations: int = 3
    max_evidence_gap_rounds: int = Field(default=1, ge=0, le=5)
    enable_iteration: bool = True
    iteration_module_target: str = "m4"

    # ---- per-node wall-clock budgets (seconds) ----
    # A single stage that blows its budget stops the run with the stage named,
    # instead of a whole-question timeout.  Keys not listed fall back to
    # ``node_timeout_default``; 0 disables the budget for that stage.
    node_timeouts: Dict[str, float] = Field(
        default_factory=lambda: {
            "m1": 180.0,
            "m2": 900.0,
            "m3": 480.0,
            "m4": 600.0,
            "m5": 180.0,
            "m6": 180.0,
        }
    )
    node_timeout_default: float = 600.0

    # ---- iteration core (feature switches — default OFF keeps the 6 baseline
    # configs behaving exactly as before) ----
    # Three-counter semantics (single source of truth, mirrored in README):
    #   iteration_count = total number of M6 reviews; +1 per M6 execution and
    #     the GLOBAL hard stop is ``iteration_count >= max_iterations`` — it is
    #     NOT added to max_search_rounds; supplement search rounds deliberately
    #     consume one M6 review of this budget (total-cost cap);
    #   search_round    = number of M2 executions (fresh + supplements), capped
    #     by ``max_search_rounds``; only constrains the supplement re-hop;
    #   revision_count  = M6→M4 revision hops, audit/display only — it never
    #     participates in any stop condition.
    followup_routing: bool = False      # M1 may skip M2/M3 for search-free followups
    m6_evidence_revisit: bool = True   # M6 evidence-sufficiency verdict may re-route to M2
    max_plan_revisions: int = 2        # consecutive plan-only (revise_m5) rounds before forcing a hypothesis revision
    max_search_rounds: int = 2          # max M2 executions (fresh + supplement rounds)
    supplement_paper_budget: int = 6    # paper budget per supplement search round
    gap_no_improvement_limit: int = 3   # mark a gap unimprovable after N rounds without improvement
    gap_no_gain_limit: int = 3          # stop supplementing a gap after N rounds with zero evidence gain

    # ---- LLM assignment ----
    qwen: QwenModelsConfig = Field(default_factory=QwenModelsConfig)

    # ---- search ----
    search: SearchConfig = Field(default_factory=SearchConfig)

    # ---- per-module overrides ----
    module_overrides: Dict[str, ModuleOverride] = Field(default_factory=dict)

    # ---- scoring / evaluation ----
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    # ---- persistence ----
    memory_cache_dir: str = ""  # if non-empty, M3 persists knowledge graph here
    entity_cache_dir: str = ".cache/hypoforge"  # stable entity IDs/pair decisions

    # ---- M3 grounding (full-text → claims → relations) ----
    grounding: GroundingConfig = Field(default_factory=GroundingConfig)

    # ---- skills (middleware) ----
    enabled_skills: List[str] = Field(default_factory=list)
    skill_fail_fast: bool = False

    @model_validator(mode="after")
    def _validate_feature_dependencies(self) -> "PipelineConfig":
        if self.grounding.enabled and self.search.implementation != "agentic":
            raise ValueError(
                "grounding.enabled=true requires search.implementation='agentic' "
                "because M3 consumes M2KnowledgeExport evidence"
            )
        if not self.advanced_model_tiers:
            base = self.qwen.base
            for tier in (self.qwen.max, self.qwen.plus, self.qwen.turbo):
                tier.model = base.model
                tier.api_base = base.api_base
                tier.api_key = base.api_key
        return self

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
            raw.setdefault(
                "max_evidence_gap_rounds",
                pipeline_raw.get("max_evidence_gap_rounds", 1),
            )
            raw.setdefault("enable_iteration", pipeline_raw.get("enable_iteration", True))
            raw.setdefault("iteration_module_target", pipeline_raw.get("iteration_module_target", "m4"))
            for key in (
                "followup_routing",
                "m6_evidence_revisit",
                "max_plan_revisions",
                "max_search_rounds",
                "supplement_paper_budget",
                "gap_no_improvement_limit",
                "gap_no_gain_limit",
            ):
                if key in pipeline_raw:
                    raw.setdefault(key, pipeline_raw[key])

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
            kwargs["implementation"] = "agentic"
            if self.search.implementation == "agentic":
                # Only the integrated variant's factory accepts search-budget
                # wiring. The minimal (PubMed-only, rule-based) factory hard-
                # codes its own budget/sources and takes only final_k +
                # source_timeout_seconds, so injecting these would raise
                # TypeError on unexpected kwargs.
                variant = kwargs.get("variant", "integrated")
                if variant == "integrated":
                    kwargs.setdefault(
                        "budget",
                        {
                            "max_rounds": self.search.max_rounds,
                            "max_queries": self.search.max_queries,
                            "max_papers": self.search.max_papers_total,
                            "max_tokens": self.search.max_tokens,
                            "max_seconds": self.search.max_seconds,
                        },
                    )
                    kwargs.setdefault(
                        "per_query_limit",
                        self.search.papers_per_sub_question,
                    )
                    kwargs.setdefault("enabled_sources", list(self.search.tools))
                    kwargs.setdefault(
                        "zero_result_relaxation",
                        self.search.zero_result_relaxation,
                    )
        return kwargs


# ============================================================================
# Task C.5: Ablation Configs
# ============================================================================

class AblationConfig(BaseModel):
    """Strict configuration for running the ablation matrix script."""
    model_config = ConfigDict(extra="forbid")

    questions: List[str] = Field(default_factory=list)
    configs_dir: str = "configs/ablation"
    repeats: int = Field(default=3, ge=1)
    max_budget_usd: float = Field(default=10.0, gt=0.0)
    output_csv: str = "ablation_results.csv"

    @model_validator(mode="after")
    def _validate_ablation(self) -> "AblationConfig":
        if not self.questions:
            raise ValueError("questions list cannot be empty in AblationConfig")
        if self.max_budget_usd <= 0:
            raise ValueError("max_budget_usd must be greater than 0")
        return self

class MasterEvaluationConfig(BaseModel):
    """The root model for configs/evaluation.yaml"""
    model_config = ConfigDict(extra="forbid")

    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    ablation: AblationConfig = Field(default_factory=AblationConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "MasterEvaluationConfig":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Evaluation config file not found: {path}")
        import yaml
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        return cls.model_validate(raw)
