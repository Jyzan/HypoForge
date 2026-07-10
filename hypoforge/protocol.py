"""
Core protocols (interfaces) for HypoForge.

Every module, tool, skill, and metric must implement its corresponding protocol.
This ensures that implementations are swappable — a stub can be replaced by a
full LLM-powered implementation without changing any orchestration code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from .state import HypothesisCard, KnowledgeEntry, PipelineState


# ---------------------------------------------------------------------------
# Module Protocol — the primary interface for M1–M6
# ---------------------------------------------------------------------------

class ModuleProtocol(ABC):
    """
    Protocol that every pipeline module (M1–M6) must implement.

    A module is a single LangGraph node.  It receives the full PipelineState,
    does its work, and returns a dictionary of fields to *merge* back into
    the state (LangGraph's partial-state-update semantics).

    Class attributes
    ----------------
    module_name : str
        Stable identifier, e.g. ``"m1"``, ``"m2"``, …
    module_version : str
        Semver string so the registry can track which implementation is active.
    description : str
        One-line human-readable summary (shown in CLI output).
    """

    # ---- subclasses MUST override these ----
    module_name: str
    module_version: str
    description: str

    # ---- subclasses MUST implement these ----

    @abstractmethod
    async def __call__(
        self,
        state: PipelineState,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute the module.

        Parameters
        ----------
        state : PipelineState
            The full, current pipeline state.  Read input fields from here.
        config : dict or None
            Optional runtime overrides (merged on top of the YAML config).

        Returns
        -------
        dict
            A dictionary of ``{field_name: new_value}`` that LangGraph will
            merge back into ``PipelineState``.
        """
        ...

    @classmethod
    @abstractmethod
    def get_input_fields(cls) -> List[str]:
        """Return the names of ``PipelineState`` fields that this module reads."""
        ...

    @classmethod
    @abstractmethod
    def get_output_fields(cls) -> List[str]:
        """Return the names of ``PipelineState`` fields that this module writes."""
        ...


# ---------------------------------------------------------------------------
# Tool Protocol — external search / API wrappers
# ---------------------------------------------------------------------------

class ToolProtocol(ABC):
    """
    Interface for external search and data-access tools.

    Examples: Semantic Scholar, PubMed E-utilities, Qwen web-search.
    Each tool is registered in ``ToolRegistry`` and instantiated by modules
    that need it (especially M2).
    """

    tool_name: str
    tool_description: str

    @abstractmethod
    async def search(self, query: str, limit: int = 20, **kwargs) -> List[dict]:
        """Run a search query and return a list of standardised hit dicts."""
        ...

    @abstractmethod
    async def fetch(self, identifier: str, **kwargs) -> dict:
        """Fetch full details for a single item (paper, entity, …)."""
        ...


# ---------------------------------------------------------------------------
# Skill Protocol — lightweight middleware / post-processing
# ---------------------------------------------------------------------------

class SkillProtocol(ABC):
    """
    Optional middleware that wraps around module execution.

    Skills are applied *before* and *after* every module (or a selected subset).
    Use-cases: citation formatting, language translation of output, logging,
    metric collection, etc.
    """

    skill_name: str

    @abstractmethod
    async def before(self, state: PipelineState, **kwargs) -> PipelineState:
        """Transform the state *before* a module runs."""
        ...

    @abstractmethod
    async def after(self, state: PipelineState, **kwargs) -> PipelineState:
        """Transform the state *after* a module runs."""
        ...


# ---------------------------------------------------------------------------
# Metric Protocol — automated evaluation
# ---------------------------------------------------------------------------

class MetricProtocol(ABC):
    """
    Interface for an automated evaluation metric.

    Each metric receives a hypothesis (or research plan) and the body of
    literature / evidence it was derived from, and returns a float score.
    """

    metric_name: str
    metric_description: str

    @abstractmethod
    async def compute(
        self,
        hypothesis: HypothesisCard,
        knowledge_entries: List[KnowledgeEntry],
        **kwargs,
    ) -> float:
        """Compute the metric score for a single hypothesis."""
        ...

    @abstractmethod
    async def batch_compute(
        self,
        hypotheses: List[HypothesisCard],
        knowledge_entries: List[KnowledgeEntry],
        **kwargs,
    ) -> List[float]:
        """Compute the metric for multiple hypotheses (may be optimised)."""
        ...
