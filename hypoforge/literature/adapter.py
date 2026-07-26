"""Agentic M2 module — config-driven wrapper with internal DI adapter.

Track A delivers two classes:

* ``AgenticM2Adapter`` — internal dependency-injection adapter.  Accepts
  pre-built ``search_agent``, ``reading_workflow`` and ``budget``.
  Used by ``integrated.py`` / ``minimal.py`` factories and scripts.

* ``AgenticM2Module`` — config-driven public wrapper loaded by
  ``ModuleRegistry`` when ``search.implementation == "agentic"``.
  Accepts ``llm_config`` + ``variant`` and builds the internal adapter
  via ``build_adapter_from_config()``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..protocol import ModuleProtocol
from ..state import LiteratureResult, M2KnowledgeExport, PipelineState
from .export import build_m2_knowledge_export_run
from .models import SearchBudget, StopReason
from .protocols import ReadingExtractionWorkflowProtocol
from .search import IterativeSearchAgent


# ============================================================================
# Internal DI adapter (used by factories and scripts)
# ============================================================================


class AgenticM2Adapter(ModuleProtocol):
    """Internal dependency-injection adapter.

    Accepts pre-built *search_agent*, *reading_workflow* and *budget*.
    Not registered — loaded programmatically by factories or scripts.
    """

    module_name = "m2"
    module_version = "0.1.0-agentic-adapter"
    description = "Iterative literature search and evidence-linked reading adapter"

    def __init__(
        self,
        *,
        search_agent: IterativeSearchAgent,
        reading_workflow: ReadingExtractionWorkflowProtocol,
        budget: Optional[SearchBudget] = None,
    ) -> None:
        self.search_agent = search_agent
        self.reading_workflow = reading_workflow
        self.budget = budget

    async def __call__(
        self,
        state: PipelineState,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        problem_card = state.problem_card
        sub_questions = (
            problem_card.sub_questions
            if problem_card and problem_card.sub_questions
            else [state.input_question]
        )
        key_entities = problem_card.key_entities if problem_card else []
        domains = problem_card.domain if problem_card else []
        question_type = problem_card.question_type.value if problem_card else ""

        literature_results: list[LiteratureResult] = []
        export_runs = []
        for sub_question in sub_questions:
            search_result = await self.search_agent.run(
                sub_question,
                key_entities=key_entities,
                domains=domains,
                question_type=question_type,
                budget=self.budget,
            )
            if search_result.stop_reason is StopReason.ERROR:
                detail = "; ".join(search_result.errors) or "unrecoverable search error"
                raise RuntimeError(f"Agentic M2 search failed: {detail}")

            reading_results = await self.reading_workflow.run(
                sub_question,
                search_result.final_papers,
                search_context=search_result,
            )
            export_run = build_m2_knowledge_export_run(
                sub_question,
                search_result,
                reading_results,
            )
            export_runs.append(export_run)
            literature_results.append(
                LiteratureResult(
                    sub_question=sub_question,
                    papers_retrieved=len(export_run.papers),
                    knowledge_entries=list(export_run.knowledge_entries),
                )
            )

        return {
            "literature_results": literature_results,
            "m2_knowledge_export": M2KnowledgeExport(runs=export_runs),
        }

    @classmethod
    def get_input_fields(cls) -> list[str]:
        return ["problem_card"]

    @classmethod
    def get_output_fields(cls) -> list[str]:
        return ["literature_results", "m2_knowledge_export"]


# ============================================================================
# Public config-driven wrapper (loaded by ModuleRegistry)
# ============================================================================


class AgenticM2Module(ModuleProtocol):
    """Config-driven agentic M2 module.

    Loaded by ``ModuleRegistry`` when ``search.implementation == "agentic"``.
    Internally builds an ``AgenticM2Adapter`` via ``build_adapter_from_config()``
    and delegates all calls to it.
    """

    module_name = "m2"
    module_version = "0.2.0-agentic"
    description = "Agentic literature search and evidence-linked reading module"

    def __init__(
        self,
        llm_config: Optional[Any] = None,
        variant: str = "integrated",
        **kwargs,
    ) -> None:
        self.adapter = build_adapter_from_config(
            llm_config=llm_config,
            variant=variant,
            **kwargs,
        )

    async def __call__(
        self,
        state: PipelineState,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return await self.adapter(state, config)

    @classmethod
    def get_input_fields(cls) -> list[str]:
        return ["problem_card"]

    @classmethod
    def get_output_fields(cls) -> list[str]:
        return ["literature_results", "m2_knowledge_export"]


# ============================================================================
# Factory
# ============================================================================


def build_adapter_from_config(
    *,
    llm_config: Optional[Any] = None,
    variant: str = "integrated",
    **kwargs,
) -> AgenticM2Adapter:
    """Build an ``AgenticM2Adapter`` from config parameters.

    Parameters
    ----------
    llm_config :
        LLM configuration for the Qwen client (required for ``variant="integrated"``).
    variant :
        ``"integrated"`` — full multi-source search + reading pipeline.
        ``"minimal"`` — PubMed-only rule-based pipeline (no LLM required).
    **kwargs :
        Forwarded to ``build_integrated_search_adapter()`` or
        ``build_minimal_pubmed_adapter()`` (e.g. *final_k*, *budget*,
        *source_timeout_seconds*).
    """
    if variant == "integrated":
        if llm_config is None:
            raise ValueError("variant='integrated' requires llm_config")
        # Lazy import to avoid circular dependency (integrated.py imports
        # AgenticM2Adapter from this module).
        from ..tools.qwen_client import QwenClient
        from .integrated import build_integrated_search_adapter

        client = QwenClient.from_config(llm_config)
        return build_integrated_search_adapter(client=client, **kwargs)

    if variant == "minimal":
        from .minimal import build_minimal_pubmed_adapter
        return build_minimal_pubmed_adapter(**kwargs)

    raise ValueError(
        f"Unknown variant {variant!r}; expected 'integrated' or 'minimal'"
    )
