"""Adapter boundary between agentic literature workflows and legacy M2 state."""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..protocol import ModuleProtocol
from ..state import KnowledgeEntry, LiteratureResult, PipelineState
from .models import SearchBudget, StopReason
from .protocols import ReadingExtractionWorkflowProtocol
from .search import IterativeSearchAgent


class AgenticM2Adapter(ModuleProtocol):
    """Run agentic search and reading while preserving the M2 public output."""

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
            )
            papers_by_id = {
                paper.paper_id: paper for paper in search_result.final_papers
            }
            knowledge_entries: list[KnowledgeEntry] = []
            for reading in reading_results:
                source_paper = papers_by_id.get(reading.paper_id)
                source_title = source_paper.title if source_paper else ""
                for entry in reading.knowledge_entries:
                    knowledge_entries.append(
                        KnowledgeEntry(
                            id=entry.entry_id,
                            type=entry.entry_type,
                            content=entry.content,
                            confidence=entry.confidence,
                            source_paper_id=reading.paper_id,
                            source_paper_title=source_title,
                            entities=entry.entities,
                        )
                    )

            literature_results.append(
                LiteratureResult(
                    sub_question=sub_question,
                    papers_retrieved=len(search_result.final_papers),
                    knowledge_entries=knowledge_entries,
                )
            )

        return {"literature_results": literature_results}

    @classmethod
    def get_input_fields(cls) -> list[str]:
        return ["problem_card"]

    @classmethod
    def get_output_fields(cls) -> list[str]:
        return ["literature_results"]
