from __future__ import annotations

from collections.abc import Sequence

import pytest

from hypoforge.literature.adapter import AgenticM2Adapter, AgenticM2Module
from hypoforge.literature.models import (
    EvidenceChunk,
    EvidenceLinkedKnowledge,
    PaperReadingResult,
    PaperRecord,
    SearchRunResult,
    StopReason,
)
from hypoforge.modules.m2_literature_search import M2LiteratureSearch
from hypoforge.registry import ModuleRegistry
from hypoforge.state import (
    ConfidenceLevel,
    KnowledgeEntryType,
    PipelineState,
    ProblemCard,
)

from .fakes import FakeReadingWorkflow


class FakeSearchAgent:
    def __init__(self, results: Sequence[SearchRunResult]) -> None:
        self.results = list(results)
        self.calls: list[str] = []

    async def run(self, sub_question: str, **kwargs) -> SearchRunResult:
        self.calls.append(sub_question)
        return self.results[len(self.calls) - 1].model_copy(deep=True)


# ---------------------------------------------------------------------------
# AgenticM2Adapter (DI) tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_adapter_maps_evidence_linked_entries_to_legacy_literature_results() -> None:
    paper = PaperRecord(
        paper_id="paper-1",
        title="Canonical paper title",
        sources=["pubmed"],
    )
    search_agent = FakeSearchAgent(
        [
            SearchRunResult(
                sub_question="How does the mechanism work?",
                final_papers=[paper],
                stop_reason=StopReason.COVERAGE_SATISFIED,
            )
        ]
    )
    reading = FakeReadingWorkflow(
        [
            PaperReadingResult(
                paper_id="paper-1",
                evidence=[
                    EvidenceChunk(
                        evidence_id="evidence-1",
                        paper_id="paper-1",
                        chunk_id="chunk-1",
                        section="results",
                        page=2,
                        quote="The mechanism depends on ATP.",
                        normalized_claim="The mechanism depends on ATP.",
                        relevance_score=0.9,
                    )
                ],
                knowledge_entries=[
                    EvidenceLinkedKnowledge(
                        entry_id="entry-1",
                        entry_type=KnowledgeEntryType.MECHANISTIC_CONCLUSION,
                        content="The mechanism depends on ATP.",
                        confidence=ConfidenceLevel.HIGH,
                        entities=["ATP"],
                        evidence_ids=["evidence-1"],
                    )
                ],
            )
        ]
    )
    adapter = AgenticM2Adapter(search_agent=search_agent, reading_workflow=reading)
    state = PipelineState(
        input_question="Original question",
        problem_card=ProblemCard(
            original_question="Original question",
            sub_questions=["How does the mechanism work?"],
            key_entities=["ATP"],
            domain=["biology"],
        ),
    )

    output = await adapter(state)

    result = output["literature_results"][0]
    assert result.sub_question == "How does the mechanism work?"
    assert result.papers_retrieved == 1
    assert len(result.knowledge_entries) == 1
    entry = result.knowledge_entries[0]
    assert entry.id == "entry-1"
    assert entry.source_paper_id == "paper-1"
    assert entry.source_paper_title == "Canonical paper title"
    assert entry.entities == ["ATP"]
    assert entry.evidence_ids == ["evidence-1"]
    package = output["m2_knowledge_export"]
    assert package.schema_version == "m2-knowledge-export/v1"
    assert len(package.runs) == 1
    assert package.runs[0].papers[0].paper_id == "paper-1"
    assert package.runs[0].evidence[0].evidence_id == "evidence-1"
    assert package.runs[0].knowledge_entries[0] == entry
    assert set(adapter.get_output_fields()) == {
        "literature_results",
        "m2_knowledge_export",
    }
    assert reading.calls == [["paper-1"]]
    assert reading.search_contexts == [search_agent.results[0]]


@pytest.mark.asyncio
async def test_adapter_preserves_sub_question_order_in_knowledge_export() -> None:
    paper = PaperRecord(
        paper_id="paper-1",
        title="Canonical paper title",
        sources=["pubmed"],
    )
    adapter = AgenticM2Adapter(
        search_agent=FakeSearchAgent(
            [
                SearchRunResult(
                    sub_question="First sub-question",
                    final_papers=[paper],
                    stop_reason=StopReason.COVERAGE_SATISFIED,
                ),
                SearchRunResult(
                    sub_question="Second sub-question",
                    final_papers=[paper],
                    stop_reason=StopReason.COVERAGE_SATISFIED,
                ),
            ]
        ),
        reading_workflow=FakeReadingWorkflow(
            [PaperReadingResult(paper_id="paper-1")]
        ),
    )

    output = await adapter(
        PipelineState(
            input_question="Original question",
            problem_card=ProblemCard(
                original_question="Original question",
                sub_questions=["First sub-question", "Second sub-question"],
            ),
        )
    )

    assert [run.sub_question for run in output["m2_knowledge_export"].runs] == [
        "First sub-question",
        "Second sub-question",
    ]


@pytest.mark.asyncio
async def test_adapter_rejects_unrecoverable_search_error() -> None:
    adapter = AgenticM2Adapter(
        search_agent=FakeSearchAgent(
            [
                SearchRunResult(
                    sub_question="question",
                    stop_reason=StopReason.ERROR,
                    errors=["all sources failed"],
                )
            ]
        ),
        reading_workflow=FakeReadingWorkflow([]),
    )

    with pytest.raises(RuntimeError, match="all sources failed"):
        await adapter(PipelineState(input_question="question"))


# ---------------------------------------------------------------------------
# AgenticM2Module (config wrapper) tests
# ---------------------------------------------------------------------------


def test_importing_module_does_not_replace_registered_legacy_m2() -> None:
    assert ModuleRegistry.get("m2") is M2LiteratureSearch


def test_agentic_module_requires_llm_config_for_integrated_variant() -> None:
    with pytest.raises(ValueError, match="variant='integrated' requires llm_config"):
        AgenticM2Module()


def test_agentic_module_rejects_unknown_variant() -> None:
    with pytest.raises(ValueError, match="Unknown variant"):
        AgenticM2Module(llm_config=object(), variant="nonexistent")


def test_agentic_module_builds_minimal_variant_without_llm_config() -> None:
    module = AgenticM2Module(variant="minimal")
    assert isinstance(module.adapter, AgenticM2Adapter)
    assert module.adapter.search_agent is not None
    assert module.adapter.reading_workflow is not None
