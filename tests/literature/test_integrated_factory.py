from __future__ import annotations

import json
import re

import pytest

from hypoforge.literature.integrated import build_integrated_search_adapter
from hypoforge.literature.minimal import AbstractReadingWorkflow
from hypoforge.literature.reading import (
    FullTextReadingWorkflow,
    RoutingDocumentParser,
    RoutingFulltextResolver,
)
from hypoforge.literature.search import (
    CoverageEvaluator,
    PaperDeduplicator,
    PaperRanker,
    ScoutReader,
)
from hypoforge.literature.search.search_tool import LiteratureSearchTool
from hypoforge.literature.sources.academic_source import AcademicSource
from hypoforge.literature.sources.arxiv_source import ArxivSource
from hypoforge.literature.sources.pubmed_source import PubMedSource
from hypoforge.state import PipelineState


class FakeClient:
    async def structured_chat(self, **kwargs):
        properties = kwargs.get("output_schema", {}).get("properties", {})
        if "notes" in properties:
            payload = json.loads(
                kwargs["user_prompt"].split("Papers to screen:\n", 1)[1]
            )
            return {
                "notes": [
                    {
                        "paper_id": item["paper_id"],
                        "relevance": 0.9,
                        "directness": 0.8,
                        "relation": "supports",
                        "supporting_sentence_ids": [
                            item["sentences"][0]["sentence_id"]
                        ],
                        "contradicting_sentence_ids": [],
                        "study_type": "experimental",
                        "mechanisms": ["Hippo signaling"],
                        "entities": ["YAP", "TAZ"],
                        "limitations": [],
                    }
                    for item in payload
                    if item["sentences"]
                ]
            }
        if "facets" in properties:
            return {
                "facets": [
                    {
                        "facet": "Hippo signaling mechanism",
                        "status": "covered",
                        "paper_ids": ["PMID:1"],
                        "sentence_ids": ["PMID:1:S1"],
                    }
                ],
                "missing_topics": [],
                "sufficient": True,
                "rationale": "The supplied abstracts ground the mechanism.",
            }
        if "summary" in properties:
            evidence_ids = re.findall(
                r"\[Evidence ID: ([^\]]+)\]", kwargs.get("user_prompt", "")
            )
            return {
                "summary": "The supplied passage reports relevant evidence.",
                "summary_evidence_ids": evidence_ids[:1],
                "entries": [
                    {
                        "type": "established_fact",
                        "content": "The supplied passage reports relevant evidence.",
                        "confidence": "medium",
                        "entities": ["Hippo"],
                        "evidence_ids": evidence_ids[:1],
                    }
                ],
            }
        return {
            "queries": [
                {
                    "text": "Hippo",
                    "tool": "pubmed",
                    "purpose": "core",
                    "reasoning": "biomedical",
                },
                {
                    "text": "YAP TAZ",
                    "tool": "semantic_scholar",
                    "purpose": "supporting",
                    "reasoning": "cross-source",
                },
            ]
        }


class FakeLegacyTool:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    async def search(self, query: str, limit: int = 20):
        self.calls.append((query, limit))
        return list(self.rows)


def make_search_tool() -> LiteratureSearchTool:
    return LiteratureSearchTool(
        pubmed=PubMedSource(
            tool=FakeLegacyTool(
                [
                    {
                        "pmid": "1",
                        "title": "PubMed paper",
                        "abstract": "PubMed evidence.",
                        "year": 2024,
                    }
                ]
            )
        ),
        academic=AcademicSource(
            tool=FakeLegacyTool(
                [
                    {
                        "paper_id": "S2-1",
                        "source": "semantic_scholar",
                        "title": "Academic paper",
                        "abstract": "Academic evidence.",
                        "year": 2023,
                    }
                ]
            )
        ),
        arxiv_source=ArxivSource(tool=FakeLegacyTool([])),
    )


def test_factory_wires_all_literature_sources() -> None:
    adapter = build_integrated_search_adapter(
        client=FakeClient(), search_tool=make_search_tool(), final_k=5
    )

    assert set(adapter.search_agent.sources) == {
        "pubmed",
        "semantic_scholar",
        "arxiv",
    }
    assert type(adapter.search_agent.query_planner).__name__ == "QueryPlanner"
    assert adapter.search_agent.query_planner.strict is True


def test_factory_wires_real_paper_selection_tools() -> None:
    client = FakeClient()
    adapter = build_integrated_search_adapter(
        client=client, search_tool=make_search_tool(), final_k=5
    )
    agent = adapter.search_agent

    assert isinstance(agent.deduplicator, PaperDeduplicator)
    assert isinstance(agent.ranker, PaperRanker)
    assert isinstance(agent.scout_reader, ScoutReader)
    assert agent.scout_reader.client is client
    assert isinstance(agent.coverage_evaluator, CoverageEvaluator)
    assert agent.coverage_evaluator.client is client
    assert agent.coverage_evaluator.selection_limit == 5
    assert agent.candidate_limit == 20


def test_factory_replaces_abstract_placeholder_with_fulltext_workflow(tmp_path) -> None:
    adapter = build_integrated_search_adapter(
        client=FakeClient(),
        search_tool=make_search_tool(),
        reading_cache_dir=tmp_path,
    )

    assert isinstance(adapter.reading_workflow, FullTextReadingWorkflow)
    assert not isinstance(adapter.reading_workflow, AbstractReadingWorkflow)
    assert isinstance(
        adapter.reading_workflow.resolver,
        RoutingFulltextResolver,
    )
    assert isinstance(
        adapter.reading_workflow.parser,
        RoutingDocumentParser,
    )


def test_factory_wires_download_and_per_paper_timeout_limits(tmp_path) -> None:
    adapter = build_integrated_search_adapter(
        client=FakeClient(),
        search_tool=make_search_tool(),
        reading_cache_dir=tmp_path,
        arxiv_download_timeout_seconds=17.0,
        resolver_timeout_seconds=23.0,
    )

    workflow = adapter.reading_workflow
    assert workflow.resolver_timeout_seconds == 23.0
    assert workflow.resolver.arxiv_resolver.download_timeout_seconds == 17.0


@pytest.mark.asyncio
async def test_integrated_factory_runs_m2_without_minimal_fallback(tmp_path) -> None:
    async def unavailable_pmc(url: str, timeout: float) -> bytes:
        raise OSError("PMC unavailable in offline integration test")

    adapter = build_integrated_search_adapter(
        client=FakeClient(),
        search_tool=make_search_tool(),
        final_k=5,
        reading_cache_dir=tmp_path,
        pmc_backend=unavailable_pmc,
    )
    # The factory's full-text wiring is covered separately.  Keep this
    # end-to-end semantic/search/export test entirely thread- and network-free.
    adapter.reading_workflow = AbstractReadingWorkflow()
    output = await adapter(PipelineState(input_question="Hippo YAP TAZ"))

    result = output["literature_results"][0]
    assert result.papers_retrieved == 2
    assert {entry.source_paper_id for entry in result.knowledge_entries} == {
        "PMID:1",
        "S2:S2-1",
    }
    package = output["m2_knowledge_export"]
    assert package.schema_version == "m2-knowledge-export/v1"
    for run in package.runs:
        evidence_ids = {evidence.evidence_id for evidence in run.evidence}
        for entry in run.knowledge_entries:
            assert set(entry.evidence_ids) <= evidence_ids


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "error"),
    [
        (None, OSError("Qwen unavailable")),
        ({"queries": []}, None),
        ({"queries": [{"text": "", "tool": "unknown"}]}, None),
    ],
)
async def test_integrated_planner_failure_returns_error_without_source_calls(
    response, error
) -> None:
    class FailingClient:
        async def structured_chat(self, **kwargs):
            if error is not None:
                raise error
            return response

    search_tool = make_search_tool()
    adapter = build_integrated_search_adapter(
        client=FailingClient(), search_tool=search_tool
    )

    result = await adapter.search_agent.run("Hippo YAP TAZ")

    assert result.stop_reason.value == "error"
    assert result.queries == []
    assert result.failed_sources == []
    assert result.source_result_counts == {}
    assert any("query_planner" in item for item in result.errors)
    assert search_tool._pubmed_source._tool.calls == []
    assert search_tool._academic_source._tool.calls == []
