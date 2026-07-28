from __future__ import annotations

import json
from typing import Any

import pytest

from hypoforge.literature.models import (
    EvidenceBucket,
    PaperRecord,
    QueryIntent,
    SearchBudget,
    SearchQuery,
    StopReason,
)
from hypoforge.literature.search import (
    CoverageEvaluator,
    IterativeSearchAgent,
    PaperDeduplicator,
    PaperRanker,
    ScoutReader,
)

from .fakes import FakePlanner, FakeSource


def query(query_id: str, text: str) -> SearchQuery:
    return SearchQuery(
        query_id=query_id,
        text=text,
        intent=QueryIntent.CORE,
        target_source="pubmed",
        purpose="Fill an evidence gap",
        relation_to_question="Directly addresses the sub-question",
    )


def paper(index: int, **kwargs: object) -> PaperRecord:
    return PaperRecord(
        paper_id=f"paper-{index}",
        title=kwargs.pop("title", f"Hsp70 mechanism paper {index}"),
        abstract=kwargs.pop("abstract", f"Neutral abstract for paper {index}."),
        year=kwargs.pop("year", 2024),
        citation_count=kwargs.pop("citation_count", 10),
        sources=["pubmed"],
        **kwargs,
    )


class ScoutClient:
    async def structured_chat(self, **kwargs: Any) -> dict[str, Any]:
        payload = json.loads(kwargs["user_prompt"].split("Papers to screen:\n", 1)[1])
        notes = []
        for item in payload:
            paper_id = item["paper_id"]
            relation = (
                "contradicts" if paper_id == "paper-4"
                else "insufficient" if paper_id == "paper-2"
                else "supports"
            )
            notes.append(
                {
                    "paper_id": paper_id,
                    "relevance": 0.9,
                    "directness": 0.9,
                    "relation": relation,
                    "supporting_sentence_ids": (
                        [f"{paper_id}:S1"] if relation == "supports" else []
                    ),
                    "contradicting_sentence_ids": (
                        [f"{paper_id}:S1"] if relation == "contradicts" else []
                    ),
                    "study_type": (
                        "review" if paper_id == "paper-2"
                        else "method" if paper_id == "paper-1"
                        else "experimental"
                    ),
                    "mechanisms": ["Hsp70 regulation"],
                    "entities": ["HSP70"],
                    "limitations": [],
                }
            )
        return {
            "notes": notes
        }


class CoverageClient:
    async def structured_chat(self, **kwargs: Any) -> dict[str, Any]:
        payload = json.loads(
            kwargs["user_prompt"].split("Grounded Scout judgments:\n", 1)[1]
        )
        complete = any(item["paper_id"] == "paper-4" for item in payload)
        return {
            "facets": [
                {
                    "facet": "negative evidence",
                    "status": "covered" if complete else "missing",
                    "paper_ids": ["paper-4"] if complete else [],
                    "sentence_ids": ["paper-4:S1"] if complete else [],
                }
            ],
            "missing_topics": [] if complete else ["negative evidence"],
            "sufficient": complete,
            "rationale": (
                "Positive and negative evidence are represented."
                if complete
                else "Negative evidence is still missing."
            ),
        }


@pytest.mark.asyncio
async def test_real_paper_selection_tools_fill_gap_across_two_agent_rounds() -> None:
    planner = FakePlanner(
        [
            [query("q-1", "initial evidence")],
            [query("q-2", "negative evidence")],
        ]
    )
    source = FakeSource(
        "pubmed",
        {
            "initial evidence": [
                paper(
                    index,
                    abstract=(
                        "This study demonstrates Hsp70 regulation of protein folding."
                    ),
                )
                for index in range(4)
            ],
            "negative evidence": [
                paper(
                    4,
                    abstract=(
                        "This experiment found no association between Hsp70 "
                        "activity and protein folding."
                    ),
                ),
                paper(
                    5,
                    year=2010,
                    citation_count=150,
                    abstract=(
                        "This study demonstrates Hsp70 regulation of protein folding."
                    ),
                ),
            ],
        },
    )
    agent = IterativeSearchAgent(
        query_planner=planner,
        sources=[source],
        deduplicator=PaperDeduplicator(),
        ranker=PaperRanker(current_year=2026),
        scout_reader=ScoutReader(ScoutClient(), current_year=2026),
        coverage_evaluator=CoverageEvaluator(CoverageClient(), current_year=2026),
        final_k=10,
    )

    result = await agent.run(
        "Does Hsp70 regulate protein folding?",
        budget=SearchBudget(max_rounds=3, max_papers=20),
    )

    assert result.stop_reason is StopReason.COVERAGE_SATISFIED
    assert result.iterations == 2
    assert result.coverage.sufficient is True
    assert len(result.final_papers) == 6
    assert any(
        "negative evidence" in topic for topic in planner.states[1].missing_topics
    )
    assert all("total" in item.rank_scores for item in result.final_papers)
