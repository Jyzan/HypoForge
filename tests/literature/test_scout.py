from __future__ import annotations

import json
from typing import Any, Callable

import pytest

from hypoforge.literature.models import EvidenceBucket, PaperRecord
from hypoforge.literature.search.scout import ScoutReader


class FakeStructuredClient:
    def __init__(
        self,
        responder: Callable[[dict[str, Any]], Any] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.responder = responder
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def structured_chat(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        response = self.responder(kwargs) if self.responder else {"notes": []}
        return response if isinstance(response, dict) else {}


def paper(index: int, **kwargs: object) -> PaperRecord:
    return PaperRecord(
        paper_id=f"paper-{index}",
        title=kwargs.pop("title", f"Hsp70 paper {index}"),
        abstract=kwargs.pop(
            "abstract", "This study demonstrates regulation of Hsp70 activity."
        ),
        sources=["test"],
        **kwargs,
    )


def payload_from_call(call: dict[str, Any]) -> list[dict[str, Any]]:
    marker = "Papers to screen:\n"
    return json.loads(call["user_prompt"].split(marker, 1)[1])


@pytest.mark.asyncio
async def test_scout_batches_requests_and_restores_input_order() -> None:
    def respond(call: dict[str, Any]) -> dict[str, Any]:
        payload = payload_from_call(call)
        return {
            "notes": [
                {
                    "paper_id": item["paper_id"],
                    "main_topic": item["title"],
                    "key_terms": ["Hsp70"],
                    "entities": ["HSP70"],
                    "mechanisms": ["Hsp70 regulates folding."],
                    "important_authors": [],
                    "controversies": [],
                    "candidate_citations": [],
                    "relevance_to_question": 0.9,
                    "evidence_buckets": ["supporting"],
                    "study_design": "in vitro study",
                    "evidence_summary": "The abstract supports the mechanism.",
                }
                for item in reversed(payload)
            ]
        }

    client = FakeStructuredClient(responder=respond)
    papers = [paper(index) for index in range(10)]

    notes = await ScoutReader(
        client, batch_size=4, max_concurrency=2, current_year=2026
    ).read("How does Hsp70 regulate folding?", papers)

    assert [note.paper_id for note in notes] == [item.paper_id for item in papers]
    assert len(client.calls) == 3
    assert all(call["disable_thinking"] is True for call in client.calls)
    assert all(call["temperature"] == 0.0 for call in client.calls)
    assert EvidenceBucket.SUPPORTING in notes[0].evidence_buckets


@pytest.mark.asyncio
async def test_scout_filters_unknown_and_duplicate_ids_and_fills_missing_notes() -> None:
    def respond(_: dict[str, Any]) -> dict[str, Any]:
        return {
            "notes": [
                {
                    "paper_id": "unknown",
                    "relevance_to_question": 1,
                    "evidence_buckets": ["supporting"],
                },
                {
                    "paper_id": "paper-0",
                    "main_topic": "Model topic",
                    "relevance_to_question": 2,
                    "evidence_buckets": ["review", "not-a-bucket"],
                },
                {
                    "paper_id": "paper-0",
                    "main_topic": "Duplicate should be ignored",
                    "relevance_to_question": 0,
                    "evidence_buckets": [],
                },
            ]
        }

    client = FakeStructuredClient(responder=respond)
    papers = [
        paper(0),
        paper(
            1,
            title="A cohort study of Hsp70",
            abstract="The cohort found no association with Hsp70 activity.",
        ),
    ]

    notes = await ScoutReader(client, current_year=2026).read("Hsp70 activity", papers)

    assert [note.paper_id for note in notes] == ["paper-0", "paper-1"]
    assert notes[0].main_topic == "Model topic"
    assert notes[0].relevance_to_question == 1.0
    assert EvidenceBucket.REVIEW in notes[0].evidence_buckets
    assert all(bucket.value != "not-a-bucket" for bucket in notes[0].evidence_buckets)
    assert notes[0].evidence_summary
    assert notes[1].study_design == "cohort study"
    assert EvidenceBucket.CONTRADICTING in notes[1].evidence_buckets


@pytest.mark.asyncio
async def test_scout_uses_conservative_fallback_when_llm_fails() -> None:
    source = paper(
        0,
        title="Systematic review of Hsp70 assays",
        abstract=(
            "This systematic review found inconsistent results. "
            "Several experiments did not support the proposed mechanism."
        ),
        year=2026,
        citation_count=120,
    )
    client = FakeStructuredClient(error=RuntimeError("offline"))

    notes = await ScoutReader(client, current_year=2026).read(
        "Hsp70 mechanism", [source]
    )

    assert len(notes) == 1
    assert notes[0].evidence_summary
    assert EvidenceBucket.REVIEW in notes[0].evidence_buckets
    assert EvidenceBucket.RECENT in notes[0].evidence_buckets
    assert EvidenceBucket.CONTRADICTING in notes[0].evidence_buckets


@pytest.mark.asyncio
async def test_scout_without_client_is_offline_and_empty_input_skips_calls() -> None:
    reader = ScoutReader(current_year=2026)

    assert await reader.read("question", []) == []
    notes = await reader.read("Hsp70", [paper(0)])

    assert len(notes) == 1
    assert notes[0].paper_id == "paper-0"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"batch_size": 0},
        {"max_concurrency": 0},
        {"max_tokens": 0},
        {"abstract_char_limit": 0},
    ],
)
def test_scout_rejects_nonpositive_limits(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        ScoutReader(**kwargs)
