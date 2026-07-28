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


def semantic_note(
    paper_id: str,
    *,
    relevance: float = 0.9,
    directness: float = 0.9,
    relation: str = "insufficient",
    supporting: list[str] | None = None,
    contradicting: list[str] | None = None,
    study_type: str = "experimental",
) -> dict[str, Any]:
    return {
        "paper_id": paper_id,
        "relevance": relevance,
        "directness": directness,
        "relation": relation,
        "supporting_sentence_ids": supporting or [],
        "contradicting_sentence_ids": contradicting or [],
        "study_type": study_type,
        "mechanisms": ["protein homeostasis"],
        "entities": ["HSP70"],
        "limitations": [],
    }


@pytest.mark.asyncio
async def test_scout_batches_requests_and_restores_input_order() -> None:
    def respond(call: dict[str, Any]) -> dict[str, Any]:
        payload = payload_from_call(call)
        return {
            "notes": [
                semantic_note(
                    item["paper_id"],
                    relation="supports",
                    supporting=[item["sentences"][0]["sentence_id"]],
                )
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
    assert EvidenceBucket.SUPPORTING in notes[0].evidence_buckets


@pytest.mark.asyncio
async def test_scout_invalid_duplicate_and_missing_results_fallback_per_paper() -> None:
    def respond(_: dict[str, Any]) -> dict[str, Any]:
        return {
            "notes": [
                semantic_note("unknown", relation="supports", supporting=["unknown:S1"]),
                semantic_note(
                    "paper-0",
                    relevance=2,
                    relation="supports",
                    supporting=["paper-0:S1"],
                    study_type="illegal-type",
                ),
                semantic_note("paper-0", relevance=0.0),
            ]
        }

    papers = [paper(0), paper(1, year=2026)]
    notes = await ScoutReader(
        FakeStructuredClient(responder=respond), current_year=2026
    ).read("Hsp70 activity", papers)

    assert [item.paper_id for item in notes] == ["paper-0", "paper-1"]
    assert notes[0].relevance_to_question <= 0.49
    assert EvidenceBucket.SUPPORTING not in notes[0].evidence_buckets
    assert notes[1].relevance_to_question <= 0.49
    assert notes[1].evidence_buckets == {EvidenceBucket.RECENT}


@pytest.mark.asyncio
async def test_generic_however_is_not_contradicting() -> None:
    source = paper(
        0,
        abstract=(
            "Hsp70 ATPase activity was measured during aging. "
            "However, the assay protocol required calibration."
        ),
    )

    notes = await ScoutReader(
        FakeStructuredClient(
            responder=lambda _: {"notes": [semantic_note("paper-0")]}
        )
    ).read("Does aging regulate Hsp70 ATPase activity?", [source])

    assert EvidenceBucket.CONTRADICTING not in notes[0].evidence_buckets


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "abstract",
    [
        "The study did not test whether Hsp70 regulates ATPase activity.",
        "Hsp70 was significantly elevated in aged cells.",
    ],
)
async def test_non_directional_language_remains_insufficient(abstract: str) -> None:
    notes = await ScoutReader(
        FakeStructuredClient(
            responder=lambda _: {"notes": [semantic_note("paper-0")]}
        )
    ).read("Does Hsp70 regulate ATPase activity?", [paper(0, abstract=abstract)])

    assert EvidenceBucket.SUPPORTING not in notes[0].evidence_buckets
    assert EvidenceBucket.CONTRADICTING not in notes[0].evidence_buckets


@pytest.mark.asyncio
async def test_mixed_requires_valid_evidence_in_both_directions() -> None:
    source = paper(
        0,
        abstract="Hsp70 increased ATPase activity. Hsp70 had no effect in aged cells.",
    )
    client = FakeStructuredClient(
        responder=lambda _: {
            "notes": [
                semantic_note(
                    "paper-0",
                    relation="mixed",
                    supporting=["paper-0:S1"],
                    contradicting=["paper-0:S2"],
                )
            ]
        }
    )

    note = (await ScoutReader(client).read("Does Hsp70 affect ATPase?", [source]))[0]

    assert note.supporting_evidence == ["Hsp70 increased ATPase activity."]
    assert note.contradicting_evidence == ["Hsp70 had no effect in aged cells."]
    assert {
        EvidenceBucket.SUPPORTING,
        EvidenceBucket.CONTRADICTING,
    }.issubset(note.evidence_buckets)


@pytest.mark.asyncio
async def test_invalid_or_cross_paper_sentence_ids_are_rejected() -> None:
    sources = [
        paper(0, abstract="Hsp70 increased ATPase activity."),
        paper(1, abstract="A separate experiment found an effect."),
    ]
    client = FakeStructuredClient(
        responder=lambda _: {
            "notes": [
                semantic_note(
                    "paper-0",
                    relation="supports",
                    supporting=["paper-1:S1", "paper-0:S99"],
                ),
                semantic_note("paper-1"),
            ]
        }
    )

    notes = await ScoutReader(client).read("question", sources)

    assert notes[0].supporting_evidence == []
    assert EvidenceBucket.SUPPORTING not in notes[0].evidence_buckets


@pytest.mark.asyncio
async def test_llm_semantics_can_resolve_synonyms_without_lexical_cap() -> None:
    source = paper(
        0,
        title="Acute myocardial infarction outcomes",
        abstract="Mortality after acute myocardial infarction was measured.",
    )
    client = FakeStructuredClient(
        responder=lambda _: {
            "notes": [
                semantic_note(
                    "paper-0",
                    relevance=0.95,
                    directness=0.9,
                    relation="supports",
                    supporting=["paper-0:S1"],
                    study_type="observational",
                )
            ]
        }
    )

    note = (await ScoutReader(client).read("heart attack mortality", [source]))[0]

    assert note.relevance_to_question == 0.95
    assert note.directness_to_question == 0.9


@pytest.mark.asyncio
async def test_peer_review_words_do_not_force_review_article_type() -> None:
    source = paper(
        0,
        title="Peer review comments on an ATPase assay",
        abstract="We experimentally validated the revised ATPase assay.",
    )
    client = FakeStructuredClient(
        responder=lambda _: {
            "notes": [semantic_note("paper-0", study_type="experimental")]
        }
    )

    note = (await ScoutReader(client).read("ATPase assay", [source]))[0]

    assert note.study_design == "experimental"
    assert EvidenceBucket.REVIEW not in note.evidence_buckets


@pytest.mark.asyncio
async def test_not_applicable_clamps_scores_and_has_no_direction() -> None:
    client = FakeStructuredClient(
        responder=lambda _: {
            "notes": [
                semantic_note(
                    "paper-0",
                    relevance=0.99,
                    directness=0.99,
                    relation="not_applicable",
                )
            ]
        }
    )
    note = (await ScoutReader(client).read("unrelated question", [paper(0)]))[0]

    assert note.relevance_to_question <= 0.2
    assert note.directness_to_question <= 0.2
    assert EvidenceBucket.SUPPORTING not in note.evidence_buckets


@pytest.mark.asyncio
async def test_scout_failure_fallback_is_metadata_only_and_non_directional() -> None:
    source = paper(
        0,
        title="Systematic review of Hsp70 assays",
        abstract="However, experiments did not support the mechanism.",
        year=2026,
        citation_count=120,
    )
    notes = await ScoutReader(
        FakeStructuredClient(error=RuntimeError("offline")), current_year=2026
    ).read("Hsp70 mechanism", [source])

    assert notes[0].evidence_buckets == {EvidenceBucket.RECENT}
    assert notes[0].relevance_to_question <= 0.49
    assert notes[0].supporting_evidence == []
    assert notes[0].contradicting_evidence == []


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
