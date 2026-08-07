from __future__ import annotations

import asyncio

import pytest

from hypoforge.literature.models import EvidenceChunk, PaperRecord
from hypoforge.literature.reading.reader import QwenPaperReader
from hypoforge.state import ConfidenceLevel, KnowledgeEntryType


class FakeClient:
    def __init__(self, response=None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict] = []

    async def structured_chat(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


def paper() -> PaperRecord:
    return PaperRecord(
        paper_id="PMID:1",
        title="Hippo mechanics",
        abstract="YAP responds to mechanics.",
        sources=["pubmed"],
    )


def evidence(identifier: str = "e1", *, citable: bool = True) -> EvidenceChunk:
    return EvidenceChunk(
        evidence_id=identifier,
        paper_id="PMID:1",
        chunk_id="c1",
        section="results",
        quote="Mechanical tension promotes nuclear YAP accumulation.",
        normalized_claim="Mechanical tension promotes nuclear YAP accumulation.",
        relevance_score=0.9,
        citable=citable,
    )


def valid_response() -> dict:
    return {
        "summary": "Mechanical tension promotes nuclear YAP.",
        "summary_evidence_ids": ["e1"],
        "entries": [
            {
                "type": "mechanistic_conclusion",
                "content": "Mechanical tension promotes nuclear YAP.",
                "confidence": "high",
                "entities": ["YAP", "mechanical tension"],
                "evidence_ids": ["e1"],
            }
        ],
    }


@pytest.mark.asyncio
async def test_reader_returns_validated_evidence_linked_knowledge() -> None:
    client = FakeClient(valid_response())

    result = await QwenPaperReader(client).read("How is YAP regulated?", paper(), [evidence()])

    assert result.summary == "Mechanical tension promotes nuclear YAP."
    assert [item.evidence_id for item in result.evidence] == ["e1"]
    assert len(result.knowledge_entries) == 1
    entry = result.knowledge_entries[0]
    assert entry.entry_type is KnowledgeEntryType.MECHANISTIC_CONCLUSION
    assert entry.confidence is ConfidenceLevel.HIGH
    assert entry.evidence_ids == ["e1"]
    assert client.calls[0]["temperature"] == 0.0
    assert client.calls[0]["disable_thinking"] is True
    assert len(client.calls[0]["user_prompt"]) < 20_000


@pytest.mark.asyncio
async def test_reader_repairs_mixed_ids_without_losing_valid_entries() -> None:
    response = valid_response()
    response["entries"].append(
        {
            "type": "method",
            "content": "A method claim.",
            "confidence": "medium",
            "entities": [],
            "evidence_ids": ["e1", "invented"],
        }
    )
    response["entries"][0]["evidence_ids"] = ["invented"]

    result = await QwenPaperReader(FakeClient(response)).read(
        "question", paper(), [evidence()]
    )

    assert len(result.knowledge_entries) == 1
    assert result.knowledge_entries[0].content == "A method claim."
    assert result.knowledge_entries[0].evidence_ids == ["e1"]
    assert [item.evidence_id for item in result.evidence] == ["e1"]
    assert any("rejected 1" in item for item in result.errors)
    assert any("repaired" in item for item in result.errors)


@pytest.mark.asyncio
async def test_reader_rejects_context_only_evidence_ids() -> None:
    response = valid_response()
    response["summary_evidence_ids"] = ["context"]
    response["entries"][0]["evidence_ids"] = ["context"]

    result = await QwenPaperReader(FakeClient(response)).read(
        "question",
        paper(),
        [evidence(), evidence("context", citable=False)],
    )

    assert result.summary == ""
    assert result.knowledge_entries == []
    assert any("invalid evidence IDs" in item for item in result.errors)
    assert any("rejected 1" in item for item in result.errors)


@pytest.mark.asyncio
async def test_reader_deduplicates_and_skips_invalid_entries() -> None:
    response = valid_response()
    response["entries"] = [
        response["entries"][0],
        dict(response["entries"][0]),
        {
            "type": "not-a-type",
            "content": "Invalid.",
            "confidence": "high",
            "entities": [],
            "evidence_ids": ["e1"],
        },
        {
            "type": "method",
            "content": "   ",
            "confidence": None,
            "entities": [],
            "evidence_ids": ["e1"],
        },
    ]

    result = await QwenPaperReader(FakeClient(response)).read(
        "question", paper(), [evidence()]
    )

    assert len(result.knowledge_entries) == 1
    assert any("rejected 2" in item for item in result.errors)


@pytest.mark.asyncio
async def test_reader_reports_model_error_without_fabricating_content() -> None:
    result = await QwenPaperReader(FakeClient(error=OSError("model down"))).read(
        "question", paper(), [evidence()]
    )

    assert result.evidence == []
    assert result.knowledge_entries == []
    assert result.summary == ""
    assert "model down" in result.errors[0]


@pytest.mark.asyncio
async def test_reader_skips_model_when_no_evidence_and_propagates_cancellation() -> None:
    client = FakeClient(valid_response())
    empty = await QwenPaperReader(client).read("question", paper(), [])
    assert empty.errors == ["no retrievable evidence"]
    assert client.calls == []

    cancelled = FakeClient(error=asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        await QwenPaperReader(cancelled).read("question", paper(), [evidence()])
