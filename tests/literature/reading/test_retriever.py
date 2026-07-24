from __future__ import annotations

import pytest

from hypoforge.literature.models import DocumentChunk
from hypoforge.literature.reading.retriever import HybridEvidenceRetriever
from hypoforge.literature.reading.store import InMemoryChunkStore


def chunk(
    chunk_id: str,
    section: str,
    text: str,
    *,
    paper_id: str = "PMID:1",
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=f"doc:{paper_id}",
        paper_id=paper_id,
        section=section,
        text=text,
    )


@pytest.mark.asyncio
async def test_retriever_covers_mechanism_method_and_limitation_intents() -> None:
    chunks = [
        chunk(
            "c1",
            "results",
            "Mechanical tension activates YAP and TAZ to increase organ size.",
        ),
        chunk(
            "c2",
            "methods",
            "CRISPR knockout and RNA sequencing measured YAP pathway activity.",
        ),
        chunk(
            "c3",
            "limitations",
            "No effect was detected in zebrafish and the small sample limits inference.",
        ),
        chunk("c4", "introduction", "Hippo signaling regulates tissue growth."),
    ]
    store = InMemoryChunkStore()
    store.replace("PMID:1", chunks)
    retriever = HybridEvidenceRetriever(store)

    first = await retriever.retrieve(
        "YAP TAZ mechanotransduction organ size", ["PMID:1"], top_k=6
    )
    second = await retriever.retrieve(
        "YAP TAZ mechanotransduction organ size", ["PMID:1"], top_k=6
    )

    assert {item.section for item in first} >= {"results", "methods", "limitations"}
    assert len({item.chunk_id for item in first}) == len(first)
    assert all(0.0 <= item.relevance_score <= 1.0 for item in first)
    assert {item.chunk_id: item.quote for item in first}.items() <= {
        item.chunk_id: item.text for item in chunks
    }.items()
    assert first == second


@pytest.mark.asyncio
async def test_retriever_filters_papers_and_rewards_exact_entities() -> None:
    store = InMemoryChunkStore()
    store.replace(
        "PMID:1",
        [
            chunk("exact", "results", "YAP TAZ jointly control organ size."),
            chunk("generic", "results", "The pathway controls organ size."),
        ],
    )
    store.replace(
        "PMID:2",
        [chunk("other", "results", "YAP TAZ evidence.", paper_id="PMID:2")],
    )

    results = await HybridEvidenceRetriever(store).retrieve(
        "YAP TAZ organ size", ["PMID:1"], top_k=2
    )

    assert results[0].chunk_id == "exact"
    assert all(item.paper_id == "PMID:1" for item in results)


@pytest.mark.asyncio
async def test_retriever_adds_same_section_neighbor_within_budget() -> None:
    store = InMemoryChunkStore()
    store.replace(
        "PMID:1",
        [
            chunk("before", "results", "Cells were placed on a stiff matrix."),
            chunk("hit", "results", "Stiffness caused nuclear YAP accumulation."),
            chunk("after", "results", "The response disappeared after YAP knockout."),
        ],
    )

    results = await HybridEvidenceRetriever(store, max_context_chars=500).retrieve(
        "stiffness nuclear YAP", ["PMID:1"], top_k=2
    )

    ids = {item.chunk_id for item in results}
    assert "hit" in ids
    assert ids & {"before", "after"}
    primary = next(item for item in results if item.chunk_id == "hit")
    neighbor = next(item for item in results if item.chunk_id != "hit")
    assert primary.citable is True
    assert neighbor.citable is False


@pytest.mark.asyncio
async def test_retriever_returns_empty_for_empty_query_or_catalog() -> None:
    store = InMemoryChunkStore()
    retriever = HybridEvidenceRetriever(store)

    assert await retriever.retrieve("question", ["missing"]) == []
    store.replace("PMID:1", [chunk("c1", "results", "YAP evidence")])
    assert await retriever.retrieve("", ["PMID:1"]) == []
