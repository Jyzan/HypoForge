from __future__ import annotations

import asyncio
import time

import pytest

from hypoforge.literature.models import (
    ContentLevel,
    CoverageReport,
    DocumentChunk,
    DocumentRecord,
    EvidenceChunk,
    PaperReadingResult,
    PaperRecord,
    QueryIntent,
    ScoutNote,
    SearchQuery,
    SearchRunResult,
)
from hypoforge.literature.reading.store import InMemoryChunkStore
from hypoforge.literature.reading.workflow import FullTextReadingWorkflow


def paper(identifier: str) -> PaperRecord:
    return PaperRecord(paper_id=identifier, title=identifier, sources=["pubmed"])


class FakeResolver:
    def __init__(self, levels: dict[str, ContentLevel], delay: float = 0.0) -> None:
        self.levels = levels
        self.delay = delay
        self.active = 0
        self.max_active = 0

    async def resolve(self, item: PaperRecord) -> DocumentRecord:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            level = self.levels[item.paper_id]
            return DocumentRecord(
                document_id=f"doc:{item.paper_id}",
                paper_id=item.paper_id,
                content_level=level,
                local_path="unused" if level is not ContentLevel.METADATA else "",
                retrieval_error=(
                    "full text unavailable; used abstract"
                    if level is ContentLevel.ABSTRACT
                    else "no readable content" if level is ContentLevel.METADATA else ""
                ),
            )
        finally:
            self.active -= 1


class FakeParser:
    def __init__(self, failing: set[str] | None = None) -> None:
        self.failing = failing or set()

    async def parse(self, document: DocumentRecord) -> list[DocumentChunk]:
        if document.paper_id in self.failing:
            raise ValueError("broken BioC")
        return [
            DocumentChunk(
                chunk_id=f"chunk:{document.paper_id}",
                document_id=document.document_id,
                paper_id=document.paper_id,
                section="results",
                text=f"YAP evidence for {document.paper_id}.",
            )
        ]


class FakeRetriever:
    def __init__(self) -> None:
        self.queries: list[str] = []

    async def retrieve(self, query: str, paper_ids, top_k: int = 10):
        self.queries.append(query)
        identifier = paper_ids[0]
        return [
            EvidenceChunk(
                evidence_id=f"evidence:{identifier}",
                paper_id=identifier,
                chunk_id=f"chunk:{identifier}",
                section="results",
                quote=f"YAP evidence for {identifier}.",
                normalized_claim=f"YAP evidence for {identifier}.",
                relevance_score=0.9,
            )
        ][:top_k]


class FakeReader:
    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay

    async def read(self, sub_question, item, evidence):
        if self.delay:
            await asyncio.sleep(self.delay)
        return PaperReadingResult(
            paper_id=item.paper_id,
            summary=f"Summary for {item.paper_id}",
            evidence=list(evidence),
        )


class TwoEvidenceRetriever(FakeRetriever):
    async def retrieve(self, query: str, paper_ids, top_k: int = 10):
        self.queries.append(query)
        identifier = paper_ids[0]
        return [
            EvidenceChunk(
                evidence_id=f"evidence:{identifier}:first",
                paper_id=identifier,
                chunk_id=f"chunk:{identifier}:first",
                section="results",
                quote="First retrieved evidence.",
                normalized_claim="First retrieved evidence.",
                relevance_score=0.9,
            ),
            EvidenceChunk(
                evidence_id=f"evidence:{identifier}:second",
                paper_id=identifier,
                chunk_id=f"chunk:{identifier}:second",
                section="discussion",
                quote="Second retrieved evidence.",
                normalized_claim="Second retrieved evidence.",
                relevance_score=0.8,
            ),
        ][:top_k]


@pytest.mark.asyncio
async def test_workflow_keeps_all_retrieved_evidence_when_reader_cites_a_subset() -> None:
    class SubsetReader:
        async def read(self, sub_question, item, evidence):
            return PaperReadingResult(
                paper_id=item.paper_id,
                summary="Reader selected one citation.",
                evidence=[evidence[0]],
            )

    workflow = FullTextReadingWorkflow(
        resolver=FakeResolver({"p1": ContentLevel.STRUCTURED_FULLTEXT}),
        parser=FakeParser(),
        retriever=TwoEvidenceRetriever(),
        reader=SubsetReader(),
        store=InMemoryChunkStore(),
    )

    result = (await workflow.run("question", [paper("p1")]))[0]

    assert [item.evidence_id for item in result.evidence] == [
        "evidence:p1:first",
        "evidence:p1:second",
    ]


@pytest.mark.asyncio
async def test_workflow_keeps_uncited_context_evidence_authoritatively() -> None:
    class ContextOnlyRetriever(FakeRetriever):
        async def retrieve(self, query: str, paper_ids, top_k: int = 10):
            evidence = await super().retrieve(query, paper_ids, top_k)
            return [evidence[0].model_copy(update={"citable": False})]

    class ContextOnlyReader:
        async def read(self, sub_question, item, evidence):
            return PaperReadingResult(
                paper_id=item.paper_id,
                summary="Useful context but no directly cited evidence.",
                evidence=[],
            )

    workflow = FullTextReadingWorkflow(
        resolver=FakeResolver({"p1": ContentLevel.STRUCTURED_FULLTEXT}),
        parser=FakeParser(),
        retriever=ContextOnlyRetriever(),
        reader=ContextOnlyReader(),
        store=InMemoryChunkStore(),
    )

    result = (await workflow.run("question", [paper("p1")]))[0]

    assert [item.evidence_id for item in result.evidence] == ["evidence:p1"]
    assert result.evidence[0].citable is False


@pytest.mark.asyncio
async def test_workflow_keeps_retrieved_evidence_when_reader_raises() -> None:
    class ExplodingReader:
        async def read(self, sub_question, item, evidence):
            raise RuntimeError("reader failed")

    workflow = FullTextReadingWorkflow(
        resolver=FakeResolver({"p1": ContentLevel.STRUCTURED_FULLTEXT}),
        parser=FakeParser(),
        retriever=TwoEvidenceRetriever(),
        reader=ExplodingReader(),
        store=InMemoryChunkStore(),
    )

    result = (await workflow.run("question", [paper("p1")]))[0]

    assert [item.evidence_id for item in result.evidence] == [
        "evidence:p1:first",
        "evidence:p1:second",
    ]
    assert "reader failed" in " ".join(result.errors)


@pytest.mark.asyncio
async def test_workflow_keeps_retrieved_evidence_when_reader_returns_wrong_paper() -> None:
    class MismatchedReader:
        async def read(self, sub_question, item, evidence):
            return PaperReadingResult(paper_id="other-paper", evidence=[])

    workflow = FullTextReadingWorkflow(
        resolver=FakeResolver({"p1": ContentLevel.STRUCTURED_FULLTEXT}),
        parser=FakeParser(),
        retriever=TwoEvidenceRetriever(),
        reader=MismatchedReader(),
        store=InMemoryChunkStore(),
    )

    result = (await workflow.run("question", [paper("p1")]))[0]

    assert [item.evidence_id for item in result.evidence] == [
        "evidence:p1:first",
        "evidence:p1:second",
    ]
    assert "mismatched paper_id" in " ".join(result.errors)


def search_context() -> SearchRunResult:
    return SearchRunResult(
        sub_question="中文问题",
        queries=[
            SearchQuery(
                query_id="q1",
                text="YAP TAZ mechanotransduction organ size",
                intent=QueryIntent.CORE,
                target_source="pubmed",
                purpose="mechanism",
                relation_to_question="direct",
            )
        ],
        scout_notes=[
            ScoutNote(
                paper_id="full",
                key_terms=["stiffness"],
                mechanisms=["nuclear localization"],
            )
        ],
        coverage=CoverageReport(missing_topics=["contradicting evidence"]),
    )


@pytest.mark.asyncio
async def test_workflow_preserves_order_degrades_and_uses_search_context() -> None:
    papers = [paper("full"), paper("abstract"), paper("empty")]
    retriever = FakeRetriever()
    workflow = FullTextReadingWorkflow(
        resolver=FakeResolver(
            {
                "full": ContentLevel.STRUCTURED_FULLTEXT,
                "abstract": ContentLevel.ABSTRACT,
                "empty": ContentLevel.METADATA,
            }
        ),
        parser=FakeParser(),
        retriever=retriever,
        reader=FakeReader(),
        store=InMemoryChunkStore(),
    )

    results = await workflow.run("中文问题", papers, search_context=search_context())

    assert [item.paper_id for item in results] == [item.paper_id for item in papers]
    assert results[0].content_level is ContentLevel.STRUCTURED_FULLTEXT
    assert results[0].chunks_parsed == 1
    assert results[0].chunks_retrieved == 1
    assert results[1].degraded_to_abstract is True
    assert "full text unavailable" in " ".join(results[1].errors)
    assert results[2].knowledge_entries == []
    assert "no readable content" in " ".join(results[2].errors)
    assert "YAP TAZ mechanotransduction organ size" in retriever.queries[0]
    assert "stiffness" in retriever.queries[0]
    assert "contradicting evidence" in retriever.queries[0]
    assert set(results[0].stage_elapsed_seconds) == {
        "fulltext_resolver",
        "document_parser",
        "evidence_retriever",
        "paper_reader",
    }


@pytest.mark.asyncio
async def test_workflow_preserves_document_source_and_license() -> None:
    class ProvenanceResolver:
        async def resolve(self, item: PaperRecord) -> DocumentRecord:
            return DocumentRecord(
                document_id="doc:licensed",
                paper_id=item.paper_id,
                content_level=ContentLevel.STRUCTURED_FULLTEXT,
                source_uri="https://example.test/fulltext/1",
                local_path="fulltext.json",
                license="CC BY 4.0",
            )

    workflow = FullTextReadingWorkflow(
        resolver=ProvenanceResolver(),
        parser=FakeParser(),
        retriever=FakeRetriever(),
        reader=FakeReader(),
        store=InMemoryChunkStore(),
    )

    result = (await workflow.run("question", [paper("p1")]))[0]

    assert result.document_source_uri == "https://example.test/fulltext/1"
    assert result.document_license == "CC BY 4.0"


@pytest.mark.asyncio
async def test_workflow_preserves_metadata_document_source_and_license() -> None:
    class MetadataProvenanceResolver:
        async def resolve(self, item: PaperRecord) -> DocumentRecord:
            return DocumentRecord(
                document_id="doc:metadata",
                paper_id=item.paper_id,
                content_level=ContentLevel.METADATA,
                source_uri="https://example.test/metadata/1",
                license="CC0 1.0",
            )

    class ParserThatMustNotRun:
        async def parse(self, document: DocumentRecord) -> list[DocumentChunk]:
            raise AssertionError("metadata-only documents must not be parsed")

    workflow = FullTextReadingWorkflow(
        resolver=MetadataProvenanceResolver(),
        parser=ParserThatMustNotRun(),
        retriever=FakeRetriever(),
        reader=FakeReader(),
        store=InMemoryChunkStore(),
    )

    result = (await workflow.run("question", [paper("p1")]))[0]

    assert result.document_source_uri == "https://example.test/metadata/1"
    assert result.document_license == "CC0 1.0"


@pytest.mark.asyncio
async def test_workflow_isolates_parser_and_reader_timeout_failures() -> None:
    parser_failure = FullTextReadingWorkflow(
        resolver=FakeResolver({"p1": ContentLevel.STRUCTURED_FULLTEXT}),
        parser=FakeParser({"p1"}),
        retriever=FakeRetriever(),
        reader=FakeReader(),
        store=InMemoryChunkStore(),
    )
    parsed = await parser_failure.run("question", [paper("p1")])
    assert "broken BioC" in " ".join(parsed[0].errors)

    reader_timeout = FullTextReadingWorkflow(
        resolver=FakeResolver({"p1": ContentLevel.STRUCTURED_FULLTEXT}),
        parser=FakeParser(),
        retriever=FakeRetriever(),
        reader=FakeReader(delay=0.1),
        store=InMemoryChunkStore(),
        reader_timeout_seconds=0.01,
    )
    timed = await reader_timeout.run("question", [paper("p1")])
    assert "timed out" in " ".join(timed[0].errors)
    assert [item.evidence_id for item in timed[0].evidence] == ["evidence:p1"]


@pytest.mark.asyncio
async def test_workflow_degrades_only_slow_resolver_and_keeps_fast_sibling() -> None:
    class MixedSpeedResolver:
        def __init__(self) -> None:
            self.fallback_calls: list[str] = []

        async def resolve(self, item: PaperRecord) -> DocumentRecord:
            if item.paper_id == "slow":
                await asyncio.sleep(0.1)
            return DocumentRecord(
                document_id=f"doc:{item.paper_id}:full",
                paper_id=item.paper_id,
                content_level=ContentLevel.STRUCTURED_FULLTEXT,
                local_path="fulltext.json",
            )

        async def resolve_abstract(self, item: PaperRecord) -> DocumentRecord:
            self.fallback_calls.append(item.paper_id)
            return DocumentRecord(
                document_id=f"doc:{item.paper_id}:abstract",
                paper_id=item.paper_id,
                content_level=ContentLevel.ABSTRACT,
                local_path="abstract.json",
            )

    resolver = MixedSpeedResolver()
    workflow = FullTextReadingWorkflow(
        resolver=resolver,
        parser=FakeParser(),
        retriever=FakeRetriever(),
        reader=FakeReader(),
        store=InMemoryChunkStore(),
        resolver_timeout_seconds=0.01,
    )

    results = await workflow.run("question", [paper("slow"), paper("fast")])

    assert resolver.fallback_calls == ["slow"]
    assert results[0].content_level is ContentLevel.ABSTRACT
    assert results[0].degraded_to_abstract is True
    assert "fulltext_resolver timed out" in " ".join(results[0].errors)
    assert results[1].content_level is ContentLevel.STRUCTURED_FULLTEXT
    assert results[1].summary == "Summary for fast"
    assert not results[1].errors


@pytest.mark.asyncio
async def test_resolver_timeout_does_not_wait_for_cancel_suppression() -> None:
    class CancellationResistantResolver:
        def __init__(self) -> None:
            self.fallback_calls: list[str] = []

        async def resolve(self, item: PaperRecord) -> DocumentRecord:
            try:
                await asyncio.sleep(1.0)
            except asyncio.CancelledError:
                await asyncio.sleep(0.1)
            return DocumentRecord(
                document_id="doc:late",
                paper_id=item.paper_id,
                content_level=ContentLevel.METADATA,
            )

        async def resolve_abstract(self, item: PaperRecord) -> DocumentRecord:
            self.fallback_calls.append(item.paper_id)
            return DocumentRecord(
                document_id="doc:abstract",
                paper_id=item.paper_id,
                content_level=ContentLevel.ABSTRACT,
                local_path="abstract.json",
            )

    resolver = CancellationResistantResolver()
    workflow = FullTextReadingWorkflow(
        resolver=resolver,
        parser=FakeParser(),
        retriever=FakeRetriever(),
        reader=FakeReader(),
        store=InMemoryChunkStore(),
        resolver_timeout_seconds=0.01,
    )

    started = time.monotonic()
    result = (await workflow.run("question", [paper("slow")]))[0]
    elapsed = time.monotonic() - started

    assert elapsed < 0.05
    assert resolver.fallback_calls == ["slow"]
    assert result.degraded_to_abstract is True
    assert "timed out" in " ".join(result.errors)
    await asyncio.sleep(0.11)


@pytest.mark.asyncio
async def test_workflow_degrades_unreadable_pdf_to_real_abstract() -> None:
    class PDFThenAbstractResolver:
        def __init__(self) -> None:
            self.fallback_calls: list[str] = []

        async def resolve(self, item: PaperRecord) -> DocumentRecord:
            return DocumentRecord(
                document_id="doc:pdf",
                paper_id=item.paper_id,
                content_level=ContentLevel.PDF,
                local_path="unreadable.pdf",
            )

        async def resolve_abstract(self, item: PaperRecord) -> DocumentRecord:
            self.fallback_calls.append(item.paper_id)
            return DocumentRecord(
                document_id="doc:abstract",
                paper_id=item.paper_id,
                content_level=ContentLevel.ABSTRACT,
                local_path="abstract.json",
                retrieval_error="PDF parser failed; used abstract",
            )

    class PDFFailingParser:
        async def parse(self, document: DocumentRecord) -> list[DocumentChunk]:
            if document.content_level is ContentLevel.PDF:
                raise ValueError("scanned PDF has no readable text")
            return [
                DocumentChunk(
                    chunk_id="chunk:abstract",
                    document_id=document.document_id,
                    paper_id=document.paper_id,
                    section="abstract",
                    text="Real abstract evidence.",
                )
            ]

    resolver = PDFThenAbstractResolver()
    source_paper = PaperRecord(
        paper_id="ARXIV:1",
        title="Scanned preprint",
        abstract="Real abstract evidence.",
        sources=["arxiv"],
    )
    workflow = FullTextReadingWorkflow(
        resolver=resolver,
        parser=PDFFailingParser(),
        retriever=FakeRetriever(),
        reader=FakeReader(),
        store=InMemoryChunkStore(),
    )

    results = await workflow.run("question", [source_paper])

    assert resolver.fallback_calls == ["ARXIV:1"]
    assert results[0].content_level is ContentLevel.ABSTRACT
    assert results[0].degraded_to_abstract is True
    assert results[0].chunks_parsed == 1
    assert "scanned PDF" in " ".join(results[0].errors)
    assert "used abstract" in " ".join(results[0].errors)


@pytest.mark.asyncio
async def test_workflow_bounds_fetch_concurrency() -> None:
    papers = [paper(f"p{index}") for index in range(5)]
    resolver = FakeResolver(
        {item.paper_id: ContentLevel.STRUCTURED_FULLTEXT for item in papers},
        delay=0.02,
    )
    workflow = FullTextReadingWorkflow(
        resolver=resolver,
        parser=FakeParser(),
        retriever=FakeRetriever(),
        reader=FakeReader(),
        store=InMemoryChunkStore(),
        fetch_concurrency=2,
    )

    await workflow.run("question", papers)

    assert resolver.max_active == 2


@pytest.mark.asyncio
async def test_workflow_propagates_global_timeout_and_cancellation() -> None:
    timeout = FullTextReadingWorkflow(
        resolver=FakeResolver({"p1": ContentLevel.STRUCTURED_FULLTEXT}, delay=0.1),
        parser=FakeParser(),
        retriever=FakeRetriever(),
        reader=FakeReader(),
        store=InMemoryChunkStore(),
        workflow_timeout_seconds=0.01,
    )
    with pytest.raises(asyncio.TimeoutError):
        await timeout.run("question", [paper("p1")])

    class CancelledResolver:
        async def resolve(self, item):
            raise asyncio.CancelledError

    cancelled = FullTextReadingWorkflow(
        resolver=CancelledResolver(),
        parser=FakeParser(),
        retriever=FakeRetriever(),
        reader=FakeReader(),
        store=InMemoryChunkStore(),
    )
    with pytest.raises(asyncio.CancelledError):
        await cancelled.run("question", [paper("p1")])
