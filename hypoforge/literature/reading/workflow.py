"""Orchestrate full-text resolution, RAG retrieval, and paper reading."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Sequence
from typing import TypeVar

from ..models import (
    ContentLevel,
    DocumentRecord,
    PaperReadingResult,
    PaperRecord,
    SearchRunResult,
)
from ..protocols import (
    DocumentParserProtocol,
    EvidenceRetrieverProtocol,
    FulltextResolverProtocol,
    PaperReaderProtocol,
    ReadingExtractionWorkflowProtocol,
)
from .store import InMemoryChunkStore

T = TypeVar("T")


def _error(stage: str, exc: BaseException) -> str:
    detail = " ".join(f"{type(exc).__name__}: {exc}".split())[:450]
    return f"{stage} failed: {detail}"


def _retrieval_query(
    sub_question: str,
    context: SearchRunResult | None,
    limit: int = 4000,
) -> str:
    values: list[str] = [sub_question]
    if context is not None:
        values.extend(item.text for item in context.queries)
        values.extend(context.coverage.missing_topics)
        for note in context.scout_notes:
            values.extend(note.key_terms)
            values.extend(note.entities)
            values.extend(note.mechanisms)
            values.extend(note.controversies)
    output: list[str] = []
    seen: set[str] = set()
    used = 0
    for value in values:
        clean = " ".join(str(value or "").split())
        key = clean.casefold()
        if not clean or key in seen:
            continue
        extra = len(clean) + (1 if output else 0)
        if used + extra > limit:
            remaining = limit - used - (1 if output else 0)
            if remaining > 0:
                output.append(clean[:remaining])
            break
        seen.add(key)
        output.append(clean)
        used += extra
    return " ".join(output)


class FullTextReadingWorkflow(ReadingExtractionWorkflowProtocol):
    def __init__(
        self,
        *,
        resolver: FulltextResolverProtocol,
        parser: DocumentParserProtocol,
        retriever: EvidenceRetrieverProtocol,
        reader: PaperReaderProtocol,
        store: InMemoryChunkStore,
        fetch_concurrency: int = 3,
        read_concurrency: int = 2,
        resolver_timeout_seconds: float = 210.0,
        reader_timeout_seconds: float = 120.0,
        workflow_timeout_seconds: float = 600.0,
        top_k: int = 8,
    ) -> None:
        if fetch_concurrency <= 0 or read_concurrency <= 0:
            raise ValueError("reading concurrency limits must be positive")
        if (
            resolver_timeout_seconds <= 0
            or reader_timeout_seconds <= 0
            or workflow_timeout_seconds <= 0
        ):
            raise ValueError("reading timeouts must be positive")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self.resolver = resolver
        self.parser = parser
        self.retriever = retriever
        self.reader = reader
        self.store = store
        self.fetch_concurrency = fetch_concurrency
        self.read_concurrency = read_concurrency
        self.resolver_timeout_seconds = resolver_timeout_seconds
        self.reader_timeout_seconds = reader_timeout_seconds
        self.workflow_timeout_seconds = workflow_timeout_seconds
        self.top_k = top_k
        self._detached_resolver_tasks: set[asyncio.Task[DocumentRecord]] = set()

    def _detach_resolver_task(
        self,
        task: asyncio.Task[DocumentRecord],
    ) -> None:
        """Keep and reap a resolver that ignored cancellation after timeout."""

        self._detached_resolver_tasks.add(task)

        def discard(completed: asyncio.Task[DocumentRecord]) -> None:
            self._detached_resolver_tasks.discard(completed)
            try:
                completed.exception()
            except (asyncio.CancelledError, Exception):
                pass

        task.add_done_callback(discard)

    async def _resolve_with_timeout(self, paper: PaperRecord) -> DocumentRecord:
        task = asyncio.create_task(self.resolver.resolve(paper))
        try:
            done, _ = await asyncio.wait(
                {task},
                timeout=self.resolver_timeout_seconds,
            )
        except asyncio.CancelledError:
            task.cancel()
            self._detach_resolver_task(task)
            raise
        if task not in done:
            task.cancel()
            self._detach_resolver_task(task)
            raise asyncio.TimeoutError
        return task.result()

    @staticmethod
    async def _measure(
        timings: dict[str, float],
        stage: str,
        operation: Callable[[], Awaitable[T]],
    ) -> T:
        started = time.monotonic()
        try:
            return await operation()
        finally:
            timings[stage] = timings.get(stage, 0.0) + (time.monotonic() - started)

    @staticmethod
    def _document_result_fields(document: DocumentRecord) -> dict[str, str]:
        return {
            "document_id": document.document_id,
            "document_source_uri": document.source_uri,
            "document_license": document.license,
        }

    async def _read_one(
        self,
        sub_question: str,
        retrieval_query: str,
        paper: PaperRecord,
        fetch_semaphore: asyncio.Semaphore,
        read_semaphore: asyncio.Semaphore,
    ) -> PaperReadingResult:
        timings: dict[str, float] = {}
        resolver_errors: list[str] = []
        try:
            async with fetch_semaphore:
                document = await self._measure(
                    timings,
                    "fulltext_resolver",
                    lambda: self._resolve_with_timeout(paper),
                )
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:
            fallback = getattr(self.resolver, "resolve_abstract", None)
            if not callable(fallback):
                return PaperReadingResult(
                    paper_id=paper.paper_id,
                    stage_elapsed_seconds=timings,
                    errors=["fulltext_resolver timed out; abstract fallback unavailable"],
                )
            resolver_errors.append("fulltext_resolver timed out; used abstract")
            try:
                document = await self._measure(
                    timings,
                    "abstract_fallback",
                    lambda: fallback(paper),
                )
            except asyncio.CancelledError:
                raise
            except Exception as fallback_exc:
                return PaperReadingResult(
                    paper_id=paper.paper_id,
                    stage_elapsed_seconds=timings,
                    errors=[
                        *resolver_errors,
                        _error("abstract_fallback", fallback_exc),
                    ],
                )
        except Exception as exc:
            return PaperReadingResult(
                paper_id=paper.paper_id,
                stage_elapsed_seconds=timings,
                errors=[_error("fulltext_resolver", exc)],
            )

        document_errors = [
            *resolver_errors,
            *([document.retrieval_error] if document.retrieval_error else []),
        ]
        if document.content_level is ContentLevel.METADATA or not document.local_path:
            if not document_errors:
                document_errors.append("no readable content")
            return PaperReadingResult(
                paper_id=paper.paper_id,
                content_level=document.content_level,
                **self._document_result_fields(document),
                stage_elapsed_seconds=timings,
                errors=document_errors,
            )

        try:
            chunks = await self._measure(
                timings, "document_parser", lambda: self.parser.parse(document)
            )
            self.store.replace(paper.paper_id, chunks)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            parser_error = _error("document_parser", exc)
            fallback = getattr(self.resolver, "resolve_abstract", None)
            if document.content_level is not ContentLevel.PDF or not callable(fallback):
                return PaperReadingResult(
                    paper_id=paper.paper_id,
                    content_level=document.content_level,
                    **self._document_result_fields(document),
                    degraded_to_abstract=(
                        document.content_level is ContentLevel.ABSTRACT
                    ),
                    stage_elapsed_seconds=timings,
                    errors=[*document_errors, parser_error],
                )
            try:
                fallback_document = await self._measure(
                    timings,
                    "abstract_fallback",
                    lambda: fallback(paper),
                )
                if (
                    fallback_document.content_level is not ContentLevel.ABSTRACT
                    or not fallback_document.local_path
                ):
                    raise ValueError("abstract fallback returned no readable abstract")
                chunks = await self._measure(
                    timings,
                    "document_parser",
                    lambda: self.parser.parse(fallback_document),
                )
                self.store.replace(paper.paper_id, chunks)
                document = fallback_document
                document_errors = [
                    *document_errors,
                    parser_error,
                    *(
                        [fallback_document.retrieval_error]
                        if fallback_document.retrieval_error
                        else []
                    ),
                ]
            except asyncio.CancelledError:
                raise
            except Exception as fallback_exc:
                return PaperReadingResult(
                    paper_id=paper.paper_id,
                    content_level=document.content_level,
                    **self._document_result_fields(document),
                    stage_elapsed_seconds=timings,
                    errors=[
                        *document_errors,
                        parser_error,
                        _error("abstract_fallback", fallback_exc),
                    ],
                )

        try:
            evidence = await self._measure(
                timings,
                "evidence_retriever",
                lambda: self.retriever.retrieve(
                    retrieval_query, [paper.paper_id], top_k=self.top_k
                ),
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return PaperReadingResult(
                paper_id=paper.paper_id,
                content_level=document.content_level,
                **self._document_result_fields(document),
                chunks_parsed=len(chunks),
                degraded_to_abstract=document.content_level is ContentLevel.ABSTRACT,
                stage_elapsed_seconds=timings,
                errors=[*document_errors, _error("evidence_retriever", exc)],
            )

        try:
            async with read_semaphore:
                reading = await self._measure(
                    timings,
                    "paper_reader",
                    lambda: asyncio.wait_for(
                        self.reader.read(sub_question, paper, evidence),
                        timeout=self.reader_timeout_seconds,
                    ),
                )
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:
            return PaperReadingResult(
                paper_id=paper.paper_id,
                content_level=document.content_level,
                **self._document_result_fields(document),
                chunks_parsed=len(chunks),
                chunks_retrieved=len(evidence),
                evidence=list(evidence),
                degraded_to_abstract=document.content_level is ContentLevel.ABSTRACT,
                stage_elapsed_seconds=timings,
                errors=[*document_errors, "paper_reader timed out"],
            )
        except Exception as exc:
            return PaperReadingResult(
                paper_id=paper.paper_id,
                content_level=document.content_level,
                **self._document_result_fields(document),
                chunks_parsed=len(chunks),
                chunks_retrieved=len(evidence),
                evidence=list(evidence),
                degraded_to_abstract=document.content_level is ContentLevel.ABSTRACT,
                stage_elapsed_seconds=timings,
                errors=[*document_errors, _error("paper_reader", exc)],
            )

        if reading.paper_id != paper.paper_id:
            return PaperReadingResult(
                paper_id=paper.paper_id,
                content_level=document.content_level,
                **self._document_result_fields(document),
                chunks_parsed=len(chunks),
                chunks_retrieved=len(evidence),
                evidence=list(evidence),
                degraded_to_abstract=document.content_level is ContentLevel.ABSTRACT,
                stage_elapsed_seconds=timings,
                errors=[*document_errors, "paper_reader returned a mismatched paper_id"],
            )
        return reading.model_copy(
            update={
                "content_level": document.content_level,
                **self._document_result_fields(document),
                "chunks_parsed": len(chunks),
                "chunks_retrieved": len(evidence),
                "evidence": list(evidence),
                "degraded_to_abstract": (
                    reading.degraded_to_abstract
                    or document.content_level is ContentLevel.ABSTRACT
                ),
                "stage_elapsed_seconds": timings,
                "errors": [*document_errors, *reading.errors],
            }
        )

    async def run(
        self,
        sub_question: str,
        papers: Sequence[PaperRecord],
        search_context: SearchRunResult | None = None,
    ) -> list[PaperReadingResult]:
        query = _retrieval_query(sub_question, search_context)
        fetch_semaphore = asyncio.Semaphore(self.fetch_concurrency)
        read_semaphore = asyncio.Semaphore(self.read_concurrency)

        async def run_all() -> list[PaperReadingResult]:
            return list(
                await asyncio.gather(
                    *(
                        self._read_one(
                            sub_question,
                            query,
                            paper,
                            fetch_semaphore,
                            read_semaphore,
                        )
                        for paper in papers
                    )
                )
            )

        return await asyncio.wait_for(run_all(), timeout=self.workflow_timeout_seconds)
