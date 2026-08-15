"""Orchestrate full-text resolution, RAG retrieval, and paper reading."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Sequence
from typing import TypeVar

from ...observability import emit_event
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
from .attribution import OTHER, attribute_error_text
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
        self._reading_cache: dict[tuple[str, str], PaperReadingResult] = {}

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

    async def probe_parsed_fulltext(self, paper: PaperRecord) -> bool:
        """Download and parse once before spending PaperReader LLM tokens.

        The resolver/parser caches make a subsequent ``run()`` inexpensive.
        Abstract and metadata fallbacks intentionally return ``False``.
        """

        try:
            document = await self._resolve_with_timeout(paper)
            if (
                document.content_level not in {
                    ContentLevel.STRUCTURED_FULLTEXT,
                    ContentLevel.PDF,
                    ContentLevel.HTML,
                    ContentLevel.OCR,
                }
                or not document.local_path
            ):
                return False
            chunks = await self.parser.parse(document)
            if not chunks:
                return False
            self.store.replace(paper.paper_id, chunks)
            return True
        except asyncio.CancelledError:
            raise
        except Exception:
            return False

    @staticmethod
    async def _measure(
        timings: dict[str, float],
        stage: str,
        operation: Callable[[], Awaitable[T]],
        *,
        details: dict[str, object] | None = None,
    ) -> T:
        started = time.monotonic()
        emit_event(
            "tool_started",
            module="m2",
            tool=stage,
            status="running",
            message=f"M2 阅读 Tool 开始：{stage}",
            details=details,
        )
        try:
            result = await operation()
        except BaseException as exc:
            emit_event(
                "tool_failed",
                module="m2",
                tool=stage,
                status="failed",
                message=f"M2 阅读 Tool 失败：{stage}：{type(exc).__name__}: {exc}",
                elapsed_seconds=time.monotonic() - started,
                details=details,
            )
            raise
        else:
            emit_event(
                "tool_completed",
                module="m2",
                tool=stage,
                status="completed",
                message=f"M2 阅读 Tool 完成：{stage}",
                elapsed_seconds=time.monotonic() - started,
                details=details,
            )
            return result
        finally:
            timings[stage] = timings.get(stage, 0.0) + (time.monotonic() - started)

    @staticmethod
    def _document_result_fields(document: DocumentRecord) -> dict[str, str]:
        return {
            "document_id": document.document_id,
            "document_source_uri": document.source_uri,
            "document_license": document.license,
            "fulltext_failure_category": document.retrieval_failure_category,
            "fulltext_failure_detail": document.retrieval_failure_detail,
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
                    details={"paper_id": paper.paper_id, "title": paper.title},
                )
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:
            fallback = getattr(self.resolver, "resolve_abstract", None)
            if not callable(fallback):
                message = "fulltext_resolver timed out; abstract fallback unavailable"
                return PaperReadingResult(
                    paper_id=paper.paper_id,
                    stage_elapsed_seconds=timings,
                    fulltext_failure_category=attribute_error_text(message),
                    fulltext_failure_detail=message,
                    errors=[message],
                )
            resolver_errors.append("fulltext_resolver timed out; used abstract")
            try:
                document = await self._measure(
                    timings,
                    "abstract_fallback",
                    lambda: fallback(paper),
                    details={"paper_id": paper.paper_id, "title": paper.title},
                )
            except asyncio.CancelledError:
                raise
            except Exception as fallback_exc:
                fallback_error = _error("abstract_fallback", fallback_exc)
                return PaperReadingResult(
                    paper_id=paper.paper_id,
                    stage_elapsed_seconds=timings,
                    fulltext_failure_category=attribute_error_text(fallback_error),
                    fulltext_failure_detail=fallback_error,
                    errors=[
                        *resolver_errors,
                        fallback_error,
                    ],
                )
            document = document.model_copy(
                update={
                    "retrieval_failure_category": OTHER,
                    "retrieval_failure_detail": "全文解析超时，已降级为摘要",
                }
            )
        except Exception as exc:
            resolver_error = _error("fulltext_resolver", exc)
            return PaperReadingResult(
                paper_id=paper.paper_id,
                stage_elapsed_seconds=timings,
                fulltext_failure_category=attribute_error_text(resolver_error),
                fulltext_failure_detail=resolver_error,
                errors=[resolver_error],
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
                timings,
                "document_parser",
                lambda: self.parser.parse(document),
                details={"paper_id": paper.paper_id, "title": paper.title},
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
                    details={"paper_id": paper.paper_id, "title": paper.title},
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
                    details={"paper_id": paper.paper_id, "title": paper.title},
                )
                self.store.replace(paper.paper_id, chunks)
                document = fallback_document.model_copy(
                    update={
                        # Retrieval itself succeeded; the parse failure is the
                        # real reason this paper degraded to its abstract.
                        "retrieval_failure_category": OTHER,
                        "retrieval_failure_detail": parser_error,
                    }
                )
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
                details={
                    "paper_id": paper.paper_id,
                    "title": paper.title,
                    "top_k": self.top_k,
                },
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
                    details={
                        "paper_id": paper.paper_id,
                        "title": paper.title,
                        "evidence_chunks": len(evidence),
                    },
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
        reader_evidence = {item.evidence_id: item for item in reading.evidence}
        merged_evidence = [
            item.model_copy(update={
                "normalized_claim": reader_evidence[item.evidence_id].normalized_claim
            })
            if item.evidence_id in reader_evidence
            else item
            for item in evidence
        ]
        return reading.model_copy(
            update={
                "content_level": document.content_level,
                **self._document_result_fields(document),
                "chunks_parsed": len(chunks),
                "chunks_retrieved": len(evidence),
                "evidence": merged_evidence,
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
        started_at = time.monotonic()
        emit_event(
            "tool_started",
            module="m2",
            tool="reading_workflow",
            status="running",
            message=f"开始全文/RAG 阅读 {len(papers)} 篇入选论文",
            details={"papers": len(papers), "sub_question": sub_question},
        )
        fetch_semaphore = asyncio.Semaphore(self.fetch_concurrency)
        read_semaphore = asyncio.Semaphore(self.read_concurrency)

        async def run_all() -> list[PaperReadingResult]:
            async def cached_read(paper: PaperRecord) -> PaperReadingResult:
                key = (" ".join(sub_question.casefold().split()), paper.paper_id)
                cached = self._reading_cache.get(key)
                if cached is not None:
                    return cached.model_copy(deep=True)
                result = await self._read_one(
                    sub_question,
                    query,
                    paper,
                    fetch_semaphore,
                    read_semaphore,
                )
                self._reading_cache[key] = result.model_copy(deep=True)
                return result

            return list(await asyncio.gather(*(cached_read(paper) for paper in papers)))

        try:
            results = await asyncio.wait_for(
                run_all(), timeout=self.workflow_timeout_seconds
            )
        except BaseException as exc:
            emit_event(
                "tool_failed",
                module="m2",
                tool="reading_workflow",
                status="failed",
                message=f"全文/RAG 阅读流程失败：{type(exc).__name__}: {exc}",
                elapsed_seconds=time.monotonic() - started_at,
            )
            raise
        emit_event(
            "tool_completed",
            module="m2",
            tool="reading_workflow",
            status="completed",
            message=f"全文/RAG 阅读完成：{len(results)} 篇",
            elapsed_seconds=time.monotonic() - started_at,
            details={
                "papers": len(results),
                "with_errors": sum(bool(result.errors) for result in results),
                "knowledge_entries": sum(
                    len(result.knowledge_entries) for result in results
                ),
            },
        )
        return results
