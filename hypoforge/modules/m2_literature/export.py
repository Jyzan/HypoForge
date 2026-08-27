from __future__ import annotations

from collections.abc import Sequence
import re

from ...state import (
    KnowledgeEntry,
    M2CoverageExport,
    M2EvidenceExport,
    M2KnowledgeRun,
    M2PaperExport,
    M2PaperRetentionExport,
    M2SearchProvenance,
    M2SearchQueryExport,
)
from .models import ContentLevel, FulltextStatus, PaperReadingResult, PaperRecord, SearchRunResult


_MAX_EXPORTED_ERRORS = 20
_MAX_EXPORTED_ERROR_LENGTH = 500
_SENSITIVE_VALUE = (
    r"api[_-]?key|key|token|access[_-]?token|secret|signature|sig"
)
_SENSITIVE_ASSIGNMENT = re.compile(
    rf"(?i)\b(?P<name>{_SENSITIVE_VALUE})\b(?P<separator>\s*[=:]\s*)"
    r"""(?P<value>"[^"]*"|'[^']*'|[^\s&;,"']+)"""
)
_BEARER_TOKEN = re.compile(r"(?i)\bBearer\s+[^\s,;]+")
_LIKELY_API_SECRET = re.compile(r"\b(?:sk|rk|pk)[_-][A-Za-z0-9_-]{12,}\b")
_QUOTED_WINDOWS_PATH = re.compile(
    r"""(?P<quote>["'])(?P<path>[A-Za-z]:[\\/][^"']+)(?P=quote)"""
)
_QUOTED_POSIX_PATH = re.compile(
    r"""(?P<quote>["'])(?P<path>/[^"']+)(?P=quote)"""
)
_DIAGNOSTIC_SUFFIX = (
    r"(?:failed|failure|error|exception|timeout|timed out|"
    r"not found|denied|unavailable)"
)
_UNQUOTED_WINDOWS_PATH_WITH_DIAGNOSTIC_SUFFIX = re.compile(
    rf"(?i)(?<![A-Za-z0-9])[A-Za-z]:[\\/][^\r\n\"';,:]*"
    rf"(?=\s+{_DIAGNOSTIC_SUFFIX}(?=$|[\s,;:.]))"
)
_UNQUOTED_POSIX_PATH_WITH_DIAGNOSTIC_SUFFIX = re.compile(
    rf"(?i)(?<![:/\w])/[^\r\n\"';,:]*"
    rf"(?=\s+{_DIAGNOSTIC_SUFFIX}(?=$|[\s,;:.]))"
)
_UNQUOTED_WINDOWS_PATH_TO_BOUNDARY = re.compile(
    r"(?i)(?<![A-Za-z0-9])[A-Za-z]:[\\/][^\r\n\"';,:]*"
    r"(?=[;,:]|\.(?=\s|$)|$)"
)
_UNQUOTED_POSIX_PATH_TO_BOUNDARY = re.compile(
    r"(?i)(?<![:/\w])/[^\r\n\"';,:]*"
    r"(?=[;,:]|\.(?=\s|$)|$)"
)
_WINDOWS_PATH = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/][^\s\"';,]+")
_UNC_PATH = re.compile(r"(?<!\\)\\\\[^\s\"';,]+")
_FILE_URI_PATH = re.compile(r"(?i)\bfile:///[^\s\"';,]+")
_POSIX_PATH = re.compile(r"(?<![:/\w])/(?:[^\s\"';,]+)")


def _redact_quoted_path(match: re.Match[str]) -> str:
    quote = match.group("quote")
    return f"{quote}[REDACTED_PATH]{quote}"


def _sanitize_error_messages(errors: Sequence[str]) -> list[str]:
    """Return portable, bounded diagnostics for the M2 export artifact."""

    sanitized: list[str] = []
    for error in errors[:_MAX_EXPORTED_ERRORS]:
        message = " ".join(str(error).split())
        message = _QUOTED_WINDOWS_PATH.sub(_redact_quoted_path, message)
        message = _QUOTED_POSIX_PATH.sub(_redact_quoted_path, message)
        message = _UNQUOTED_WINDOWS_PATH_WITH_DIAGNOSTIC_SUFFIX.sub(
            "[REDACTED_PATH]", message
        )
        message = _UNQUOTED_POSIX_PATH_WITH_DIAGNOSTIC_SUFFIX.sub(
            "[REDACTED_PATH]", message
        )
        message = _UNQUOTED_WINDOWS_PATH_TO_BOUNDARY.sub(
            "[REDACTED_PATH]", message
        )
        message = _UNQUOTED_POSIX_PATH_TO_BOUNDARY.sub(
            "[REDACTED_PATH]", message
        )
        message = _WINDOWS_PATH.sub("[REDACTED_PATH]", message)
        message = _UNC_PATH.sub("[REDACTED_PATH]", message)
        message = _FILE_URI_PATH.sub("[REDACTED_PATH]", message)
        message = _POSIX_PATH.sub("[REDACTED_PATH]", message)
        message = _BEARER_TOKEN.sub("Bearer [REDACTED]", message)
        message = _LIKELY_API_SECRET.sub("[REDACTED]", message)
        message = _SENSITIVE_ASSIGNMENT.sub(
            lambda match: f"{match.group('name')}{match.group('separator')}[REDACTED]",
            message,
        )
        if len(message) > _MAX_EXPORTED_ERROR_LENGTH:
            message = message[: _MAX_EXPORTED_ERROR_LENGTH - 3].rstrip() + "..."
        if message:
            sanitized.append(message)
    return sanitized


def _enum_value(value: object) -> str:
    return str(getattr(value, "value", value or ""))


def _paper_export(
    paper: PaperRecord,
    reading: PaperReadingResult,
) -> M2PaperExport:
    if reading.content_level is ContentLevel.STRUCTURED_FULLTEXT:
        resolved_status = FulltextStatus.DOWNLOADED.value
    elif reading.content_level is ContentLevel.PDF:
        resolved_status = FulltextStatus.DOWNLOADED.value
    elif reading.content_level is ContentLevel.HTML:
        resolved_status = FulltextStatus.HTML_AVAILABLE.value
    elif reading.content_level is ContentLevel.ABSTRACT:
        resolved_status = FulltextStatus.ABSTRACT_ONLY.value
    elif reading.errors:
        resolved_status = FulltextStatus.UNAVAILABLE.value
    else:
        resolved_status = _enum_value(paper.fulltext_status)
    failure_category = reading.fulltext_failure_category
    failure_detail = reading.fulltext_failure_detail
    if resolved_status == FulltextStatus.DOWNLOADED.value:
        # Successful full-text retrieval carries no failure attribution.
        failure_category = ""
        failure_detail = ""
    sanitized_failure_detail = (
        _sanitize_error_messages([failure_detail])[0] if failure_detail else ""
    )
    return M2PaperExport(
        paper_id=paper.paper_id,
        title=paper.title,
        abstract=paper.abstract,
        authors=list(paper.authors),
        year=paper.year,
        journal=paper.journal,
        doi=paper.doi,
        pmid=paper.pmid,
        pmcid=paper.pmcid,
        external_ids=dict(paper.external_ids),
        citation_count=paper.citation_count,
        publication_type=paper.publication_type,
        sources=list(paper.sources),
        is_open_access=paper.is_open_access,
        # Reading is authoritative: search-time availability is stale once
        # resolution/parsing has completed.
        fulltext_status=resolved_status,
        rank_scores=dict(paper.rank_scores),
        reading_summary=reading.summary,
        content_level=_enum_value(reading.content_level),
        document_id=reading.document_id,
        document_source_uri=reading.document_source_uri,
        document_license=reading.document_license,
        degraded_to_abstract=reading.degraded_to_abstract,
        fulltext_failure_category=failure_category,
        fulltext_failure_detail=sanitized_failure_detail,
        chunks_parsed=reading.chunks_parsed,
        chunks_retrieved=reading.chunks_retrieved,
        stage_elapsed_seconds=dict(reading.stage_elapsed_seconds),
        errors=_sanitize_error_messages(reading.errors),
    )


def _evidence_exports(reading: PaperReadingResult) -> list[M2EvidenceExport]:
    return [M2EvidenceExport(**item.model_dump(mode="python")) for item in reading.evidence]


def _knowledge_exports(
    paper: PaperRecord,
    reading: PaperReadingResult,
) -> list[KnowledgeEntry]:
    return [
        KnowledgeEntry(
            id=item.entry_id,
            type=item.entry_type,
            content=item.content,
            confidence=item.confidence,
            source_paper_id=paper.paper_id,
            source_paper_title=paper.title,
            entities=list(item.entities),
            evidence_ids=list(item.evidence_ids),
        )
        for item in reading.knowledge_entries
    ]


def _search_provenance(search_result: SearchRunResult) -> M2SearchProvenance:
    coverage = search_result.coverage
    return M2SearchProvenance(
        queries=[
            M2SearchQueryExport(
                query_id=query.query_id,
                text=query.text,
                round_index=query.round_index,
                intent=_enum_value(query.intent),
                target_source=query.target_source,
                purpose=query.purpose,
                target_gap=query.target_gap,
                relation_to_question=query.relation_to_question,
            )
            for query in search_result.queries
        ],
        coverage=M2CoverageExport(
            covered_buckets=sorted(
                _enum_value(bucket) for bucket in coverage.covered_buckets
            ),
            missing_buckets=sorted(
                _enum_value(bucket) for bucket in coverage.missing_buckets
            ),
            covered_topics=list(coverage.covered_topics),
            missing_topics=list(coverage.missing_topics),
            sufficient=coverage.sufficient,
            rationale=coverage.rationale,
        ),
        source_result_counts=dict(search_result.source_result_counts),
        failed_sources=list(search_result.failed_sources),
        iterations=search_result.iterations,
        stop_reason=(
            _enum_value(search_result.stop_reason)
            if search_result.stop_reason is not None
            else None
        ),
        errors=_sanitize_error_messages(search_result.errors),
        stage_elapsed_seconds=dict(search_result.stage_elapsed_seconds),
        papers_found=search_result.papers_found,
        papers_after_dedup=search_result.papers_after_dedup,
        retention_decisions=[
            M2PaperRetentionExport(**decision.model_dump(mode="python"))
            for decision in search_result.retention_decisions
        ],
    )


def _reading_by_paper_id(
    search_result: SearchRunResult,
    reading_results: Sequence[PaperReadingResult],
) -> dict[str, PaperReadingResult]:
    readings: dict[str, PaperReadingResult] = {}
    for reading in reading_results:
        if reading.paper_id in readings:
            raise ValueError(f"duplicate reading result for paper ID: {reading.paper_id}")
        for evidence in reading.evidence:
            if evidence.paper_id != reading.paper_id:
                raise ValueError(
                    "reading result for paper ID: "
                    f"{reading.paper_id} contains evidence {evidence.evidence_id!r} "
                    f"for paper ID: {evidence.paper_id}"
                )
        readings[reading.paper_id] = reading

    final_paper_ids = {paper.paper_id for paper in search_result.final_papers}
    for paper_id in readings:
        if paper_id not in final_paper_ids:
            raise ValueError(f"reading result for unknown paper ID: {paper_id}")
    for paper in search_result.final_papers:
        if paper.paper_id not in readings:
            raise ValueError(f"missing reading result for paper ID: {paper.paper_id}")
    return readings


def build_m2_knowledge_export_run(
    sub_question: str,
    search_result: SearchRunResult,
    reading_results: Sequence[PaperReadingResult],
) -> M2KnowledgeRun:
    readings = _reading_by_paper_id(search_result, reading_results)
    papers = []
    evidence = []
    knowledge_entries = []
    for paper in search_result.final_papers:
        reading = readings[paper.paper_id]
        papers.append(_paper_export(paper, reading))
        evidence.extend(_evidence_exports(reading))
        knowledge_entries.extend(_knowledge_exports(paper, reading))

    return M2KnowledgeRun(
        sub_question=sub_question,
        papers=papers,
        evidence=evidence,
        knowledge_entries=knowledge_entries,
        search_provenance=_search_provenance(search_result),
    )
