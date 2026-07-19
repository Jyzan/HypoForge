from __future__ import annotations

import pytest

from hypoforge.literature.export import build_m2_knowledge_export_run
from hypoforge.literature.models import (
    ContentLevel,
    CoverageReport,
    EvidenceBucket,
    EvidenceChunk,
    EvidenceLinkedKnowledge,
    FulltextStatus,
    PaperReadingResult,
    PaperRecord,
    QueryIntent,
    SearchQuery,
    SearchRunResult,
    StopReason,
)
from hypoforge.state import ConfidenceLevel, KnowledgeEntryType


def make_paper(paper_id: str, title: str) -> PaperRecord:
    return PaperRecord(
        paper_id=paper_id,
        title=title,
        sources=["pubmed"],
    )


def make_reading(paper_id: str) -> PaperReadingResult:
    return PaperReadingResult(paper_id=paper_id)


def test_builder_exports_papers_evidence_knowledge_and_search_provenance() -> None:
    paper = PaperRecord(
        paper_id="PMID:1",
        title="Force and YAP",
        abstract="Force changes YAP transport.",
        authors=["A. Author"],
        year=2025,
        journal="Mechanobiology",
        doi="10.1/example",
        pmid="1",
        pmcid="PMC1",
        external_ids={"arxiv": "1234.5678"},
        citation_count=12,
        publication_type="article",
        sources=["pubmed"],
        is_open_access=True,
        fulltext_status=FulltextStatus.XML_AVAILABLE,
        rank_scores={"relevance": 0.9},
    )
    evidence = EvidenceChunk(
        evidence_id="ev-1",
        paper_id="PMID:1",
        chunk_id="chunk-1",
        section="results",
        page=4,
        quote="Force changed YAP transport.",
        normalized_claim="Force changes YAP transport.",
        relevance_score=0.95,
    )
    reading = PaperReadingResult(
        paper_id="PMID:1",
        summary="Force promotes transport.",
        evidence=[evidence],
        knowledge_entries=[
            EvidenceLinkedKnowledge(
                entry_id="knowledge-1",
                entry_type=KnowledgeEntryType.MECHANISTIC_CONCLUSION,
                content="Force changes YAP transport.",
                confidence=ConfidenceLevel.HIGH,
                entities=["YAP"],
                evidence_ids=["ev-1"],
            )
        ],
        content_level=ContentLevel.STRUCTURED_FULLTEXT,
        document_id="doc-1",
        document_source_uri="https://example.test/fulltext/1",
        document_license="CC BY 4.0",
        chunks_parsed=8,
        chunks_retrieved=1,
        stage_elapsed_seconds={"parse": 0.1},
        errors=["minor extraction warning"],
    )
    search = SearchRunResult(
        sub_question="question",
        queries=[
            SearchQuery(
                query_id="q-1",
                text="YAP force",
                round_index=1,
                intent=QueryIntent.MESH,
                target_source="pubmed",
                purpose="mechanism",
                target_gap="supporting",
                relation_to_question="Direct mechanism evidence.",
            )
        ],
        papers_found=3,
        papers_after_dedup=2,
        final_papers=[paper],
        coverage=CoverageReport(
            covered_buckets={EvidenceBucket.RECENT, EvidenceBucket.SUPPORTING},
            missing_buckets={EvidenceBucket.CONTRADICTING},
            covered_topics=["force"],
            missing_topics=["context"],
            sufficient=True,
            rationale="Enough direct evidence.",
        ),
        failed_sources=["arxiv"],
        iterations=2,
        stop_reason=StopReason.COVERAGE_SATISFIED,
        errors=["arxiv timeout"],
        source_result_counts={"pubmed": 1},
        stage_elapsed_seconds={"source_search": 0.2},
    )

    run = build_m2_knowledge_export_run("question", search, [reading])

    assert run.sub_question == "question"
    assert run.papers[0].model_dump(mode="python") == {
        "paper_id": "PMID:1",
        "title": "Force and YAP",
        "abstract": "Force changes YAP transport.",
        "authors": ["A. Author"],
        "year": 2025,
        "journal": "Mechanobiology",
        "doi": "10.1/example",
        "pmid": "1",
        "pmcid": "PMC1",
        "external_ids": {"arxiv": "1234.5678"},
        "citation_count": 12,
        "publication_type": "article",
        "sources": ["pubmed"],
        "is_open_access": True,
        "fulltext_status": "xml_available",
        "rank_scores": {"relevance": 0.9},
        "reading_summary": "Force promotes transport.",
        "content_level": "structured_fulltext",
        "document_id": "doc-1",
        "document_source_uri": "https://example.test/fulltext/1",
        "document_license": "CC BY 4.0",
        "degraded_to_abstract": False,
        "chunks_parsed": 8,
        "chunks_retrieved": 1,
        "stage_elapsed_seconds": {"parse": 0.1},
        "errors": ["minor extraction warning"],
    }
    assert run.evidence[0].model_dump(mode="python") == evidence.model_dump(mode="python")
    assert run.knowledge_entries[0].model_dump(mode="python") == {
        "id": "knowledge-1",
        "type": KnowledgeEntryType.MECHANISTIC_CONCLUSION,
        "content": "Force changes YAP transport.",
        "confidence": ConfidenceLevel.HIGH,
        "source_paper_id": "PMID:1",
        "source_paper_title": "Force and YAP",
        "entities": ["YAP"],
        "evidence_ids": ["ev-1"],
    }
    assert run.search_provenance.model_dump(mode="python") == {
        "queries": [
            {
                "query_id": "q-1",
                "text": "YAP force",
                "round_index": 1,
                "intent": "mesh",
                "target_source": "pubmed",
                "purpose": "mechanism",
                "target_gap": "supporting",
                "relation_to_question": "Direct mechanism evidence.",
            }
        ],
        "coverage": {
            "covered_buckets": ["recent", "supporting"],
            "missing_buckets": ["contradicting"],
            "covered_topics": ["force"],
            "missing_topics": ["context"],
            "sufficient": True,
            "rationale": "Enough direct evidence.",
        },
        "source_result_counts": {"pubmed": 1},
        "failed_sources": ["arxiv"],
        "iterations": 2,
        "stop_reason": "coverage_satisfied",
        "errors": ["arxiv timeout"],
        "stage_elapsed_seconds": {"source_search": 0.2},
        "papers_found": 3,
        "papers_after_dedup": 2,
    }


def test_builder_preserves_final_k_evidence_and_knowledge_order() -> None:
    first = make_paper("PMID:1", "First")
    second = make_paper("PMID:2", "Second")
    first_reading = PaperReadingResult(
        paper_id="PMID:1",
        evidence=[
            EvidenceChunk(
                evidence_id="ev-1b",
                paper_id="PMID:1",
                chunk_id="chunk-1b",
                quote="Second evidence.",
                normalized_claim="Second claim.",
                relevance_score=0.5,
            ),
            EvidenceChunk(
                evidence_id="ev-1a",
                paper_id="PMID:1",
                chunk_id="chunk-1a",
                quote="First evidence.",
                normalized_claim="First claim.",
                relevance_score=0.6,
            ),
        ],
        knowledge_entries=[
            EvidenceLinkedKnowledge(
                entry_id="knowledge-1b",
                entry_type=KnowledgeEntryType.MECHANISTIC_CONCLUSION,
                content="Second knowledge.",
                evidence_ids=["ev-1b"],
            ),
            EvidenceLinkedKnowledge(
                entry_id="knowledge-1a",
                entry_type=KnowledgeEntryType.MECHANISTIC_CONCLUSION,
                content="First knowledge.",
                evidence_ids=["ev-1a"],
            ),
        ],
    )
    second_reading = make_reading("PMID:2")
    search = SearchRunResult(sub_question="question", final_papers=[first, second])

    run = build_m2_knowledge_export_run(
        "question", search, [second_reading, first_reading]
    )

    assert [item.paper_id for item in run.papers] == ["PMID:1", "PMID:2"]
    assert [item.evidence_id for item in run.evidence] == ["ev-1b", "ev-1a"]
    assert [item.id for item in run.knowledge_entries] == [
        "knowledge-1b",
        "knowledge-1a",
    ]


def test_builder_exports_a_zero_paper_run() -> None:
    search = SearchRunResult(sub_question="question")

    run = build_m2_knowledge_export_run("question", search, [])

    assert run.sub_question == "question"
    assert run.papers == []
    assert run.evidence == []
    assert run.knowledge_entries == []
    assert run.search_provenance == run.search_provenance.model_construct()


def test_builder_sanitizes_portability_and_secret_leaks_from_exported_errors() -> None:
    paper = make_paper("PMID:1", "First")
    reading = PaperReadingResult(
        paper_id="PMID:1",
        errors=[
            "cache C:\\Users\\alice\\.cache\\paper.pdf failed; "
            "Authorization: Bearer very-secret-token; "
            "request https://example.test/api?api_key=key-secret&token=token-secret; "
            "OpenAI keys sk-abcdefghijklmnop and sk-proj-qrstuvwxyzabcdef; "
            "UNC \\\\server\\restricted\\paper.pdf; "
            "local file:///home/alice/.cache/fulltext.json; "
            'quoted API api_key="double-secret" failed; '
            "quoted token token='single-secret' failed; "
            'Windows file "C:\\Users\\Alice\\Private Project\\paper.pdf" failed; '
            "POSIX file '/home/alice/Private Project/paper.pdf' failed"
        ],
    )
    search = SearchRunResult(
        sub_question="question",
        final_papers=[paper],
        errors=[
            "download /home/alice/.cache/fulltext.json failed; "
            "access_token=access-secret signature=sig-secret; retry later"
        ],
    )

    run = build_m2_knowledge_export_run("question", search, [reading])
    exported_errors = [
        *run.papers[0].errors,
        *run.search_provenance.errors,
    ]
    combined = "\n".join(exported_errors)

    for secret in [
        "C:\\Users\\alice",
        "/home/alice",
        "very-secret-token",
        "key-secret",
        "token-secret",
        "access-secret",
        "sig-secret",
        "sk-abcdefghijklmnop",
        "sk-proj-qrstuvwxyzabcdef",
        "\\\\server\\restricted",
        "/home/alice/.cache/fulltext.json",
        "double-secret",
        "single-secret",
        "C:\\Users\\Alice\\Private Project\\paper.pdf",
        "/home/alice/Private Project/paper.pdf",
    ]:
        assert secret not in combined
    assert "cache" in combined
    assert "download" in combined
    assert "retry later" in combined
    assert "quoted API api_key=[REDACTED] failed" in combined
    assert "quoted token token=[REDACTED] failed" in combined
    assert 'Windows file "[REDACTED_PATH]" failed' in combined
    assert "POSIX file '[REDACTED_PATH]' failed" in combined


def test_builder_truncates_a_useful_message_within_the_exported_error_limit() -> None:
    paper = make_paper("PMID:1", "First")
    useful_prefix = "full-text resolver failed after retrying PMC endpoint: "
    reading = PaperReadingResult(
        paper_id="PMID:1",
        errors=[useful_prefix + ("x" * 2000)],
    )
    search = SearchRunResult(sub_question="question", final_papers=[paper])

    run = build_m2_knowledge_export_run("question", search, [reading])

    exported = run.papers[0].errors[0]
    assert len(exported) <= 500
    assert exported.startswith(useful_prefix)
    assert exported.endswith("...")


def test_builder_redacts_unquoted_absolute_paths_with_spaces_without_losing_context() -> None:
    paper = make_paper("PMID:1", "First")
    reading = PaperReadingResult(
        paper_id="PMID:1",
        errors=[
            "cache C:\\Users\\Alice\\Private Project\\paper.pdf failed",
            "cache /home/alice/Private Project/paper.pdf failed",
            "cache C:\\Users\\Alice\\Private Project\\papers failed",
            "cache /home/alice/Private Project/papers failed",
        ],
    )
    search = SearchRunResult(sub_question="question", final_papers=[paper])

    run = build_m2_knowledge_export_run("question", search, [reading])

    assert run.papers[0].errors == [
        "cache [REDACTED_PATH] failed",
        "cache [REDACTED_PATH] failed",
        "cache [REDACTED_PATH] failed",
        "cache [REDACTED_PATH] failed",
    ]


def test_builder_bounds_exported_error_list_sizes() -> None:
    paper = make_paper("PMID:1", "First")
    reading = PaperReadingResult(
        paper_id="PMID:1",
        errors=[f"reading warning {index}" for index in range(30)],
    )
    search = SearchRunResult(
        sub_question="question",
        final_papers=[paper],
        errors=[f"search warning {index}" for index in range(30)],
    )

    run = build_m2_knowledge_export_run("question", search, [reading])

    assert len(run.papers[0].errors) <= 20
    assert len(run.search_provenance.errors) <= 20
    assert all(len(item) <= 500 for item in run.papers[0].errors)


@pytest.mark.parametrize(
    ("readings", "paper_id", "message"),
    [
        (
            [make_reading("PMID:1"), make_reading("PMID:1")],
            "PMID:1",
            "duplicate reading result",
        ),
        ([], "PMID:1", "missing reading result"),
        ([make_reading("PMID:2")], "PMID:2", "unknown paper"),
    ],
)
def test_builder_rejects_invalid_reading_results(
    readings: list[PaperReadingResult], paper_id: str, message: str
) -> None:
    search = SearchRunResult(
        sub_question="question",
        final_papers=[make_paper("PMID:1", "First")],
    )

    with pytest.raises(ValueError, match=f"{message}.*{paper_id}"):
        build_m2_knowledge_export_run("question", search, readings)


def test_builder_rejects_reading_evidence_from_a_different_paper() -> None:
    search = SearchRunResult(
        sub_question="question",
        final_papers=[make_paper("PMID:1", "First")],
    )
    reading = PaperReadingResult(
        paper_id="PMID:1",
        evidence=[
            EvidenceChunk(
                evidence_id="ev-other",
                paper_id="PMID:2",
                chunk_id="chunk-other",
                quote="Misattributed evidence.",
                normalized_claim="Misattributed evidence.",
                relevance_score=0.8,
            )
        ],
    )

    with pytest.raises(
        ValueError,
        match="reading result for paper ID: PMID:1 contains evidence.*PMID:2",
    ):
        build_m2_knowledge_export_run("question", search, [reading])
