from __future__ import annotations

import pytest
from pydantic import ValidationError

from hypoforge.literature.models import (
    ContentLevel,
    CoverageReport,
    DocumentChunk,
    EvidenceBucket,
    EvidenceChunk,
    FulltextStatus,
    PaperRecord,
    QueryIntent,
    RemainingSearchBudget,
    SearchBudget,
    SearchQuery,
    SearchRunResult,
    SearchState,
    StopReason,
)


def test_search_query_strips_text_and_keeps_search_intent() -> None:
    query = SearchQuery(
        query_id="q-1",
        text="  Hsp70 AND proteostasis  ",
        intent=QueryIntent.CORE,
        target_source="pubmed",
        purpose="Find direct mechanistic evidence",
        relation_to_question="Directly tests the proposed mechanism",
    )

    assert query.text == "Hsp70 AND proteostasis"
    assert query.intent is QueryIntent.CORE


def test_paper_record_normalizes_doi_and_rejects_negative_citations() -> None:
    paper = PaperRecord(
        paper_id="paper-1",
        title="A study of Hsp70",
        doi="https://doi.org/10.1000/ABC.1",
        sources=["pubmed"],
        citation_count=12,
        fulltext_status=FulltextStatus.XML_AVAILABLE,
    )

    assert paper.doi == "10.1000/abc.1"

    with pytest.raises(ValidationError):
        PaperRecord(
            paper_id="paper-2",
            title="Invalid citations",
            sources=["pubmed"],
            citation_count=-1,
        )


def test_search_budget_requires_positive_limits() -> None:
    with pytest.raises(ValidationError):
        SearchBudget(max_rounds=0)


def test_remaining_budget_allows_exhausted_dimensions() -> None:
    remaining = RemainingSearchBudget(
        max_rounds=0,
        max_queries=0,
        max_papers=0,
        max_tokens=0,
        max_seconds=0,
    )

    assert remaining.max_queries == 0


def test_search_state_tracks_iterative_usage() -> None:
    state = SearchState(
        queries_executed=2,
        unique_papers_seen=7,
        estimated_tokens_used=120,
        elapsed_seconds=1.5,
        consecutive_low_gain_rounds=1,
        consecutive_no_result_rounds=0,
    )

    assert state.queries_executed == 2
    assert state.elapsed_seconds == 1.5


def test_document_and_evidence_chunks_require_non_empty_source_text() -> None:
    with pytest.raises(ValidationError):
        DocumentChunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            paper_id="paper-1",
            section="Results",
            text="   ",
        )

    evidence = EvidenceChunk(
        evidence_id="evidence-1",
        paper_id="paper-1",
        chunk_id="chunk-1",
        section="Results",
        quote="Hsp70 activity increased after treatment.",
        normalized_claim="Treatment increases Hsp70 activity.",
        relevance_score=0.91,
    )

    assert evidence.quote.startswith("Hsp70")


def test_search_run_result_round_trips_with_coverage_and_stop_reason() -> None:
    coverage = CoverageReport(
        covered_buckets={EvidenceBucket.SUPPORTING, EvidenceBucket.RECENT},
        missing_buckets={EvidenceBucket.CONTRADICTING},
        missing_topics=["negative results"],
        sufficient=False,
        rationale="No contradictory study has been found.",
    )
    result = SearchRunResult(
        sub_question="Does NAD+ regulate Hsp70 activity?",
        papers_found=20,
        papers_after_dedup=15,
        coverage=coverage,
        iterations=2,
        stop_reason=StopReason.MAX_ROUNDS,
    )

    restored = SearchRunResult.model_validate_json(result.model_dump_json())

    assert restored.coverage.missing_buckets == {EvidenceBucket.CONTRADICTING}
    assert restored.stop_reason is StopReason.MAX_ROUNDS


def test_search_result_carries_agent_trace() -> None:
    result = SearchRunResult(
        sub_question="Does NAD+ regulate Hsp70 activity?",
        source_result_counts={"pubmed": 3},
        reused_paper_ids=["paper-1"],
        final_state=SearchState(),
    )

    assert result.source_result_counts == {"pubmed": 3}
    assert result.reused_paper_ids == ["paper-1"]
    assert result.final_state is not None


def test_document_record_tracks_content_level_without_storing_full_text() -> None:
    from hypoforge.literature.models import DocumentRecord

    document = DocumentRecord(
        document_id="doc-1",
        paper_id="paper-1",
        content_level=ContentLevel.STRUCTURED_FULLTEXT,
        source_uri="https://example.org/article.xml",
        local_path="documents/paper-1.xml",
        license="CC-BY-4.0",
    )

    assert document.content_level is ContentLevel.STRUCTURED_FULLTEXT
    assert not hasattr(document, "full_text")


def test_literature_package_exposes_shared_contracts() -> None:
    from hypoforge import literature

    assert literature.PaperRecord is PaperRecord
    assert literature.SearchRunResult is SearchRunResult
