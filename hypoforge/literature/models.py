"""Shared data contracts for agentic literature search and reading.

These models contain metadata and references, not complete paper text.  Large
documents live in an external document store and are addressed by stable IDs.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Dict, List, Optional, Set

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

from ..state import ConfidenceLevel, KnowledgeEntryType


NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class LiteratureModel(BaseModel):
    """Strict base model used by every M2 public contract."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class QueryIntent(str, Enum):
    CORE = "core"
    SYNONYM = "synonym"
    MESH = "mesh"
    CONTRADICTORY = "contradictory"
    RECENT = "recent"
    REVIEW = "review"
    CITATION = "citation"


class ContentLevel(str, Enum):
    METADATA = "metadata"
    ABSTRACT = "abstract"
    STRUCTURED_FULLTEXT = "structured_fulltext"
    HTML = "html"
    PDF = "pdf"
    OCR = "ocr"


class FulltextStatus(str, Enum):
    UNKNOWN = "unknown"
    UNAVAILABLE = "unavailable"
    ABSTRACT_ONLY = "abstract_only"
    XML_AVAILABLE = "xml_available"
    HTML_AVAILABLE = "html_available"
    PDF_AVAILABLE = "pdf_available"
    DOWNLOADED = "downloaded"
    FAILED = "failed"


class EvidenceBucket(str, Enum):
    SUPPORTING = "supporting"
    CONTRADICTING = "contradicting"
    REVIEW = "review"
    RECENT = "recent"
    CLASSIC = "classic"
    METHODOLOGICAL = "methodological"


class StopReason(str, Enum):
    COVERAGE_SATISFIED = "coverage_satisfied"
    MAX_ROUNDS = "max_rounds"
    QUERY_BUDGET = "query_budget"
    PAPER_BUDGET = "paper_budget"
    TOKEN_BUDGET = "token_budget"
    TIME_BUDGET = "time_budget"
    LOW_MARGINAL_GAIN = "low_marginal_gain"
    NO_RESULTS = "no_results"
    ERROR = "error"


class SearchQuery(LiteratureModel):
    query_id: NonEmptyStr
    text: NonEmptyStr
    round_index: int = Field(default=0, ge=0)
    intent: QueryIntent = QueryIntent.CORE
    target_source: NonEmptyStr
    purpose: NonEmptyStr
    target_gap: str = ""
    relation_to_question: NonEmptyStr


class PaperRecord(LiteratureModel):
    paper_id: NonEmptyStr
    title: NonEmptyStr
    abstract: str = ""
    authors: List[str] = Field(default_factory=list)
    year: Optional[int] = Field(default=None, ge=1000, le=3000)
    journal: str = ""
    doi: str = ""
    pmid: str = ""
    pmcid: str = ""
    external_ids: Dict[str, str] = Field(default_factory=dict)
    citation_count: Optional[int] = Field(default=None, ge=0)
    publication_type: str = ""
    sources: List[NonEmptyStr] = Field(min_length=1)
    is_open_access: Optional[bool] = None
    fulltext_status: FulltextStatus = FulltextStatus.UNKNOWN
    rank_scores: Dict[str, float] = Field(default_factory=dict)

    @field_validator("doi", mode="before")
    @classmethod
    def normalize_doi(cls, value: object) -> str:
        doi = str(value or "").strip().lower()
        for prefix in ("https://doi.org/", "http://doi.org/", "doi:", "doi "):
            if doi.startswith(prefix):
                doi = doi[len(prefix) :]
                break
        return doi.strip()


class ScoutNote(LiteratureModel):
    paper_id: NonEmptyStr
    main_topic: str = ""
    key_terms: List[str] = Field(default_factory=list)
    entities: List[str] = Field(default_factory=list)
    mechanisms: List[str] = Field(default_factory=list)
    important_authors: List[str] = Field(default_factory=list)
    controversies: List[str] = Field(default_factory=list)
    candidate_citations: List[str] = Field(default_factory=list)
    relevance_to_question: float = Field(default=0.0, ge=0.0, le=1.0)
    # ``None`` preserves compatibility with notes produced before directness
    # screening was introduced. New ``ScoutReader`` results always populate it.
    directness_to_question: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    # These are exact, question-anchored abstract sentences. They make a
    # directional bucket auditable without changing the ScoutReader Protocol.
    supporting_evidence: List[str] = Field(default_factory=list)
    contradicting_evidence: List[str] = Field(default_factory=list)
    evidence_buckets: Set[EvidenceBucket] = Field(default_factory=set)
    study_design: str = ""
    evidence_summary: str = ""


class CoverageReport(LiteratureModel):
    covered_buckets: Set[EvidenceBucket] = Field(default_factory=set)
    missing_buckets: Set[EvidenceBucket] = Field(default_factory=set)
    covered_topics: List[str] = Field(default_factory=list)
    missing_topics: List[str] = Field(default_factory=list)
    sufficient: bool = False
    rationale: str = ""


class SearchBudget(LiteratureModel):
    max_rounds: int = Field(default=3, ge=1)
    max_queries: int = Field(default=12, ge=1)
    max_papers: int = Field(default=100, ge=1)
    max_tokens: int = Field(default=100_000, ge=1)
    max_seconds: int = Field(default=900, ge=1)


class RemainingSearchBudget(LiteratureModel):
    max_rounds: int = Field(default=0, ge=0)
    max_queries: int = Field(default=0, ge=0)
    max_papers: int = Field(default=0, ge=0)
    max_tokens: int = Field(default=0, ge=0)
    max_seconds: float = Field(default=0.0, ge=0.0)


class SearchState(LiteratureModel):
    question_type: str = ""
    key_entities: Set[str] = Field(default_factory=set)
    domains: Set[str] = Field(default_factory=set)
    round_index: int = Field(default=0, ge=0)
    queries_used: List[SearchQuery] = Field(default_factory=list)
    known_terms: Set[str] = Field(default_factory=set)
    covered_topics: Set[str] = Field(default_factory=set)
    missing_topics: Set[str] = Field(default_factory=set)
    unavailable_sources: Set[str] = Field(default_factory=set)
    candidate_paper_ids: List[str] = Field(default_factory=list)
    bucket_counts: Dict[EvidenceBucket, int] = Field(default_factory=dict)
    remaining_budget: RemainingSearchBudget = Field(
        default_factory=RemainingSearchBudget
    )
    queries_executed: int = Field(default=0, ge=0)
    unique_papers_seen: int = Field(default=0, ge=0)
    estimated_tokens_used: int = Field(default=0, ge=0)
    elapsed_seconds: float = Field(default=0.0, ge=0.0)
    consecutive_low_gain_rounds: int = Field(default=0, ge=0)
    consecutive_no_result_rounds: int = Field(default=0, ge=0)


class SearchRunResult(LiteratureModel):
    sub_question: NonEmptyStr
    queries: List[SearchQuery] = Field(default_factory=list)
    papers_found: int = Field(default=0, ge=0)
    papers_after_dedup: int = Field(default=0, ge=0)
    candidates: List[PaperRecord] = Field(default_factory=list)
    final_papers: List[PaperRecord] = Field(default_factory=list)
    coverage: CoverageReport = Field(default_factory=CoverageReport)
    failed_sources: List[str] = Field(default_factory=list)
    iterations: int = Field(default=0, ge=0)
    stop_reason: Optional[StopReason] = None
    errors: List[str] = Field(default_factory=list)
    source_result_counts: Dict[str, int] = Field(default_factory=dict)
    stage_elapsed_seconds: Dict[str, float] = Field(default_factory=dict)
    scout_notes: List[ScoutNote] = Field(default_factory=list)
    reused_paper_ids: List[str] = Field(default_factory=list)
    final_state: Optional[SearchState] = None


class DocumentRecord(LiteratureModel):
    document_id: NonEmptyStr
    paper_id: NonEmptyStr
    content_level: ContentLevel
    source_uri: str = ""
    local_path: str = ""
    license: str = ""
    retrieval_error: str = ""


class DocumentChunk(LiteratureModel):
    chunk_id: NonEmptyStr
    document_id: NonEmptyStr
    paper_id: NonEmptyStr
    section: str = ""
    page: Optional[int] = Field(default=None, ge=1)
    start_offset: Optional[int] = Field(default=None, ge=0)
    end_offset: Optional[int] = Field(default=None, ge=0)
    text: NonEmptyStr


class EvidenceChunk(LiteratureModel):
    evidence_id: NonEmptyStr
    paper_id: NonEmptyStr
    chunk_id: NonEmptyStr
    section: str = ""
    page: Optional[int] = Field(default=None, ge=1)
    quote: NonEmptyStr
    normalized_claim: NonEmptyStr
    relevance_score: float = Field(ge=0.0, le=1.0)
    citable: bool = True


class EvidenceLinkedKnowledge(LiteratureModel):
    entry_id: NonEmptyStr
    entry_type: KnowledgeEntryType
    content: NonEmptyStr
    confidence: Optional[ConfidenceLevel] = None
    entities: List[str] = Field(default_factory=list)
    evidence_ids: List[NonEmptyStr] = Field(min_length=1)


class PaperReadingResult(LiteratureModel):
    paper_id: NonEmptyStr
    summary: str = ""
    evidence: List[EvidenceChunk] = Field(default_factory=list)
    knowledge_entries: List[EvidenceLinkedKnowledge] = Field(default_factory=list)
    content_level: ContentLevel = ContentLevel.METADATA
    document_id: str = ""
    document_source_uri: str = ""
    document_license: str = ""
    chunks_parsed: int = Field(default=0, ge=0)
    chunks_retrieved: int = Field(default=0, ge=0)
    stage_elapsed_seconds: Dict[str, float] = Field(default_factory=dict)
    degraded_to_abstract: bool = False
    errors: List[str] = Field(default_factory=list)
