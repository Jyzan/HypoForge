"""
Pipeline state definitions for HypoForge.

All data that flows through the M1→M2→M3→M4→M5→M6 pipeline lives in
``PipelineState`` (a Pydantic BaseModel).  Each module reads from and
writes to this shared state object.

The sub-models below correspond to the structured outputs defined in the
competition specification (§三).
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ============================================================================
# Enums
# ============================================================================

class QuestionType(str, Enum):
    MECHANISM = "mechanism_explanation"
    METHOD = "method_development"
    DISCOVERY = "phenomenon_discovery"


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class KnowledgeEntryType(str, Enum):
    ESTABLISHED_FACT = "established_fact"
    MECHANISTIC_CONCLUSION = "mechanistic_conclusion"
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    METHOD = "method"
    KNOWLEDGE_GAP = "knowledge_gap"
    KEY_ENTITY = "key_entity"


class EvidenceNodeType(str, Enum):
    CLAIM = "claim"
    EVIDENCE = "evidence"
    SOURCE = "source"
    LIMITATION = "limitation"
    CONFLICT = "conflict"
    ENTITY = "entity"


class EvidenceEdgeRelation(str, Enum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    EXTENDS = "extends"
    LIMITS = "limits"
    INVOLVES = "involves"


class ReviewerDimension(str, Enum):
    SCIENTIFIC_LOGIC = "scientific_logic"
    EVIDENCE_CONSISTENCY = "evidence_consistency"
    METHOD_FEASIBILITY = "method_feasibility"
    OVERALL = "overall"


# ============================================================================
# M1 产出 — ProblemCard
# ============================================================================

class ProblemCard(BaseModel):
    """Structured decomposition of a scientific question (M1 output)."""

    original_question: str
    domain: List[str] = Field(default_factory=list)
    sub_questions: List[str] = Field(default_factory=list)
    key_entities: List[str] = Field(default_factory=list)
    question_type: QuestionType = QuestionType.MECHANISM

    @field_validator("original_question", mode="before")
    @classmethod
    def normalize_original_question(cls, value: Any) -> str:
        """Keep model-generated line breaks from leaking into terminal/UI output."""
        return re.sub(r"\s+", " ", str(value or "")).strip()

    @field_validator("domain", "sub_questions", "key_entities", mode="before")
    @classmethod
    def normalize_text_lists(cls, value: Any) -> List[str]:
        """Normalize whitespace in list fields while preserving item boundaries."""
        if value is None:
            return []
        return [re.sub(r"\s+", " ", str(item or "")).strip() for item in value]


# ============================================================================
# M2 产出 — Knowledge Entries
# ============================================================================

class KnowledgeEntry(BaseModel):
    """One extracted piece of knowledge from the literature."""

    id: str
    type: KnowledgeEntryType
    content: str
    confidence: Optional[ConfidenceLevel] = None
    source_paper_id: str = ""
    source_paper_title: str = ""
    entities: List[str] = Field(default_factory=list)
    evidence_ids: List[str] = Field(default_factory=list)


class LiteratureResult(BaseModel):
    """Aggregated results for one sub-question (M2 output)."""

    sub_question: str
    papers_retrieved: int = 0
    knowledge_entries: List[KnowledgeEntry] = Field(default_factory=list)


class M2SearchQueryExport(BaseModel):
    query_id: str
    text: str
    round_index: int = 0
    intent: str = "core"
    target_source: str
    purpose: str
    target_gap: str = ""
    relation_to_question: str = ""


class M2CoverageExport(BaseModel):
    covered_buckets: List[str] = Field(default_factory=list)
    missing_buckets: List[str] = Field(default_factory=list)
    covered_topics: List[str] = Field(default_factory=list)
    missing_topics: List[str] = Field(default_factory=list)
    sufficient: bool = False
    rationale: str = ""


class M2SearchProvenance(BaseModel):
    queries: List[M2SearchQueryExport] = Field(default_factory=list)
    coverage: M2CoverageExport = Field(default_factory=M2CoverageExport)
    source_result_counts: Dict[str, int] = Field(default_factory=dict)
    failed_sources: List[str] = Field(default_factory=list)
    iterations: int = 0
    stop_reason: Optional[str] = None
    errors: List[str] = Field(default_factory=list)
    stage_elapsed_seconds: Dict[str, float] = Field(default_factory=dict)
    papers_found: int = 0
    papers_after_dedup: int = 0


class M2PaperExport(BaseModel):
    paper_id: str
    title: str
    abstract: str = ""
    authors: List[str] = Field(default_factory=list)
    year: Optional[int] = None
    journal: str = ""
    doi: str = ""
    pmid: str = ""
    pmcid: str = ""
    external_ids: Dict[str, str] = Field(default_factory=dict)
    citation_count: Optional[int] = None
    publication_type: str = ""
    sources: List[str] = Field(default_factory=list)
    is_open_access: Optional[bool] = None
    fulltext_status: str = "unknown"
    rank_scores: Dict[str, float] = Field(default_factory=dict)
    reading_summary: str = ""
    content_level: str = "metadata"
    document_id: str = ""
    document_source_uri: str = ""
    document_license: str = ""
    degraded_to_abstract: bool = False
    chunks_parsed: int = 0
    chunks_retrieved: int = 0
    stage_elapsed_seconds: Dict[str, float] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)


class M2EvidenceExport(BaseModel):
    evidence_id: str
    paper_id: str
    chunk_id: str
    section: str = ""
    page: Optional[int] = None
    quote: str
    normalized_claim: str
    relevance_score: float
    citable: bool = True


class M2KnowledgeRun(BaseModel):
    sub_question: str
    papers: List[M2PaperExport] = Field(default_factory=list)
    evidence: List[M2EvidenceExport] = Field(default_factory=list)
    knowledge_entries: List[KnowledgeEntry] = Field(default_factory=list)
    search_provenance: M2SearchProvenance = Field(
        default_factory=M2SearchProvenance
    )

    @model_validator(mode="after")
    def validate_provenance(self) -> "M2KnowledgeRun":
        paper_ids = [item.paper_id for item in self.papers]
        if len(paper_ids) != len(set(paper_ids)):
            raise ValueError("duplicate paper_id in M2 knowledge run")

        evidence_ids = [item.evidence_id for item in self.evidence]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("duplicate evidence_id in M2 knowledge run")

        knowledge_ids = [item.id for item in self.knowledge_entries]
        if len(knowledge_ids) != len(set(knowledge_ids)):
            raise ValueError("duplicate knowledge id in M2 knowledge run")

        paper_set = set(paper_ids)
        evidence_by_id = {item.evidence_id: item for item in self.evidence}
        for item in self.evidence:
            if item.paper_id not in paper_set:
                raise ValueError(
                    f"evidence {item.evidence_id!r} references unknown paper"
                )
        for item in self.knowledge_entries:
            if item.source_paper_id not in paper_set:
                raise ValueError(
                    f"knowledge {item.id!r} references unknown paper"
                )
            if not item.evidence_ids:
                raise ValueError(f"knowledge {item.id!r} has no evidence_ids")
            unknown = [
                evidence_id
                for evidence_id in item.evidence_ids
                if evidence_id not in evidence_by_id
            ]
            if unknown:
                raise ValueError(
                    f"knowledge {item.id!r} references unknown evidence: {unknown}"
                )
            cross_paper = [
                evidence_id
                for evidence_id in item.evidence_ids
                if evidence_by_id[evidence_id].paper_id != item.source_paper_id
            ]
            if cross_paper:
                evidence_id = cross_paper[0]
                evidence_paper_id = evidence_by_id[evidence_id].paper_id
                raise ValueError(
                    f"knowledge {item.id!r} references evidence {evidence_id!r} "
                    f"from paper {evidence_paper_id!r}, expected source paper "
                    f"{item.source_paper_id!r}"
                )
        return self


class M2KnowledgeExport(BaseModel):
    schema_version: Literal["m2-knowledge-export/v1"] = (
        "m2-knowledge-export/v1"
    )
    runs: List[M2KnowledgeRun] = Field(default_factory=list)


# ============================================================================
# M3 产出 — Evidence Graph
# ============================================================================

class EvidenceNode(BaseModel):
    """A single node in the evidence graph."""

    id: str
    type: EvidenceNodeType
    label: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EvidenceEdge(BaseModel):
    """A directed, typed edge in the evidence graph."""

    source: str
    target: str
    relation: EvidenceEdgeRelation


class EvidenceGraph(BaseModel):
    """Full evidence graph (M3 output)."""

    nodes: List[EvidenceNode] = Field(default_factory=list)
    edges: List[EvidenceEdge] = Field(default_factory=list)

    # Convenience lists of KnowledgeEntry ids, populated by M3
    established_facts: List[str] = Field(default_factory=list)
    conflicts: List[str] = Field(default_factory=list)
    knowledge_gaps: List[str] = Field(default_factory=list)


# ============================================================================
# M4 产出 — Hypothesis Cards
# ============================================================================

class HypothesisCard(BaseModel):
    """A single candidate hypothesis (M4 output)."""

    hypothesis_id: str
    statement: str
    mechanism: str = ""
    observable_predictions: List[str] = Field(default_factory=list)
    falsification_conditions: List[str] = Field(default_factory=list)
    supporting_evidence: List[str] = Field(default_factory=list)

    # Multi-dimensional scores (populated by Ranker in M4)
    scores: Dict[str, float] = Field(default_factory=dict)


# ============================================================================
# M5 产出 — Research Plan
# ============================================================================

class ResearchPlan(BaseModel):
    """A detailed research plan for one hypothesis (M5 output)."""

    hypothesis_id: str = ""
    study_subjects: str = ""
    independent_variables: List[str] = Field(default_factory=list)
    dependent_variables: List[str] = Field(default_factory=list)
    control_groups: List[str] = Field(default_factory=list)
    procedures: List[str] = Field(default_factory=list)
    measurement_metrics: List[str] = Field(default_factory=list)
    analysis_methods: List[str] = Field(default_factory=list)
    expected_results_if_supported: str = ""
    expected_results_if_refuted: str = ""
    timeline: str = ""
    risks_and_alternatives: str = ""


# ============================================================================
# M6 产出 — Review Results
# ============================================================================

class ReviewResult(BaseModel):
    """A single reviewer's assessment (M6 output)."""

    dimension: ReviewerDimension
    score: float  # 1.0 – 5.0
    comments: str = ""
    suggestions: str = ""
    version: int = 1


# ============================================================================
# Global Pipeline State
# ============================================================================

class PipelineState(BaseModel):
    """
    The single source of truth that flows through the entire pipeline.

    LangGraph treats this as the graph's state schema.  Each node returns
    a ``dict[str, Any]`` that is *merged* back into this model (additive /
    overwrite semantics depend on the field type).
    """

    # ---- Input ----
    input_question: str = ""

    # ---- M1 ----
    problem_card: Optional[ProblemCard] = None

    # ---- M2 ----
    literature_results: List[LiteratureResult] = Field(default_factory=list)
    m2_knowledge_export: Optional[M2KnowledgeExport] = None

    # ---- M3 ----
    evidence_graph: Optional[EvidenceGraph] = None

    # ---- M4 ----
    candidate_hypotheses: List[HypothesisCard] = Field(default_factory=list)
    top_hypotheses: List[HypothesisCard] = Field(default_factory=list)

    # ---- M5 ----
    research_plans: List[ResearchPlan] = Field(default_factory=list)

    # ---- M6 + iteration ----
    reviews: List[ReviewResult] = Field(default_factory=list)
    iteration_count: int = 0
    max_iterations: int = 3

    # ---- persistence ----
    memory_cache_dir: str = ""  # non-empty enables persistent knowledge graph

    # ---- metadata ----
    run_id: str = ""
    errors: List[str] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)

    # ---- token tracking (populated by LLM-calling modules) ----
    total_input_tokens: int = 0
    total_output_tokens: int = 0
