"""
Pipeline state definitions for HypoForge.

All data that flows through the M1→M2→M3→M4→M5→M6 pipeline lives in
``PipelineState`` (a Pydantic BaseModel).  Each module reads from and
writes to this shared state object.

The sub-models below correspond to the structured outputs defined in the
competition specification (§三).
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


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


class LiteratureResult(BaseModel):
    """Aggregated results for one sub-question (M2 output)."""

    sub_question: str
    papers_retrieved: int = 0
    knowledge_entries: List[KnowledgeEntry] = Field(default_factory=list)


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

    # ---- metadata ----
    run_id: str = ""
    errors: List[str] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)

    # ---- token tracking (populated by LLM-calling modules) ----
    total_input_tokens: int = 0
    total_output_tokens: int = 0
