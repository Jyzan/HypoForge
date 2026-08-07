"""
Pipeline state definitions for HypoForge.

All data that flows through the M1→M2→M3→M4→M5→M6 pipeline lives in
``PipelineState`` (a Pydantic BaseModel).  Each module reads from and
writes to this shared state object.

The sub-models below correspond to the structured outputs defined in the
competition specification (§三).
"""

from __future__ import annotations

import hashlib
import re
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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
    SAME_AS = "same_as"
    REFINES = "refines"


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
    """A directed, typed edge in the evidence graph.

    .. versionchanged:: 0.2.0
        Added optional ``confidence``, ``rationale``, and ``evidence_ids``
        fields to support M3 grounding's enhanced relation output.
    """

    source: str
    target: str
    relation: EvidenceEdgeRelation

    # ---- M3 grounding enhanced fields (optional) ----
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    rationale: Optional[str] = None
    evidence_ids: List[str] = Field(default_factory=list)


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

    # Reasoning written by the Ranker *before* the numeric scores (reason-before-score).
    ranking_rationale: str = ""

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
    reasoning: str = ""  # written *before* the score (reason-before-score)
    score: float  # 1.0 – 5.0
    comments: str = ""
    suggestions: str = ""
    version: int = 1


# ============================================================================
# M2 增强导出 — canonical KnowledgeExport (agentic M2 → M3)
# ============================================================================

class M2SearchQueryExport(BaseModel):
    """One search query issued by the IterativeSearchAgent."""

    query_id: str
    text: str
    round_index: int = 0
    intent: str = "core"
    target_source: str
    purpose: str
    target_gap: str = ""
    relation_to_question: str = ""


class M2CoverageExport(BaseModel):
    """Coverage assessment from the CoverageEvaluator."""

    covered_buckets: List[str] = Field(default_factory=list)
    missing_buckets: List[str] = Field(default_factory=list)
    covered_topics: List[str] = Field(default_factory=list)
    missing_topics: List[str] = Field(default_factory=list)
    sufficient: bool = False
    rationale: str = ""


class M2SearchProvenance(BaseModel):
    """Full provenance trail for a single sub-question search run."""

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
    """Canonical paper metadata exported by agentic M2."""

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
    """One evidence item extracted during full-text reading."""

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
    """Aggregated results for one sub-question (agentic M2 output)."""

    sub_question: str
    papers: List[M2PaperExport] = Field(default_factory=list)
    evidence: List[M2EvidenceExport] = Field(default_factory=list)
    knowledge_entries: List[KnowledgeEntry] = Field(default_factory=list)
    search_provenance: M2SearchProvenance = Field(default_factory=M2SearchProvenance)

    @model_validator(mode="after")
    def validate_provenance(self) -> "M2KnowledgeRun":
        """Ensure every exported knowledge item is traceable to local evidence."""
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
                raise ValueError(
                    f"knowledge {item.id!r} references evidence {evidence_id!r} "
                    "from a different paper"
                )
        return self


class M2KnowledgeExport(BaseModel):
    """Complete M2 agentic-search export — the data contract between M2 and M3."""

    schema_version: Literal["m2-knowledge-export/v1"] = "m2-knowledge-export/v1"
    runs: List[M2KnowledgeRun] = Field(default_factory=list)


# ============================================================================
# M3 Grounding 模型 — 全文证据接地
# ============================================================================

class AtomicClaim(BaseModel):
    """An atomic claim grounded in one or more M2 evidence records."""

    id: str
    statement: str
    evidence_ids: List[str] = Field(default_factory=list)
    paper_ids: List[str] = Field(default_factory=list)
    entities: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class EvidenceRelation(BaseModel):
    """A judged semantic relation before conversion to an EvidenceEdge.

    .. versionchanged:: 0.2.0
        Added ``model_config`` with ``extra="allow"`` so M3 grounding can
        pass through debug/trace fields (``retrieval_score``,
        ``condition_comparability``, ``candidate_origin``) without
        serialisation errors.
    """

    model_config = ConfigDict(extra="allow")

    id: str
    source: str
    target: str
    relation: Literal[
        "supports", "contradicts", "extends", "limits", "same_as", "refines"
    ]
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    rationale: str = ""
    evidence_ids: List[str] = Field(default_factory=list)
    source_paper_ids: List[str] = Field(default_factory=list)
    target_paper_ids: List[str] = Field(default_factory=list)


class GroundingReport(BaseModel):
    """M3 grounding 产出报告。

    .. note::
        ``evidence_records`` and ``relations_total`` are **count** fields
        (integers), not the full record/relation payloads.  The actual
        records live inside ``EvidenceGraph`` nodes and edges.
    """

    papers_total: int = 0
    full_text_papers: int = 0
    abstract_only_papers: int = 0
    fallback_papers: int = 0
    chunks_total: int = 0
    queries_total: int = 0
    evidence_records: int = 0  # count, not list
    claims_total: int = 0
    relations_total: int = 0
    relation_pairs_recalled: int = 0
    relation_candidates_judged: int = 0
    relation_candidates_selected: int = 0
    relation_selection_mode: str = "direct"
    relation_search: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)


class GroundingResult(BaseModel):
    """Typed result returned by the future M3 grounding workflow."""

    claims: List[AtomicClaim] = Field(default_factory=list)
    relations: List[EvidenceRelation] = Field(default_factory=list)
    report: GroundingReport = Field(default_factory=GroundingReport)


# ============================================================================
# Iteration core — evidence sufficiency, routing & followup contracts
# ============================================================================

def _normalise_gap_part(text: str) -> str:
    """Lowercase / collapse-whitespace normalisation shared by all gap-id parts."""
    return re.sub(r"\s+", " ", str(text or "").lower()).strip()


def make_gap_id(
    target_sub_question: str, gap_type: str, canonical_entities: Sequence[str]
) -> str:
    """Stable composite id for an evidence gap (v2 contract).

    ``sha1(target_sub_question | gap_type | ",".join(sorted(entities)))[:12]``
    with every part lowercased / whitespace-normalised and the canonical
    entities **comma-joined** after sorting, so the *same* gap — even
    described with different wording but bound to the same sub-question,
    type and canonical entities — always maps to the same id across review
    rounds.  Entity order does not matter (sorted before hashing).
    """
    sub_q = _normalise_gap_part(target_sub_question)
    gtype = _normalise_gap_part(gap_type)
    entities = ",".join(
        sorted(_normalise_gap_part(e) for e in canonical_entities if str(e or "").strip())
    )
    raw = f"{sub_q}|{gtype}|{entities}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


GapType = Literal[
    "mechanism", "population", "dosage", "conflict", "coverage", "other"
]


class EvidenceGap(BaseModel):
    """One identified gap in the evidence base (M6 verdict output).

    ``gap_id`` is derived **code-side** from
    ``(target_sub_question, gap_type, canonical_entities)`` — never from the
    free-text description alone, and never trusted from LLM output.  When
    ``target_sub_question`` is missing the description is used as the hash
    anchor instead, so id derivation never crashes.

    ``status`` values:

    * ``open`` — freshly identified, awaiting a supplement search;
    * ``pending_grounding`` — M2 tried to mitigate it, waiting for M3
      grounding to confirm the new evidence actually closes the gap;
    * ``closed`` — resolved (confirmed or disappeared while sufficient);
    * ``unimprovable`` — cannot be addressed by more searching.
    """

    gap_id: str = ""
    description: str
    gap_type: GapType = "other"
    canonical_entities: List[str] = Field(default_factory=list)
    suggested_queries: List[str] = Field(default_factory=list)
    target_sub_question: str = ""
    source_review_version: int = 0
    status: Literal["open", "pending_grounding", "closed", "unimprovable"] = "open"
    attempts: int = 0

    @model_validator(mode="after")
    def _ensure_gap_id(self) -> "EvidenceGap":
        if not self.gap_id:
            anchor = self.target_sub_question or self.description
            self.gap_id = make_gap_id(anchor, self.gap_type, self.canonical_entities)
        return self


class EvidenceSufficiencyVerdict(BaseModel):
    """M6 structured verdict: is the current evidence base sufficient?"""

    sufficient: bool
    gaps: List[EvidenceGap] = Field(default_factory=list)
    rationale: str = ""


class RoutingDecision(BaseModel):
    """One auditable routing decision recorded in ``routing_history``."""

    round: int
    from_module: str
    to_module: str
    decided_by: Literal["m6", "m1", "policy"]
    reason: str = ""
    gap_ids: List[str] = Field(default_factory=list)


class FollowupRequest(BaseModel):
    """A follow-up question issued on top of a completed parent run."""

    text: str
    parent_run_id: str = ""
    skip_search: Optional[bool] = None  # decided by M1 (None = not yet decided)


class SearchLedger(BaseModel):
    """Cross-round ledger of M2 search activity."""

    queries_issued: List[str] = Field(default_factory=list)
    paper_keys: List[str] = Field(default_factory=list)


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
    m2_knowledge_export: Optional[M2KnowledgeExport] = None  # agentic M2 enhanced export

    # ---- M3 ----
    evidence_graph: Optional[EvidenceGraph] = None
    grounding_report: Optional[GroundingReport] = None  # M3 grounding variant

    # ---- M4 ----
    candidate_hypotheses: List[HypothesisCard] = Field(default_factory=list)
    top_hypotheses: List[HypothesisCard] = Field(default_factory=list)
    best_hypotheses: List[HypothesisCard] = Field(default_factory=list)  # keep-best across iterations

    # ---- M5 ----
    research_plans: List[ResearchPlan] = Field(default_factory=list)

    # ---- M6 + iteration ----
    reviews: List[ReviewResult] = Field(default_factory=list)
    # iteration_count = number of M6 review rounds; doubles as the GLOBAL hard
    # stop (supplement rounds also consume this budget: global cap = max_iterations).
    iteration_count: int = 0
    max_iterations: int = 3
    review_score_threshold: float = 4.0  # M6 overall (1–5) at/above which iteration stops
    user_guidance: List[str] = Field(default_factory=list)  # human guidance injected between iterations

    # ---- iteration core (evidence sufficiency / routing / followup) ----
    # All fields have defaults so old checkpoints (without them) still load.
    evidence_verdict: Optional[EvidenceSufficiencyVerdict] = None
    evidence_gaps: List[EvidenceGap] = Field(default_factory=list)
    followup: Optional[FollowupRequest] = None
    parent_run_id: str = ""
    # search_round = number of M2 executions (fresh + supplements), capped by
    # config.max_search_rounds; independent of iteration_count.
    search_round: int = 0
    # revision_count = number of M4 revision rounds (audit/display only —
    # never gates routing).
    revision_count: int = 0
    search_ledger: SearchLedger = Field(default_factory=SearchLedger)
    routing_history: List[RoutingDecision] = Field(default_factory=list)

    # ---- persistence ----
    memory_cache_dir: str = ""  # non-empty enables persistent knowledge graph

    # ---- metadata ----
    run_id: str = ""
    errors: List[str] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)

    # ---- token tracking (populated by LLM-calling modules) ----
    total_input_tokens: int = 0
    total_output_tokens: int = 0
