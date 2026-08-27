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
    HYPOTHESIS = "hypothesis"


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
    METHOD_FEASIBILITY = "method_feasibility"
    TASK_ALIGNMENT = "task_alignment"
    TASK_COVERAGE = "task_coverage"
    NOVELTY_SCORE = "novelty"
    EVIDENCE_RELIABILITY = "evidence_reliability"
    EXPERIMENTAL_RIGOR = "experimental_rigor"
    STATISTICS_REPRODUCIBILITY = "statistics_reproducibility"
    TECHNICAL_FEASIBILITY = "technical_feasibility"
    TESTABILITY_DIMENSION = "testability"
    EVIDENCE_COVERAGE = "evidence_coverage_gate"
    ANSWER_COMPLETENESS = "answer_completeness_gate"
    SOURCE_QUALITY = "source_quality_gate"
    TESTABILITY = "testability_metric"
    NOVELTY = "novelty_metric"
    OBJECTIVE_EVIDENCE_CONSISTENCY = "objective_evidence_consistency"
    EXPERIMENTAL_VALIDATION_COVERAGE = "experimental_validation_coverage"
    OVERALL = "overall"


# ============================================================================
# M1 产出 — ProblemCard
# ============================================================================

class TaskEntity(BaseModel):
    """One task-scoped entity with aliases supplied by M1, not source code."""

    entity_id: str
    name: str
    source_mention: str = ""
    aliases: List[str] = Field(default_factory=list)
    role: Literal[
        "primary_object", "intervention", "outcome", "method",
        "context", "constraint", "other",
    ] = "other"
    required: bool = False

    @field_validator("entity_id", "name", "source_mention", mode="before")
    @classmethod
    def normalize_scalar(cls, value: Any) -> str:
        return re.sub(r"\s+", " ", str(value or "")).strip()

    @field_validator("aliases", mode="before")
    @classmethod
    def normalize_aliases(cls, value: Any) -> List[str]:
        if value is None:
            return []
        return list(dict.fromkeys(
            re.sub(r"\s+", " ", str(item or "")).strip()
            for item in value
            if str(item or "").strip()
        ))


class TaskRequirement(BaseModel):
    """An atomic user requirement mapped to one M1 sub-question."""

    requirement_id: str
    sub_question: str
    primary_entity_id: str = ""
    related_entity_ids: List[str] = Field(default_factory=list)
    relation: str
    required: bool = True

    @field_validator(
        "requirement_id", "sub_question", "primary_entity_id", "relation",
        mode="before",
    )
    @classmethod
    def normalize_scalar(cls, value: Any) -> str:
        return re.sub(r"\s+", " ", str(value or "")).strip()

    @field_validator("related_entity_ids", mode="before")
    @classmethod
    def normalize_entity_ids(cls, value: Any) -> List[str]:
        if value is None:
            return []
        return list(dict.fromkeys(str(item or "").strip() for item in value if str(item or "").strip()))


class TaskContract(BaseModel):
    """Domain-neutral task identity propagated from M1 through M6."""

    source: Literal["m1", "derived"] = "m1"
    entities: List[TaskEntity] = Field(default_factory=list)
    requirements: List[TaskRequirement] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_references(self) -> "TaskContract":
        entity_ids = [entity.entity_id for entity in self.entities]
        requirement_ids = [item.requirement_id for item in self.requirements]
        if len(entity_ids) != len(set(entity_ids)):
            raise ValueError("task contract contains duplicate entity IDs")
        if len(requirement_ids) != len(set(requirement_ids)):
            raise ValueError("task contract contains duplicate requirement IDs")
        known = set(entity_ids)
        for requirement in self.requirements:
            referenced = [
                requirement.primary_entity_id,
                *requirement.related_entity_ids,
            ]
            unknown = [item for item in referenced if item and item not in known]
            if unknown:
                raise ValueError(
                    f"requirement {requirement.requirement_id!r} references "
                    f"unknown task entities: {unknown}"
                )
        return self


class TaskTraceReference(BaseModel):
    """A contract ID plus a literal excerpt from the generated output."""

    contract_id: str
    output_excerpt: str


class TaskTrace(BaseModel):
    """Auditable claim that an output addresses task entities/requirements."""

    entity_mentions: List[TaskTraceReference] = Field(default_factory=list)
    requirement_mentions: List[TaskTraceReference] = Field(default_factory=list)


class ProblemCard(BaseModel):
    """Structured decomposition of a scientific question (M1 output)."""

    original_question: str
    domain: List[str] = Field(default_factory=list)
    sub_questions: List[str] = Field(default_factory=list)
    key_entities: List[str] = Field(default_factory=list)
    task_contract: TaskContract = Field(default_factory=TaskContract)

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

    @model_validator(mode="after")
    def provide_backward_compatible_contract(self) -> "ProblemCard":
        """Derive a conservative contract for snapshots created before v2."""

        if self.task_contract.entities or self.task_contract.requirements:
            if self.task_contract.source == "m1":
                self.key_entities = [
                    entity.name
                    for entity in self.task_contract.entities
                    if entity.name
                ]
            return self
        entities = [
            TaskEntity(
                entity_id=f"E{index}",
                name=name,
                source_mention=name,
                role="primary_object" if index == 1 else "other",
                required=index == 1,
            )
            for index, name in enumerate(self.key_entities, start=1)
            if name
        ]
        primary_id = entities[0].entity_id if entities else ""
        requirements = [
            TaskRequirement(
                requirement_id=f"R{index}",
                sub_question=question,
                primary_entity_id=primary_id,
                relation="address the atomic sub-question",
            )
            for index, question in enumerate(self.sub_questions, start=1)
            if question
        ]
        self.task_contract = TaskContract(
            source="derived",
            entities=entities,
            requirements=requirements,
        )
        return self


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
    # Explicit routing provenance.  Text matching is not reliable once an M6
    # evidence-gap description is converted into a supplement sub-question.
    origin_gap_ids: List[str] = Field(default_factory=list)
    origin_gap_request_ids: List[str] = Field(default_factory=list)


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


class GraphCorrectionRequest(BaseModel):
    """A reviewer-proposed, not-yet-trusted graph mutation."""

    request_id: str = ""
    operation: Literal["add_edge", "remove_edge", "reclassify_edge"]
    source_node_id: str
    target_node_id: str
    current_relation: Optional[EvidenceEdgeRelation] = None
    proposed_relation: Optional[EvidenceEdgeRelation] = None
    evidence_ids: List[str] = Field(default_factory=list)
    reason: str
    requested_by: str = "objective_evidence_consistency"
    iteration: int = Field(default=0, ge=0)
    status: Literal["pending", "applied", "rejected"] = "pending"
    rejection_reason: str = ""


class GraphAuditRecord(BaseModel):
    """Append-only before/after record for one graph correction request."""

    sequence: int = Field(ge=1)
    graph_version_before: int = Field(ge=1)
    graph_version_after: int = Field(ge=1)
    request_id: str
    operation: str
    status: Literal["applied", "rejected"]
    before: Dict[str, Any] = Field(default_factory=dict)
    after: Dict[str, Any] = Field(default_factory=dict)
    reason: str = ""
    evidence_ids: List[str] = Field(default_factory=list)
    iteration: int = Field(default=0, ge=0)


class EntityMergeMember(BaseModel):
    """Lossless snapshot of one entity before a graph-level merge."""

    node_id: str
    name: str
    similarity_to_canonical: float = Field(ge=-1.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EntityMergeRecord(BaseModel):
    """Append-only audit record for one final-graph entity merge group."""

    merge_id: str
    round_index: int = Field(default=0, ge=0)
    threshold: float = Field(default=0.92, ge=-1.0, le=1.0)
    embedding_model: str = ""
    method: Literal["exact", "embedding_cosine"]
    canonical_node_id: str
    canonical_name: str
    members: List[EntityMergeMember] = Field(default_factory=list)
    node_count_before: int = Field(default=0, ge=0)
    node_count_after: int = Field(default=0, ge=0)
    redirected_edge_count: int = Field(default=0, ge=0)
    deduplicated_edge_count: int = Field(default=0, ge=0)
    removed_self_loop_count: int = Field(default=0, ge=0)


class EvidenceGraph(BaseModel):
    """Full evidence graph (M3 output)."""

    nodes: List[EvidenceNode] = Field(default_factory=list)
    edges: List[EvidenceEdge] = Field(default_factory=list)

    # Convenience lists of KnowledgeEntry ids, populated by M3
    established_facts: List[str] = Field(default_factory=list)
    conflicts: List[str] = Field(default_factory=list)
    knowledge_gaps: List[str] = Field(default_factory=list)
    version: int = Field(default=1, ge=1)
    audit_log: List[GraphAuditRecord] = Field(default_factory=list)
    entity_merge_log: List[EntityMergeRecord] = Field(default_factory=list)


# ============================================================================
# M4 产出 — Hypothesis Cards
# ============================================================================

HypothesisGroundingStatus = Literal[
    "legacy_unknown",
    "evidence_backed",
    "mixed",
    "bridge_only",
]

class HypothesisPremise(BaseModel):
    """One epistemically explicit premise used by an M4 hypothesis.

    ``evidence_backed`` premises may be strengthened only by canonical
    evidence IDs.  ``unverified_bridge`` premises are graph hypotheses
    created by M3 after an effective search leaves a relation unresolved;
    they must never claim paper or evidence provenance.
    """

    premise_id: str
    claim: str
    kind: Literal["evidence_backed", "unverified_bridge"]
    required: bool = True
    supporting_evidence_ids: List[str] = Field(default_factory=list)
    source_paper_ids: List[str] = Field(default_factory=list)
    bridge_hypothesis_node_id: str = ""
    origin_gap_id: str = ""
    audit_verdict: Literal[
        "supported", "partially_supported", "unsupported",
        "contradicted", "invalid_citation", "not_applicable",
    ] = "not_applicable"
    audit_reason: str = ""
    original_claim: str = ""

    @model_validator(mode="after")
    def validate_epistemic_source(self) -> "HypothesisPremise":
        if self.kind == "evidence_backed":
            if self.bridge_hypothesis_node_id:
                raise ValueError("evidence-backed premise cannot reference a bridge node")
        else:
            if self.supporting_evidence_ids or self.source_paper_ids:
                raise ValueError("unverified bridge cannot claim canonical evidence")
            if not self.bridge_hypothesis_node_id:
                raise ValueError("unverified bridge requires a hypothesis node ID")
        return self


class HypothesisCard(BaseModel):
    """A single candidate hypothesis (M4 output)."""

    hypothesis_id: str
    statement: str
    mechanism: str = ""
    observable_predictions: List[str] = Field(default_factory=list)
    falsification_conditions: List[str] = Field(default_factory=list)
    supporting_evidence: List[str] = Field(default_factory=list)
    source_paper_ids: List[str] = Field(default_factory=list)
    task_trace: TaskTrace = Field(default_factory=TaskTrace)

    # Reasoning written by the Ranker *before* the numeric scores (reason-before-score).
    ranking_rationale: str = ""

    # Multi-dimensional scores (populated by Ranker in M4)
    scores: Dict[str, float] = Field(default_factory=dict)

    # Explicit epistemic structure.  Defaults keep historical checkpoints
    # loadable and preserve the legacy M5/M6 fields above.
    factual_premises: List[HypothesisPremise] = Field(default_factory=list)
    working_assumptions: List[HypothesisPremise] = Field(default_factory=list)
    research_gap: str = ""
    # New M4 outputs must replace this compatibility value with an explicit
    # epistemic status at the module boundary.  Keeping the default here lets
    # historical checkpoints load without a migration step.
    grounding_status: HypothesisGroundingStatus = "legacy_unknown"


class EvidenceGapRequest(BaseModel):
    """A searchable M4 evidence gap with an auditable bounded lifecycle."""

    gap_id: str
    hypothesis_id: str = ""
    requirement_id: str = ""
    sub_question: str
    task_entity_ids: List[str] = Field(default_factory=list)
    relation: str = ""
    missing_evidence_type: Literal[
        "supporting_evidence", "contradicting_evidence", "method_evidence",
        "recent_evidence", "source_quality",
    ] = "supporting_evidence"
    suggested_queries: List[str] = Field(default_factory=list)
    executed_queries: List[str] = Field(default_factory=list)
    status: Literal[
        "pending", "searched", "indexed", "resolved", "hypothesized", "exhausted",
    ] = "pending"
    attempts: int = Field(default=0, ge=0)
    created_iteration: int = Field(default=0, ge=0)
    resolution_evidence_ids: List[str] = Field(default_factory=list)
    audit_claim: str = ""
    bridge_hypothesis_node_id: str = ""
    rationale: str = ""
    premise_id: str = ""
    scientific_resolution: Literal[
        "unreviewed", "supported", "contradicted", "unresolved",
    ] = "unreviewed"
    contradicting_evidence_ids: List[str] = Field(default_factory=list)
    search_completed: bool = False
    technical_errors: List[str] = Field(default_factory=list)
    corrected_claim: str = ""


# ============================================================================
# M5 产出 — Research Plan
# ============================================================================

class ResearchPlanEvidenceLink(BaseModel):
    """Trace one plan claim/step to evidence, or mark it as unverified."""

    plan_element: str
    claim: str = ""
    supporting_evidence_ids: List[str] = Field(default_factory=list)
    source_paper_ids: List[str] = Field(default_factory=list)
    support_status: Literal[
        "supported", "hypothesis_to_validate", "unsupported"
    ] = "hypothesis_to_validate"


class ValidationTarget(BaseModel):
    """One M4 claim that M5 must be able to test or falsify."""

    target_id: str
    hypothesis_id: str
    target_kind: Literal[
        "statement", "mechanism", "prediction",
        "falsification", "working_assumption",
    ]
    target_text: str
    required: bool = True
    bridge_hypothesis_node_id: str = ""


class ValidationCoverageItem(BaseModel):
    """M6's auditable mapping from one M4 target to M5 methods."""

    target_id: str
    target_kind: str
    target_text: str
    verdict: Literal["covered", "partial", "missing"]
    procedure_refs: List[str] = Field(default_factory=list)
    measurement_refs: List[str] = Field(default_factory=list)
    control_refs: List[str] = Field(default_factory=list)
    analysis_refs: List[str] = Field(default_factory=list)
    bridge_validation_refs: List[str] = Field(default_factory=list)
    falsification_text: str = ""
    rationale: str = ""


class ExperimentalValidationVerdict(BaseModel):
    """Structured M6 verdict for whether M5 can test M4's hypothesis."""

    sufficient: bool
    items: List[ValidationCoverageItem] = Field(default_factory=list)
    rationale: str = ""


class WorkingAssumptionValidation(BaseModel):
    """An explicit M5 experiment covering one unresolved M3 bridge."""

    bridge_hypothesis_node_id: str
    procedure: str
    measurement: str
    falsification_condition: str


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
    supporting_evidence_ids: List[str] = Field(default_factory=list)
    source_paper_ids: List[str] = Field(default_factory=list)
    evidence_links: List[ResearchPlanEvidenceLink] = Field(default_factory=list)
    bridge_validations: List[WorkingAssumptionValidation] = Field(default_factory=list)
    task_trace: TaskTrace = Field(default_factory=TaskTrace)


# ============================================================================
# M6 产出 — Review Results
# ============================================================================

class ReviewResult(BaseModel):
    """A single reviewer's assessment (M6 output)."""

    dimension: ReviewerDimension
    attribution: Literal["hypothesis", "plan", "both"] = "both"
    reasoning: str = ""  # written *before* the score (reason-before-score)
    # 1–5 for LLM reviewers; objective gates/metrics scale 0–1 × 5 and may
    # legitimately score 0 (e.g. 0% evidence coverage).
    score: float = Field(ge=0.0, le=5.0)
    comments: str = ""
    suggestions: str = ""
    evidence_ids: List[str] = Field(default_factory=list)
    hard_gate_passed: Optional[bool] = None
    graph_correction_requests: List[GraphCorrectionRequest] = Field(default_factory=list)
    version: int = 1

    @field_validator("reasoning", "suggestions", mode="before")
    @classmethod
    def _normalise_review_text(cls, value: Any) -> Any:
        """Accept the list form emitted by some structured-review responses.

        The public state contract keeps ``reasoning`` and ``suggestions`` as
        text because downstream modules and interfaces consume displayable
        strings. Qwen may nevertheless return bullet points as a JSON array,
        so join those entries at the contract boundary instead of failing the
        entire M6 iteration.
        """
        if isinstance(value, (list, tuple)):
            return "\n".join(
                str(item).strip()
                for item in value
                if str(item).strip()
            )
        return value


class ScoreDimensionDetail(BaseModel):
    """One auditable dimension of the modern M6 score."""

    dimension: Literal[
        "task_coverage",
        "novelty",
        "scientific_logic",
        "evidence_reliability",
        "testability",
        "experimental_rigor",
        "statistics_reproducibility",
        "technical_feasibility",
    ]
    score: float = Field(ge=1.0, le=5.0)
    weight: float = Field(gt=0.0, le=1.0)
    weighted_contribution: float = Field(ge=0.0, le=5.0)
    source: Literal[
        "deterministic", "independent", "llm", "hybrid", "degraded"
    ]
    confidence: float = Field(ge=0.0, le=1.0)
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    deductions: List[str] = Field(default_factory=list)


class ScoreCap(BaseModel):
    """A conservative upper bound applied after weighted aggregation."""

    rule_id: str
    maximum: float = Field(ge=1.0, le=5.0)
    reason: str
    attribution: Literal["hypothesis", "plan", "both"] = "both"


class M6ScoringSummary(BaseModel):
    """Persisted modern M6 score and its audit trail."""

    version: int = 1
    raw_score: float = Field(ge=1.0, le=5.0)
    final_score: float = Field(ge=1.0, le=5.0)
    dimensions: List[ScoreDimensionDetail]
    applied_caps: List[ScoreCap] = Field(default_factory=list)
    rationale: str = ""
    complete: bool = True


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


class M2PaperRetentionExport(BaseModel):
    paper_id: str
    decision: Literal["retain", "reject"]
    roles: List[str] = Field(default_factory=list)
    reason: str
    rank_position: int


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
    retention_decisions: List[M2PaperRetentionExport] = Field(default_factory=list)


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
    # Structured full-text failure attribution; empty when full text was
    # retrieved successfully. Categories: publisher_blocked / invalid_pdf_link
    # / no_fulltext_available / other.
    fulltext_failure_category: str = ""
    fulltext_failure_detail: str = ""
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
    origin_gap_ids: List[str] = Field(default_factory=list)
    origin_gap_request_ids: List[str] = Field(default_factory=list)

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

    The ``scientific_resolution`` field is authoritative for modern M6
    supplement rounds; ``status`` remains a lifecycle marker for backward
    compatibility with legacy checkpoints.
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
    # Gap-specific scientific outcome.  Defaults keep old checkpoints
    # loadable while preventing M3 from using evidence-count as a verdict.
    hypothesis_ids: List[str] = Field(default_factory=list)
    # Stable provenance back to the M4/M5 claim that produced this gap.  The
    # defaults keep historical checkpoints loadable and let routing
    # distinguish a missing fact from a contradicted one without parsing
    # free-form descriptions.
    source_claim_type: Literal["", "factual_premise", "plan_fact"] = ""
    source_claim_id: str = ""
    scientific_resolution: Literal[
        "unreviewed", "supported", "contradicted", "unresolved",
    ] = "unreviewed"
    resolution_evidence_ids: List[str] = Field(default_factory=list)
    contradicting_evidence_ids: List[str] = Field(default_factory=list)
    search_completed: bool = False
    technical_errors: List[str] = Field(default_factory=list)
    bridge_hypothesis_node_id: str = ""
    rationale: str = ""

    @model_validator(mode="after")
    def _ensure_gap_id(self) -> "EvidenceGap":
        if not self.gap_id:
            anchor = self.target_sub_question or self.description
            self.gap_id = make_gap_id(anchor, self.gap_type, self.canonical_entities)
        return self


class FactualPremiseAudit(BaseModel):
    """M6's persisted per-premise audit summary for the result/UI layer."""

    premise_id: str
    claim: str
    verdict: Literal[
        "supported", "partially_supported", "unsupported",
        "contradicted", "invalid_citation", "not_applicable",
    ] = "unsupported"
    evidence_ids: List[str] = Field(default_factory=list)
    corrected_claim: str = ""
    rationale: str = ""


class EvidenceSufficiencyVerdict(BaseModel):
    """M6 structured verdict: is the current evidence base sufficient?"""

    sufficient: bool
    gaps: List[EvidenceGap] = Field(default_factory=list)
    evidence_ids: List[str] = Field(default_factory=list)
    premise_audits: List[FactualPremiseAudit] = Field(default_factory=list)
    rationale: str = ""


class RoutingDecision(BaseModel):
    """One auditable routing decision recorded in ``routing_history``."""

    round: int
    from_module: str
    to_module: str
    decided_by: Literal["m6", "m4", "m1", "policy"]
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
    evidence_gap_requests: List[EvidenceGapRequest] = Field(default_factory=list)
    evidence_gap_search_rounds: int = 0
    max_evidence_gap_rounds: int = 1
    graph_correction_requests: List[GraphCorrectionRequest] = Field(default_factory=list)

    # ---- M5 ----
    research_plans: List[ResearchPlan] = Field(default_factory=list)
    research_plan_history: Dict[int, List[ResearchPlan]] = Field(default_factory=dict)

    # ---- M6 + iteration ----
    reviews: List[ReviewResult] = Field(default_factory=list)
    m6_scoring_summary: Optional[M6ScoringSummary] = None
    experimental_validation_verdict: Optional[ExperimentalValidationVerdict] = None
    # iteration_count = number of M6 review rounds; doubles as the GLOBAL hard
    # stop (supplement rounds also consume this budget: global cap = max_iterations).
    iteration_count: int = 0
    max_iterations: int = 3
    # Legacy display/config field retained for checkpoint compatibility.  M6
    # numeric scores no longer control iteration routing.
    review_score_threshold: float = 4.0
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
    # plan_revision_count = number of CONSECUTIVE plan-only (revise_m5)
    # rounds. GATES routing: capped by config.max_plan_revisions so a stuck
    # plan cannot starve M4's hypothesis-revision budget.
    plan_revision_count: int = 0
    search_ledger: SearchLedger = Field(default_factory=SearchLedger)
    routing_history: List[RoutingDecision] = Field(default_factory=list)

    # ---- persistence ----
    memory_cache_dir: str = ""  # non-empty enables persistent knowledge graph
    entity_cache_dir: str = ""  # non-empty enables cross-run entity decisions

    # ---- metadata ----
    run_id: str = ""
    errors: List[str] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)

    # ---- token tracking (populated by LLM-calling modules) ----
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    # Aggregated across repeated visits (for example M6 -> M2 -> M3 -> M4).
    # Each row has ``input``, ``output`` and successful response ``calls``.
    token_usage_by_module: Dict[str, Dict[str, int]] = Field(default_factory=dict)
    # Post-pipeline independent evaluation is intentionally separate: it runs
    # after M6 and must not make M6 appear more expensive than it is.
    scoring_token_usage: Dict[str, int] = Field(default_factory=dict)
