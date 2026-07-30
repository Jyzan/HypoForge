"""
M3 Grounding intermediate types.

Canonical models (AtomicClaim, EvidenceRelation, GroundingReport, GroundingResult)
are imported from ``hypoforge.state``.  This module defines only the M3-specific
intermediate types that bridge M2EvidenceExport items through the grounding workflow.

.. versionchanged:: 0.2.0 (Track B)
    Removed ``PaperSource`` and ``FullTextChunk`` — M3 no longer downloads or
    parses full-text papers.  All evidence originates from M2's
    ``M2EvidenceExport`` items.  ``EvidenceRecord`` is now a retrieval-time
    wrapper over M2 evidence rather than a chunk-derived record.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from hypoforge.state import AtomicClaim, EvidenceRelation, GroundingReport, GroundingResult  # noqa: F401


# ============================================================================
# M3-specific intermediate types
# ============================================================================


class EvidenceRecord(BaseModel):
    """A retrieval-time bridge between an M2EvidenceExport item and a user query.

    Created during the *retrieve* step when a BM25/embedding search matches
    an ``M2EvidenceExport`` to one of the planned queries.  It enriches the
    M2 evidence with query context and an LLM-written RCS summary, but the
    canonical source of truth remains ``evidence_id`` → ``M2EvidenceExport``.
    """

    id: str
    evidence_id: str  # points to M2EvidenceExport.evidence_id
    paper_id: str
    chunk_id: str = ""
    section: str = ""
    page: Optional[int] = None
    query: str = ""
    quote: str = ""
    normalized_claim: str = ""
    summary: str = ""
    excerpt: str = ""
    relevance_score: float = Field(default=0.0, ge=0.0, le=10.0)
    retrieval_score: float = 0.0
    claims: List[str] = Field(default_factory=list)
    entities: List[str] = Field(default_factory=list)
    methods: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    epistemic_status: str = "reported"
    context: Dict[str, Any] = Field(default_factory=dict)


class RelationPair(BaseModel):
    """A recalled pair of nodes worth sending to the relation judge."""

    id: str
    source: str
    target: str
    source_type: Literal["evidence_record", "claim"] = "claim"
    target_type: Literal["claim"] = "claim"
    retrieval_score: float = Field(default=0.0, ge=0.0, le=1.0)
    entity_overlap: List[str] = Field(default_factory=list)
    candidate_origin: List[str] = Field(default_factory=list)
    source_evidence_ids: List[str] = Field(default_factory=list)
    target_evidence_ids: List[str] = Field(default_factory=list)
    source_paper_ids: List[str] = Field(default_factory=list)
    target_paper_ids: List[str] = Field(default_factory=list)


class RelationCandidate(BaseModel):
    """A judged semantic edge candidate; ``unrelated`` never enters the graph."""

    id: str
    source: str
    target: str
    relation: Literal[
        "supports", "contradicts", "extends", "limits", "same_as", "refines", "unrelated"
    ]
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    rationale: str = ""
    evidence_ids: List[str] = Field(default_factory=list)
    source_paper_ids: List[str] = Field(default_factory=list)
    target_paper_ids: List[str] = Field(default_factory=list)
    retrieval_score: float = Field(default=0.0, ge=0.0, le=1.0)
    condition_comparability: float = Field(default=0.5, ge=0.0, le=1.0)
    entity_overlap: List[str] = Field(default_factory=list)
    candidate_origin: List[str] = Field(default_factory=list)

    def to_relation(self) -> EvidenceRelation:
        """Convert to a canonical EvidenceRelation (extra fields are dropped)."""
        if self.relation == "unrelated":
            raise ValueError("unrelated candidates cannot become graph edges")
        return EvidenceRelation(
            id=self.id,
            source=self.source,
            target=self.target,
            relation=self.relation,  # type: ignore[arg-type]
            confidence=self.confidence,
            rationale=self.rationale,
            evidence_ids=self.evidence_ids,
            source_paper_ids=self.source_paper_ids,
            target_paper_ids=self.target_paper_ids,
        )
