"""Typed objects used by the PaperQA-inspired M3 grounding subgraph."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class PaperSource(BaseModel):
    id: str
    title: str = ""
    doi: str = ""
    pmid: str = ""
    url: str = ""
    full_text_path: str = ""
    acquisition_status: str = "pending"
    acquisition_note: str = ""
    seed_entry_ids: List[str] = Field(default_factory=list)
    seed_text: str = ""


class FullTextChunk(BaseModel):
    id: str
    paper_id: str
    text: str
    section: str = ""
    page: Optional[int] = None
    start_char: int = 0
    end_char: int = 0
    source_path: str = ""


class EvidenceRecord(BaseModel):
    id: str
    chunk_id: str
    paper_id: str
    query: str
    summary: str
    excerpt: str
    relevance_score: float = Field(default=0.0, ge=0.0, le=10.0)
    retrieval_score: float = 0.0
    claims: List[str] = Field(default_factory=list)
    entities: List[str] = Field(default_factory=list)
    methods: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    epistemic_status: str = "reported"
    context: Dict[str, Any] = Field(default_factory=dict)


class AtomicClaim(BaseModel):
    id: str
    statement: str
    evidence_record_ids: List[str] = Field(default_factory=list)
    entities: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class EvidenceRelation(BaseModel):
    id: str = ""
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
    retrieval_score: float = Field(default=0.0, ge=0.0, le=1.0)
    condition_comparability: float = Field(default=0.5, ge=0.0, le=1.0)
    candidate_origin: List[str] = Field(default_factory=list)


class RelationPair(BaseModel):
    """A recalled pair worth sending to the relation judge."""

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
        if self.relation == "unrelated":
            raise ValueError("unrelated candidates cannot become graph edges")
        return EvidenceRelation(
            id=self.id,
            source=self.source,
            target=self.target,
            relation=self.relation,
            confidence=self.confidence,
            rationale=self.rationale,
            evidence_ids=self.evidence_ids,
            source_paper_ids=self.source_paper_ids,
            target_paper_ids=self.target_paper_ids,
            retrieval_score=self.retrieval_score,
            condition_comparability=self.condition_comparability,
            candidate_origin=self.candidate_origin,
        )


class GroundingReport(BaseModel):
    papers_total: int = 0
    full_text_papers: int = 0
    fallback_papers: int = 0
    chunks_total: int = 0
    queries_total: int = 0
    evidence_records: int = 0
    claims_total: int = 0
    relations_total: int = 0
    relation_pairs_recalled: int = 0
    relation_candidates_judged: int = 0
    relation_candidates_selected: int = 0
    relation_selection_mode: str = "direct"
    relation_search: Dict[str, Any] = Field(default_factory=dict)
    retrieval_backend: str = "bm25"
    warnings: List[str] = Field(default_factory=list)
    paper_status: Dict[str, str] = Field(default_factory=dict)
