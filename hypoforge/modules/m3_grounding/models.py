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
    source: str
    target: str
    relation: Literal[
        "supports", "contradicts", "extends", "limits", "same_as", "refines"
    ]
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    rationale: str = ""


class GroundingReport(BaseModel):
    papers_total: int = 0
    full_text_papers: int = 0
    fallback_papers: int = 0
    chunks_total: int = 0
    queries_total: int = 0
    evidence_records: int = 0
    claims_total: int = 0
    relations_total: int = 0
    retrieval_backend: str = "bm25"
    warnings: List[str] = Field(default_factory=list)
    paper_status: Dict[str, str] = Field(default_factory=dict)
