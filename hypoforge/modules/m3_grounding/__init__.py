"""Full-text evidence grounding used internally by M3."""

from .models import (
    AtomicClaim,
    EvidenceRecord,
    EvidenceRelation,
    FullTextChunk,
    GroundingReport,
    PaperSource,
    RelationCandidate,
    RelationPair,
)
from .evidence_gams import EvidenceGraphGAMS
from .relation_retrieval import RelationCandidateRetriever
from .workflow import FullTextEvidenceGrounding

__all__ = [
    "AtomicClaim", "EvidenceRecord", "EvidenceRelation", "FullTextChunk",
    "GroundingReport", "PaperSource", "RelationCandidate", "RelationPair",
    "EvidenceGraphGAMS", "RelationCandidateRetriever", "FullTextEvidenceGrounding",
]
