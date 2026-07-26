"""M3 Evidence Grounding — consumes M2KnowledgeExport, produces EvidenceGraph claims & relations.

Unlike earlier development versions, this module does **not** download or parse
full-text papers.  All evidence originates from ``M2EvidenceExport`` items
produced by Track A's ``FullTextReadingWorkflow``.
"""

from .models import (
    AtomicClaim,
    EvidenceRecord,
    EvidenceRelation,
    GroundingReport,
    GroundingResult,
    RelationCandidate,
    RelationPair,
)
from .evidence_gams import EvidenceGraphGAMS
from .relation_retrieval import RelationCandidateRetriever
from .workflow import GroundingWorkflow

__all__ = [
    "AtomicClaim",
    "EvidenceRecord",
    "EvidenceRelation",
    "GroundingReport",
    "GroundingResult",
    "RelationCandidate",
    "RelationPair",
    "EvidenceGraphGAMS",
    "RelationCandidateRetriever",
    "GroundingWorkflow",
]
