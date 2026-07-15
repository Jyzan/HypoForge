"""Shared contracts for the agentic M2 literature pipeline.

Feature branches should import public contracts from this module instead of
depending on another tool's private implementation.
"""

from .models import (
    ContentLevel,
    CoverageReport,
    DocumentChunk,
    DocumentRecord,
    EvidenceBucket,
    EvidenceChunk,
    EvidenceLinkedKnowledge,
    FulltextStatus,
    PaperReadingResult,
    PaperRecord,
    QueryIntent,
    ScoutNote,
    SearchBudget,
    SearchQuery,
    SearchRunResult,
    SearchState,
    StopReason,
)
from .protocols import (
    CoverageEvaluatorProtocol,
    DocumentParserProtocol,
    EvidenceRetrieverProtocol,
    FulltextResolverProtocol,
    LiteratureSourceProtocol,
    PaperDeduplicatorProtocol,
    PaperRankerProtocol,
    PaperReaderProtocol,
    QueryPlannerProtocol,
    ScoutReaderProtocol,
)

__all__ = [
    "ContentLevel",
    "CoverageEvaluatorProtocol",
    "CoverageReport",
    "DocumentChunk",
    "DocumentParserProtocol",
    "DocumentRecord",
    "EvidenceBucket",
    "EvidenceChunk",
    "EvidenceLinkedKnowledge",
    "EvidenceRetrieverProtocol",
    "FulltextResolverProtocol",
    "FulltextStatus",
    "LiteratureSourceProtocol",
    "PaperDeduplicatorProtocol",
    "PaperRankerProtocol",
    "PaperReaderProtocol",
    "PaperReadingResult",
    "PaperRecord",
    "QueryIntent",
    "QueryPlannerProtocol",
    "ScoutNote",
    "ScoutReaderProtocol",
    "SearchBudget",
    "SearchQuery",
    "SearchRunResult",
    "SearchState",
    "StopReason",
]
