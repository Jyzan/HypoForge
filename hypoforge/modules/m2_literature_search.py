"""
M2: Literature Search & Knowledge Extraction.

Retrieves papers from Semantic Scholar / PubMed and extracts six categories
of structured knowledge from each paper.

Output: ``literature_results`` (list of ``LiteratureResult``) in state.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..protocol import ModuleProtocol
from ..registry import ModuleRegistry
from ..state import (
    ConfidenceLevel,
    KnowledgeEntry,
    KnowledgeEntryType,
    LiteratureResult,
    PipelineState,
)


@ModuleRegistry.register
class M2LiteratureSearch(ModuleProtocol):
    module_name = "m2"
    module_version = "0.1.0"
    description = "Literature retrieval + six-category structured knowledge extraction"

    # ------------------------------------------------------------------
    # Stub data
    # ------------------------------------------------------------------

    _STUB_ENTRIES = [
        KnowledgeEntry(
            id="KE001",
            type=KnowledgeEntryType.ESTABLISHED_FACT,
            content="Hsp70 chaperones assist protein folding via an ATP-dependent cycle.",
            confidence=ConfidenceLevel.HIGH,
            source_paper_id="PMID:32012345",
            source_paper_title="Hsp70 chaperone cycle: structure and mechanism",
            entities=["Hsp70", "ATP"],
        ),
        KnowledgeEntry(
            id="KE002",
            type=KnowledgeEntryType.MECHANISTIC_CONCLUSION,
            content="NAD+ depletion impairs Hsp70 ATPase activity in aged cells.",
            confidence=ConfidenceLevel.MEDIUM,
            source_paper_id="PMID:31987654",
            source_paper_title="NAD+ metabolism regulates proteostasis in aging",
            entities=["NAD+", "Hsp70", "aging"],
        ),
        KnowledgeEntry(
            id="KE003",
            type=KnowledgeEntryType.CONFLICTING_EVIDENCE,
            content="Some studies suggest Hsp70 is NAD+-independent; others show NAD+ modulation.",
            confidence=ConfidenceLevel.LOW,
            source_paper_id="PMID:31876543",
            source_paper_title="Debate: metabolic regulation of chaperone activity",
            entities=["Hsp70", "NAD+"],
        ),
        KnowledgeEntry(
            id="KE004",
            type=KnowledgeEntryType.METHOD,
            content="ATPase activity assay, FRET-based folding sensor, Cryo-EM.",
            confidence=ConfidenceLevel.HIGH,
            source_paper_id="PMID:32012345",
            source_paper_title="Hsp70 chaperone cycle: structure and mechanism",
            entities=["Hsp70"],
        ),
        KnowledgeEntry(
            id="KE005",
            type=KnowledgeEntryType.KNOWLEDGE_GAP,
            content="The direct link between cellular NAD+/NADH ratio and Hsp70 folding efficiency in vivo is unknown.",
            confidence=None,
            source_paper_id="PMID:31987654",
            source_paper_title="NAD+ metabolism regulates proteostasis in aging",
            entities=["NAD+", "NADH", "Hsp70"],
        ),
        KnowledgeEntry(
            id="KE006",
            type=KnowledgeEntryType.ESTABLISHED_FACT,
            content="Amyloid-beta aggregation is a hallmark of Alzheimer disease pathology.",
            confidence=ConfidenceLevel.HIGH,
            source_paper_id="PMID:31765432",
            source_paper_title="Amyloid cascade hypothesis: 2024 update",
            entities=["amyloid-beta", "Alzheimer disease"],
        ),
        KnowledgeEntry(
            id="KE007",
            type=KnowledgeEntryType.KNOWLEDGE_GAP,
            content="Whether enhancing chaperone activity can clear pre-formed amyloid aggregates remains controversial.",
            confidence=None,
            source_paper_id="PMID:31654321",
            source_paper_title="Chaperone-based therapies for neurodegeneration",
            entities=["chaperone", "amyloid"],
        ),
    ]

    # ------------------------------------------------------------------
    # ModuleProtocol implementation
    # ------------------------------------------------------------------

    def __init__(self, search_tools: Optional[List[str]] = None, **kwargs):
        self.search_tools = search_tools or ["semantic_scholar", "pubmed"]

    async def __call__(
        self,
        state: PipelineState,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        # In stub mode, produce one LiteratureResult per sub-question
        sub_questions = (
            state.problem_card.sub_questions
            if state.problem_card
            else ["Unknown sub-question"]
        )

        results: List[LiteratureResult] = []
        for sq in sub_questions:
            results.append(LiteratureResult(
                sub_question=sq,
                papers_retrieved=5,
                knowledge_entries=self._STUB_ENTRIES,
            ))

        return {"literature_results": results}

    @classmethod
    def get_input_fields(cls) -> List[str]:
        return ["problem_card"]

    @classmethod
    def get_output_fields(cls) -> List[str]:
        return ["literature_results"]
