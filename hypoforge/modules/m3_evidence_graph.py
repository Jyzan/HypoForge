"""
M3: Evidence Graph Construction.

Takes structured knowledge entries from M2 and builds a typed graph of
claims, evidence, sources, limitations, conflicts, and entities.

Output: ``evidence_graph`` in state.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from ..protocol import ModuleProtocol
from ..registry import ModuleRegistry
from ..state import (
    EvidenceEdge,
    EvidenceEdgeRelation,
    EvidenceGraph,
    EvidenceNode,
    EvidenceNodeType,
    KnowledgeEntryType,
    PipelineState,
)


@ModuleRegistry.register
class M3EvidenceGraph(ModuleProtocol):
    module_name = "m3"
    module_version = "0.1.0"
    description = "Build typed evidence graph from structured knowledge entries"

    # ------------------------------------------------------------------
    # ModuleProtocol implementation
    # ------------------------------------------------------------------

    def __init__(self, **kwargs):
        pass

    async def __call__(
        self,
        state: PipelineState,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        # --- collect all knowledge entries ---
        all_entries = []
        for lr in state.literature_results:
            all_entries.extend(lr.knowledge_entries)

        if not all_entries:
            # Edge case: no literature — return empty graph
            return {"evidence_graph": EvidenceGraph()}

        # --- build nodes ---
        nodes: List[EvidenceNode] = []
        edges: List[EvidenceEdge] = []
        node_ids: Dict[str, str] = {}  # entry_id → node_id

        # Source nodes (one per unique paper)
        seen_papers: Dict[str, str] = {}
        for e in all_entries:
            pid = e.source_paper_id
            if pid and pid not in seen_papers:
                nid = f"SRC_{pid}"
                seen_papers[pid] = nid
                nodes.append(EvidenceNode(
                    id=nid,
                    type=EvidenceNodeType.SOURCE,
                    label=e.source_paper_title or pid,
                ))

        # Claim / Evidence / ... nodes (one per entry)
        for e in all_entries:
            nid = f"N_{e.id}" if e.id else f"N_{uuid.uuid4().hex[:6]}"
            node_ids[e.id] = nid

            node_type = {
                KnowledgeEntryType.ESTABLISHED_FACT: EvidenceNodeType.EVIDENCE,
                KnowledgeEntryType.MECHANISTIC_CONCLUSION: EvidenceNodeType.CLAIM,
                KnowledgeEntryType.CONFLICTING_EVIDENCE: EvidenceNodeType.CONFLICT,
                KnowledgeEntryType.METHOD: EvidenceNodeType.EVIDENCE,
                KnowledgeEntryType.KNOWLEDGE_GAP: EvidenceNodeType.LIMITATION,
                KnowledgeEntryType.KEY_ENTITY: EvidenceNodeType.ENTITY,
            }.get(e.type, EvidenceNodeType.EVIDENCE)

            nodes.append(EvidenceNode(
                id=nid,
                type=node_type,
                label=e.content[:120],
                metadata={"entry_type": e.type.value, "confidence": e.confidence.value if e.confidence else None},
            ))

            # Edge: source → this node (involves)
            if e.source_paper_id and e.source_paper_id in seen_papers:
                edges.append(EvidenceEdge(
                    source=seen_papers[e.source_paper_id],
                    target=nid,
                    relation=EvidenceEdgeRelation.INVOLVES,
                ))

        # --- categorise entries ---
        established = []
        conflicts = []
        gaps = []
        for e in all_entries:
            if e.type == KnowledgeEntryType.ESTABLISHED_FACT:
                established.append(e.id)
            elif e.type == KnowledgeEntryType.CONFLICTING_EVIDENCE:
                conflicts.append(e.id)
            elif e.type == KnowledgeEntryType.KNOWLEDGE_GAP:
                gaps.append(e.id)

        graph = EvidenceGraph(
            nodes=nodes,
            edges=edges,
            established_facts=established,
            conflicts=conflicts,
            knowledge_gaps=gaps,
        )

        return {"evidence_graph": graph}

    @classmethod
    def get_input_fields(cls) -> List[str]:
        return ["literature_results"]

    @classmethod
    def get_output_fields(cls) -> List[str]:
        return ["evidence_graph"]
