"""M3: PaperQA-inspired full-text grounding and evidence-graph construction."""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional

from ..protocol import ModuleProtocol
from ..registry import ModuleRegistry
from ..state import (
    ConfidenceLevel,
    EvidenceEdge,
    EvidenceEdgeRelation,
    EvidenceGraph,
    EvidenceNode,
    EvidenceNodeType,
    KnowledgeEntry,
    KnowledgeEntryType,
    LiteratureResult,
    PipelineState,
)
from ..tools.qwen_client import QwenClient
from .m3_grounding import FullTextEvidenceGrounding
from .m3_grounding.models import AtomicClaim, EvidenceRecord, EvidenceRelation, FullTextChunk, PaperSource

logger = logging.getLogger(__name__)


@ModuleRegistry.register
class M3EvidenceGraph(ModuleProtocol):
    """Ground M2 seeds in paper text and build an auditable evidence graph.

    ``mode="llm"`` uses Qwen for RCS and context-aware claim relations.
    ``mode="rule"`` still performs full-text acquisition, parsing and
    BM25/MMR retrieval, but uses deterministic extractive evidence records.
    """

    module_name = "m3"
    module_version = "0.4.0"
    description = "Full-text grounding + candidate relation recall + optional Evidence-GAMS"

    def __init__(
        self,
        mode: str = "rule",
        llm_config: Optional[Any] = None,
        cache_dir: str = ".hypoforge_cache/m3_grounding",
        local_paper_dir: str = "papers",
        enable_network_fulltext: bool = False,
        embedding_model: str = "",
        chunk_chars: int = 9000,
        chunk_overlap_chars: int = 1000,
        retrieve_k: int = 30,
        evidence_k: int = 12,
        min_relevance: float = 5.0,
        max_queries: int = 16,
        max_sources: int = 60,
        grounding_max_concurrency: int = 4,
        relation_selection_mode: str = "direct",
        relation_candidate_k: int = 12,
        relation_max_pairs: int = 60,
        relation_batch_size: int = 10,
        relation_min_confidence: float = 0.65,
        evidence_gams_iterations: int = 128,
        evidence_gams_exploration_weight: float = 0.35,
        evidence_gams_seed: int = 42,
        llm_call_timeout: float = 120.0,
        relation_judge_retries: int = 2,
        **_: Any,
    ):
        self.mode = mode
        self.client = QwenClient.from_config(llm_config) if llm_config else None
        self.grounder = FullTextEvidenceGrounding(
            client=self.client,
            mode=mode,
            cache_dir=cache_dir,
            local_paper_dir=local_paper_dir,
            enable_network_fulltext=enable_network_fulltext,
            embedding_model=embedding_model,
            chunk_chars=chunk_chars,
            chunk_overlap_chars=chunk_overlap_chars,
            retrieve_k=retrieve_k,
            evidence_k=evidence_k,
            min_relevance=min_relevance,
            max_queries=max_queries,
            max_sources=max_sources,
            max_concurrency=grounding_max_concurrency,
            relation_selection_mode=relation_selection_mode,
            relation_candidate_k=relation_candidate_k,
            relation_max_pairs=relation_max_pairs,
            relation_batch_size=relation_batch_size,
            relation_min_confidence=relation_min_confidence,
            evidence_gams_iterations=evidence_gams_iterations,
            evidence_gams_exploration_weight=evidence_gams_exploration_weight,
            evidence_gams_seed=evidence_gams_seed,
            llm_call_timeout=llm_call_timeout,
            relation_judge_retries=relation_judge_retries,
        )

    async def __call__(
        self, state: PipelineState, config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        all_entries = [e for result in state.literature_results for e in result.knowledge_entries]
        if not all_entries:
            return {"evidence_graph": EvidenceGraph(), "literature_results": state.literature_results}

        graph = self._build_seed_graph(all_entries)
        try:
            grounded = await self.grounder.run(state)
            graph = self._merge_grounding(
                graph,
                grounded.get("sources", []),
                grounded.get("chunks", []),
                grounded.get("evidence_records", []),
                grounded.get("claims", []),
                grounded.get("relations", []),
                grounded.get("report"),
            )
            literature_results = self._append_grounded_entries(
                state.literature_results,
                grounded.get("evidence_records", []),
                grounded.get("claims", []),
                grounded.get("relations", []),
            )
        except Exception as exc:
            logger.exception("M3 full-text grounding failed; retaining seed graph")
            graph.grounding_report = {"warnings": [f"Grounding subgraph failed: {exc}"]}
            literature_results = state.literature_results

        if state.memory_cache_dir:
            self._persist_graph(graph, state.memory_cache_dir)
        return {"evidence_graph": graph, "literature_results": literature_results}

    @staticmethod
    def _entry_node_type(entry: KnowledgeEntry) -> EvidenceNodeType:
        return {
            KnowledgeEntryType.ESTABLISHED_FACT: EvidenceNodeType.EVIDENCE,
            KnowledgeEntryType.MECHANISTIC_CONCLUSION: EvidenceNodeType.CLAIM,
            KnowledgeEntryType.CONFLICTING_EVIDENCE: EvidenceNodeType.CONFLICT,
            KnowledgeEntryType.METHOD: EvidenceNodeType.EVIDENCE,
            KnowledgeEntryType.KNOWLEDGE_GAP: EvidenceNodeType.LIMITATION,
            KnowledgeEntryType.KEY_ENTITY: EvidenceNodeType.ENTITY,
        }.get(entry.type, EvidenceNodeType.EVIDENCE)

    def _build_seed_graph(self, entries: List[KnowledgeEntry]) -> EvidenceGraph:
        nodes: List[EvidenceNode] = []
        edges: List[EvidenceEdge] = []
        sources: Dict[str, str] = {}
        for entry in entries:
            if entry.source_paper_id and entry.source_paper_id not in sources:
                node_id = "SRC_" + uuid.uuid5(uuid.NAMESPACE_URL, entry.source_paper_id).hex[:16]
                sources[entry.source_paper_id] = node_id
                nodes.append(EvidenceNode(
                    id=node_id, type=EvidenceNodeType.SOURCE,
                    label=entry.source_paper_title or entry.source_paper_id,
                    metadata={"paper_id": entry.source_paper_id, "grounding_status": "m2_seed"},
                ))
        for entry in entries:
            node_id = "N_" + entry.id
            nodes.append(EvidenceNode(
                id=node_id, type=self._entry_node_type(entry), label=entry.content[:180],
                metadata={"entry_id": entry.id, "entry_type": entry.type.value,
                          "confidence": entry.confidence.value if entry.confidence else None,
                          "provenance_level": "m2_seed"},
            ))
            if entry.source_paper_id in sources:
                edges.append(EvidenceEdge(
                    source=sources[entry.source_paper_id], target=node_id,
                    relation=EvidenceEdgeRelation.CONTAINS,
                    rationale="M2 extracted this seed entry from the paper metadata/abstract.",
                ))
        return EvidenceGraph(
            nodes=nodes, edges=edges,
            established_facts=[e.id for e in entries if e.type == KnowledgeEntryType.ESTABLISHED_FACT],
            conflicts=[e.id for e in entries if e.type == KnowledgeEntryType.CONFLICTING_EVIDENCE],
            knowledge_gaps=[e.id for e in entries if e.type == KnowledgeEntryType.KNOWLEDGE_GAP],
        )

    @staticmethod
    def _merge_grounding(
        graph: EvidenceGraph,
        sources: List[PaperSource],
        chunks: List[FullTextChunk],
        records: List[EvidenceRecord],
        claims: List[AtomicClaim],
        relations: List[EvidenceRelation],
        report: Any,
    ) -> EvidenceGraph:
        source_nodes = {
            str(n.metadata.get("paper_id")): n.id
            for n in graph.nodes if n.type == EvidenceNodeType.SOURCE
        }
        selected_chunk_ids = {record.chunk_id for record in records}
        selected_chunks = {chunk.id: chunk for chunk in chunks if chunk.id in selected_chunk_ids}
        record_map = {record.id: record for record in records}
        claim_map = {claim.id: claim for claim in claims}
        for source in sources:
            node_id = source_nodes.get(source.id)
            if node_id:
                node = next(n for n in graph.nodes if n.id == node_id)
                node.metadata.update({
                    "acquisition_status": source.acquisition_status,
                    "acquisition_note": source.acquisition_note,
                    "doi": source.doi, "pmid": source.pmid, "url": source.url,
                    "full_text_path": source.full_text_path,
                })
        for chunk in selected_chunks.values():
            graph.nodes.append(EvidenceNode(
                id=chunk.id, type=EvidenceNodeType.CHUNK,
                label=f"{chunk.section or 'section'} p.{chunk.page or '?'}: {chunk.text[:120]}",
                metadata=chunk.model_dump(exclude={"text"}) | {"text_preview": chunk.text[:500]},
            ))
            if chunk.paper_id in source_nodes:
                graph.edges.append(EvidenceEdge(
                    source=source_nodes[chunk.paper_id], target=chunk.id,
                    relation=EvidenceEdgeRelation.CONTAINS,
                ))
        for record in records:
            graph.nodes.append(EvidenceNode(
                id=record.id, type=EvidenceNodeType.EVIDENCE_RECORD,
                label=record.summary[:180], metadata=record.model_dump(),
            ))
            graph.edges.append(EvidenceEdge(
                source=record.chunk_id, target=record.id,
                relation=EvidenceEdgeRelation.GROUNDS,
                confidence=record.relevance_score / 10,
                rationale="RCS derived from this exact chunk and locator.",
            ))
        for claim in claims:
            graph.nodes.append(EvidenceNode(
                id=claim.id, type=EvidenceNodeType.CLAIM,
                label=claim.statement[:180], metadata=claim.model_dump(),
            ))
        relation_enum = {
            "supports": EvidenceEdgeRelation.SUPPORTS,
            "contradicts": EvidenceEdgeRelation.CONTRADICTS,
            "extends": EvidenceEdgeRelation.EXTENDS,
            "limits": EvidenceEdgeRelation.LIMITS,
            "same_as": EvidenceEdgeRelation.SAME_AS,
            "refines": EvidenceEdgeRelation.REFINES,
        }
        seen_edges = {(e.source, e.target, e.relation.value) for e in graph.edges}
        for rel in relations:
            if rel.source not in record_map and rel.source not in claim_map:
                continue
            if rel.target not in record_map and rel.target not in claim_map:
                continue
            key = (rel.source, rel.target, rel.relation)
            if key not in seen_edges:
                graph.edges.append(EvidenceEdge(
                    source=rel.source, target=rel.target, relation=relation_enum[rel.relation],
                    confidence=rel.confidence, rationale=rel.rationale,
                    evidence_ids=rel.evidence_ids,
                    metadata={
                        "relation_id": rel.id,
                        "source_paper_ids": rel.source_paper_ids,
                        "target_paper_ids": rel.target_paper_ids,
                        "retrieval_score": rel.retrieval_score,
                        "condition_comparability": rel.condition_comparability,
                        "candidate_origin": rel.candidate_origin,
                    },
                ))
                seen_edges.add(key)

        # M4 consumes KnowledgeEntry IDs from these buckets. The companion method
        # appends matching full-text entries to literature_results.
        # M4's existing prompt consumes only the three legacy buckets.  M3 has
        # already filtered records by its relevance threshold (5.0 by default),
        # so retain those accepted full-text records here instead of silently
        # hiding useful medium/high-relevance evidence from the downstream step.
        accepted_records = [record.id for record in records if record.relevance_score >= 5]
        support_sources: Dict[str, set[str]] = {claim.id: set() for claim in claims}
        for relation in relations:
            if relation.relation != "supports" or relation.target not in claim_map:
                continue
            support_sources[relation.target].update(relation.source_paper_ids)
            support_sources[relation.target].update(relation.target_paper_ids)
            for evidence_id in relation.evidence_ids:
                record = record_map.get(evidence_id)
                if record and record.paper_id:
                    support_sources[relation.target].add(record.paper_id)

        contradicted = {
            endpoint
            for relation in relations
            if relation.relation == "contradicts" and relation.confidence >= 0.65
            for endpoint in (relation.source, relation.target)
            if endpoint in claim_map
        }
        established_claims = [
            claim.id for claim in claims
            if len(support_sources.get(claim.id, set())) >= 2 and claim.id not in contradicted
        ]
        under_supported_claims = [
            claim.id for claim in claims
            if len(support_sources.get(claim.id, set())) < 2 and claim.id not in contradicted
        ]
        graph.established_facts = list(dict.fromkeys(
            graph.established_facts + accepted_records + established_claims
        ))
        graph.conflicts = list(dict.fromkeys(graph.conflicts + sorted(contradicted)))
        graph.knowledge_gaps = list(dict.fromkeys(
            graph.knowledge_gaps + under_supported_claims
        ))
        graph.grounding_report = report.model_dump() if report else {}
        return graph

    @staticmethod
    def _append_grounded_entries(
        results: List[LiteratureResult],
        records: List[EvidenceRecord],
        claims: List[AtomicClaim],
        relations: List[EvidenceRelation],
    ) -> List[LiteratureResult]:
        copied = [result.model_copy(deep=True) for result in results]
        if not copied:
            copied = [LiteratureResult(sub_question="full-text grounding")]
        existing = {entry.id for result in copied for entry in result.knowledge_entries}
        paper_titles = {
            entry.source_paper_id: entry.source_paper_title
            for result in copied for entry in result.knowledge_entries
        }
        for record in records:
            if record.id in existing:
                continue
            locator = ", ".join(
                f"{key}={value}" for key, value in record.context.items()
                if key in {"section", "page", "start_char", "end_char"} and value not in {None, ""}
            )
            content = f"{record.summary}\nExact excerpt: {record.excerpt}\nLocator: {locator or record.chunk_id}"
            entry_type = (
                KnowledgeEntryType.CONFLICTING_EVIDENCE
                if "contradict" in record.epistemic_status.lower()
                else KnowledgeEntryType.ESTABLISHED_FACT
            )
            target = max(copied, key=lambda r: len(set(r.sub_question.lower().split()) & set(record.query.lower().split())))
            target.knowledge_entries.append(KnowledgeEntry(
                id=record.id, type=entry_type, content=content,
                confidence=ConfidenceLevel.HIGH if record.relevance_score >= 8 else ConfidenceLevel.MEDIUM,
                source_paper_id=record.paper_id,
                source_paper_title=paper_titles.get(record.paper_id, record.paper_id),
                entities=record.entities,
            ))
            existing.add(record.id)
        conflict_claims = {r.source for r in relations if r.relation == "contradicts"} | {
            r.target for r in relations if r.relation == "contradicts"
        }
        record_by_id = {record.id: record for record in records}
        for claim in claims:
            if claim.id in existing:
                continue
            first_record = record_by_id.get(claim.evidence_record_ids[0]) if claim.evidence_record_ids else None
            copied[0].knowledge_entries.append(KnowledgeEntry(
                id=claim.id,
                type=(KnowledgeEntryType.CONFLICTING_EVIDENCE if claim.id in conflict_claims
                      else KnowledgeEntryType.MECHANISTIC_CONCLUSION),
                content=claim.statement,
                confidence=ConfidenceLevel.HIGH if claim.confidence >= 0.8 else ConfidenceLevel.MEDIUM,
                source_paper_id=first_record.paper_id if first_record else "",
                source_paper_title=paper_titles.get(first_record.paper_id, "") if first_record else "",
                entities=claim.entities,
            ))
            existing.add(claim.id)
        return copied

    @staticmethod
    def _persist_graph(graph: EvidenceGraph, cache_dir: str) -> None:
        from pathlib import Path
        from ..memory.graph_manager import KnowledgeGraphManager
        try:
            KnowledgeGraphManager(cache_dir=Path(cache_dir)).save_from_evidence_graph(graph)
        except Exception as exc:
            logger.warning("M3: failed to persist evidence graph: %s", exc)

    @classmethod
    def get_input_fields(cls) -> List[str]:
        return ["literature_results", "problem_card"]

    @classmethod
    def get_output_fields(cls) -> List[str]:
        return ["evidence_graph", "literature_results"]
