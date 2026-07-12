"""
M3: Evidence Graph Construction.

Takes structured knowledge entries from M2 and builds a typed graph of
claims, evidence, sources, limitations, conflicts, and entities.

In ``mode="rule"`` (default), nodes and ``INVOLVES`` edges are built from
entry metadata only — fast, no API calls, but all cross-entry edges are
limited to the ``INVOLVES`` type.

In ``mode="llm"``, after rule-based construction, Qwen is called to
discover richer relationships: ``SUPPORTS``, ``CONTRADICTS``, ``EXTENDS``,
``LIMITS`` — producing a more semantically rich evidence graph.

Output: ``evidence_graph`` in state.  Optionally persisted to disk via
``KnowledgeGraphManager`` when ``state.memory_cache_dir`` is set.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any, Dict, List, Optional

from ..protocol import ModuleProtocol
from ..prompts.m3_prompts import (
    M3_BATCH_RELATION_SYSTEM_PROMPT,
    M3_BATCH_RELATION_USER_TEMPLATE,
    M3_RELATION_SYSTEM_PROMPT,
    M3_RELATION_USER_TEMPLATE,
)
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
from ..tools.qwen_client import QwenClient

logger = logging.getLogger(__name__)


@ModuleRegistry.register
class M3EvidenceGraph(ModuleProtocol):
    """Build typed evidence graph from structured knowledge entries.

    Two modes:
      - ``"rule"`` — fast, deterministic, no LLM calls (current behaviour).
      - ``"llm"`` — adds Qwen-powered cross-entry relation extraction on top
        of the rule-based graph.  Falls back to rule-only on failure.

    TODO (组员可扩展):
        - **Incremental graph update**: 当前每次 M3 执行都重建整张图；
          可改为增量更新——仅处理 M2 新产出的条目，追加到已有图中。
        - **Entity linking**: 用 UMLS / MeSH / Gene Ontology 做实体归一化，
          将 "Hsp70", "HSP70", "HSPA1A" 映射到同一规范 ID。
        - **Confidence-weighted edges**: Qwen 提取关系时可同时输出置信度分数，
          用于下游假设生成时的证据权重计算。
        - **Graph visualisation export**: 将 EvidenceGraph 导出为 Cytoscape.js
          或 Mermaid 格式，便于前端 Demo 可视化。
        - **Iterative graph refinement**: 在 M6 评审反馈后，回补或修正图中的
          关系边（当前图在首次构建后保持不变）。
    """

    module_name = "m3"
    module_version = "0.2.0"
    description = "Build typed evidence graph + optional LLM relation extraction"

    # ------------------------------------------------------------------
    # Configurable
    # ------------------------------------------------------------------

    def __init__(
        self,
        mode: str = "rule",  # "rule" | "llm"
        llm_config: Optional[Any] = None,
        max_relation_entries: int = 0,
        relation_batch_size: int = 20,
        relation_max_concurrency: int = 4,
        relation_max_tokens: int = 16384,
        enable_cross_batch: bool = True,
        bridge_batch_size: int = 10,
        **kwargs,
    ):
        self.mode = mode
        self.llm_config = llm_config
        self.max_relation_entries = max(0, max_relation_entries)
        self.relation_batch_size = max(1, relation_batch_size)
        self.relation_max_concurrency = max(1, relation_max_concurrency)
        self.relation_max_tokens = max(1024, relation_max_tokens)
        self.enable_cross_batch = enable_cross_batch
        self.bridge_batch_size = max(1, bridge_batch_size)
        self.client = QwenClient.from_config(llm_config) if llm_config else None

    # ------------------------------------------------------------------
    # ModuleProtocol implementation
    # ------------------------------------------------------------------

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
            return {"evidence_graph": EvidenceGraph()}

        # --- Step 1: rule-based graph construction (always runs) ---
        graph = self._build_rule_graph(all_entries)

        # --- Step 2: LLM-enhanced relation extraction (optional) ---
        if self.mode in {"llm", "direct", "api"} and self.client:
            try:
                graph = await self._enhance_with_llm_batched(graph, all_entries)
            except Exception as exc:
                logger.warning("M3 LLM enhancement failed; using rule-only graph: %s", exc)

        # --- persist to disk (if enabled) ---
        if getattr(state, "memory_cache_dir", ""):
            self._persist_graph(graph, state.memory_cache_dir)

        return {"evidence_graph": graph}

    # ------------------------------------------------------------------
    # Step 1 — rule-based graph construction
    # ------------------------------------------------------------------

    def _build_rule_graph(self, all_entries: list) -> EvidenceGraph:
        """Build nodes and ``INVOLVES`` edges from entry metadata.

        This is the deterministic baseline that always runs — no LLM needed.
        """
        nodes: List[EvidenceNode] = []
        edges: List[EvidenceEdge] = []

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

        # Claim / Evidence / … nodes (one per entry)
        for e in all_entries:
            nid = f"N_{e.id}" if e.id else f"N_{uuid.uuid4().hex[:6]}"

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
                metadata={
                    "entry_type": e.type.value,
                    "confidence": e.confidence.value if e.confidence else None,
                },
            ))

            # Edge: source → this node (involves)
            if e.source_paper_id and e.source_paper_id in seen_papers:
                edges.append(EvidenceEdge(
                    source=seen_papers[e.source_paper_id],
                    target=nid,
                    relation=EvidenceEdgeRelation.INVOLVES,
                ))

        # Categorise entries
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

        return EvidenceGraph(
            nodes=nodes,
            edges=edges,
            established_facts=established,
            conflicts=conflicts,
            knowledge_gaps=gaps,
        )

    # ------------------------------------------------------------------
    # Step 2 — LLM-enhanced relation extraction
    # ------------------------------------------------------------------

    async def _enhance_with_llm_batched(
        self,
        graph: EvidenceGraph,
        all_entries: list,
    ) -> EvidenceGraph:
        """Extract semantic relations concurrently in bounded batches."""
        assert self.client is not None

        node_id_map = {
            n.id[2:]: n.id for n in graph.nodes if n.id.startswith("N_")
        }
        relation_entries = all_entries
        if self.max_relation_entries:
            relation_entries = relation_entries[: self.max_relation_entries]
        batches = [
            relation_entries[i:i + self.relation_batch_size]
            for i in range(0, len(relation_entries), self.relation_batch_size)
        ]
        semaphore = asyncio.Semaphore(self.relation_max_concurrency)

        async def run_one(batch: list) -> dict:
            async with semaphore:
                return await self._extract_relation_batch(batch)

        jobs = [("local", batch) for batch in batches]
        if self.enable_cross_batch:
            for left, right in zip(batches, batches[1:]):
                bridge = left[-self.bridge_batch_size:] + right[:self.bridge_batch_size]
                jobs.append(("bridge", bridge))

        results = await asyncio.gather(
            *(run_one(batch) for _, batch in jobs),
            return_exceptions=True,
        )

        added = 0
        successful = 0
        for index, result in enumerate(results):
            job_type, job_entries = jobs[index]
            if isinstance(result, Exception):
                logger.warning("M3 %s relation batch %d failed: %s", job_type, index + 1, result)
                continue
            if not isinstance(result, dict) or result.get("_parse_error"):
                logger.warning("M3 %s relation batch %d returned invalid JSON", job_type, index + 1)
                continue
            successful += 1
            valid_ids = {entry.id for entry in job_entries}
            added += self._merge_relation_edges(
                graph, node_id_map, result.get("edges") or [], valid_ids
            )

        logger.info(
            "M3 batched relation extraction: %d/%d jobs succeeded, %d edges added "
            "(%d local batches, %d bridge batches)",
            successful, len(jobs), added, len(batches), len(jobs) - len(batches),
        )
        return graph

    async def _extract_relation_batch(self, entries: list) -> dict:
        """Call the model for one small batch so JSON stays within budget.

        Uses an edge-only prompt + thinking-disabled mode so that reasoning
        models (Qwen3, DeepSeek-R1, …) reserve the full ``max_tokens``
        budget for the visible JSON output instead of spending it on
        internal chain-of-thought.
        """
        entries_json = [
            {
                "entry_id": entry.id,
                "type": entry.type.value,
                "content": entry.content[:600],
                "entities": entry.entities,
                "source": entry.source_paper_id,
            }
            for entry in entries
        ]
        schema = {
            "type": "object",
            "properties": {
                "edges": {
                    "type": "array",
                    "maxItems": 30,
                    "items": {
                        "type": "object",
                        "properties": {
                            "source": {"type": "string"},
                            "target": {"type": "string"},
                            "relation": {
                                "type": "string",
                                "enum": ["supports", "contradicts", "extends", "limits"],
                            },
                        },
                        "required": ["source", "target", "relation"],
                    },
                },
            },
            "required": ["edges"],
        }
        user_prompt = M3_BATCH_RELATION_USER_TEMPLATE.format(
            knowledge_entries_json=json.dumps(entries_json, ensure_ascii=False, indent=2),
        )
        return await self.client.structured_chat(
            system_prompt=M3_BATCH_RELATION_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            output_schema=schema,
            max_tokens=self.relation_max_tokens,
            temperature=getattr(self.llm_config, "temperature", 0.1),
            disable_thinking=True,
        )

    @staticmethod
    def _merge_relation_edges(
        graph: EvidenceGraph,
        node_id_map: Dict[str, str],
        raw_edges: list,
        valid_ids: set,
    ) -> int:
        """Validate and deduplicate edges returned by one batch."""
        added = 0
        for raw_edge in raw_edges:
            if not isinstance(raw_edge, dict):
                continue
            source = raw_edge.get("source", "")
            target = raw_edge.get("target", "")
            if source not in valid_ids or target not in valid_ids:
                continue
            try:
                relation = EvidenceEdgeRelation(raw_edge.get("relation", ""))
            except ValueError:
                continue
            src_node = node_id_map.get(source, f"N_{source}")
            tgt_node = node_id_map.get(target, f"N_{target}")
            if any(
                edge.source == src_node
                and edge.target == tgt_node
                and edge.relation == relation
                for edge in graph.edges
            ):
                continue
            graph.edges.append(EvidenceEdge(
                source=src_node,
                target=tgt_node,
                relation=relation,
            ))
            added += 1
        return added

    async def _enhance_with_llm(
        self,
        graph: EvidenceGraph,
        all_entries: list,
    ) -> EvidenceGraph:
        """Call Qwen to discover SUPPORTS / CONTRADICTS / EXTENDS / LIMITS edges.

        The LLM sees all knowledge entries (id, type, content, entities) and
        returns a list of cross-entry relationships.  We merge these into the
        existing rule-based graph.
        """
        assert self.client is not None

        # Serialize entries for the prompt — keep it compact
        entries_json = []
        node_id_map: Dict[str, str] = {}  # entry_id → node_id
        for n in graph.nodes:
            # Extract entry_id from node id (N_KE001 → KE001)
            if n.id.startswith("N_"):
                node_id_map[n.id[2:]] = n.id

        # Relation extraction is the expensive part. Keep the deterministic
        # graph complete, but cap the semantic-extraction payload so the model
        # can return valid JSON within its completion budget.
        relation_entries = all_entries[: self.max_relation_entries]
        for e in relation_entries:
            entries_json.append({
                "entry_id": e.id,
                "type": e.type.value,
                "content": e.content[:600],
                "entities": e.entities,
                "source": e.source_paper_id,
            })

        user_prompt = M3_RELATION_USER_TEMPLATE.format(
            knowledge_entries_json=json.dumps(entries_json, ensure_ascii=False, indent=2),
        )

        schema = {
            "type": "object",
            "properties": {
                "edges": {
                    "type": "array",
                    "maxItems": 30,
                    "items": {
                        "type": "object",
                        "properties": {
                            "source": {"type": "string", "description": "entry_id of source"},
                            "target": {"type": "string", "description": "entry_id of target"},
                            "relation": {
                                "type": "string",
                                "enum": ["supports", "contradicts", "extends", "limits"],
                            },
                        },
                        "required": ["source", "target", "relation"],
                    },
                },
                "revised_established_facts": {
                    "type": "array", "items": {"type": "string"},
                },
                "revised_conflicts": {
                    "type": "array", "items": {"type": "string"},
                },
                "revised_knowledge_gaps": {
                    "type": "array", "items": {"type": "string"},
                },
            },
        }

        try:
            result = await self.client.structured_chat(
                system_prompt=M3_RELATION_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                output_schema=schema,
                max_tokens=self.relation_max_tokens,
                temperature=getattr(self.llm_config, "temperature", 0.1),
                disable_thinking=True,
            )
        except Exception:
            logger.exception("M3 LLM call failed")
            return graph

        if not isinstance(result, dict) or result.get("_parse_error"):
            logger.warning("M3 LLM returned unparseable result; keeping rule-based graph")
            return graph

        # --- Merge LLM edges into the graph ---
        llm_edges = result.get("edges") or []
        edge_count = 0
        for raw_edge in llm_edges:
            if not isinstance(raw_edge, dict):
                continue
            src_entry = raw_edge.get("source", "")
            tgt_entry = raw_edge.get("target", "")
            rel_str = raw_edge.get("relation", "")

            # Map entry_id → node_id
            src_node = node_id_map.get(src_entry, f"N_{src_entry}")
            tgt_node = node_id_map.get(tgt_entry, f"N_{tgt_entry}")

            try:
                relation = EvidenceEdgeRelation(rel_str)
            except ValueError:
                continue

            # Avoid duplicate edges
            already_exists = any(
                e.source == src_node and e.target == tgt_node and e.relation == relation
                for e in graph.edges
            )
            if already_exists:
                continue

            graph.edges.append(EvidenceEdge(
                source=src_node,
                target=tgt_node,
                relation=relation,
            ))
            edge_count += 1

        logger.info(
            "M3 LLM enhancement: added %d cross-entry edges (%d nodes, %d total edges)",
            edge_count, len(graph.nodes), len(graph.edges),
        )

        # --- Optionally adopt LLM-revised categorisations ---
        if result.get("revised_established_facts"):
            graph.established_facts = result["revised_established_facts"]
        if result.get("revised_conflicts"):
            graph.conflicts = result["revised_conflicts"]
        if result.get("revised_knowledge_gaps"):
            graph.knowledge_gaps = result["revised_knowledge_gaps"]

        return graph

    # ------------------------------------------------------------------
    # Persistence helper
    # ------------------------------------------------------------------

    @staticmethod
    def _persist_graph(graph: EvidenceGraph, cache_dir: str) -> None:
        """Save the evidence graph to a JSONL-backed persistent store."""
        from pathlib import Path

        from ..memory.graph_manager import KnowledgeGraphManager

        _logger = logging.getLogger(__name__)
        try:
            mgr = KnowledgeGraphManager(cache_dir=Path(cache_dir))
            mgr.save_from_evidence_graph(graph)
            _logger.info(
                "M3: persisted evidence graph (%d nodes, %d edges) to %s",
                len(graph.nodes), len(graph.edges), cache_dir,
            )
        except Exception as exc:
            _logger.warning("M3: failed to persist evidence graph: %s", exc)

    @classmethod
    def get_input_fields(cls) -> List[str]:
        return ["literature_results"]

    @classmethod
    def get_output_fields(cls) -> List[str]:
        return ["evidence_graph"]
