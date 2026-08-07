"""
M3: Evidence Graph Construction.

Takes structured knowledge entries from M2 and builds a typed graph of
claims, evidence, sources, limitations, conflicts, and entities.

In ``mode="rule"`` (default), nodes and ``INVOLVES`` edges are built from
entry metadata only — fast, no API calls, but all cross-entry edges are
limited to the ``INVOLVES`` type.

In ``mode="llm"``, after rule-based construction, Qwen is called to
discover richer relationships: ``SUPPORTS``, ``CONTRADICTS``, ``EXTENDS``,
``LIMITS`` — with confidence scores and rationales.

When ``grounding_enabled=True``, the M3 grounding workflow is run after the
rule-based + LLM graph construction.  It consumes ``M2KnowledgeExport``
from the agentic M2 pipeline and adds grounded claims + evidence relations
to the graph without re-downloading papers.

.. versionchanged:: 0.4.0
    - Incremental graph update (new entries appended, existing graph reused)
    - Entity normalisation (synonym map + case folding)
    - Confidence-weighted LLM edges (confidence + rationale in schema)
    - Mermaid diagram export (``evidence_graph_to_mermaid``)
    - Iterative graph refinement from M6 review feedback
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Set

from ..memory.hypoforge_types import GapGain, stable_entry_id
from ..protocol import ModuleProtocol
from ..observability import emit_event
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
    GroundingReport,
    KnowledgeEntryType,
    PipelineState,
)
from ..tools.qwen_client import QwenClient
from ..vocabulary import (
    add_runtime_alias,
    legacy_synonym_map,
    normalize_entity,
)
from .m3_grounding import GroundingWorkflow
from .m3_grounding.models import (
    AtomicClaim,
    EvidenceRecord,
    EvidenceRelation,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Entity synonym map — normalises common biomedical name variants
# ============================================================================

# Backward-compatible view.  The source of truth now lives in
# ``hypoforge/data/science125_vocabulary.json`` and spans all 12 Science-125
# domains.  Related (but non-equivalent) terms are intentionally excluded.
ENTITY_SYNONYMS: Dict[str, Set[str]] = legacy_synonym_map()


def add_entity_synonym(canonical: str, variant: str) -> None:
    """Register a new entity synonym at runtime (no restart needed)."""
    canonical_lower = canonical.strip().lower()
    variant_lower = variant.strip().lower()
    ENTITY_SYNONYMS.setdefault(canonical_lower, set()).add(variant_lower)
    add_runtime_alias(canonical_lower, variant_lower)


# ============================================================================
# Mermaid export
# ============================================================================

_RELATION_STYLE: Dict[str, str] = {
    "supports": "-->|supports|",
    "contradicts": "-->|contradicts|",
    "extends": "-->|extends|",
    "limits": "-->|limits|",
    "involves": "-->|involves|",
    "same_as": "-->|same_as|",
    "refines": "-->|refines|",
}

_NODE_SHAPE: Dict[str, tuple[str, str]] = {
    "claim": ("[", "]"),
    "evidence": ("(", ")"),
    "source": ("{", "}"),
    "limitation": ("[/", "/]"),
    "conflict": ("{{", "}}"),
    "entity": ("[(", ")]"),
}


def _mermaid_safe(text: str, max_len: int = 60) -> str:
    """Escape Mermaid-unfriendly characters and truncate."""
    safe = text.replace('"', "'").replace("\n", " ").replace("\r", "")
    return safe[:max_len] + ("…" if len(safe) > max_len else "")


def evidence_graph_to_mermaid(graph: EvidenceGraph, title: str = "Evidence Graph") -> str:
    """Export an EvidenceGraph as a Mermaid flowchart diagram.

    Returns a Mermaid string suitable for embedding in markdown (`` ```mermaid``)
    or rendering via https://mermaid.live.

    Node shapes encode type:
      - claim: [rectangle]
      - evidence: (rounded)
      - source: {hexagon}
      - limitation: [/parallelogram/]
      - conflict: {{double-brace}}
      - entity: [(cylinder)]
    """
    lines: List[str] = [
        "flowchart LR",
        f"    title[{_mermaid_safe(title)}]",
        "",
        "    %% ── Nodes ──",
    ]

    id_map: Dict[str, str] = {}
    for idx, node in enumerate(graph.nodes):
        safe_id = re.sub(r"[^a-zA-Z0-9_]", "_", node.id)
        id_map[node.id] = safe_id
        label = _mermaid_safe(node.label or node.id, 50)
        open_shape, close_shape = _NODE_SHAPE.get(node.type.value, ("[", "]"))
        lines.append(
            f"    {safe_id}{open_shape}\"{label}\"{close_shape}"
        )

    lines.append("")
    lines.append("    %% ── Edges ──")

    for edge in graph.edges:
        src = id_map.get(edge.source, edge.source)
        tgt = id_map.get(edge.target, edge.target)
        rel_str = _RELATION_STYLE.get(edge.relation.value, "-->")
        label_parts: List[str] = []
        if edge.confidence is not None:
            label_parts.append(f"c={edge.confidence:.2f}")
        if edge.rationale:
            label_parts.append(_mermaid_safe(edge.rationale, 30))
        if label_parts:
            lines.append(f"    {src} {rel_str} {tgt}")
            lines.append(f"    %%      {' | '.join(label_parts)}")
        else:
            lines.append(f"    {src} {rel_str} {tgt}")

    lines.append("")
    lines.append("    %% ── Buckets ──")
    if graph.established_facts:
        facts = ", ".join(graph.established_facts[:10])
        lines.append(f"    %% established: {facts}")
    if graph.conflicts:
        conflicts = ", ".join(graph.conflicts[:10])
        lines.append(f"    %% conflicts: {conflicts}")
    if graph.knowledge_gaps:
        gaps = ", ".join(graph.knowledge_gaps[:10])
        lines.append(f"    %% gaps: {gaps}")

    return "\n".join(lines)


# ============================================================================
# M3EvidenceGraph Module
# ============================================================================


@ModuleRegistry.register
class M3EvidenceGraph(ModuleProtocol):
    """Build typed evidence graph from structured knowledge entries.

    Two modes (plus optional grounding):
      - ``"rule"`` — fast, deterministic, no LLM calls (current behaviour).
      - ``"llm"`` — adds Qwen-powered cross-entry relation extraction on top
        of the rule-based graph.  Falls back to rule-only on failure.
      - ``grounding_enabled=True`` — runs the M3 grounding workflow
        (consuming ``M2KnowledgeExport``) and merges grounded claims +
        relations into the evidence graph.

    .. versionchanged:: 0.4.0
        - Incremental graph update: existing ``evidence_graph`` in state
          is reused; only new entries trigger node/edge construction.
        - Entity normalisation: ``normalize_entity()`` maps common gene/
          protein name variants to canonical forms.
        - Confidence-weighted edges: LLM extraction now requests
          ``confidence`` and ``rationale`` per edge.
        - Mermaid export: ``evidence_graph_to_mermaid(graph)`` produces
          a flowchart diagram string.
        - Iterative refinement: M6 reviews are used to prompt the LLM
          for graph corrections when ``iteration_count > 0``.
    """

    module_name = "m3"
    module_version = "0.4.0"
    description = (
        "Build typed evidence graph + optional LLM relation extraction "
        "+ optional M3 grounding + incremental update + entity linking"
    )

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
        # ---- grounding settings ----
        grounding_enabled: bool = False,
        grounding_mode: str = "rule",
        grounding_cache_dir: str = ".hypoforge_cache/m3_grounding",
        grounding_embedding_model: str = "",
        grounding_retrieve_k: int = 30,
        grounding_evidence_k: int = 12,
        grounding_min_relevance: float = 5.0,
        grounding_max_queries: int = 16,
        grounding_max_evidence_items: int = 200,
        grounding_max_concurrency: int = 4,
        grounding_relation_selection_mode: str = "direct",
        grounding_relation_candidate_k: int = 12,
        grounding_relation_max_pairs: int = 60,
        grounding_relation_batch_size: int = 10,
        grounding_relation_min_confidence: float = 0.65,
        grounding_gams_iterations: int = 128,
        grounding_gams_exploration_weight: float = 0.35,
        grounding_gams_seed: int = 42,
        grounding_llm_call_timeout: float = 120.0,
        grounding_relation_judge_retries: int = 2,
        # ---- knowledge-retention settings ----
        gap_no_improvement_limit: int = 1,
        output_dir: str = "",
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

        # Gap confirmation: how many no-improvement rounds before a
        # pending_grounding gap is declared unimprovable.
        self.gap_no_improvement_limit = max(1, int(gap_no_improvement_limit))
        # Run output dir for lossless graph-round snapshots (falls back to
        # state.memory_cache_dir when unset).
        self.output_dir = output_dir

        # Grounding workflow (lazy — only built when enabled)
        self.grounding_enabled = grounding_enabled
        self._grounder: Optional[GroundingWorkflow] = None
        if grounding_enabled:
            self._grounder = GroundingWorkflow(
                client=self.client,
                mode=grounding_mode if grounding_mode in {"rule", "llm", "api", "direct"} else mode,
                cache_dir=grounding_cache_dir,
                embedding_model=grounding_embedding_model,
                retrieve_k=grounding_retrieve_k,
                evidence_k=grounding_evidence_k,
                min_relevance=grounding_min_relevance,
                max_queries=grounding_max_queries,
                max_evidence_items=grounding_max_evidence_items,
                max_concurrency=grounding_max_concurrency,
                relation_selection_mode=grounding_relation_selection_mode,
                relation_candidate_k=grounding_relation_candidate_k,
                relation_max_pairs=grounding_relation_max_pairs,
                relation_batch_size=grounding_relation_batch_size,
                relation_min_confidence=grounding_relation_min_confidence,
                evidence_gams_iterations=grounding_gams_iterations,
                evidence_gams_exploration_weight=grounding_gams_exploration_weight,
                evidence_gams_seed=grounding_gams_seed,
                llm_call_timeout=grounding_llm_call_timeout,
                relation_judge_retries=grounding_relation_judge_retries,
            )

        # NOTE: incremental dedup is rebuilt on the fly from the existing
        # graph's node ids / metadata each run — no instance-level id sets.

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

        existing_graph = state.evidence_graph

        if not all_entries:
            # No knowledge entries at all: never wipe an existing graph.
            # Only return a genuinely empty graph when there is nothing.
            if existing_graph is not None and existing_graph.nodes:
                logger.info(
                    "M3: no knowledge entries; reusing existing graph "
                    "(%d nodes, %d edges)",
                    len(existing_graph.nodes), len(existing_graph.edges),
                )
                emit_event(
                    "module_notice",
                    module="m3",
                    status="completed",
                    message=(
                        "无知识条目，保留既有证据图（"
                        f"{len(existing_graph.nodes)} 节点 / "
                        f"{len(existing_graph.edges)} 边），不清空重建"
                    ),
                )
                return {"evidence_graph": existing_graph}
            # State carries no graph — fall back to the persisted one
            # (lossless round snapshot first, then the JSONL memory store).
            persisted = self._load_persisted_graph(state)
            if persisted is not None and persisted.nodes:
                logger.info(
                    "M3: no knowledge entries; reusing persisted graph "
                    "(%d nodes, %d edges)",
                    len(persisted.nodes), len(persisted.edges),
                )
                emit_event(
                    "module_notice",
                    module="m3",
                    status="completed",
                    message=(
                        "无知识条目，从持久化记忆恢复证据图（"
                        f"{len(persisted.nodes)} 节点 / "
                        f"{len(persisted.edges)} 边）"
                    ),
                )
                return {"evidence_graph": persisted}
            return {"evidence_graph": EvidenceGraph()}

        # --- Step 1: rule-based graph construction (with incremental update) ---
        rule_started_at = time.monotonic()
        emit_event(
            "tool_started",
            module="m3",
            tool="rule_graph_builder",
            status="running",
            message=f"开始将 {len(all_entries)} 条知识构造成基础证据图",
        )

        new_entries: list = []
        if existing_graph is not None and existing_graph.nodes:
            new_entries = self._find_new_entries(all_entries, existing_graph)
            if not new_entries:
                # No new entries: skip rule construction and LLM enhancement
                # below, but still run grounding merge / gap confirmation /
                # persistence so the round completes consistently.
                logger.info(
                    "M3 incremental: no new entries; reusing existing graph "
                    "(%d nodes, %d edges)",
                    len(existing_graph.nodes), len(existing_graph.edges),
                )
                graph = existing_graph
                emit_event(
                    "tool_completed",
                    module="m3",
                    tool="rule_graph_builder",
                    status="completed",
                    message=f"增量更新：无新条目，复用现有图 ({len(graph.nodes)} 节点 / {len(graph.edges)} 边)",
                    elapsed_seconds=time.monotonic() - rule_started_at,
                )
            else:
                logger.info(
                    "M3 incremental: %d new entries; appending to existing graph "
                    "(%d nodes, %d edges)",
                    len(new_entries), len(existing_graph.nodes), len(existing_graph.edges),
                )
                graph = self._build_rule_graph(new_entries)
                graph = self._merge_graphs(existing_graph, graph)
                emit_event(
                    "tool_completed",
                    module="m3",
                    tool="rule_graph_builder",
                    status="completed",
                    message=f"基础证据图完成：{len(graph.nodes)} 节点 / {len(graph.edges)} 边",
                    elapsed_seconds=time.monotonic() - rule_started_at,
                )
        else:
            new_entries = list(all_entries)
            graph = self._build_rule_graph(all_entries)
            emit_event(
                "tool_completed",
                module="m3",
                tool="rule_graph_builder",
                status="completed",
                message=f"基础证据图完成：{len(graph.nodes)} 节点 / {len(graph.edges)} 边",
                elapsed_seconds=time.monotonic() - rule_started_at,
            )

        # --- Step 2: LLM-enhanced relation extraction (optional) ---
        # Only runs when this round contributed new entries, and then only
        # over the new entries — old entries never get re-extracted.
        if new_entries and self.mode in {"llm", "direct", "api"} and self.client:
            llm_started_at = time.monotonic()
            emit_event(
                "tool_started",
                module="m3",
                tool="qwen_relation_extractor",
                status="running",
                message="开始抽取跨知识条目的语义关系",
            )
            try:
                graph = await self._enhance_with_llm_batched(graph, new_entries)
            except Exception as exc:
                logger.warning(
                    "M3 LLM enhancement failed; using rule-only graph: %s", exc
                )
                emit_event(
                    "tool_failed",
                    module="m3",
                    tool="qwen_relation_extractor",
                    status="failed",
                    message=f"语义关系抽取失败，保留规则图：{type(exc).__name__}: {exc}",
                    elapsed_seconds=time.monotonic() - llm_started_at,
                )
            else:
                emit_event(
                    "tool_completed",
                    module="m3",
                    tool="qwen_relation_extractor",
                    status="completed",
                    message=f"语义关系抽取完成：证据图现有 {len(graph.edges)} 条边",
                    elapsed_seconds=time.monotonic() - llm_started_at,
                )

        # --- Step 3: Iterative refinement from M6 reviews ---
        if (
            state.iteration_count > 0
            and state.reviews
            and self.client
            and self.mode in {"llm", "direct", "api"}
        ):
            try:
                graph = await self._refine_with_reviews(graph, state)
            except Exception as exc:
                logger.warning(
                    "M3 iterative refinement failed; keeping unrefined graph: %s", exc
                )

        # --- Step 4: M3 grounding (optional — consumes M2KnowledgeExport) ---
        result: Dict[str, Any] = {"evidence_graph": graph}
        if self.grounding_enabled and self._grounder is not None:
            if state.m2_knowledge_export is None:
                logger.warning(
                    "M3 grounding is enabled but m2_knowledge_export is None. "
                    "Grounding requires the agentic M2 pipeline "
                    "(search.implementation='agentic'). Skipping grounding."
                )
                result["errors"] = state.errors + [
                    "[m3] grounding.enabled=True but no m2_knowledge_export "
                    "available — run the agentic M2 pipeline first."
                ]
            else:
                try:
                    grounded = await self._grounder.run(state)
                    graph = self._merge_grounding(
                        graph,
                        grounded.get("evidence_records", []),
                        grounded.get("claims", []),
                        grounded.get("relations", []),
                        grounded.get("report"),
                    )
                    result["evidence_graph"] = graph
                except Exception as exc:
                    logger.exception(
                        "M3 grounding workflow failed; retaining non-grounded graph: %s",
                        exc,
                    )

        # Track existing IDs for next incremental run
        # (rebuilt on the fly from the graph itself — no instance state).

        # --- Step 5: pending_grounding gap confirmation ---
        new_entry_ids = {e.id for e in new_entries if e.id}
        gap_gain = self._compute_gap_gain(state, new_entry_ids)
        gap_updates = self._evaluate_pending_gaps(state, gap_gain)
        if gap_updates is not None:
            result["evidence_gaps"] = gap_updates

        # CONTRACT (P2): per-gap new-effective-evidence counts
        # (dict[gap_id, int]).  PipelineState has no dedicated ``gap_gain``
        # field and pipeline.py rejects unknown patch keys, so the counts
        # travel inside the existing ``metrics`` dict under ``m3_gap_gain``.
        # P2 consumers read ``state.metrics["m3_gap_gain"]``.
        if state.evidence_gaps:
            result["metrics"] = {**state.metrics, "m3_gap_gain": gap_gain}

        # --- persist to disk (merge semantics) + lossless round snapshot ---
        if getattr(state, "memory_cache_dir", ""):
            self._persist_graph(graph, state.memory_cache_dir)
        snapshot_dir = self.output_dir or getattr(state, "memory_cache_dir", "")
        if snapshot_dir:
            self._write_round_snapshot(graph, snapshot_dir, state)

        return result

    # ------------------------------------------------------------------
    # Incremental update helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_new_entries(
        all_entries: list,
        existing_graph: EvidenceGraph,
    ) -> list:
        """Return entries whose IDs are not already represented in the graph.

        The id set is rebuilt locally from the graph every call (no
        module-level mutable state), so a fresh module instance — or a
        resumed run — deduplicates exactly as well as a warm one.
        """
        existing_entry_ids: Set[str] = set()
        for node in existing_graph.nodes:
            # Rule-built nodes have id "N_{entry_id}"
            if node.id.startswith("N_"):
                existing_entry_ids.add(node.id[2:])
            # Also check metadata for grounding nodes
            entry_id = (node.metadata or {}).get("entry_id", "")
            if entry_id:
                existing_entry_ids.add(entry_id)
        return [e for e in all_entries if e.id not in existing_entry_ids]

    @staticmethod
    def _merge_graphs(base: EvidenceGraph, additions: EvidenceGraph) -> EvidenceGraph:
        """Merge *additions* into *base*, deduplicating nodes and edges."""
        existing_node_ids = {n.id for n in base.nodes}
        for node in additions.nodes:
            if node.id not in existing_node_ids:
                base.nodes.append(node)
                existing_node_ids.add(node.id)

        existing_edges = {
            (e.source, e.target, e.relation.value) for e in base.edges
        }
        for edge in additions.edges:
            key = (edge.source, edge.target, edge.relation.value)
            if key not in existing_edges:
                base.edges.append(edge)
                existing_edges.add(key)

        # Merge bucket lists (deduplicate)
        base.established_facts = list(dict.fromkeys(
            base.established_facts + additions.established_facts
        ))
        base.conflicts = list(dict.fromkeys(
            base.conflicts + additions.conflicts
        ))
        base.knowledge_gaps = list(dict.fromkeys(
            base.knowledge_gaps + additions.knowledge_gaps
        ))
        return base

    # ------------------------------------------------------------------
    # Step 1 — rule-based graph construction
    # ------------------------------------------------------------------

    def _build_rule_graph(self, all_entries: list) -> EvidenceGraph:
        """Build nodes and ``INVOLVES`` edges from entry metadata.

        Entities are normalised through ``normalize_entity()`` so that
        "Hsp70", "HSP70", and "HSPA1A" all map to the same canonical name.
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

        # Track entity nodes to avoid duplicates
        seen_entities: Dict[str, str] = {}

        # Claim / Evidence / … nodes (one per entry)
        for e in all_entries:
            # Stable node id: entries arriving without an id get the
            # canonical content-based id (same algorithm M2 uses) instead
            # of a random uuid, so re-runs stay dedup-compatible.
            entry_id = e.id or stable_entry_id(e.content, e.source_paper_id)
            nid = f"N_{entry_id}"

            node_type = {
                KnowledgeEntryType.ESTABLISHED_FACT: EvidenceNodeType.EVIDENCE,
                KnowledgeEntryType.MECHANISTIC_CONCLUSION: EvidenceNodeType.CLAIM,
                KnowledgeEntryType.CONFLICTING_EVIDENCE: EvidenceNodeType.CONFLICT,
                KnowledgeEntryType.METHOD: EvidenceNodeType.EVIDENCE,
                KnowledgeEntryType.KNOWLEDGE_GAP: EvidenceNodeType.LIMITATION,
                KnowledgeEntryType.KEY_ENTITY: EvidenceNodeType.ENTITY,
            }.get(e.type, EvidenceNodeType.EVIDENCE)

            # Normalise entities in metadata
            normalised_entities = [
                normalize_entity(name) for name in e.entities
            ]

            nodes.append(EvidenceNode(
                id=nid,
                type=node_type,
                label=e.content[:120],
                metadata={
                    "entry_type": e.type.value,
                    "confidence": e.confidence.value if e.confidence else None,
                    "entities": normalised_entities,
                },
            ))

            # Edge: source → this node (involves)
            if e.source_paper_id and e.source_paper_id in seen_papers:
                edges.append(EvidenceEdge(
                    source=seen_papers[e.source_paper_id],
                    target=nid,
                    relation=EvidenceEdgeRelation.INVOLVES,
                ))

            # Entity nodes + involves edges (deduplicated)
            for name in normalised_entities:
                if name not in seen_entities:
                    eid = f"ENT_{name}"
                    seen_entities[name] = eid
                    nodes.append(EvidenceNode(
                        id=eid,
                        type=EvidenceNodeType.ENTITY,
                        label=name,
                        metadata={"normalised_from": e.entities},
                    ))
                entity_nid = seen_entities[name]
                # Avoid duplicate entity→entry edges
                ent_edge_key = (entity_nid, nid, EvidenceEdgeRelation.INVOLVES.value)
                if not any(
                    eg.source == entity_nid and eg.target == nid
                    and eg.relation == EvidenceEdgeRelation.INVOLVES
                    for eg in edges
                ):
                    edges.append(EvidenceEdge(
                        source=entity_nid,
                        target=nid,
                        relation=EvidenceEdgeRelation.INVOLVES,
                    ))

        # Categorise entries
        established = []
        conflicts = []
        gaps = []
        for e in all_entries:
            entry_id = e.id or stable_entry_id(e.content, e.source_paper_id)
            if e.type == KnowledgeEntryType.ESTABLISHED_FACT:
                established.append(entry_id)
            elif e.type == KnowledgeEntryType.CONFLICTING_EVIDENCE:
                conflicts.append(entry_id)
            elif e.type == KnowledgeEntryType.KNOWLEDGE_GAP:
                gaps.append(entry_id)

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
        """Extract semantic relations concurrently in bounded batches.

        .. versionchanged:: 0.4.0
            Edge schema now includes optional ``confidence`` (0–1) and
            ``rationale`` (string) fields for confidence-weighted edges.
        """
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
                logger.warning(
                    "M3 %s relation batch %d failed: %s",
                    job_type, index + 1, result,
                )
                continue
            if not isinstance(result, dict) or result.get("_parse_error"):
                logger.warning(
                    "M3 %s relation batch %d returned invalid JSON",
                    job_type, index + 1,
                )
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
        """Call the model for one small batch.

        .. versionchanged:: 0.4.0
            Schema now requests optional ``confidence`` (0–1) and
            ``rationale`` per edge.  Model output is validated and
            confidence scores are attached to the resulting
            ``EvidenceEdge`` objects.
        """
        entries_json = [
            {
                "entry_id": entry.id,
                "type": entry.type.value,
                "content": entry.content[:600],
                "entities": [normalize_entity(name) for name in entry.entities],
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
                                "enum": [
                                    "supports", "contradicts",
                                    "extends", "limits",
                                ],
                            },
                            "confidence": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                                "description": "How confident the model is that this edge is correct",
                            },
                            "rationale": {
                                "type": "string",
                                "description": "One-sentence justification for this edge",
                            },
                        },
                        "required": ["source", "target", "relation"],
                    },
                },
            },
        }
        user_prompt = M3_BATCH_RELATION_USER_TEMPLATE.format(
            knowledge_entries_json=json.dumps(
                entries_json, ensure_ascii=False, indent=2,
            ),
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
        """Validate and deduplicate edges returned by one batch.

        .. versionchanged:: 0.4.0
            Attaches ``confidence`` and ``rationale`` from the LLM output
            when the fields are present and valid.
        """
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
            # ---- confidence + rationale (optional, v0.4.0) ----
            confidence: Optional[float] = None
            raw_conf = raw_edge.get("confidence")
            if isinstance(raw_conf, (int, float)) and 0 <= raw_conf <= 1:
                confidence = float(raw_conf)
            rationale: Optional[str] = None
            raw_rationale = raw_edge.get("rationale", "")
            if isinstance(raw_rationale, str) and raw_rationale.strip():
                rationale = raw_rationale.strip()[:500]
            graph.edges.append(EvidenceEdge(
                source=src_node,
                target=tgt_node,
                relation=relation,
                confidence=confidence,
                rationale=rationale,
            ))
            added += 1
        return added

    async def _enhance_with_llm(
        self,
        graph: EvidenceGraph,
        all_entries: list,
    ) -> EvidenceGraph:
        """Call Qwen to discover SUPPORTS / CONTRADICTS / EXTENDS / LIMITS edges.

        .. versionchanged:: 0.4.0
            Schema extended with optional ``confidence`` and ``rationale``.
        """
        assert self.client is not None

        entries_json = []
        node_id_map: Dict[str, str] = {}
        for n in graph.nodes:
            if n.id.startswith("N_"):
                node_id_map[n.id[2:]] = n.id

        relation_entries = all_entries[: self.max_relation_entries]
        for e in relation_entries:
            entries_json.append({
                "entry_id": e.id,
                "type": e.type.value,
                "content": e.content[:600],
                "entities": [normalize_entity(name) for name in e.entities],
                "source": e.source_paper_id,
            })

        user_prompt = M3_RELATION_USER_TEMPLATE.format(
            knowledge_entries_json=json.dumps(
                entries_json, ensure_ascii=False, indent=2,
            ),
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
                            "source": {
                                "type": "string",
                                "description": "entry_id of source",
                            },
                            "target": {
                                "type": "string",
                                "description": "entry_id of target",
                            },
                            "relation": {
                                "type": "string",
                                "enum": [
                                    "supports", "contradicts",
                                    "extends", "limits",
                                ],
                            },
                            "confidence": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                            },
                            "rationale": {"type": "string"},
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
            logger.warning(
                "M3 LLM returned unparseable result; keeping rule-based graph"
            )
            return graph

        llm_edges = result.get("edges") or []
        edge_count = 0
        for raw_edge in llm_edges:
            if not isinstance(raw_edge, dict):
                continue
            src_entry = raw_edge.get("source", "")
            tgt_entry = raw_edge.get("target", "")
            rel_str = raw_edge.get("relation", "")

            src_node = node_id_map.get(src_entry, f"N_{src_entry}")
            tgt_node = node_id_map.get(tgt_entry, f"N_{tgt_entry}")

            try:
                relation = EvidenceEdgeRelation(rel_str)
            except ValueError:
                continue

            already_exists = any(
                e.source == src_node
                and e.target == tgt_node
                and e.relation == relation
                for e in graph.edges
            )
            if already_exists:
                continue

            confidence: Optional[float] = None
            raw_conf = raw_edge.get("confidence")
            if isinstance(raw_conf, (int, float)) and 0 <= raw_conf <= 1:
                confidence = float(raw_conf)
            rationale: Optional[str] = None
            raw_rationale = raw_edge.get("rationale", "")
            if isinstance(raw_rationale, str) and raw_rationale.strip():
                rationale = raw_rationale.strip()[:500]

            graph.edges.append(EvidenceEdge(
                source=src_node,
                target=tgt_node,
                relation=relation,
                confidence=confidence,
                rationale=rationale,
            ))
            edge_count += 1

        logger.info(
            "M3 LLM enhancement: added %d cross-entry edges "
            "(%d nodes, %d total edges)",
            edge_count, len(graph.nodes), len(graph.edges),
        )

        if result.get("revised_established_facts"):
            graph.established_facts = result["revised_established_facts"]
        if result.get("revised_conflicts"):
            graph.conflicts = result["revised_conflicts"]
        if result.get("revised_knowledge_gaps"):
            graph.knowledge_gaps = result["revised_knowledge_gaps"]

        return graph

    # ------------------------------------------------------------------
    # Step 3 — iterative refinement from M6 reviews
    # ------------------------------------------------------------------

    async def _refine_with_reviews(
        self,
        graph: EvidenceGraph,
        state: PipelineState,
    ) -> EvidenceGraph:
        """Use M6 review feedback to correct or augment the evidence graph.

        Only runs when ``iteration_count > 0`` (i.e., M6 has already
        produced reviews in a previous iteration).  The LLM receives the
        current graph summary + reviewer critiques and returns a list of
        edge additions, removals, or re-classifications.
        """
        assert self.client is not None

        # Summarise the current graph
        node_summary = []
        for n in graph.nodes[:80]:
            node_summary.append(
                f"  {n.id} [{n.type.value}] {n.label[:100]}"
            )
        edge_summary = []
        for e in graph.edges[:120]:
            conf_str = f" c={e.confidence:.2f}" if e.confidence is not None else ""
            rat_str = f" ({e.rationale[:60]})" if e.rationale else ""
            edge_summary.append(
                f"  {e.source} --[{e.relation.value}{conf_str}]--> {e.target}{rat_str}"
            )

        review_text = "\n".join(
            f"[{r.dimension.value}] score={r.score:.1f} "
            f"comments={r.comments[:200]} suggestions={r.suggestions[:200]}"
            for r in state.reviews[-4:]  # last 4 reviews
        )

        schema = {
            "type": "object",
            "properties": {
                "add_edges": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source": {"type": "string"},
                            "target": {"type": "string"},
                            "relation": {
                                "type": "string",
                                "enum": [
                                    "supports", "contradicts",
                                    "extends", "limits",
                                ],
                            },
                            "confidence": {
                                "type": "number", "minimum": 0, "maximum": 1,
                            },
                            "rationale": {"type": "string"},
                        },
                        "required": ["source", "target", "relation"],
                    },
                },
                "remove_edges": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source": {"type": "string"},
                            "target": {"type": "string"},
                            "reason": {"type": "string"},
                        },
                        "required": ["source", "target"],
                    },
                },
                "reclassify_edges": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source": {"type": "string"},
                            "target": {"type": "string"},
                            "new_relation": {
                                "type": "string",
                                "enum": [
                                    "supports", "contradicts",
                                    "extends", "limits",
                                ],
                            },
                            "rationale": {"type": "string"},
                        },
                        "required": ["source", "target", "new_relation"],
                    },
                },
                "refinement_summary": {"type": "string"},
            },
        }

        prompt = (
            "You are reviewing an evidence graph that was used to generate "
            "scientific hypotheses.  Reviewer feedback suggests improvements. "
            "Your task: suggest concrete edge corrections to improve the graph.\n\n"
            f"## Review Feedback\n{review_text}\n\n"
            f"## Current Graph Nodes ({len(graph.nodes)} total, showing first 80)\n"
            + "\n".join(node_summary)
            + f"\n\n## Current Graph Edges ({len(graph.edges)} total, showing first 120)\n"
            + "\n".join(edge_summary)
            + "\n\nSuggest edge additions, removals, and reclassifications "
            "based on the reviewer feedback.  Be conservative — only suggest "
            "changes directly motivated by reviewer comments."
        )

        try:
            result = await self.client.structured_chat(
                system_prompt=(
                    "You are a biomedical knowledge graph curator. "
                    "Given reviewer feedback, suggest targeted corrections "
                    "to an evidence graph. Be precise: use exact node IDs. "
                    "Only suggest changes with clear scientific justification."
                ),
                user_prompt=prompt,
                output_schema=schema,
                max_tokens=8192,
                temperature=0.1,
                disable_thinking=True,
            )
        except Exception:
            logger.exception("M3 refinement LLM call failed")
            return graph

        if not isinstance(result, dict) or result.get("_parse_error"):
            return graph

        # Apply removals
        remove_keys = {
            (r.get("source", ""), r.get("target", ""))
            for r in (result.get("remove_edges") or [])
            if isinstance(r, dict)
        }
        if remove_keys:
            before = len(graph.edges)
            graph.edges = [
                e for e in graph.edges
                if (e.source, e.target) not in remove_keys
            ]
            logger.info(
                "M3 refinement: removed %d edges", before - len(graph.edges)
            )

        # Apply reclassifications
        reclassify = [
            r for r in (result.get("reclassify_edges") or [])
            if isinstance(r, dict)
        ]
        for rc in reclassify:
            src = rc.get("source", "")
            tgt = rc.get("target", "")
            new_rel_str = rc.get("new_relation", "")
            try:
                new_rel = EvidenceEdgeRelation(new_rel_str)
            except ValueError:
                continue
            rationale = str(rc.get("rationale", "") or "")[:500]
            for edge in graph.edges:
                if edge.source == src and edge.target == tgt:
                    edge.relation = new_rel
                    if rationale:
                        edge.rationale = rationale
                    break
        if reclassify:
            logger.info("M3 refinement: reclassified %d edges", len(reclassify))

        # Apply additions
        add_edges = [
            a for a in (result.get("add_edges") or [])
            if isinstance(a, dict)
        ]
        added = 0
        for ae in add_edges:
            src = ae.get("source", "")
            tgt = ae.get("target", "")
            rel_str = ae.get("relation", "")
            try:
                rel = EvidenceEdgeRelation(rel_str)
            except ValueError:
                continue
            if any(
                e.source == src and e.target == tgt and e.relation == rel
                for e in graph.edges
            ):
                continue
            confidence: Optional[float] = None
            raw_conf = ae.get("confidence")
            if isinstance(raw_conf, (int, float)) and 0 <= raw_conf <= 1:
                confidence = float(raw_conf)
            rationale = str(ae.get("rationale", "") or "")[:500] or None
            graph.edges.append(EvidenceEdge(
                source=src,
                target=tgt,
                relation=rel,
                confidence=confidence,
                rationale=rationale,
            ))
            added += 1
        if added:
            logger.info("M3 refinement: added %d edges", added)

        summary = str(result.get("refinement_summary", "") or "")
        if summary:
            logger.info("M3 refinement summary: %s", summary[:200])

        return graph

    # ------------------------------------------------------------------
    # Step 4 — merge grounding results into the evidence graph
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_grounding(
        graph: EvidenceGraph,
        records: List[EvidenceRecord],
        claims: List[AtomicClaim],
        relations: List[EvidenceRelation],
        report: Any,
    ) -> EvidenceGraph:
        """Merge grounding workflow output into the existing evidence graph.

        Creates CLAIM nodes for each ``AtomicClaim`` and adds provenance
        edges (evidence → claim) and semantic relation edges (claim → claim).
        The ``evidence_ids`` on both claims and relations point back to
        ``M2EvidenceExport.evidence_id``, which is recorded in node metadata.
        """
        record_map = {record.evidence_id: record for record in records}
        claim_map: Dict[str, str] = {}

        # --- Create CLAIM nodes ---
        for claim in claims:
            node_id = "GCLM_" + claim.id.replace("CLM_", "")
            claim_map[claim.id] = node_id
            evidence_paper_ids: List[str] = []
            for eid in claim.evidence_ids:
                rec = record_map.get(eid)
                if rec and rec.paper_id:
                    evidence_paper_ids.append(rec.paper_id)
            graph.nodes.append(EvidenceNode(
                id=node_id,
                type=EvidenceNodeType.CLAIM,
                label=claim.statement[:200],
                metadata={
                    "claim_id": claim.id,
                    "confidence": claim.confidence,
                    "evidence_ids": claim.evidence_ids,
                    "paper_ids": list(dict.fromkeys(evidence_paper_ids)),
                    "entities": [
                        normalize_entity(name)
                        for name in claim.entities
                    ],
                    "provenance_level": "m2_fulltext_grounded",
                },
            ))

        # --- Create EVIDENCE nodes for each EvidenceRecord ---
        for record in records:
            node_id = "GEV_" + record.evidence_id[:20]
            graph.nodes.append(EvidenceNode(
                id=node_id,
                type=EvidenceNodeType.EVIDENCE,
                label=record.summary[:200],
                metadata={
                    "evidence_id": record.evidence_id,
                    "paper_id": record.paper_id,
                    "section": record.section,
                    "page": record.page,
                    "quote": record.quote[:500],
                    "normalized_claim": record.normalized_claim[:500],
                    "relevance_score": record.relevance_score,
                    "provenance_level": "m2_fulltext_grounded",
                },
            ))

        # --- Add provenance edges (evidence → claim) ---
        for claim in claims:
            claim_node = claim_map.get(claim.id)
            if not claim_node:
                continue
            for eid in claim.evidence_ids:
                ev_node = "GEV_" + eid[:20]
                already = any(
                    e.source == ev_node and e.target == claim_node
                    and e.relation == EvidenceEdgeRelation.SUPPORTS
                    for e in graph.edges
                )
                if not already:
                    graph.edges.append(EvidenceEdge(
                        source=ev_node,
                        target=claim_node,
                        relation=EvidenceEdgeRelation.SUPPORTS,
                        rationale="Atomic claim grounded in this M2 evidence item.",
                        evidence_ids=[eid],
                    ))

        # --- Add semantic relation edges (claim → claim) ---
        for rel in relations:
            source_node = claim_map.get(rel.source)
            target_node = claim_map.get(rel.target)
            if not source_node or not target_node:
                continue
            try:
                edge_rel = EvidenceEdgeRelation(rel.relation)
            except ValueError:
                continue
            already = any(
                e.source == source_node and e.target == target_node
                and e.relation == edge_rel
                for e in graph.edges
            )
            if not already:
                graph.edges.append(EvidenceEdge(
                    source=source_node,
                    target=target_node,
                    relation=edge_rel,
                    confidence=rel.confidence,
                    rationale=rel.rationale,
                    evidence_ids=rel.evidence_ids,
                ))

        logger.info(
            "M3 grounding: merged %d claims, %d evidence records, %d relations "
            "(%d total nodes, %d total edges)",
            len(claims), len(records), len(relations),
            len(graph.nodes), len(graph.edges),
        )

        return graph

    # ------------------------------------------------------------------
    # Step 5 — pending_grounding gap confirmation
    # ------------------------------------------------------------------

    @staticmethod
    def _sub_question_matches(sub_question: str, target: str) -> bool:
        """Tolerant match between a literature sub-question and a gap target."""
        a = " ".join(str(sub_question or "").split()).casefold()
        b = " ".join(str(target or "").split()).casefold()
        if not a or not b:
            return False
        if a == b:
            return True
        return a in b or b in a

    def _compute_gap_gain(
        self,
        state: PipelineState,
        new_entry_ids: Set[str],
    ) -> GapGain:
        """Per-gap count of this round's new effective evidence entries.

        Gain for a gap = number of entries in ``new_entry_ids`` that belong
        to literature results whose sub-question matches the gap's
        ``target_sub_question``.  Every gap in state gets an entry (0 when
        nothing matched) so consumers can rely on total coverage.
        """
        gain: GapGain = {}
        for gap in state.evidence_gaps:
            gain[gap.gap_id] = sum(
                1
                for lr in state.literature_results
                if self._sub_question_matches(
                    lr.sub_question, gap.target_sub_question
                )
                for entry in lr.knowledge_entries
                if entry.id in new_entry_ids
            )
        return gain

    def _evaluate_pending_gaps(
        self,
        state: PipelineState,
        gap_gain: GapGain,
    ) -> Optional[List[Any]]:
        """Confirm ``pending_grounding`` gaps after this M3 round.

        Uses the pre-computed *gap_gain* map (see :meth:`_compute_gap_gain`):

        * gain > 0 → gap closed;
        * gain == 0 → ``attempts += 1``; once ``attempts`` reaches
          ``gap_no_improvement_limit`` the gap becomes ``unimprovable``.

        Returns the full updated ``evidence_gaps`` list (a modified copy),
        or ``None`` when there is nothing pending.
        """
        if not any(
            g.status == "pending_grounding" for g in state.evidence_gaps
        ):
            return None

        updated = [gap.model_copy(deep=True) for gap in state.evidence_gaps]
        limit = self.gap_no_improvement_limit

        for gap in updated:
            if gap.status != "pending_grounding":
                continue
            gain = gap_gain.get(gap.gap_id, 0)
            if gain > 0:
                gap.status = "closed"
                emit_event(
                    "evidence_gap_closed",
                    module="m3",
                    status="completed",
                    message=f"证据缺口已闭合：新增 {gain} 条有效证据",
                    details={
                        "gap_id": gap.gap_id,
                        "gain": gain,
                        "status": "closed",
                    },
                )
            else:
                gap.attempts += 1
                if gap.attempts >= limit:
                    gap.status = "unimprovable"
                emit_event(
                    "evidence_gap_status_changed",
                    module="m3",
                    status="completed",
                    message=(
                        f"证据缺口无新增证据（attempts={gap.attempts}，"
                        f"status={gap.status}）"
                    ),
                    details={
                        "gap_id": gap.gap_id,
                        "gain": 0,
                        "attempts": gap.attempts,
                        "status": gap.status,
                    },
                )
        return updated

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _load_persisted_graph(
        self,
        state: PipelineState,
    ) -> Optional[EvidenceGraph]:
        """Load the most recent persisted graph (empty-entry fallback).

        Preference order:

        1. Lossless ``graph-round-{N}.json`` snapshot (authoritative,
           no conversion loss);
        2. ``KnowledgeGraphManager`` JSONL store via
           ``load_to_evidence_graph`` (lossy KnowledgeGraph round-trip).

        Returns ``None`` when neither source has a non-empty graph.
        """
        from pathlib import Path

        from ..memory.graph_manager import KnowledgeGraphManager
        from ..memory.snapshots import load_latest_graph_round

        snapshot_dir = self.output_dir or getattr(state, "memory_cache_dir", "")
        if snapshot_dir:
            payload = load_latest_graph_round(snapshot_dir)
            if payload:
                try:
                    graph = EvidenceGraph(**payload)
                    if graph.nodes:
                        return graph
                except Exception as exc:
                    logger.warning(
                        "M3: failed to parse round snapshot in %s: %s",
                        snapshot_dir, exc,
                    )

        cache_dir = getattr(state, "memory_cache_dir", "")
        if cache_dir:
            try:
                mgr = KnowledgeGraphManager(cache_dir=Path(cache_dir))
                loaded = mgr.load_to_evidence_graph()
                if loaded is not None and loaded.nodes:
                    return loaded
            except Exception as exc:
                logger.warning(
                    "M3: failed to load persisted evidence graph from %s: %s",
                    cache_dir, exc,
                )
        return None

    @staticmethod
    def _persist_graph(graph: EvidenceGraph, cache_dir: str) -> None:
        """Merge the evidence graph into the JSONL-backed persistent store.

        Uses ``merge_from_evidence_graph`` (union semantics) instead of the
        overwriting ``save_from_evidence_graph`` so knowledge accumulated
        across rounds / supplement searches is retained in the index layer.
        """
        from pathlib import Path

        from ..memory.graph_manager import KnowledgeGraphManager

        _logger = logging.getLogger(__name__)
        try:
            mgr = KnowledgeGraphManager(cache_dir=Path(cache_dir))
            mgr.merge_from_evidence_graph(graph)
            _logger.info(
                "M3: merged evidence graph (%d nodes, %d edges) into %s",
                len(graph.nodes), len(graph.edges), cache_dir,
            )
        except Exception as exc:
            _logger.warning("M3: failed to persist evidence graph: %s", exc)

    def _write_round_snapshot(
        self,
        graph: EvidenceGraph,
        snapshot_dir: str,
        state: PipelineState,
        round_no: Optional[int] = None,
    ) -> None:
        """Losslessly snapshot this round's graph as ``graph-round-{N}.json``.

        *round_no* may be passed explicitly (e.g. by a caller that tracks
        its own round counter).  Otherwise ``search_round`` is used when
        it is positive (supplement rounds), falling back to
        ``iteration_count`` for plain review iterations.
        """
        from ..memory.snapshots import save_graph_round_snapshot

        if round_no is None:
            round_no = int(getattr(state, "search_round", 0) or 0)
            if round_no <= 0:
                round_no = max(0, int(getattr(state, "iteration_count", 0) or 0))
        save_graph_round_snapshot(graph, snapshot_dir, max(0, int(round_no)))

    @classmethod
    def get_input_fields(cls) -> List[str]:
        return ["literature_results", "evidence_graph", "reviews", "iteration_count"]

    @classmethod
    def get_output_fields(cls) -> List[str]:
        return ["evidence_graph"]
