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
import hashlib
import json
import logging
import re
import uuid
from typing import Any, Dict, List, Optional, Set

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
    GroundingReport,
    KnowledgeEntryType,
    PipelineState,
)
from ..tools.qwen_client import QwenClient
from .m3_grounding import GroundingWorkflow
from .m3_grounding.models import (
    AtomicClaim,
    EvidenceRecord,
    EvidenceRelation,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Optional emit_event import (from feat/full-pipeline-web-ui)
# ============================================================================

try:
    from ..observability import emit_event  # noqa: F401
    HAS_EMIT_EVENT = True
except ImportError:
    HAS_EMIT_EVENT = False
    # Define a no-op so callers don't need conditional guards at every site.
    def emit_event(*args, **kwargs) -> None:  # type: ignore[no-redef]
        pass


# ============================================================================
# Entity synonym map — normalises common biomedical name variants
# ============================================================================

# Canonical → {variants}.  Keys are lowercase; values are sets of known
# aliases (also lowercase).  组员可扩展: add UMLS / MeSH / GO lookups here.
ENTITY_SYNONYMS: Dict[str, Set[str]] = {
    "hsp70": {"hsp70", "hspa1a", "hspa1b", "hsp72", "hsp70-1", "heat shock protein 70"},
    "hsp90": {"hsp90", "hsp90aa1", "hsp90ab1", "hspc1", "heat shock protein 90"},
    "p53": {"p53", "tp53", "tumor protein p53", "tumour protein p53"},
    "nf-kb": {"nf-kb", "nfkb", "nf-κb", "nuclear factor kappa b", "nuclear factor κb"},
    "tnf-α": {"tnf-α", "tnf-alpha", "tnfa", "tnfα", "tumor necrosis factor alpha"},
    "il-6": {"il-6", "il6", "interleukin-6", "interleukin 6"},
    "akt": {"akt", "akt1", "pkb", "protein kinase b", "rac-alpha"},
    "mtor": {"mtor", "mtorc1", "mtorc2", "mammalian target of rapamycin", "frap1"},
    "ampk": {"ampk", "amp-activated protein kinase", "prkaa1", "prkaa2"},
    "mapk": {"mapk", "map kinase", "mitogen-activated protein kinase", "erk", "erk1", "erk2"},
    "nad+": {"nad+", "nad", "nicotinamide adenine dinucleotide"},
    "ros": {"ros", "reactive oxygen species", "oxidative stress"},
    "caspase-3": {"caspase-3", "casp3", "caspase 3"},
    "bcl-2": {"bcl-2", "bcl2", "b-cell lymphoma 2"},
    "vegf": {"vegf", "vegf-a", "vascular endothelial growth factor"},
    "egfr": {"egfr", "epidermal growth factor receptor", "erbb1", "her1"},
    "pi3k": {"pi3k", "phosphatidylinositol 3-kinase", "pi3 kinase", "pik3ca"},
    "wnt": {"wnt", "wingless", "wnt/β-catenin", "wnt/beta-catenin"},
    "notch": {"notch", "notch1", "notch signaling"},
    "hedgehog": {"hedgehog", "shh", "sonic hedgehog", "hh signaling"},
}


def _build_entity_index() -> Dict[str, str]:
    """Build a lookup mapping every variant → canonical form."""
    index: Dict[str, str] = {}
    for canonical, variants in ENTITY_SYNONYMS.items():
        for variant in variants:
            index[variant] = canonical
    return index


_ENTITY_INDEX: Dict[str, str] = _build_entity_index()


def normalize_entity(name: str) -> str:
    """Normalize a biomedical entity name to its canonical form.

    Steps:
    1. Strip whitespace and lowercase.
    2. Remove trailing punctuation (commas, periods, semicolons).
    3. Look up in the synonym index.
    4. Fall back to the cleaned original if no synonym match.
    """
    cleaned = name.strip().lower().rstrip(".,;:)-]")
    # Remove leading punctuation like opening parens
    cleaned = cleaned.lstrip("([")
    if cleaned in _ENTITY_INDEX:
        return _ENTITY_INDEX[cleaned]
    return cleaned


def add_entity_synonym(canonical: str, variant: str) -> None:
    """Register a new entity synonym at runtime (no restart needed)."""
    canonical_lower = canonical.strip().lower()
    variant_lower = variant.strip().lower()
    ENTITY_SYNONYMS.setdefault(canonical_lower, set()).add(variant_lower)
    _ENTITY_INDEX[variant_lower] = canonical_lower


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

        # Track existing node/edge IDs for incremental updates
        self._existing_node_ids: Set[str] = set()
        self._existing_entry_ids: Set[str] = set()

    # ------------------------------------------------------------------
    # ModuleProtocol implementation
    # ------------------------------------------------------------------

    async def __call__(
        self,
        state: PipelineState,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        emit_event("m3:start", {"mode": self.mode, "grounding": self.grounding_enabled})

        # --- collect all knowledge entries ---
        all_entries = []
        for lr in state.literature_results:
            all_entries.extend(lr.knowledge_entries)

        if not all_entries:
            emit_event("m3:done", {"nodes": 0, "edges": 0, "reason": "no_entries"})
            return {"evidence_graph": EvidenceGraph()}

        # --- Incremental update: reuse existing graph when present ---
        existing_graph = state.evidence_graph
        if existing_graph is not None and existing_graph.nodes:
            new_entries = self._find_new_entries(all_entries, existing_graph)
            if not new_entries:
                logger.info(
                    "M3 incremental: no new entries; reusing existing graph "
                    "(%d nodes, %d edges)",
                    len(existing_graph.nodes), len(existing_graph.edges),
                )
                emit_event("m3:done", {
                    "nodes": len(existing_graph.nodes),
                    "edges": len(existing_graph.edges),
                    "reason": "incremental_no_new",
                })
                return {"evidence_graph": existing_graph}
            logger.info(
                "M3 incremental: %d new entries; appending to existing graph "
                "(%d nodes, %d edges)",
                len(new_entries), len(existing_graph.nodes), len(existing_graph.edges),
            )
            emit_event("m3:rule_graph_start", {
                "new_entries": len(new_entries),
                "existing_nodes": len(existing_graph.nodes),
            })
            graph = self._build_rule_graph(new_entries)
            graph = self._merge_graphs(existing_graph, graph)
        else:
            emit_event("m3:rule_graph_start", {"entries": len(all_entries)})
            graph = self._build_rule_graph(all_entries)

        emit_event("m3:rule_graph_done", {
            "nodes": len(graph.nodes), "edges": len(graph.edges),
        })

        # --- Step 2: LLM-enhanced relation extraction (optional) ---
        if self.mode in {"llm", "direct", "api"} and self.client:
            try:
                emit_event("m3:llm_enhance_start", {"mode": self.mode})
                graph = await self._enhance_with_llm_batched(graph, all_entries)
                emit_event("m3:llm_enhance_done", {
                    "nodes": len(graph.nodes), "edges": len(graph.edges),
                })
            except Exception as exc:
                logger.warning(
                    "M3 LLM enhancement failed; using rule-only graph: %s", exc
                )
                emit_event("m3:llm_enhance_failed", {"error": str(exc)})

        # --- Step 3: Iterative refinement from M6 reviews ---
        if (
            state.iteration_count > 0
            and state.reviews
            and self.client
            and self.mode in {"llm", "direct", "api"}
        ):
            try:
                emit_event("m3:refine_start", {"iteration": state.iteration_count})
                graph = await self._refine_with_reviews(graph, state)
                emit_event("m3:refine_done", {
                    "nodes": len(graph.nodes), "edges": len(graph.edges),
                })
            except Exception as exc:
                logger.warning(
                    "M3 iterative refinement failed; keeping unrefined graph: %s", exc
                )
                emit_event("m3:refine_failed", {"error": str(exc)})

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
                emit_event("m3:grounding_skipped", {"reason": "no_m2_knowledge_export"})
            else:
                try:
                    emit_event("m3:grounding_start", {})
                    grounded = await self._grounder.run(state)
                    graph = self._merge_grounding(
                        graph,
                        grounded.get("evidence_records", []),
                        grounded.get("claims", []),
                        grounded.get("relations", []),
                        grounded.get("report"),
                    )
                    result["evidence_graph"] = graph
                    emit_event("m3:grounding_done", {
                        "nodes": len(graph.nodes),
                        "edges": len(graph.edges),
                        "claims": len(grounded.get("claims", [])),
                        "relations": len(grounded.get("relations", [])),
                    })
                except Exception as exc:
                    logger.exception(
                        "M3 grounding workflow failed; retaining non-grounded graph: %s",
                        exc,
                    )
                    emit_event("m3:grounding_failed", {"error": str(exc)})

        # Track existing IDs for next incremental run
        self._existing_node_ids = {n.id for n in graph.nodes}
        self._existing_entry_ids = {
            n.id[2:] for n in graph.nodes if n.id.startswith("N_")
        }

        # --- persist to disk (if enabled) ---
        if getattr(state, "memory_cache_dir", ""):
            self._persist_graph(graph, state.memory_cache_dir)

        emit_event("m3:done", {
            "nodes": len(graph.nodes), "edges": len(graph.edges),
        })

        return result

    # ------------------------------------------------------------------
    # Incremental update helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_new_entries(
        all_entries: list,
        existing_graph: EvidenceGraph,
    ) -> list:
        """Return entries whose IDs are not already represented in the graph."""
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
                title = e.source_paper_title or pid
                nodes.append(EvidenceNode(
                    id=nid,
                    type=EvidenceNodeType.SOURCE,
                    label=title,
                    metadata={
                        "paper_id": pid,
                        "searchable_text": title,
                    },
                ))

        # Track entity nodes to avoid duplicates
        seen_entities: Dict[str, str] = {}

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

            # Normalise entities in metadata
            normalised_entities = [
                normalize_entity(name) for name in e.entities
            ]

            # Build searchable_text aggregating all searchable fields
            searchable_text = " ".join(filter(None, [
                e.content,
                " ".join(normalised_entities),
                e.source_paper_title or "",
            ]))

            nodes.append(EvidenceNode(
                id=nid,
                type=node_type,
                label=e.content[:120],
                metadata={
                    "entry_type": e.type.value,
                    "confidence": e.confidence.value if e.confidence else None,
                    "entities": normalised_entities,
                    "searchable_text": searchable_text,
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
                    # Include original entity names in searchable_text
                    ent_searchable = " ".join(filter(None, [
                        name,
                        " ".join(e.entities),
                    ]))
                    nodes.append(EvidenceNode(
                        id=eid,
                        type=EvidenceNodeType.ENTITY,
                        label=name,
                        metadata={
                            "normalised_from": e.entities,
                            "searchable_text": ent_searchable,
                        },
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
    def _make_gev_node_id(evidence_id: str) -> str:
        """Create a collision-resistant GEV_ node ID from an evidence_id.

        Uses SHA-256 hash (first 16 hex chars) instead of raw truncation
        to avoid collisions when different evidence_ids share the same
        first 20 characters.
        """
        suffix = hashlib.sha256(evidence_id.encode()).hexdigest()[:16]
        return f"GEV_{suffix}"

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
        edges (evidence → claim), semantic relations (claim → claim),
        and **bridge edges** connecting the grounding subgraph to the
        rule-based subgraph so that BFS traversals (e.g. NoveltyMetric)
        can reach nodes in both subgraphs.

        .. versionchanged:: 0.4.1
            - GEV_ node IDs use SHA-256 hash instead of ``evidence_id[:20]``.
            - ``searchable_text`` is added to all new node metadata for
              BM25 / keyword search compatibility (k1=0.5+).
            - Bridge edges (SRC_ → GEV_, N_xxx → GCLM_) connect the two
              previously disconnected subgraphs.
        """
        record_map = {record.evidence_id: record for record in records}
        claim_map: Dict[str, str] = {}
        ev_node_map: Dict[str, str] = {}  # evidence_id → GEV_ node id

        # --- Pre-compute rule-graph lookup tables for bridge edges ---
        # SOURCE nodes indexed by paper_id
        src_by_paper: Dict[str, str] = {}
        # CLAIM/EVIDENCE rule-graph nodes with their normalised entities
        rule_claim_nodes: List[Dict[str, Any]] = []
        for node in graph.nodes:
            if node.type == EvidenceNodeType.SOURCE:
                pid = (node.metadata or {}).get("paper_id", "")
                if pid:
                    src_by_paper[pid] = node.id
            if node.type in (EvidenceNodeType.CLAIM, EvidenceNodeType.EVIDENCE) and node.id.startswith("N_"):
                rule_claim_nodes.append({
                    "id": node.id,
                    "label": node.label,
                    "entities": set((node.metadata or {}).get("entities", [])),
                })

        # --- Create CLAIM nodes ---
        for claim in claims:
            node_id = "GCLM_" + claim.id.replace("CLM_", "")
            claim_map[claim.id] = node_id
            evidence_paper_ids: List[str] = []
            for eid in claim.evidence_ids:
                rec = record_map.get(eid)
                if rec and rec.paper_id:
                    evidence_paper_ids.append(rec.paper_id)
            normalised_claim_entities = [
                normalize_entity(name) for name in claim.entities
            ]
            # Build searchable_text for BM25/kw search compatibility
            gclm_searchable = " ".join(filter(None, [
                claim.statement,
                " ".join(normalised_claim_entities),
            ]))
            graph.nodes.append(EvidenceNode(
                id=node_id,
                type=EvidenceNodeType.CLAIM,
                label=claim.statement[:200],
                metadata={
                    "claim_id": claim.id,
                    "confidence": claim.confidence,
                    "evidence_ids": claim.evidence_ids,
                    "paper_ids": list(dict.fromkeys(evidence_paper_ids)),
                    "entities": normalised_claim_entities,
                    "provenance_level": "m2_fulltext_grounded",
                    "searchable_text": gclm_searchable,
                },
            ))

        # --- Create EVIDENCE nodes for each EvidenceRecord ---
        for record in records:
            node_id = M3EvidenceGraph._make_gev_node_id(record.evidence_id)
            ev_node_map[record.evidence_id] = node_id
            # Build searchable_text aggregating all searchable evidence fields
            gev_searchable = " ".join(filter(None, [
                record.summary,
                record.quote[:500] if record.quote else "",
                record.normalized_claim[:500] if record.normalized_claim else "",
            ]))
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
                    "searchable_text": gev_searchable,
                },
            ))

        # --- Bridge edge type A: SRC_ → GEV_ (source paper contains evidence) ---
        bridge_src_count = 0
        for record in records:
            if record.paper_id and record.paper_id in src_by_paper:
                src_node = src_by_paper[record.paper_id]
                gev_node = ev_node_map.get(record.evidence_id)
                if gev_node:
                    already = any(
                        e.source == src_node and e.target == gev_node
                        for e in graph.edges
                    )
                    if not already:
                        graph.edges.append(EvidenceEdge(
                            source=src_node,
                            target=gev_node,
                            relation=EvidenceEdgeRelation.INVOLVES,
                            rationale=f"Source paper contains M2 evidence item {record.evidence_id}.",
                        ))
                        bridge_src_count += 1

        # --- Bridge edge type B: N_xxx → GCLM_ (semantic SAME_AS via entity overlap) ---
        bridge_sameas_count = 0
        gclm_entities = {}
        for claim in claims:
            gclm_id = claim_map[claim.id]
            gclm_entities[gclm_id] = set(
                normalize_entity(name) for name in claim.entities
            )

        for gclm_id, gclm_ent_set in gclm_entities.items():
            if not gclm_ent_set:
                continue
            for rcn in rule_claim_nodes:
                if not rcn["entities"]:
                    continue
                overlap = len(gclm_ent_set & rcn["entities"])
                # Require ≥2 shared entities OR ≥50% Jaccard for a match
                jaccard = overlap / len(gclm_ent_set | rcn["entities"]) if overlap else 0
                if overlap >= 2 or jaccard >= 0.5:
                    already = any(
                        e.source == rcn["id"] and e.target == gclm_id
                        and e.relation == EvidenceEdgeRelation.SAME_AS
                        for e in graph.edges
                    )
                    if not already:
                        graph.edges.append(EvidenceEdge(
                            source=rcn["id"],
                            target=gclm_id,
                            relation=EvidenceEdgeRelation.SAME_AS,
                            confidence=min(0.9, 0.5 + jaccard),
                            rationale=f"Semantic match: {overlap} shared entities (Jaccard={jaccard:.2f}).",
                        ))
                        bridge_sameas_count += 1

        # --- Add provenance edges (evidence → claim) ---
        for claim in claims:
            claim_node = claim_map.get(claim.id)
            if not claim_node:
                continue
            for eid in claim.evidence_ids:
                ev_node = ev_node_map.get(eid)
                if not ev_node:
                    continue
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
            "(%d total nodes, %d total edges). "
            "Bridge edges: %d SRC→GEV, %d N→GCLM (SAME_AS).",
            len(claims), len(records), len(relations),
            len(graph.nodes), len(graph.edges),
            bridge_src_count, bridge_sameas_count,
        )

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
        return ["literature_results", "evidence_graph", "reviews", "iteration_count"]

    @classmethod
    def get_output_fields(cls) -> List[str]:
        return ["evidence_graph"]
