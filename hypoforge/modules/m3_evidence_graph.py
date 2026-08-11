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
from typing import Any, Callable, Dict, List, Optional, Set

from ..memory.hypoforge_types import GapGain, stable_entry_id
from ..protocol import ModuleProtocol
from ..entity_normalization import (
    EntityNormalizationService,
    _embed_raw,
    clean_entity_surface,
)
from ..entity_graph_merge import EmbedTexts, EntityGraphMerger
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
    GraphAuditRecord,
    GraphCorrectionRequest,
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
      - ``"rule"`` — fast, deterministic, no LLM calls.
      - ``"llm"`` — adds Qwen-powered cross-entry relation extraction on top
        of the rule-based graph.  When a client is explicitly configured and
        mode is ``"llm"``, LLM failure raises ``RuntimeError`` (no silent
        fallback to rule-only).
      - ``grounding_enabled=True`` — runs the M3 grounding workflow
        (consuming ``M2KnowledgeExport``) whenever there is grounding
        evidence to process, regardless of whether this round contributed
        new entries.  Grounding failure raises ``RuntimeError``.

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
        entity_embedding_model: str = "",
        entity_merge_enabled: bool = True,
        entity_merge_similarity_threshold: float = 0.92,
        entity_merge_embedding_backend: Optional[EmbedTexts] = None,
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
        self.entity_embedding_model = (
            entity_embedding_model or grounding_embedding_model
        )
        self.entity_merge_enabled = bool(entity_merge_enabled)
        self.entity_merge_similarity_threshold = float(
            entity_merge_similarity_threshold
        )
        self.entity_merge_embedding_backend = entity_merge_embedding_backend

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

        # Incremental dedup is rebuilt from graph metadata each run. Only the
        # task-local canonical surface map is instance state.
        self._task_entity_index: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # ModuleProtocol implementation
    # ------------------------------------------------------------------

    async def __call__(
        self,
        state: PipelineState,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        indexed_gap_requests = [
            gap.model_copy(update={"status": "indexed"})
            if gap.status == "searched" else gap.model_copy(deep=True)
            for gap in state.evidence_gap_requests
        ]
        # --- collect all knowledge entries ---
        all_entries = []
        for lr in state.literature_results:
            all_entries.extend(lr.knowledge_entries)

        # One public normalizer owns task aliases, optional embedding recall,
        # conservative LLM identity judgments, and persistent pair decisions.
        namespace = "|".join(sorted(
            state.problem_card.domain if state.problem_card else []
        )) or "global"
        normalizer = EntityNormalizationService(
            cache_dir=state.entity_cache_dir or state.memory_cache_dir,
            namespace=namespace,
            client=self.client,
            embedding_model=self.entity_embedding_model,
            # Per-run cache isolation: share the run_id stamp fixed at run
            # start so M2/M3 within one run use the same cache file.
            run_stamp=state.run_id,
        )
        # Compatibility vocabularies are explicit inputs to the public service;
        # the current task contract is registered last and is authoritative.
        for canonical, aliases in ENTITY_SYNONYMS.items():
            normalizer.register_alias_group(canonical, aliases)
        if state.problem_card:
            normalizer.register_task_contract(state.problem_card.task_contract)
        resolved_entities = await normalizer.resolve_batch(
            entity
            for entry in all_entries
            for entity in entry.entities
        )
        self._task_entity_index = {
            clean_entity_surface(original): resolved.canonical_name
            for original, resolved in resolved_entities.items()
            if clean_entity_surface(original)
        }

        existing_graph = state.evidence_graph
        if existing_graph is None and state.memory_cache_dir:
            existing_graph = self._load_persistent_graph(
                state.memory_cache_dir,
                " ".join(filter(None, [
                    state.input_question,
                    " ".join(state.problem_card.domain)
                    if state.problem_card else "",
                    " ".join(state.problem_card.key_entities)
                    if state.problem_card else "",
                ])),
            )

        if not all_entries:
            if not (existing_graph and existing_graph.nodes):
                existing_graph = self._load_persisted_graph(state)
            if existing_graph is not None and existing_graph.nodes:
                graph = existing_graph
                logger.info(
                    "M3: no knowledge entries; reusing existing graph "
                    "(%d nodes, %d edges)",
                    len(graph.nodes), len(graph.edges),
                )
            else:
                raise RuntimeError(
                    "M3 cannot build an evidence graph because no usable "
                    "knowledge entries or task-relevant historical graph "
                    "are available.  M2 must produce at least one "
                    "knowledge entry before M3 can proceed."
                )
            # Continue through the shared correction → grounding → gap
            # confirmation → entity merge → persistence path below.  A
            # checkpoint-resume round can have no new KnowledgeEntry objects
            # while still carrying new M2 evidence that must be grounded.

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
                if self.mode in {"llm", "direct", "api"} and self.client:
                    raise RuntimeError(
                        f"M3 LLM enhancement failed in mode={self.mode!r}.  "
                        f"LLM-based relation extraction was explicitly "
                        f"requested and cannot be silently skipped."
                    ) from exc
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

        # --- Step 3: validated, audited corrections requested by M6 ---
        correction_updates = [
            request.model_copy(deep=True)
            for request in state.graph_correction_requests
        ]
        if any(request.status == "pending" for request in correction_updates):
            graph, correction_updates = self._apply_graph_corrections(
                graph,
                correction_updates,
                iteration=state.iteration_count,
            )

        # --- Step 4: M3 grounding (optional — consumes M2KnowledgeExport) ---
        # Grounding runs when it is enabled AND there is evidence to ground,
        # regardless of whether this round contributed new entries.  This
        # supports checkpoint resume: a previously-built rule-only graph can
        # be grounded on resume if M2 evidence is now available.
        result: Dict[str, Any] = {"evidence_graph": graph}
        if correction_updates != state.graph_correction_requests:
            result["graph_correction_requests"] = correction_updates
        should_ground = (
            self.grounding_enabled
            and self._grounder is not None
            and state.m2_knowledge_export is not None
            and any(
                hasattr(run, "evidence") and run.evidence
                for run in state.m2_knowledge_export.runs
            )
        )
        if should_ground:
            # m2_knowledge_export is verified non-None by should_ground above.
            assert state.m2_knowledge_export is not None
            try:
                grounded = await self._grounder.run(state)
                graph = self._merge_grounding(
                    graph,
                    grounded.get("evidence_records", []),
                    grounded.get("claims", []),
                    grounded.get("relations", []),
                    grounded.get("report"),
                    entity_normalizer=self._normalise_task_entity,
                )
                result["evidence_graph"] = graph
            except Exception as exc:
                if self.grounding_enabled:
                    raise RuntimeError(
                        "M3 grounding workflow failed.  "
                        "grounding.enabled=True requires a successful "
                        "grounding run; a non-grounded graph does not "
                        "satisfy the evidence contract."
                    ) from exc
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

        # --- Step 6: final-graph entity similarity merge ---
        graph = await self._merge_final_entities(graph, state)
        result["evidence_graph"] = graph

        # --- persist to disk (merge semantics) + lossless round snapshot ---
        if getattr(state, "memory_cache_dir", ""):
            self._persist_graph(graph, state.memory_cache_dir)
        snapshot_dir = self.output_dir or getattr(state, "memory_cache_dir", "")
        if snapshot_dir:
            self._write_round_snapshot(graph, snapshot_dir, state)

        if indexed_gap_requests != state.evidence_gap_requests:
            result["evidence_gap_requests"] = indexed_gap_requests

        return result

    async def _merge_final_entities(
        self,
        graph: EvidenceGraph,
        state: PipelineState,
    ) -> EvidenceGraph:
        """Apply the auditable final-graph entity merge without failing M3."""
        if not self.entity_merge_enabled:
            return graph

        entity_count = sum(
            1 for node in graph.nodes if node.type == EvidenceNodeType.ENTITY
        )
        started_at = time.monotonic()
        emit_event(
            "entity_merge_started",
            module="m3",
            tool="entity_similarity_merger",
            status="running",
            message=(
                f"开始检查 {entity_count} 个实体节点，"
                f"合并阈值 {self.entity_merge_similarity_threshold:.2f}"
            ),
            details={
                "entity_count": entity_count,
                "threshold": self.entity_merge_similarity_threshold,
                "embedding_model": self.entity_embedding_model,
            },
        )
        task_entity_names: list[str] = []
        if state.problem_card:
            task_entity_names = [
                entity.name
                for entity in state.problem_card.task_contract.entities
                if entity.name.strip()
            ]
            if not task_entity_names:
                task_entity_names = list(state.problem_card.key_entities)

        embedding_backend = (
            self.entity_merge_embedding_backend or self._embed_final_entity_names
        )
        outcome = await EntityGraphMerger(
            threshold=self.entity_merge_similarity_threshold,
            embedding_model=self.entity_embedding_model,
            embed_texts=embedding_backend,
        ).merge(
            graph,
            task_entity_names=task_entity_names,
            round_index=state.iteration_count,
        )
        elapsed = time.monotonic() - started_at
        if outcome.degraded_reason:
            emit_event(
                "entity_merge_degraded",
                module="m3",
                tool="entity_similarity_merger",
                status="degraded",
                message=(
                    "实体 embedding 不可用，已仅执行完全同名合并："
                    f"{outcome.degraded_reason}"
                ),
                elapsed_seconds=elapsed,
                details={"reason": outcome.degraded_reason},
            )
        emit_event(
            "entity_merge_completed",
            module="m3",
            tool="entity_similarity_merger",
            status="completed",
            message=(
                f"实体合并完成：{outcome.entity_count_before} → "
                f"{outcome.entity_count_after}，共 {outcome.merge_count} 组"
            ),
            elapsed_seconds=elapsed,
            details={
                "merge_count": outcome.merge_count,
                "entity_count_before": outcome.entity_count_before,
                "entity_count_after": outcome.entity_count_after,
                "redirected_edge_count": outcome.redirected_edge_count,
                "deduplicated_edge_count": outcome.deduplicated_edge_count,
                "removed_self_loop_count": outcome.removed_self_loop_count,
                "degraded": bool(outcome.degraded_reason),
            },
        )
        return outcome.graph

    async def _embed_final_entity_names(
        self,
        texts: list[str],
        model: str,
    ) -> list[list[float]]:
        """Reuse the configured Qwen-compatible embedding transport."""
        return await _embed_raw(
            model=model,
            inputs=texts,
            base_url=self.client.api_base if self.client else "",
            api_key=self.client.api_key if self.client else "",
        )

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
            entry_ids = (node.metadata or {}).get("entry_ids", [])
            if isinstance(entry_ids, str):
                entry_ids = [entry_ids]
            existing_entry_ids.update(
                str(item) for item in entry_ids if str(item or "").strip()
            )
        return [
            entry
            for entry in all_entries
            if (entry.id or stable_entry_id(entry.content, entry.source_paper_id))
            not in existing_entry_ids
        ]

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

    def _normalise_task_entity(self, name: str) -> str:
        surface = clean_entity_surface(name)
        if surface in self._task_entity_index:
            return self._task_entity_index[surface]
        cleaned = normalize_entity(name)
        return self._task_entity_index.get(cleaned, cleaned)

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
                normalized
                for name in e.entities
                if (normalized := self._normalise_task_entity(name))
            ]
            node_label = e.content[:120]
            if e.type is KnowledgeEntryType.KEY_ENTITY:
                canonical = (
                    normalised_entities[0]
                    if normalised_entities else self._normalise_task_entity(e.content)
                )
                if canonical:
                    node_label = canonical

            nodes.append(EvidenceNode(
                id=nid,
                type=node_type,
                label=node_label,
                metadata={
                    "entry_id": e.id,
                    "entry_type": e.type.value,
                    "confidence": e.confidence.value if e.confidence else None,
                    "entities": normalised_entities,
                    "evidence_ids": list(e.evidence_ids),
                    "source_paper_id": e.source_paper_id,
                    "source_paper_title": e.source_paper_title,
                    "full_text": e.content[:2000],
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
                        metadata={
                            "normalised_from": [
                                original
                                for original in e.entities
                                if self._normalise_task_entity(original) == name
                            ],
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
                "entities": [self._normalise_task_entity(name) for name in entry.entities],
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
                "entities": [self._normalise_task_entity(name) for name in e.entities],
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

    @staticmethod
    def _apply_graph_corrections(
        graph: EvidenceGraph,
        requests: List[GraphCorrectionRequest],
        *,
        iteration: int,
    ) -> tuple[EvidenceGraph, List[GraphCorrectionRequest]]:
        """Validate correction requests and append a before/after audit record."""

        node_ids = {node.id for node in graph.nodes}
        valid_evidence_ids = {
            str(evidence_id)
            for node in graph.nodes
            for evidence_id in (node.metadata or {}).get("evidence_ids", [])
        }
        valid_evidence_ids.update(
            evidence_id for edge in graph.edges for evidence_id in edge.evidence_ids
        )
        updated: List[GraphCorrectionRequest] = []

        for request in requests:
            if request.status != "pending":
                updated.append(request)
                continue
            before_version = graph.version
            before: Dict[str, Any] = {}
            after: Dict[str, Any] = {}
            rejection = ""
            matching = [
                edge for edge in graph.edges
                if edge.source == request.source_node_id
                and edge.target == request.target_node_id
                and (
                    request.current_relation is None
                    or edge.relation == request.current_relation
                )
            ]

            if request.source_node_id not in node_ids or request.target_node_id not in node_ids:
                rejection = "request references an unknown graph node"
            elif not request.reason.strip():
                rejection = "request has no correction reason"
            elif not request.evidence_ids:
                rejection = "request has no canonical supporting evidence ID"
            elif any(
                evidence_id not in valid_evidence_ids
                for evidence_id in request.evidence_ids
            ):
                rejection = "request references evidence not present in the graph"
            elif request.operation in {"remove_edge", "reclassify_edge"} and len(matching) != 1:
                rejection = "request does not identify exactly one existing edge"
            elif request.operation in {"add_edge", "reclassify_edge"} and request.proposed_relation is None:
                rejection = "request has no proposed relation"

            if not rejection:
                if request.operation == "add_edge":
                    duplicate = any(
                        edge.source == request.source_node_id
                        and edge.target == request.target_node_id
                        and edge.relation == request.proposed_relation
                        for edge in graph.edges
                    )
                    if duplicate:
                        rejection = "the proposed edge already exists"
                    else:
                        edge = EvidenceEdge(
                            source=request.source_node_id,
                            target=request.target_node_id,
                            relation=request.proposed_relation,
                            rationale=request.reason[:500],
                            evidence_ids=list(request.evidence_ids),
                        )
                        graph.edges.append(edge)
                        after = edge.model_dump(mode="json")
                elif request.operation == "remove_edge":
                    edge = matching[0]
                    before = edge.model_dump(mode="json")
                    graph.edges.remove(edge)
                else:
                    edge = matching[0]
                    before = edge.model_dump(mode="json")
                    edge.relation = request.proposed_relation
                    edge.rationale = request.reason[:500]
                    edge.evidence_ids = list(dict.fromkeys([
                        *edge.evidence_ids, *request.evidence_ids,
                    ]))
                    after = edge.model_dump(mode="json")

            if rejection:
                resolved = request.model_copy(update={
                    "status": "rejected",
                    "rejection_reason": rejection,
                })
                status = "rejected"
            else:
                graph.version += 1
                resolved = request.model_copy(update={
                    "status": "applied",
                    "rejection_reason": "",
                })
                status = "applied"
            graph.audit_log.append(GraphAuditRecord(
                sequence=len(graph.audit_log) + 1,
                graph_version_before=before_version,
                graph_version_after=graph.version,
                request_id=request.request_id,
                operation=request.operation,
                status=status,
                before=before,
                after=after,
                reason=rejection or request.reason,
                evidence_ids=list(request.evidence_ids),
                iteration=iteration,
            ))
            updated.append(resolved)

        return graph, updated

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
        entity_normalizer: Optional[Callable[[str], str]] = None,
    ) -> EvidenceGraph:
        """Merge grounding workflow output into the existing evidence graph.

        Creates CLAIM nodes for each ``AtomicClaim`` and adds provenance
        edges (evidence → claim) and semantic relation edges (claim → claim).
        The ``evidence_ids`` on both claims and relations point back to
        ``M2EvidenceExport.evidence_id``, which is recorded in node metadata.
        """
        normalise = entity_normalizer or normalize_entity
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
                        normalise(name)
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
    def _load_persistent_graph(cache_dir: str, query: str) -> Optional[EvidenceGraph]:
        """Retrieve a task-relevant historical subgraph through the BM25 cache."""

        from pathlib import Path
        from ..memory.graph_manager import KnowledgeGraphManager
        from ..memory.schema import knowledge_graph_to_evidence_graph

        try:
            manager = KnowledgeGraphManager(cache_dir=Path(cache_dir))
            if query.strip():
                knowledge_graph = manager.search_nodes(
                    query, context="evidence_graph", top_k=30
                )
                graph = knowledge_graph_to_evidence_graph(knowledge_graph)
            else:
                graph = manager.load_to_evidence_graph()
            if graph is None or not graph.nodes:
                return None
            # A retrieved subgraph may not include the persistent _buckets meta
            # entity. Reconstruct buckets from canonical node metadata.
            for node in graph.nodes:
                metadata = node.metadata or {}
                entry_id = str(metadata.get("entry_id") or "")
                entry_type = str(metadata.get("entry_type") or "")
                if not entry_id:
                    continue
                if entry_type == KnowledgeEntryType.ESTABLISHED_FACT.value:
                    graph.established_facts.append(entry_id)
                elif entry_type == KnowledgeEntryType.CONFLICTING_EVIDENCE.value:
                    graph.conflicts.append(entry_id)
                elif entry_type == KnowledgeEntryType.KNOWLEDGE_GAP.value:
                    graph.knowledge_gaps.append(entry_id)
            graph.established_facts = list(dict.fromkeys(graph.established_facts))
            graph.conflicts = list(dict.fromkeys(graph.conflicts))
            graph.knowledge_gaps = list(dict.fromkeys(graph.knowledge_gaps))
            logger.info(
                "M3 loaded %d task-relevant nodes from persistent memory",
                len(graph.nodes),
            )
            return graph
        except Exception as exc:
            logger.warning("M3 persistent graph retrieval failed: %s", exc)
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
        return [
            "literature_results", "m2_knowledge_export", "evidence_graph",
            "reviews", "iteration_count", "memory_cache_dir", "problem_card",
        ]

    @classmethod
    def get_output_fields(cls) -> List[str]:
        return ["evidence_graph"]
