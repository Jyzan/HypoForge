"""
Persistent knowledge-graph manager (JSONL-backed).

Adapted from BioDSA's ``KnowledgeGraphManager``
(``biodsa/memory/memory_graph/graph.py``) and simplified for HypoForge:

* Synchronous API (called from within async M3 ``__call__``).
* BM25 search index (optional, falls back to linear matching).
* No NetworkX visualisation.
* Uses ``"hypoforge"`` file-marker for safety.
* Retains entity/relation caching and streaming-append for small batches.
"""

from __future__ import annotations

import json
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from .schema import (
    Entity,
    KnowledgeGraph,
    Relation,
    calculate_entities_hash,
    evidence_graph_to_knowledge_graph,
    knowledge_graph_to_evidence_graph,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# File-format constants
# ---------------------------------------------------------------------------

FILE_MARKER = {"type": "_hypoforge", "source": "knowledge-graph"}

DEFAULT_CACHE_DIR = Path.home() / ".hypoforge" / "knowledge_graph"


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def get_memory_file_path(cache_dir: Path, context: str = "evidence_graph") -> Path:
    """JSONL file for a given context."""
    return cache_dir / f"memory-{context}.jsonl"


def get_index_file_path(cache_dir: Path, context: str = "evidence_graph") -> Path:
    """Pickle file for the BM25 index of a given context."""
    return cache_dir / f"index-{context}.pkl"


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class KnowledgeGraphManager:
    """Persistent knowledge-graph store backed by a JSONL file.

    Usage::

        mgr = KnowledgeGraphManager(cache_dir=Path("./kg_cache"))
        mgr.save_from_evidence_graph(evidence_graph, context="run_001")
        # … later …
        kg = mgr.read_graph(context="run_001")
    """

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    def __init__(self, cache_dir: Optional[Path] = None):
        if cache_dir is None:
            cache_dir = DEFAULT_CACHE_DIR
        if isinstance(cache_dir, str):
            cache_dir = Path(cache_dir)
        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        # In-memory graph cache
        self._cached_graph: Optional[KnowledgeGraph] = None
        self._cached_graph_context: Optional[str] = None
        self._cached_graph_file_mtime: Optional[float] = None

        # Entity lookup cache
        self._entity_cache: Dict[str, Entity] = {}
        self._cache_dirty: bool = True

        # Relation lookup cache
        self._relation_cache: Dict[Tuple[str, str, str], Relation] = {}
        self._relation_cache_dirty: bool = True

        # BM25 search index (lazy-initialised on first search or save)
        self._search_index: object = None  # BM25SearchIndex
        self._index_dirty: bool = True

    # ==================================================================
    # Public API
    # ==================================================================

    # ------------------------------------------------------------------
    # EvidenceGraph convenience
    # ------------------------------------------------------------------

    def save_from_evidence_graph(
        self,
        evidence_graph,
        context: str = "evidence_graph",
    ) -> KnowledgeGraph:
        """Convert an ``EvidenceGraph`` → ``KnowledgeGraph`` and persist.

        Also builds/updates the BM25 search index so the graph is
        immediately searchable.

        Returns the ``KnowledgeGraph`` that was written to disk.
        """
        kg = evidence_graph_to_knowledge_graph(evidence_graph)
        self._save_graph(kg, context)

        # Build / update BM25 index
        self._ensure_index_built(context)

        logger.info(
            "Persisted evidence graph to %s (context=%s): %d entities, %d relations",
            self._cache_dir, context, len(kg.entities), len(kg.relations),
        )
        return kg

    def load_to_evidence_graph(self, context: str = "evidence_graph"):
        """Load a persisted ``KnowledgeGraph`` and convert back to ``EvidenceGraph``.

        Returns ``None`` if no persisted graph exists for *context*.
        """
        kg = self._load_graph(context)
        if kg.is_empty:
            return None
        return knowledge_graph_to_evidence_graph(kg)

    def search_nodes(
        self, query: str, context: str = "evidence_graph", top_k: Optional[int] = None
    ) -> KnowledgeGraph:
        """BM25-powered semantic search over entities.

        Returns a ``KnowledgeGraph`` containing only the matching entities
        and their inter-relations (a relevance-scored subgraph).
        Falls back to linear substring matching if ``rank_bm25`` is not
        installed.
        """
        self._ensure_index_built(context)
        if self._search_index is None:
            return KnowledgeGraph()

        matching_names = self._search_index.search(query, top_k)
        if not matching_names:
            return KnowledgeGraph()

        graph = self._load_graph(context)
        name_set = set(matching_names)
        filtered_entities = [e for e in graph.entities if e.name in name_set]
        filtered_relations = [
            r for r in graph.relations
            if r.from_entity in name_set and r.to_entity in name_set
        ]
        return KnowledgeGraph(entities=filtered_entities, relations=filtered_relations)

    def open_nodes(
        self, names: List[str], context: str = "evidence_graph"
    ) -> KnowledgeGraph:
        """Retrieve specific entities by name, including their inter-relations."""
        graph = self._load_graph(context)
        name_set = set(names)
        filtered_entities = [e for e in graph.entities if e.name in name_set]
        filtered_names = {e.name for e in filtered_entities}
        filtered_relations = [
            r for r in graph.relations
            if r.from_entity in filtered_names and r.to_entity in filtered_names
        ]
        return KnowledgeGraph(entities=filtered_entities, relations=filtered_relations)

    def get_text_overview(
        self,
        context: str = "evidence_graph",
        max_entities: Optional[int] = None,
        max_observations_per_entity: int = 5,
    ) -> str:
        """Return a human/LLM-readable markdown overview of the full graph."""
        graph = self._load_graph(context)

        if not graph.entities and not graph.relations:
            return "# Knowledge Graph\n\nThe graph is currently empty."

        # Collect implicit entities (appear in relations but not as Entity objects)
        explicit = {e.name for e in graph.entities}
        implicit: Set[str] = set()
        for r in graph.relations:
            if r.from_entity not in explicit:
                implicit.add(r.from_entity)
            if r.to_entity not in explicit:
                implicit.add(r.to_entity)

        all_entities = list(graph.entities) + [
            Entity(name=n, entity_type="implicit", observations=["(referenced in relations)"])
            for n in sorted(implicit)
        ]

        # Sort by connectivity
        connectivity: Dict[str, int] = {}
        for e in all_entities:
            connectivity[e.name] = sum(
                1 for r in graph.relations
                if r.from_entity == e.name or r.to_entity == e.name
            )
        all_entities.sort(key=lambda e: (-connectivity.get(e.name, 0), e.name))

        if max_entities:
            all_entities = all_entities[:max_entities]

        entity_names_set = {e.name for e in all_entities}

        lines = [
            "# Knowledge Graph",
            "",
            f"**Entities:** {len(graph.entities)}  |  **Relations:** {len(graph.relations)}",
            "",
            "## Entities",
            "",
        ]

        # Group by type
        by_type: Dict[str, list] = {}
        for e in all_entities:
            by_type.setdefault(e.entity_type or "uncategorized", []).append(e)

        for etype, entities in sorted(by_type.items()):
            lines.append(f"### {etype.upper()}")
            lines.append("")
            for e in sorted(entities, key=lambda x: (-connectivity.get(x.name, 0), x.name)):
                lines.append(f"**{e.name}** ({e.entity_type})")
                if e.observations:
                    for obs in e.observations[:max_observations_per_entity]:
                        lines.append(f"  - {obs}")
                    if len(e.observations) > max_observations_per_entity:
                        lines.append(
                            f"  - ... ({len(e.observations) - max_observations_per_entity} more)"
                        )
                lines.append("")
            lines.append("")

        # Relations
        rels_in_view = [
            r for r in graph.relations
            if r.from_entity in entity_names_set and r.to_entity in entity_names_set
        ]
        if rels_in_view:
            lines.append("## Relations")
            lines.append("")
            for r in rels_in_view:
                lines.append(f"- {r.from_entity} **{r.relation_type}** → {r.to_entity}")
            lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Graph-level operations
    # ------------------------------------------------------------------

    def read_graph(self, context: str = "evidence_graph") -> KnowledgeGraph:
        """Return the full persisted graph for *context*."""
        return self._load_graph(context)

    def clear_graph(self, context: str = "evidence_graph") -> None:
        """Delete all entities & relations for *context* (writes empty file)."""
        empty = KnowledgeGraph()
        self._save_graph(empty, context)
        logger.info("Cleared knowledge graph (context=%s)", context)

    def list_contexts(self) -> List[str]:
        """List available graph contexts (by scanning ``*.jsonl`` files)."""
        contexts: List[str] = []
        try:
            for f in sorted(self._cache_dir.glob("*.jsonl")):
                name = f.name
                if name == "memory.jsonl":
                    contexts.append("evidence_graph")  # default
                elif name.startswith("memory-") and name.endswith(".jsonl"):
                    contexts.append(name[7:-6])  # strip "memory-" prefix, ".jsonl" suffix
        except OSError:
            pass
        return contexts

    # ------------------------------------------------------------------
    # Entity CRUD
    # ------------------------------------------------------------------

    def create_entities(
        self, entities: List[Entity], context: str = "evidence_graph"
    ) -> List[Entity]:
        """Create entities (deduplicated by name).

        Uses streaming append for ≤10 entities, full-load otherwise.
        """
        if not entities:
            return []
        if len(entities) <= 10:
            return self._create_entities_streaming(entities, context)
        return self._create_entities_bulk(entities, context)

    def delete_entities(
        self, entity_names: List[str], context: str = "evidence_graph"
    ) -> None:
        """Delete entities and their associated relations."""
        graph = self._load_graph(context)
        names_set = set(entity_names)
        graph.entities = [e for e in graph.entities if e.name not in names_set]
        graph.relations = [
            r for r in graph.relations
            if r.from_entity not in names_set and r.to_entity not in names_set
        ]
        self._save_graph(graph, context)

    # ------------------------------------------------------------------
    # Relation CRUD
    # ------------------------------------------------------------------

    def create_relations(
        self, relations: List[Relation], context: str = "evidence_graph"
    ) -> List[Relation]:
        """Create relations; auto-creates missing entities with type ``"auto_created"``."""
        if not relations:
            return []
        graph = self._load_graph(context)
        existing_names = {e.name for e in graph.entities}

        referenced: Set[str] = set()
        for r in relations:
            referenced.add(r.from_entity)
            referenced.add(r.to_entity)
        missing = referenced - existing_names

        if missing:
            for name in sorted(missing):
                graph.entities.append(
                    Entity(
                        name=name,
                        entity_type="auto_created",
                        observations=[f"Auto-created from relation ({self._timestamp()})"],
                    )
                )
            self._entity_cache.clear()

        self._ensure_relation_cache_built(graph.relations)
        new_relations = [
            r for r in relations
            if not self._relation_exists_fast(r, graph.relations)
        ]

        if new_relations or missing:
            graph.relations.extend(new_relations)
            self._save_graph(graph, context)

        return new_relations

    # ------------------------------------------------------------------
    # Observation CRUD
    # ------------------------------------------------------------------

    def add_observations(
        self,
        observations: List[Dict[str, object]],
        context: str = "evidence_graph",
    ) -> List[Dict[str, object]]:
        """Add observations to entities (auto-creates missing entities).

        Each entry: ``{"entityName": "...", "contents": ["obs1", ...]}``
        """
        graph = self._load_graph(context)
        results: List[Dict[str, object]] = []
        entities_updated: List[Entity] = []
        entities_created: List[Entity] = []

        for obs in observations:
            entity_name = str(obs["entityName"])
            contents = list(obs["contents"])  # type: ignore[arg-type]

            entity = self._get_entity_fast(entity_name, graph.entities)
            if entity is None:
                entity = Entity(
                    name=entity_name,
                    entity_type="auto_generated",
                    observations=list(contents),
                )
                graph.entities.append(entity)
                entities_created.append(entity)
                results.append({
                    "entityName": entity_name,
                    "addedObservations": list(contents),
                    "entity_created": True,
                })
            else:
                new_obs = [c for c in contents if c not in entity.observations]
                if new_obs:
                    entity.observations.extend(new_obs)
                    entities_updated.append(entity)
                results.append({
                    "entityName": entity_name,
                    "addedObservations": new_obs,
                    "entity_created": False,
                })

        if entities_updated or entities_created:
            self._save_graph(graph, context)

        return results

    # ==================================================================
    # Internal — file I/O
    # ==================================================================

    def _load_graph(self, context: str) -> KnowledgeGraph:
        """Load graph from JSONL, with mtime-based caching."""
        file_path = get_memory_file_path(self._cache_dir, context)

        if self._can_use_cached_graph(context, file_path):
            return self._cached_graph  # type: ignore[return-value]

        try:
            current_mtime = os.path.getmtime(file_path) if file_path.exists() else 0.0
        except OSError:
            current_mtime = 0.0

        if not file_path.exists():
            graph = KnowledgeGraph()
            self._update_graph_cache(graph, context, current_mtime)
            return graph

        with open(file_path, "r", encoding="utf-8") as fh:
            lines = [line.strip() for line in fh if line.strip()]

        if not lines:
            graph = KnowledgeGraph()
            self._update_graph_cache(graph, context, current_mtime)
            return graph

        # Validate safety marker
        first_line = json.loads(lines[0])
        if first_line.get("type") != "_hypoforge" or first_line.get("source") != "knowledge-graph":
            raise ValueError(
                f"File {file_path} missing _hypoforge safety marker. "
                f"This file may not belong to the HypoForge knowledge graph system."
            )

        entities: List[Entity] = []
        relations: List[Relation] = []

        for line in lines[1:]:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("type") == "entity":
                entities.append(Entity.from_dict({k: v for k, v in item.items() if k != "type"}))
            elif item.get("type") == "relation":
                relations.append(Relation.from_dict({k: v for k, v in item.items() if k != "type"}))

        graph = KnowledgeGraph(entities=entities, relations=relations)
        self._update_graph_cache(graph, context, current_mtime)
        self._cache_dirty = True
        self._relation_cache_dirty = True
        return graph

    def _save_graph(self, graph: KnowledgeGraph, context: str) -> None:
        """Write full graph to JSONL (marker + entities + relations)."""
        file_path = get_memory_file_path(self._cache_dir, context)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        lines: List[str] = [json.dumps(FILE_MARKER, ensure_ascii=False)]
        for entity in graph.entities:
            lines.append(json.dumps({"type": "entity", **entity.to_dict()}, ensure_ascii=False))
        for relation in graph.relations:
            lines.append(json.dumps({"type": "relation", **relation.to_dict()}, ensure_ascii=False))

        with open(file_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")

        try:
            new_mtime = os.path.getmtime(file_path)
            self._update_graph_cache(graph, context, new_mtime)
        except OSError:
            self._invalidate_graph_cache()

        self._cache_dirty = True
        self._relation_cache_dirty = True

    def _append_entities_to_file(self, entities: List[Entity], context: str) -> None:
        """Stream-append entities without full graph reload."""
        file_path = get_memory_file_path(self._cache_dir, context)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if not file_path.exists():
            with open(file_path, "w", encoding="utf-8") as fh:
                fh.write(json.dumps(FILE_MARKER, ensure_ascii=False) + "\n")

        with open(file_path, "a", encoding="utf-8") as fh:
            for entity in entities:
                fh.write(json.dumps({"type": "entity", **entity.to_dict()}, ensure_ascii=False) + "\n")

        self._invalidate_graph_cache()
        self._cache_dirty = True

    # ------------------------------------------------------------------
    # Streaming helpers
    # ------------------------------------------------------------------

    def _entity_exists_streaming(self, entity_name: str, context: str) -> bool:
        """Check entity existence by streaming (avoids full load)."""
        file_path = get_memory_file_path(self._cache_dir, context)
        if not file_path.exists():
            return False

        try:
            with open(file_path, "r", encoding="utf-8") as fh:
                first = True
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if first:
                        first = False
                        if data.get("type") == "_hypoforge":
                            continue
                    if data.get("type") == "entity" and data.get("name") == entity_name:
                        return True
        except FileNotFoundError:
            return False
        return False

    # ------------------------------------------------------------------
    # Entity creation (two paths)
    # ------------------------------------------------------------------

    def _create_entities_streaming(
        self, entities: List[Entity], context: str
    ) -> List[Entity]:
        """≤10 entities: stream-check each for existence, then append."""
        new_entities = [
            e for e in entities
            if not self._entity_exists_streaming(e.name, context)
        ]
        if new_entities:
            self._append_entities_to_file(new_entities, context)
        return new_entities

    def _create_entities_bulk(
        self, entities: List[Entity], context: str
    ) -> List[Entity]:
        """>10 entities: load full graph, check in-memory, save."""
        graph = self._load_graph(context)
        self._ensure_cache_built(graph.entities)
        new_entities = [e for e in entities if e.name not in self._entity_cache]
        if new_entities:
            graph.entities.extend(new_entities)
            self._save_graph(graph, context)
        return new_entities

    # ==================================================================
    # Cache maintenance
    # ==================================================================

    def _can_use_cached_graph(self, context: str, file_path: Path) -> bool:
        if self._cached_graph is None:
            return False
        if self._cached_graph_context != context:
            return False
        if not file_path.exists():
            return self._cached_graph_file_mtime == 0.0
        try:
            return os.path.getmtime(file_path) == self._cached_graph_file_mtime
        except OSError:
            return False

    def _update_graph_cache(self, graph: KnowledgeGraph, context: str, mtime: float) -> None:
        self._cached_graph = graph
        self._cached_graph_context = context
        self._cached_graph_file_mtime = mtime

    def _invalidate_graph_cache(self) -> None:
        self._cached_graph = None
        self._cached_graph_context = None
        self._cached_graph_file_mtime = None

    # ------------------------------------------------------------------
    # Entity cache
    # ------------------------------------------------------------------

    def _build_entity_cache(self, entities: List[Entity]) -> None:
        self._entity_cache = {e.name: e for e in entities}
        self._cache_dirty = False

    def _ensure_cache_built(self, entities: List[Entity]) -> None:
        if self._cache_dirty or not self._entity_cache:
            self._build_entity_cache(entities)

    def _get_entity_fast(self, name: str, entities: List[Entity]) -> Optional[Entity]:
        self._ensure_cache_built(entities)
        return self._entity_cache.get(name)

    # ------------------------------------------------------------------
    # Relation cache
    # ------------------------------------------------------------------

    def _build_relation_cache(self, relations: List[Relation]) -> None:
        self._relation_cache = {
            (r.from_entity, r.to_entity, r.relation_type): r
            for r in relations
        }
        self._relation_cache_dirty = False

    def _ensure_relation_cache_built(self, relations: List[Relation]) -> None:
        if self._relation_cache_dirty or not self._relation_cache:
            self._build_relation_cache(relations)

    def _relation_exists_fast(self, relation: Relation, relations: List[Relation]) -> bool:
        self._ensure_relation_cache_built(relations)
        return (relation.from_entity, relation.to_entity, relation.relation_type) in self._relation_cache

    # ------------------------------------------------------------------
    # BM25 index
    # ------------------------------------------------------------------

    def _ensure_index_built(self, context: str = "evidence_graph") -> None:
        """Build or load-from-disk the BM25 index for *context*."""
        from .bm25_index import BM25SearchIndex

        if self._search_index is None:
            self._search_index = BM25SearchIndex()
            self._index_dirty = True

        graph = self._load_graph(context)

        # Try loading from disk first
        if not self._index_dirty and self._search_index.is_valid_for_entities(graph.entities):
            return

        index_file = get_index_file_path(self._cache_dir, context)
        entities_hash = calculate_entities_hash(graph.entities)

        if self._search_index.load_from_disk(index_file, entities_hash):
            self._index_dirty = False
            return

        # Build from scratch
        self._search_index.build_index(graph.entities)
        self._search_index.save_to_disk(index_file)
        self._index_dirty = False

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _timestamp() -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
