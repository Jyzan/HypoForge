"""
LangChain-compatible tool wrappers for the persistent knowledge graph.

These are **optional** — they require ``langchain_core`` to be installed.
When available, LLM agents can call ``add_to_evidence_graph`` and
``retrieve_from_evidence_graph`` at runtime to interact with the
persistent Entity-Relation-Observation store.

Adapted from BioDSA's ``biodsa/memory/graph.py`` (``AddToGraph`` and
``RetrieveFromGraph`` BaseTool subclasses).

Usage (from within an LLM-powered module)::

    from hypoforge.memory.tools import (
        AddToEvidenceGraph,
        RetrieveFromEvidenceGraph,
    )

    tools = [
        AddToEvidenceGraph(cache_dir="./kg_cache", context="run_001"),
        RetrieveFromEvidenceGraph(cache_dir="./kg_cache", context="run_001"),
    ]
    # bind to LLM, let agent call tools autonomously …
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

# ---------------------------------------------------------------------------
# Optional langchain import
# ---------------------------------------------------------------------------
_BASETOOL_AVAILABLE = False
try:
    from langchain_core.tools import BaseTool
    from pydantic import BaseModel as PydanticBaseModel, Field

    _BASETOOL_AVAILABLE = True
except ImportError:  # pragma: no cover
    # langchain_core not installed — the classes below will raise at init
    BaseTool = object  # type: ignore[assignment,misc]
    PydanticBaseModel = object  # type: ignore[assignment,misc]
    Field = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _get_manager(
    cache_dir: Optional[str] = None, context: str = "evidence_graph"
) -> Any:
    """Return a ``KnowledgeGraphManager`` for *cache_dir*."""
    from .graph_manager import DEFAULT_CACHE_DIR, KnowledgeGraphManager

    cd = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    return KnowledgeGraphManager(cache_dir=cd)


def _to_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


# ============================================================================
# Tool input schemas (pydantic)
# ============================================================================

if _BASETOOL_AVAILABLE:

    class _EntityInput(PydanticBaseModel):
        """Pydantic schema for an entity in a tool call."""

        name: str
        entity_type: str
        observations: List[str] = Field(default_factory=list)

    class _RelationInput(PydanticBaseModel):
        """Pydantic schema for a relation in a tool call."""

        from_entity: str
        to_entity: str
        relation_type: str

    class _AddToEvidenceGraphInput(PydanticBaseModel):
        """Input schema for ``add_to_evidence_graph``."""

        entities: Optional[List[_EntityInput]] = None
        relations: Optional[List[_RelationInput]] = None
        observations: Optional[_EntityInput] = None

    class _RetrieveFromEvidenceGraphInput(PydanticBaseModel):
        """Input schema for ``retrieve_from_evidence_graph``."""

        query: Optional[str] = Field(
            default=None,
            description="Natural-language search query (BM25-scored).",
        )
        entity_names: Optional[str] = Field(
            default=None,
            description="JSON-encoded list of entity names to fetch exactly.",
        )
        get_full_map: bool = Field(
            default=False,
            description="Return the entire graph as a markdown overview.",
        )
        top_k: int = Field(default=10, description="Max results for BM25 search.")

else:
    _EntityInput = object  # type: ignore[assignment,misc]
    _RelationInput = object  # type: ignore[assignment,misc]
    _AddToEvidenceGraphInput = object  # type: ignore[assignment,misc]
    _RetrieveFromEvidenceGraphInput = object  # type: ignore[assignment,misc]


# ============================================================================
# AddToEvidenceGraph
# ============================================================================


class AddToEvidenceGraph(BaseTool if _BASETOOL_AVAILABLE else object):  # type: ignore[misc]
    """LangChain tool: persist entities / relations / observations.

    An LLM agent calls this to record findings in the knowledge graph.
    """

    if _BASETOOL_AVAILABLE:
        name: str = "add_to_evidence_graph"
        description: str = (
            "Add scientific or engineering entities, relations, and observations to the "
            "persistent evidence knowledge graph.  Call this after extracting "
            "a new finding from the literature.  Entities are deduplicated by "
            "name; relations auto-create missing entities."
        )
        args_schema: Type[PydanticBaseModel] = _AddToEvidenceGraphInput

        # --- instance fields ---
        cache_dir: Optional[str] = None
        context: str = "evidence_graph"

        def _run(
            self,
            entities: Optional[List[dict]] = None,
            relations: Optional[List[dict]] = None,
            observations: Optional[dict] = None,
        ) -> str:
            """Synchronous entry point (called by LangChain)."""
            from hypoforge.memory.schema import Entity, Relation

            mgr = _get_manager(self.cache_dir, self.context)
            results: Dict[str, Any] = {}

            try:
                if entities:
                    elist = [
                        Entity(
                            name=e["name"],
                            entity_type=e.get("entity_type", e.get("entityType", "")),
                            observations=list(e.get("observations", [])),
                        )
                        for e in entities
                    ]
                    created = mgr.create_entities(elist, self.context)
                    results["entities_created"] = len(created)

                if relations:
                    rlist = [
                        Relation(
                            from_entity=r.get("from_entity", r.get("from", "")),
                            to_entity=r.get("to_entity", r.get("to", "")),
                            relation_type=r.get("relation_type", r.get("relationType", "")),
                        )
                        for r in relations
                    ]
                    created = mgr.create_relations(rlist, self.context)
                    results["relations_created"] = len(created)

                if observations:
                    obs_data = {
                        "entityName": observations.get("name", ""),
                        "contents": observations.get("observations", []),
                    }
                    obs_results = mgr.add_observations([obs_data], self.context)
                    results["observations_added"] = obs_results

                return _to_json({"success": True, "results": results})

            except Exception as exc:
                return _to_json({"success": False, "error": str(exc)})


# ============================================================================
# RetrieveFromEvidenceGraph
# ============================================================================


class RetrieveFromEvidenceGraph(BaseTool if _BASETOOL_AVAILABLE else object):  # type: ignore[misc]
    """LangChain tool: search / browse the persistent knowledge graph.

    Three retrieval modes (evaluated in order):
    1. ``get_full_map=True`` — full graph as LLM-optimized markdown.
    2. ``query`` — BM25 semantic search over entities.
    3. ``entity_names`` — exact look-up by entity name(s).
    """

    if _BASETOOL_AVAILABLE:
        name: str = "retrieve_from_evidence_graph"
        description: str = (
            "Search or browse the persistent evidence knowledge graph.  "
            "Use ``query`` for BM25-ranked semantic search, "
            "``entity_names`` for exact look-up, or ``get_full_map=True`` "
            "to dump the entire graph as markdown text."
        )
        args_schema: Type[PydanticBaseModel] = _RetrieveFromEvidenceGraphInput

        # --- instance fields ---
        cache_dir: Optional[str] = None
        context: str = "evidence_graph"

        def _run(
            self,
            query: Optional[str] = None,
            entity_names: Optional[str] = None,
            get_full_map: bool = False,
            top_k: int = 10,
            max_entities: Optional[int] = None,
            max_observations_per_entity: int = 5,
        ) -> str:
            """Synchronous entry point (called by LangChain)."""
            mgr = _get_manager(self.cache_dir, self.context)

            try:
                # --- Mode 1: full map ---
                if get_full_map:
                    return mgr.get_text_overview(
                        self.context,
                        max_entities=max_entities,
                        max_observations_per_entity=max_observations_per_entity,
                    )

                # --- Mode 2: BM25 search ---
                if query:
                    sub = mgr.search_nodes(query, self.context, top_k=top_k)
                    return _to_json(sub.to_dict())

                # --- Mode 3: exact entity look-up ---
                if entity_names:
                    try:
                        names: List[str] = json.loads(entity_names)
                    except json.JSONDecodeError:
                        names = [n.strip() for n in entity_names.split(",")]
                    filtered = mgr.open_nodes(names, self.context)
                    return _to_json(filtered.to_dict())

                return _to_json({"error": "Provide query, entity_names, or get_full_map=True."})

            except Exception as exc:
                return _to_json({"success": False, "error": str(exc)})
