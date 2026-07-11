"""
Persistent knowledge-graph layer for HypoForge.

Adapted from BioDSA's ``biodsa/memory/`` subsystem.  Provides four layers:

1. **Data model** — ``Entity`` / ``Relation`` / ``KnowledgeGraph`` dataclasses
2. **Persistence** — ``KnowledgeGraphManager`` (JSONL-backed store)
3. **Search** — ``BM25SearchIndex`` (with disk persistence)
4. **Tools** — ``AddToEvidenceGraph`` / ``RetrieveFromEvidenceGraph``
   (optional LangChain-compatible tool wrappers for LLM agents)

Plus bidirectional conversion between HypoForge's in-memory
``EvidenceGraph`` (Pydantic) and the persistent ``KnowledgeGraph`` format.

Quick start::

    from hypoforge.memory import KnowledgeGraphManager, BM25SearchIndex

    # Persist
    mgr = KnowledgeGraphManager(cache_dir=Path("./kg_cache"))
    mgr.save_from_evidence_graph(evidence_graph, context="run_001")

    # Search
    idx = BM25SearchIndex()
    idx.build_index(mgr.read_graph("run_001").entities)
    results = idx.search("Hsp70 protein folding", top_k=10)
"""

from .bm25_index import BM25SearchIndex, HAS_BM25, HAS_TIKTOKEN
from .graph_manager import KnowledgeGraphManager
from .schema import (
    Entity,
    KnowledgeGraph,
    Relation,
    evidence_graph_to_knowledge_graph,
    knowledge_graph_to_evidence_graph,
)

# Tools are optional (depend on langchain_core)
try:
    from .tools import AddToEvidenceGraph, RetrieveFromEvidenceGraph
except ImportError:
    AddToEvidenceGraph = None  # type: ignore[assignment]
    RetrieveFromEvidenceGraph = None  # type: ignore[assignment]

__all__ = [
    # Schema
    "Entity",
    "KnowledgeGraph",
    "Relation",
    # Persistence
    "KnowledgeGraphManager",
    # Search
    "BM25SearchIndex",
    "HAS_BM25",
    "HAS_TIKTOKEN",
    # Tools (optional)
    "AddToEvidenceGraph",
    "RetrieveFromEvidenceGraph",
    # Conversion
    "evidence_graph_to_knowledge_graph",
    "knowledge_graph_to_evidence_graph",
]
