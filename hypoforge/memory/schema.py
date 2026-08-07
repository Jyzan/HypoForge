"""
Data model for the persistent knowledge graph layer.

Adapted from BioDSA's ``biodsa/memory/memory_graph/schema.py``.
Provides Entity-Relation-Observation dataclasses and bidirectional
conversion functions between HypoForge's ``EvidenceGraph`` (Pydantic,
in-memory) and the persistent ``KnowledgeGraph`` (dataclass, JSONL-backed).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Forward reference — imported lazily to avoid circular imports
# ---------------------------------------------------------------------------

EvidenceGraph = None  # set by _ensure_imports()


def _ensure_imports():
    """Lazy-import EvidenceGraph types to avoid circular imports at module level."""
    global EvidenceGraph
    if EvidenceGraph is None:
        from hypoforge.state import (  # noqa: F811
            EvidenceEdge,
            EvidenceEdgeRelation,
            EvidenceGraph as _EvidenceGraph,
            EvidenceNode,
            EvidenceNodeType,
        )
        EvidenceGraph = _EvidenceGraph


# ============================================================================
# Dataclass data model (mirrors BioDSA's schema.py)
# ============================================================================


@dataclass
class Entity:
    """A node in the persistent knowledge graph.

    Mirrors BioDSA's ``Entity`` dataclass with identical serialisation format
    (``entityType`` / ``observations`` keys) so JSONL files are compatible.
    """

    name: str
    entity_type: str
    observations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "entityType": self.entity_type,
            "observations": list(self.observations),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Entity":
        return cls(
            name=data["name"],
            entity_type=data.get("entityType", ""),
            observations=data.get("observations", []),
        )


@dataclass
class Relation:
    """A directed, typed edge in the persistent knowledge graph.

    Serialisation uses ``from`` / ``to`` / ``relationType`` keys for
    BioDSA compatibility.  The optional ``confidence`` / ``rationale``
    fields (added for merge-based knowledge retention) are extra keys
    that older readers can safely ignore.
    """

    from_entity: str
    to_entity: str
    relation_type: str
    confidence: float | None = None
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "from": self.from_entity,
            "to": self.to_entity,
            "relationType": self.relation_type,
        }
        if self.confidence is not None:
            data["confidence"] = self.confidence
        if self.rationale:
            data["rationale"] = self.rationale
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Relation":
        raw_conf = data.get("confidence")
        confidence = (
            float(raw_conf)
            if isinstance(raw_conf, (int, float))
            else None
        )
        return cls(
            from_entity=data["from"],
            to_entity=data["to"],
            relation_type=data["relationType"],
            confidence=confidence,
            rationale=str(data.get("rationale", "") or ""),
        )


@dataclass
class KnowledgeGraph:
    """Container for entities + relations (the persistent unit)."""

    entities: List[Entity] = field(default_factory=list)
    relations: List[Relation] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entities": [e.to_dict() for e in self.entities],
            "relations": [r.to_dict() for r in self.relations],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeGraph":
        entities = [Entity.from_dict(e) for e in data.get("entities", [])]
        relations = [Relation.from_dict(r) for r in data.get("relations", [])]
        return cls(entities=entities, relations=relations)

    @property
    def is_empty(self) -> bool:
        return len(self.entities) == 0 and len(self.relations) == 0


# ============================================================================
# Conversion: EvidenceGraph (Pydantic, in-memory) ↔ KnowledgeGraph (persistent)
# ============================================================================


def evidence_graph_to_knowledge_graph(eg: "EvidenceGraph") -> KnowledgeGraph:
    """Convert HypoForge's in-memory EvidenceGraph to a persistent KnowledgeGraph.

    Mapping rules
    -------------
    * Each ``EvidenceNode`` → one ``Entity``:
      - ``name`` = node.id
      - ``entity_type`` = node.type.value
      - ``observations`` = [node.label, json(metadata)]
    * Each ``EvidenceEdge`` → one ``Relation``:
      - ``from_entity`` = edge.source
      - ``to_entity`` = edge.target
      - ``relation_type`` = edge.relation.value
    * The three bucket lists (``established_facts``, ``conflicts``,
      ``knowledge_gaps``) are stored as observations on a special
      ``_buckets`` meta-entity so they survive round-trips.
    """
    _ensure_imports()

    entities: List[Entity] = []
    relations: List[Relation] = []

    # --- nodes → entities ---
    for node in eg.nodes:
        import json

        observations = [node.label]
        if node.metadata:
            observations.append(json.dumps(node.metadata, ensure_ascii=False))
        entities.append(
            Entity(
                name=node.id,
                entity_type=node.type.value,
                observations=observations,
            )
        )

    # --- edges → relations ---
    for edge in eg.edges:
        relations.append(
            Relation(
                from_entity=edge.source,
                to_entity=edge.target,
                relation_type=edge.relation.value,
                confidence=edge.confidence,
                rationale=edge.rationale or "",
            )
        )

    # --- bucket lists as meta-entity observations ---
    bucket_obs = []
    if eg.established_facts:
        bucket_obs.append("established_facts:" + ",".join(eg.established_facts))
    if eg.conflicts:
        bucket_obs.append("conflicts:" + ",".join(eg.conflicts))
    if eg.knowledge_gaps:
        bucket_obs.append("knowledge_gaps:" + ",".join(eg.knowledge_gaps))
    if bucket_obs:
        entities.append(
            Entity(
                name="_buckets",
                entity_type="meta",
                observations=bucket_obs,
            )
        )

    return KnowledgeGraph(entities=entities, relations=relations)


def knowledge_graph_to_evidence_graph(kg: KnowledgeGraph) -> "EvidenceGraph":
    """Reconstruct an in-memory EvidenceGraph from a persistent KnowledgeGraph.

    This is the inverse of :func:`evidence_graph_to_knowledge_graph`.
    Entities whose ``entity_type`` is ``"meta"`` and ``name == "_buckets"``
    are interpreted as the bucket-list carrier (not as real graph nodes).
    """
    _ensure_imports()
    from hypoforge.state import (  # noqa: F811
        EvidenceEdge,
        EvidenceEdgeRelation,
        EvidenceGraph,
        EvidenceNode,
        EvidenceNodeType,
    )

    import json

    nodes: List[EvidenceNode] = []
    edges: List[EvidenceEdge] = []
    established_facts: List[str] = []
    conflicts: List[str] = []
    knowledge_gaps: List[str] = []

    for entity in kg.entities:
        # --- meta entities ---
        if entity.entity_type == "meta" and entity.name == "_buckets":
            for obs in entity.observations:
                if obs.startswith("established_facts:"):
                    payload = obs[len("established_facts:"):]
                    established_facts = payload.split(",") if payload else []
                elif obs.startswith("conflicts:"):
                    payload = obs[len("conflicts:"):]
                    conflicts = payload.split(",") if payload else []
                elif obs.startswith("knowledge_gaps:"):
                    payload = obs[len("knowledge_gaps:"):]
                    knowledge_gaps = payload.split(",") if payload else []
            continue

        # --- regular entity → EvidenceNode ---
        label = ""
        metadata: Dict[str, Any] = {}
        if entity.observations:
            label = entity.observations[0]
            if len(entity.observations) > 1:
                try:
                    metadata = json.loads(entity.observations[1])
                except (json.JSONDecodeError, IndexError):
                    metadata = {}

        try:
            node_type = EvidenceNodeType(entity.entity_type)
        except ValueError:
            node_type = EvidenceNodeType.EVIDENCE  # fallback

        nodes.append(
            EvidenceNode(
                id=entity.name,
                type=node_type,
                label=label,
                metadata=metadata,
            )
        )

    # --- relations → EvidenceEdge ---
    for rel in kg.relations:
        try:
            edge_rel = EvidenceEdgeRelation(rel.relation_type)
        except ValueError:
            edge_rel = EvidenceEdgeRelation.INVOLVES
        edges.append(
            EvidenceEdge(
                source=rel.from_entity,
                target=rel.to_entity,
                relation=edge_rel,
                confidence=rel.confidence,
                rationale=rel.rationale or None,
            )
        )

    return EvidenceGraph(
        nodes=nodes,
        edges=edges,
        established_facts=established_facts,
        conflicts=conflicts,
        knowledge_gaps=knowledge_gaps,
    )


# ============================================================================
# Utility
# ============================================================================


def calculate_entities_hash(entities: List[Entity]) -> str:
    """Lightweight change-detection hash (mirrors BioDSA).

    Uses entity count + sorted names of first 10 / last 10 entities.
    Not cryptographically strong — just enough to detect meaningful changes.
    """
    if not entities:
        return hashlib.md5(b"").hexdigest()

    hash_input = f"{len(entities)}"
    if len(entities) <= 20:
        names = [e.name for e in entities]
    else:
        names = [e.name for e in entities[:10]] + [e.name for e in entities[-10:]]
    hash_input += "|".join(sorted(names))

    return hashlib.md5(hash_input.encode()).hexdigest()
