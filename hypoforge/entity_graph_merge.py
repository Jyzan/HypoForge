"""Auditable final-graph entity merging for M3.

This post-processor is deliberately separate from pre-graph entity
normalisation.  It works only on ``ENTITY`` nodes in an already-built graph,
returns a deep copy, and records every destructive graph rewrite.
"""

from __future__ import annotations

import hashlib
import math
import re
import string
import unicodedata
from copy import deepcopy
from dataclasses import dataclass
from typing import Awaitable, Callable, Sequence

from .state import (
    EvidenceEdge,
    EvidenceGraph,
    EvidenceNode,
    EvidenceNodeType,
    EntityMergeMember,
    EntityMergeRecord,
)


EmbedTexts = Callable[[list[str], str], Awaitable[list[list[float]]]]

_OUTER_PUNCTUATION = string.punctuation + "，。！？；：、‘’“”（）【】《》〈〉"


def _normalise_name(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value or ""))
    value = re.sub(r"\s+", " ", value).strip()
    return value.strip(_OUTER_PUNCTUATION + " ").casefold()


def _unique_strings(values: Sequence[object]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


@dataclass(frozen=True)
class EntityMergeOutcome:
    graph: EvidenceGraph
    merge_count: int
    entity_count_before: int
    entity_count_after: int
    redirected_edge_count: int
    deduplicated_edge_count: int
    removed_self_loop_count: int
    degraded_reason: str = ""


class EntityGraphMerger:
    """Merge equivalent final-graph entity nodes without mutating input."""

    def __init__(
        self,
        *,
        threshold: float = 0.92,
        embedding_model: str = "",
        embed_texts: EmbedTexts | None = None,
    ) -> None:
        self.threshold = float(threshold)
        self.embedding_model = str(embedding_model or "")
        self.embed_texts = embed_texts

    async def merge(
        self,
        graph: EvidenceGraph,
        *,
        task_entity_names: Sequence[str],
        round_index: int,
    ) -> EntityMergeOutcome:
        working = graph.model_copy(deep=True)
        entities = [
            node
            for node in working.nodes
            if node.type == EvidenceNodeType.ENTITY and _normalise_name(node.label)
        ]
        entity_count_before = len(entities)
        by_name: dict[str, list[EvidenceNode]] = {}
        for node in entities:
            by_name.setdefault(_normalise_name(node.label), []).append(node)

        degree = self._degree_by_node(working)
        task_names = {_normalise_name(name) for name in task_entity_names}
        exact_groups: list[list[EvidenceNode]] = []
        for normalised_name in sorted(by_name):
            members = by_name[normalised_name]
            exact_groups.append(
                sorted(
                    members,
                    key=lambda node: self._canonical_rank(
                        node, task_names=task_names, degree=degree
                    ),
                )
            )

        raw_to_representative = {
            member.id: group[0].id
            for group in exact_groups
            for member in group
        }
        similarities: dict[tuple[str, str], float] = {}
        semantic_group_indices: list[list[int]] = [[index] for index in range(len(exact_groups))]
        degraded_reason = ""
        if len(exact_groups) > 1:
            if self.embed_texts is None or not self.embedding_model:
                degraded_reason = "embedding backend or model is not configured"
            else:
                try:
                    representative_names = [
                        _normalise_name(group[0].label) for group in exact_groups
                    ]
                    vectors = await self.embed_texts(
                        representative_names,
                        self.embedding_model,
                    )
                    self._validate_vectors(vectors, len(exact_groups))
                    similarities = self._pairwise_similarities(exact_groups, vectors)
                    semantic_group_indices = self._complete_linkage_clusters(
                        exact_groups,
                        similarities,
                    )
                except Exception as exc:
                    degraded_reason = f"{type(exc).__name__}: {exc}"

        clusters: list[list[EvidenceNode]] = []
        for group_indices in semantic_group_indices:
            members = [
                member
                for group_index in group_indices
                for member in exact_groups[group_index]
            ]
            if len(members) > 1:
                clusters.append(members)

        if not clusters:
            return EntityMergeOutcome(
                graph=working,
                merge_count=0,
                entity_count_before=entity_count_before,
                entity_count_after=entity_count_before,
                redirected_edge_count=0,
                deduplicated_edge_count=0,
                removed_self_loop_count=0,
                degraded_reason=degraded_reason,
            )

        rewritten, stats = self._rewrite_graph(
            working,
            clusters=clusters,
            task_names=task_names,
            degree=degree,
            round_index=max(0, int(round_index)),
            similarities=similarities,
            raw_to_representative=raw_to_representative,
        )
        return EntityMergeOutcome(
            graph=rewritten,
            merge_count=len(clusters),
            entity_count_before=entity_count_before,
            entity_count_after=sum(
                1 for node in rewritten.nodes if node.type == EvidenceNodeType.ENTITY
            ),
            redirected_edge_count=stats[0],
            deduplicated_edge_count=stats[1],
            removed_self_loop_count=stats[2],
            degraded_reason=degraded_reason,
        )

    @staticmethod
    def _validate_vectors(vectors: list[list[float]], expected: int) -> None:
        if len(vectors) != expected:
            raise ValueError(
                f"embedding count mismatch: expected {expected}, got {len(vectors)}"
            )
        if not vectors or not vectors[0]:
            raise ValueError("embedding vectors must be non-empty")
        dimension = len(vectors[0])
        for vector in vectors:
            if len(vector) != dimension:
                raise ValueError("embedding vector dimensions do not match")
            if not all(math.isfinite(float(value)) for value in vector):
                raise ValueError("embedding vectors must contain finite numbers")
            if not any(float(value) != 0.0 for value in vector):
                raise ValueError("embedding vectors must have non-zero magnitude")

    @staticmethod
    def _pairwise_similarities(
        exact_groups: list[list[EvidenceNode]],
        vectors: list[list[float]],
    ) -> dict[tuple[str, str], float]:
        similarities: dict[tuple[str, str], float] = {}
        for left_index, left_group in enumerate(exact_groups):
            left_id = left_group[0].id
            similarities[(left_id, left_id)] = 1.0
            for right_index in range(left_index + 1, len(exact_groups)):
                right_id = exact_groups[right_index][0].id
                score = cosine_similarity(vectors[left_index], vectors[right_index])
                similarities[(left_id, right_id)] = score
                similarities[(right_id, left_id)] = score
        return similarities

    def _complete_linkage_clusters(
        self,
        exact_groups: list[list[EvidenceNode]],
        similarities: dict[tuple[str, str], float],
    ) -> list[list[int]]:
        clusters: list[list[int]] = [[index] for index in range(len(exact_groups))]
        candidates: list[tuple[float, str, str, int, int]] = []
        for left_index in range(len(exact_groups)):
            left_id = exact_groups[left_index][0].id
            for right_index in range(left_index + 1, len(exact_groups)):
                right_id = exact_groups[right_index][0].id
                score = similarities[(left_id, right_id)]
                candidates.append(
                    (score, min(left_id, right_id), max(left_id, right_id), left_index, right_index)
                )
        candidates.sort(key=lambda item: (-round(item[0], 12), item[1], item[2]))

        for _score, _left_id, _right_id, left_index, right_index in candidates:
            left_cluster = next(
                (cluster for cluster in clusters if left_index in cluster),
                None,
            )
            right_cluster = next(
                (cluster for cluster in clusters if right_index in cluster),
                None,
            )
            if left_cluster is None or right_cluster is None or left_cluster is right_cluster:
                continue
            complete_score = min(
                similarities[
                    (
                        exact_groups[left_member][0].id,
                        exact_groups[right_member][0].id,
                    )
                ]
                for left_member in left_cluster
                for right_member in right_cluster
            )
            if complete_score < self.threshold:
                continue
            merged = sorted([*left_cluster, *right_cluster])
            clusters = [
                cluster
                for cluster in clusters
                if cluster is not left_cluster and cluster is not right_cluster
            ]
            clusters.append(merged)
            clusters.sort(key=lambda cluster: min(cluster))
        return clusters

    @staticmethod
    def _degree_by_node(graph: EvidenceGraph) -> dict[str, int]:
        degree: dict[str, int] = {}
        for edge in graph.edges:
            degree[edge.source] = degree.get(edge.source, 0) + 1
            degree[edge.target] = degree.get(edge.target, 0) + 1
        return degree

    @staticmethod
    def _canonical_rank(
        node: EvidenceNode,
        *,
        task_names: set[str],
        degree: dict[str, int],
    ) -> tuple[int, int, int, int, str]:
        normalised = _normalise_name(node.label)
        return (
            0 if normalised in task_names else 1,
            0 if node.id.startswith("ENT_") else 1,
            -degree.get(node.id, 0),
            len(normalised),
            node.id,
        )

    def _rewrite_graph(
        self,
        graph: EvidenceGraph,
        *,
        clusters: list[list[EvidenceNode]],
        task_names: set[str],
        degree: dict[str, int],
        round_index: int,
        similarities: dict[tuple[str, str], float],
        raw_to_representative: dict[str, str],
    ) -> tuple[EvidenceGraph, tuple[int, int, int]]:
        original_metadata = {
            node.id: deepcopy(node.metadata)
            for members in clusters
            for node in members
        }
        canonical_by_old_id: dict[str, str] = {}
        canonical_by_cluster: list[tuple[EvidenceNode, list[EvidenceNode]]] = []
        for members in clusters:
            ordered = sorted(
                members,
                key=lambda node: self._canonical_rank(
                    node, task_names=task_names, degree=degree
                ),
            )
            canonical = ordered[0]
            canonical_by_cluster.append((canonical, ordered))
            for member in ordered:
                canonical_by_old_id[member.id] = canonical.id

        removed_ids = {
            member.id
            for canonical, members in canonical_by_cluster
            for member in members
            if member.id != canonical.id
        }
        nodes_by_id = {node.id: node for node in graph.nodes}
        for canonical, members in canonical_by_cluster:
            canonical_node = nodes_by_id[canonical.id]
            metadata = dict(canonical_node.metadata)
            metadata["aliases"] = _unique_strings(
                [*metadata.get("aliases", []), *(member.label for member in members)]
            )
            metadata["merged_node_ids"] = _unique_strings(
                [
                    *metadata.get("merged_node_ids", []),
                    *(member.id for member in members if member.id != canonical.id),
                ]
            )
            metadata["normalised_from"] = _unique_strings(
                source
                for member in members
                for source in member.metadata.get("normalised_from", [])
            )
            metadata["entry_ids"] = _unique_strings(
                entry_id
                for member in members
                for entry_id in [
                    *(
                        member.metadata.get("entry_ids", [])
                        if isinstance(member.metadata.get("entry_ids", []), list)
                        else [member.metadata.get("entry_ids", "")]
                    ),
                    member.metadata.get("entry_id", ""),
                    member.id[2:] if member.id.startswith("N_") else "",
                ]
            )
            canonical_node.metadata = metadata

        new_nodes = [node for node in graph.nodes if node.id not in removed_ids]
        record_edge_stats = {
            canonical.id: [0, 0, 0]
            for canonical, members in canonical_by_cluster
        }
        rewritten_edges: list[tuple[EvidenceEdge, set[str]]] = []
        redirected = 0
        removed_self_loops = 0
        for edge in graph.edges:
            new_source = canonical_by_old_id.get(edge.source, edge.source)
            new_target = canonical_by_old_id.get(edge.target, edge.target)
            touched_groups: set[str] = set()
            if new_source != edge.source:
                touched_groups.add(new_source)
            if new_target != edge.target:
                touched_groups.add(new_target)
            if edge.source != edge.target and new_source == new_target:
                removed_self_loops += 1
                for canonical_id in touched_groups:
                    record_edge_stats[canonical_id][2] += 1
                continue
            if new_source != edge.source or new_target != edge.target:
                redirected += 1
                for canonical_id in touched_groups:
                    record_edge_stats[canonical_id][0] += 1
            rewritten_edges.append(
                (
                    edge.model_copy(update={"source": new_source, "target": new_target}),
                    touched_groups,
                )
            )

        deduplicated = 0
        edge_index: dict[tuple[str, str, object], EvidenceEdge] = {}
        edge_touched_groups: dict[tuple[str, str, object], set[str]] = {}
        edge_order: list[tuple[str, str, object]] = []
        for edge, touched_groups in rewritten_edges:
            key = (edge.source, edge.target, edge.relation)
            existing = edge_index.get(key)
            if existing is None:
                edge_index[key] = edge.model_copy(deep=True)
                edge_touched_groups[key] = set(touched_groups)
                edge_order.append(key)
                continue
            deduplicated += 1
            responsible_groups = edge_touched_groups[key] | touched_groups
            for canonical_id in responsible_groups:
                record_edge_stats[canonical_id][1] += 1
            edge_touched_groups[key].update(touched_groups)
            existing.evidence_ids = _unique_strings(
                [*existing.evidence_ids, *edge.evidence_ids]
            )
            confidence_values = [
                value
                for value in (existing.confidence, edge.confidence)
                if value is not None
            ]
            existing.confidence = max(confidence_values) if confidence_values else None
            rationales = _unique_strings([existing.rationale, edge.rationale])
            existing.rationale = " | ".join(rationales) or None

        node_count_before = len(graph.nodes)
        node_count_after = len(new_nodes)
        new_records: list[EntityMergeRecord] = []
        for canonical, members in canonical_by_cluster:
            cluster_redirected, cluster_deduplicated, cluster_self_loops = record_edge_stats[
                canonical.id
            ]
            member_ids = sorted(member.id for member in members)
            digest = hashlib.sha256(
                f"{round_index}|{'|'.join(member_ids)}".encode("utf-8")
            ).hexdigest()[:16]
            member_snapshots: list[EntityMergeMember] = []
            canonical_representative = raw_to_representative[canonical.id]
            for member in members:
                similarity = 1.0
                if member.id != canonical.id:
                    member_representative = raw_to_representative[member.id]
                    similarity = similarities.get(
                        (canonical_representative, member_representative),
                        similarities.get(
                            (member_representative, canonical_representative),
                            1.0,
                        ),
                    )
                member_snapshots.append(
                    EntityMergeMember(
                        node_id=member.id,
                        name=member.label.strip(),
                        similarity_to_canonical=similarity,
                        metadata=original_metadata[member.id],
                    )
                )
            new_records.append(
                EntityMergeRecord(
                    merge_id=f"entity-merge-{digest}",
                    round_index=round_index,
                    threshold=self.threshold,
                    embedding_model=self.embedding_model,
                    method=(
                        "embedding_cosine"
                        if len({_normalise_name(member.label) for member in members}) > 1
                        else "exact"
                    ),
                    canonical_node_id=canonical.id,
                    canonical_name=canonical.label.strip(),
                    members=member_snapshots,
                    node_count_before=node_count_before,
                    node_count_after=node_count_after,
                    redirected_edge_count=cluster_redirected,
                    deduplicated_edge_count=cluster_deduplicated,
                    removed_self_loop_count=cluster_self_loops,
                )
            )

        graph.nodes = new_nodes
        graph.edges = [edge_index[key] for key in edge_order]
        graph.entity_merge_log.extend(new_records)
        graph.version += 1
        return graph, (redirected, deduplicated, removed_self_loops)


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Return cosine similarity, rejecting zero or mismatched vectors."""
    if not left or len(left) != len(right):
        raise ValueError("embedding vectors must be non-empty and equal length")
    dot = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = math.sqrt(sum(float(value) ** 2 for value in left))
    right_norm = math.sqrt(sum(float(value) ** 2 for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        raise ValueError("embedding vectors must have non-zero magnitude")
    return dot / (left_norm * right_norm)
