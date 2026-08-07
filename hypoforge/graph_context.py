"""Build one provenance-rich, budgeted evidence context for M4-M6."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Literal

from pydantic import BaseModel, Field

from .state import EvidenceNode, KnowledgeEntry, PipelineState, TaskContract


class GraphContextItem(BaseModel):
    entry_id: str
    kind: Literal["established_fact", "conflict", "knowledge_gap", "other"]
    text: str
    source_paper_id: str = ""
    source_paper_title: str = ""
    evidence_ids: list[str] = Field(default_factory=list)
    quotes: list[str] = Field(default_factory=list)


class GraphContext(BaseModel):
    original_question: str = ""
    domains: list[str] = Field(default_factory=list)
    key_entities: list[str] = Field(default_factory=list)
    task_contract: TaskContract = Field(default_factory=TaskContract)
    established_facts: list[GraphContextItem] = Field(default_factory=list)
    conflicts: list[GraphContextItem] = Field(default_factory=list)
    knowledge_gaps: list[GraphContextItem] = Field(default_factory=list)
    relations: list[str] = Field(default_factory=list)
    unresolved_entry_ids: list[str] = Field(default_factory=list)
    available_evidence_ids: list[str] = Field(default_factory=list)
    available_paper_ids: list[str] = Field(default_factory=list)

    @property
    def available_reference_ids(self) -> set[str]:
        entries = {
            item.entry_id
            for bucket in (self.established_facts, self.conflicts, self.knowledge_gaps)
            for item in bucket
        }
        return entries | set(self.available_evidence_ids)

    def render(self, *, max_chars: int = 24000) -> str:
        lines = [
            f"Original question: {self.original_question}",
            f"Domains: {', '.join(self.domains) or '(not supplied)'}",
            f"Key entities: {', '.join(self.key_entities) or '(not supplied)'}",
            "Binding task contract: " + self.task_contract.model_dump_json(),
        ]
        for title, items in (
            ("ESTABLISHED FACTS", self.established_facts),
            ("CONFLICTS", self.conflicts),
            ("KNOWLEDGE GAPS", self.knowledge_gaps),
        ):
            lines.append(f"\n{title}:")
            if not items:
                lines.append("- None available")
            for item in items:
                provenance = []
                if item.source_paper_id:
                    provenance.append(f"paper={item.source_paper_id}")
                if item.evidence_ids:
                    provenance.append("evidence=" + ",".join(item.evidence_ids))
                suffix = f" ({'; '.join(provenance)})" if provenance else ""
                lines.append(f"- [{item.entry_id}] {item.text}{suffix}")
                for quote in item.quotes[:2]:
                    lines.append(f"  quote: {quote}")
        if self.relations:
            lines.append("\nKEY GRAPH RELATIONS:")
            lines.extend(f"- {relation}" for relation in self.relations)
        if self.unresolved_entry_ids:
            lines.append(
                "\nUNRESOLVED GRAPH IDS: " + ", ".join(self.unresolved_entry_ids)
            )
        rendered = "\n".join(lines)
        return rendered[:max_chars]


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def build_graph_context(
    state: PipelineState,
    *,
    max_items_per_bucket: int = 12,
    max_relations: int = 20,
) -> GraphContext:
    """Resolve graph bucket IDs from every authoritative representation.

    The lookup intentionally includes all M2 export runs as well as legacy
    ``literature_results`` and graph-node metadata.  Consequently a follow-up
    run can still render historical IDs instead of leaking bare identifiers.
    """

    card = state.problem_card
    entry_by_id: dict[str, KnowledgeEntry] = {}
    paper_title: dict[str, str] = {}
    evidence_by_id: dict[str, object] = {}

    for result in state.literature_results:
        for entry in result.knowledge_entries:
            entry_by_id[entry.id] = entry
            if entry.source_paper_id and entry.source_paper_title:
                paper_title[entry.source_paper_id] = entry.source_paper_title

    if state.m2_knowledge_export:
        for run in state.m2_knowledge_export.runs:
            for paper in run.papers:
                paper_title[paper.paper_id] = paper.title
            for evidence in run.evidence:
                evidence_by_id[evidence.evidence_id] = evidence
            for entry in run.knowledge_entries:
                entry_by_id[entry.id] = entry

    graph = state.evidence_graph
    nodes = {node.id: node for node in graph.nodes} if graph else {}
    node_for_entry: dict[str, EvidenceNode] = {}
    source_for_node: dict[str, str] = {}
    if graph:
        for node in graph.nodes:
            metadata_entry_id = str((node.metadata or {}).get("entry_id") or "")
            if metadata_entry_id:
                node_for_entry[metadata_entry_id] = node
            if node.id.startswith("N_"):
                node_for_entry[node.id[2:]] = node
        for edge in graph.edges:
            if edge.source.startswith("SRC_"):
                source_for_node.setdefault(edge.target, edge.source[4:])

    unresolved: list[str] = []

    def resolve(entry_id: str, kind: str) -> GraphContextItem | None:
        entry = entry_by_id.get(entry_id)
        node = node_for_entry.get(entry_id)
        if entry:
            ids = _unique(entry.evidence_ids)
            quotes = [
                str(getattr(evidence_by_id[eid], "quote", ""))[:500]
                for eid in ids
                if eid in evidence_by_id
            ]
            return GraphContextItem(
                entry_id=entry_id,
                kind=kind,
                text=entry.content,
                source_paper_id=entry.source_paper_id,
                source_paper_title=(
                    entry.source_paper_title
                    or paper_title.get(entry.source_paper_id, "")
                ),
                evidence_ids=ids,
                quotes=_unique(quotes),
            )
        if node:
            metadata = node.metadata or {}
            evidence_ids = _unique(
                str(value) for value in metadata.get("evidence_ids", [])
            )
            paper_ids = metadata.get("paper_ids", [])
            paper_id = str(metadata.get("paper_id") or "")
            if not paper_id and isinstance(paper_ids, list) and paper_ids:
                paper_id = str(paper_ids[0])
            paper_id = paper_id or source_for_node.get(node.id, "")
            quote = str(metadata.get("quote") or "")[:500]
            return GraphContextItem(
                entry_id=entry_id,
                kind=kind,
                text=node.label,
                source_paper_id=paper_id,
                source_paper_title=paper_title.get(paper_id, ""),
                evidence_ids=evidence_ids,
                quotes=[quote] if quote else [],
            )
        unresolved.append(entry_id)
        return None

    bucket_specs = (
        ("established_facts", "established_fact"),
        ("conflicts", "conflict"),
        ("knowledge_gaps", "knowledge_gap"),
    )
    buckets: dict[str, list[GraphContextItem]] = defaultdict(list)
    if graph:
        for field, kind in bucket_specs:
            for entry_id in getattr(graph, field, [])[:max_items_per_bucket]:
                item = resolve(entry_id, kind)
                if item:
                    buckets[field].append(item)

    relation_lines: list[str] = []
    if graph:
        for edge in graph.edges:
            if len(relation_lines) >= max_relations:
                break
            source = nodes.get(edge.source)
            target = nodes.get(edge.target)
            if not source or not target or edge.relation.value == "involves":
                continue
            evidence_suffix = (
                f" evidence={','.join(edge.evidence_ids)}" if edge.evidence_ids else ""
            )
            relation_lines.append(
                f"[{source.id}] {source.label} --{edge.relation.value}--> "
                f"[{target.id}] {target.label}{evidence_suffix}"
            )

    evidence_ids = _unique(
        [*evidence_by_id]
        + [
            evidence_id
            for bucket in buckets.values()
            for item in bucket
            for evidence_id in item.evidence_ids
        ]
    )
    return GraphContext(
        original_question=(card.original_question if card else state.input_question),
        domains=list(card.domain if card else []),
        key_entities=list(card.key_entities if card else []),
        task_contract=(card.task_contract if card else TaskContract(source="derived")),
        established_facts=buckets["established_facts"],
        conflicts=buckets["conflicts"],
        knowledge_gaps=buckets["knowledge_gaps"],
        relations=relation_lines,
        unresolved_entry_ids=_unique(unresolved),
        available_evidence_ids=evidence_ids,
        available_paper_ids=_unique(paper_title),
    )
