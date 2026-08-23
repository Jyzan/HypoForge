"""Purpose-aware selection and rendering of provenance-rich graph context."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable

from ..graph_context import GraphContext, GraphContextItem
from ..observability import emit_event
from .models import ContextManifest, ContextPack, ContextRequest
from .token_budget import estimate_input_tokens


@dataclass(frozen=True)
class _PurposeProfile:
    facts: int
    conflicts: int
    gaps: int
    quotes_per_item: int
    relations: int


_PROFILES = {
    "m4_generate": _PurposeProfile(8, 6, 6, 1, 10),
    "m4_critic": _PurposeProfile(4, 8, 4, 1, 8),
    "m4_rank": _PurposeProfile(3, 3, 3, 0, 0),
    "m5_plan": _PurposeProfile(6, 4, 4, 1, 6),
    "m6_logic": _PurposeProfile(5, 6, 3, 1, 6),
    "m6_feasibility": _PurposeProfile(4, 4, 4, 1, 4),
    "m6_sufficiency": _PurposeProfile(6, 6, 6, 0, 8),
}


class ContextPlanner:
    """Select complete evidence cards under a deterministic input budget."""

    def plan(
        self,
        context: GraphContext,
        request: ContextRequest,
    ) -> ContextPack:
        profile = _PROFILES[request.purpose]
        budget = max(1, request.max_input_tokens - request.reserve_tokens)
        focus_entries = set(request.focus_entry_ids)
        focus_evidence = set(request.focus_evidence_ids)

        buckets = {
            "established_facts": list(context.established_facts),
            "conflicts": list(context.conflicts),
            "knowledge_gaps": list(context.knowledge_gaps),
        }
        caps = {
            "established_facts": profile.facts,
            "conflicts": profile.conflicts,
            "knowledge_gaps": profile.gaps,
        }
        source_order = {
            item.entry_id: index
            for index, item in enumerate(
                item for values in buckets.values() for item in values
            )
        }

        def priority(item: GraphContextItem) -> tuple[int, int]:
            focused = (
                item.entry_id in focus_entries
                or bool(set(item.evidence_ids) & focus_evidence)
            )
            return (0 if focused else 1, source_order[item.entry_id])

        selected: dict[str, list[GraphContextItem]] = {
            name: [] for name in buckets
        }
        dropped_reasons: dict[str, str] = {}
        candidates: list[tuple[str, GraphContextItem]] = []
        for name, values in buckets.items():
            candidates.extend((name, item) for item in values)
        candidates.sort(key=lambda pair: priority(pair[1]))

        # Reserve room for section headings and an auditable omitted-ID line.
        content_budget = max(1, budget - 96)
        for bucket_name, item in candidates:
            if len(selected[bucket_name]) >= caps[bucket_name]:
                dropped_reasons[item.entry_id] = "purpose_profile_limit"
                continue
            tentative = {
                name: list(values) for name, values in selected.items()
            }
            tentative[bucket_name].append(item)
            rendered = self._render(
                context,
                request,
                tentative,
                relations=[],
                dropped_ids=[],
                quotes_per_item=profile.quotes_per_item,
            )
            if estimate_input_tokens(rendered) <= content_budget:
                selected[bucket_name].append(item)
            else:
                dropped_reasons[item.entry_id] = "token_budget"

        included_ids = [
            item.entry_id
            for name in buckets
            for item in selected[name]
        ]
        all_ids = [
            item.entry_id
            for name in buckets
            for item in buckets[name]
        ]
        dropped_ids = [entry_id for entry_id in all_ids if entry_id not in included_ids]
        for entry_id in dropped_ids:
            dropped_reasons.setdefault(entry_id, "token_budget")

        relations: list[str] = []
        for relation in context.relations[: profile.relations]:
            tentative = [*relations, relation]
            rendered = self._render(
                context,
                request,
                selected,
                relations=tentative,
                dropped_ids=dropped_ids,
                quotes_per_item=profile.quotes_per_item,
            )
            if estimate_input_tokens(rendered) <= budget:
                relations.append(relation)
            else:
                break

        rendered = self._render(
            context,
            request,
            selected,
            relations=relations,
            dropped_ids=dropped_ids,
            quotes_per_item=profile.quotes_per_item,
        )
        while estimate_input_tokens(rendered) > budget and included_ids:
            remove_id = included_ids.pop()
            for values in selected.values():
                values[:] = [item for item in values if item.entry_id != remove_id]
            if remove_id not in dropped_ids:
                dropped_ids.append(remove_id)
            dropped_reasons[remove_id] = "token_budget"
            rendered = self._render(
                context,
                request,
                selected,
                relations=relations,
                dropped_ids=dropped_ids,
                quotes_per_item=profile.quotes_per_item,
            )

        estimated = estimate_input_tokens(rendered)
        return ContextPack(
            rendered=rendered,
            manifest=ContextManifest(
                purpose=request.purpose,
                budget_tokens=budget,
                estimated_tokens=estimated,
                included_ids=included_ids,
                dropped_ids=dropped_ids,
                drop_reasons=dropped_reasons,
                render_hash=hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
            ),
        )

    @staticmethod
    def _render(
        context: GraphContext,
        request: ContextRequest,
        selected: dict[str, list[GraphContextItem]],
        *,
        relations: Iterable[str],
        dropped_ids: list[str],
        quotes_per_item: int,
    ) -> str:
        lines = [
            f"Context purpose: {request.purpose}",
            f"Original question: {context.original_question}",
            f"Domains: {', '.join(context.domains) or '(not supplied)'}",
            f"Key entities: {', '.join(context.key_entities) or '(not supplied)'}",
            "Binding task contract: " + context.task_contract.model_dump_json(),
        ]
        for bucket_name, title in (
            ("established_facts", "ESTABLISHED FACTS"),
            ("conflicts", "CONFLICTS"),
            ("knowledge_gaps", "KNOWLEDGE GAPS"),
        ):
            lines.append(f"\n{title}:")
            items = selected[bucket_name]
            if not items:
                lines.append("- None selected")
            for item in items:
                provenance = []
                if item.source_paper_id:
                    provenance.append(f"paper={item.source_paper_id}")
                if item.evidence_ids:
                    provenance.append("evidence=" + ",".join(item.evidence_ids))
                suffix = f" ({'; '.join(provenance)})" if provenance else ""
                lines.append(f"- [{item.entry_id}] {item.text}{suffix}")
                for quote in item.quotes[:quotes_per_item]:
                    lines.append(f"  quote: {quote}")
        relation_list = list(relations)
        if relation_list:
            lines.append("\nKEY GRAPH RELATIONS:")
            lines.extend(f"- {relation}" for relation in relation_list)
        if dropped_ids:
            lines.append("\nOMITTED ENTRY IDS: " + ", ".join(dropped_ids))
        if context.unresolved_entry_ids:
            lines.append(
                "UNRESOLVED GRAPH IDS: " + ", ".join(context.unresolved_entry_ids)
            )
        return "\n".join(lines)


def emit_context_built(module: str, tool: str, pack: ContextPack) -> None:
    """Record a compact manifest without persisting the raw prompt."""

    manifest = pack.manifest
    emit_event(
        "llm_context_built",
        module=module,
        tool=tool,
        status="completed",
        message=f"Context pack built for {manifest.purpose}",
        details={
            "purpose": manifest.purpose,
            "budget_tokens": manifest.budget_tokens,
            "estimated_tokens": manifest.estimated_tokens,
            "included_count": len(manifest.included_ids),
            "dropped_count": len(manifest.dropped_ids),
            "included_ids": list(manifest.included_ids),
            "dropped_ids": list(manifest.dropped_ids),
            "render_hash": manifest.render_hash,
        },
    )
