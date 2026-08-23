"""Deterministic M4-M6 context-budget comparison for regression checks."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from hypoforge.context import ContextPlanner, ContextRequest, estimate_input_tokens
from hypoforge.graph_context import GraphContext, GraphContextItem


def _large_context() -> GraphContext:
    def bucket(kind: str, prefix: str) -> list[GraphContextItem]:
        return [GraphContextItem(
            entry_id=f"{prefix}-{index:02d}",
            kind=kind,
            text=(
                f"{prefix} claim {index}: "
                + "mechanism evidence constraint observation " * 24
            ),
            source_paper_id=f"paper-{prefix}-{index:02d}",
            evidence_ids=[f"ev-{prefix}-{index:02d}"],
            quotes=[
                f"Exact supporting passage {index}: "
                + "measured result with provenance " * 12
            ],
        ) for index in range(12)]

    return GraphContext(
        original_question="How can an evidence-grounded system solve the task?",
        domains=["computer science", "biomedicine"],
        key_entities=["evidence", "hypothesis", "research plan"],
        established_facts=bucket("established_fact", "fact"),
        conflicts=bucket("conflict", "conflict"),
        knowledge_gaps=bucket("knowledge_gap", "gap"),
        relations=[
            f"[N_{index}] source mechanism --supports--> target outcome "
            + "with bounded provenance " * 8
            for index in range(20)
        ],
    )


def measure_context_budget() -> dict[str, object]:
    context = _large_context()
    purposes = [
        "m4_generate",
        "m4_critic",
        "m4_critic",
        "m4_rank",
        "m5_plan",
        "m5_plan",
        "m5_plan",
        "m6_logic",
        "m6_feasibility",
        "m6_sufficiency",
    ]
    legacy_one = estimate_input_tokens(context.render())
    legacy_total = legacy_one * len(purposes)
    planner = ContextPlanner()
    current_total = 0
    missing_focused_ids: list[str] = []
    pack_tokens: list[dict[str, object]] = []
    for index, purpose in enumerate(purposes):
        focused_id = f"fact-{index % 12:02d}"
        pack = planner.plan(context, ContextRequest(
            purpose=purpose,
            focus_entry_ids=(focused_id,),
        ))
        current_total += pack.manifest.estimated_tokens
        if focused_id not in pack.manifest.included_ids:
            missing_focused_ids.append(focused_id)
        pack_tokens.append({
            "purpose": purpose,
            "estimated_tokens": pack.manifest.estimated_tokens,
            "included": len(pack.manifest.included_ids),
            "dropped": len(pack.manifest.dropped_ids),
        })
    reduction = 1.0 - (current_total / legacy_total)
    return {
        "legacy_estimated_tokens": legacy_total,
        "current_estimated_tokens": current_total,
        "reduction_ratio": round(reduction, 4),
        "missing_focused_ids": missing_focused_ids,
        "packs": pack_tokens,
    }


if __name__ == "__main__":
    report = measure_context_budget()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["reduction_ratio"] < 0.50:
        raise SystemExit("Context token reduction is below the 50% gate")
    if report["missing_focused_ids"]:
        raise SystemExit("A focused evidence ID was omitted")
