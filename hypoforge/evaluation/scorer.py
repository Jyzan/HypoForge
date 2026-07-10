"""
Automated scoring pipeline.

Runs all registered metrics against a completed PipelineState and produces
a scoring matrix suitable for ablation comparisons.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import List

from ..state import HypothesisCard, KnowledgeEntry, PipelineState, ResearchPlan
from .metrics import MetricRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_knowledge_entries(state: PipelineState) -> List[KnowledgeEntry]:
    entries: List[KnowledgeEntry] = []
    for lr in state.literature_results:
        entries.extend(lr.knowledge_entries)
    return entries


def _is_async_context() -> bool:
    """Return True if we are inside a running asyncio event loop."""
    try:
        loop = asyncio.get_running_loop()
        return loop.is_running()
    except RuntimeError:
        return False


# ---------------------------------------------------------------------------
# Hypothesis scoring (sync + async)
# ---------------------------------------------------------------------------

async def score_hypothesis_async(
    hypothesis: HypothesisCard,
    knowledge_entries: List[KnowledgeEntry],
    metric_names: List[str] | None = None,
) -> dict[str, float]:
    """Compute all (or selected) metrics for a single hypothesis (async)."""
    if metric_names is None:
        metric_names = MetricRegistry.list_all()

    scores = {}
    for name in metric_names:
        metric_cls = MetricRegistry.get(name)
        if metric_cls is None:
            continue
        metric = metric_cls()
        scores[name] = await metric.compute(hypothesis, knowledge_entries)
    return scores


def score_hypothesis(
    hypothesis: HypothesisCard,
    knowledge_entries: List[KnowledgeEntry],
    metric_names: List[str] | None = None,
) -> dict[str, float]:
    """Synchronous wrapper — safe in both sync and async contexts."""
    if _is_async_context():
        # Fall back to hypothesis's own scores when async; callers in
        # async code should use score_hypothesis_async() directly.
        return (
            hypothesis.scores
            if hypothesis.scores
            else {m: 0.5 for m in (metric_names or MetricRegistry.list_all())}
        )
    return asyncio.run(
        score_hypothesis_async(hypothesis, knowledge_entries, metric_names)
    )


# ---------------------------------------------------------------------------
# Plan completeness
# ---------------------------------------------------------------------------

def score_plan_completeness(plan: ResearchPlan) -> float:
    """Checklist-based completeness score (0.0–1.0)."""
    fields = [
        plan.study_subjects,
        plan.independent_variables,
        plan.dependent_variables,
        plan.control_groups,
        plan.procedures,
        plan.measurement_metrics,
        plan.analysis_methods,
        plan.expected_results_if_supported,
        plan.expected_results_if_refuted,
        plan.timeline,
        plan.risks_and_alternatives,
    ]
    filled = sum(1 for f in fields if f)  # non-empty string or non-empty list
    return filled / len(fields)


# ---------------------------------------------------------------------------
# Full pipeline scoring
# ---------------------------------------------------------------------------

async def score_pipeline_state_async(state: PipelineState) -> dict:
    """Produce a full scoring report (async — preferred)."""
    knowledge_entries = _collect_knowledge_entries(state)

    hypothesis_scores = []
    for h in state.top_hypotheses:
        scores = await score_hypothesis_async(h, knowledge_entries)
        hypothesis_scores.append({"hypothesis_id": h.hypothesis_id, **scores})

    plan_scores = []
    for p in state.research_plans:
        plan_scores.append({
            "hypothesis_id": p.hypothesis_id,
            "plan_completeness": score_plan_completeness(p),
        })

    return {
        "run_id": state.run_id,
        "question": state.input_question,
        "iterations": state.iteration_count,
        "hypothesis_scores": hypothesis_scores,
        "plan_scores": plan_scores,
        "errors": state.errors,
    }


def score_pipeline_state(state: PipelineState) -> dict:
    """Synchronous wrapper — safe in both sync and async contexts."""
    if _is_async_context():
        return {
            "run_id": state.run_id,
            "question": state.input_question,
            "iterations": state.iteration_count,
            "hypothesis_count": len(state.top_hypotheses),
            "plan_count": len(state.research_plans),
            "note": "Use score_pipeline_state_async() from async contexts for full scores.",
        }
    return asyncio.run(score_pipeline_state_async(state))


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_scoring_report(state: PipelineState, output_dir: str = "./output") -> Path:
    """Score the pipeline state and save the report as JSON."""
    if _is_async_context():
        # In async context, save a lightweight summary
        report = score_pipeline_state(state)
    else:
        report = score_pipeline_state(state)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{state.run_id}_scores.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    return out_path
