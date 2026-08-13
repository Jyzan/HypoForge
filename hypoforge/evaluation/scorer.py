"""
Automated scoring pipeline.

Runs the registered *independent* metrics against a completed
``PipelineState`` and produces a scoring report suitable for ablation
comparisons and the competition scoring matrix.

The report deliberately separates two things that used to be conflated:

* ``self_reported`` — the scores M4 gave *itself* (kept for transparency, but
  clearly labelled — it is **not** an independent evaluation).
* ``independent``   — scores recomputed here from objective signals only.

The synchronous wrappers raise inside a running event loop rather than
silently returning a degenerate placeholder; the pipeline uses the async
functions directly.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from ..state import KnowledgeEntry, PipelineState, ResearchPlan
from ..graph_context import build_graph_context
from ..task_alignment import assess_task_alignment
from .metrics import MetricRegistry
from .rubric import (
    HYPOTHESIS_RUBRIC,
    composite_score,
    normalise_weights,
)


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
        return asyncio.get_running_loop().is_running()
    except RuntimeError:
        return False


# ---------------------------------------------------------------------------
# Hypothesis scoring
# ---------------------------------------------------------------------------

async def score_hypothesis_async(
    hypothesis: Any,
    knowledge_entries: List[KnowledgeEntry],
    weights: Optional[Mapping[str, float]] = None,
    llm_config: Any = None,
    embed_config: Any = None,
    evidence_graph: Optional[Any] = None,
) -> Dict[str, Any]:
    """Score a single hypothesis.

    Returns a dict with the generator's self-reported scores, any
    independently-recomputed metrics, and the (weighted) composite over the
    self-reported dimensions.
    """
    self_reported = dict(hypothesis.scores or {})

    independent: Dict[str, float] = {}
    traces: Dict[str, Any] = {}
    for name in MetricRegistry.list_all():
        metric_cls = MetricRegistry.get(name)
        if metric_cls is None:
            continue
        if not getattr(metric_cls, "implemented", True):
            continue
        if not getattr(metric_cls, "independent", True):
            continue
        try:
            # Metrics that accept an LLM client receive one so they can
            # evaluate quality with a lightweight model rather than just
            # checking for non-empty fields.
            kwargs = {}
            if llm_config: kwargs['llm_config'] = llm_config
            if embed_config: kwargs['embed_config'] = embed_config
            metric = metric_cls(**kwargs)
            result = await metric.compute(
                hypothesis, 
                knowledge_entries, 
                evidence_graph=evidence_graph
            )
            if isinstance(result, tuple) and len(result) == 2:
                independent[name] = float(result[0])
                traces[name] = result[1]
            else:
                independent[name] = float(result)
                traces[name] = None
        except NotImplementedError:
            continue

    return {
        "hypothesis_id": hypothesis.hypothesis_id,
        "self_reported": self_reported,
        "independent": independent,
        "composite": composite_score(self_reported, weights) if self_reported else None,
        "traces": traces,
    }


# ---------------------------------------------------------------------------
# Plan completeness (placeholder-aware)
# ---------------------------------------------------------------------------

_PLACEHOLDERS = {"", "tbd", "todo", "n/a", "na", "none", "null", "?", "待定", "无", "暂无", "略"}


def _is_filled(value: Any) -> bool:
    """True if *value* carries real content (not empty / a placeholder / too short)."""
    if value is None:
        return False
    if isinstance(value, str):
        v = value.strip().lower()
        return len(v) >= 3 and v not in _PLACEHOLDERS
    if isinstance(value, (list, tuple, set)):
        return any(_is_filled(item) for item in value)
    if isinstance(value, dict):
        return any(_is_filled(item) for item in value.values())
    return bool(value)


def score_plan_completeness(plan: ResearchPlan) -> float:
    """Checklist-based completeness score (0.0–1.0) over the 11 plan elements.

    Unlike a naive truthiness check, placeholder strings ("TBD", "待定", …)
    and sub-3-character stubs do *not* count as filled.
    """
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
    filled = sum(1 for f in fields if _is_filled(f))
    return round(filled / len(fields), 4)


# ---------------------------------------------------------------------------
# Aggregate (one comparable number per run, for the ablation matrix)
# ---------------------------------------------------------------------------

def _latest_overall_review(state: PipelineState) -> Optional[float]:
    """The M6 'overall' review score (1–5) from the most recent iteration."""
    if not state.reviews:
        return None
    max_version = max(r.version for r in state.reviews)
    overalls = [
        r.score for r in state.reviews
        if r.version == max_version and r.dimension.value == "overall"
    ]
    if not overalls:
        return None
    return round(sum(overalls) / len(overalls), 2)


def _aggregate(
    state: PipelineState,
    hypothesis_scores: List[Dict[str, Any]],
    plan_scores: List[Dict[str, Any]],
) -> Dict[str, Any]:
    composites = [h["composite"] for h in hypothesis_scores if h.get("composite") is not None]
    completeness = [p["plan_completeness"] for p in plan_scores]
    return {
        "top1_composite": composites[0] if composites else None,
        "mean_composite": round(sum(composites) / len(composites), 4) if composites else None,
        "mean_plan_completeness": round(sum(completeness) / len(completeness), 4) if completeness else None,
        "latest_overall_review": _latest_overall_review(state),
    }


def _quality_gates(state: PipelineState) -> Dict[str, Any]:
    """Independent object/evidence/coverage/source gates (O-09)."""

    context = build_graph_context(state)
    valid_evidence = set(context.available_evidence_ids)
    alignment_rows = []
    for hypothesis in state.top_hypotheses:
        plan = next(
            (item for item in state.research_plans if item.hypothesis_id == hypothesis.hypothesis_id),
            None,
        )
        hypothesis_assessment = assess_task_alignment(
            state,
            hypothesis.model_dump_json(exclude={"task_trace"}),
            subject_text="\n".join([hypothesis.statement, hypothesis.mechanism]),
            trace=hypothesis.task_trace,
            require_contract=False,
        )
        plan_assessment = (
            assess_task_alignment(
                state,
                plan.model_dump_json(exclude={"task_trace"}),
                subject_text=plan.study_subjects,
                trace=plan.task_trace,
                require_contract=False,
            ) if plan else None
        )
        passed = hypothesis_assessment.passed and bool(
            plan_assessment and plan_assessment.passed
        )
        score = min(
            hypothesis_assessment.score,
            plan_assessment.score if plan_assessment else 0.0,
        )
        alignment_rows.append({
            "hypothesis_id": hypothesis.hypothesis_id,
            "passed": passed,
            "score": score,
            "matched_anchors": list(hypothesis_assessment.matched_anchors),
            "conflicts": list(hypothesis_assessment.conflicts)
            + (list(plan_assessment.conflicts) if plan_assessment else []),
            "rationale": " ".join(filter(None, [
                f"Hypothesis: {hypothesis_assessment.rationale}",
                f"Plan: {plan_assessment.rationale}" if plan_assessment else "Plan: missing",
            ])),
        })

    claim_units = 0
    supported_units = 0
    for hypothesis in state.top_hypotheses:
        claim_units += 1
        cited = set(hypothesis.supporting_evidence)
        if cited and all(item in valid_evidence for item in cited):
            supported_units += 1
    for plan in state.research_plans:
        claim_units += 1
        cited = set(plan.supporting_evidence_ids)
        cited.update(
            evidence_id
            for link in plan.evidence_links
            for evidence_id in link.supporting_evidence_ids
            if link.support_status == "supported"
        )
        if cited and all(item in valid_evidence for item in cited):
            supported_units += 1
    evidence_coverage = supported_units / claim_units if claim_units else 0.0

    plan_completeness = [score_plan_completeness(plan) for plan in state.research_plans]
    anchor_coverage = [row["score"] for row in alignment_rows]
    answer_completeness = (
        sum(plan_completeness) / len(plan_completeness)
        if plan_completeness else 0.0
    )

    paper_quality: list[float] = []
    if state.m2_knowledge_export:
        for run in state.m2_knowledge_export.runs:
            for paper in run.papers:
                semantic = paper.rank_scores.get("scout_relevance")
                if isinstance(semantic, (int, float)):
                    paper_quality.append(float(semantic))
                else:
                    paper_quality.append(
                        1.0 if paper.content_level in {"structured_fulltext", "pdf", "html"}
                        else 0.6 if paper.content_level == "abstract" else 0.0
                    )
    source_quality = sum(paper_quality) / len(paper_quality) if paper_quality else 0.0

    return {
        "task_alignment": {
            "passed": bool(alignment_rows) and all(row["passed"] for row in alignment_rows),
            "items": alignment_rows,
        },
        "evidence_coverage": round(evidence_coverage, 4),
        "answer_completeness": round(answer_completeness, 4),
        "source_quality": round(source_quality, 4),
    }


# ---------------------------------------------------------------------------
# Full pipeline scoring
# ---------------------------------------------------------------------------

async def score_pipeline_state_async(
    state: PipelineState,
    weights: Optional[Mapping[str, float]] = None,
    llm_config: Any = None,
    embed_config: Any = None,
) -> Dict[str, Any]:
    """Produce a full scoring report (async — the canonical entry point)."""
    knowledge_entries = _collect_knowledge_entries(state)

    hypothesis_scores = [
        await score_hypothesis_async(
            h,
            knowledge_entries,
            weights,
            llm_config=llm_config,
            embed_config=embed_config,
            evidence_graph=state.evidence_graph
        )
        for h in state.top_hypotheses
    ]
    plan_scores = [
        {"hypothesis_id": p.hypothesis_id, "plan_completeness": score_plan_completeness(p)}
        for p in state.research_plans
    ]

    quality_gates = _quality_gates(state)
    aggregate = _aggregate(state, hypothesis_scores, plan_scores)
    aggregate.update({
        "task_alignment_passed": quality_gates["task_alignment"]["passed"],
        "evidence_coverage": quality_gates["evidence_coverage"],
        "answer_completeness": quality_gates["answer_completeness"],
        "source_quality": quality_gates["source_quality"],
    })
    return {
        "run_id": state.run_id,
        "question": state.input_question,
        "iterations": state.iteration_count,
        "hypothesis_scores": hypothesis_scores,
        "plan_scores": plan_scores,
        "aggregate": aggregate,
        "quality_gates": quality_gates,
        "rubric": {
            "weights": normalise_weights(weights),
            "dimensions": HYPOTHESIS_RUBRIC,
            "note": (
                "self_reported = M4 generator self-assessment (not an independent "
                "evaluation); independent = objectively recomputed metrics only."
            ),
        },
        "errors": state.errors,
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

async def save_scoring_report_async(
    state: PipelineState,
    output_dir: str = "./output",
    weights: Optional[Mapping[str, float]] = None,
    llm_config: Any = None,
    embed_config: Any = None,
) -> Path:
    """Convenience wrapper to score a state and write the JSON report."""
    report = await score_pipeline_state_async(
        state, weights=weights, llm_config=llm_config, embed_config=embed_config
    )
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{state.run_id}_scores.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    return out_path


# ---------------------------------------------------------------------------
# Synchronous wrappers (for use *outside* an event loop only)
# ---------------------------------------------------------------------------

def score_pipeline_state(
    state: PipelineState,
    weights: Optional[Mapping[str, float]] = None,
) -> Dict[str, Any]:
    """Synchronous wrapper — only valid outside a running event loop."""
    if _is_async_context():
        raise RuntimeError(
            "score_pipeline_state() cannot run inside an event loop; "
            "await score_pipeline_state_async() instead."
        )
    return asyncio.run(score_pipeline_state_async(state, weights))


def save_scoring_report(
    state: PipelineState,
    output_dir: str = "./output",
    weights: Optional[Mapping[str, float]] = None,
) -> Path:
    """Synchronous wrapper — only valid outside a running event loop."""
    if _is_async_context():
        raise RuntimeError(
            "save_scoring_report() cannot run inside an event loop; "
            "await save_scoring_report_async() instead."
        )
    return asyncio.run(save_scoring_report_async(state, output_dir, weights))
