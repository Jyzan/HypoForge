"""Reusable Rich rendering functions for HypoForge pipeline output."""

from __future__ import annotations

from typing import List

from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich import box

from . import COLORS, console

# Import state models for type-safe rendering
from ..state import (
    EvidenceGraph,
    HypothesisCard,
    PipelineState,
    ProblemCard,
    ResearchPlan,
    ReviewResult,
)


# ============================================================================
# Phase / Module headers
# ============================================================================

_PHASE_ICONS = {"running": "▶", "done": "✓", "error": "✗"}
_PHASE_COLORS = {"running": "primary", "done": "success", "error": "error"}


def render_phase_header(
    module_name: str,
    title: str,
    status: str = "running",
) -> None:
    """Print a stylised rule marking the start (or end) of a module phase.

    Example output::

        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        ▶ [M1] Problem Understanding — 问题理解与分解
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """
    icon = _PHASE_ICONS.get(status, "▶")
    color = COLORS[_PHASE_COLORS.get(status, "primary")]
    console.print()
    console.print(Rule(
        f"[bold {color}]{icon} [{module_name.upper()}] {title}",
        style=color,
        characters="━",
    ))


def render_phase_done(module_name: str) -> None:
    """Shorthand for a completed phase."""
    console.print(f"  [{COLORS['success']}]✓ [{module_name.upper()}] complete[/{COLORS['success']}]")


# ============================================================================
# Problem Card (M1)
# ============================================================================

def render_problem_card(card: ProblemCard) -> None:
    """Print the structured problem decomposition."""
    console.print(f"  [{COLORS['muted']}]Domain:[/] {', '.join(card.domain)}")
    console.print(f"  [{COLORS['muted']}]Sub-questions:[/] {len(card.sub_questions)}")
    for i, sq in enumerate(card.sub_questions, 1):
        console.print(f"    {i}. {sq}")
    console.print(f"  [{COLORS['muted']}]Key entities:[/] {', '.join(card.key_entities)}")
    console.print(f"  [{COLORS['muted']}]Question type:[/] {card.question_type.value}")


# ============================================================================
# Literature Search (M2)
# ============================================================================

def render_literature_summary(
    papers_retrieved: int,
    knowledge_entries: int,
    sub_questions: int,
) -> None:
    """Print a one-line summary of literature search results."""
    console.print(
        f"  [{COLORS['success']}]✓ Retrieved {papers_retrieved} papers "
        f"across {sub_questions} sub-questions, "
        f"extracted {knowledge_entries} knowledge entries[/{COLORS['success']}]"
    )


# ============================================================================
# Evidence Graph (M3)
# ============================================================================

def render_evidence_summary(graph: EvidenceGraph) -> None:
    """Print a Rich table summarising the evidence graph."""
    table = Table(
        title="📊 Evidence Graph Summary",
        box=box.ROUNDED,
        title_style=f"bold {COLORS['primary']}",
    )
    table.add_column("Category", style=f"bold {COLORS['primary']}")
    table.add_column("Count", justify="right")

    table.add_row("Nodes", str(len(graph.nodes)))
    table.add_row("Edges", str(len(graph.edges)))
    table.add_row(
        "Established Facts",
        str(len(graph.established_facts)),
        style=COLORS["success"],
    )
    table.add_row(
        "Conflicts",
        str(len(graph.conflicts)),
        style=COLORS["warning"],
    )
    table.add_row(
        "Knowledge Gaps",
        str(len(graph.knowledge_gaps)),
        style=COLORS["highlight"],
    )
    console.print(table)


# ============================================================================
# Hypothesis Cards (M4)
# ============================================================================

def render_hypothesis_card(h: HypothesisCard, rank: int, max_width: int = 80) -> None:
    """Render a single hypothesis as a Rich Panel."""
    scores_str = " | ".join(
        f"{k.replace('_', ' ').title()}={v:.2f}"
        for k, v in h.scores.items()
    ) if h.scores else "N/A"

    body = Text()
    body.append(f"Rank #{rank}: ", style=f"bold {COLORS['highlight']}")
    body.append(f"{h.statement}\n\n")

    if h.mechanism:
        body.append("Mechanism: ", style="bold")
        body.append(f"{h.mechanism}\n\n")

    if h.observable_predictions:
        body.append("Predictions:\n", style="bold")
        for p in h.observable_predictions:
            body.append(f"  • {p}\n")
        body.append("\n")

    if h.falsification_conditions:
        body.append("Falsification:\n", style="bold")
        for c in h.falsification_conditions:
            body.append(f"  • {c}\n")
        body.append("\n")

    body.append("Scores: ", style="bold")
    body.append(scores_str)

    console.print(Panel(
        body,
        title=f"[bold]💡 Hypothesis {h.hypothesis_id}",
        border_style=COLORS["highlight"],
        box=box.ROUNDED,
        width=max_width,
    ))
    console.print()


def render_hypotheses_summary(
    candidates: int,
    passed: int,
    top_n: int,
    hypotheses: List[HypothesisCard],
) -> None:
    """Print M4 pipeline summary + top hypotheses as cards."""
    console.print(
        f"  Generator → {candidates} candidates\n"
        f"  Critic → {passed} passed\n"
        f"  Falsifiability Check → {top_n} selected"
    )
    console.print()
    for i, h in enumerate(hypotheses, 1):
        render_hypothesis_card(h, i)


# ============================================================================
# Research Plan (M5)
# ============================================================================

def render_research_plan(plan: ResearchPlan) -> None:
    """Render a research plan as a structured panel."""
    lines = []
    lines.append(f"[bold]Study Subjects:[/] {plan.study_subjects}")
    lines.append(f"[bold]Independent Variables:[/] {', '.join(plan.independent_variables)}")
    lines.append(f"[bold]Dependent Variables:[/] {', '.join(plan.dependent_variables)}")
    lines.append(f"[bold]Control Groups:[/] {', '.join(plan.control_groups)}")
    lines.append(f"[bold]Procedures:[/]")
    for p in plan.procedures:
        lines.append(f"  • {p}")
    lines.append(f"[bold]Metrics:[/] {', '.join(plan.measurement_metrics)}")
    lines.append(f"[bold]Analysis:[/] {', '.join(plan.analysis_methods)}")
    lines.append(f"[bold]If Supported:[/] {plan.expected_results_if_supported[:200]}")
    lines.append(f"[bold]If Refuted:[/] {plan.expected_results_if_refuted[:200]}")

    console.print(Panel(
        "\n".join(lines),
        title=f"[bold]📋 Research Plan for {plan.hypothesis_id}",
        border_style=COLORS["primary"],
        box=box.ROUNDED,
    ))
    console.print()


# ============================================================================
# Review & Iteration (M6)
# ============================================================================

def render_review_result(review: ReviewResult, idx: int) -> None:
    """Print one reviewer's assessment."""
    color = COLORS["success"] if review.score >= 3.5 else COLORS["warning"]
    console.print(
        f"  [{color}]Reviewer {review.dimension.value}: "
        f"{review.score:.1f}/5 → \"{review.comments[:120]}\"[/{color}]"
    )


def render_iteration_header(iteration: int, max_iterations: int) -> None:
    """Print a header for the current iteration round."""
    console.print()
    console.print(Rule(
        f"[bold {COLORS['highlight']}]🔄 Iteration Round {iteration}/{max_iterations}",
        style=COLORS["highlight"],
    ))


def render_iteration_comparison(reviews: List[ReviewResult]) -> None:
    """Print a before/after comparison table for iteration scores."""
    if len(reviews) < 2:
        return

    table = Table(title="📈 Iteration Score Comparison", box=box.ROUNDED)
    table.add_column("Dimension", style=f"bold {COLORS['primary']}")
    table.add_column("Before", justify="right")
    table.add_column("After", justify="right")
    table.add_column("Δ", justify="right")

    # Group reviews by dimension across versions
    by_dim: dict = {}
    for r in reviews:
        by_dim.setdefault(r.dimension.value, {})[r.version] = r.score

    for dim, scores in by_dim.items():
        versions = sorted(scores.keys())
        if len(versions) >= 2:
            before = scores[versions[0]]
            after = scores[versions[-1]]
            delta = after - before
            delta_color = COLORS["success"] if delta > 0 else COLORS["error"]
            table.add_row(
                dim,
                f"{before:.1f}",
                f"{after:.1f}",
                f"[{delta_color}]{delta:+.1f}[/{delta_color}]",
            )

    console.print(table)


# ============================================================================
# Final summary
# ============================================================================

def render_final_summary(state: PipelineState) -> None:
    """Print a closing summary of the full pipeline run."""
    console.print()
    console.print(Rule(
        f"[bold {COLORS['success']}]✓ Pipeline Complete — {state.run_id}",
        style=COLORS["success"],
        characters="═",
    ))
    console.print(f"  Question: {state.input_question}")
    console.print(f"  Top hypotheses: {len(state.top_hypotheses)}")
    console.print(f"  Research plans: {len(state.research_plans)}")
    console.print(f"  Iterations: {state.iteration_count}")
    console.print(f"  Total tokens: {state.total_input_tokens:,} in / {state.total_output_tokens:,} out")
    if state.errors:
        console.print(f"  [{COLORS['error']}]Errors: {len(state.errors)}[/{COLORS['error']}]")
    console.print()
