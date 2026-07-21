"""Reusable Rich rendering functions for HypoForge pipeline output."""

from __future__ import annotations

import re
from typing import Any, Dict, List

from rich.panel import Panel
from rich.padding import Padding
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich import box
from rich.console import Group

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

# ---------------------------------------------------------------------------
# Knowledge entry type → colour mapping
# ---------------------------------------------------------------------------

_KE_COLORS = {
    "established_fact": COLORS["success"],
    "mechanistic_conclusion": COLORS["primary"],
    "conflicting_evidence": COLORS["error"],
    "method": COLORS["warning"],
    "key_entity": COLORS["info"],
    "knowledge_gap": COLORS["highlight"],
    "speculation": "#9ca3af",
}

_KE_LABELS = {
    "established_fact": "FACT",
    "mechanistic_conclusion": "MECH",
    "conflicting_evidence": "CONFLICT",
    "method": "METHOD",
    "knowledge_gap": "GAP",
    "key_entity": "ENTITY",
    "speculation": "SPEC",
}


def _single_line(value: Any) -> str:
    """Prevent embedded model whitespace from creating accidental display lines."""
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _numbered_lines(value: Any) -> str:
    """Put inline numbered recommendations on separate terminal lines."""
    text = _single_line(value)
    if not text:
        return ""
    parts = re.split(r"\s+(?=\d+[.)]\s)", text)
    return "\n".join(part.strip() for part in parts if part.strip())


def _split_numbered(value: Any) -> List[tuple] | None:
    """Parse an inline numbered list into ``(marker, body)`` pairs.

    Markers are normalised to ``"N."`` form (so a model that writes ``1)`` and
    one that writes ``1.`` render identically).  Returns ``None`` when *value*
    contains no numbered markers (i.e. it is prose that should render inline).
    """
    text = _single_line(value)
    if not text:
        return None
    parts = [p.strip() for p in re.split(r"\s+(?=\d+[.)]\s)", text) if p.strip()]
    items: List[tuple] = []
    for part in parts:
        match = re.match(r"(\d+)[.)]\s*(.*)", part)
        if match:
            items.append((f"{match.group(1)}.", match.group(2).strip()))
        else:
            # Text before the first marker (a preamble) — keep it, unmarked.
            items.append((None, part))
    has_marker = any(marker is not None for marker, _ in items)
    return items if has_marker else None


def _split_risks_and_alternatives(value: Any) -> tuple[List[str], List[str]]:
    """Extract repeated ``Risk:`` / ``Alternative:`` pairs from model prose.

    Models often number only the first pair and then emit one continuous string,
    for example ``1. Risk: ... Alternative: ... Risk: ... Alternative: ...``.
    Number-only parsing treats that as one item, so use the semantic labels as
    the primary delimiters and fall back to a conventional numbered list.
    """
    text = _single_line(value)
    if not text:
        return [], []

    label_pattern = re.compile(
        r"(?i)(?<!\S)(?:\d+[.)]\s*)?(risks?|alternatives?)\s*(?:#?\d+)?\s*:\s*"
    )
    matches = list(label_pattern.finditer(text))
    risks: List[str] = []
    alternatives: List[str] = []

    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[match.end():end].strip(" \t\r\n-;|\u2502")
        if not body:
            continue
        label = match.group(1).lower()
        if label.startswith("risk"):
            risks.append(body)
        else:
            alternatives.append(body)

    if risks or alternatives:
        return risks, alternatives

    parsed = _split_numbered(text)
    if parsed:
        return [body for _, body in parsed if body], []
    return [text], []


def _preview_text(value: Any, limit: int = 64) -> str:
    """Keep dense M5 table cells readable without another LLM call."""
    text = _single_line(value)
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def _labelled_wrapped_table(label: str, value: Any) -> Table:
    """Return a label/value row whose wrapped lines use a hanging indent."""
    table = Table(
        box=None,
        show_header=False,
        show_edge=False,
        pad_edge=False,
        padding=(0, 1, 0, 0),
        expand=True,
    )
    table.add_column(style=COLORS["muted"], no_wrap=True)
    table.add_column(overflow="fold", ratio=1)
    table.add_row(label, _single_line(value))
    return table


def _numbered_wrapped_table(
    items: List[str],
    *,
    marker_style: str,
    body_style: str = "white",
) -> Table:
    """Return a numbered list whose continuation lines align with the body."""
    table = Table(
        box=None,
        show_header=False,
        show_edge=False,
        pad_edge=False,
        padding=(0, 1, 0, 0),
        expand=True,
    )
    table.add_column(justify="right", no_wrap=True, style=marker_style)
    table.add_column(overflow="fold", ratio=1, style=body_style)
    for index, body in enumerate(items, 1):
        table.add_row(f"{index}.", _single_line(body))
    return table


def _preview_cell(value: Any, limit: int) -> Text:
    """Return a compact table cell with a highlighted truncation marker."""
    text = _single_line(value)
    if len(text) <= limit:
        return Text(text)
    result = Text(text[: limit - 3].rstrip())
    result.append("...", style=COLORS["warning"])
    return result


def _source_url(source_id: Any) -> str:
    """Normalize paper identifiers to directly usable source URLs."""
    source = _single_line(source_id)
    if not source:
        return ""
    if source.startswith(("http://", "https://")):
        return source
    if source.upper().startswith("PMID:"):
        return f"https://pubmed.ncbi.nlm.nih.gov/{source.split(':', 1)[1].strip()}/"
    if source.upper().startswith("DOI:"):
        return f"https://doi.org/{source.split(':', 1)[1].strip()}"
    if source.startswith("10."):
        return f"https://doi.org/{source}"
    return source


def _metric_label(value: Any) -> str:
    """Show the concise metric name; retain details in the saved JSON output."""
    text = _single_line(value)
    return text.split(":", 1)[0].strip() if ":" in text else _preview_text(text, 48)


def _timeline_text(value: Any) -> Text:
    """Render timeline phases with their date ranges visually emphasized."""
    text = _single_line(value)
    result = Text()
    phases = [phase.strip() for phase in re.split(r"\s*;\s*", text) if phase.strip()]
    for index, phase in enumerate(phases):
        if index:
            result.append("\n")
        match = re.match(r"((?:Months?|Weeks?|Days?)\s+[\w\-–—]+\s*:)(.*)", phase)
        if match:
            result.append(match.group(1), style="bold white")
            result.append(f" {match.group(2).strip()}")
        else:
            result.append(phase)
    return result


# ============================================================================
# Phase / Module headers
# ============================================================================

_PHASE_ICONS = {"running": ">", "done": "[OK]", "error": "[ERR]"}
_PHASE_COLORS = {"running": "primary", "done": "success", "error": "error"}


def render_phase_header(
    module_name: str,
    title: str,
    status: str = "running",
) -> None:
    """Print a stylised rule marking the start (or end) of a module phase."""
    icon = _PHASE_ICONS.get(status, ">")
    color = COLORS[_PHASE_COLORS.get(status, "primary")]
    console.print()
    console.print(Rule(
        f"[bold {color}]{icon} [{module_name.upper()}] {title}",
        style=color,
        characters="━",
    ))


def render_phase_done(module_name: str) -> None:
    """Shorthand for a completed phase."""
    console.print(f"  [{COLORS['success']}][OK] [{module_name.upper()}] complete[/{COLORS['success']}]")


# ============================================================================
# Problem Card (M1) — simple, proven style
# ============================================================================

def render_problem_card(card: ProblemCard) -> None:
    """Print the structured problem decomposition."""
    console.print(f"  [{COLORS['muted']}]Domain:[/] {', '.join(card.domain)}")
    console.print(f"  [{COLORS['muted']}]Sub-questions:[/] {len(card.sub_questions)}")
    sub_question_table = Table(
        box=None,
        show_header=False,
        show_edge=False,
        pad_edge=False,
        padding=(0, 1, 0, 0),
    )
    sub_question_table.add_column(justify="right", no_wrap=True)
    sub_question_table.add_column(overflow="fold")
    for i, sq in enumerate(card.sub_questions, 1):
        sub_question_table.add_row(f"{i}.", _single_line(sq))
    if card.sub_questions:
        console.print(Padding(sub_question_table, (0, 0, 0, 4)))
    console.print(Padding(
        _labelled_wrapped_table("Key entities:", ", ".join(card.key_entities)),
        (0, 0, 0, 2),
    ))
    console.print(f"  [{COLORS['muted']}]Question type:[/] {card.question_type.value}")


# ============================================================================
# Literature Search (M2)
# ============================================================================

def _ke_badge(ke_type: str) -> str:
    """Return a coloured Rich-markup badge for a knowledge entry type.

    Rich markup in *table cells* is parsed by Rich, so this is safe.
    """
    color = _KE_COLORS.get(ke_type, COLORS["muted"])
    label = _KE_LABELS.get(ke_type, ke_type[:6].upper())
    return f"[bold {color}]{label}[/bold {color}]"


def render_literature_summary(
    papers_retrieved: int,
    knowledge_entries: int,
    sub_questions: int,
) -> None:
    """Print a one-line summary of literature search results."""
    console.print(
        f"  [{COLORS['success']}]Retrieved {papers_retrieved} papers "
        f"across {sub_questions} sub-questions, "
        f"extracted {knowledge_entries} knowledge entries[/{COLORS['success']}]"
    )


def _render_m2_results(results: list) -> None:
    papers = sum(r.papers_retrieved for r in results)
    entries = sum(len(r.knowledge_entries) for r in results)
    render_literature_summary(papers, entries, len(results))

    # ---- type distribution ----
    type_counts: Dict[str, int] = {}
    for r in results:
        for e in r.knowledge_entries:
            t = e.type.value if hasattr(e.type, "value") else str(e.type)
            type_counts[t] = type_counts.get(t, 0) + 1
    if type_counts:
        dist = "  ".join(
            f"{_ke_badge(t)} x{c}" for t, c in sorted(
                type_counts.items(), key=lambda item: _KE_LABELS.get(item[0], item[0])
            )
        )
        console.print(f"  [bold]Distribution:[/] {dist}")

    console.print()

    # ---- knowledge entries table ----
    table = Table(
        title="Knowledge Extraction",
        box=box.ROUNDED,
        title_style=f"bold {COLORS['primary']}",
        show_lines=True,
        header_style="bold white",
    )
    table.add_column("Type", no_wrap=True, width=10)
    table.add_column("Content", ratio=3, overflow="fold")
    table.add_column("Entities", ratio=2, overflow="fold")
    table.add_column("Source", ratio=2, overflow="fold")
    table.add_column("Confidence", justify="center", no_wrap=True, width=10)

    confidence_styles = {"high": COLORS["success"], "medium": COLORS["warning"],
                         "low": COLORS["error"]}

    # Keep related knowledge entries together, independent of retrieval order.
    type_order = {
        "established_fact": 0,
        "mechanistic_conclusion": 1,
        "conflicting_evidence": 2,
        "knowledge_gap": 3,
        "method": 4,
        "key_entity": 5,
        "speculation": 6,
    }
    all_entries = [
        (entry, entry.type.value if hasattr(entry.type, "value") else str(entry.type))
        for result in results
        for entry in result.knowledge_entries
    ]
    all_entries.sort(key=lambda item: type_order.get(item[1], 99))

    grouped: Dict[str, list] = {}
    for entry, ke_type in all_entries:
        grouped.setdefault(ke_type, []).append(entry)

    # Show up to 3 entries per category so no single type dominates the
    # preview.  When a category has fewer than 3 entries all are shown.
    ordered_types = sorted(grouped, key=lambda t: type_order.get(t, 99))
    max_per_category = 3
    previous_type = None
    remaining = 0
    for ke_type in ordered_types:
        entries_for_type = grouped[ke_type]
        shown = entries_for_type[:max_per_category]
        remaining += max(0, len(entries_for_type) - max_per_category)
        for entry in shown:
            if previous_type is not None and ke_type != previous_type:
                table.add_section()
            conf = entry.confidence.value if entry.confidence and hasattr(entry.confidence, "value") else ""
            conf_style = confidence_styles.get(conf, COLORS["muted"])
            source = _source_url(entry.source_paper_id)
            table.add_row(
                _ke_badge(ke_type),
                _preview_cell(entry.content, 110),
                _preview_cell(", ".join(entry.entities[:4]), 76) if entry.entities else "-",
                source or "-",
                f"[{conf_style}]{conf.upper()}[/{conf_style}]" if conf else "-",
            )
            previous_type = ke_type

    has_rows = previous_type is not None
    if has_rows:
        console.print(table)
        if remaining > 0:
            console.print(f"  [{COLORS['warning']}]... and {remaining} more entries[/{COLORS['warning']}]")
    console.print()


# ============================================================================
# Evidence Graph (M3)
# ============================================================================

def _lookup_entry_content(entry_id: str, state: PipelineState) -> str:
    """Find a knowledge entry's content by ID from literature results."""
    for lr in state.literature_results:
        for ke in lr.knowledge_entries:
            if ke.id == entry_id:
                return ke.content
    return entry_id


def render_evidence_summary(graph: EvidenceGraph, state: PipelineState | None = None) -> None:
    """Print the evidence graph: summary table + sample items."""
    # ---- summary table ----
    table = Table(
        title="Evidence Graph",
        box=box.ROUNDED,
        title_style=f"bold {COLORS['primary']}",
    )
    table.add_column("Category", style=f"bold {COLORS['primary']}")
    table.add_column("Count", justify="right")

    table.add_row(f"[{COLORS['success']}]Established Facts[/{COLORS['success']}]",
                  str(len(graph.established_facts)))
    table.add_row(f"[{COLORS['warning']}]Conflicts[/{COLORS['warning']}]",
                  str(len(graph.conflicts)))
    table.add_row(f"[{COLORS['highlight']}]Knowledge Gaps[/{COLORS['highlight']}]",
                  str(len(graph.knowledge_gaps)))
    table.add_row(f"[{COLORS['info']}]Nodes[/{COLORS['info']}]", str(len(graph.nodes)))
    table.add_row(f"[{COLORS['primary']}]Edges[/{COLORS['primary']}]", str(len(graph.edges)))
    console.print(table)

    # ---- sample items ----
    if state is None:
        return

    console.print()
    for label, color, ids, max_show in [
        ("Established Facts", COLORS["success"], graph.established_facts, 5),
        ("Conflicts", COLORS["warning"], graph.conflicts, 3),
        ("Knowledge Gaps", COLORS["highlight"], graph.knowledge_gaps, 3),
    ]:
        if not ids:
            continue
        console.print(f"  [bold {color}]{label}[/bold {color}]")
        for eid in ids[:max_show]:
            content = _lookup_entry_content(eid, state)
            # Keep each evidence preview on one terminal line.  Category labels
            # start at two spaces; entries sit one indentation level beneath
            # them and Rich replaces overflow with an ellipsis.
            row = Text(no_wrap=True, overflow="ellipsis")
            row.append("    ")
            row.append(_single_line(eid), style=COLORS["muted"])
            row.append(" ")
            row.append(_single_line(content))
            console.print(row, no_wrap=True, overflow="ellipsis")
        if len(ids) > max_show:
            console.print(
                f"    [{COLORS['muted']}]... and {len(ids) - max_show} more[/{COLORS['muted']}]"
            )
        console.print()


# ============================================================================
# Hypothesis Cards (M4)
# ============================================================================

_SCORE_LABELS = {
    "novelty": "Novelty",
    "scientific_soundness": "Soundness",
    "testability": "Testability",
    "evidence_consistency": "Evidence",
    "composite": "Composite",
}

_SCORE_BAR_SEGMENTS = 10


def _score_bar(value: float) -> str:
    """Draw a mini horizontal bar for a score (0–1).  Returns a Rich markup string."""
    filled = max(0, min(_SCORE_BAR_SEGMENTS, int(value * _SCORE_BAR_SEGMENTS)))
    bar = "█" * filled + "░" * (_SCORE_BAR_SEGMENTS - filled)
    if value >= 0.8:
        color = COLORS["success"]
    elif value >= 0.6:
        color = COLORS["primary"]
    elif value >= 0.4:
        color = COLORS["warning"]
    else:
        color = COLORS["error"]
    return f"[{color}]{bar}[/{color}]"


def render_hypothesis_card(h: HypothesisCard, rank: int, max_width: int = 80) -> None:
    """Render a single hypothesis as a Rich Panel with score bars.

    IMPORTANT: We use console.print() markup-style strings directly, NOT
    Text.append() + embedded style tags (which are treated as literal text).
    """
    body = Text()

    # Statement
    body.append(f"{h.statement}\n\n")

    # Mechanism
    if h.mechanism:
        body.append("Mechanism  ", style="bold")
        body.append(f"{h.mechanism}\n\n", style=COLORS["muted"])

    # Predictions
    if h.observable_predictions:
        body.append("Predictions\n", style="bold")
        for p in h.observable_predictions:
            body.append("  [+] ", style=COLORS["success"])
            body.append(f"{p}\n")

    # Falsification
    if h.falsification_conditions:
        body.append("Falsification Conditions\n", style="bold")
        for c in h.falsification_conditions:
            body.append("  [-] ", style=COLORS["error"])
            body.append(f"{c}\n")

    # Scores with bars — use Rich markup in console.print() below
    # to render the score bars, since Text.append() doesn't parse markup

    console.print(Panel(
        body,
        title=f"[bold {COLORS['highlight']}]Hypothesis {h.hypothesis_id}  —  Rank #{rank}",
        border_style=COLORS["highlight"],
        box=box.ROUNDED,
        width=max_width,
    ))

    # Scores are rendered separately via console.print() so the bar markup works
    if h.scores:
        for key, label in _SCORE_LABELS.items():
            if key in h.scores:
                val = h.scores[key]
                bar = _score_bar(val)
                console.print(
                    f"    {label:<12} {bar} "
                    f"[{COLORS['muted']}]{val:.2f}[/{COLORS['muted']}]"
                )
    console.print()


def render_hypotheses_summary(
    candidates: int,
    passed: int,
    top_n: int,
    hypotheses: List[HypothesisCard],
) -> None:
    """Print M4 pipeline summary + top hypotheses as cards."""
    console.print(
        f"  [{COLORS['muted']}]Generator ->[/{COLORS['muted']}] "
        f"[{COLORS['primary']}]{candidates} candidates[/{COLORS['primary']}]  "
        f"[{COLORS['muted']}]Critic ->[/{COLORS['muted']}] "
        f"[{COLORS['warning']}]{passed} passed[/{COLORS['warning']}]  "
        f"[{COLORS['muted']}]Falsifiability ->[/{COLORS['muted']}] "
        f"[{COLORS['success']}]{top_n} selected[/{COLORS['success']}]"
    )
    console.print()
    for i, h in enumerate(hypotheses, 1):
        render_hypothesis_card(h, i)


# ============================================================================
# Research Plan (M5)
# ============================================================================

def render_research_plan(plan: ResearchPlan) -> None:
    """Render a research plan as a structured, multi-section card."""
    sections: List[Panel] = []

    # ── Overview ──
    overview = Text()
    overview.append("Subjects\n", style=f"bold {COLORS['primary']}")
    overview.append(f"  {_single_line(plan.study_subjects)}")
    overview.append("\n\n")
    overview.append("Timeline\n", style=f"bold {COLORS['primary']}")
    timeline = _timeline_text(plan.timeline)
    for index, phase in enumerate(timeline.split("\n")):
        if index:
            overview.append("\n")
        overview.append("  ")
        overview.append(phase)
    sections.append(Panel(overview, title="Overview", border_style=COLORS["primary"],
                          box=box.ROUNDED, padding=(1, 2)))

    # ── Variables & Controls ──
    # A vertical layout gives long experimental variables the full panel width
    # instead of forcing them into three narrow, heavily wrapped columns.
    variable_groups = [
        ("Independent", plan.independent_variables),
        ("Dependent", plan.dependent_variables),
        ("Controls", plan.control_groups),
    ]
    variable_renderables: List[Any] = []
    for group_index, (label, values) in enumerate(variable_groups):
        block = Text()
        block.append(label, style=f"bold {COLORS['highlight']}")
        block.append("\n")
        if values:
            for item_index, value in enumerate(values, 1):
                if item_index > 1:
                    block.append("\n")
                block.append(f"  {item_index}. ", style=f"bold {COLORS['highlight']}")
                block.append(_single_line(value), style="white")
        else:
            block.append("  -", style=COLORS["muted"])
        variable_renderables.append(block)
        if group_index < len(variable_groups) - 1:
            variable_renderables.append(
                Rule(style=COLORS["highlight"], characters="─")
            )

    sections.append(Panel(
        Group(*variable_renderables),
        title="Variables & Controls",
        border_style=COLORS["highlight"],
        box=box.ROUNDED,
        padding=(1, 2),
    ))

    # ── Procedures ──
    proc_text = Text()
    for i, p in enumerate(plan.procedures, 1):
        proc_text.append(f"  {i}. ", style=f"bold {COLORS['warning']}")
        proc_text.append(f"{p}\n")
    sections.append(Panel(proc_text, title="Procedures", border_style=COLORS["warning"],
                          box=box.ROUNDED, padding=(1, 2)))

    # ── Metrics & Analysis ──
    ma_table = Table(
        box=box.SIMPLE,
        show_header=True,
        header_style=f"bold {COLORS['primary']}",
    )
    ma_table.add_column("Metrics", style="white", no_wrap=True, overflow="ellipsis")
    ma_table.add_column("Analysis Methods", style="white", no_wrap=True, overflow="ellipsis")
    max_ma = max(len(plan.measurement_metrics), len(plan.analysis_methods))
    for i in range(max_ma):
        metric_cell = Text()
        if i < len(plan.measurement_metrics):
            metric_cell.append(f"{i + 1}. ", style=f"bold {COLORS['info']}")
            metric_cell.append(_metric_label(plan.measurement_metrics[i]), style="white")

        analysis_cell = Text()
        if i < len(plan.analysis_methods):
            analysis_cell.append(f"{i + 1}. ", style=f"bold {COLORS['info']}")
            analysis_cell.append(_preview_text(plan.analysis_methods[i]), style="white")

        ma_table.add_row(
            metric_cell,
            analysis_cell,
        )
    sections.append(Panel(ma_table, title="Metrics & Analysis", border_style=COLORS["primary"],
                          box=box.ROUNDED, padding=(1, 2)))

    # ── Expected Results ──
    er_text = Text()
    if plan.expected_results_if_supported:
        er_text.append("If Supported\n", style=f"bold {COLORS['success']}")
        er_text.append(f"  {plan.expected_results_if_supported}\n\n")
    if plan.expected_results_if_refuted:
        er_text.append("If Refuted\n", style=f"bold {COLORS['error']}")
        er_text.append(f"  {plan.expected_results_if_refuted}")
    sections.append(Panel(er_text, title="Expected Results", border_style=COLORS["success"],
                          box=box.ROUNDED, padding=(1, 2)))

    # ── Risks ──
    if plan.risks_and_alternatives:
        risk_items, alternative_items = _split_risks_and_alternatives(
            plan.risks_and_alternatives
        )
        risk_renderables: List[Any] = []

        def append_warning_list(label: str, items: List[str]) -> None:
            risk_renderables.append(Text(label, style=f"bold {COLORS['warning']}"))
            if not items:
                risk_renderables.append(Text("  -", style=COLORS["muted"]))
                return
            risk_renderables.append(Padding(
                _numbered_wrapped_table(
                    items,
                    marker_style=f"bold {COLORS['warning']}",
                ),
                (0, 0, 0, 2),
            ))

        append_warning_list("Risk", risk_items)
        if alternative_items:
            risk_renderables.append(Rule(style=COLORS["warning"], characters="─"))
            append_warning_list("Alternative", alternative_items)

        sections.append(Panel(
            Group(*risk_renderables),
            title="Risks & Alternatives",
            border_style=COLORS["warning"],
            box=box.ROUNDED,
            padding=(1, 2),
        ))

    # ── Assemble ──
    console.print(
        f"[bold {COLORS['primary']}]Research Plan[/bold {COLORS['primary']}]  "
        f"{plan.hypothesis_id}"
    )
    console.print(Group(*sections))
    console.print()


def render_research_plans_summary(plans: List[ResearchPlan], max_plans: int = 2) -> None:
    """Render a compact summary + structured detailed views."""
    if not plans:
        console.print(f"  [{COLORS['muted']}]No research plans generated.[/{COLORS['muted']}]")
        return

    # Compact summary table
    table = Table(
        title="Research Plans Overview",
        box=box.ROUNDED,
        title_style=f"bold {COLORS['primary']}",
        show_lines=False,
    )
    table.add_column(
        "ID", style=f"bold {COLORS['highlight']}",
        width=20, no_wrap=True, overflow="ellipsis",
    )
    table.add_column(
        "Subjects", width=24, no_wrap=True, overflow="ellipsis",
    )
    table.add_column("IV", justify="center", width=4, no_wrap=True)
    table.add_column("DV", justify="center", width=4, no_wrap=True)
    table.add_column("Controls", justify="center", width=8, no_wrap=True)
    table.add_column(
        "Timeline", width=24, no_wrap=True, overflow="ellipsis",
    )

    for plan in plans:
        table.add_row(
            plan.hypothesis_id,
            _single_line(plan.study_subjects),
            str(len(plan.independent_variables)),
            str(len(plan.dependent_variables)),
            str(len(plan.control_groups)),
            _single_line(plan.timeline),
        )
    console.print(table)
    console.print()

    # Detailed views for top plans
    for plan in plans[:max_plans]:
        render_research_plan(plan)


# ============================================================================
# Review & Iteration (M6)
# ============================================================================

def render_review_result(review: ReviewResult, idx: int) -> None:
    """Print one reviewer's assessment (compact line — used for iteration header)."""
    if review.score >= 4.0:
        color = COLORS["success"]
    elif review.score >= 3.0:
        color = COLORS["warning"]
    else:
        color = COLORS["error"]
    console.print(
        f"  [{color}]Reviewer {review.dimension.value}: "
        f"{review.score:.1f}/5 -> \"{review.comments[:120]}\"[/{color}]"
    )


def render_iteration_header(iteration: int, max_iterations: int) -> None:
    """Print a header for the current iteration round."""
    console.print()
    console.print(Rule(
        f"[bold {COLORS['highlight']}]Iteration Round {iteration}/{max_iterations}",
        style=COLORS["highlight"],
    ))


def render_iteration_comparison(reviews: List[ReviewResult]) -> None:
    """Print a before/after comparison table for iteration scores."""
    if len(reviews) < 2:
        return

    table = Table(title="Iteration Score Comparison", box=box.ROUNDED)
    table.add_column("Dimension", style=f"bold {COLORS['primary']}")
    table.add_column("Before", justify="right")
    table.add_column("After", justify="right")
    table.add_column("Delta", justify="right")

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


def render_guidance_prompt(iteration: int) -> None:
    """Render the Claude-Code-style cue that asks the user for revision guidance.

    Prints the visual cue only; the caller performs the blocking ``input()``
    (I/O is kept out of the display layer).  Shared by M4's interactive step and
    the M6 preview harness so the on-screen prompt is tuned in exactly one place.
    """
    hint = Text()
    hint.append("Guide the next M4 revision — e.g. ", style=COLORS["muted"])
    hint.append('"focus on the mechanism"', style="italic")
    hint.append(".\n", style=COLORS["muted"])
    hint.append("Press ", style=COLORS["muted"])
    hint.append("Enter", style=f"bold {COLORS['primary']}")
    hint.append(" to skip.", style=COLORS["muted"])
    console.print()
    console.print(Panel(
        hint,
        title=f"[bold {COLORS['highlight']}]Your guidance · iteration {iteration}",
        border_style=COLORS["highlight"],
        box=box.ROUNDED,
        padding=(0, 2),
    ))


def render_guidance_result(guidance: str) -> None:
    """Confirm whether typed guidance will be sent to the M4 revision."""
    guidance = _single_line(guidance)
    if guidance:
        console.print(
            f"\n  [{COLORS['success']}]Captured → M4 revision context:"
            f"[/{COLORS['success']}] {guidance}"
        )
        return
    console.print(
        f"\n  [{COLORS['muted']}]No guidance entered (Enter = skip); "
        f"M4 will revise from the reviews alone.[/{COLORS['muted']}]"
    )


def _labelled_block(label: str, body: Any, *, indent: int = 4) -> None:
    """Print ``<label>: <body>`` with wrapped lines left-aligned under the label.

    A uniform *indent* means a continuation line sits directly below the label
    word rather than jumping back to column 0.
    """
    text = Text()
    text.append(f"{label}: ", style="bold")
    text.append(_single_line(body))
    console.print(Padding(text, (0, 0, 0, indent)))


def _numbered_block(label: str, items: List[tuple], *, indent: int = 4) -> None:
    """Print a bold *label* then a numbered list.

    Each marker sits at *indent* (aligned under the label) while wrapped body
    lines hang under the item's own text — achieved with a borderless two-column
    table so CJK width is measured correctly.
    """
    console.print(Padding(Text(f"{label}:", style="bold"), (0, 0, 0, indent)))
    table = Table(
        box=None,
        show_header=False,
        show_edge=False,
        pad_edge=False,
        padding=(0, 1, 0, 0),
    )
    table.add_column(justify="left", no_wrap=True, style="bold")
    table.add_column(overflow="fold")
    for marker, body in items:
        table.add_row(marker or "", Text(_single_line(body)))
    console.print(Padding(table, (0, 0, 0, indent)))


def _review_block_separator(*, indent: int = 4) -> None:
    """Separate M6 detail blocks with an indented white horizontal rule."""
    console.print(Padding(Rule(style="white", characters="─"), (0, 0, 0, indent)))


def _render_m6_reviews(reviews: List[ReviewResult]) -> None:
    """Print review results as a Rich table with colour-coded scores."""
    if not reviews:
        console.print(f"  [{COLORS['muted']}]No reviews generated.[/{COLORS['muted']}]")
        return

    latest_version = max(review.version for review in reviews)
    latest_reviews = [review for review in reviews if review.version == latest_version]

    table = Table(
        title=f"Review Round {latest_version}",
        box=box.ROUNDED,
        title_style=f"bold {COLORS['highlight']}",
        show_lines=False,
        header_style="bold white",
    )
    table.add_column("Dimension", style=f"bold {COLORS['primary']}", no_wrap=True, width=22)
    table.add_column("Score", justify="center", no_wrap=True, width=10)

    for review in latest_reviews:
        if review.score >= 4.0:
            score_style = COLORS["success"]
        elif review.score >= 3.0:
            score_style = COLORS["warning"]
        else:
            score_style = COLORS["error"]

        is_overall = review.dimension.value == "overall"
        dim_label = (
            f"[bold]{review.dimension.value} (computed)[/bold]"
            if is_overall
            else review.dimension.value
        )

        table.add_row(dim_label, f"[bold {score_style}]{review.score:.1f}/5[/bold {score_style}]")

    console.print(table)
    for review in latest_reviews:
        console.print()
        console.print(f"  [bold {COLORS['primary']}]{review.dimension.value}[/bold {COLORS['primary']}]")
        if review.reasoning:
            _labelled_block("Reasoning", review.reasoning)
            _review_block_separator()
        _labelled_block("Assessment", review.comments)
        if review.suggestions:
            _review_block_separator()
            # "overall" is a summary line whose Reasoning / Assessment do not
            # wrap onto their own lines, so render its suggestion inline to match.
            # Specialist reviews carry a numbered list → hanging-indent block.
            is_overall = review.dimension.value == "overall"
            items = None if is_overall else _split_numbered(review.suggestions)
            if items:
                _numbered_block("Suggestions", items)
            else:
                _labelled_block("Suggestions", review.suggestions)
    console.print()


# ============================================================================
# Per-module result dispatcher
# ============================================================================

def render_module_result(
    module_name: str,
    state: PipelineState,
    result: Dict[str, Any],
) -> None:
    """Render a module's returned state update in a readable terminal form."""
    if not result:
        console.print(f"  [{COLORS['muted']}]No state updates returned.[/{COLORS['muted']}]")
        return

    if module_name == "m1" and result.get("problem_card"):
        render_problem_card(result["problem_card"])
        return

    if module_name == "m2" and result.get("literature_results") is not None:
        _render_m2_results(result["literature_results"])
        return

    if module_name == "m3" and result.get("evidence_graph"):
        render_evidence_summary(result["evidence_graph"], state)
        return

    if module_name == "m4":
        candidates = result.get("candidate_hypotheses", [])
        top = result.get("top_hypotheses", [])
        render_hypotheses_summary(
            candidates=len(candidates),
            passed=len(candidates),
            top_n=len(top),
            hypotheses=top,
        )
        return

    if module_name == "m5" and result.get("research_plans") is not None:
        render_research_plans_summary(result["research_plans"])
        return

    if module_name == "m6":
        all_reviews = result.get("reviews", state.reviews)
        _render_m6_reviews(all_reviews)
        return

    updated_fields = ", ".join(sorted(result.keys()))
    console.print(Panel(
        f"Updated fields: {updated_fields}",
        title=f"[bold]Result: {module_name.upper()}",
        border_style=COLORS["primary"],
        box=box.ROUNDED,
    ))


# ============================================================================
# Final summary
# ============================================================================

def render_final_summary(state: PipelineState) -> None:
    """Print a closing summary of the full pipeline run."""
    console.print()
    console.print(Rule(
        f"[bold {COLORS['success']}][OK] Pipeline Complete — {state.run_id}",
        style=COLORS["success"],
        characters="═",
    ))

    body = Text()
    body.append("Question:       ", style="bold")
    body.append(f"{state.input_question}\n")
    body.append("Top Hypotheses: ", style="bold")
    body.append(f"{len(state.top_hypotheses)}\n", style=COLORS["highlight"])
    body.append("Research Plans: ", style="bold")
    body.append(f"{len(state.research_plans)}\n", style=COLORS["primary"])
    body.append("Iterations:     ", style="bold")
    body.append(f"{state.iteration_count}\n", style=COLORS["warning"])
    body.append("Tokens:         ", style="bold")
    body.append(f"{state.total_input_tokens:,} in", style=COLORS["success"])
    body.append(" / ")
    body.append(f"{state.total_output_tokens:,} out", style=COLORS["primary"])
    if state.errors:
        body.append("\nErrors:         ", style="bold")
        body.append(f"{len(state.errors)}", style=COLORS["error"])

    console.print(Panel(
        body,
        title=f"[bold {COLORS['success']}]Pipeline Summary",
        border_style=COLORS["success"],
        box=box.ROUNDED,
        padding=(1, 2),
    ))
    console.print()
