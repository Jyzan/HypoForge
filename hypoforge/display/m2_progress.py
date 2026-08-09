from __future__ import annotations

from collections.abc import Mapping
import re
import time
from typing import Any, Protocol

from rich import box
from rich.markup import escape
from rich.panel import Panel

from . import COLORS, console


_STAGE_LABELS = {
    "query_planner": "Query planning",
    "source_search": "Source search",
    "paper_deduplicator": "Deduplication",
    "paper_ranker": "Ranking",
    "scout_reader": "Scout screening",
    "coverage_evaluator": "Coverage",
    "retention_judge": "LLM retention judge",
    "reading_workflow": "Full-text / abstract reading",
    "document_resolver": "Document resolution",
    "document_parser": "Document chunking",
    "evidence_retriever": "Evidence retrieval",
    "paper_reader": "Knowledge extraction",
    "m2_export": "M2 -> M3 evidence export",
}

# Keep this order aligned with the Agentic M2 orchestration.  A numbered
# stage indicator is easier to scan than a wall of similarly styled lines,
# especially when several source calls run in parallel.
_STAGE_ORDER = (
    "query_planner",
    "source_search",
    "paper_deduplicator",
    "paper_ranker",
    "scout_reader",
    "coverage_evaluator",
    "retention_judge",
    "reading_workflow",
    "document_resolver",
    "document_parser",
    "evidence_retriever",
    "paper_reader",
    "m2_export",
)


def _short(value: object, limit: int = 72) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "..."


def _duration(event: Mapping[str, Any]) -> str:
    elapsed = event.get("elapsed_seconds")
    if elapsed is None:
        return ""
    try:
        seconds = float(elapsed)
    except (TypeError, ValueError):
        return ""
    return f" | {seconds:.1f}s"


def _failure_text(event: Mapping[str, Any]) -> str:
    """Keep failure output useful even when legacy event text is garbled."""
    message = " ".join(str(event.get("message") or "").split())
    # Existing event producers contain a few translated separators. Prefer
    # the exception-class suffix while retaining its useful message.
    match = re.search(
        r"(?P<diagnostic>(?:[A-Za-z_][\w.]*?(?:Error|Exception)|Timeout)\s*:\s*.+)$",
        message,
    )
    if match:
        message = match.group("diagnostic")
    elif ":" in message:
        message = message.rsplit(":", 1)[-1].strip()
    return _short(message, 120) or "stage failed"


class _ConsoleLike(Protocol):
    def print(self, *objects: object, **kwargs: object) -> object:
        ...


class M2ProgressReporter:
    """Compact Rich reporter for the long-running Agentic M2 phase.

    It consumes the same observability events used by the web UI. The
    reporter is deliberately best-effort: terminal rendering must never
    interrupt search, reading, or the M2 -> M3 evidence contract.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        run_id: str = "",
        output: _ConsoleLike | None = None,
    ) -> None:
        self.enabled = enabled
        self.run_id = run_id
        self.output = output or console
        self._active = False
        self._round = 0
        self._started_at = 0.0
        # ``tool_completed`` is often followed by a richer ``tool_result``.
        # Hold the first event briefly so the terminal gets one polished line
        # instead of two almost-identical [OK] lines.
        self._pending_completed: dict[tuple[str, int, str], Mapping[str, Any]] = {}

    def handle_event(self, event: Mapping[str, Any]) -> None:
        if not self.enabled or event.get("module") != "m2":
            return
        event_type = str(event.get("event_type", ""))
        tool = str(event.get("tool", ""))
        details = event.get("details") or {}
        if not isinstance(details, Mapping):
            details = {}

        if event_type == "module_started":
            self._active = True
            self._round = 0
            self._started_at = time.monotonic()
            self._pending_completed.clear()
            run_line = (
                f"\n[dim]run: {_short(self.run_id, 48)}[/dim]"
                if self.run_id
                else ""
            )
            self.output.print(
                Panel(
                    "[bold]Agentic literature search[/bold]\n"
                    "[dim]plan[/dim] -> [dim]search[/dim] -> [dim]screen[/dim] -> "
                    "[dim]read evidence[/dim] -> [dim]export to M3[/dim]" + run_line,
                    title=f"[bold {COLORS['highlight']}]M2 / Agentic[/bold {COLORS['highlight']}]",
                    border_style=COLORS["highlight"],
                    box=box.ROUNDED,
                    expand=False,
                )
            )
            return
        if not self._active:
            return

        if event_type == "tool_started":
            self._flush_pending()
            round_index = details.get("round")
            if round_index is not None:
                try:
                    self._round = max(self._round, int(round_index))
                except (TypeError, ValueError):
                    pass
            label = self._label(tool)
            if tool.startswith("source:"):
                label = f"Source search / {escape(tool.split(':', 1)[1])}"
            suffix = f"  [dim]round {self._round}[/dim]" if self._round else ""
            if details.get("query"):
                suffix += f"  [dim]{escape(_short(details['query']))}[/dim]"
            self.output.print(
                f"  [bold {COLORS['info']}]>>[/bold {COLORS['info']}] "
                f"[dim]{self._stage_number(tool)}[/dim] [bold]{label}[/bold]{suffix}"
            )
            return

        if event_type == "tool_completed":
            key = self._event_key(tool, event, details)
            self._pending_completed[key] = event
            return

        if event_type == "tool_result":
            key = self._event_key(tool, event, details)
            completed = self._pending_completed.pop(key, None)
            self._render_completed(tool, details, completed or event)
            return

        if event_type == "tool_failed":
            self._pending_completed.pop(self._event_key(tool, event, details), None)
            label = self._label(tool)
            if tool.startswith("source:"):
                label = f"Source search / {escape(tool.split(':', 1)[1])}"
            self.output.print(
                f"  [bold {COLORS['error']}][X][/bold {COLORS['error']}] "
                f"[bold]{label}[/bold]  [red]{escape(_failure_text(event))}[/red]"
                f"[dim]{_duration(event)}[/dim]"
            )
            return

        if event_type == "module_completed":
            self._flush_pending()
            summary = self._summary_from_module(details)
            total = time.monotonic() - self._started_at if self._started_at else None
            if total is not None:
                summary = f"{summary} | total {total:.1f}s" if summary else f"total {total:.1f}s"
            self.output.print(
                Panel(
                    summary or "M2 complete; evidence-linked export is ready for M3.",
                    title=f"[bold {COLORS['success']}]M2 complete[/bold {COLORS['success']}]",
                    border_style=COLORS["success"],
                    box=box.ROUNDED,
                    expand=False,
                )
            )
            self._active = False
        elif event_type == "module_failed":
            self._flush_pending()
            self.output.print(
                Panel(
                    f"[red]{escape(_failure_text(event))}[/red]",
                    title=f"[bold {COLORS['error']}]M2 failed[/bold {COLORS['error']}]",
                    border_style=COLORS["error"],
                    box=box.ROUNDED,
                    expand=False,
                )
            )
            self._active = False

    @staticmethod
    def _event_round(event: Mapping[str, Any], details: Mapping[str, Any]) -> int:
        value = details.get("round", event.get("round", 0))
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    @classmethod
    def _event_key(
        cls,
        tool: str,
        event: Mapping[str, Any],
        details: Mapping[str, Any],
    ) -> tuple[str, int, str]:
        # A reading workflow can run once per sub-question in the same search
        # round.  Include its query/sub-question discriminator so one pending
        # completion cannot overwrite another one.
        discriminator = details.get("sub_question") or details.get("query") or ""
        return (
            tool,
            cls._event_round(event, details),
            " ".join(str(discriminator).split()),
        )

    @staticmethod
    def _label(tool: str) -> str:
        return _STAGE_LABELS.get(tool, tool.replace("_", " ").title())

    @classmethod
    def _stage_number(cls, tool: str) -> str:
        base = tool.split(":", 1)[0]
        try:
            index = _STAGE_ORDER.index(base) + 1
        except ValueError:
            return "--"
        return f"{index:02d}/{len(_STAGE_ORDER):02d}"

    def _render_completed(
        self,
        tool: str,
        details: Mapping[str, Any],
        event: Mapping[str, Any],
    ) -> None:
        label = self._label(tool)
        if tool.startswith("source:"):
            label = f"Source search / {escape(tool.split(':', 1)[1])}"
        summary = self._summary(tool, details)
        self.output.print(
            f"  [bold {COLORS['success']}][OK][/bold {COLORS['success']}] "
            f"[dim]{self._stage_number(tool)}[/dim] [bold]{label}[/bold]"
            + (f"  [dim]|[/dim] {escape(summary)}" if summary else "")
            + f"[dim]{_duration(event)}[/dim]"
        )

    def _flush_pending(self) -> None:
        for (tool, _round, _discriminator), event in list(self._pending_completed.items()):
            self._render_completed(tool, {}, event)
        self._pending_completed.clear()

    @staticmethod
    def _summary(tool: str, details: Mapping[str, Any]) -> str:
        if tool == "query_planner":
            queries = details.get("queries")
            if isinstance(queries, list):
                count = len(queries)
                return f"{count} query{'' if count == 1 else 's'}"
            return "queries ready"
        if tool == "source_search":
            counts = details.get("source_result_counts")
            if isinstance(counts, Mapping):
                total = sum(
                    int(value) for value in counts.values()
                    if isinstance(value, (int, float))
                )
                return f"{total} results"
            return "search complete"
        if tool == "paper_deduplicator":
            return f"{details.get('papers', 0)} unique papers"
        if tool == "paper_ranker":
            titles = details.get("top_titles")
            return f"{len(titles)} candidates" if isinstance(titles, list) else "candidates ranked"
        if tool == "scout_reader":
            return f"{details.get('notes', 0)} papers screened"
        if tool == "coverage_evaluator":
            if details.get("sufficient"):
                return "sufficient coverage"
            missing = details.get("missing_topics")
            return f"{len(missing)} topics missing" if isinstance(missing, list) else "coverage gaps remain"
        if tool == "reading_workflow":
            return (
                f"{details.get('papers', 0)} papers | "
                f"{details.get('knowledge_entries', 0)} knowledge entries"
            )
        if tool == "m2_export":
            return (
                f"papers {details.get('papers', 0)} | "
                f"evidence {details.get('evidence', 0)} | "
                f"knowledge {details.get('knowledge_entries', 0)}"
            )
        if tool == "retention_judge":
            return f"{details.get('candidates', 0)} candidates"
        if tool.startswith("source:"):
            if details.get("papers") is not None:
                return f"{details.get('papers', 0)} papers"
            if details.get("cache_hit"):
                return "cache hit"
            return "search complete"
        return ""

    @staticmethod
    def _summary_from_module(details: Mapping[str, Any]) -> str:
        labels = (
            ("sub-questions", "sub_questions"),
            ("papers", "papers_retrieved"),
            ("knowledge", "knowledge_entries"),
            ("exports", "export_runs"),
        )
        values = [(label, details[key]) for label, key in labels if key in details]
        return " | ".join(f"{label} {value}" for label, value in values)
