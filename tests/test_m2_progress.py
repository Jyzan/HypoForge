from __future__ import annotations

from rich.console import Console

from hypoforge.display.m2_progress import M2ProgressReporter
from hypoforge.observability import bind_event_sink, notify_event


def _event(event_type: str, *, tool: str = "", details: dict | None = None, **extra) -> dict:
    return {
        "event_type": event_type,
        "module": "m2",
        "tool": tool,
        "details": details or {},
        **extra,
    }


def test_m2_progress_ignores_other_modules_and_disabled_output() -> None:
    output = Console(record=True, force_terminal=False, width=100)
    reporter = M2ProgressReporter(output=output)

    reporter.handle_event({"event_type": "module_started", "module": "m1"})
    reporter.handle_event(_event("module_started"))
    disabled_output = Console(record=True, force_terminal=False, width=100)
    M2ProgressReporter(enabled=False, output=disabled_output).handle_event(
        _event("module_started")
    )

    assert "M2 / Agentic" in output.export_text()
    assert disabled_output.export_text() == ""


def test_m2_progress_renders_reading_and_grounding_summary() -> None:
    output = Console(record=True, force_terminal=False, width=100)
    reporter = M2ProgressReporter(output=output)

    reporter.handle_event(_event("module_started"))
    reporter.handle_event(
        _event("tool_started", tool="query_planner", details={"round": 1})
    )
    reporter.handle_event(
        _event(
            "tool_result",
            tool="m2_export",
            details={"papers": 5, "evidence": 18, "knowledge_entries": 12},
            elapsed_seconds=1.25,
        )
    )
    reporter.handle_event(
        _event(
            "module_completed",
            details={
                "sub_questions": 1,
                "papers_retrieved": 5,
                "knowledge_entries": 12,
                "export_runs": 1,
            },
        )
    )

    rendered = output.export_text()
    assert "Query planning" in rendered
    assert "M2 -> M3 evidence export" in rendered
    assert "papers 5 | evidence 18 | knowledge 12" in rendered
    assert "sub-questions 1 | papers 5 | knowledge 12 | exports 1" in rendered
    assert "M2 complete" in rendered


def test_event_sink_presentation_failure_is_isolated() -> None:
    class BrokenSink:
        def handle_event(self, event):
            raise RuntimeError("terminal failure")

    with bind_event_sink(BrokenSink()):
        notify_event(_event("module_started"))


def test_m2_progress_failure_keeps_exception_diagnostic_readable() -> None:
    output = Console(record=True, force_terminal=False, width=100)
    reporter = M2ProgressReporter(output=output)

    reporter.handle_event(_event("module_started"))
    reporter.handle_event(
        _event(
            "tool_failed",
            tool="source:pubmed",
            message="M2 Tool failure: source:pubmed: TimeoutError: upstream unavailable",
        )
    )

    rendered = output.export_text()
    assert "Source search / pubmed" in rendered
    assert "TimeoutError: upstream unavailable" in rendered
    assert "M2 Tool failure" not in rendered


def test_m2_progress_escapes_untrusted_query_markup() -> None:
    output = Console(record=True, force_terminal=False, width=100)
    reporter = M2ProgressReporter(output=output)

    reporter.handle_event(_event("module_started"))
    reporter.handle_event(
        _event(
            "tool_started",
            tool="query_planner",
            details={"round": 1, "query": "[malformed markup"},
        )
    )

    assert "[malformed markup" in output.export_text()


def test_m2_progress_merges_completion_and_result_into_one_status_line() -> None:
    output = Console(record=True, force_terminal=False, width=100)
    reporter = M2ProgressReporter(output=output)

    reporter.handle_event(_event("module_started"))
    reporter.handle_event(
        _event(
            "tool_completed",
            tool="query_planner",
            details={"round": 1},
            elapsed_seconds=0.4,
        )
    )
    reporter.handle_event(
        _event(
            "tool_result",
            tool="query_planner",
            details={"round": 1, "queries": [{"text": "robot arm"}]},
        )
    )

    rendered = output.export_text()
    assert rendered.count("[OK]") == 1
    assert "01/13" in rendered
    assert "1 query" in rendered


def test_m2_progress_keeps_same_stage_completions_for_each_sub_question() -> None:
    output = Console(record=True, force_terminal=False, width=100)
    reporter = M2ProgressReporter(output=output)

    reporter.handle_event(_event("module_started"))
    reporter.handle_event(
        _event(
            "tool_completed",
            tool="reading_workflow",
            details={"sub_question": "question one"},
            elapsed_seconds=1.0,
        )
    )
    reporter.handle_event(
        _event(
            "tool_completed",
            tool="reading_workflow",
            details={"sub_question": "question two"},
            elapsed_seconds=2.0,
        )
    )
    reporter.handle_event(_event("module_completed", details={}))

    rendered = output.export_text()
    assert rendered.count("Full-text / abstract reading") == 2
