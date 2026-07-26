from __future__ import annotations

import io
import json

import pytest

from hypoforge.literature.adapter import AgenticM2Module
from hypoforge.literature.models import (
    CoverageReport,
    PaperReadingResult,
    PaperRecord,
    SearchBudget,
    SearchQuery,
    SearchRunResult,
    StopReason,
)
from scripts import run_m2_integrated


QUESTION = "Hippo通路如何调控器官大小？"


def make_search_result() -> SearchRunResult:
    paper = PaperRecord(
        paper_id="PMID:1",
        title="Hippo pathway paper",
        abstract="YAP and TAZ regulate organ size.",
        year=2024,
        pmid="1",
        sources=["pubmed"],
    )
    query = SearchQuery(
        query_id="q1",
        text="Hippo YAP TAZ organ size",
        target_source="pubmed",
        purpose="core mechanism",
        relation_to_question="Directly tests the proposed mechanism.",
    )
    return SearchRunResult(
        sub_question=QUESTION,
        queries=[query],
        papers_found=1,
        papers_after_dedup=1,
        candidates=[paper],
        final_papers=[paper],
        coverage=CoverageReport(
            covered_topics=["Hippo signaling"],
            sufficient=True,
            rationale="Core mechanism covered.",
        ),
        iterations=1,
        stop_reason=StopReason.COVERAGE_SATISFIED,
        source_result_counts={"pubmed": 1},
    )


def make_reading_result() -> PaperReadingResult:
    return PaperReadingResult(
        paper_id="PMID:1",
        summary="YAP and TAZ connect Hippo signaling to organ size.",
        degraded_to_abstract=True,
        errors=["full text unavailable; used abstract"],
    )


class FakeSearchAgent:
    def __init__(self, result: SearchRunResult) -> None:
        self.result = result
        self.calls = 0

    async def run(self, *args, **kwargs) -> SearchRunResult:
        self.calls += 1
        return self.result


class FakeReadingWorkflow:
    def __init__(self, result: PaperReadingResult) -> None:
        self.result = result
        self.calls = 0

    async def run(self, *args, **kwargs) -> list[PaperReadingResult]:
        self.calls += 1
        return [self.result]


@pytest.mark.asyncio
async def test_run_uses_integrated_adapter_once_and_returns_complete_trace(
    monkeypatch,
) -> None:
    search_result = make_search_result()
    reading_result = make_reading_result()
    search_agent = FakeSearchAgent(search_result)
    reading_workflow = FakeReadingWorkflow(reading_result)
    budget = SearchBudget(max_rounds=2, max_queries=4)
    base_adapter = AgenticM2Module(
        search_agent=search_agent,
        reading_workflow=reading_workflow,
        budget=budget,
    )
    client_models: list[str] = []
    builder_calls: list[dict] = []

    class FakeQwenClient:
        def __init__(self, *, model: str) -> None:
            client_models.append(model)

    def fake_builder(**kwargs):
        builder_calls.append(kwargs)
        return base_adapter

    monkeypatch.setattr(run_m2_integrated, "QwenClient", FakeQwenClient)
    monkeypatch.setattr(
        run_m2_integrated, "build_integrated_search_adapter", fake_builder
    )

    payload = await run_m2_integrated._run(
        question=QUESTION,
        model="qwen3.6-plus",
        limit=7,
        timeout=12.5,
    )

    assert client_models == ["qwen3.6-plus"]
    assert len(builder_calls) == 1
    assert isinstance(builder_calls[0]["client"], FakeQwenClient)
    assert builder_calls[0]["final_k"] == 7
    assert builder_calls[0]["source_timeout_seconds"] == 12.5
    assert search_agent.calls == 1
    assert reading_workflow.calls == 1
    assert payload == {
        "status": "ok",
        "trace": {
            "search_runs": [search_result.model_dump(mode="json")],
            "reading_results": [reading_result.model_dump(mode="json")],
        },
        "literature_results": [
            {
                "sub_question": QUESTION,
                "papers_retrieved": 1,
                "knowledge_entries": [],
            }
        ],
    }


def test_cli_emits_utf8_trace_and_final_output(monkeypatch, capsys) -> None:
    async def fake_run(**kwargs):
        assert kwargs == {
            "question": QUESTION,
            "model": "qwen3.6-plus",
            "limit": 5,
            "timeout": 30.0,
        }
        return {
            "status": "ok",
            "trace": {
                "search_runs": [{"stop_reason": "coverage_satisfied"}],
                "reading_results": [{"paper_id": "PMID:1"}],
            },
            "literature_results": [
                {"sub_question": QUESTION, "papers_retrieved": 1}
            ],
        }

    monkeypatch.setattr(run_m2_integrated, "_run", fake_run)

    code = run_m2_integrated.main(
        ["--question", QUESTION, "--model", "qwen3.6-plus"]
    )

    streams = capsys.readouterr()
    payload = json.loads(streams.out)
    assert code == 0
    assert streams.err == ""
    assert payload["status"] == "ok"
    assert payload["trace"]["search_runs"][0]["stop_reason"] == (
        "coverage_satisfied"
    )
    assert QUESTION in streams.out
    assert "\\u" not in streams.out


def test_cli_success_falls_back_to_ascii_json_on_strict_gbk_stdout(
    monkeypatch,
) -> None:
    async def fake_run(**kwargs):
        return {
            "status": "ok",
            "trace": {"search_runs": [], "reading_results": []},
            "literature_results": [{"title": "Jörg Hippo paper"}],
        }

    raw_stdout = io.BytesIO()
    gbk_stdout = io.TextIOWrapper(
        raw_stdout, encoding="gbk", errors="strict", newline=""
    )
    monkeypatch.setattr(run_m2_integrated, "_run", fake_run)
    monkeypatch.setattr(run_m2_integrated.sys, "stdout", gbk_stdout)

    code = run_m2_integrated.main(["--question", "question"])

    gbk_stdout.flush()
    output = raw_stdout.getvalue().decode("gbk")
    assert code == 0
    assert json.loads(output)["literature_results"][0]["title"] == (
        "Jörg Hippo paper"
    )
    assert "\\u00f6" in output


def test_cli_error_falls_back_to_ascii_json_on_strict_gbk_stderr(
    monkeypatch,
) -> None:
    async def fail(**kwargs):
        raise RuntimeError("source Jörg unavailable")

    raw_stderr = io.BytesIO()
    gbk_stderr = io.TextIOWrapper(
        raw_stderr, encoding="gbk", errors="strict", newline=""
    )
    monkeypatch.setattr(run_m2_integrated, "_run", fail)
    monkeypatch.setattr(run_m2_integrated.sys, "stderr", gbk_stderr)

    code = run_m2_integrated.main(["--question", "question"])

    gbk_stderr.flush()
    output = raw_stderr.getvalue().decode("gbk")
    assert code == 1
    assert json.loads(output)["message"] == "source Jörg unavailable"
    assert "\\u00f6" in output


def test_cli_emits_sanitized_structured_runtime_error(monkeypatch, capsys) -> None:
    async def fail(**kwargs):
        raise RuntimeError("source\n\t unavailable " + "x" * 600)

    monkeypatch.setattr(run_m2_integrated, "_run", fail)

    code = run_m2_integrated.main(["--question", "question"])

    streams = capsys.readouterr()
    payload = json.loads(streams.err)
    assert code == 1
    assert streams.out == ""
    assert payload["status"] == "error"
    assert payload["error_type"] == "RuntimeError"
    assert payload["message"].startswith("source unavailable ")
    assert len(payload["message"]) == 500
    assert "\n" not in payload["message"]
    assert "\t" not in payload["message"]


@pytest.mark.parametrize(
    ("argv", "message_fragment"),
    [
        ([], "--question"),
        (["--question", "question", "--limit", "many"], "invalid int value"),
        (
            ["--question", "question", "--timeout", "forever"],
            "invalid float value",
        ),
    ],
)
def test_cli_argument_errors_are_json(
    argv, message_fragment, capsys
) -> None:
    code = run_m2_integrated.main(argv)

    streams = capsys.readouterr()
    payload = json.loads(streams.err)
    assert code == 1
    assert streams.out == ""
    assert payload["status"] == "error"
    assert payload["error_type"] == "CLIArgumentError"
    assert message_fragment in payload["message"]


def test_cli_help_keeps_standard_success_exit(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        run_m2_integrated.main(["--help"])

    streams = capsys.readouterr()
    assert exc_info.value.code == 0
    assert streams.out.startswith("usage:")
    assert "--model" in streams.out
    assert "--limit" in streams.out
    assert "--timeout" in streams.out
    assert streams.err == ""
