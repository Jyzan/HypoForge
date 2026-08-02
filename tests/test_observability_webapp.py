from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from hypoforge.observability import RunEventRecorder, bind_recorder, emit_event
from hypoforge.state import PipelineState
from hypoforge.webapp import RunManager


def test_recorder_persists_events_and_snapshots(tmp_path: Path) -> None:
    recorder = RunEventRecorder(tmp_path / "run-1", "run-1")
    recorder.emit(
        "run_started",
        status="running",
        message="start",
        details={"question": "q"},
    )
    with bind_recorder(recorder):
        emit_event(
            "tool_completed",
            module="m2",
            tool="pubmed",
            status="completed",
            details={"papers": 3},
        )
    snapshot = recorder.save_snapshot("m2-iteration-0", {"papers": [1, 2, 3]})
    recorder.write_manifest({"run_id": "run-1", "status": "running"})

    events = recorder.read_events()
    assert [event["sequence"] for event in events] == [1, 2]
    assert events[1]["tool"] == "pubmed"
    assert json.loads(snapshot.read_text(encoding="utf-8"))["papers"] == [1, 2, 3]
    assert json.loads(recorder.manifest_path.read_text(encoding="utf-8"))[
        "run_id"
    ] == "run-1"


def test_run_manager_reads_persisted_result_and_artifacts(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("enabled_modules: [m1]\n", encoding="utf-8")
    manager = RunManager(config_path=config_path, output_root=tmp_path / "runs")
    run_id = "ui-test"
    run_dir = tmp_path / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"run_id": run_id, "status": "completed"}),
        encoding="utf-8",
    )
    (run_dir / f"{run_id}.json").write_text(
        json.dumps(
            {
                "input_question": "q",
                "top_hypotheses": [{"hypothesis_id": "H1", "statement": "s"}],
                "research_plans": [],
                "reviews": [],
                "errors": [],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / f"{run_id}_scores.json").write_text(
        json.dumps({"aggregate": {"top1_composite": 0.8}}),
        encoding="utf-8",
    )

    assert manager.get(run_id)["status"] == "completed"
    assert manager.result(run_id)["scores"]["aggregate"]["top1_composite"] == 0.8
    names = {item["name"] for item in manager.artifacts(run_id)}
    assert f"{run_id}.json" in names
    assert f"{run_id}_scores.json" in names


def test_run_credentials_are_memory_only_and_applied_per_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
enabled_modules: [m1]
qwen:
  base: {model: default-model}
  max: {model: default-model}
  plus: {model: default-model}
  turbo: {model: default-model}
module_overrides:
  m2:
    kwargs: {}
""",
        encoding="utf-8",
    )
    manager = RunManager(config_path=config_path, output_root=tmp_path / "runs")
    captured = {}

    class DeferredThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self) -> None:
            pass

    class FakeRunner:
        def __init__(self, config, event_recorder=None):
            captured["config"] = config

        async def run(self, question: str, run_id: str):
            return PipelineState(input_question=question, run_id=run_id)

    monkeypatch.setattr(threading, "Thread", DeferredThread)
    monkeypatch.setattr("hypoforge.webapp.PipelineRunner", FakeRunner)
    run = manager.start(
        "q",
        model_name="qwen-custom",
        qwen_api_key="qwen-secret-for-test",
        semantic_scholar_api_key="s2-secret-for-test",
    )
    assert "secret" not in json.dumps(run)

    manager._run_pipeline(run["run_id"], "q")
    config = captured["config"]
    assert config.qwen.plus.model == "qwen-custom"
    assert config.qwen.plus.api_key == "qwen-secret-for-test"
    assert (
        config.module_overrides["m2"].kwargs["semantic_scholar_api_key"]
        == "s2-secret-for-test"
    )
    persisted = (
        tmp_path / "runs" / run["run_id"] / "manifest.json"
    ).read_text(encoding="utf-8")
    assert "qwen-secret-for-test" not in persisted
    assert "s2-secret-for-test" not in persisted


def test_web_ui_preserves_open_event_details_and_has_ephemeral_key_fields() -> None:
    html_path = (
        Path(__file__).resolve().parent.parent
        / "hypoforge"
        / "web"
        / "index.html"
    )
    html = html_path.read_text(encoding="utf-8")

    assert 'id="modelName"' in html
    assert 'id="qwenApiKey"' in html
    assert 'id="semanticApiKey"' in html
    assert "localStorage" not in html
    assert "openSequences" in html
    assert "data-event-sequence" in html
