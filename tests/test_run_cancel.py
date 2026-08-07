"""Run cancellation: cooperative stop, knowledge retention, cancel endpoint.

Covers task #24:

1. The cancel flag stops the execution loop after the current module, and
   the accumulated state (problem card / literature knowledge) is persisted.
2. ``POST /api/runs/{run_id}/cancel`` status codes:
   202 cancelling / 409 finished / 404 unknown / 400 malformed id.
3. A cancelled run keeps its knowledge artifacts and can serve as the parent
   of a follow-up run (``build_followup_seed`` + ``RunManager.start``).
4. Cancelling an already-finished run is rejected.
"""

from __future__ import annotations

import http.client
import json
import threading
from pathlib import Path

import pytest

from hypoforge.config import PipelineConfig
from hypoforge.observability import RunEventRecorder
from hypoforge.pipeline import PipelineRunner, build_followup_seed
from hypoforge.protocol import ModuleProtocol
from hypoforge.state import (
    HypothesisCard,
    M2KnowledgeExport,
    M2KnowledgeRun,
    PipelineState,
    ProblemCard,
)
from hypoforge.webapp import RunManager, build_server


# --------------------------------------------------------------------------- #
# Fake modules for the graph-level cancellation test
# --------------------------------------------------------------------------- #

def _make_fake_module(name: str, patch: dict, hook=None):
    class FakeModule(ModuleProtocol):
        module_name = name
        module_version = "test"
        description = f"fake {name}"

        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def __call__(self, state, config=None):
            if hook is not None:
                hook()
            return dict(patch)

        @classmethod
        def get_input_fields(cls):
            return []

        @classmethod
        def get_output_fields(cls):
            return list(patch)

    return FakeModule()


def _runner_with_fakes(tmp_path: Path, cancel_event, m1_hook=None, m4_hook=None):
    """A PipelineRunner wired with fake m1/m4 modules (no LLM / network).

    Returns ``(runner, recorder, modules)``; callers must monkeypatch
    ``ModuleRegistry.build_all`` to return *modules*.
    """
    problem_card = ProblemCard(
        original_question="问题", sub_questions=["子问题 A"]
    )
    hypothesis = HypothesisCard(hypothesis_id="H1", statement="假设陈述")
    modules = {
        "m1": _make_fake_module(
            "m1", {"problem_card": problem_card}, hook=m1_hook
        ),
        "m4": _make_fake_module(
            "m4", {"top_hypotheses": [hypothesis]}, hook=m4_hook
        ),
    }
    config = PipelineConfig(verbose=False, enabled_modules=["m1", "m4"])
    config.output_dir = str(tmp_path)
    recorder = RunEventRecorder(tmp_path, "run-cancel")
    runner = PipelineRunner(
        config, event_recorder=recorder, cancel_event=cancel_event
    )
    return runner, recorder, modules


@pytest.mark.asyncio
async def test_cancel_flag_stops_after_current_module_and_persists_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cancel set during m1 → m4 must never run; state file keeps m1 output."""
    from hypoforge.registry import ModuleRegistry

    cancel_event = threading.Event()
    m4_calls = {"count": 0}

    def cancel_after_m1():
        cancel_event.set()  # "module completes then stop" (cooperative)

    def count_m4():
        m4_calls["count"] += 1

    runner, recorder, modules = _runner_with_fakes(
        tmp_path, cancel_event, m1_hook=cancel_after_m1, m4_hook=count_m4
    )
    monkeypatch.setattr(
        ModuleRegistry, "build_all", staticmethod(lambda cfg: modules)
    )
    state = await runner.run("问题", run_id="run-cancel")

    assert runner.cancelled is True
    assert m4_calls["count"] == 0
    # m1 artifacts retained, m4 never produced anything
    assert state.problem_card is not None
    assert state.problem_card.sub_questions == ["子问题 A"]
    assert state.top_hypotheses == []

    # Final state persisted → usable as a followup parent
    state_path = tmp_path / "run-cancel.json"
    assert state_path.exists()
    persisted = json.loads(state_path.read_text(encoding="utf-8"))
    assert persisted["problem_card"]["sub_questions"] == ["子问题 A"]

    events = recorder.read_events()
    types = [event["event_type"] for event in events]
    assert "module_cancelled" in types  # m4 refused to start
    assert "run_cancelled" in types
    assert "run_completed" not in types
    cancelled_event = next(
        event for event in events if event["event_type"] == "run_cancelled"
    )
    assert cancelled_event["status"] == "cancelled"


@pytest.mark.asyncio
async def test_no_cancel_runs_to_completion(tmp_path: Path, monkeypatch):
    """Control: without a cancel request the same fake graph completes."""
    from hypoforge.registry import ModuleRegistry

    runner, recorder, modules = _runner_with_fakes(
        tmp_path, threading.Event()
    )
    monkeypatch.setattr(
        ModuleRegistry, "build_all", staticmethod(lambda cfg: modules)
    )
    state = await runner.run("问题", run_id="run-full")

    assert runner.cancelled is False
    assert state.problem_card is not None
    assert [h.hypothesis_id for h in state.top_hypotheses] == ["H1"]
    types = [event["event_type"] for event in recorder.read_events()]
    assert "run_completed" in types
    assert "run_cancelled" not in types


# --------------------------------------------------------------------------- #
# Knowledge retention: cancelled parent → followup seed
# --------------------------------------------------------------------------- #

def test_build_followup_seed_from_cancelled_parent_state() -> None:
    """A cancelled run's persisted state must feed build_followup_seed with
    the problem card and the M2 literature knowledge intact."""
    seed_state = PipelineState(
        input_question="原问题",
        run_id="ui-cancelled-parent",
        problem_card=ProblemCard(
            original_question="原问题", sub_questions=["子问题 A"]
        ),
        m2_knowledge_export=M2KnowledgeExport(
            runs=[M2KnowledgeRun(sub_question="子问题 A")]
        ),
    ).model_dump(mode="json")

    seed = build_followup_seed(
        seed_state=seed_state,
        followup_text="细化对照组设计",
        run_id="ui-child",
        config=PipelineConfig(verbose=False),
    )

    assert seed.parent_run_id == "ui-cancelled-parent"
    assert seed.problem_card is not None
    assert seed.problem_card.sub_questions == ["子问题 A"]
    assert seed.m2_knowledge_export is not None
    assert seed.m2_knowledge_export.runs[0].sub_question == "子问题 A"
    assert seed.input_question == "细化对照组设计"


# --------------------------------------------------------------------------- #
# RunManager: request_cancel semantics + worker cancelled status
# --------------------------------------------------------------------------- #

def _write_config(tmp_path: Path) -> Path:
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
    return config_path


class DeferredThread:
    """threading.Thread stand-in that never starts the worker."""

    def __init__(self, *args, **kwargs):
        pass

    def start(self) -> None:
        pass


class CancellingRunner:
    """FakeRunner honouring the cooperative cancel flag like the real one."""

    def __init__(self, config, event_recorder=None):
        self.config = config
        self.cancel_event = None
        self.cancelled = False

    async def run(self, question: str, run_id: str, **kwargs):
        self.cancelled = bool(
            self.cancel_event is not None and self.cancel_event.is_set()
        )
        state = PipelineState(
            input_question=question,
            run_id=run_id,
            problem_card=ProblemCard(
                original_question=question,
                sub_questions=["已保留的问题卡"],
            ),
        )
        if self.cancelled:
            # Mimic PipelineRunner._save_output on a graceful stop: the
            # accumulated state must land on disk for followup seeding.
            out_dir = Path(self.config.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"{run_id}.json").write_text(
                json.dumps(state.model_dump(mode="json"), ensure_ascii=False),
                encoding="utf-8",
            )
        return state


def test_request_cancel_marks_cancelled_and_retains_knowledge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = RunManager(
        config_path=_write_config(tmp_path), output_root=tmp_path / "runs"
    )
    monkeypatch.setattr(threading, "Thread", DeferredThread)
    monkeypatch.setattr("hypoforge.webapp.PipelineRunner", CancellingRunner)

    run = manager.start("原问题")
    run_id = run["run_id"]

    info = manager.request_cancel(run_id)
    assert info == {"run_id": run_id, "status": "cancelling"}
    assert manager._cancel_events[run_id].is_set()
    assert manager.get(run_id)["cancel_requested"] is True
    # idempotent while cancelling
    assert manager.request_cancel(run_id)["status"] == "cancelling"

    # The worker observes the flag and the run ends as cancelled.
    manager._run_pipeline(run_id, "原问题")
    assert manager.get(run_id)["status"] == "cancelled"

    manifest = json.loads(
        (tmp_path / "runs" / run_id / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["status"] == "cancelled"
    assert "停止" in manifest["cancel_message"]

    events = RunEventRecorder(
        tmp_path / "runs" / run_id, run_id
    ).read_events()
    types = [event["event_type"] for event in events]
    # run_cancelled itself is emitted by the real pipeline runner (covered
    # by the graph-level test above); the web layer records the request.
    assert "cancel_requested" in types

    # Knowledge retention: the final state keeps the problem card …
    state = json.loads(
        (tmp_path / "runs" / run_id / f"{run_id}.json").read_text(
            encoding="utf-8"
        )
    )
    assert state["problem_card"]["sub_questions"] == ["已保留的问题卡"]

    # … and the cancelled run is accepted as a followup parent.
    followup = manager.start(
        "细化对照组", parent_run_id=run_id, followup="细化对照组"
    )
    assert followup["parent_run_id"] == run_id


def test_request_cancel_rejects_finished_unknown_and_malformed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = RunManager(
        config_path=_write_config(tmp_path), output_root=tmp_path / "runs"
    )
    monkeypatch.setattr(threading, "Thread", DeferredThread)
    monkeypatch.setattr("hypoforge.webapp.PipelineRunner", CancellingRunner)

    run = manager.start("原问题")
    run_id = run["run_id"]
    manager._run_pipeline(run_id, "原问题")  # finishes (no cancel requested)
    assert manager.get(run_id)["status"] == "completed"

    with pytest.raises(RuntimeError, match="已结束"):
        manager.request_cancel(run_id)
    with pytest.raises(LookupError, match="不存在"):
        manager.request_cancel("ui-ghost-run")
    with pytest.raises(ValueError, match="非法字符"):
        manager.request_cancel("../escape")

    # Persisted-only run (server restarted) → "已结束", not 404.
    finished_dir = tmp_path / "runs" / "ui-old-finished"
    finished_dir.mkdir(parents=True)
    (finished_dir / "manifest.json").write_text(
        json.dumps({"run_id": "ui-old-finished", "status": "completed"}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="已结束"):
        manager.request_cancel("ui-old-finished")


# --------------------------------------------------------------------------- #
# HTTP endpoint contract
# --------------------------------------------------------------------------- #

def test_http_cancel_endpoint_statuses(tmp_path: Path) -> None:
    _write_config(tmp_path)
    server = build_server(
        host="127.0.0.1",
        port=0,
        config_path=tmp_path / "config.yaml",
        output_root=tmp_path / "runs",
        static_dir=tmp_path,
    )
    # A live in-memory running run.
    server.manager._runs["ui-live"] = {
        "run_id": "ui-live",
        "status": "running",
        "cancel_requested": False,
    }
    server.manager._cancel_events["ui-live"] = threading.Event()
    # An in-memory finished run.
    server.manager._runs["ui-done"] = {
        "run_id": "ui-done",
        "status": "completed",
    }
    # A persisted-only finished run.
    done_dir = tmp_path / "runs" / "ui-disk-done"
    done_dir.mkdir(parents=True)
    (done_dir / "manifest.json").write_text(
        json.dumps({"run_id": "ui-disk-done", "status": "completed"}),
        encoding="utf-8",
    )

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)

        # 运行中 → 202 cancelling，标志置位 + cancel_requested 事件落盘。
        conn.request("POST", "/api/runs/ui-live/cancel")
        response = conn.getresponse()
        assert response.status == 202
        assert json.loads(response.read()) == {
            "run_id": "ui-live",
            "status": "cancelling",
        }
        assert server.manager._cancel_events["ui-live"].is_set()
        events = RunEventRecorder(
            tmp_path / "runs" / "ui-live", "ui-live"
        ).read_events()
        assert any(
            event["event_type"] == "cancel_requested" for event in events
        )

        # 幂等：再次请求仍是 cancelling。
        conn.request("POST", "/api/runs/ui-live/cancel")
        response = conn.getresponse()
        assert response.status == 202
        assert json.loads(response.read())["status"] == "cancelling"

        # 已结束（内存记录 / 仅落盘）→ 409。
        conn.request("POST", "/api/runs/ui-done/cancel")
        response = conn.getresponse()
        assert response.status == 409
        assert "已结束" in json.loads(response.read())["error"]

        conn.request("POST", "/api/runs/ui-disk-done/cancel")
        response = conn.getresponse()
        assert response.status == 409
        assert "已结束" in json.loads(response.read())["error"]

        # 不存在 → 404。
        conn.request("POST", "/api/runs/ui-ghost/cancel")
        response = conn.getresponse()
        assert response.status == 404
        assert "不存在" in json.loads(response.read())["error"]

        # 非法 run_id → 400。
        conn.request("POST", "/api/runs/..%2Fxxx/cancel")
        response = conn.getresponse()
        assert response.status == 400
    finally:
        server.shutdown()
        server.server_close()
