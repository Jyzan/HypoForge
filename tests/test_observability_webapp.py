from __future__ import annotations

import http.client
import json
import threading
from pathlib import Path

import pytest

from hypoforge.observability import RunEventRecorder, bind_recorder, emit_event
from hypoforge.state import PipelineState
from hypoforge.webapp import RunManager, build_server


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
        openalex_api_key="oa-secret-for-test",
        openalex_mailto="lab@example.org",
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
    assert (
        config.module_overrides["m2"].kwargs["openalex_api_key"]
        == "oa-secret-for-test"
    )
    assert (
        config.module_overrides["m2"].kwargs["openalex_mailto"]
        == "lab@example.org"
    )
    persisted = (
        tmp_path / "runs" / run["run_id"] / "manifest.json"
    ).read_text(encoding="utf-8")
    assert "qwen-secret-for-test" not in persisted
    assert "s2-secret-for-test" not in persisted
    assert "oa-secret-for-test" not in persisted


def test_run_manager_rejects_malformed_openalex_credentials(
    tmp_path: Path,
) -> None:
    manager = RunManager(
        config_path=_write_config(tmp_path), output_root=tmp_path / "runs"
    )
    with pytest.raises(ValueError, match="API Key 长度异常"):
        manager.start("q", openalex_api_key="x" * 4097)
    with pytest.raises(ValueError, match="Mailto 长度异常"):
        manager.start("q", openalex_mailto="a" * 321)


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
    assert 'id="openalexApiKey"' in html
    assert 'id="openalexMailto"' in html
    # Ephemeral keys: never echoed back, persisted, or sent in plaintext.
    assert "localStorage" not in html
    assert 'id="openalexApiKey" type="password"' in html
    assert "openSequences" in html
    assert "data-event-sequence" in html


class DeferredThread:
    def __init__(self, *args, **kwargs):
        pass

    def start(self) -> None:
        pass


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


def _make_parent_run(tmp_path: Path, parent_id: str = "ui-parent") -> Path:
    parent_dir = tmp_path / "runs" / parent_id
    parent_dir.mkdir(parents=True)
    (parent_dir / f"{parent_id}.json").write_text(
        json.dumps(
            {
                "run_id": parent_id,
                "input_question": "parent question",
                "problem_card": {"original_question": "parent question"},
                "top_hypotheses": [{"statement": "parent H1"}],
                "errors": [],
            }
        ),
        encoding="utf-8",
    )
    (parent_dir / "manifest.json").write_text(
        json.dumps({"run_id": parent_id, "status": "completed"}),
        encoding="utf-8",
    )
    return parent_dir


def test_followup_defaults_keep_behavior_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = RunManager(
        config_path=_write_config(tmp_path), output_root=tmp_path / "runs"
    )
    captured: dict = {}

    class FakeRunner:
        def __init__(self, config, event_recorder=None):
            pass

        async def run(self, question: str, run_id: str, **kwargs):
            captured["kwargs"] = kwargs
            return PipelineState(input_question=question, run_id=run_id)

    monkeypatch.setattr(threading, "Thread", DeferredThread)
    monkeypatch.setattr("hypoforge.webapp.PipelineRunner", FakeRunner)
    run = manager.start("q")
    assert "parent_run_id" not in run
    manager._run_pipeline(run["run_id"], "q")
    # No followup -> run() called without seed_state/followup_text kwargs.
    assert captured["kwargs"] == {}
    manifest = json.loads(
        (tmp_path / "runs" / run["run_id"] / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert "parent_run_id" not in manifest


def test_followup_missing_parent_returns_error(tmp_path: Path) -> None:
    manager = RunManager(
        config_path=_write_config(tmp_path), output_root=tmp_path / "runs"
    )
    with pytest.raises(ValueError, match="不存在"):
        manager.start("追问", parent_run_id="ghost-run", followup="再细化一下")
    with pytest.raises(ValueError):
        manager.start("追问", followup="只有追问没有父运行")
    with pytest.raises(ValueError, match="非法字符"):
        manager.start("追问", parent_run_id="../escape", followup="x")


def test_followup_corrupt_parent_state_returns_error(tmp_path: Path) -> None:
    manager = RunManager(
        config_path=_write_config(tmp_path), output_root=tmp_path / "runs"
    )
    parent_dir = tmp_path / "runs" / "ui-corrupt"
    parent_dir.mkdir(parents=True)
    (parent_dir / "ui-corrupt.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="损坏"):
        manager.start("追问", parent_run_id="ui-corrupt", followup="x")


def test_followup_passes_seed_state_and_marks_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = RunManager(
        config_path=_write_config(tmp_path), output_root=tmp_path / "runs"
    )
    _make_parent_run(tmp_path)
    captured: dict = {}

    class FakeRunner:
        def __init__(self, config, event_recorder=None):
            pass

        async def run(self, question: str, run_id: str, **kwargs):
            captured["kwargs"] = kwargs
            return PipelineState(input_question=question, run_id=run_id)

    monkeypatch.setattr(threading, "Thread", DeferredThread)
    monkeypatch.setattr("hypoforge.webapp.PipelineRunner", FakeRunner)
    run = manager.start(
        "请细化对照组设计",
        parent_run_id="ui-parent",
        followup="请细化对照组设计",
    )
    assert run["parent_run_id"] == "ui-parent"
    manager._run_pipeline(run["run_id"], "请细化对照组设计")
    kwargs = captured["kwargs"]
    assert kwargs["followup_text"] == "请细化对照组设计"
    assert kwargs["seed_state"]["run_id"] == "ui-parent"
    manifest = json.loads(
        (tmp_path / "runs" / run["run_id"] / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["parent_run_id"] == "ui-parent"
    assert manifest["followup"] == "请细化对照组设计"


def test_single_run_mutex_error_message_mentions_waiting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = RunManager(
        config_path=_write_config(tmp_path), output_root=tmp_path / "runs"
    )
    monkeypatch.setattr(threading, "Thread", DeferredThread)
    manager.start("第一条")
    with pytest.raises(RuntimeError, match="当前运行结束后才能提交"):
        manager.start("第二条")


def test_followup_rejected_with_409_while_run_in_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Submitting a followup while another run is executing must be rejected
    with the mutex error (HTTP layer maps RuntimeError → 409); no queueing."""
    manager = RunManager(
        config_path=_write_config(tmp_path), output_root=tmp_path / "runs"
    )
    _make_parent_run(tmp_path)
    monkeypatch.setattr(threading, "Thread", DeferredThread)
    manager.start("第一条")
    with pytest.raises(RuntimeError, match="当前运行结束后才能提交"):
        manager.start("追问", parent_run_id="ui-parent", followup="细化对照组")


def _write_routing_fixture(tmp_path: Path, run_id: str) -> None:
    run_dir = tmp_path / "runs" / run_id
    recorder = RunEventRecorder(run_dir, run_id)
    recorder.emit(
        "module_started", module="m1", status="running",
        message="M1 开始", details={"iteration_count": 0},
    )
    recorder.emit(
        "routing_decision", module="m1", status="completed", message="route",
        details={"from": "m1", "to": "search_m2", "decided_by": "policy",
                 "reason": "standard path", "gap_ids": [], "round": 1},
    )
    for name in ("m2", "m3", "m4", "m5", "m6"):
        recorder.emit(
            "module_started", module=name, status="running",
            message=f"{name} 开始", details={"iteration_count": 0},
        )
    recorder.emit(
        "routing_decision", module="m6", status="completed", message="route",
        details={"from": "m6", "to": "revise_m4", "decided_by": "policy",
                 "reason": "revise hypotheses", "gap_ids": ["G-1"], "round": 2},
    )
    for name in ("m4", "m5", "m6"):
        recorder.emit(
            "module_started", module=name, status="running",
            message=f"{name} 再来一轮", details={"iteration_count": 1},
        )
    recorder.emit(
        "routing_decision", module="m6", status="completed", message="route",
        details={"from": "m6", "to": "end", "decided_by": "policy",
                 "reason": "threshold met", "gap_ids": [], "round": 3},
    )
    (run_dir / f"{run_id}.json").write_text(
        json.dumps(
            {
                "input_question": "q",
                "reviews": [
                    {"dimension": "overall", "score": 3.5, "version": 1},
                    {"dimension": "overall", "score": 4.5, "version": 2},
                ],
                "evidence_gaps": [{"gap_id": "G-1", "status": "closed"}],
                "top_hypotheses": [{"statement": "final H1"}],
                "research_plans": [{"hypothesis_id": "H1"}],
                "iteration_count": 2,
                "search_round": 1,
                "routing_history": [
                    {"round": 1, "from_module": "m1", "to_module": "search_m2",
                     "decided_by": "policy", "reason": "standard path", "gap_ids": []},
                    {"round": 2, "from_module": "m6", "to_module": "revise_m4",
                     "decided_by": "policy", "reason": "revise hypotheses",
                     "gap_ids": ["G-1"]},
                    {"round": 3, "from_module": "m6", "to_module": "end",
                     "decided_by": "policy", "reason": "threshold met", "gap_ids": []},
                ],
                "errors": [],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "snapshots").mkdir(exist_ok=True)
    (run_dir / "snapshots" / "001-m4-r0-iter0.json").write_text(
        json.dumps({"top_hypotheses": [{"statement": "v1 H1"}], "research_plans": [{}]}),
        encoding="utf-8",
    )
    (run_dir / "snapshots" / "002-m6-r0-iter1.json").write_text(
        json.dumps(
            {
                "top_hypotheses": [{"statement": "v2 H1"}],
                "research_plans": [{}, {}],
                "reviews": [{"dimension": "overall", "score": 4.5, "version": 2}],
            }
        ),
        encoding="utf-8",
    )


def test_rounds_endpoint_contract_shape_from_state_json(tmp_path: Path) -> None:
    manager = RunManager(
        config_path=_write_config(tmp_path), output_root=tmp_path / "runs"
    )
    _write_routing_fixture(tmp_path, "ui-rounds")
    payload = manager.rounds("ui-rounds")
    # Contract shape: {"routing_history": [...], "search_round": n}
    assert set(payload) == {"routing_history", "search_round"}
    assert payload["search_round"] == 1
    assert [d["to_module"] for d in payload["routing_history"]] == [
        "search_m2", "revise_m4", "end",
    ]
    assert payload["routing_history"][1]["gap_ids"] == ["G-1"]
    # Missing run → empty default shape (never a crash, never a raw list).
    assert manager.rounds("missing-run") == {"routing_history": [], "search_round": 0}


def test_versions_endpoint_groups_reviews_and_snapshots(tmp_path: Path) -> None:
    manager = RunManager(
        config_path=_write_config(tmp_path), output_root=tmp_path / "runs"
    )
    _write_routing_fixture(tmp_path, "ui-vers")
    versions = manager.versions("ui-vers")
    assert manager.versions("missing-run") == []
    assert [v["version"] for v in versions] == [1, 2]
    # Contract shape: each version entry carries version/overall/timestamp.
    for entry in versions:
        assert {"version", "overall", "timestamp"} <= set(entry)
    v1, v2 = versions
    assert v1["overall"] == 3.5
    assert v1["top_hypothesis_titles"] == ["v1 H1"]
    assert v1["research_plans_count"] == 1
    assert v1["research_plans"] == [{}]
    assert v1["snapshots"] == ["snapshots/001-m4-r0-iter0.json"]
    assert v2["overall"] == 4.5
    # Final state is authoritative for the latest version.
    assert v2["top_hypothesis_titles"] == ["final H1"]
    assert v2["research_plans_count"] == 1
    assert v2["research_plans"] == [{"hypothesis_id": "H1"}]
    assert "snapshots/002-m6-r0-iter1.json" in v2["snapshots"]


def test_result_exposes_routing_fields(tmp_path: Path) -> None:
    manager = RunManager(
        config_path=_write_config(tmp_path), output_root=tmp_path / "runs"
    )
    _write_routing_fixture(tmp_path, "ui-res")
    result = manager.result("ui-res")
    assert result["search_round"] == 1
    assert result["revision_count"] == 0
    assert [d["to_module"] for d in result["routing_history"]] == [
        "search_m2", "revise_m4", "end",
    ]
    assert result["evidence_gaps"][0]["status"] == "closed"


def _make_run(
    tmp_path: Path, run_id: str, parent: str = "", status: str = "completed"
) -> Path:
    run_dir = tmp_path / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"run_id": run_id, "status": status}
    if parent:
        manifest["parent_run_id"] = parent
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return run_dir


def test_delete_run_removes_directory_and_returns_list(tmp_path: Path) -> None:
    manager = RunManager(
        config_path=_write_config(tmp_path), output_root=tmp_path / "runs"
    )
    _make_run(tmp_path, "ui-del-a")
    deleted = manager.delete("ui-del-a")
    assert deleted == ["ui-del-a"]
    assert not (tmp_path / "runs" / "ui-del-a").exists()


def test_delete_chain_root_cascades_to_descendants(tmp_path: Path) -> None:
    manager = RunManager(
        config_path=_write_config(tmp_path), output_root=tmp_path / "runs"
    )
    _make_run(tmp_path, "ui-root")
    _make_run(tmp_path, "ui-child", parent="ui-root")
    _make_run(tmp_path, "ui-grandchild", parent="ui-child")
    _make_run(tmp_path, "ui-unrelated")
    deleted = manager.delete("ui-root")
    assert set(deleted) == {"ui-root", "ui-child", "ui-grandchild"}
    for run_id in ("ui-root", "ui-child", "ui-grandchild"):
        assert not (tmp_path / "runs" / run_id).exists()
    assert (tmp_path / "runs" / "ui-unrelated").exists()


def test_delete_middle_run_cascades_only_its_subtree(tmp_path: Path) -> None:
    manager = RunManager(
        config_path=_write_config(tmp_path), output_root=tmp_path / "runs"
    )
    _make_run(tmp_path, "ui-root")
    _make_run(tmp_path, "ui-child", parent="ui-root")
    _make_run(tmp_path, "ui-grandchild", parent="ui-child")
    deleted = manager.delete("ui-child")
    assert set(deleted) == {"ui-child", "ui-grandchild"}
    assert (tmp_path / "runs" / "ui-root").exists()


def test_delete_missing_run_raises_lookup_error(tmp_path: Path) -> None:
    manager = RunManager(
        config_path=_write_config(tmp_path), output_root=tmp_path / "runs"
    )
    with pytest.raises(LookupError, match="不存在"):
        manager.delete("ui-ghost")


def test_delete_illegal_run_id_rejected_and_nothing_deleted(
    tmp_path: Path,
) -> None:
    manager = RunManager(
        config_path=_write_config(tmp_path), output_root=tmp_path / "runs"
    )
    kept = _make_run(tmp_path, "ui-keep")
    outside = tmp_path / "outside"
    outside.mkdir()
    for bad_id in ("../outside", "ui-../outside", "../xxx", "plain-name", ""):
        with pytest.raises(ValueError):
            manager.delete(bad_id)
    assert kept.exists()
    assert outside.exists()


def test_delete_running_run_rejected_with_conflict(tmp_path: Path) -> None:
    """Running state is injected into the manager's in-memory registry (the
    same mechanism start() uses) since a live pipeline run cannot be built
    in a unit test."""
    manager = RunManager(
        config_path=_write_config(tmp_path), output_root=tmp_path / "runs"
    )
    run_dir = _make_run(tmp_path, "ui-busy", status="running")
    manager._runs["ui-busy"] = {"run_id": "ui-busy", "status": "running"}
    with pytest.raises(RuntimeError, match="正在执行中"):
        manager.delete("ui-busy")
    assert run_dir.exists()


def test_delete_rejected_when_descendant_is_running(tmp_path: Path) -> None:
    manager = RunManager(
        config_path=_write_config(tmp_path), output_root=tmp_path / "runs"
    )
    root_dir = _make_run(tmp_path, "ui-root")
    child_dir = _make_run(tmp_path, "ui-child", parent="ui-root", status="running")
    manager._runs["ui-child"] = {"run_id": "ui-child", "status": "running"}
    with pytest.raises(RuntimeError, match="正在执行中"):
        manager.delete("ui-root")
    assert root_dir.exists()
    assert child_dir.exists()


def test_http_delete_endpoint_statuses(tmp_path: Path) -> None:
    _write_config(tmp_path)
    server = build_server(
        host="127.0.0.1",
        port=0,
        config_path=tmp_path / "config.yaml",
        output_root=tmp_path / "runs",
        static_dir=tmp_path,
    )
    _make_run(tmp_path, "ui-http-del")
    _make_run(tmp_path, "ui-http-root")
    _make_run(tmp_path, "ui-http-child", parent="ui-http-root")
    _make_run(tmp_path, "ui-http-busy", status="running")
    server.manager._runs["ui-http-busy"] = {
        "run_id": "ui-http-busy",
        "status": "running",
    }
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)

        # 删除存在的运行 → 200 + deleted 列表，目录消失。
        conn.request("DELETE", "/api/runs/ui-http-del")
        response = conn.getresponse()
        assert response.status == 200
        assert json.loads(response.read())["deleted"] == ["ui-http-del"]
        assert not (tmp_path / "runs" / "ui-http-del").exists()

        # 删除链根 → 后代一并删除。
        conn.request("DELETE", "/api/runs/ui-http-root")
        response = conn.getresponse()
        assert response.status == 200
        assert set(json.loads(response.read())["deleted"]) == {
            "ui-http-root",
            "ui-http-child",
        }

        # 不存在 → 404。
        conn.request("DELETE", "/api/runs/ui-ghost")
        response = conn.getresponse()
        assert response.status == 404
        assert "不存在" in json.loads(response.read())["error"]

        # 非法 run_id → 400，且不删除任何目录。
        conn.request("DELETE", "/api/runs/..%2Fxxx")
        response = conn.getresponse()
        assert response.status == 400
        assert (tmp_path / "runs" / "ui-http-busy").exists()

        # 运行中 → 409 中文提示。
        conn.request("DELETE", "/api/runs/ui-http-busy")
        response = conn.getresponse()
        assert response.status == 409
        assert "正在执行中" in json.loads(response.read())["error"]
    finally:
        server.shutdown()
        server.server_close()


def test_http_endpoints_rounds_versions_and_followup_error(
    tmp_path: Path,
) -> None:
    _write_config(tmp_path)
    server = build_server(
        host="127.0.0.1",
        port=0,
        config_path=tmp_path / "config.yaml",
        output_root=tmp_path / "runs",
        static_dir=tmp_path,
    )
    _write_routing_fixture(tmp_path, "ui-http")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)

        conn.request("GET", "/api/runs/ui-http/rounds")
        response = conn.getresponse()
        assert response.status == 200
        rounds_payload = json.loads(response.read())
        # Contract shape at the HTTP layer too.
        assert set(rounds_payload) == {"routing_history", "search_round"}
        assert len(rounds_payload["routing_history"]) == 3
        assert rounds_payload["search_round"] == 1

        conn.request("GET", "/api/runs/ui-http/versions")
        response = conn.getresponse()
        assert response.status == 200
        assert len(json.loads(response.read())["versions"]) == 2

        conn.request(
            "POST",
            "/api/runs",
            body=json.dumps(
                {"question": "q", "parent_run_id": "ghost", "followup": "x"}
            ),
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        assert response.status == 400
        assert "不存在" in json.loads(response.read())["error"]

        # Followup submitted while a run is in progress → 409 (mutex, no queue).
        server.manager._runs["busy-run"] = {"run_id": "busy-run", "status": "running"}
        _make_parent_run(tmp_path)
        conn.request(
            "POST",
            "/api/runs",
            body=json.dumps(
                {
                    "question": "追问",
                    "parent_run_id": "ui-parent",
                    "followup": "细化对照组",
                }
            ),
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        assert response.status == 409
        assert "当前运行结束后才能提交" in json.loads(response.read())["error"]
    finally:
        server.shutdown()
        server.server_close()
