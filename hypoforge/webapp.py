"""Dependency-free local web application for observing M1-M6 runs."""

from __future__ import annotations

import asyncio
import json
import os
import threading
import traceback
import uuid
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from dotenv import dotenv_values

from .config import ModuleOverride, PipelineConfig
from .observability import RunEventRecorder
from .pipeline import PipelineRunner


def load_runtime_environment(env_file: str | Path | None = None) -> None:
    """Load a local env file and map the team's FENJIN names to OpenAI names."""

    if env_file:
        values = dotenv_values(Path(env_file))
        for key, value in values.items():
            if value is not None:
                os.environ[str(key)] = str(value)
    if not os.environ.get("OPENAI_API_KEY") and os.environ.get(
        "FENJIN_VISION_API_KEY"
    ):
        os.environ["OPENAI_API_KEY"] = os.environ["FENJIN_VISION_API_KEY"]
    if not os.environ.get("OPENAI_BASE_URL") and os.environ.get(
        "FENJIN_VISION_BASE_URL"
    ):
        os.environ["OPENAI_BASE_URL"] = os.environ["FENJIN_VISION_BASE_URL"]


class RunManager:
    """Start pipeline runs in background threads and expose persisted results."""

    def __init__(
        self,
        *,
        config_path: str | Path,
        output_root: str | Path,
    ) -> None:
        self.config_path = Path(config_path).resolve()
        self.output_root = Path(output_root).resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._runs: dict[str, dict[str, Any]] = {}
        self._run_credentials: dict[str, dict[str, str]] = {}

    def start(
        self,
        question: str,
        *,
        model_name: str = "",
        qwen_api_key: str = "",
        semantic_scholar_api_key: str = "",
    ) -> dict[str, Any]:
        question = " ".join(str(question or "").split())
        model_name = " ".join(str(model_name or "").split())
        qwen_api_key = str(qwen_api_key or "").strip()
        semantic_scholar_api_key = str(semantic_scholar_api_key or "").strip()
        if not question:
            raise ValueError("问题不能为空")
        if len(question) > 4000:
            raise ValueError("问题过长，请控制在 4000 字以内")
        if len(model_name) > 128:
            raise ValueError("模型名称过长")
        if len(qwen_api_key) > 4096 or len(semantic_scholar_api_key) > 4096:
            raise ValueError("API Key 长度异常")
        with self._lock:
            if any(item.get("status") == "running" for item in self._runs.values()):
                raise RuntimeError("已有一条流程正在运行，请等待完成后再提交")
            run_id = (
                "ui-"
                + datetime.now().strftime("%Y%m%d-%H%M%S")
                + "-"
                + uuid.uuid4().hex[:6]
            )
            run_dir = self.output_root / run_id
            record = {
                "run_id": run_id,
                "question": question,
                "status": "running",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "completed_at": None,
                "run_dir": str(run_dir),
                "error": "",
            }
            self._runs[run_id] = record
            # Credentials are intentionally kept outside the public run record.
            # They are consumed once by the worker and never persisted.
            self._run_credentials[run_id] = {
                "model_name": model_name,
                "qwen_api_key": qwen_api_key,
                "semantic_scholar_api_key": semantic_scholar_api_key,
            }
        thread = threading.Thread(
            target=self._run_pipeline,
            args=(run_id, question),
            name=f"hypoforge-{run_id}",
            daemon=True,
        )
        thread.start()
        return dict(record)

    def _run_pipeline(self, run_id: str, question: str) -> None:
        run_dir = self.output_root / run_id
        recorder = RunEventRecorder(run_dir, run_id)
        with self._lock:
            credentials = self._run_credentials.pop(run_id, {})
        try:
            config = PipelineConfig.from_yaml(self.config_path)
            config.output_dir = str(run_dir)
            config.verbose = False
            config.interactive = False
            model_name = credentials.get("model_name", "")
            qwen_api_key = credentials.get("qwen_api_key", "")
            semantic_scholar_api_key = credentials.get(
                "semantic_scholar_api_key", ""
            )
            for tier in (
                config.qwen.base,
                config.qwen.max,
                config.qwen.plus,
                config.qwen.turbo,
            ):
                if model_name:
                    tier.model = model_name
                if qwen_api_key:
                    tier.api_key = qwen_api_key
            if semantic_scholar_api_key:
                m2_override = config.module_overrides.setdefault(
                    "m2", ModuleOverride()
                )
                m2_override.kwargs["semantic_scholar_api_key"] = (
                    semantic_scholar_api_key
                )
            manifest = {
                "run_id": run_id,
                "question": question,
                "status": "running",
                "started_at": self._runs[run_id]["started_at"],
                "config_path": str(self.config_path),
                "models": {
                    "base": config.qwen.base.model,
                    "max": config.qwen.max.model,
                    "plus": config.qwen.plus.model,
                    "turbo": config.qwen.turbo.model,
                },
                "enabled_modules": list(config.enabled_modules),
                "output_dir": str(run_dir),
            }
            recorder.write_manifest(manifest)
            state = asyncio.run(
                PipelineRunner(config, event_recorder=recorder).run(
                    question=question,
                    run_id=run_id,
                )
            )
            status = "completed" if not state.errors else "completed_with_errors"
            completed_at = datetime.now(timezone.utc).isoformat()
            with self._lock:
                self._runs[run_id].update(
                    {"status": status, "completed_at": completed_at}
                )
            manifest.update(
                {
                    "status": status,
                    "completed_at": completed_at,
                    "errors": len(state.errors),
                    "final_state": str(run_dir / f"{run_id}.json"),
                    "scores": str(run_dir / f"{run_id}_scores.json"),
                }
            )
            recorder.write_manifest(manifest)
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
            completed_at = datetime.now(timezone.utc).isoformat()
            recorder.emit(
                "run_failed",
                status="failed",
                message=f"流程异常终止：{error}",
                details={"traceback": traceback.format_exc()},
            )
            with self._lock:
                self._runs[run_id].update(
                    {
                        "status": "failed",
                        "completed_at": completed_at,
                        "error": error,
                    }
                )
            recorder.write_manifest(
                {
                    "run_id": run_id,
                    "question": question,
                    "status": "failed",
                    "started_at": self._runs[run_id]["started_at"],
                    "completed_at": completed_at,
                    "error": error,
                    "config_path": str(self.config_path),
                    "output_dir": str(run_dir),
                }
            )

    def get(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            if run_id in self._runs:
                return dict(self._runs[run_id])
        manifest_path = self.output_root / run_id / "manifest.json"
        if not manifest_path.exists():
            return None
        try:
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    def list_runs(self) -> list[dict[str, Any]]:
        records: dict[str, dict[str, Any]] = {}
        for manifest_path in self.output_root.glob("*/manifest.json"):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if manifest.get("run_id"):
                records[str(manifest["run_id"])] = manifest
        with self._lock:
            records.update({key: dict(value) for key, value in self._runs.items()})
        return sorted(
            records.values(),
            key=lambda item: str(item.get("started_at", "")),
            reverse=True,
        )

    def events(self, run_id: str, after: int = 0) -> list[dict[str, Any]]:
        run_dir = self.output_root / run_id
        if not run_dir.is_dir():
            return []
        return RunEventRecorder(run_dir, run_id).read_events(after=after)

    def artifacts(self, run_id: str) -> list[dict[str, Any]]:
        run_dir = self.output_root / run_id
        if not run_dir.is_dir():
            return []
        return [
            {
                "name": path.name,
                "relative_path": str(path.relative_to(run_dir)),
                "size_bytes": path.stat().st_size,
            }
            for path in sorted(run_dir.rglob("*"))
            if path.is_file()
        ]

    def result(self, run_id: str) -> dict[str, Any] | None:
        run_dir = self.output_root / run_id
        state_path = run_dir / f"{run_id}.json"
        if not state_path.exists():
            return None
        state = json.loads(state_path.read_text(encoding="utf-8"))
        scores_path = run_dir / f"{run_id}_scores.json"
        scores = (
            json.loads(scores_path.read_text(encoding="utf-8"))
            if scores_path.exists()
            else None
        )
        return {
            "run_id": run_id,
            "question": state.get("input_question", ""),
            "problem_card": state.get("problem_card"),
            "literature_results": state.get("literature_results", []),
            "evidence_graph": state.get("evidence_graph"),
            "top_hypotheses": state.get("top_hypotheses", []),
            "best_hypotheses": state.get("best_hypotheses", []),
            "research_plans": state.get("research_plans", []),
            "reviews": state.get("reviews", []),
            "iteration_count": state.get("iteration_count", 0),
            "errors": state.get("errors", []),
            "metrics": state.get("metrics", {}),
            "token_usage": {
                "input": state.get("total_input_tokens", 0),
                "output": state.get("total_output_tokens", 0),
            },
            "scores": scores,
        }


class HypoForgeHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        manager: RunManager,
        static_dir: str | Path,
    ) -> None:
        self.manager = manager
        self.static_dir = Path(static_dir).resolve()
        super().__init__(server_address, HypoForgeRequestHandler)


class HypoForgeRequestHandler(BaseHTTPRequestHandler):
    server: HypoForgeHTTPServer

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _json(self, payload: Any, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _serve_index(self) -> None:
        path = self.server.static_dir / "index.html"
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    @staticmethod
    def _run_route(path: str) -> tuple[str, str] | None:
        parts = [part for part in path.split("/") if part]
        if len(parts) >= 3 and parts[0:2] == ["api", "runs"]:
            return parts[2], parts[3] if len(parts) > 3 else ""
        return None

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._serve_index()
            return
        if parsed.path == "/api/runs":
            self._json({"runs": self.server.manager.list_runs()})
            return
        route = self._run_route(parsed.path)
        if route:
            run_id, action = route
            if action == "events":
                try:
                    after = int(parse_qs(parsed.query).get("after", ["0"])[0])
                except ValueError:
                    after = 0
                self._json(
                    {
                        "events": self.server.manager.events(run_id, after),
                        "run": self.server.manager.get(run_id),
                    }
                )
                return
            if action == "result":
                result = self.server.manager.result(run_id)
                self._json(
                    {"result": result},
                    HTTPStatus.OK if result is not None else HTTPStatus.ACCEPTED,
                )
                return
            if action == "artifacts":
                self._json({"artifacts": self.server.manager.artifacts(run_id)})
                return
            if not action:
                run = self.server.manager.get(run_id)
                self._json(
                    {"run": run},
                    HTTPStatus.OK if run is not None else HTTPStatus.NOT_FOUND,
                )
                return
        self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/runs":
            self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 16_384:
                raise ValueError("请求内容过大")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            run = self.server.manager.start(
                payload.get("question", ""),
                model_name=payload.get("model_name", ""),
                qwen_api_key=payload.get("qwen_api_key", ""),
                semantic_scholar_api_key=payload.get(
                    "semantic_scholar_api_key", ""
                ),
            )
        except ValueError as exc:
            self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        except RuntimeError as exc:
            self._json({"error": str(exc)}, HTTPStatus.CONFLICT)
            return
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._json({"error": "请求 JSON 无效"}, HTTPStatus.BAD_REQUEST)
            return
        self._json({"run": run}, HTTPStatus.ACCEPTED)


def build_server(
    *,
    host: str,
    port: int,
    config_path: str | Path,
    output_root: str | Path,
    static_dir: str | Path,
) -> HypoForgeHTTPServer:
    return HypoForgeHTTPServer(
        (host, port),
        RunManager(config_path=config_path, output_root=output_root),
        static_dir,
    )
