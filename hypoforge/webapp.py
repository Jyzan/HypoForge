"""Dependency-free local web application for observing M1-M6 runs."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
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
    """Load local Qwen credentials and map them to the OpenAI-compatible client.

    ``FENJIN_VISION_*`` remains a compatibility fallback for older team env
    files, while ``QWEN_*`` is the formal project-facing naming.
    """

    if env_file:
        values = dotenv_values(Path(env_file))
        for key, value in values.items():
            if value is not None:
                os.environ[str(key)] = str(value)
    if not os.environ.get("QWEN_API_KEY") and os.environ.get("FENJIN_VISION_API_KEY"):
        os.environ["QWEN_API_KEY"] = os.environ["FENJIN_VISION_API_KEY"]
    if not os.environ.get("QWEN_BASE_URL") and os.environ.get("FENJIN_VISION_BASE_URL"):
        os.environ["QWEN_BASE_URL"] = os.environ["FENJIN_VISION_BASE_URL"]
    if not os.environ.get("OPENAI_API_KEY") and os.environ.get("QWEN_API_KEY"):
        os.environ["OPENAI_API_KEY"] = os.environ["QWEN_API_KEY"]
    if not os.environ.get("OPENAI_BASE_URL") and os.environ.get("QWEN_BASE_URL"):
        os.environ["OPENAI_BASE_URL"] = os.environ["QWEN_BASE_URL"]


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
        self._run_followups: dict[str, dict[str, Any]] = {}
        # Pure-resume seeds: checkpoint JSON of a failed run, consumed once by
        # the worker.  Kept separate from followups because a resume carries
        # no followup semantics (no M1 triage, no fresh iteration budget).
        self._run_resumes: dict[str, dict[str, Any]] = {}
        # One cancellation flag per live run: set by ``request_cancel`` and
        # polled by the pipeline both during and between async modules.
        self._cancel_events: dict[str, threading.Event] = {}

    def start(
        self,
        question: str,
        *,
        model_name: str = "",
        qwen_api_key: str = "",
        semantic_scholar_api_key: str = "",
        openalex_api_key: str = "",
        openalex_mailto: str = "",
        parent_run_id: str = "",
        followup: str = "",
        resume_of: str = "",
    ) -> dict[str, Any]:
        question = " ".join(str(question or "").split())
        model_name = " ".join(str(model_name or "").split())
        qwen_api_key = str(qwen_api_key or "").strip()
        semantic_scholar_api_key = str(semantic_scholar_api_key or "").strip()
        openalex_api_key = str(openalex_api_key or "").strip()
        openalex_mailto = " ".join(str(openalex_mailto or "").split())
        parent_run_id = " ".join(str(parent_run_id or "").split())
        followup = " ".join(str(followup or "").split())
        resume_of = " ".join(str(resume_of or "").split())
        if resume_of and (parent_run_id or followup):
            raise ValueError("resume_of 不能与 parent_run_id/followup 同时提供")
        if not question:
            raise ValueError("问题不能为空")
        if len(question) > 4000:
            raise ValueError("问题过长，请控制在 4000 字以内")
        if len(model_name) > 128:
            raise ValueError("模型名称过长")
        if (
            len(qwen_api_key) > 4096
            or len(semantic_scholar_api_key) > 4096
            or len(openalex_api_key) > 4096
        ):
            raise ValueError("API Key 长度异常")
        if len(openalex_mailto) > 320:
            raise ValueError("OpenAlex Mailto 长度异常")
        if len(followup) > 4000:
            raise ValueError("追问内容过长，请控制在 4000 字以内")
        seed_state: dict[str, Any] | None = None
        if parent_run_id or followup:
            if not parent_run_id or not followup:
                raise ValueError("追问运行需要同时提供 parent_run_id 与 followup")
            seed_state = self._load_parent_state(parent_run_id)
        elif resume_of:
            # Pure resume: the parent's checkpoint is authoritative for the
            # question text, so the caller cannot drift from the failed run.
            seed_state = self._load_checkpoint_state(resume_of)
            question = str(seed_state.get("input_question") or question)
        with self._lock:
            if any(item.get("status") == "running" for item in self._runs.values()):
                raise RuntimeError("已有一条流程正在运行，当前运行结束后才能提交")
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
                "cancel_requested": False,
            }
            if parent_run_id:
                record["parent_run_id"] = parent_run_id
            if resume_of:
                record["resume_of"] = resume_of
            self._runs[run_id] = record
            self._cancel_events[run_id] = threading.Event()
            # Credentials are intentionally kept outside the public run record.
            # They are consumed once by the worker and never persisted.
            self._run_credentials[run_id] = {
                "model_name": model_name,
                "qwen_api_key": qwen_api_key,
                "semantic_scholar_api_key": semantic_scholar_api_key,
                "openalex_api_key": openalex_api_key,
                "openalex_mailto": openalex_mailto,
            }
            if seed_state is not None:
                # Memory-only, consumed once by the worker (like credentials).
                if resume_of:
                    self._run_resumes[run_id] = {
                        "resume_of": resume_of,
                        "seed_state": seed_state,
                    }
                else:
                    self._run_followups[run_id] = {
                        "parent_run_id": parent_run_id,
                        "followup": followup,
                        "seed_state": seed_state,
                    }
        thread = threading.Thread(
            target=self._run_pipeline,
            args=(run_id, question),
            name=f"hypoforge-{run_id}",
            daemon=True,
        )
        thread.start()
        return dict(record)

    def request_cancel(self, run_id: str) -> dict[str, Any]:
        """Request a cooperative stop for a running run.

        Sets the run's cancellation flag; the pipeline cancels active async
        module work, persists the last completed state, then ends.  Raises
        ``ValueError`` for malformed
        ids, ``LookupError`` for unknown runs, and ``RuntimeError`` when the
        run has already finished (cancel is idempotent while cancelling).
        """

        if not re.fullmatch(r"[A-Za-z0-9._-]+", run_id):
            raise ValueError("run_id 含非法字符")
        with self._lock:
            record = self._runs.get(run_id)
            if record is not None and record.get("status") != "running":
                raise RuntimeError(
                    f"运行 {run_id} 已结束（状态：{record.get('status')}），"
                    "无法停止"
                )
            if record is None:
                # Not live in this process — check the persisted manifest so
                # already-finished runs report "已结束" instead of 404.
                manifest = self.get(run_id)
                if manifest is not None:
                    raise RuntimeError(
                        f"运行 {run_id} 已结束"
                        f"（状态：{manifest.get('status', '')}），无法停止"
                    )
                raise LookupError(f"运行 {run_id} 不存在")
            already = bool(record.get("cancel_requested"))
            record["cancel_requested"] = True
            cancel_event = self._cancel_events.get(run_id)
        if cancel_event is not None:
            cancel_event.set()
        if not already:
            RunEventRecorder(self.output_root / run_id, run_id).emit(
                "cancel_requested",
                status="cancelling",
                message=(
                    "收到停止请求：正在取消当前异步任务，"
                    "已完成模块的成果将保留"
                ),
            )
        return {"run_id": run_id, "status": "cancelling"}

    def _load_parent_state(self, parent_run_id: str) -> dict[str, Any]:
        """Load and validate a parent run's final state for a follow-up run."""

        if not re.fullmatch(r"[A-Za-z0-9._-]+", parent_run_id):
            raise ValueError("parent_run_id 含非法字符")
        state_path = self.output_root / parent_run_id / f"{parent_run_id}.json"
        if not state_path.exists():
            raise ValueError(
                f"父运行 {parent_run_id} 的最终 state 不存在，无法发起追问"
            )
        try:
            data = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"父运行 {parent_run_id} 的 state 文件损坏，无法发起追问：{exc}"
            ) from exc
        if not isinstance(data, dict):
            raise ValueError(f"父运行 {parent_run_id} 的 state 格式无效")
        return data

    def _load_checkpoint_state(self, run_id: str) -> dict[str, Any]:
        """Load and validate a run's checkpoint for a pure resume."""

        if not re.fullmatch(r"[A-Za-z0-9._-]+", run_id):
            raise ValueError("resume_of 含非法字符")
        record = self._runs.get(run_id)
        if record and record.get("status") == "running":
            raise ValueError("父运行仍在运行中，无法从断点重试")
        ckpt_path = self.output_root / run_id / f"{run_id}_checkpoint.json"
        if not ckpt_path.exists():
            raise ValueError(
                f"父运行 {run_id} 没有可用的断点（checkpoint）——"
                "运行可能在第一个模块完成前失败，请重新发起运行"
            )
        try:
            data = json.loads(ckpt_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"父运行 {run_id} 的断点文件损坏，无法续传：{exc}"
            ) from exc
        if not isinstance(data, dict):
            raise ValueError(f"父运行 {run_id} 的断点格式无效")
        return data

    def _read_final_state(self, run_id: str) -> dict[str, Any] | None:
        state_path = self.output_root / run_id / f"{run_id}.json"
        if not state_path.exists():
            return None
        try:
            data = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return data if isinstance(data, dict) else None

    def _run_pipeline(self, run_id: str, question: str) -> None:
        run_dir = self.output_root / run_id
        recorder = RunEventRecorder(run_dir, run_id)
        with self._lock:
            credentials = self._run_credentials.pop(run_id, {})
            followup_info = self._run_followups.pop(run_id, {})
            resume_info = self._run_resumes.pop(run_id, {})
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
            openalex_api_key = credentials.get("openalex_api_key", "")
            openalex_mailto = credentials.get("openalex_mailto", "")
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
            if openalex_api_key or openalex_mailto:
                m2_override = config.module_overrides.setdefault(
                    "m2", ModuleOverride()
                )
                if openalex_api_key:
                    m2_override.kwargs["openalex_api_key"] = openalex_api_key
                if openalex_mailto:
                    m2_override.kwargs["openalex_mailto"] = openalex_mailto
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
                "credential_status": {
                    "llm_configured": bool(config.qwen.base.api_key),
                    "semantic_scholar_configured": bool(semantic_scholar_api_key),
                    "openalex_configured": bool(os.environ.get("OPENALEX_API_KEY")),
                },
                "output_dir": str(run_dir),
            }
            if followup_info:
                manifest["parent_run_id"] = followup_info.get("parent_run_id", "")
                manifest["followup"] = followup_info.get("followup", "")
            if resume_info:
                manifest["resume_of"] = resume_info.get("resume_of", "")
            recorder.write_manifest(manifest)
            run_kwargs: dict[str, Any] = {}
            if followup_info:
                run_kwargs = {
                    "followup_text": followup_info.get("followup", ""),
                    "seed_state": followup_info.get("seed_state"),
                }
            elif resume_info:
                # Pure resume: checkpoint seed without followup_text, so the
                # pipeline skips completed modules and re-runs the failed one.
                run_kwargs = {"seed_state": resume_info.get("seed_state")}
            runner = PipelineRunner(config, event_recorder=recorder)
            # Active cancellation flag (set by request_cancel).
            runner.cancel_event = self._cancel_events.get(run_id)
            state = asyncio.run(
                runner.run(
                    question=question,
                    run_id=run_id,
                    **run_kwargs,
                )
            )
            cancelled = bool(getattr(runner, "cancelled", False))
            if cancelled:
                status = "cancelled"
            else:
                status = (
                    "completed" if not state.errors else "completed_with_errors"
                )
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
            if cancelled:
                manifest["cancel_message"] = (
                    "用户手动停止：已完成模块的成果已保留，"
                    "可作为后续追问的父运行"
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
            failed_manifest = {
                "run_id": run_id,
                "question": question,
                "status": "failed",
                "started_at": self._runs[run_id]["started_at"],
                "completed_at": completed_at,
                "error": error,
                "config_path": str(self.config_path),
                "output_dir": str(run_dir),
            }
            if followup_info:
                failed_manifest["parent_run_id"] = followup_info.get(
                    "parent_run_id", ""
                )
            recorder.write_manifest(failed_manifest)

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

    def rename(self, run_id: str, display_name: str) -> dict[str, Any]:
        """Set a display name for a run (sidebar only; question untouched)."""

        if not re.fullmatch(r"[A-Za-z0-9._-]+", run_id):
            raise ValueError("run_id 含非法字符")
        display_name = " ".join(str(display_name or "").split())
        if not display_name:
            raise ValueError("名称不能为空")
        if len(display_name) > 200:
            raise ValueError("名称过长，请控制在 200 字以内")
        manifest_path = self.output_root / run_id / "manifest.json"
        if not manifest_path.exists():
            raise LookupError(f"运行 {run_id} 不存在")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"运行 {run_id} 的 manifest 损坏，无法重命名：{exc}") from exc
        manifest["display_name"] = display_name
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        with self._lock:
            if run_id in self._runs:
                self._runs[run_id]["display_name"] = display_name
        return manifest

    def delete(self, run_id: str) -> list[str]:
        """Delete a run directory together with its whole followup chain.

        Raises ``ValueError`` for malformed ids, ``LookupError`` when the run
        directory does not exist, and ``RuntimeError`` while any run in the
        affected chain is still executing.  Removal is confined to
        ``output_root``: every resolved path is verified to live inside it
        before ``shutil.rmtree`` is called.
        """

        if not re.fullmatch(r"ui-[A-Za-z0-9._-]+", run_id):
            raise ValueError("run_id 非法，仅允许 ui- 开头的运行目录名")
        run_dir = (self.output_root / run_id).resolve()
        try:
            run_dir.relative_to(self.output_root)
        except ValueError as exc:
            raise ValueError("run_id 指向的路径不在运行目录内") from exc
        if not run_dir.is_dir():
            raise LookupError(f"运行 {run_id} 不存在")

        # parent → children map built from persisted manifests plus the
        # in-memory records of live runs.
        children: dict[str, set[str]] = {}
        for manifest_path in self.output_root.glob("*/manifest.json"):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            rid = str(manifest.get("run_id") or "")
            parent = str(manifest.get("parent_run_id") or "")
            if rid and parent:
                children.setdefault(parent, set()).add(rid)
        with self._lock:
            running = {
                rid
                for rid, item in self._runs.items()
                if item.get("status") == "running"
            }
            for rid, item in self._runs.items():
                parent = str(item.get("parent_run_id") or "")
                if parent:
                    children.setdefault(parent, set()).add(str(rid))

        # Cascade: collect the full descendant chain (cycle-safe BFS).
        targets: list[str] = [run_id]
        seen = {run_id}
        stack = [run_id]
        while stack:
            current = stack.pop()
            for child in children.get(current, ()):
                if child not in seen:
                    seen.add(child)
                    targets.append(child)
                    stack.append(child)

        busy = running & seen
        if busy:
            raise RuntimeError(
                f"运行 {'、'.join(sorted(busy))} 正在执行中，"
                "请等待其结束后再删除"
            )

        deleted: list[str] = []
        for rid in targets:
            target = (self.output_root / rid).resolve()
            try:
                target.relative_to(self.output_root)
            except ValueError:
                continue
            if target.is_dir():
                shutil.rmtree(target)
            deleted.append(rid)
        with self._lock:
            for rid in deleted:
                self._runs.pop(rid, None)
                self._run_credentials.pop(rid, None)
                self._run_followups.pop(rid, None)
                self._cancel_events.pop(rid, None)
        return deleted

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

    @staticmethod
    def _public_state_payload(
        run_id: str,
        state: dict[str, Any],
        *,
        scores: dict[str, Any] | None = None,
        is_final: bool,
        snapshot_source: str = "",
    ) -> dict[str, Any]:
        """Return the browser-safe subset of a persisted pipeline state."""

        return {
            "run_id": run_id,
            "question": state.get("input_question", ""),
            "problem_card": state.get("problem_card"),
            "literature_results": state.get("literature_results", []),
            "m2_knowledge_export": state.get("m2_knowledge_export"),
            "evidence_graph": state.get("evidence_graph"),
            "grounding_report": state.get("grounding_report"),
            "candidate_hypotheses": state.get("candidate_hypotheses", []),
            "top_hypotheses": state.get("top_hypotheses", []),
            "best_hypotheses": state.get("best_hypotheses", []),
            "evidence_gap_requests": state.get("evidence_gap_requests", []),
            "graph_correction_requests": state.get("graph_correction_requests", []),
            "research_plans": state.get("research_plans", []),
            "research_plan_history": state.get("research_plan_history", {}),
            "reviews": state.get("reviews", []),
            "iteration_count": state.get("iteration_count", 0),
            "routing_history": state.get("routing_history", []),
            "evidence_gaps": state.get("evidence_gaps", []),
            "search_round": state.get("search_round", 0),
            "revision_count": state.get("revision_count", 0),
            "parent_run_id": state.get("parent_run_id", ""),
            "errors": state.get("errors", []),
            "metrics": state.get("metrics", {}),
            "token_usage": {
                "input": state.get("total_input_tokens", 0),
                "output": state.get("total_output_tokens", 0),
            },
            "scores": scores,
            "last_module": state.get("_last_module", ""),
            "is_final": is_final,
            "snapshot_source": snapshot_source,
        }

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
        return self._public_state_payload(
            run_id,
            state,
            scores=scores,
            is_final=True,
            snapshot_source=state_path.name,
        )

    def latest_state(self, run_id: str) -> dict[str, Any] | None:
        """Return the newest readable state for final, failed, or live runs.

        The final state remains authoritative. Before it exists, the atomic
        checkpoint is preferred, followed by the latest completed-module
        snapshot. This preserves inspectability when a later module fails.
        """

        final = self.result(run_id)
        if final is not None:
            return final

        run_dir = self.output_root / run_id
        candidates = [run_dir / f"{run_id}_checkpoint.json"]
        snapshots_dir = run_dir / "snapshots"
        if snapshots_dir.is_dir():
            candidates.extend(sorted(snapshots_dir.glob("*.json"), reverse=True))

        for path in candidates:
            if not path.is_file():
                continue
            try:
                state = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            return self._public_state_payload(
                run_id,
                state,
                is_final=False,
                snapshot_source=str(path.relative_to(run_dir)),
            )
        return None

    def rounds(self, run_id: str) -> dict[str, Any]:
        """Round timeline with a stable contract shape (read from state.json):

        ``{"routing_history": [...], "search_round": n}``

        ``routing_history`` is the auditable list of RoutingDecision records
        written by the M1/M6 node wrappers; ``search_round`` is the number of
        M2 executions.  Missing/corrupt runs yield the empty default shape.
        """

        state = self._read_final_state(run_id) or {}
        routing_history = state.get("routing_history")
        if not isinstance(routing_history, list):
            routing_history = []
        search_round = state.get("search_round", 0)
        if not isinstance(search_round, int):
            try:
                search_round = int(search_round)
            except (TypeError, ValueError):
                search_round = 0
        return {"routing_history": routing_history, "search_round": search_round}

    def versions(self, run_id: str) -> list[dict[str, Any]]:
        """Summarise each review version (vN) — stable contract shape:

        ``[{"version", "overall", "timestamp", ...}, ...]``

        ``timestamp`` is the event-stream time at which the M6 module of that
        review version completed (``None`` when the event log is missing).
        Per-round hypothesis/plan counts come from the snapshots
        (``snapshots/*-{module}-r{R}-iter{N}.json``); snapshot label
        ``iter{N}`` records the pre-execution iteration count, and M6 stamps
        reviews with ``iteration_count + 1``, so a snapshot belongs to version
        ``N + 1``.
        """

        run_dir = self.output_root / run_id
        if not run_dir.is_dir():
            return []

        def new_entry(version: int) -> dict[str, Any]:
            return {
                "version": version,
                "overall": None,
                "timestamp": None,
                "reviews": [],
                "top_hypothesis_titles": [],
                "top_hypotheses_count": 0,
                "research_plans_count": 0,
                "research_plans": [],
                "snapshots": [],
            }

        entries: dict[int, dict[str, Any]] = {}
        state = self._read_final_state(run_id)
        for review in (state or {}).get("reviews") or []:
            if not isinstance(review, dict):
                continue
            try:
                version = int(review.get("version") or 1)
            except (TypeError, ValueError):
                continue
            entry = entries.setdefault(version, new_entry(version))
            entry["reviews"].append(
                {
                    "dimension": review.get("dimension"),
                    "score": review.get("score"),
                    "reasoning": review.get("reasoning", ""),
                    "comments": review.get("comments", ""),
                        "suggestions": review.get("suggestions", ""),
                        "evidence_ids": review.get("evidence_ids", []),
                        "hard_gate_passed": review.get("hard_gate_passed"),
                        "version": version,
                }
            )
            if review.get("dimension") == "overall":
                entry["overall"] = review.get("score")

        # Per-version completion timestamps from the event stream (best
        # effort): the M6 ``module_completed`` event carries the review
        # version as ``iteration_count`` in its details.
        timestamp_by_version: dict[Any, str] = {}
        for event in self.events(run_id):
            if (
                event.get("event_type") == "module_completed"
                and event.get("module") == "m6"
            ):
                version = (event.get("details") or {}).get("iteration_count")
                if version is not None and event.get("timestamp"):
                    timestamp_by_version[version] = event["timestamp"]
        for version, entry in entries.items():
            entry["timestamp"] = timestamp_by_version.get(version)

        # Per-round hypothesis/plan counts from snapshots (best effort).
        snapshot_pattern = re.compile(r"-m\d-r(\d+)-iter(\d+)\.json$")
        snapshots_dir = run_dir / "snapshots"
        if snapshots_dir.is_dir():
            for path in sorted(snapshots_dir.glob("*.json")):
                match = snapshot_pattern.search(path.name)
                if not match:
                    continue
                version = int(match.group(2)) + 1
                entry = entries.setdefault(version, new_entry(version))
                entry["snapshots"].append(
                    str(path.relative_to(run_dir)).replace(os.sep, "/")
                )
                try:
                    snapshot = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if not isinstance(snapshot, dict):
                    continue
                hypotheses = snapshot.get("top_hypotheses")
                if isinstance(hypotheses, list) and hypotheses:
                    entry["top_hypotheses_count"] = len(hypotheses)
                    entry["top_hypothesis_titles"] = [
                        str(
                            item.get("statement")
                            or item.get("title")
                            or item.get("hypothesis_id")
                            or ""
                        )
                        for item in hypotheses
                        if isinstance(item, dict)
                    ][:10]
                plans = snapshot.get("research_plans")
                if isinstance(plans, list) and plans:
                    entry["research_plans_count"] = len(plans)
                    entry["research_plans"] = plans

        # Final state is authoritative for the latest version.
        if state:
            version = max(entries) if entries else 1
            entry = entries.setdefault(version, new_entry(version))
            hypotheses = state.get("top_hypotheses") or state.get(
                "best_hypotheses"
            ) or []
            entry["top_hypotheses_count"] = len(hypotheses)
            entry["top_hypothesis_titles"] = [
                str(
                    item.get("statement")
                    or item.get("title")
                    or item.get("hypothesis_id")
                    or ""
                )
                for item in hypotheses
                if isinstance(item, dict)
            ][:10]
            entry["research_plans_count"] = len(state.get("research_plans") or [])
            entry["research_plans"] = state.get("research_plans") or []

        return [entries[version] for version in sorted(entries)]


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
            if action == "state":
                state = self.server.manager.latest_state(run_id)
                self._json(
                    {"state": state},
                    HTTPStatus.OK if state is not None else HTTPStatus.ACCEPTED,
                )
                return
            if action == "artifacts":
                self._json({"artifacts": self.server.manager.artifacts(run_id)})
                return
            if action == "rounds":
                # Contract shape: {"routing_history": [...], "search_round": n}
                self._json(self.server.manager.rounds(run_id))
                return
            if action == "versions":
                self._json({"versions": self.server.manager.versions(run_id)})
                return
            if not action:
                run = self.server.manager.get(run_id)
                self._json(
                    {"run": run},
                    HTTPStatus.OK if run is not None else HTTPStatus.NOT_FOUND,
                )
                return
        self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_DELETE(self) -> None:
        route = self._run_route(urlparse(self.path).path)
        if not route or route[1]:
            self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            return
        try:
            deleted = self.server.manager.delete(route[0])
        except ValueError as exc:
            self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        except LookupError as exc:
            self._json({"error": str(exc)}, HTTPStatus.NOT_FOUND)
            return
        except RuntimeError as exc:
            self._json({"error": str(exc)}, HTTPStatus.CONFLICT)
            return
        self._json({"deleted": deleted})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        route = self._run_route(parsed.path)
        if route is not None and route[1] == "cancel":
            try:
                info = self.server.manager.request_cancel(route[0])
            except ValueError as exc:
                self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            except LookupError as exc:
                self._json({"error": str(exc)}, HTTPStatus.NOT_FOUND)
                return
            except RuntimeError as exc:
                self._json({"error": str(exc)}, HTTPStatus.CONFLICT)
                return
            self._json(info, HTTPStatus.ACCEPTED)
            return
        if route is not None and route[1] == "rename":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length > 4096:
                    raise ValueError("请求内容过大")
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                info = self.server.manager.rename(
                    route[0], payload.get("name", "")
                )
            except ValueError as exc:
                self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            except LookupError as exc:
                self._json({"error": str(exc)}, HTTPStatus.NOT_FOUND)
                return
            except (json.JSONDecodeError, UnicodeDecodeError):
                self._json({"error": "请求 JSON 无效"}, HTTPStatus.BAD_REQUEST)
                return
            self._json({"run": info}, HTTPStatus.OK)
            return
        if parsed.path != "/api/runs":
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
                openalex_api_key=payload.get("openalex_api_key", ""),
                openalex_mailto=payload.get("openalex_mailto", ""),
                parent_run_id=payload.get("parent_run_id", ""),
                followup=payload.get("followup", ""),
                resume_of=payload.get("resume_of", ""),
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
