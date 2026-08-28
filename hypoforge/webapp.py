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
from .modules.m2_literature.search.search_tool import literature_credential_warnings
from .observability import RunEventRecorder
from .pipeline import PipelineRunner


_LEGACY_M6_METRIC_MAP = {
    "objective_evidence_consistency": "evidence_consistency",
    "novelty_metric": "novelty",
    "testability_metric": "testability",
}


def _is_legacy_m6_metric_review(review: dict[str, Any]) -> bool:
    """Identify the old live proxies that ran before posthoc scoring.

    Current M6 uses ``objective_evidence_consistency`` for a real specialist
    review, so that dimension is legacy only when it carries the old generated
    ``Calculated ... score`` rationale.  Novelty/testability metric reviews
    were only ever posthoc proxies in the live review list.
    """

    dimension = str(review.get("dimension") or "")
    if dimension in {"novelty_metric", "testability_metric"}:
        return True
    if dimension != "objective_evidence_consistency":
        return False
    reasoning = str(review.get("reasoning") or "").casefold()
    return "calculated evidence_consistency score" in reasoning


def _posthoc_independent_scores(scores: dict[str, Any] | None) -> dict[str, float]:
    if not isinstance(scores, dict):
        return {}
    rows = scores.get("hypothesis_scores")
    if not isinstance(rows, list) or not rows or not isinstance(rows[0], dict):
        return {}
    independent = rows[0].get("independent")
    if not isinstance(independent, dict):
        return {}
    output: dict[str, float] = {}
    for name in _LEGACY_M6_METRIC_MAP.values():
        value = independent.get(name)
        if isinstance(value, (int, float)):
            output[name] = min(1.0, max(0.0, float(value)))
    return output


def _reconcile_legacy_m6_reviews(
    reviews: Any,
    scores: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return browser-facing M6 reviews without treating missing scores as 0.

    This is deliberately presentation-only: persisted pipeline state and the
    routing decisions already taken by that historical run remain untouched.
    """

    copied = [dict(item) for item in reviews if isinstance(item, dict)] if isinstance(reviews, list) else []
    legacy = [item for item in copied if _is_legacy_m6_metric_review(item)]
    if not legacy:
        return copied, {"status": "not_applicable", "dimensions": []}

    independent = _posthoc_independent_scores(scores)
    reconciled_dimensions: list[str] = []
    affected_versions: set[int] = set()
    final_available = bool(independent)

    for review in legacy:
        dimension = str(review.get("dimension") or "")
        metric_name = _LEGACY_M6_METRIC_MAP[dimension]
        version = int(review.get("version") or 1)
        affected_versions.add(version)
        review["superseded_live_score"] = review.get("score")
        if metric_name not in independent:
            review.update({
                "score": None,
                "hard_gate_passed": None,
                "score_source": "posthoc_independent",
                "score_status": "awaiting_posthoc",
                "comments": (
                    "最终独立评分尚未完成；旧版实时阶段写入的默认 0 分不计入硬门。"
                ),
                "reasoning": (
                    f"Final independent {metric_name} score is unavailable; "
                    "the historical live default is not treated as a real zero."
                ),
            })
            continue

        score = round(independent[metric_name] * 5.0, 1)
        review.update({
            "score": score,
            "hard_gate_passed": score >= 3.0,
            "score_source": "posthoc_independent",
            "score_status": "final",
                "comments": "最终独立评分已替代旧版实时阶段的临时分数。",
            "reasoning": (
                f"Final post-pipeline independent {metric_name} score is "
                f"{independent[metric_name]:.2f} (scaled to {score:.1f}/5); "
                "it supersedes the historical live default."
            ),
        })
        reconciled_dimensions.append(dimension)

    for version in affected_versions:
        version_reviews = [
            item for item in copied if int(item.get("version") or 1) == version
        ]
        overall = next(
            (item for item in version_reviews if item.get("dimension") == "overall"),
            None,
        )
        if overall is None:
            continue
        overall["superseded_live_score"] = overall.get("score")
        overall["score_source"] = "reconciled_m6_display"
        if not final_available:
            overall["hard_gate_passed"] = None
            overall["score_status"] = "awaiting_posthoc"
            continue

        failed_gates = [
            str(item.get("dimension"))
            for item in version_reviews
            if item.get("dimension") != "overall"
            and item.get("hard_gate_passed") is False
        ]
        core_scores = [
            float(item["score"])
            for item in version_reviews
            if item.get("dimension") in {
                "scientific_logic",
                "objective_evidence_consistency",
                "method_feasibility",
            }
            and isinstance(item.get("score"), (int, float))
        ]
        overall_score = sum(core_scores) / len(core_scores) if core_scores else 3.0
        if "task_alignment" in failed_gates:
            overall_score = min(overall_score, 1.9)
        elif failed_gates:
            overall_score = min(overall_score, 2.9)
        overall.update({
            "score": round(overall_score, 1),
            "hard_gate_passed": not failed_gates,
            "score_status": "final",
            "reasoning": (
                "Display score reconciled with final independent metrics; "
                "hard-gate failures: "
                + (", ".join(failed_gates) if failed_gates else "none")
                + ". Historical routing is unchanged."
            ),
        })

    status = "final" if final_available else "pending"
    return copied, {
        "status": status,
        "dimensions": list(dict.fromkeys(reconciled_dimensions)),
        "historical_routing_unchanged": True,
    }


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
        serper_api_key: str = "",
        ads_api_token: str = "",
        unpaywall_email: str = "",
        crossref_mailto: str = "",
        parent_run_id: str = "",
        followup: str = "",
        resume_of: str = "",
        run_mode: str = "standard",
    ) -> dict[str, Any]:
        question = " ".join(str(question or "").split())
        model_name = " ".join(str(model_name or "").split())
        qwen_api_key = str(qwen_api_key or "").strip()
        semantic_scholar_api_key = str(semantic_scholar_api_key or "").strip()
        openalex_api_key = str(openalex_api_key or "").strip()
        openalex_mailto = " ".join(str(openalex_mailto or "").split())
        serper_api_key = str(serper_api_key or "").strip()
        ads_api_token = str(ads_api_token or "").strip()
        unpaywall_email = " ".join(str(unpaywall_email or "").split())
        crossref_mailto = " ".join(str(crossref_mailto or "").split())
        parent_run_id = " ".join(str(parent_run_id or "").split())
        followup = " ".join(str(followup or "").split())
        resume_of = " ".join(str(resume_of or "").split())
        run_mode = "fast" if str(run_mode or "standard").strip().casefold() == "fast" else "standard"
        if resume_of and (parent_run_id or followup):
            raise ValueError("resume_of cannot be combined with parent_run_id/followup")
        if not question:
            raise ValueError("question cannot be empty")
        if len(question) > 4000:
            raise ValueError("question too long; please keep it within 4000 characters")
        if len(model_name) > 128:
            raise ValueError("model name too long")
        if (
            len(qwen_api_key) > 4096
            or len(semantic_scholar_api_key) > 4096
            or len(openalex_api_key) > 4096
            or len(serper_api_key) > 4096
            or len(ads_api_token) > 4096
        ):
            raise ValueError("API key length is invalid")
        if (
            len(openalex_mailto) > 320
            or len(unpaywall_email) > 320
            or len(crossref_mailto) > 320
        ):
            raise ValueError("email address length is invalid")
        if len(followup) > 4000:
            raise ValueError("follow-up content too long; please keep it within 4000 characters")
        seed_state: dict[str, Any] | None = None
        if parent_run_id or followup:
            if not parent_run_id or not followup:
                raise ValueError("a follow-up run requires both parent_run_id and followup")
            seed_state = self._load_parent_state(parent_run_id)
        elif resume_of:
            # Pure resume: the parent's checkpoint is authoritative for the
            # question text, so the caller cannot drift from the failed run.
            seed_state = self._load_checkpoint_state(resume_of)
            question = str(seed_state.get("input_question") or question)

        # Resolve capability status before starting the worker so the first
        # API response can immediately tell the browser about degraded search
        # sources.  Only booleans and warning messages are exposed; credential
        # values remain in the memory-only credential store below.
        preview_config = PipelineConfig.from_yaml(self.config_path)
        if run_mode == "fast":
            preview_config.apply_fast_mode_preset()
        m2_override = preview_config.module_overrides.get("m2")
        m2_kwargs = dict(m2_override.kwargs) if m2_override is not None else {}
        effective_qwen = str(
            qwen_api_key or preview_config.qwen.base.api_key or ""
        ).strip()
        effective_semantic_scholar = str(
            semantic_scholar_api_key
            or m2_kwargs.get("semantic_scholar_api_key", "")
            or os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
        ).strip()
        effective_openalex = str(
            openalex_api_key
            or m2_kwargs.get("openalex_api_key", "")
            or os.environ.get("OPENALEX_API_KEY", "")
        ).strip()
        effective_openalex_mailto = " ".join(str(
            openalex_mailto
            or m2_kwargs.get("openalex_mailto", "")
            or os.environ.get("OPENALEX_MAILTO", "")
        ).split())
        effective_serper = str(
            serper_api_key
            or m2_kwargs.get("serper_api_key", "")
            or os.environ.get("SERPER_API_KEY", "")
        ).strip()
        effective_ads = str(
            ads_api_token
            or m2_kwargs.get("ads_api_token", "")
            or os.environ.get("ADS_API_TOKEN", "")
        ).strip()
        effective_unpaywall = " ".join(str(
            unpaywall_email
            or m2_kwargs.get("unpaywall_email", "")
            or os.environ.get("UNPAYWALL_EMAIL", "")
        ).split())
        effective_crossref = " ".join(str(
            crossref_mailto
            or m2_kwargs.get("crossref_mailto", "")
            or os.environ.get("CROSSREF_MAILTO", "")
        ).split())
        enabled_sources = list(preview_config.search.tools)
        credential_warnings = (
            literature_credential_warnings(
                enabled_sources,
                serper_api_key=effective_serper,
                ads_api_token=effective_ads,
            )
            if "m2" in preview_config.enabled_modules
            else []
        )
        credential_status = {
            "llm_configured": bool(
                effective_qwen
            ),
            "semantic_scholar_configured": bool(effective_semantic_scholar),
            "openalex_configured": bool(effective_openalex),
            "serper_configured": bool(effective_serper),
            "ads_configured": bool(effective_ads),
            "unpaywall_configured": bool(
                effective_unpaywall or effective_openalex_mailto
            ),
        }
        with self._lock:
            if any(item.get("status") == "running" for item in self._runs.values()):
                raise RuntimeError(
                    "a run is already in progress; submit after the current run finishes"
                )
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
                "credential_status": credential_status,
                "credential_warnings": credential_warnings,
                "run_mode": run_mode,
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
                "qwen_api_key": effective_qwen,
                "semantic_scholar_api_key": effective_semantic_scholar,
                "openalex_api_key": effective_openalex,
                "openalex_mailto": effective_openalex_mailto,
                "serper_api_key": effective_serper,
                "ads_api_token": effective_ads,
                "unpaywall_email": effective_unpaywall,
                "crossref_mailto": effective_crossref,
                "run_mode": run_mode,
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
            raise ValueError("run_id contains invalid characters")
        with self._lock:
            record = self._runs.get(run_id)
            if record is not None and record.get("status") != "running":
                raise RuntimeError(
                    f"Run {run_id} already finished (status: {record.get('status')}), "
                    "cannot be stopped"
                )
            if record is None:
                # Not live in this process — check the persisted manifest so
                # already-finished runs report "already finished" instead of 404.
                manifest = self.get(run_id)
                if manifest is not None:
                    raise RuntimeError(
                        f"Run {run_id} already finished"
                        f" (status: {manifest.get('status', '')}); cannot be stopped"
                    )
                raise LookupError(f"Run {run_id} does not exist")
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
                    "stop requested: cancelling the running async task; "
                    "results of completed modules are preserved"
                ),
            )
        return {"run_id": run_id, "status": "cancelling"}

    def _load_parent_state(self, parent_run_id: str) -> dict[str, Any]:
        """Load and validate a parent run's final state for a follow-up run."""

        if not re.fullmatch(r"[A-Za-z0-9._-]+", parent_run_id):
            raise ValueError("parent_run_id contains invalid characters")
        state_path = self.output_root / parent_run_id / f"{parent_run_id}.json"
        if not state_path.exists():
            raise ValueError(
                f"Parent run {parent_run_id} has no final state; cannot start a follow-up"
            )
        try:
            data = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Parent run {parent_run_id} state file is corrupted; cannot start a follow-up: {exc}"
            ) from exc
        if not isinstance(data, dict):
            raise ValueError(f"Parent run {parent_run_id} has an invalid state format")
        return data

    def _load_checkpoint_state(self, run_id: str) -> dict[str, Any]:
        """Load and validate a run's checkpoint for a pure resume."""

        if not re.fullmatch(r"[A-Za-z0-9._-]+", run_id):
            raise ValueError("resume_of contains invalid characters")
        record = self._runs.get(run_id)
        if record and record.get("status") == "running":
            raise ValueError("parent run is still running; cannot retry from checkpoint")
        ckpt_path = self.output_root / run_id / f"{run_id}_checkpoint.json"
        if not ckpt_path.exists():
            raise ValueError(
                f"Parent run {run_id} has no usable checkpoint — "
                "it may have failed before the first module completed; please start a new run"
            )
        try:
            data = json.loads(ckpt_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Parent run {run_id} checkpoint file is corrupted; cannot resume: {exc}"
            ) from exc
        if not isinstance(data, dict):
            raise ValueError(f"Parent run {run_id} has an invalid checkpoint format")
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
            run_mode = credentials.get("run_mode", "standard")
            if run_mode == "fast":
                config.apply_fast_mode_preset()
            model_name = credentials.get("model_name", "")
            if run_mode == "fast" and str(model_name or "").strip() in (
                "", "qwen3.7-plus"
            ):
                # Fast mode should use the fast preset model unless the user
                # explicitly chose a different model in the UI.
                model_name = ""
            qwen_api_key = credentials.get("qwen_api_key", "")
            semantic_scholar_api_key = credentials.get(
                "semantic_scholar_api_key", ""
            )
            openalex_api_key = credentials.get("openalex_api_key", "")
            openalex_mailto = credentials.get("openalex_mailto", "")
            serper_api_key = credentials.get("serper_api_key", "")
            ads_api_token = credentials.get("ads_api_token", "")
            unpaywall_email = credentials.get("unpaywall_email", "")
            crossref_mailto = credentials.get("crossref_mailto", "")
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
            m2_extra_credentials = {
                "serper_api_key": serper_api_key,
                "ads_api_token": ads_api_token,
                "unpaywall_email": unpaywall_email,
                "crossref_mailto": crossref_mailto,
            }
            if any(m2_extra_credentials.values()):
                m2_override = config.module_overrides.setdefault(
                    "m2", ModuleOverride()
                )
                for key, value in m2_extra_credentials.items():
                    if value:
                        m2_override.kwargs[key] = value
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
                "credential_status": dict(
                    self._runs[run_id].get("credential_status", {})
                ),
                "credential_warnings": list(
                    self._runs[run_id].get("credential_warnings", [])
                ),
                "run_mode": run_mode,
                "output_dir": str(run_dir),
            }
            if followup_info:
                manifest["parent_run_id"] = followup_info.get("parent_run_id", "")
                manifest["followup"] = followup_info.get("followup", "")
            if resume_info:
                manifest["resume_of"] = resume_info.get("resume_of", "")
            recorder.write_manifest(manifest)
            for notice in manifest["credential_warnings"]:
                recorder.emit(
                    "credential_warning",
                    module="m2",
                    tool=str(notice.get("source") or "credential"),
                    status="warning",
                    message=str(notice.get("message") or "检索凭证未配置"),
                    details={
                        "code": str(notice.get("code") or ""),
                        "source": str(notice.get("source") or ""),
                    },
                )
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
                    "user stopped the run: results of completed modules are preserved "
                    "and can be used as the parent for a follow-up"
                )
            recorder.write_manifest(manifest)
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
            completed_at = datetime.now(timezone.utc).isoformat()
            recorder.emit(
                "run_failed",
                status="failed",
                message=f"pipeline terminated abnormally: {error}",
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
                "credential_status": dict(
                    self._runs[run_id].get("credential_status", {})
                ),
                "credential_warnings": list(
                    self._runs[run_id].get("credential_warnings", [])
                ),
                "run_mode": run_mode,
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
            raise ValueError("run_id contains invalid characters")
        display_name = " ".join(str(display_name or "").split())
        if not display_name:
            raise ValueError("name cannot be empty")
        if len(display_name) > 200:
            raise ValueError("name too long; please keep it within 200 characters")
        manifest_path = self.output_root / run_id / "manifest.json"
        if not manifest_path.exists():
            raise LookupError(f"Run {run_id} does not exist")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Run {run_id} manifest is corrupted; cannot rename: {exc}"
            ) from exc
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
            raise ValueError("invalid run_id: only run directories starting with 'ui-' are allowed")
        run_dir = (self.output_root / run_id).resolve()
        try:
            run_dir.relative_to(self.output_root)
        except ValueError as exc:
            raise ValueError("run_id points outside the runs directory") from exc
        if not run_dir.is_dir():
            raise LookupError(f"Run {run_id} does not exist")

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
                f"Run(s) {', '.join(sorted(busy))} still running; "
                "wait for them to finish before deleting"
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

        reviews, score_reconciliation = _reconcile_legacy_m6_reviews(
            state.get("reviews", []),
            scores,
        )
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
            "clarification_request": state.get("clarification_request"),
            "evidence_gap_requests": state.get("evidence_gap_requests", []),
            "graph_correction_requests": state.get("graph_correction_requests", []),
            "research_plans": state.get("research_plans", []),
            "research_plan_history": state.get("research_plan_history", {}),
            "reviews": reviews,
            # New M6 score contract.  Older runs simply omit this field and
            # remain renderable through the legacy review reconciliation above.
            "m6_scoring_summary": state.get("m6_scoring_summary"),
            "experimental_validation_verdict": state.get("experimental_validation_verdict"),
            "evidence_verdict": state.get("evidence_verdict"),
            "m6_score_reconciliation": score_reconciliation,
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
            "token_usage_by_module": state.get("token_usage_by_module", {}),
            "scoring_token_usage": state.get("scoring_token_usage", {}),
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

    _STATIC_ASSETS = {
        "/assets/ui-polish.css": (
            "ui-polish.css",
            "text/css; charset=utf-8",
        ),
    }

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

    def _serve_static_asset(self, request_path: str) -> bool:
        asset = self._STATIC_ASSETS.get(request_path)
        if asset is None:
            return False
        filename, content_type = asset
        path = self.server.static_dir / filename
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return True
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)
        return True

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
        if parsed.path.startswith("/assets/"):
            if self._serve_static_asset(parsed.path):
                return
            self.send_error(HTTPStatus.NOT_FOUND)
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
                    raise ValueError("request content too large")
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
                self._json({"error": "invalid request JSON"}, HTTPStatus.BAD_REQUEST)
                return
            self._json({"run": info}, HTTPStatus.OK)
            return
        if parsed.path != "/api/runs":
            self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 16_384:
                raise ValueError("request content too large")
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
                serper_api_key=payload.get("serper_api_key", ""),
                ads_api_token=payload.get("ads_api_token", ""),
                unpaywall_email=payload.get("unpaywall_email", ""),
                crossref_mailto=payload.get("crossref_mailto", ""),
                parent_run_id=payload.get("parent_run_id", ""),
                followup=payload.get("followup", ""),
                resume_of=payload.get("resume_of", ""),
                run_mode=payload.get("run_mode", "standard"),
            )
        except ValueError as exc:
            self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        except RuntimeError as exc:
            self._json({"error": str(exc)}, HTTPStatus.CONFLICT)
            return
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._json({"error": "invalid request JSON"}, HTTPStatus.BAD_REQUEST)
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
