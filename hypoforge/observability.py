"""Lightweight run observability shared by the pipeline and local web UI.

The recorder is deliberately dependency-free.  Events are appended to JSONL as
soon as they happen, while full state snapshots are stored separately so the
event stream stays small enough for frequent browser polling.
"""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Protocol

from pydantic import BaseModel


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class EventSink(Protocol):
    """Small protocol used by optional live terminal/web event consumers."""

    def handle_event(self, event: Mapping[str, Any]) -> None:
        ...


class RunEventRecorder:
    """Append-only event recorder with per-stage JSON snapshots."""

    def __init__(self, run_dir: str | Path, run_id: str) -> None:
        self.run_dir = Path(run_dir)
        self.run_id = run_id
        self.events_path = self.run_dir / "events.jsonl"
        self.manifest_path = self.run_dir / "manifest.json"
        self.snapshots_dir = self.run_dir / "snapshots"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._sequence = self._read_last_sequence()
        self._snapshot_sequence = len(list(self.snapshots_dir.glob("*.json")))

    def _read_last_sequence(self) -> int:
        if not self.events_path.exists():
            return 0
        try:
            lines = self.events_path.read_text(encoding="utf-8").splitlines()
            return int(json.loads(lines[-1]).get("sequence", 0)) if lines else 0
        except (OSError, ValueError, json.JSONDecodeError):
            return 0

    def emit(
        self,
        event_type: str,
        *,
        module: str = "",
        tool: str = "",
        status: str = "",
        message: str = "",
        elapsed_seconds: float | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            self._sequence += 1
            event = {
                "sequence": self._sequence,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "run_id": self.run_id,
                "event_type": event_type,
                "module": module,
                "tool": tool,
                "status": status,
                "message": message,
                "elapsed_seconds": (
                    round(float(elapsed_seconds), 4)
                    if elapsed_seconds is not None
                    else None
                ),
                "details": _jsonable(details or {}),
            }
            with self.events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
            return event

    def save_snapshot(self, label: str, state: Any) -> Path:
        safe_label = "".join(
            char if char.isalnum() or char in {"-", "_"} else "-"
            for char in label
        ).strip("-") or "snapshot"
        with self._lock:
            self._snapshot_sequence += 1
            path = self.snapshots_dir / (
                f"{self._snapshot_sequence:03d}-{safe_label}.json"
            )
            path.write_text(
                json.dumps(_jsonable(state), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return path

    def write_manifest(self, data: Mapping[str, Any]) -> None:
        with self._lock:
            self.manifest_path.write_text(
                json.dumps(_jsonable(data), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def read_events(self, after: int = 0) -> list[dict[str, Any]]:
        if not self.events_path.exists():
            return []
        events: list[dict[str, Any]] = []
        with self._lock:
            for line in self.events_path.read_text(encoding="utf-8").splitlines():
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if int(event.get("sequence", 0)) > after:
                    events.append(event)
        return events


_CURRENT_RECORDER: ContextVar[RunEventRecorder | None] = ContextVar(
    "hypoforge_run_event_recorder",
    default=None,
)
_CURRENT_EVENT_SINK: ContextVar[EventSink | None] = ContextVar(
    "hypoforge_event_sink",
    default=None,
)


@contextmanager
def bind_recorder(recorder: RunEventRecorder | None) -> Iterator[None]:
    """Make a recorder visible to nested async Tools via ``contextvars``."""

    token = _CURRENT_RECORDER.set(recorder)
    try:
        yield
    finally:
        _CURRENT_RECORDER.reset(token)


@contextmanager
def bind_event_sink(sink: EventSink | None) -> Iterator[None]:
    """Bind an optional live event consumer for the current async context."""

    token = _CURRENT_EVENT_SINK.set(sink)
    try:
        yield
    finally:
        _CURRENT_EVENT_SINK.reset(token)


def notify_event(event: Mapping[str, Any]) -> None:
    """Send an already-recorded event to the current live consumer."""

    sink = _CURRENT_EVENT_SINK.get()
    if sink is None:
        return
    try:
        sink.handle_event(event)
    except Exception:
        # Terminal presentation must never break a scientific run.
        return


def emit_event(
    event_type: str,
    *,
    module: str = "",
    tool: str = "",
    status: str = "",
    message: str = "",
    elapsed_seconds: float | None = None,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Emit to the current recorder, or do nothing in ordinary CLI runs."""

    recorder = _CURRENT_RECORDER.get()
    event = (
        recorder.emit(
            event_type,
            module=module,
            tool=tool,
            status=status,
            message=message,
            elapsed_seconds=elapsed_seconds,
            details=details,
        )
        if recorder is not None
        else {
            "event_type": event_type,
            "module": module,
            "tool": tool,
            "status": status,
            "message": message,
            "elapsed_seconds": elapsed_seconds,
            "details": dict(details or {}),
        }
    )
    notify_event(event)
    return event if recorder is not None else None
