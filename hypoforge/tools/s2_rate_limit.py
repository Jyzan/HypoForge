"""Per-key Semantic Scholar request spacing shared by local processes.

Search, paper metadata and external scoring all use this coordinator. State
contains only timestamps; the filename is a credential digest. Linux and
macOS file locks also coordinate separate HypoForge processes on one machine.
"""

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
import re
from pathlib import Path
import tempfile
import time


def relevance_query(text: str) -> str:
    """Plain-text syntax for /paper/search; never silently invert a negation."""
    if re.search(r"\bNOT\b", text):
        raise ValueError("Semantic Scholar relevance search cannot express Boolean NOT")
    text = re.sub(r"\[[^\]]+\]", "", text)
    text = re.sub(r"\b(?:AND|OR)\b", " ", text)
    text = re.sub(r'["()\-–—]', " ", text)
    return " ".join(text.split())


def _state_path(api_key: str) -> Path:
    root = Path(os.environ.get(
        "SEMANTIC_SCHOLAR_RATE_LIMIT_DIR",
        str(Path(tempfile.gettempdir()) / f"hypoforge-s2-{os.getuid()}"),
    ))
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    name = hashlib.sha256(api_key.encode()).hexdigest()
    return root / f"{name}.json"


def _sleep(seconds: float, deadline: float | None) -> None:
    if deadline is not None and time.monotonic() + seconds >= deadline:
        raise TimeoutError("Semantic Scholar shared request queue exceeded its deadline")
    if seconds > 0:
        time.sleep(seconds)


@contextmanager
def _locked_state(api_key: str, deadline: float | None = None):
    fd = os.open(_state_path(api_key), os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "r+") as stream:
        while True:
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                _sleep(0.05, deadline)
        try:
            raw = stream.read()
            state = json.loads(raw) if raw else {}
            yield state
            stream.seek(0)
            json.dump(state, stream)
            stream.truncate()
            stream.flush()
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def wait_for_slot(api_key: str, min_interval: float, deadline: float | None = None) -> None:
    """Reserve one request start, including any shared HTTP 429 cooldown."""
    with _locked_state(api_key, deadline) as state:
        now = time.monotonic()
        last = float(state.get("last_start", 0.0))
        blocked = float(state.get("blocked_until", 0.0))
        # Monotonic timestamps from a previous boot must not block this boot.
        if last > now:
            last = blocked = 0.0
        wait = max(0.0, last + min_interval - now, blocked - now)
        _sleep(wait, deadline)
        state["last_start"] = time.monotonic()


def defer_requests(api_key: str, seconds: float) -> None:
    """Tell every local caller to respect the same provider cooldown."""
    with _locked_state(api_key) as state:
        state["blocked_until"] = max(
            float(state.get("blocked_until", 0.0)),
            time.monotonic() + max(0.0, seconds),
        )
