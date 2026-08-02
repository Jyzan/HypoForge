"""Resolve selected arXiv papers to bounded, cached PDF documents."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from time import monotonic as _monotonic
import urllib.parse
import urllib.request
from collections.abc import Awaitable, Callable
from pathlib import Path

from ..models import ContentLevel, DocumentRecord, PaperRecord
from ..protocols import FulltextResolverProtocol


ArxivPDFFetchBackend = Callable[[str, float], Awaitable[bytes]]


class ArxivDownloadTimeoutError(TimeoutError):
    """Raised when one PDF exceeds its end-to-end download deadline."""


def _safe_error(exc: BaseException) -> str:
    return " ".join(f"{type(exc).__name__}: {exc}".split())[:500]


def _set_response_read_timeout(response: object, seconds: float) -> bool:
    """Best-effort socket timeout update for urllib/http.client responses."""

    fp = getattr(response, "fp", None)
    raw = getattr(fp, "raw", None)
    candidates = (response, fp, raw, getattr(raw, "_sock", None))
    for candidate in candidates:
        setter = getattr(candidate, "settimeout", None)
        if not callable(setter):
            continue
        try:
            setter(seconds)
        except (OSError, TypeError, ValueError):
            continue
        return True
    return False


def _read_bounded_response(
    response: object,
    *,
    max_bytes: int,
    deadline: float,
    clock: Callable[[], float],
    timeout_seconds: float = 180.0,
    socket_timeout_seconds: float = 30.0,
    chunk_size: int = 65_536,
) -> bytes:
    """Read a blocking HTTP response under byte and monotonic-time limits."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    headers = getattr(response, "headers", {})
    raw_length = headers.get("Content-Length") if headers is not None else None
    if raw_length:
        try:
            content_length = int(raw_length)
        except (TypeError, ValueError):
            content_length = 0
        if content_length > max_bytes:
            raise ValueError("arXiv PDF exceeds configured size limit")

    chunks: list[bytes] = []
    used = 0
    while True:
        now = clock()
        if now >= deadline:
            raise ArxivDownloadTimeoutError(
                "arXiv PDF exceeded "
                f"{timeout_seconds:g} second download deadline"
            )
        remaining = deadline - now
        deadline_limited = remaining <= socket_timeout_seconds
        _set_response_read_timeout(
            response,
            max(0.001, min(socket_timeout_seconds, remaining)),
        )
        try:
            chunk = response.read(chunk_size)
        except TimeoutError as exc:
            if deadline_limited or clock() >= deadline:
                raise ArxivDownloadTimeoutError(
                    "arXiv PDF exceeded "
                    f"{timeout_seconds:g} second download deadline"
                ) from exc
            raise
        if clock() >= deadline:
            raise ArxivDownloadTimeoutError(
                "arXiv PDF exceeded "
                f"{timeout_seconds:g} second download deadline"
            )
        if not chunk:
            break
        used += len(chunk)
        if used > max_bytes:
            raise ValueError("arXiv PDF exceeds configured size limit")
        chunks.append(chunk)
    return b"".join(chunks)


async def _default_fetch(
    url: str,
    timeout: float,
    max_bytes: int,
    download_timeout_seconds: float,
) -> bytes:
    def fetch() -> bytes:
        started = _monotonic()
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "HypoForge/0.1 arxiv-fulltext"},
        )
        socket_timeout = min(timeout, 30.0, download_timeout_seconds)
        with urllib.request.urlopen(request, timeout=socket_timeout) as response:
            return _read_bounded_response(
                response,
                max_bytes=max_bytes,
                deadline=started + download_timeout_seconds,
                clock=_monotonic,
                timeout_seconds=download_timeout_seconds,
                socket_timeout_seconds=socket_timeout,
            )

    return await asyncio.to_thread(fetch)


class ArxivPDFResolver(FulltextResolverProtocol):
    """Download and cache arXiv PDFs, with explicit abstract fallback."""

    def __init__(
        self,
        cache_dir: str | Path = ".cache/hypoforge/literature/documents",
        timeout_seconds: float = 30.0,
        download_timeout_seconds: float = 180.0,
        max_pdf_bytes: int = 52_428_800,
        backend: ArxivPDFFetchBackend | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if download_timeout_seconds <= 0:
            raise ValueError("download_timeout_seconds must be positive")
        if max_pdf_bytes <= 0:
            raise ValueError("max_pdf_bytes must be positive")
        self.cache_dir = Path(cache_dir)
        self.timeout_seconds = float(timeout_seconds)
        self.download_timeout_seconds = float(download_timeout_seconds)
        self.max_pdf_bytes = int(max_pdf_bytes)
        self.backend = backend

    def _paper_dir(self, paper: PaperRecord) -> Path:
        stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", paper.paper_id).strip("_")
        digest = hashlib.sha256(paper.paper_id.encode("utf-8")).hexdigest()[:12]
        return self.cache_dir / f"{stem[:60] or 'paper'}-{digest}"

    @staticmethod
    def _document_id(paper: PaperRecord, suffix: str) -> str:
        digest = hashlib.sha256(paper.paper_id.encode("utf-8")).hexdigest()[:16]
        return f"document:{digest}:{suffix}"

    @staticmethod
    def _write_atomic(path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(payload)
        temporary.replace(path)

    @staticmethod
    def _arxiv_id(paper: PaperRecord) -> str:
        external = str(paper.external_ids.get("arxiv") or "").strip()
        if external:
            return external
        if paper.paper_id.upper().startswith("ARXIV:"):
            return paper.paper_id.split(":", 1)[1].strip()
        return ""

    def _valid_cached_pdf(self, path: Path) -> bool:
        try:
            if not path.is_file() or not (
                5 <= path.stat().st_size <= self.max_pdf_bytes
            ):
                return False
            with path.open("rb") as handle:
                return handle.read(5) == b"%PDF-"
        except OSError:
            return False

    def _pdf_document(
        self,
        paper: PaperRecord,
        path: Path,
        source_uri: str,
    ) -> DocumentRecord:
        return DocumentRecord(
            document_id=self._document_id(paper, "arxiv-pdf"),
            paper_id=paper.paper_id,
            content_level=ContentLevel.PDF,
            source_uri=source_uri,
            local_path=str(path.resolve()),
        )

    def _abstract_document(self, paper: PaperRecord, error: str) -> DocumentRecord:
        abstract = " ".join(paper.abstract.split())
        if not abstract:
            detail = f"{error}; no readable content" if error else "no readable content"
            return DocumentRecord(
                document_id=self._document_id(paper, "metadata"),
                paper_id=paper.paper_id,
                content_level=ContentLevel.METADATA,
                retrieval_error=detail,
            )
        path = self._paper_dir(paper) / "abstract.json"
        payload = json.dumps(
            {"format": "hypoforge_abstract_v1", "text": abstract},
            ensure_ascii=False,
        ).encode("utf-8")
        self._write_atomic(path, payload)
        return DocumentRecord(
            document_id=self._document_id(paper, "abstract"),
            paper_id=paper.paper_id,
            content_level=ContentLevel.ABSTRACT,
            local_path=str(path.resolve()),
            retrieval_error=error,
        )

    async def resolve_abstract(self, paper: PaperRecord) -> DocumentRecord:
        """Materialize the real abstract without another network request."""
        return self._abstract_document(
            paper,
            "arXiv full text unavailable; used abstract",
        )

    async def resolve(self, paper: PaperRecord) -> DocumentRecord:
        arxiv_id = self._arxiv_id(paper)
        if not arxiv_id:
            return self._abstract_document(paper, "arXiv identifier unavailable")
        quoted_id = urllib.parse.quote(arxiv_id, safe="/.")
        source_uri = f"https://arxiv.org/pdf/{quoted_id}"
        path = self._paper_dir(paper) / "paper.pdf"
        if self._valid_cached_pdf(path):
            return self._pdf_document(paper, path, source_uri)
        try:
            if self.backend is None:
                fetch_task = asyncio.create_task(
                    _default_fetch(
                        source_uri,
                        self.timeout_seconds,
                        self.max_pdf_bytes,
                        self.download_timeout_seconds,
                    )
                )
                try:
                    done, _ = await asyncio.wait(
                        {fetch_task},
                        timeout=self.download_timeout_seconds,
                    )
                except asyncio.CancelledError:
                    fetch_task.cancel()
                    raise
                if fetch_task not in done:
                    fetch_task.cancel()
                    raise ArxivDownloadTimeoutError(
                        "arXiv PDF exceeded "
                        f"{self.download_timeout_seconds:g} second download deadline"
                    )
                payload = fetch_task.result()
            else:
                payload = await self.backend(source_uri, self.timeout_seconds)
            if len(payload) > self.max_pdf_bytes:
                raise ValueError("arXiv PDF exceeds configured size limit")
            if not payload.lstrip().startswith(b"%PDF-"):
                raise ValueError("invalid PDF response from arXiv")
            self._write_atomic(path, payload)
            return self._pdf_document(paper, path, source_uri)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return self._abstract_document(paper, _safe_error(exc))
