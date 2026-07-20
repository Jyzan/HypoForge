"""Resolve legally reusable PMC full text with an explicit abstract fallback."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import urllib.request
from collections.abc import Awaitable, Callable
from pathlib import Path

from ..models import ContentLevel, DocumentRecord, PaperRecord
from ..protocols import FulltextResolverProtocol

FetchBackend = Callable[[str, float], Awaitable[bytes]]


def _safe_error(exc: BaseException) -> str:
    return " ".join(f"{type(exc).__name__}: {exc}".split())[:500]


def _documents(payload: object) -> list[dict]:
    if isinstance(payload, list):
        documents: list[dict] = []
        for item in payload:
            documents.extend(_documents(item))
        return documents
    if isinstance(payload, dict):
        raw = payload.get("documents")
        if isinstance(raw, list):
            documents: list[dict] = []
            for item in raw:
                documents.extend(_documents(item))
            return documents
        if isinstance(payload.get("passages"), list):
            return [payload]
    return []


def _has_nonempty_passage(payload: object) -> bool:
    for document in _documents(payload):
        passages = document.get("passages", [])
        if not isinstance(passages, list):
            continue
        for passage in passages:
            if isinstance(passage, dict) and str(passage.get("text") or "").strip():
                return True
    return False


async def _default_fetch(url: str, timeout: float) -> bytes:
    def fetch() -> bytes:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "HypoForge/0.1 literature-reading"},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()

    return await asyncio.to_thread(fetch)


class PMCFulltextResolver(FulltextResolverProtocol):
    """Fetch BioC JSON from the PMC Open Access API and cache it locally."""

    API_ROOT = (
        "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/"
        "pmcoa.cgi/BioC_json"
    )

    def __init__(
        self,
        cache_dir: str | Path = ".cache/hypoforge/literature/documents",
        timeout_seconds: float = 30.0,
        backend: FetchBackend | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.cache_dir = Path(cache_dir)
        self.timeout_seconds = float(timeout_seconds)
        self.backend = backend or _default_fetch

    def _paper_dir(self, paper: PaperRecord) -> Path:
        stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", paper.paper_id).strip("_")
        digest = hashlib.sha256(paper.paper_id.encode("utf-8")).hexdigest()[:12]
        return self.cache_dir / f"{stem[:60] or 'paper'}-{digest}"

    @staticmethod
    def _document_id(paper: PaperRecord, suffix: str) -> str:
        digest = hashlib.sha256(paper.paper_id.encode("utf-8")).hexdigest()[:16]
        return f"document:{digest}:{suffix}"

    @staticmethod
    def _identifier(paper: PaperRecord) -> str:
        return (paper.pmcid or paper.pmid).strip()

    @staticmethod
    def _write_atomic(path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(payload)
        temporary.replace(path)

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

    async def resolve(self, paper: PaperRecord) -> DocumentRecord:
        identifier = self._identifier(paper)
        if not identifier:
            return self._abstract_document(paper, "PMC identifier unavailable")

        path = self._paper_dir(paper) / "bioc.json"
        source_uri = f"{self.API_ROOT}/{identifier}/unicode"
        if path.is_file():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if _has_nonempty_passage(payload):
                    return DocumentRecord(
                        document_id=self._document_id(paper, "pmc"),
                        paper_id=paper.paper_id,
                        content_level=ContentLevel.STRUCTURED_FULLTEXT,
                        source_uri=source_uri,
                        local_path=str(path.resolve()),
                        license="PMC Open Access subset",
                    )
            except (OSError, UnicodeError, json.JSONDecodeError):
                pass

        try:
            raw = await self.backend(source_uri, self.timeout_seconds)
            decoded = raw.decode("utf-8")
            if decoded.lstrip().casefold().startswith("[error]"):
                detail = " ".join(decoded.split())[:300]
                raise ValueError(f"PMC Open Access full text unavailable: {detail}")
            payload = json.loads(decoded)
            if not _has_nonempty_passage(payload):
                raise ValueError("PMC BioC response has no non-empty passages")
            normalized = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self._write_atomic(path, normalized)
            return DocumentRecord(
                document_id=self._document_id(paper, "pmc"),
                paper_id=paper.paper_id,
                content_level=ContentLevel.STRUCTURED_FULLTEXT,
                source_uri=source_uri,
                local_path=str(path.resolve()),
                license="PMC Open Access subset",
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return self._abstract_document(paper, _safe_error(exc))

    async def resolve_abstract(self, paper: PaperRecord) -> DocumentRecord:
        """Materialize the paper's real abstract without another network request."""

        return self._abstract_document(
            paper,
            "full text retrieval unavailable; used abstract",
        )
