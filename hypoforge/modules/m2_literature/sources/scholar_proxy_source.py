"""Google Scholar discovery through explicitly configured proxy APIs."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from typing import Any

from ..models import FulltextStatus, PaperRecord, SearchQuery
from ..protocols import LiteratureSourceProtocol


_YEAR_RE = re.compile(r"\b((?:19|20)\d{2})\b")
_DOI_RE = re.compile(r"(?:https?://doi\.org/)?(10\.\d{4,9}/[^\s?#]+)", re.I)
_SERPER_RATE_LOCK = threading.Lock()
_SERPER_NEXT_START = 0.0


class ScholarProxyError(OSError):
    """Raised when a Scholar proxy cannot complete a request."""


def _request_json(
    url: str,
    *,
    timeout: float,
    headers: dict[str, str] | None = None,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    merged_headers = {
        "Accept": "application/json",
        "User-Agent": "HypoForge/0.1 (academic research)",
    }
    if headers:
        merged_headers.update(headers)
    body = None
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        merged_headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=body, headers=merged_headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read(500).decode("utf-8", errors="replace")
        raise ScholarProxyError(
            f"HTTP {exc.code} from Scholar proxy: {detail}"
        ) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ScholarProxyError(f"Scholar proxy request failed: {exc}") from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ScholarProxyError(f"Scholar proxy returned invalid JSON: {exc}") from exc


def _safe_int(value: Any) -> int | None:
    try:
        return int(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _record(row: dict[str, Any]) -> PaperRecord | None:
    title = " ".join(str(row.get("title") or "").split())
    if not title:
        return None
    link = str(row.get("link") or "").strip()
    snippet = " ".join(str(row.get("snippet") or "").split())
    publication = " ".join(str(row.get("publicationInfo") or "").split())
    doi = ""
    for value in (link, snippet, publication):
        match = _DOI_RE.search(value)
        if match:
            doi = PaperRecord.normalize_doi(match.group(1).rstrip(".,);"))
            if doi.casefold().endswith(".pdf"):
                doi = doi[:-4].rstrip(".,);")
            break
    year = _safe_int(row.get("year"))
    if year is None:
        matches = _YEAR_RE.findall(publication)
        year = int(matches[-1]) if matches else None
    authors_text = publication.split(" - ", 1)[0].strip()
    authors = [item.strip() for item in authors_text.split(",") if item.strip()]
    digest = hashlib.sha256(title.casefold().encode("utf-8")).hexdigest()[:20]
    external_ids = {"text_kind": "search_snippet"}
    if link:
        external_ids["serper_url"] = link
    return PaperRecord(
        paper_id=f"DOI:{doi}" if doi else f"SERPER_SCHOLAR-TITLE:{digest}",
        title=title,
        abstract=snippet,
        authors=authors,
        year=year,
        journal=publication,
        doi=doi,
        external_ids=external_ids,
        citation_count=_safe_int(row.get("citedBy")),
        sources=["serper_scholar"],
        fulltext_status=(
            FulltextStatus.ABSTRACT_ONLY if snippet else FulltextStatus.UNKNOWN
        ),
    )


class ScholarProxySource(LiteratureSourceProtocol):
    """Search Google Scholar through Serper's Scholar endpoint.

    The process-wide start limiter defaults to four requests per second, one
    request below Serper's observed Scholar ceiling.  It protects concurrent
    sub-question searches that share one key.
    """

    source_name = "serper_scholar"

    def __init__(
        self,
        *,
        api_key: str = "",
        timeout_seconds: float = 20.0,
        requests_per_second: float | None = None,
    ) -> None:
        self.api_key = str(api_key or os.environ.get("SERPER_API_KEY", "")).strip()
        if not self.api_key:
            raise ValueError("SERPER_API_KEY is required for domain-routed search")
        self.timeout_seconds = float(timeout_seconds)
        resolved_rate = (
            float(os.environ.get("SERPER_REQUESTS_PER_SECOND", "4"))
            if requests_per_second is None else float(requests_per_second)
        )
        if self.timeout_seconds <= 0 or resolved_rate <= 0:
            raise ValueError("Scholar timeout and request rate must be positive")
        self.requests_per_second = resolved_rate

    def _wait_for_slot(self) -> None:
        global _SERPER_NEXT_START
        interval = 1.0 / self.requests_per_second
        with _SERPER_RATE_LOCK:
            now = time.monotonic()
            scheduled = max(now, _SERPER_NEXT_START)
            _SERPER_NEXT_START = scheduled + interval
        delay = scheduled - time.monotonic()
        if delay > 0:
            time.sleep(delay)

    def _search_sync(self, text: str, limit: int) -> list[PaperRecord]:
        self._wait_for_slot()
        payload = _request_json(
            "https://google.serper.dev/scholar",
            timeout=self.timeout_seconds,
            headers={"X-API-KEY": self.api_key},
            payload={"q": text, "num": min(limit, 20), "hl": "en"},
        )
        return [
            record for row in payload.get("organic", [])
            if (record := _record(row)) is not None
        ]

    async def search(self, query: SearchQuery, limit: int = 20) -> list[PaperRecord]:
        return await asyncio.to_thread(
            self._search_sync, query.text, min(max(limit, 1), 20)
        )
