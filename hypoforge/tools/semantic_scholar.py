"""
Academic literature search tool — Semantic Scholar / OpenAlex dual backend.

- If ``SEMANTIC_SCHOLAR_API_KEY`` is set in ``.env`` → uses Semantic Scholar
  (higher rate limit, better precision for CS / biomedicine).
- Otherwise → falls back to **OpenAlex** (free, no key required, 10 req/s,
  covers all disciplines, works without a VPN).

Both backends return the same standardised dict schema so M2 doesn't care
which backend is active.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
import threading
import time
import urllib.parse
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from ..protocol import ToolProtocol
from ..registry import ToolRegistry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Resolve config from environment (once at import time)
# ---------------------------------------------------------------------------

def _resolve_config() -> tuple[str, str]:
    """Load ``.env``, return ``(s2_api_key, backend_name)``."""
    from pathlib import Path

    candidates = [
        Path(__file__).resolve().parent.parent.parent / ".env",
        Path.cwd() / ".env",
    ]
    for p in candidates:
        if p.exists():
            load_dotenv(p, override=True)
            break
    else:
        load_dotenv(override=True)

    key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
    return key, "semantic_scholar" if key else "openalex"


_S2_API_KEY, _BACKEND = _resolve_config()
_OPENALEX_API_KEY = os.environ.get("OPENALEX_API_KEY", "")

# Module-level 429 circuit-breaker.  When S2 repeatedly rate-limits us,
# subsequent calls in the same process skip S2 entirely and go directly
# to OpenAlex.  The breaker auto-resets after a quiet period so that a
# later supplement round can try S2 again.
_s2_circuit_open_until: float = 0.0
_CIRCUIT_COOLDOWN_SECONDS = 120.0  # keep S2 off for 2 min after a 429 storm


def _s2_circuit_open() -> bool:
    """Return True when S2 should be skipped (breaker is tripped)."""
    return time.monotonic() < _s2_circuit_open_until


def _s2_circuit_break() -> None:
    """Trip the S2 circuit breaker for the cooldown period."""
    global _s2_circuit_open_until
    _s2_circuit_open_until = time.monotonic() + _CIRCUIT_COOLDOWN_SECONDS


def _s2_circuit_reset() -> None:
    """Reset the S2 circuit breaker (e.g. after a successful call)."""
    global _s2_circuit_open_until
    _s2_circuit_open_until = 0.0

# Free S2 keys are limited to ~1 request per second, but some trial /
# academic keys have even stricter quotas.  The default is deliberately
# conservative; set SEMANTIC_SCHOLAR_MIN_INTERVAL_SECONDS in .env to
# adjust for your specific key tier.
_S2_RATE_LIMIT = max(
    1.0,
    float(os.environ.get("SEMANTIC_SCHOLAR_MIN_INTERVAL_SECONDS", "5.0")),
)
_OPENALEX_RATE_LIMIT = 0.12
_last_request_time: float = 0.0
_rate_limit_lock = threading.Lock()


def _rate_limit(min_interval: float) -> None:
    global _last_request_time
    with _rate_limit_lock:
        now = time.monotonic()
        gap = min_interval - (now - _last_request_time)
        if gap > 0:
            time.sleep(gap)
        _last_request_time = time.monotonic()


def _http_get_json(url: str, *, s2_api_key: str = "") -> dict:
    """GET *url*, parse JSON, with rate-limit + retry on transient errors."""
    last_exc = None
    request_interval = (
        _S2_RATE_LIMIT
        if s2_api_key and "semanticscholar.org" in url
        else _OPENALEX_RATE_LIMIT
    )
    for attempt in range(4):
        _rate_limit(request_interval)
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "HypoForge/0.1.0 (Academic Research)")
        if s2_api_key and "semanticscholar.org" in url:
            req.add_header("x-api-key", s2_api_key)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            last_exc = e
            if e.code == 429 and attempt < 3:
                retry_after = e.headers.get("Retry-After", "")
                if retry_after:
                    try:
                        wait = float(retry_after)
                    except (ValueError, TypeError):
                        wait = 2.0
                else:
                    base = 2 ** attempt
                    wait = base + random.uniform(0, base * 0.25)
                logger.debug("429 rate-limited, retry in %.1fs", wait)
                time.sleep(wait)
            else:
                raise
        except (urllib.error.URLError, OSError) as e:
            last_exc = e
            if attempt < 3:
                base = 2 ** attempt
                wait = base + random.uniform(0, base * 0.25)
                logger.debug("Network error (%s), retry in %.1fs: %s", type(e).__name__, wait, e)
                time.sleep(wait)
            else:
                raise

    raise last_exc  # type: ignore[misc]


# ============================================================================
# Backend: Semantic Scholar
# ============================================================================

_S2_BASE = "https://api.semanticscholar.org/graph/v1"


def _s2_search(query: str, limit: int, api_key: str = "") -> List[dict]:
    """Search Semantic Scholar, return standardised paper dicts."""
    params: Dict[str, str] = {
        "query": query,
        "limit": str(min(limit, 100)),
        "fields": "title,year,authors,journal,externalIds,citationCount,abstract,openAccessPdf,isOpenAccess",
    }
    qs = urllib.parse.urlencode(params)
    url = f"{_S2_BASE}/paper/search?{qs}"

    data = _http_get_json(url, s2_api_key=api_key)
    papers = data.get("data", [])
    logger.info(
        "S2 search: query=%r → total=%s returned=%d",
        query, data.get("total", "?"), len(papers),
    )

    return [_s2_normalise(p) for p in papers]


def _s2_fetch(paper_id: str, api_key: str = "") -> dict:
    """Fetch a single paper from Semantic Scholar by ID."""
    fields = "title,abstract,year,authors,journal,externalIds,citationCount,openAccessPdf,isOpenAccess"
    url = f"{_S2_BASE}/paper/{paper_id}?fields={fields}"
    return _s2_normalise(_http_get_json(url, s2_api_key=api_key))


def _s2_normalise(raw: dict) -> dict:
    """Convert an S2 paper dict to the standardised schema."""
    authors = [a.get("name", "") for a in raw.get("authors", [])]
    journal_info = raw.get("journal") or {}
    ext_ids = raw.get("externalIds") or {}
    oa_pdf = raw.get("openAccessPdf") or {}
    oa_pdf_url = str(oa_pdf.get("url") or "") if isinstance(oa_pdf, dict) else ""
    return {
        "source": "semantic_scholar",
        "paper_id": raw.get("paperId", ""),
        "title": raw.get("title") or "",
        "abstract": raw.get("abstract") or "",
        "year": raw.get("year") or 0,
        "doi": ext_ids.get("DOI", ""),
        "pmid": ext_ids.get("PubMed", ""),
        "pmcid": ext_ids.get("PubMedCentral", ""),
        "external_ids": {
            str(key): str(value)
            for key, value in ext_ids.items()
            if value not in (None, "")
        },
        "oa_pdf_url": oa_pdf_url,
        "is_open_access": bool(raw.get("isOpenAccess")),
        "authors": authors,
        "journal": journal_info.get("name", ""),
        "citation_count": raw.get("citationCount", 0),
    }


# ============================================================================
# Backend: OpenAlex
# ============================================================================

_OA_BASE = "https://api.openalex.org"


def _oa_search(query: str, limit: int, api_key: str = "") -> List[dict]:
    """Search OpenAlex, return standardised paper dicts."""
    params = {
        "search": query,
        "per_page": str(min(limit, 200)),
        "sort": "relevance_score:desc",
    }
    if api_key:
        params["api_key"] = api_key
    url = f"{_OA_BASE}/works?{urllib.parse.urlencode(params)}"
    data = _http_get_json(url)
    papers = data.get("results", [])
    logger.info(
        "OpenAlex search: query=%r → total=%s returned=%d",
        query, data.get("meta", {}).get("count", "?"), len(papers),
    )
    return [_oa_normalise(p) for p in papers]


def _oa_fetch(work_id: str, api_key: str = "") -> dict:
    """Fetch a single work from OpenAlex by ID.

    The *work_id* can be a full OpenAlex URL
    (``https://openalex.org/W...``), a bare ID (``W...``),
    or an API URL (``https://api.openalex.org/works/W...``).
    """
    # Normalise to bare ID
    if "openalex.org/" in work_id:
        # https://openalex.org/W123 → W123
        # https://api.openalex.org/works/W123 → W123
        work_id = work_id.rsplit("/", 1)[-1]

    url = f"{_OA_BASE}/works/{work_id}"
    if api_key:
        url = f"{url}?{urllib.parse.urlencode({'api_key': api_key})}"
    return _oa_normalise(_http_get_json(url))


def _oa_normalise(raw: dict) -> dict:
    """Convert an OpenAlex work dict to the standardised schema."""
    # Authors
    authors = [
        a.get("author", {}).get("display_name", "")
        for a in raw.get("authorships", [])
    ]

    # Journal
    venue = raw.get("primary_location", {}) or {}
    source = venue.get("source") or {}
    journal = source.get("display_name", "")

    # DOI
    doi_raw = raw.get("doi") or ""
    # Strip OpenAlex's https://doi.org/ prefix if present
    if doi_raw.startswith("https://doi.org/"):
        doi_raw = doi_raw[len("https://doi.org/"):]

    # Abstract — reconstruct from inverted index
    abstract = ""
    inv_idx = raw.get("abstract_inverted_index")
    if inv_idx:
        try:
            max_pos = max(max(v) for v in inv_idx.values())
            words: list[str] = [""] * (max_pos + 1)
            for word, positions in inv_idx.items():
                for pos in positions:
                    words[pos] = word
            abstract = " ".join(words)
        except (ValueError, TypeError):
            pass

    return {
        "source": "openalex",
        "paper_id": raw.get("id", ""),
        "title": raw.get("title") or "",
        "abstract": abstract,
        "year": raw.get("publication_year") or 0,
        "doi": doi_raw,
        "external_ids": {
            str(key): str(value)
            for key, value in (raw.get("ids") or {}).items()
            if value not in (None, "")
        },
        "oa_pdf_url": str(
            ((raw.get("best_oa_location") or {}).get("pdf_url") or "")
        ),
        "is_open_access": bool((raw.get("open_access") or {}).get("is_oa")),
        "retrieval_relevance": raw.get("relevance_score"),
        "authors": authors,
        "journal": journal,
        "citation_count": raw.get("cited_by_count", 0),
    }


def _s2_with_oa_fallback(
    query: str,
    limit: int,
    s2_api_key: str,
    oa_api_key: str,
    s2_error: "Exception | None" = None,
) -> List[dict]:
    """Try Semantic Scholar (unless *s2_error* is already set), then OpenAlex.

    ``s2_error`` pre-populates the error from an earlier S2 attempt (e.g. from
    a circuit-breaker path that already failed).  When ``None`` a fresh S2
    attempt is made first.
    """
    if s2_error is None:
        try:
            rows = _s2_search(query, limit, s2_api_key)
            if rows:
                return rows
            logger.warning("Semantic Scholar returned zero results; falling back to OpenAlex")
        except Exception as exc:
            s2_error = exc
            logger.warning(
                "Semantic Scholar failed (%s); falling back to OpenAlex",
                type(exc).__name__,
            )
    oa_query = re.sub(r"\[[^\]]+\]", "", query)
    oa_query = re.sub(r"\b(?:title|abstract|author):", "", oa_query, flags=re.I)
    try:
        return _oa_search(" ".join(oa_query.split()), limit, oa_api_key)
    except Exception as oa_error:
        if s2_error is not None:
            raise RuntimeError(
                f"Semantic Scholar failed ({s2_error}); "
                f"OpenAlex fallback failed ({oa_error})"
            ) from oa_error
        raise RuntimeError(f"OpenAlex fallback failed: {oa_error}") from oa_error


# ============================================================================
# Dispatcher
# ============================================================================

def _search(query: str, limit: int = 20) -> List[dict]:
    """Dispatch search to the active backend.

    When Semantic Scholar is the primary backend but is persistently
    rate-limiting us (HTTP 429), a circuit-breaker opens and subsequent
    calls skip straight to OpenAlex for a cooldown period.  This avoids
    wasting 20+ seconds on doomed retries for every query in a batch.
    """
    if _BACKEND == "semantic_scholar":
        s2_error: Exception | None = None
        if _s2_circuit_open():
            logger.info("S2 circuit is open (rate-limited); using OpenAlex directly")
            s2_error = RuntimeError("S2 circuit open")
        else:
            try:
                rows = _s2_search(query, limit, _S2_API_KEY)
                if rows:
                    return rows
                logger.warning(
                    "Semantic Scholar returned zero results; falling back to OpenAlex"
                )
            except urllib.error.HTTPError as exc:
                s2_error = exc
                if exc.code == 429:
                    _s2_circuit_break()
                    logger.warning(
                        "Semantic Scholar rate-limited (429); "
                        "circuit-breaker open, falling back to OpenAlex"
                    )
                else:
                    logger.warning(
                        "Semantic Scholar failed (%s); falling back to OpenAlex",
                        type(exc).__name__,
                    )
            except Exception as exc:
                s2_error = exc
                logger.warning(
                    "Semantic Scholar failed (%s); falling back to OpenAlex",
                    type(exc).__name__,
                )
        return _s2_with_oa_fallback(
            query, limit, _S2_API_KEY, _OPENALEX_API_KEY, s2_error=s2_error
        )
    return _oa_search(query, limit, _OPENALEX_API_KEY)


def _fetch(identifier: str) -> dict:
    """Dispatch single-paper fetch to the active backend."""
    if _BACKEND == "semantic_scholar":
        return _s2_fetch(identifier, _S2_API_KEY)
    else:
        return _oa_fetch(identifier, _OPENALEX_API_KEY)


# ============================================================================
# SemanticScholarTool — implements ToolProtocol
# ============================================================================

@ToolRegistry.register
class SemanticScholarTool(ToolProtocol):
    """Search academic papers via Semantic Scholar or OpenAlex.

    Backend selection is automatic:
      - ``SEMANTIC_SCHOLAR_API_KEY`` set  →  Semantic Scholar (with key)
      - empty / not set                    →  OpenAlex (free, 10 req/s)

    Parameters
    ----------
    api_key : str
        Semantic Scholar API key (optional — uses env if empty).
    """

    tool_name = "semantic_scholar"
    tool_description = (
        "Search academic papers via Semantic Scholar / OpenAlex "
        "(auto-selects backend based on API key availability)"
    )

    def __init__(self, api_key: str = "", openalex_api_key: str = ""):
        self.api_key = api_key or _S2_API_KEY
        self.openalex_api_key = openalex_api_key or _OPENALEX_API_KEY
        # If caller explicitly chose openalex (no s2 key, but has openalex key),
        # route to OpenAlex. Otherwise default to semantic_scholar if we have a key.
        if not api_key and openalex_api_key:
            self._backend = "openalex"
        elif not api_key and not openalex_api_key:
            self._backend = "openalex" if not _S2_API_KEY else "semantic_scholar"
        else:
            self._backend = "semantic_scholar" if self.api_key else "openalex"

    @property
    def backend_name(self) -> str:
        return self._backend

    def _search_instance(self, query: str, limit: int) -> List[dict]:
        if (
            self.api_key == _S2_API_KEY
            and self.openalex_api_key == _OPENALEX_API_KEY
            and self._backend == _BACKEND
        ):
            return _search(query, limit)
        if self._backend == "semantic_scholar":
            return _s2_with_oa_fallback(
                query, limit, self.api_key, self.openalex_api_key
            )
        return _oa_search(query, limit, self.openalex_api_key)

    def _fetch_instance(self, identifier: str) -> dict:
        if self._backend == "semantic_scholar":
            return _s2_fetch(identifier, self.api_key)
        return _oa_fetch(identifier, self.openalex_api_key)

    async def search(self, query: str, limit: int = 20, **kwargs) -> List[dict]:
        """Run an academic literature search.

        Parameters
        ----------
        query : str
            Search query (supports phrase quotes, boolean operators).
        limit : int
            Max papers to return.

        Returns
        -------
        list[dict]
            Standardised fields: ``source``, ``paper_id``, ``title``,
            ``abstract``, ``year``, ``doi``, ``authors``, ``journal``,
            ``citation_count``.
        """
        try:
            return await asyncio.to_thread(
                self._search_instance, query, min(max(limit, 1), 200)
            )
        except Exception:
            logger.exception("Search failed (backend=%s, query=%r)", self._backend, query)
            return []

    async def search_strict(
        self,
        query: str,
        limit: int = 20,
        **kwargs,
    ) -> List[dict]:
        """Search in a worker thread and propagate backend failures."""
        bounded_limit = min(max(limit, 1), 200)
        return await asyncio.to_thread(self._search_instance, query, bounded_limit)

    async def fetch(self, identifier: str, **kwargs) -> dict:
        """Fetch a single paper by its backend-specific ID.

        Parameters
        ----------
        identifier : str
            Semantic Scholar ``paperId`` or OpenAlex work ID (full URL or bare ``W…``).

        Returns
        -------
        dict
            Standardised paper dict; returns ``{"paper_id": identifier}`` on failure.
        """
        try:
            return await asyncio.to_thread(self._fetch_instance, identifier)
        except Exception:
            logger.exception("Fetch failed (backend=%s, id=%s)", self._backend, identifier)
            return {"paper_id": identifier, "source": self._backend}


# ============================================================================
# Convenience — module-level functions (used by M2)
# ============================================================================

async def search_academic(query: str, limit: int = 20) -> List[dict]:
    """One-shot academic literature search via the active backend."""
    return await SemanticScholarTool().search(query, limit=limit)


async def fetch_paper(identifier: str) -> dict:
    """Fetch one paper by ID via the active backend."""
    return await SemanticScholarTool().fetch(identifier)
