"""
Academic literature search tool for Semantic Scholar.

OpenAlex is an independent source adapter. A failed or empty Semantic Scholar
query is reported as such and is not redirected to OpenAlex.
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
    # OpenAlex has its own source adapter.  Do not select it as a hidden
    # replacement for this Semantic Scholar source.
    return key, "semantic_scholar"


_S2_API_KEY, _BACKEND = _resolve_config()
_OPENALEX_API_KEY = os.environ.get("OPENALEX_API_KEY", "")
_OPENALEX_MAILTO = os.environ.get("OPENALEX_MAILTO", "")

# Module-level 429 circuit-breaker.  When S2 repeatedly rate-limits us,
# subsequent calls in the same process fail fast for a short cooldown.
# OpenAlex is a separate configured source, not a fallback from this module.
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
_OPENALEX_RATE_LIMIT = max(
    0.12,
    float(os.environ.get("OPENALEX_MIN_INTERVAL_SECONDS", "0.25")),
)
_last_request_time: float = 0.0
_rate_limit_lock = threading.Lock()

# Bypass the system proxy (e.g. Clash on 127.0.0.1:7897) for the API.
# A shared proxy exit IP can trip Semantic Scholar's per-IP limits.
_NO_PROXY_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

# Cap on the S2 stage of a search: rate-limit sleep + request + retries must
# fit inside the agent's per-source timeout (default 30.0s).
_S2_STAGE_DEADLINE_SECONDS = 15.0
_OPENALEX_STAGE_DEADLINE_SECONDS = 15.0


def _remaining_seconds(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Semantic Scholar request deadline exceeded")
    return remaining


def _sleep_with_deadline(seconds: float, deadline: float | None) -> None:
    if seconds <= 0:
        return
    if deadline is not None and seconds >= _remaining_seconds(deadline):
        raise TimeoutError("Semantic Scholar deadline would be exceeded")
    time.sleep(seconds)


def _rate_limit(min_interval: float, deadline: float | None = None) -> None:
    global _last_request_time
    with _rate_limit_lock:
        now = time.monotonic()
        gap = min_interval - (now - _last_request_time)
        if gap > 0:
            _sleep_with_deadline(gap, deadline)
        _last_request_time = time.monotonic()


def _http_get_json(
    url: str, *, s2_api_key: str = "", deadline: float | None = None
) -> dict:
    """GET *url*, parse JSON, with rate-limit + retry on transient errors.

    ``deadline`` (absolute ``time.monotonic()`` time) bounds the whole call:
    rate-limit waits, backoff sleeps and the socket timeout all respect it.
    Semantic Scholar HTTP 429 fails fast so its circuit breaker can engage.
    OpenAlex 429 is retried within the same bounded stage because concurrent
    evidence-gap searches can briefly exceed its burst allowance. Other
    providers are not called from this helper; source independence is handled
    by the search agent.
    """
    # The endpoint determines the provider; an absent key must not make an
    # S2 request look like an OpenAlex request or use OpenAlex's rate limit.
    is_s2 = "semanticscholar.org" in url
    if is_s2 and _s2_circuit_open():
        # A concurrent worker already tripped the breaker; don't queue
        # behind an 8s rate-limit sleep for a doomed request.
        raise RuntimeError("Semantic Scholar circuit is open (rate-limited)")
    request_interval = _S2_RATE_LIMIT if is_s2 else _OPENALEX_RATE_LIMIT
    last_exc = None
    for attempt in range(4):
        _rate_limit(request_interval, deadline)
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "HypoForge/0.1.0 (Academic Research)")
        if is_s2:
            req.add_header("x-api-key", s2_api_key)
        try:
            with _NO_PROXY_OPENER.open(
                req,
                timeout=min(30.0, _remaining_seconds(deadline))
                if deadline is not None else 30.0,
            ) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if deadline is not None:
                    _remaining_seconds(deadline)
                return data
        except urllib.error.HTTPError as e:
            last_exc = e
            if e.code == 429:
                if is_s2:
                    logger.debug(
                        "Semantic Scholar 429 (attempt %d); failing fast",
                        attempt,
                    )
                    raise
                if attempt < 3:
                    raw_retry_after = str(
                        (e.headers or {}).get("Retry-After", "")
                    ).strip()
                    retry_after: float | None = None
                    try:
                        if raw_retry_after:
                            retry_after = float(raw_retry_after)
                    except (TypeError, ValueError):
                        retry_after = None
                    wait = (
                        max(0.0, retry_after)
                        if retry_after is not None
                        else (2 ** attempt) + random.uniform(0, 0.25)
                    )
                    logger.warning(
                        "OpenAlex rate-limited one request; retrying in %.1fs "
                        "(attempt %d/4)",
                        wait,
                        attempt + 1,
                    )
                    _sleep_with_deadline(wait, deadline)
                    continue
            raise
        except (urllib.error.URLError, OSError) as e:
            last_exc = e
            if attempt < 3:
                base = 2 ** attempt
                wait = base + random.uniform(0, base * 0.25)
                logger.debug("Network error (%s), retry in %.1fs: %s", type(e).__name__, wait, e)
                _sleep_with_deadline(wait, deadline)
            else:
                raise

    raise last_exc  # type: ignore[misc]


# ============================================================================
# Backend: Semantic Scholar
# ============================================================================

_S2_BASE = "https://api.semanticscholar.org/graph/v1"


def _s2_search(
    query: str, limit: int, api_key: str = "", deadline: float | None = None
) -> List[dict]:
    """Search Semantic Scholar, return standardised paper dicts."""
    params: Dict[str, str] = {
        "query": query,
        "limit": str(min(limit, 100)),
        "fields": "title,year,authors,journal,externalIds,citationCount,abstract,openAccessPdf,isOpenAccess",
    }
    qs = urllib.parse.urlencode(params)
    url = f"{_S2_BASE}/paper/search?{qs}"

    data = _http_get_json(url, s2_api_key=api_key, deadline=deadline)
    papers = data.get("data", [])
    logger.info(
        "S2 search: query=%r → total=%s returned=%d",
        query, data.get("total", "?"), len(papers),
    )

    return [_s2_normalise(p) for p in papers]


def _s2_fetch(paper_id: str, api_key: str = "", deadline: float | None = None) -> dict:
    """Fetch a single paper from Semantic Scholar by ID."""
    fields = "title,abstract,year,authors,journal,externalIds,citationCount,openAccessPdf,isOpenAccess"
    url = f"{_S2_BASE}/paper/{paper_id}?fields={fields}"
    return _s2_normalise(
        _http_get_json(url, s2_api_key=api_key, deadline=deadline)
    )


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


def _normalize_openalex_query(query: str) -> str:
    """Convert backend-specific Boolean syntax to OpenAlex free text."""
    text = re.sub(r"\[[^\]]+\]", " ", str(query or ""))
    text = re.sub(r"\b(?:title|abstract|author):", " ", text, flags=re.I)
    text = re.sub(r"\b(?:AND|OR|NOT)\b", " ", text, flags=re.I)
    # OpenAlex interprets ``?`` and ``*`` as wildcard operators.  A normal
    # question ending in ``?`` therefore produces HTTP 400 under its default
    # stemmed search mode.  M2 sends free-text discovery queries, not exact
    # wildcard expressions, so remove both operators before URL encoding.
    text = re.sub(r"[?*]+", " ", text)
    text = re.sub(r'''[(){}\[\]"'“”‘’]+''', " ", text)
    return " ".join(text.split())


def _oa_search(
    query: str,
    limit: int,
    api_key: str = "",
    mailto: str = "",
    deadline: float | None = None,
    open_access_only: bool = False,
) -> List[dict]:
    """Search OpenAlex, return standardised paper dicts."""
    clean_query = _normalize_openalex_query(query)
    if not clean_query:
        raise ValueError("OpenAlex query must not be blank")
    params = {
        "search": clean_query,
        "per_page": str(min(limit, 200)),
        "sort": (
            "cited_by_count:desc" if open_access_only
            else "relevance_score:desc"
        ),
    }
    if open_access_only:
        params["filter"] = "is_oa:true"
    if api_key:
        params["api_key"] = api_key
    if mailto:
        params["mailto"] = mailto
    url = f"{_OA_BASE}/works?{urllib.parse.urlencode(params)}"
    data = _http_get_json(url, deadline=deadline)
    papers = data.get("results", [])
    logger.info(
        "OpenAlex search: query=%r → total=%s returned=%d",
        clean_query, data.get("meta", {}).get("count", "?"), len(papers),
    )
    return [_oa_normalise(p) for p in papers]


def _oa_fetch(
    work_id: str,
    api_key: str = "",
    mailto: str = "",
    deadline: float | None = None,
) -> dict:
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
    params: Dict[str, str] = {}
    if api_key:
        params["api_key"] = api_key
    if mailto:
        params["mailto"] = mailto
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    return _oa_normalise(_http_get_json(url, deadline=deadline))


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


# ============================================================================
# Dispatcher
# ============================================================================

def _search(query: str, limit: int = 20) -> List[dict]:
    """Dispatch search to the active backend.

    When Semantic Scholar is persistently rate-limiting us (HTTP 429), a
    circuit-breaker opens and subsequent calls fail fast for a cooldown period.
    OpenAlex is not invoked here; it has its own independent source adapter.
    """
    deadline = time.monotonic() + _S2_STAGE_DEADLINE_SECONDS
    if _s2_circuit_open():
        raise RuntimeError("Semantic Scholar circuit is open (rate-limited)")
    try:
        return _s2_search(query, limit, _S2_API_KEY, deadline=deadline)
    except urllib.error.HTTPError as exc:
        if exc.code == 429:
            _s2_circuit_break()
        raise
    except TimeoutError:
        _s2_circuit_break()
        raise


def _fetch(identifier: str) -> dict:
    """Dispatch single-paper fetch to the active backend."""
    deadline = time.monotonic() + _S2_STAGE_DEADLINE_SECONDS
    return _s2_fetch(identifier, _S2_API_KEY, deadline=deadline)


# ============================================================================
# SemanticScholarTool — implements ToolProtocol
# ============================================================================

@ToolRegistry.register
class SemanticScholarTool(ToolProtocol):
    """Search academic papers via Semantic Scholar only.

    Semantic Scholar is always the selected backend.  OpenAlex is exposed by
    a separate source adapter and is not selected as an error fallback.

    Parameters
    ----------
    api_key : str
        Semantic Scholar API key (optional — uses env if empty).
    """

    tool_name = "semantic_scholar"
    tool_description = "Search academic papers via Semantic Scholar."

    def __init__(self, api_key: str = "", openalex_api_key: str = ""):
        self.api_key = api_key or _S2_API_KEY
        # Kept as a compatibility-only argument for callers that used the old
        # dual-backend constructor. It is intentionally ignored.
        self._backend = "semantic_scholar"

    @property
    def backend_name(self) -> str:
        return self._backend

    def _search_instance(self, query: str, limit: int) -> List[dict]:
        if self.api_key == _S2_API_KEY and self._backend == _BACKEND:
            return _search(query, limit)
        deadline = time.monotonic() + _S2_STAGE_DEADLINE_SECONDS
        return _s2_search(query, limit, self.api_key, deadline=deadline)

    def _fetch_instance(self, identifier: str) -> dict:
        deadline = time.monotonic() + _S2_STAGE_DEADLINE_SECONDS
        return _s2_fetch(identifier, self.api_key, deadline=deadline)

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


async def search_openalex_strict(
    query: str,
    limit: int = 20,
    api_key: str = "",
    mailto: str = "",
    open_access_only: bool = False,
) -> List[dict]:
    """Search OpenAlex directly and propagate transport failures."""
    clean_query = _normalize_openalex_query(query)
    if not clean_query:
        raise ValueError("query must not be blank")
    bounded_limit = min(max(limit, 1), 200)
    resolved_key = str(api_key or "").strip() or _OPENALEX_API_KEY
    resolved_mailto = str(mailto or "").strip() or _OPENALEX_MAILTO
    deadline = time.monotonic() + _OPENALEX_STAGE_DEADLINE_SECONDS
    return await asyncio.to_thread(
        _oa_search,
        clean_query,
        bounded_limit,
        resolved_key,
        resolved_mailto,
        deadline,
        open_access_only,
    )


async def fetch_paper(identifier: str) -> dict:
    """Fetch one paper by ID via the active backend."""
    return await SemanticScholarTool().fetch(identifier)
