"""
PubMed E-utilities API tool — real implementation.

NCBI E-utilities: ``esearch`` → ``efetch`` two-step workflow to retrieve
PubMed articles with full metadata (title, abstract, authors, journal, …).

Register an API key at https://ncbi.nlm.nih.gov/account/settings/ to raise
the rate limit from 3 req/s to 10 req/s.  Set it via environment variable
``NCBI_API_KEY`` (and optionally ``NCBI_BASE_URL`` to override the endpoint).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from ..protocol import ToolProtocol
from ..registry import ToolRegistry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Resolve API key + base URL from environment (once at import time)
# ---------------------------------------------------------------------------

def _resolve_config() -> tuple[str, str]:
    """Load ``.env`` and return (api_key, base_url)."""
    from pathlib import Path

    candidates = [
        Path(__file__).resolve().parent.parent.parent / ".env",  # hypoforge/tools/ → repo root
        Path.cwd() / ".env",
    ]
    for p in candidates:
        if p.exists():
            load_dotenv(p, override=True)
            break
    else:
        load_dotenv(override=True)

    key = os.environ.get("NCBI_API_KEY", "")
    base = os.environ.get("NCBI_BASE_URL", "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/")
    return key, base


_NCBI_API_KEY, _NCBI_BASE_URL = _resolve_config()

# Rate-limit padding (seconds between requests) — 0.34 s for keyless (3/s),
# 0.11 s with key (10/s), plus a small safety margin.
_RATE_LIMIT = 0.12 if _NCBI_API_KEY else 0.35
_last_request_time: float = 0.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rate_limit() -> None:
    """Sleep if we are calling the API faster than the rate limit allows."""
    global _last_request_time
    now = time.monotonic()
    gap = _RATE_LIMIT - (now - _last_request_time)
    if gap > 0:
        time.sleep(gap)
    _last_request_time = time.monotonic()


def _build_url(endpoint: str, params: Dict[str, str]) -> str:
    """Construct a full E-utilities URL."""
    if _NCBI_API_KEY:
        params["api_key"] = _NCBI_API_KEY
    return _NCBI_BASE_URL + endpoint + "?" + urllib.parse.urlencode(params)


def _http_get_json(url: str) -> dict:
    """GET *url*, parse response as JSON, with retry on transient errors."""
    last_exc = None
    for attempt in range(4):
        _rate_limit()
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            last_exc = e
            if e.code == 429 and attempt < 3:
                wait = 2 ** attempt
                logger.debug("NCBI 429 rate-limited, retry in %ds", wait)
                time.sleep(wait)
            else:
                raise
        except (urllib.error.URLError, OSError) as e:
            last_exc = e
            if attempt < 3:
                wait = 2 ** attempt
                logger.debug("NCBI network error (%s), retry in %ds: %s", type(e).__name__, wait, e)
                time.sleep(wait)
            else:
                raise
    raise last_exc  # type: ignore[misc]


def _http_get_xml_text(url: str) -> str:
    """GET *url*, return raw XML text, with retry on transient errors."""
    last_exc = None
    for attempt in range(4):
        _rate_limit()
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                return resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            last_exc = e
            if e.code == 429 and attempt < 3:
                wait = 2 ** attempt
                logger.debug("NCBI 429 rate-limited, retry in %ds", wait)
                time.sleep(wait)
            else:
                raise
        except (urllib.error.URLError, OSError) as e:
            last_exc = e
            if attempt < 3:
                wait = 2 ** attempt
                logger.debug("NCBI network error (%s), retry in %ds: %s", type(e).__name__, wait, e)
                time.sleep(wait)
            else:
                raise
    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# XML → dict parsing (efetch PubmedArticleSet)
# ---------------------------------------------------------------------------

def _parse_pubmed_article(article_elem: ET.Element) -> Dict[str, Any]:
    """Extract a standardised dict from a ``<PubmedArticle>`` element."""

    # ---- pmid ----
    pmid_elem = article_elem.find(".//PMID")
    pmid = pmid_elem.text if pmid_elem is not None else ""

    # ---- title ----
    title_elem = article_elem.find(".//ArticleTitle")
    title = "".join(title_elem.itertext()) if title_elem is not None else ""

    # ---- abstract ----
    abstract_parts = []
    for abs_elem in article_elem.findall(".//AbstractText"):
        label = abs_elem.get("Label", "")
        text = "".join(abs_elem.itertext())
        if label:
            abstract_parts.append(f"[{label}] {text}")
        else:
            abstract_parts.append(text)
    abstract = " ".join(abstract_parts)

    # ---- year ----
    year_elem = article_elem.find(".//PubDate/Year")
    # Fall back to MedlineDate parsing if Year is missing
    if year_elem is not None and year_elem.text:
        year = int(year_elem.text)
    else:
        medline_elem = article_elem.find(".//PubDate/MedlineDate")
        if medline_elem is not None and medline_elem.text:
            # MedlineDate is often "2024 Jan-Feb" or "2023-2024" — grab first 4 digits
            import re
            match = re.search(r"(\d{4})", medline_elem.text)
            year = int(match.group(1)) if match else 0
        else:
            year = 0

    # ---- journal ----
    journal_elem = article_elem.find(".//Journal/Title")
    journal = journal_elem.text if journal_elem is not None else ""

    # ---- authors ----
    authors: List[str] = []
    for author_elem in article_elem.findall(".//Author"):
        last = author_elem.findtext("LastName") or ""
        fore = author_elem.findtext("ForeName") or ""
        if last:
            authors.append(f"{last} {fore}".strip())

    # ---- doi ----
    doi_elem = article_elem.find(".//ArticleId[@IdType='doi']")
    doi = doi_elem.text if doi_elem is not None else ""

    return {
        "pmid": pmid,
        "title": title,
        "abstract": abstract,
        "year": year,
        "journal": journal,
        "authors": authors,
        "doi": doi,
    }


def _esearch(query: str, limit: int = 20) -> List[str]:
    """Run an E-utilities ``esearch`` and return a list of PMIDs."""
    url = _build_url("esearch.fcgi", {
        "db": "pubmed",
        "term": query,
        "retmax": str(limit),
        "retmode": "json",
        "sort": "relevance",
    })
    logger.debug("esearch: %s", url[:200])
    data = _http_get_json(url)
    id_list = data.get("esearchresult", {}).get("idlist", [])
    count = int(data.get("esearchresult", {}).get("count", 0))
    logger.info("esearch: query=%r → total=%s returned=%d", query, count, len(id_list))
    return id_list


def _efetch_batch(pmids: List[str]) -> List[Dict[str, Any]]:
    """Fetch full metadata for a batch of PMIDs via ``efetch`` (XML)."""
    if not pmids:
        return []

    url = _build_url("efetch.fcgi", {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
    })
    logger.debug("efetch: %d IDs", len(pmids))
    xml_text = _http_get_xml_text(url)

    root = ET.fromstring(xml_text)
    articles = []
    for article_elem in root.findall(".//PubmedArticle"):
        articles.append(_parse_pubmed_article(article_elem))

    return articles


# ============================================================================
# PubMedTool — implements ToolProtocol
# ============================================================================

@ToolRegistry.register
class PubMedTool(ToolProtocol):
    """Search biomedical literature via PubMed E-utilities API.

    Two-step workflow: ``esearch`` to get PMIDs, then ``efetch`` to get full
    metadata for each paper.

    Parameters
    ----------
    api_key : str
        NCBI API key (optional — reads ``NCBI_API_KEY`` from env if empty).
    api_base : str
        Override the E-utilities base URL (reads ``NCBI_BASE_URL`` from env).
    """

    tool_name = "pubmed"
    tool_description = "Search biomedical literature via PubMed E-utilities API"

    def __init__(self, api_key: str = "", api_base: str = ""):
        self.api_key = api_key or _NCBI_API_KEY
        self.api_base = api_base or _NCBI_BASE_URL

    # ------------------------------------------------------------------
    # ToolProtocol implementation
    # ------------------------------------------------------------------

    async def search(self, query: str, limit: int = 20, **kwargs) -> List[dict]:
        """Run a PubMed search and return standardised paper dicts.

        Parameters
        ----------
        query : str
            PubMed search query (supports full PubMed query syntax).
        limit : int
            Maximum number of papers to return (default 20, max 100 per call).

        Returns
        -------
        list[dict]
            Each dict: pmid, title, abstract, year, journal, authors, doi.
        """
        limit = min(max(limit, 1), 100)  # E-utilities caps retmax at ~10 000 but we keep it practical

        try:
            pmids = _esearch(query, limit=limit)
        except Exception:
            logger.exception("esearch failed for query=%r", query)
            return []

        if not pmids:
            return []

        try:
            articles = _efetch_batch(pmids)
        except Exception:
            logger.exception("efetch failed for %d PMIDs", len(pmids))
            return []

        return articles

    async def fetch(self, identifier: str, **kwargs) -> dict:
        """Fetch a single paper by PMID.

        Parameters
        ----------
        identifier : str
            A PubMed ID (PMID), e.g. ``"31253954"``.

        Returns
        -------
        dict
            pmid, title, abstract, year, journal, authors, doi.
            Returns an empty dict with ``pmid`` set if the fetch fails.
        """
        try:
            articles = _efetch_batch([identifier])
        except Exception:
            logger.exception("efetch failed for PMID=%s", identifier)
            return {"pmid": identifier}

        if articles:
            return articles[0]
        return {"pmid": identifier}


# ============================================================================
# Convenience — module-level functions (used by M2)
# ============================================================================

def _search_pubmed_strict_sync(query: str, limit: int) -> List[dict]:
    bounded_limit = min(max(limit, 1), 100)
    pmids = _esearch(query, limit=bounded_limit)
    return _efetch_batch(pmids)


async def search_pubmed_strict(query: str, limit: int = 20) -> List[dict]:
    """Search PubMed without swallowing failures or blocking the event loop."""
    return await asyncio.to_thread(_search_pubmed_strict_sync, query, limit)


async def search_pubmed(query: str, limit: int = 20) -> List[dict]:
    """Convenience: one-shot PubMed search (async)."""
    tool = PubMedTool()
    return await tool.search(query, limit=limit)


async def fetch_pubmed_article(pmid: str) -> dict:
    """Convenience: fetch one paper by PMID (async)."""
    tool = PubMedTool()
    return await tool.fetch(pmid)
