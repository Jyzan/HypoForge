"""Domain-specialist metadata sources used by routed M2 discovery.

These adapters only use documented public APIs.  They never scrape publisher
pages or treat an arbitrary search-result PDF as trusted full text.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from ..models import FulltextStatus, PaperRecord, SearchQuery
from ..protocols import LiteratureSourceProtocol


_CROSSREF_LOCK = threading.Lock()
_CROSSREF_NEXT_START = 0.0
logger = logging.getLogger(__name__)


class SpecialistSourceError(OSError):
    """A specialist metadata API failed or is not configured."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after_seconds = retry_after_seconds


def _json_get(
    url: str,
    *,
    timeout: float,
    headers: dict[str, str] | None = None,
) -> Any:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "HypoForge/0.1 (academic research)",
            **(headers or {}),
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read(500).decode("utf-8", errors="replace")
        retry_after: float | None = None
        retry_after_raw = (
            exc.headers.get("Retry-After", "") if exc.headers else ""
        )
        try:
            retry_after = float(retry_after_raw)
        except (TypeError, ValueError):
            pass
        raise SpecialistSourceError(
            f"HTTP {exc.code} from specialist source: {detail}",
            status_code=exc.code,
            retry_after_seconds=retry_after,
        ) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise SpecialistSourceError(str(exc)) from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise SpecialistSourceError(f"invalid JSON: {exc}") from exc


async def _get(url: str, timeout: float, headers: dict[str, str] | None = None) -> Any:
    return await asyncio.to_thread(_json_get, url, timeout=timeout, headers=headers)


def _title(value: Any) -> str:
    if isinstance(value, list):
        value = value[0] if value else ""
    if isinstance(value, dict):
        value = ": ".join(
            str(value.get(key) or "") for key in ("title", "subtitle", "addition")
            if str(value.get(key) or "").strip()
        )
    return " ".join(str(value or "").split())


def _year(value: Any) -> int | None:
    match = re.search(r"\b(?:19|20)\d{2}\b", str(value or ""))
    return int(match.group()) if match else None


def _authors(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    output: list[str] = []
    for value in values:
        if isinstance(value, dict):
            name = value.get("name") or value.get("full_name") or " ".join(
                filter(None, [value.get("given"), value.get("family")])
            )
        else:
            name = value
        normalized = " ".join(str(name or "").split())
        if normalized:
            output.append(normalized)
    return output


def _paper_id(prefix: str, title: str, doi: str = "", native: str = "") -> str:
    normalized_doi = PaperRecord.normalize_doi(doi)
    if normalized_doi:
        return f"DOI:{normalized_doi}"
    if native:
        return f"{prefix.upper()}:{native}"
    digest = hashlib.sha256(title.casefold().encode("utf-8")).hexdigest()[:20]
    return f"{prefix.upper()}-TITLE:{digest}"


class _BaseSpecialistSource(LiteratureSourceProtocol):
    source_name = "specialist"

    def __init__(self, *, timeout_seconds: float = 30.0) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.timeout_seconds = float(timeout_seconds)


class CrossrefSource(_BaseSpecialistSource):
    source_name = "crossref"

    def __init__(self, *, mailto: str = "", timeout_seconds: float = 30.0) -> None:
        super().__init__(timeout_seconds=timeout_seconds)
        self.mailto = str(
            mailto or os.environ.get("CROSSREF_MAILTO")
            or os.environ.get("UNPAYWALL_EMAIL", "")
        ).strip()

    @staticmethod
    def _wait_for_slot() -> None:
        global _CROSSREF_NEXT_START
        # One list request/second is safe in the anonymous public pool.
        interval = 1.0
        with _CROSSREF_LOCK:
            now = time.monotonic()
            scheduled = max(now, _CROSSREF_NEXT_START)
            _CROSSREF_NEXT_START = scheduled + interval
        delay = scheduled - time.monotonic()
        if delay > 0:
            time.sleep(delay)

    async def search(self, query: SearchQuery, limit: int = 20) -> list[PaperRecord]:
        params = {
            "query.bibliographic": query.text,
            "rows": str(min(max(limit, 1), 100)),
            "select": "DOI,title,author,published,container-title,is-referenced-by-count,abstract,type",
        }
        if self.mailto:
            params["mailto"] = self.mailto
        url = "https://api.crossref.org/works?" + urllib.parse.urlencode(params)

        async def request_once() -> Any:
            await asyncio.to_thread(self._wait_for_slot)
            return await _get(url, self.timeout_seconds)

        try:
            data = await request_once()
        except SpecialistSourceError as exc:
            if exc.status_code != 429:
                raise
            delay = (
                exc.retry_after_seconds
                if exc.retry_after_seconds is not None
                else 2.0
            )
            delay = min(10.0, max(0.0, delay))
            logger.warning(
                "Crossref rate-limited one query; retrying once after %.1fs",
                delay,
            )
            if delay:
                await asyncio.sleep(delay)
            data = await request_once()
        output: list[PaperRecord] = []
        for row in (data.get("message") or {}).get("items") or []:
            title = _title(row.get("title"))
            if not title:
                continue
            doi = PaperRecord.normalize_doi(row.get("DOI"))
            dates = ((row.get("published") or {}).get("date-parts") or [[]])[0]
            abstract = re.sub(r"<[^>]+>", " ", str(row.get("abstract") or ""))
            output.append(PaperRecord(
                paper_id=_paper_id(self.source_name, title, doi),
                title=title,
                abstract=" ".join(abstract.split()),
                authors=_authors(row.get("author")),
                year=int(dates[0]) if dates else None,
                journal="; ".join(row.get("container-title") or []),
                doi=doi,
                citation_count=int(row.get("is-referenced-by-count") or 0),
                publication_type=str(row.get("type") or ""),
                sources=[self.source_name],
                fulltext_status=(
                    FulltextStatus.ABSTRACT_ONLY if abstract else FulltextStatus.UNKNOWN
                ),
            ))
        return output


class EuropePMCSource(_BaseSpecialistSource):
    source_name = "europe_pmc"

    async def search(self, query: SearchQuery, limit: int = 20) -> list[PaperRecord]:
        params = urllib.parse.urlencode({
            "query": query.text,
            "format": "json",
            "pageSize": min(max(limit, 1), 100),
            "resultType": "core",
        })
        data = await _get(
            f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?{params}",
            self.timeout_seconds,
        )
        output: list[PaperRecord] = []
        for row in ((data.get("resultList") or {}).get("result") or []):
            title = _title(row.get("title"))
            if not title:
                continue
            doi = PaperRecord.normalize_doi(row.get("doi"))
            pmcid = str(row.get("pmcid") or "")
            abstract = str(row.get("abstractText") or "")
            output.append(PaperRecord(
                paper_id=_paper_id("epmc", title, doi, str(row.get("id") or "")),
                title=title,
                abstract=abstract,
                authors=[x.strip() for x in str(row.get("authorString") or "").split(",") if x.strip()],
                year=_year(row.get("pubYear")),
                journal=str(row.get("journalTitle") or ""),
                doi=doi,
                pmid=str(row.get("pmid") or ""),
                pmcid=pmcid,
                citation_count=int(row.get("citedByCount") or 0),
                publication_type=str(row.get("pubType") or ""),
                sources=[self.source_name],
                is_open_access=str(row.get("isOpenAccess") or "").upper() == "Y",
                fulltext_status=(
                    FulltextStatus.XML_AVAILABLE if pmcid
                    else FulltextStatus.ABSTRACT_ONLY if abstract
                    else FulltextStatus.UNKNOWN
                ),
            ))
        return output


class ZbMathSource(_BaseSpecialistSource):
    source_name = "zbmath"

    async def search(self, query: SearchQuery, limit: int = 20) -> list[PaperRecord]:
        params = urllib.parse.urlencode({
            "search_string": query.text,
            "results_per_page": min(max(limit, 1), 100),
        })
        try:
            data = await _get(
                f"https://api.zbmath.org/v1/document/_search?{params}",
                self.timeout_seconds,
            )
        except SpecialistSourceError as exc:
            if "HTTP 404" in str(exc) and (
                "No results found" in str(exc) or "Entry not found" in str(exc)
            ):
                return []
            raise
        output: list[PaperRecord] = []
        for row in data.get("result") or []:
            title = _title(row.get("title")) or "(Untitled)"
            links = row.get("links") or []
            doi = next(
                (str(item.get("identifier")) for item in links if item.get("type") == "doi"),
                "",
            )
            abstract = " ".join(
                str(item.get("text") or "")
                for item in row.get("editorial_contributions") or []
            )
            output.append(PaperRecord(
                paper_id=_paper_id("zbmath", title, doi, str(row.get("id") or row.get("identifier") or "")),
                title=title,
                abstract=abstract,
                authors=_authors((row.get("contributors") or {}).get("authors")),
                year=_year(row.get("publication_year") or row.get("year") or row.get("source")),
                journal=str(row.get("source") or ""),
                doi=doi,
                publication_type="mathematics",
                sources=[self.source_name],
                fulltext_status=(
                    FulltextStatus.ABSTRACT_ONLY if abstract else FulltextStatus.UNKNOWN
                ),
            ))
        return output


class InspireSource(_BaseSpecialistSource):
    source_name = "inspire"

    async def search(self, query: SearchQuery, limit: int = 20) -> list[PaperRecord]:
        params = urllib.parse.urlencode({
            "q": query.text,
            "size": min(max(limit, 1), 100),
            "sort": "mostcited",
        })
        data = await _get(
            f"https://inspirehep.net/api/literature?{params}", self.timeout_seconds
        )
        output: list[PaperRecord] = []
        for hit in ((data.get("hits") or {}).get("hits") or []):
            row = hit.get("metadata") or {}
            title = _title((row.get("titles") or [{}])[0]) or "(Untitled)"
            doi = PaperRecord.normalize_doi(((row.get("dois") or [{}])[0]).get("value"))
            arxiv = str(((row.get("arxiv_eprints") or [{}])[0]).get("value") or "")
            external_ids = {"inspire": str(hit.get("id") or "")}
            if arxiv:
                external_ids["arxiv"] = arxiv
            abstract = " ".join(str(item.get("value") or "") for item in row.get("abstracts") or [])
            output.append(PaperRecord(
                paper_id=_paper_id("inspire", title, doi, str(hit.get("id") or "")),
                title=title,
                abstract=abstract,
                authors=_authors(row.get("authors")),
                year=_year(row.get("earliest_date") or row.get("preprint_date")),
                doi=doi,
                external_ids=external_ids,
                citation_count=int(row.get("citation_count") or 0),
                publication_type="physics",
                sources=[self.source_name],
                is_open_access=True if arxiv else None,
                fulltext_status=(
                    FulltextStatus.PDF_AVAILABLE if arxiv
                    else FulltextStatus.ABSTRACT_ONLY if abstract
                    else FulltextStatus.UNKNOWN
                ),
            ))
        return output


class DblpSource(_BaseSpecialistSource):
    source_name = "dblp"

    async def search(self, query: SearchQuery, limit: int = 20) -> list[PaperRecord]:
        params = urllib.parse.urlencode({
            "q": query.text, "h": min(max(limit, 1), 100), "format": "json",
        })
        data = await _get(
            f"https://dblp.org/search/publ/api?{params}", self.timeout_seconds
        )
        hits = (((data.get("result") or {}).get("hits") or {}).get("hit") or [])
        output: list[PaperRecord] = []
        for hit in hits:
            row = hit.get("info") or {}
            title = re.sub(r"<[^>]+>", "", str(row.get("title") or "")).rstrip(".")
            if not title:
                continue
            author_rows = ((row.get("authors") or {}).get("author") or [])
            if isinstance(author_rows, dict):
                author_rows = [author_rows]
            doi = PaperRecord.normalize_doi(row.get("doi"))
            output.append(PaperRecord(
                paper_id=_paper_id("dblp", title, doi, str(row.get("key") or "")),
                title=title,
                authors=_authors(author_rows),
                year=_year(row.get("year")),
                journal=str(row.get("venue") or ""),
                doi=doi,
                publication_type=str(row.get("type") or "computer science"),
                sources=[self.source_name],
                is_open_access=str(row.get("access") or "") == "open",
            ))
        return output


class OstiSource(_BaseSpecialistSource):
    source_name = "osti"

    async def search(self, query: SearchQuery, limit: int = 20) -> list[PaperRecord]:
        params = urllib.parse.urlencode({"search": query.text, "rows": min(max(limit, 1), 100)})
        data = await _get(f"https://www.osti.gov/api/v1/records?{params}", self.timeout_seconds)
        output: list[PaperRecord] = []
        for row in data if isinstance(data, list) else []:
            title = _title(row.get("title")) or "(Untitled)"
            doi = PaperRecord.normalize_doi(row.get("doi"))
            external_ids = {"osti": str(row.get("osti_id") or "")}
            for link in row.get("links") or []:
                url = str(link.get("href") or link.get("url") or "") if isinstance(link, dict) else str(link)
                if url.casefold().endswith(".pdf"):
                    external_ids["oa_pdf_url"] = url
                    break
            output.append(PaperRecord(
                paper_id=_paper_id("osti", title, doi, str(row.get("osti_id") or "")),
                title=title,
                abstract=str(row.get("description") or ""),
                authors=_authors(row.get("authors")),
                year=_year(row.get("publication_date")),
                journal=str(row.get("journal_name") or row.get("publisher") or ""),
                doi=doi,
                external_ids=external_ids,
                publication_type=str(row.get("product_type") or "technical report"),
                sources=[self.source_name],
                is_open_access=True if external_ids.get("oa_pdf_url") else None,
                fulltext_status=(
                    FulltextStatus.PDF_AVAILABLE if external_ids.get("oa_pdf_url")
                    else FulltextStatus.ABSTRACT_ONLY
                ),
            ))
        return output


class NtrsSource(_BaseSpecialistSource):
    source_name = "ntrs"

    async def search(self, query: SearchQuery, limit: int = 20) -> list[PaperRecord]:
        params = urllib.parse.urlencode({
            "q": query.text,
            "page[size]": min(max(limit, 1), 100),
        })
        data = await _get(
            f"https://ntrs.nasa.gov/api/citations/search?{params}",
            self.timeout_seconds,
        )
        output: list[PaperRecord] = []
        for row in data.get("results") or data.get("items") or []:
            title = _title(row.get("title")) or "(Untitled)"
            native = str(row.get("id") or row.get("stiId") or "")
            doi = PaperRecord.normalize_doi(row.get("doi"))
            authors = [
                str(((item.get("meta") or {}).get("author") or {}).get("name") or "")
                for item in row.get("authorAffiliations") or []
                if str(((item.get("meta") or {}).get("author") or {}).get("name") or "").strip()
            ]
            pdf_url = ""
            for download in row.get("downloads") or []:
                links = download.get("links") or {}
                path = str(links.get("pdf") or links.get("original") or "")
                if path and str(download.get("mimetype") or "").casefold() == "application/pdf":
                    pdf_url = urllib.parse.urljoin("https://ntrs.nasa.gov", path)
                    break
            external_ids = {"ntrs": native} if native else {}
            if pdf_url:
                external_ids["oa_pdf_url"] = pdf_url
            output.append(PaperRecord(
                paper_id=_paper_id("ntrs", title, doi, native),
                title=title,
                abstract=str(row.get("abstract") or ""),
                authors=authors,
                year=_year(
                    row.get("distributionDate")
                    or row.get("publicationDate")
                    or row.get("publicationYear")
                ),
                doi=doi,
                external_ids=external_ids,
                publication_type="technical report",
                sources=[self.source_name],
                is_open_access=True if pdf_url else None,
                fulltext_status=(
                    FulltextStatus.PDF_AVAILABLE if pdf_url
                    else FulltextStatus.ABSTRACT_ONLY if row.get("abstract")
                    else FulltextStatus.UNKNOWN
                ),
            ))
        return output


class AdsSource(_BaseSpecialistSource):
    source_name = "ads"

    def __init__(self, *, api_token: str = "", timeout_seconds: float = 30.0) -> None:
        super().__init__(timeout_seconds=timeout_seconds)
        self.api_token = str(api_token or os.environ.get("ADS_API_TOKEN", "")).strip()

    async def search(self, query: SearchQuery, limit: int = 20) -> list[PaperRecord]:
        if not self.api_token:
            raise SpecialistSourceError("ADS_API_TOKEN is required for NASA ADS")
        params = urllib.parse.urlencode({
            "q": query.text,
            "rows": min(max(limit, 1), 100),
            "fl": "bibcode,title,abstract,author,year,pub,doi,citation_count,identifier",
            "sort": "citation_count desc",
        })
        data = await _get(
            f"https://api.adsabs.harvard.edu/v1/search/query?{params}",
            self.timeout_seconds,
            {"Authorization": f"Bearer {self.api_token}"},
        )
        output: list[PaperRecord] = []
        for row in (data.get("response") or {}).get("docs") or []:
            title = _title(row.get("title"))
            if not title:
                continue
            dois = row.get("doi") or []
            doi = PaperRecord.normalize_doi(dois[0] if isinstance(dois, list) and dois else dois)
            identifiers = [str(item) for item in row.get("identifier") or []]
            arxiv = next((item.split(":", 1)[1] for item in identifiers if item.casefold().startswith("arxiv:")), "")
            external_ids = {"ads": str(row.get("bibcode") or "")}
            if arxiv:
                external_ids["arxiv"] = arxiv
            output.append(PaperRecord(
                paper_id=_paper_id("ads", title, doi, str(row.get("bibcode") or "")),
                title=title,
                abstract=str(row.get("abstract") or ""),
                authors=_authors(row.get("author")),
                year=_year(row.get("year")),
                journal=str(row.get("pub") or ""),
                doi=doi,
                external_ids=external_ids,
                citation_count=int(row.get("citation_count") or 0),
                publication_type="astronomy",
                sources=[self.source_name],
                is_open_access=True if arxiv else None,
                fulltext_status=(
                    FulltextStatus.PDF_AVAILABLE if arxiv
                    else FulltextStatus.ABSTRACT_ONLY if row.get("abstract")
                    else FulltextStatus.UNKNOWN
                ),
            ))
        return output
