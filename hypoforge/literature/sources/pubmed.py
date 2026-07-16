from __future__ import annotations

import hashlib
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

from ...tools.pubmed_search import search_pubmed_strict
from ..models import FulltextStatus, PaperRecord, SearchQuery
from ..protocols import LiteratureSourceProtocol

PubMedBackend = Callable[
    [str, int],
    Awaitable[Sequence[Mapping[str, Any]]],
]


def _normalized_title(value: object) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).split())


def _paper_id(pmid: str, doi: str, title: str) -> str:
    if pmid:
        return f"PMID:{pmid}"
    if doi:
        return f"DOI:{doi}"
    digest = hashlib.sha256(_normalized_title(title).encode("utf-8")).hexdigest()[:16]
    return f"TITLE:{digest}"


def _to_paper_record(raw: Mapping[str, Any]) -> PaperRecord | None:
    pmid = str(raw.get("pmid") or "").strip()
    doi = PaperRecord.normalize_doi(raw.get("doi"))
    title = str(raw.get("title") or "").strip()
    if not pmid and not doi and not title:
        return None
    abstract = str(raw.get("abstract") or "").strip()
    year_value = raw.get("year")
    try:
        parsed_year = int(year_value) if year_value else 0
    except (TypeError, ValueError):
        parsed_year = 0
    year = parsed_year if 1000 <= parsed_year <= 3000 else None
    return PaperRecord(
        paper_id=_paper_id(pmid, doi, title),
        title=title or doi or f"PubMed {pmid}",
        abstract=abstract,
        authors=[str(item) for item in raw.get("authors") or [] if str(item).strip()],
        year=year,
        journal=str(raw.get("journal") or "").strip(),
        doi=doi,
        pmid=pmid,
        sources=["pubmed"],
        fulltext_status=(
            FulltextStatus.ABSTRACT_ONLY if abstract else FulltextStatus.UNKNOWN
        ),
    )


class PubMedLiteratureSource(LiteratureSourceProtocol):
    source_name = "pubmed"

    def __init__(self, backend: PubMedBackend | None = None) -> None:
        self.backend = backend or search_pubmed_strict

    async def search(self, query: SearchQuery, limit: int = 20) -> list[PaperRecord]:
        rows = await self.backend(query.text, limit)
        records = (_to_paper_record(row) for row in rows)
        return [record for record in records if record is not None]
