"""arXiv adapter for the iterative literature-search source protocol."""

from __future__ import annotations

import asyncio
import hashlib
import re
from typing import Optional

from hypoforge.literature.models import FulltextStatus, PaperRecord, SearchQuery
from hypoforge.literature.protocols import LiteratureSourceProtocol
from hypoforge.tools.arxiv_search import ArxivTool


class ArxivBackendError(OSError):
    """Raised when the real arXiv backend cannot complete a search."""


_VERSION_SUFFIX = re.compile(r"v\d+$", re.IGNORECASE)


def canonical_arxiv_id(value: object) -> str:
    clean = str(value or "").strip()
    for marker in ("/abs/", "/pdf/"):
        if marker in clean:
            clean = clean.split(marker, 1)[1]
            break
    clean = clean.removesuffix(".pdf").strip("/")
    return _VERSION_SUFFIX.sub("", clean)


def _identifiable(row: dict) -> bool:
    return any(
        str(row.get(key) or "").strip()
        for key in ("arxiv_id", "doi", "title")
    )


def _external_ids(row: dict) -> dict[str, str]:
    output: dict[str, str] = {}
    for source_key, target_key in (
        ("arxiv_id", "arxiv"),
        ("entry_url", "entry_url"),
        ("pdf_url", "pdf_url"),
        ("primary_category", "primary_category"),
    ):
        value = str(row.get(source_key) or "").strip()
        if value:
            output[target_key] = value
    categories = row.get("categories")
    if isinstance(categories, (list, tuple)):
        joined = ",".join(
            value
            for item in categories
            if (value := str(item or "").strip())
        )
        if joined:
            output["categories"] = joined
    return output


def _dict_to_record(row: dict) -> PaperRecord:
    raw_arxiv_id = str(row.get("arxiv_id") or "").strip()
    arxiv_id = canonical_arxiv_id(raw_arxiv_id)
    doi = PaperRecord.normalize_doi(row.get("doi"))
    title = str(row.get("title") or "").strip() or "(Untitled)"
    abstract = str(row.get("abstract") or "").strip()
    pdf_url = str(row.get("pdf_url") or "").strip()
    if arxiv_id:
        paper_id = f"ARXIV:{arxiv_id}"
    elif doi:
        paper_id = f"DOI:{doi}"
    else:
        digest = hashlib.sha256(title.encode("utf-8")).hexdigest()[:16]
        paper_id = f"ARXIV-TITLE:{digest}"
    try:
        year = int(row.get("year")) if row.get("year") else None
    except (TypeError, ValueError):
        year = None
    if pdf_url:
        status = FulltextStatus.PDF_AVAILABLE
    elif abstract:
        status = FulltextStatus.ABSTRACT_ONLY
    else:
        status = FulltextStatus.UNKNOWN
    return PaperRecord(
        paper_id=paper_id,
        title=title,
        abstract=abstract,
        authors=list(row.get("authors") or []),
        year=year,
        journal=str(row.get("journal") or "").strip(),
        doi=doi,
        external_ids=_external_ids(row),
        citation_count=None,
        publication_type="preprint",
        sources=["arxiv"],
        is_open_access=True if pdf_url else None,
        fulltext_status=status,
    )


class ArxivSource(LiteratureSourceProtocol):
    """Search arXiv and return normalized metadata-only paper records."""

    source_name = "arxiv"

    def __init__(self, tool: Optional[ArxivTool] = None) -> None:
        self._tool = tool if tool is not None else ArxivTool()

    async def search(
        self,
        query: SearchQuery,
        limit: int = 20,
    ) -> list[PaperRecord]:
        strict_search = getattr(self._tool, "search_strict", None)
        search = strict_search if callable(strict_search) else self._tool.search
        try:
            rows = await search(query.text, limit=limit)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise ArxivBackendError(
                f"arxiv backend failed: {type(exc).__name__}: {exc}"
            ) from exc
        return [_dict_to_record(row) for row in rows if _identifiable(row)]
