"""Academic literature source — ``LiteratureSourceProtocol`` adapter for
the legacy ``SemanticScholarTool`` (Semantic Scholar / OpenAlex).

The legacy tool is called unchanged; only input/output shapes are bridged.
"""

from __future__ import annotations

import math
from typing import List, Optional

from hypoforge.modules.m2_literature.models import FulltextStatus, PaperRecord, SearchQuery
from hypoforge.modules.m2_literature.protocols import LiteratureSourceProtocol
from hypoforge.tools.semantic_scholar import SemanticScholarTool


class AcademicBackendError(OSError):
    """Academic backend failure with the resolved provider in its message."""


def _dict_to_record(d: dict, source_name: str) -> PaperRecord:
    """Convert a legacy S2 / OpenAlex dict to a ``PaperRecord``."""
    doi = PaperRecord.normalize_doi(d.get("doi"))
    title = (d.get("title") or "").strip()
    abstract = str(d.get("abstract") or "").strip()
    legacy_source = (d.get("source") or "").strip()
    legacy_paper_id = (d.get("paper_id") or "").strip()
    year = d.get("year") or 0
    external_ids = {
        str(key): str(value)
        for key, value in (d.get("external_ids") or {}).items()
        if value not in (None, "")
    }
    pmid = str(d.get("pmid") or external_ids.get("PubMed") or "").strip()
    pmcid = str(
        d.get("pmcid") or external_ids.get("PubMedCentral") or ""
    ).strip()
    oa_pdf_url = str(d.get("oa_pdf_url") or "").strip()
    if oa_pdf_url:
        external_ids["oa_pdf_url"] = oa_pdf_url
    raw_native_relevance = d.get("retrieval_relevance")
    rank_scores = {}
    if (
        isinstance(raw_native_relevance, (int, float))
        and math.isfinite(float(raw_native_relevance))
    ):
        rank_scores["source_native_relevance"] = float(raw_native_relevance)

    if legacy_paper_id:
        if legacy_source == "semantic_scholar":
            paper_id = f"S2:{legacy_paper_id}"
        elif legacy_source == "openalex":
            short = (
                legacy_paper_id.rsplit("/", 1)[-1]
                if "/" in legacy_paper_id
                else legacy_paper_id
            )
            paper_id = f"OA:{short}"
        else:
            paper_id = f"{legacy_source.upper()}:{legacy_paper_id}"
    elif doi:
        paper_id = f"DOI:{doi}"
    else:
        import hashlib
        slug = hashlib.sha256(title.encode()).hexdigest()[:16] if title else "unknown"
        paper_id = f"ACADEMIC:{slug}"

    if not title:
        title = "(Untitled)"

    return PaperRecord(
        paper_id=paper_id,
        title=title,
        abstract=abstract,
        authors=d.get("authors", []),
        year=year if year else None,
        journal=(d.get("journal") or "").strip(),
        doi=doi,
        pmid=pmid,
        pmcid=pmcid,
        external_ids=external_ids,
        citation_count=d.get("citation_count", 0) or 0,
        publication_type="",
        sources=[legacy_source or source_name],
        is_open_access=(
            bool(d.get("is_open_access")) if d.get("is_open_access") is not None
            else None
        ),
        fulltext_status=(
            FulltextStatus.PDF_AVAILABLE if oa_pdf_url
            else FulltextStatus.XML_AVAILABLE if pmcid
            else FulltextStatus.ABSTRACT_ONLY if abstract
            else FulltextStatus.UNKNOWN
        ),
        rank_scores=rank_scores,
    )


class AcademicSource(LiteratureSourceProtocol):
    """Search academic literature via Semantic Scholar or OpenAlex.

    Implements ``LiteratureSourceProtocol`` for use with
    ``IterativeSearchAgent``.
    """

    source_name = "semantic_scholar"

    def __init__(self, tool: Optional[SemanticScholarTool] = None) -> None:
        """Optionally inject a stub tool for testing."""
        self._tool = tool if tool is not None else SemanticScholarTool()

    @property
    def backend_name(self) -> str:
        return str(getattr(self._tool, "backend_name", self.source_name))

    async def search(
        self,
        query: SearchQuery,
        limit: int = 20,
    ) -> List[PaperRecord]:
        strict_search = getattr(self._tool, "search_strict", None)
        search = strict_search if callable(strict_search) else self._tool.search
        try:
            raw = await search(query.text, limit=limit)
        except Exception as exc:
            raise AcademicBackendError(
                f"{self.backend_name} backend failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        records = []
        for row in raw:
            if not any(
                str(row.get(key) or "").strip()
                for key in ("paper_id", "doi", "title")
            ):
                continue
            records.append(_dict_to_record(row, self.source_name))
        return records
