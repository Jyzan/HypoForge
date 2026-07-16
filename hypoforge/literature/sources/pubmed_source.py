"""PubMed literature source — ``LiteratureSourceProtocol`` adapter for
the legacy ``PubMedTool``.

The legacy tool is called unchanged; only input/output shapes are bridged.
"""

from __future__ import annotations

from typing import List, Optional

from hypoforge.literature.models import FulltextStatus, PaperRecord, SearchQuery
from hypoforge.literature.protocols import LiteratureSourceProtocol
from hypoforge.tools.pubmed_search import PubMedTool


def _dict_to_record(d: dict, source_name: str) -> PaperRecord:
    """Convert a legacy PubMed dict to a ``PaperRecord``."""
    pmid = (d.get("pmid") or "").strip()
    doi = (d.get("doi") or "").strip()
    title = (d.get("title") or "").strip()
    year = d.get("year") or 0

    if pmid:
        paper_id = f"PMID:{pmid}"
    elif doi:
        paper_id = f"DOI:{doi}"
    else:
        import hashlib
        slug = hashlib.sha256(title.encode()).hexdigest()[:16] if title else "unknown"
        paper_id = f"PUBMED:{slug}"

    if not title:
        title = "(Untitled)"

    return PaperRecord(
        paper_id=paper_id,
        title=title,
        abstract=(d.get("abstract") or "").strip(),
        authors=d.get("authors", []),
        year=year if year else None,
        journal=(d.get("journal") or "").strip(),
        doi=doi,
        pmid=pmid,
        pmcid="",
        external_ids={"pmid": pmid} if pmid else {},
        citation_count=d.get("citation_count", 0) or 0,
        publication_type="",
        sources=[source_name],
        fulltext_status=(
            FulltextStatus.UNKNOWN if (pmid or doi)
            else FulltextStatus.ABSTRACT_ONLY
        ),
    )


class PubMedSource(LiteratureSourceProtocol):
    """Search biomedical literature via the existing PubMed E-utilities tool.

    Implements ``LiteratureSourceProtocol`` for use with
    ``IterativeSearchAgent``.
    """

    source_name = "pubmed"

    def __init__(self, tool: Optional[PubMedTool] = None) -> None:
        """Optionally inject a stub tool for testing."""
        self._tool = tool if tool is not None else PubMedTool()

    async def search(
        self,
        query: SearchQuery,
        limit: int = 20,
    ) -> List[PaperRecord]:
        raw = await self._tool.search(query.text, limit=limit)
        return [_dict_to_record(p, self.source_name) for p in raw]
