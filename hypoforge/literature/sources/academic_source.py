"""Academic literature source — ``LiteratureSourceProtocol`` adapter for
the legacy ``SemanticScholarTool`` (Semantic Scholar / OpenAlex).

The legacy tool is called unchanged; only input/output shapes are bridged.
"""

from __future__ import annotations

from typing import List, Optional

from hypoforge.literature.models import FulltextStatus, PaperRecord, SearchQuery
from hypoforge.literature.protocols import LiteratureSourceProtocol
from hypoforge.tools.semantic_scholar import SemanticScholarTool


def _dict_to_record(d: dict, source_name: str) -> PaperRecord:
    """Convert a legacy S2 / OpenAlex dict to a ``PaperRecord``."""
    doi = (d.get("doi") or "").strip()
    title = (d.get("title") or "").strip()
    legacy_source = (d.get("source") or "").strip()
    legacy_paper_id = (d.get("paper_id") or "").strip()
    year = d.get("year") or 0

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
        abstract=(d.get("abstract") or "").strip(),
        authors=d.get("authors", []),
        year=year if year else None,
        journal=(d.get("journal") or "").strip(),
        doi=doi,
        pmid="",
        pmcid="",
        external_ids={},
        citation_count=d.get("citation_count", 0) or 0,
        publication_type="",
        sources=[source_name],
        fulltext_status=(
            FulltextStatus.UNKNOWN if doi
            else FulltextStatus.ABSTRACT_ONLY
        ),
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

    async def search(
        self,
        query: SearchQuery,
        limit: int = 20,
    ) -> List[PaperRecord]:
        raw = await self._tool.search(query.text, limit=limit)
        return [_dict_to_record(p, self.source_name) for p in raw]
