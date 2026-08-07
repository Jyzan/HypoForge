"""PubMed literature source — ``LiteratureSourceProtocol`` adapter for
the legacy ``PubMedTool``.

The legacy tool is called unchanged; only input/output shapes are bridged.
"""

from __future__ import annotations

from typing import List, Optional

from hypoforge.literature.models import FulltextStatus, PaperRecord, SearchQuery
from hypoforge.literature.protocols import LiteratureSourceProtocol
from hypoforge.tools.pubmed_search import PubMedTool

from .pubmed import (
    MAX_RELAXATION_PROBES,
    PubMedCountProbe,
    _default_count_probe,
    run_zero_result_relaxation,
)


def _dict_to_record(d: dict, source_name: str) -> PaperRecord:
    """Convert a legacy PubMed dict to a ``PaperRecord``."""
    pmid = (d.get("pmid") or "").strip()
    doi = PaperRecord.normalize_doi(d.get("doi"))
    title = (d.get("title") or "").strip()
    abstract = str(d.get("abstract") or "").strip()
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
        abstract=abstract,
        authors=d.get("authors", []),
        year=year if year else None,
        journal=(d.get("journal") or "").strip(),
        doi=doi,
        pmid=pmid,
        pmcid="",
        external_ids={"pmid": pmid} if pmid else {},
        citation_count=d.get("citation_count"),
        publication_type="",
        sources=[source_name],
        fulltext_status=(
            FulltextStatus.ABSTRACT_ONLY if abstract
            else FulltextStatus.UNKNOWN
        ),
    )


class PubMedSource(LiteratureSourceProtocol):
    """Search biomedical literature via the existing PubMed E-utilities tool.

    Implements ``LiteratureSourceProtocol`` for use with
    ``IterativeSearchAgent``.
    """

    source_name = "pubmed"

    def __init__(
        self,
        tool: Optional[PubMedTool] = None,
        *,
        enable_relaxation: bool = True,
        count_probe: PubMedCountProbe | None = None,
        max_relaxation_probes: int = MAX_RELAXATION_PROBES,
    ) -> None:
        """Optionally inject a stub tool for testing."""
        if max_relaxation_probes <= 0:
            raise ValueError("max_relaxation_probes must be positive")
        self._uses_default_tool = tool is None
        self._tool = tool if tool is not None else PubMedTool()
        self.enable_relaxation = enable_relaxation
        self.count_probe = count_probe
        self.max_relaxation_probes = max_relaxation_probes

    def _resolve_count_probe(self) -> PubMedCountProbe | None:
        if self.count_probe is not None:
            return self.count_probe
        # Only probe NCBI for real when the real tool is in use; injected
        # test tools fall back to probe-free (blind) relaxation.
        return _default_count_probe if self._uses_default_tool else None

    async def search(
        self,
        query: SearchQuery,
        limit: int = 20,
    ) -> List[PaperRecord]:
        strict_search = getattr(self._tool, "search_strict", None)
        search = strict_search if callable(strict_search) else self._tool.search

        async def fetch(text: str):
            return await search(text, limit=limit)

        raw = list(await fetch(query.text))
        relaxed_from = ""
        if not raw and self.enable_relaxation:
            raw, relaxed_from = await run_zero_result_relaxation(
                query.text,
                fetch,
                count_probe=self._resolve_count_probe(),
                max_probes=self.max_relaxation_probes,
            )
        records = []
        for row in raw:
            if not any(
                str(row.get(key) or "").strip()
                for key in ("pmid", "doi", "title")
            ):
                continue
            record = _dict_to_record(row, self.source_name)
            if relaxed_from:
                record.relaxed_from = relaxed_from
            records.append(record)
        return records
