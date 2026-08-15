"""Unified literature search across biomedical and technical sources.

Provides ``tool_definitions`` for QueryPlanner and a single search entry
point that dispatches to the appropriate backend.

For use with ``IterativeSearchAgent``, prefer the individual source
implementations in ``hypoforge.literature.sources``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Dict, List, Optional

from hypoforge.literature.models import PaperRecord, SearchQuery
from hypoforge.literature.sources.academic_source import AcademicSource
from hypoforge.literature.sources.arxiv_source import ArxivSource
from hypoforge.literature.sources.openalex_source import (
    OpenAlexOpenAccessSource,
    OpenAlexSource,
)
from hypoforge.literature.sources.openalex_enriched_scholar import (
    OpenAlexEnrichedScholarSource,
)
from hypoforge.literature.sources.pubmed_source import PubMedSource
from hypoforge.literature.sources.specialist_sources import (
    AdsSource,
    CrossrefSource,
    DblpSource,
    EuropePMCSource,
    InspireSource,
    NtrsSource,
    OstiSource,
    ZbMathSource,
)
from hypoforge.tools.semantic_scholar import SemanticScholarTool


class LiteratureSearchTool:
    """Unified search across PubMed, Semantic Scholar, OpenAlex, and arXiv.

    Usage::

        tool = LiteratureSearchTool()
        defs = tool.tool_definitions      # → pass to QueryPlanner
        records = await tool.search("Hsp70 ATPase", backend="pubmed")
        results = await tool.search_multi(queries)   # parallel multi-source
    """

    TOOL_DEFINITIONS: List[Dict[str, Any]] = [
        {
            "name": "pubmed",
            "display_name": "PubMed",
            "description": (
                "Search biomedical and life sciences literature via PubMed. "
                "BEST for: clinical medicine, molecular biology, genetics, "
                "pharmacology, biochemistry, and all biomedical topics. "
                "Supports MeSH terms, field tags ([tiab], [MeSH], [au]), "
                "and full PubMed query syntax (AND, OR, NOT, parentheses). "
                "NOT suitable for: computer science, engineering, physics, "
                "or social sciences queries."
            ),
        },
        {
            "name": "semantic_scholar",
            "display_name": "Semantic Scholar",
            "description": (
                "Search academic literature across ALL disciplines via "
                "Semantic Scholar. "
                "BEST for: computer science, engineering, physics, social "
                "sciences, mathematics, and interdisciplinary research. "
                "Also works for broad biomedical queries that may appear "
                "in non-PubMed journals. "
                "NOT ideal for: specialized biomedical/clinical queries "
                "requiring MeSH terms (use PubMed instead)."
            ),
        },
        {
            "name": "openalex",
            "display_name": "OpenAlex",
            "description": (
                "Search the OpenAlex scholarly catalog across ALL disciplines "
                "with plain keyword queries. BEST for broad cross-disciplinary "
                "recall, citation-rich metadata, and works missing from PubMed "
                "or Semantic Scholar."
            ),
        },
        {
            "name": "openalex_oa",
            "display_name": "OpenAlex Open Access",
            "description": (
                "Conditional open-access discovery sorted by citation count."
            ),
        },
        {
            "name": "arxiv",
            "display_name": "arXiv",
            "description": (
                "Search recent preprints in computer science, mathematics, "
                "physics, statistics, electrical engineering, quantitative "
                "biology, quantitative finance, and economics. BEST for: "
                "recent technical methods and openly accessible preprints. "
                "Results may not have completed peer review. NOT ideal for: "
                "specialized clinical or biomedical queries requiring MeSH "
                "terms (use PubMed instead)."
            ),
        },
        {
            "name": "serper_openalex",
            "display_name": "Serper Scholar → OpenAlex",
            "description": (
                "Mandatory cross-domain Google Scholar discovery through "
                "Serper, followed by strict DOI/title enrichment in OpenAlex."
            ),
        },
        {"name": "crossref", "display_name": "Crossref", "description": "Cross-domain DOI metadata and citation counts."},
        {"name": "europe_pmc", "display_name": "Europe PMC", "description": "Biomedical metadata, citation counts, PMCID and open full text."},
        {"name": "zbmath", "display_name": "zbMATH Open", "description": "Mathematics-focused bibliographic discovery."},
        {"name": "inspire", "display_name": "INSPIRE-HEP", "description": "High-energy physics literature and arXiv identifiers."},
        {"name": "ads", "display_name": "NASA ADS", "description": "Astronomy and astrophysics literature sorted by citations."},
        {"name": "dblp", "display_name": "DBLP", "description": "Computer-science bibliographic discovery."},
        {"name": "osti", "display_name": "DOE OSTI", "description": "Energy research papers and technical reports."},
        {"name": "ntrs", "display_name": "NASA NTRS", "description": "NASA aerospace technical reports."},
    ]

    _TOOL_NAME_SET: set[str] = {d["name"] for d in TOOL_DEFINITIONS}

    def __init__(
        self,
        pubmed: Optional[PubMedSource] = None,
        academic: Optional[AcademicSource] = None,
        arxiv_source: Optional[ArxivSource] = None,
        openalex_source: Optional[OpenAlexSource] = None,
        enabled_sources: Sequence[str] | None = None,
        semantic_scholar_api_key: str = "",
        openalex_api_key: str = "",
        openalex_mailto: str = "",
        serper_api_key: str = "",
        ads_api_token: str = "",
        crossref_mailto: str = "",
        zero_result_relaxation: bool = True,
    ):
        requested = {
            str(name).strip().casefold()
            for name in (enabled_sources or self._TOOL_NAME_SET)
        }
        unknown = requested - self._TOOL_NAME_SET
        if unknown:
            raise ValueError(f"Unknown literature sources: {sorted(unknown)}")
        if not requested:
            raise ValueError("At least one literature source must be enabled")
        self._enabled_sources = requested
        self._pubmed_source = (
            pubmed
            if pubmed is not None
            else PubMedSource(enable_relaxation=zero_result_relaxation)
        )
        self._academic_source = (
            academic
            if academic is not None
            else AcademicSource(
                tool=SemanticScholarTool(api_key=semantic_scholar_api_key)
            )
        )
        self._arxiv_source = (
            arxiv_source if arxiv_source is not None else ArxivSource()
        )
        self._openalex_source = (
            openalex_source
            if openalex_source is not None
            else OpenAlexSource(api_key=openalex_api_key, mailto=openalex_mailto)
        )
        self._source_map: dict[str, object] = {
            "pubmed": self._pubmed_source,
            "semantic_scholar": self._academic_source,
            "openalex": self._openalex_source,
            "arxiv": self._arxiv_source,
        }
        if "openalex_oa" in requested:
            self._source_map["openalex_oa"] = OpenAlexOpenAccessSource(
                api_key=openalex_api_key,
                mailto=openalex_mailto,
            )
        if "serper_openalex" in requested:
            self._source_map["serper_openalex"] = OpenAlexEnrichedScholarSource(
                serper_api_key=serper_api_key,
                openalex_api_key=openalex_api_key,
                openalex_mailto=openalex_mailto,
            )
        specialist_factories = {
            "crossref": lambda: CrossrefSource(mailto=crossref_mailto),
            "europe_pmc": EuropePMCSource,
            "zbmath": ZbMathSource,
            "inspire": InspireSource,
            "ads": lambda: AdsSource(api_token=ads_api_token),
            "dblp": DblpSource,
            "osti": OstiSource,
            "ntrs": NtrsSource,
        }
        for source_name, factory in specialist_factories.items():
            if source_name in requested:
                self._source_map[source_name] = factory()

    @property
    def tool_definitions(self) -> List[Dict[str, Any]]:
        return [
            dict(definition)
            for definition in self.TOOL_DEFINITIONS
            if definition["name"] in self._enabled_sources
        ]

    @classmethod
    def is_valid_backend(cls, name: str) -> bool:
        return name in cls._TOOL_NAME_SET

    def as_source_list(self) -> List[object]:
        """Return individual ``LiteratureSourceProtocol`` implementations
        suitable for ``IterativeSearchAgent`` injection.
        """
        return [
            self._source_map[definition["name"]]
            for definition in self.TOOL_DEFINITIONS
            if definition["name"] in self._enabled_sources
        ]

    async def search(
        self, query_text: str, backend: str, limit: int = 20,
    ) -> List[PaperRecord]:
        if backend not in self._enabled_sources:
            raise ValueError(f"Backend {backend!r} is disabled")
        query = SearchQuery(
            query_id="_direct",
            text=query_text,
            target_source=backend,
            purpose="direct",
            relation_to_question="Direct search.",
        )
        source = self._source_map.get(backend)
        if source is None:
            raise ValueError(
                f"Unknown backend {backend!r}. Valid: {sorted(self._TOOL_NAME_SET)}"
            )
        return await source.search(query, limit=limit)

    async def search_multi(
        self, queries: List[SearchQuery], limit: int = 20,
    ) -> Dict[str, List[PaperRecord]]:
        import asyncio

        async def _run_one(q: SearchQuery) -> tuple[str, List[PaperRecord]]:
            try:
                records = await self.search(q.text, q.target_source, limit=limit)
            except Exception:
                records = []
            return q.query_id, records

        tasks = [_run_one(q) for q in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        output: Dict[str, List[PaperRecord]] = {}
        for r in results:
            if isinstance(r, Exception):
                continue
            qid, records = r
            output[qid] = records
        return output
