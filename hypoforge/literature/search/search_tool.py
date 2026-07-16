"""Unified literature search convenience — wraps PubMed + Semantic Scholar.

Provides ``tool_definitions`` for QueryPlanner and a single search entry
point that dispatches to the appropriate backend.

For use with ``IterativeSearchAgent``, prefer the individual source
implementations in ``hypoforge.literature.sources``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from hypoforge.literature.models import PaperRecord, SearchQuery
from hypoforge.literature.sources.academic_source import AcademicSource
from hypoforge.literature.sources.pubmed_source import PubMedSource


class LiteratureSearchTool:
    """Unified search across PubMed + Semantic Scholar.

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
            "display_name": "Semantic Scholar / OpenAlex",
            "description": (
                "Search academic literature across ALL disciplines via "
                "Semantic Scholar or OpenAlex (auto-selected). "
                "BEST for: computer science, engineering, physics, social "
                "sciences, mathematics, and interdisciplinary research. "
                "Also works for broad biomedical queries that may appear "
                "in non-PubMed journals. "
                "NOT ideal for: specialized biomedical/clinical queries "
                "requiring MeSH terms (use PubMed instead)."
            ),
        },
    ]

    _TOOL_NAME_SET: set[str] = {d["name"] for d in TOOL_DEFINITIONS}

    def __init__(
        self,
        pubmed: Optional[PubMedSource] = None,
        academic: Optional[AcademicSource] = None,
    ):
        self._pubmed_source = pubmed if pubmed is not None else PubMedSource()
        self._academic_source = academic if academic is not None else AcademicSource()

    @property
    def tool_definitions(self) -> List[Dict[str, Any]]:
        return [dict(d) for d in self.TOOL_DEFINITIONS]

    @classmethod
    def is_valid_backend(cls, name: str) -> bool:
        return name in cls._TOOL_NAME_SET

    def as_source_list(self) -> List[object]:
        """Return individual ``LiteratureSourceProtocol`` implementations
        suitable for ``IterativeSearchAgent`` injection.
        """
        return [self._pubmed_source, self._academic_source]

    async def search(
        self, query_text: str, backend: str, limit: int = 20,
    ) -> List[PaperRecord]:
        query = SearchQuery(
            query_id="_direct",
            text=query_text,
            target_source=backend,
            purpose="direct",
            relation_to_question="Direct search.",
        )
        if backend == "pubmed":
            return await self._pubmed_source.search(query, limit=limit)
        if backend == "semantic_scholar":
            return await self._academic_source.search(query, limit=limit)
        raise ValueError(
            f"Unknown backend {backend!r}. "
            f"Valid: {sorted(self._TOOL_NAME_SET)}"
        )

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
