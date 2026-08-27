"""Serper Scholar discovery followed by strict OpenAlex identity enrichment."""

from __future__ import annotations

import asyncio
from difflib import SequenceMatcher
import os
import re

from ..models import PaperRecord, SearchQuery
from ..protocols import LiteratureSourceProtocol
from .openalex_source import OpenAlexSource
from .scholar_proxy_source import ScholarProxySource


_TITLE_TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)


def _normalized_title(value: object) -> str:
    return " ".join(_TITLE_TOKEN_RE.findall(str(value or "").casefold()))


def _title_similarity(left: object, right: object) -> float:
    first = _normalized_title(left)
    second = _normalized_title(right)
    if not first or not second:
        return 0.0
    if first == second:
        return 1.0
    sequence = SequenceMatcher(None, first, second).ratio()
    left_tokens, right_tokens = set(first.split()), set(second.split())
    union = left_tokens | right_tokens
    jaccard = len(left_tokens & right_tokens) / len(union) if union else 0.0
    return 0.7 * sequence + 0.3 * jaccard


class OpenAlexEnrichedScholarSource(LiteratureSourceProtocol):
    """Keep Serper hits even when optional OpenAlex enrichment fails."""

    source_name = "serper_openalex"

    def __init__(
        self,
        *,
        scholar_source: ScholarProxySource | None = None,
        openalex_source: OpenAlexSource | None = None,
        serper_api_key: str = "",
        openalex_api_key: str = "",
        openalex_mailto: str = "",
        title_match_threshold: float = 0.92,
        enrichment_concurrency: int = 4,
        enrichment_timeout_seconds: float = 10.0,
    ) -> None:
        self.scholar_source = scholar_source or ScholarProxySource(
            api_key=serper_api_key
        )
        self.openalex_source = openalex_source or OpenAlexSource(
            api_key=openalex_api_key or os.environ.get("OPENALEX_API_KEY", ""),
            mailto=openalex_mailto or os.environ.get("OPENALEX_MAILTO", ""),
        )
        if not 0 < title_match_threshold <= 1:
            raise ValueError("title_match_threshold must be in (0, 1]")
        if enrichment_concurrency <= 0 or enrichment_timeout_seconds <= 0:
            raise ValueError("OpenAlex enrichment limits must be positive")
        self.title_match_threshold = float(title_match_threshold)
        self.enrichment_concurrency = int(enrichment_concurrency)
        self.enrichment_timeout_seconds = float(enrichment_timeout_seconds)
        self._cache: dict[str, PaperRecord | None] = {}

    @staticmethod
    def _cache_key(paper: PaperRecord) -> str:
        doi = PaperRecord.normalize_doi(paper.doi)
        return f"doi:{doi}" if doi else f"title:{_normalized_title(paper.title)}"

    async def _match(self, paper: PaperRecord) -> PaperRecord | None:
        texts = [paper.doi, paper.title] if paper.doi else [paper.title]
        for text in texts:
            query = SearchQuery(
                query_id="openalex-enrichment",
                text=text,
                target_source="openalex",
                purpose="identifier_enrichment",
                relation_to_question=paper.title,
            )
            candidates = await self.openalex_source.search(query, limit=5)
            if paper.doi:
                normalized = PaperRecord.normalize_doi(paper.doi)
                exact = next(
                    (row for row in candidates if PaperRecord.normalize_doi(row.doi) == normalized),
                    None,
                )
                if exact is not None:
                    return exact
            scored = [(_title_similarity(paper.title, row.title), row) for row in candidates]
            if scored:
                score, candidate = max(scored, key=lambda item: item[0])
                if score >= self.title_match_threshold:
                    return candidate
        return None

    @staticmethod
    def _merge(scholar: PaperRecord, openalex: PaperRecord) -> PaperRecord:
        external_ids = {**scholar.external_ids, **openalex.external_ids}
        sources = list(dict.fromkeys([*scholar.sources, *openalex.sources]))
        return openalex.model_copy(update={
            "paper_id": openalex.paper_id or scholar.paper_id,
            "title": openalex.title or scholar.title,
            "abstract": openalex.abstract or scholar.abstract,
            "authors": openalex.authors or scholar.authors,
            "year": openalex.year or scholar.year,
            "journal": openalex.journal or scholar.journal,
            "doi": openalex.doi or scholar.doi,
            "external_ids": external_ids,
            "citation_count": (
                scholar.citation_count
                if scholar.citation_count is not None else openalex.citation_count
            ),
            "sources": sources,
        })

    async def _enrich_one(
        self, paper: PaperRecord, semaphore: asyncio.Semaphore
    ) -> PaperRecord:
        key = self._cache_key(paper)
        if key in self._cache:
            cached = self._cache[key]
            return paper if cached is None else self._merge(paper, cached)
        async with semaphore:
            matched = await asyncio.wait_for(
                self._match(paper), timeout=self.enrichment_timeout_seconds
            )
        self._cache[key] = matched
        return paper if matched is None else self._merge(paper, matched)

    async def search(self, query: SearchQuery, limit: int = 20) -> list[PaperRecord]:
        papers = await self.scholar_source.search(query, limit=limit)
        semaphore = asyncio.Semaphore(self.enrichment_concurrency)
        results = await asyncio.gather(
            *(self._enrich_one(paper, semaphore) for paper in papers),
            return_exceptions=True,
        )
        return [
            paper if isinstance(result, BaseException) else result
            for paper, result in zip(papers, results)
        ]
