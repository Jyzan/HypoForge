"""Deterministic metadata-aware paper ranking."""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import datetime, timezone

from ..models import FulltextStatus, PaperRecord
from ..protocols import PaperRankerProtocol
from ._text import lexical_relevance


_WEIGHTS = {
    "query_relevance": 0.45,
    "retrieval_prior": 0.20,
    "citation_impact": 0.15,
    "recency": 0.10,
    "metadata_quality": 0.05,
    "access_quality": 0.05,
}
_ACCESS_SCORE = {
    FulltextStatus.UNKNOWN: 0.10,
    FulltextStatus.UNAVAILABLE: 0.0,
    FulltextStatus.FAILED: 0.0,
    FulltextStatus.ABSTRACT_ONLY: 0.35,
    FulltextStatus.XML_AVAILABLE: 0.80,
    FulltextStatus.HTML_AVAILABLE: 0.80,
    FulltextStatus.PDF_AVAILABLE: 0.80,
    FulltextStatus.DOWNLOADED: 1.0,
}


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _age(paper: PaperRecord, current_year: int) -> int | None:
    if paper.year is None:
        return None
    return max(0, current_year - paper.year)


def _metadata_quality(paper: PaperRecord) -> float:
    identity_present = bool(
        paper.doi or paper.pmid or paper.pmcid or paper.external_ids
    )
    values = (
        bool(paper.abstract),
        bool(paper.authors),
        paper.year is not None,
        bool(paper.journal),
        identity_present,
    )
    return sum(values) / len(values)


class PaperRanker(PaperRankerProtocol):
    """Rank papers from lexical relevance and normalized metadata signals."""

    tool_name = "paper_ranker"

    def __init__(self, *, current_year: int | None = None) -> None:
        self.current_year = current_year or datetime.now(timezone.utc).year

    async def rank(
        self,
        sub_question: str,
        papers: Sequence[PaperRecord],
        limit: int,
    ) -> list[PaperRecord]:
        if limit <= 0 or not papers:
            return []

        citation_raw: list[float] = []
        for paper in papers:
            age = _age(paper, self.current_year)
            annualized = (paper.citation_count or 0) / ((age or 0) + 1)
            citation_raw.append(math.log1p(annualized))
        max_citation = max(citation_raw, default=0.0)

        ranked: list[tuple[float, int, str, PaperRecord]] = []
        paper_count = len(papers)
        for index, paper in enumerate(papers):
            relevance = lexical_relevance(
                sub_question, paper.title, paper.abstract
            )
            supplied_prior = paper.rank_scores.get("source_relevance")
            if supplied_prior is not None and math.isfinite(supplied_prior):
                retrieval_prior = _clamp(float(supplied_prior))
            elif paper_count == 1:
                retrieval_prior = 1.0
            else:
                retrieval_prior = 1.0 - index / (paper_count - 1)
            citation_impact = (
                citation_raw[index] / max_citation if max_citation > 0 else 0.0
            )
            age = _age(paper, self.current_year)
            recency = 0.0 if age is None else 1.0 / (1.0 + age / 5.0)
            metadata_quality = _metadata_quality(paper)
            access_quality = _ACCESS_SCORE[paper.fulltext_status]
            if paper.is_open_access:
                access_quality = min(1.0, access_quality + 0.20)

            scores = {
                "query_relevance": relevance,
                "retrieval_prior": retrieval_prior,
                "citation_impact": citation_impact,
                "recency": recency,
                "metadata_quality": metadata_quality,
                "access_quality": access_quality,
            }
            total = sum(scores[name] * weight for name, weight in _WEIGHTS.items())
            scores["total"] = total
            updated_scores = dict(paper.rank_scores)
            updated_scores.update(scores)
            ranked_paper = paper.model_copy(update={"rank_scores": updated_scores})
            ranked.append((total, index, paper.paper_id, ranked_paper))

        ranked.sort(key=lambda item: (-item[0], item[1], item[2]))
        return [item[3] for item in ranked[:limit]]
