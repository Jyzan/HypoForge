from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

from .models import (
    CoverageReport,
    EvidenceBucket,
    PaperRecord,
    QueryIntent,
    ScoutNote,
    SearchQuery,
    SearchState,
)
from .protocols import (
    CoverageEvaluatorProtocol,
    PaperDeduplicatorProtocol,
    PaperRankerProtocol,
    QueryPlannerProtocol,
    ScoutReaderProtocol,
)

_STOP_WORDS = {
    "and", "are", "does", "for", "from", "how", "into", "the", "what",
    "when", "where", "which", "with",
}


def _latin_terms(text: str, limit: int = 8) -> list[str]:
    found = re.findall(r"[A-Za-z][A-Za-z0-9-]{1,}", text)
    output: list[str] = []
    seen: set[str] = set()
    for term in found:
        key = term.casefold()
        if key in _STOP_WORDS or key in seen:
            continue
        seen.add(key)
        output.append(term)
        if len(output) >= limit:
            break
    return output


class RuleBasedQueryPlanner(QueryPlannerProtocol):
    async def plan(self, sub_question: str, key_entities: Sequence[str] = (),
                   domains: Sequence[str] = (), question_type: str = "",
                   state: SearchState | None = None) -> list[SearchQuery]:
        if state and state.queries_used:
            return []
        terms = _latin_terms(" ".join([*key_entities, sub_question]), limit=6)
        text = " AND ".join(terms) if terms else sub_question.strip()
        digest = hashlib.sha256(text.casefold().encode("utf-8")).hexdigest()[:12]
        return [SearchQuery(
            query_id=f"pubmed-{digest}", text=text, intent=QueryIntent.CORE,
            target_source="pubmed", purpose="Find direct PubMed evidence",
            relation_to_question="Uses explicit entities from the sub-question",
        )]


def _normalized_title(title: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", title.casefold()).split())


def _paper_keys(paper: PaperRecord) -> list[str]:
    keys: list[str] = []
    if paper.pmid:
        keys.append(f"pmid:{paper.pmid.casefold()}")
    if paper.doi:
        keys.append(f"doi:{paper.doi.casefold()}")
    title = _normalized_title(paper.title)
    if title:
        keys.append(f"title:{title}")
    return keys


class ExactPaperDeduplicator(PaperDeduplicatorProtocol):
    async def deduplicate(
        self,
        papers: Sequence[PaperRecord],
        existing_papers: Sequence[PaperRecord] = (),
    ) -> list[PaperRecord]:
        canonical_by_key: dict[str, PaperRecord] = {}
        for paper in existing_papers:
            for key in _paper_keys(paper):
                canonical_by_key.setdefault(key, paper)

        output: list[PaperRecord] = []
        returned_ids: set[str] = set()
        for paper in papers:
            canonical = next(
                (canonical_by_key[key] for key in _paper_keys(paper)
                 if key in canonical_by_key),
                paper,
            )
            for key in [*_paper_keys(paper), *_paper_keys(canonical)]:
                canonical_by_key.setdefault(key, canonical)
            if canonical.paper_id not in returned_ids:
                returned_ids.add(canonical.paper_id)
                output.append(canonical)
        return output


class MetadataPaperRanker(PaperRankerProtocol):
    async def rank(
        self,
        sub_question: str,
        papers: Sequence[PaperRecord],
        limit: int,
    ) -> list[PaperRecord]:
        indexed = list(enumerate(papers))
        indexed.sort(key=lambda item: (
            0 if item[1].abstract.strip() else 1,
            -(item[1].year or 0),
            item[0],
        ))
        output: list[PaperRecord] = []
        for _, paper in indexed[:limit]:
            scores = dict(paper.rank_scores)
            scores.update({
                "has_abstract": 1.0 if paper.abstract.strip() else 0.0,
                "publication_year": float(paper.year or 0),
            })
            output.append(paper.model_copy(update={"rank_scores": scores}))
        return output


class AbstractScoutReader(ScoutReaderProtocol):
    async def read(
        self,
        sub_question: str,
        papers: Sequence[PaperRecord],
    ) -> list[ScoutNote]:
        notes: list[ScoutNote] = []
        for paper in papers:
            terms = _latin_terms(f"{paper.title} {paper.abstract}", limit=8)
            notes.append(ScoutNote(
                paper_id=paper.paper_id,
                main_topic=paper.title,
                key_terms=terms,
                entities=terms,
                relevance_to_question=0.8 if terms else 0.2,
            ))
        return notes


class SingleSourceCoverageEvaluator(CoverageEvaluatorProtocol):
    async def evaluate(
        self,
        sub_question: str,
        papers: Sequence[PaperRecord],
        scout_notes: Sequence[ScoutNote],
        state: SearchState,
    ) -> CoverageReport:
        return CoverageReport(
            covered_buckets={EvidenceBucket.SUPPORTING} if papers else set(),
            missing_buckets=(
                set() if papers else {EvidenceBucket.SUPPORTING}
            ),
            covered_topics=["PubMed evidence"] if papers else [],
            missing_topics=[] if papers else ["PubMed evidence"],
            sufficient=bool(papers),
            rationale=(
                "At least one valid PubMed candidate is available for workflow testing."
                if papers else "PubMed returned no valid candidate papers."
            ),
        )
