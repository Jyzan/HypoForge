from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

from ...state import ConfidenceLevel, KnowledgeEntryType
from .adapter import AgenticM2Adapter
from .models import (
    CoverageReport,
    EvidenceBucket,
    EvidenceChunk,
    EvidenceLinkedKnowledge,
    PaperRecord,
    PaperReadingResult,
    QueryIntent,
    ScoutNote,
    SearchBudget,
    SearchQuery,
    SearchRunResult,
    SearchState,
)
from .protocols import (
    CoverageEvaluatorProtocol,
    PaperDeduplicatorProtocol,
    PaperRankerProtocol,
    QueryPlannerProtocol,
    ReadingExtractionWorkflowProtocol,
    ScoutReaderProtocol,
)
from .search import IterativeSearchAgent
from .sources import PubMedBackend, PubMedLiteratureSource

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
                   state: SearchState | None = None,
                   supplement_entities: Sequence[str] = ()) -> list[SearchQuery]:
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


class AbstractReadingWorkflow(ReadingExtractionWorkflowProtocol):
    async def run(
        self,
        sub_question: str,
        papers: Sequence[PaperRecord],
        search_context: SearchRunResult | None = None,
    ) -> list[PaperReadingResult]:
        results: list[PaperReadingResult] = []
        for paper in papers:
            abstract = " ".join(paper.abstract.split())
            if not abstract:
                results.append(PaperReadingResult(
                    paper_id=paper.paper_id,
                    summary=paper.title,
                    degraded_to_abstract=True,
                    errors=["PubMed abstract unavailable"],
                ))
                continue
            sentence = re.split(r"(?<=[.!?。！？])\s+", abstract, maxsplit=1)[0][:600]
            evidence_id = f"{paper.paper_id}:abstract:1"
            evidence = EvidenceChunk(
                evidence_id=evidence_id,
                paper_id=paper.paper_id,
                chunk_id=f"{paper.paper_id}:abstract",
                section="abstract",
                quote=sentence,
                normalized_claim=sentence,
                relevance_score=0.7,
            )
            entry_content = f"PubMed abstract reports: {sentence}"
            # Stable entry id: sha1(normalised content + source paper)[:12].
            digest = hashlib.sha1(
                (
                    " ".join(entry_content.split()).casefold()
                    + "\u0000"
                    + paper.paper_id
                ).encode("utf-8")
            ).hexdigest()[:12]
            entry = EvidenceLinkedKnowledge(
                entry_id=f"pubmed-{digest}",
                entry_type=KnowledgeEntryType.ESTABLISHED_FACT,
                content=entry_content,
                confidence=ConfidenceLevel.MEDIUM,
                entities=_latin_terms(f"{sub_question} {paper.title}", limit=6),
                evidence_ids=[evidence_id],
            )
            results.append(PaperReadingResult(
                paper_id=paper.paper_id,
                summary=sentence,
                evidence=[evidence],
                knowledge_entries=[entry],
                degraded_to_abstract=True,
            ))
        return results


def build_minimal_pubmed_adapter(
    *,
    backend: PubMedBackend | None = None,
    final_k: int = 5,
    source_timeout_seconds: float = 30.0,
    zero_result_relaxation: bool = True,
) -> AgenticM2Adapter:
    if final_k <= 0:
        raise ValueError("final_k must be positive")
    source = PubMedLiteratureSource(
        backend=backend,
        timeout_seconds=source_timeout_seconds,
        enable_relaxation=zero_result_relaxation,
    )
    agent = IterativeSearchAgent(
        query_planner=RuleBasedQueryPlanner(),
        sources=[source],
        deduplicator=ExactPaperDeduplicator(),
        ranker=MetadataPaperRanker(),
        scout_reader=AbstractScoutReader(),
        coverage_evaluator=SingleSourceCoverageEvaluator(),
        final_k=final_k,
        candidate_limit=final_k,
        per_query_limit=final_k,
        source_timeout_seconds=source_timeout_seconds,
    )
    return AgenticM2Adapter(
        search_agent=agent,
        reading_workflow=AbstractReadingWorkflow(),
        budget=SearchBudget(max_rounds=2, max_queries=2, max_papers=final_k),
    )
