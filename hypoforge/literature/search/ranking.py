"""Deterministic metadata-aware paper ranking."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from datetime import datetime, timezone

from ..models import EvidenceBucket, FulltextStatus, PaperRecord, ScoutNote
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
_REVIEW_PATTERN = re.compile(
    r"\b(systematic review|meta-analysis|meta analysis|review article|review)\b",
    re.IGNORECASE,
)
_NON_EMPIRICAL_PATTERN = re.compile(
    r"\b(computational|computer[- ]?based|in silico|simulation|"
    r"mathematical model|modeling|modelling|methodology|protocol|algorithm)\b",
    re.IGNORECASE,
)
_DEFAULT_FINAL_SELECTION_WINDOW = 5
_DIVERSITY_RELEVANCE_THRESHOLD = 0.55
_DIRECTIONAL_DIRECTNESS_THRESHOLD = 0.45
_REVIEW_CAP_IN_EARLY_SELECTION = 2
_CLOSE_EVIDENCE_MARGIN = 0.25
_SCORE_EPSILON = 1e-9


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


def _safe_score(value: object, default: float) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return default
    return _clamp(score) if math.isfinite(score) else default


def _is_review(paper: PaperRecord, note: ScoutNote | None) -> bool:
    return bool(
        _REVIEW_PATTERN.search(
            " ".join(
                [
                    paper.title,
                    paper.publication_type,
                    note.study_design if note is not None else "",
                ]
            )
        )
    )


def _is_empirical(paper: PaperRecord, note: ScoutNote | None) -> bool:
    """Return whether a paper is plausibly original empirical evidence.

    A non-review is not automatically a primary study: computational models,
    protocols, and methods papers are valuable for coverage but should not get
    the same early-window boost as a study that could independently test the
    question's claim.
    """

    if _is_review(paper, note):
        return False
    return not _NON_EMPIRICAL_PATTERN.search(
        " ".join(
            [
                paper.title,
                paper.publication_type,
                note.study_design if note is not None else "",
            ]
        )
    )


def _note_relevance(note: ScoutNote | None, default: float = 0.0) -> float:
    if note is None:
        return default
    return _safe_score(note.relevance_to_question, default)


def _note_directness(note: ScoutNote | None, default: float = 0.0) -> float:
    if note is None:
        return default
    return _safe_score(note.directness_to_question, _note_relevance(note, default))


def _meets_threshold(value: float, threshold: float) -> bool:
    """Avoid classifying a decimal score just below its mathematical boundary."""

    return value + _SCORE_EPSILON >= threshold


def _is_direct_primary(paper: PaperRecord, note: ScoutNote | None) -> bool:
    """Return whether a Scout note represents directly usable original evidence.

    Relevance alone is not enough: a contextual review or a computational
    model can be highly topical without testing the relation in the question.
    The selector may still retain them for context, but it should not let them
    consume the slots intended for direct primary evidence.
    """

    return bool(
        note is not None
        and paper.abstract.strip()
        and _is_empirical(paper, note)
        and _meets_threshold(_note_relevance(note), _DIVERSITY_RELEVANCE_THRESHOLD)
        and _meets_threshold(_note_directness(note), _DIRECTIONAL_DIRECTNESS_THRESHOLD)
    )


def _credible_directional_roles(note: ScoutNote | None) -> set[str]:
    """Return directional roles that can influence Final-K selection.

    A note carrying both labels is evidence of ambiguity, not a reason to give
    one paper both bonuses.  The coverage evaluator separately enforces that
    the two directions come from distinct papers.
    """

    if note is None or not _meets_threshold(
        _note_directness(note), _DIRECTIONAL_DIRECTNESS_THRESHOLD
    ):
        return set()
    directional = {
        bucket.value
        for bucket in note.evidence_buckets
        if bucket in {EvidenceBucket.SUPPORTING, EvidenceBucket.CONTRADICTING}
    }
    return directional if len(directional) == 1 else set()


def _evidence_roles(paper: PaperRecord, note: ScoutNote | None) -> set[str]:
    if note is None:
        return set()
    roles = {bucket.value for bucket in note.evidence_buckets}
    roles.difference_update(
        {EvidenceBucket.SUPPORTING.value, EvidenceBucket.CONTRADICTING.value}
    )
    roles.update(_credible_directional_roles(note))
    if _is_direct_primary(paper, note):
        roles.add("direct_primary")
    return roles


def _selection_kind(paper: PaperRecord, note: ScoutNote | None) -> str:
    if _is_direct_primary(paper, note):
        return "direct_primary"
    if _is_review(paper, note):
        return "review"
    if _is_empirical(paper, note):
        return "indirect_primary"
    return "context"


def _is_credible_for_early_selection(
    paper: PaperRecord,
    note: ScoutNote | None,
) -> bool:
    # Old Scout notes do not have a directness field. Keep their historical
    # ordering behavior rather than silently treating them as low confidence.
    return note is None or _meets_threshold(
        _note_relevance(note), _DIVERSITY_RELEVANCE_THRESHOLD
    )


def _diversify_early_selection(
    papers: Sequence[PaperRecord],
    notes: dict[str, ScoutNote],
    *,
    selection_limit: int | None = None,
) -> list[PaperRecord]:
    """Select a credible, evidence-aware prefix of the ranked candidates.

    ``IterativeSearchAgent`` returns the first ``final_k`` papers after Scout
    reranking.  The Agent currently does not pass that setting into this B
    helper, so the default matches the integrated factory's Final-K of five;
    callers may supply another value without changing any Protocol signature.

    The prefix is a set-selection problem rather than five independent score
    comparisons: it prioritizes direct primary evidence and distinct evidence
    directions, caps contextual reviews when close credible alternatives are
    available, and leaves low-relevance papers out of the prefix while credible
    candidates remain.
    """

    if selection_limit is not None and selection_limit <= 0:
        raise ValueError("selection_limit must be positive when provided")
    ranked = list(papers)
    if len(ranked) < 2:
        return ranked

    selected: list[PaperRecord] = []
    remaining = list(ranked)
    covered_roles: set[str] = set()
    selected_sources: set[str] = set()
    selected_kinds: Counter[str] = Counter()
    window = min(selection_limit or _DEFAULT_FINAL_SELECTION_WINDOW, len(ranked))
    role_bonus = {
        EvidenceBucket.CONTRADICTING.value: 0.08,
        EvidenceBucket.SUPPORTING.value: 0.07,
        "direct_primary": 0.08,
        EvidenceBucket.REVIEW.value: 0.01,
        EvidenceBucket.METHODOLOGICAL.value: 0.02,
    }

    while remaining and len(selected) < window:
        credible = [
            paper
            for paper in remaining
            if _is_credible_for_early_selection(paper, notes.get(paper.paper_id))
        ]
        candidates = credible or remaining
        best_base = max(
            _safe_score(item.rank_scores.get("post_scout_total"), 0.0)
            for item in candidates
        )
        if selected_kinds["review"] >= _REVIEW_CAP_IN_EARLY_SELECTION:
            close_non_reviews = [
                paper
                for paper in candidates
                if not _is_review(paper, notes.get(paper.paper_id))
                and _safe_score(paper.rank_scores.get("post_scout_total"), 0.0)
                >= best_base - _CLOSE_EVIDENCE_MARGIN
            ]
            if close_non_reviews:
                candidates = close_non_reviews

        choices: list[tuple[float, float, int, str, PaperRecord]] = []
        for index, paper in enumerate(candidates):
            note = notes.get(paper.paper_id)
            base = _safe_score(paper.rank_scores.get("post_scout_total"), 0.0)
            relevance = _note_relevance(note, base)
            roles = _evidence_roles(paper, note)
            kind = _selection_kind(paper, note)
            adjustment = 0.0
            if (
                relevance >= _DIVERSITY_RELEVANCE_THRESHOLD
                and base >= best_base - _CLOSE_EVIDENCE_MARGIN
            ):
                adjustment += sum(
                    bonus
                    for role, bonus in role_bonus.items()
                    if role in roles and role not in covered_roles
                )
                if selected_sources and not (set(paper.sources) & selected_sources):
                    adjustment += 0.02
                kind_penalty = {
                    "review": 0.045,
                    "direct_primary": 0.015,
                    "indirect_primary": 0.025,
                    "context": 0.030,
                }[kind]
                adjustment -= kind_penalty * selected_kinds[kind]
                # Diversity resolves close calls only. In particular, a paper
                # with multiple metadata labels cannot leapfrog a materially
                # stronger result just by collecting bonuses.
                adjustment = max(-0.16, min(0.16, adjustment))
            choices.append((base + adjustment, base, -index, paper.paper_id, paper))

        _, base, _, _, chosen = max(choices)
        note = notes.get(chosen.paper_id)
        roles = _evidence_roles(chosen, note)
        kind = _selection_kind(chosen, note)
        adjustment = next(
            score - original
            for score, original, _, _, item in choices
            if item.paper_id == chosen.paper_id
        )
        scores = dict(chosen.rank_scores)
        scores["diversity_adjustment"] = adjustment
        scores["diversity_selection_score"] = base + adjustment
        selected.append(chosen.model_copy(update={"rank_scores": scores}))
        covered_roles.update(roles)
        selected_sources.update(chosen.sources)
        selected_kinds[kind] += 1
        remaining = [item for item in remaining if item.paper_id != chosen.paper_id]

    return [*selected, *remaining]


def rerank_with_scout(
    papers: Sequence[PaperRecord],
    notes: Sequence[ScoutNote],
    *,
    scout_weight: float = 0.65,
    selection_limit: int | None = None,
) -> list[PaperRecord]:
    """Blend metadata rank with grounded Scout relevance deterministically."""
    if not 0.0 <= scout_weight <= 1.0:
        raise ValueError("scout_weight must be between zero and one")
    if not papers:
        return []

    note_by_paper = {note.paper_id: note for note in notes}
    count = len(papers)
    reranked: list[tuple[float, int, str, PaperRecord]] = []
    for index, paper in enumerate(papers):
        default_base = 1.0 if count == 1 else 1.0 - index / (count - 1)
        raw_base = paper.rank_scores.get("total", default_base)
        base = _safe_score(raw_base, default_base)
        note = note_by_paper.get(paper.paper_id)
        scout_relevance = note.relevance_to_question if note is not None else base
        scout_directness = (
            _safe_score(note.directness_to_question, scout_relevance)
            if note is not None
            else scout_relevance
        )
        grounded_relevance = 0.75 * scout_relevance + 0.25 * scout_directness
        combined = (1.0 - scout_weight) * base + scout_weight * grounded_relevance
        scores = dict(paper.rank_scores)
        scores.update(
            {
                "scout_relevance": scout_relevance,
                "scout_directness": scout_directness,
                "post_scout_total": combined,
            }
        )
        reranked.append(
            (
                combined,
                index,
                paper.paper_id,
                paper.model_copy(update={"rank_scores": scores}),
            )
        )

    reranked.sort(key=lambda item: (-item[0], item[1], item[2]))
    return _diversify_early_selection(
        [item[3] for item in reranked],
        note_by_paper,
        selection_limit=selection_limit,
    )


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
            relevance = lexical_relevance(sub_question, paper.title, paper.abstract)
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
            active_weights = {
                name: weight
                for name, weight in _WEIGHTS.items()
                if name != "citation_impact" or paper.citation_count is not None
            }
            weight_sum = sum(active_weights.values())
            total = (
                sum(scores[name] * weight for name, weight in active_weights.items())
                / weight_sum
            )
            scores["total"] = total
            updated_scores = dict(paper.rank_scores)
            updated_scores.update(scores)
            ranked_paper = paper.model_copy(update={"rank_scores": updated_scores})
            ranked.append((total, index, paper.paper_id, ranked_paper))

        ranked.sort(key=lambda item: (-item[0], item[1], item[2]))
        return [item[3] for item in ranked[:limit]]
