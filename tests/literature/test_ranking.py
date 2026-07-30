from __future__ import annotations

import pytest

from hypoforge.literature.models import (
    EvidenceBucket,
    FulltextStatus,
    PaperRecord,
    ScoutNote,
)
from hypoforge.literature.search.ranking import PaperRanker, rerank_with_scout


def paper(paper_id: str, title: str, **kwargs: object) -> PaperRecord:
    return PaperRecord(
        paper_id=paper_id,
        title=title,
        sources=["test"],
        **kwargs,
    )


@pytest.mark.asyncio
async def test_relevant_paper_can_outrank_an_earlier_irrelevant_result() -> None:
    irrelevant = paper("irrelevant", "Unrelated ecology observations")
    relevant = paper(
        "relevant",
        "Hsp70 ATPase controls protein folding",
        abstract="Hsp70 regulates ATP-dependent protein folding.",
    )

    ranked = await PaperRanker(current_year=2026).rank(
        "Hsp70 protein folding", [irrelevant, relevant], limit=2
    )

    assert ranked[0].paper_id == "relevant"
    assert ranked[0].rank_scores["query_relevance"] > 0


@pytest.mark.asyncio
async def test_ranker_populates_all_scores_without_mutating_inputs() -> None:
    source = paper(
        "paper",
        "Protein folding",
        abstract="Mechanisms of protein folding.",
        authors=["Author"],
        year=2025,
        journal="Journal",
        doi="10.1/test",
        citation_count=12,
        is_open_access=True,
        fulltext_status=FulltextStatus.XML_AVAILABLE,
    )
    before = source.model_dump()

    ranked = await PaperRanker(current_year=2026).rank(
        "protein folding", [source], limit=1
    )

    assert source.model_dump() == before
    assert set(ranked[0].rank_scores) >= {
        "query_relevance",
        "retrieval_prior",
        "citation_impact",
        "recency",
        "metadata_quality",
        "access_quality",
        "total",
    }
    assert 0 <= ranked[0].rank_scores["total"] <= 1


@pytest.mark.asyncio
async def test_citation_impact_is_age_adjusted() -> None:
    older = paper("old", "Same topic", year=2006, citation_count=100)
    newer = paper("new", "Same topic", year=2025, citation_count=100)

    ranked = await PaperRanker(current_year=2026).rank(
        "same topic", [older, newer], limit=2
    )
    scores = {item.paper_id: item.rank_scores for item in ranked}

    assert scores["new"]["citation_impact"] > scores["old"]["citation_impact"]


@pytest.mark.asyncio
async def test_unknown_citation_metadata_is_not_treated_as_known_zero() -> None:
    unknown = paper(
        "unknown",
        "Same topic",
        citation_count=None,
        rank_scores={"source_relevance": 0.5},
    )
    known_zero = paper(
        "zero",
        "Same topic",
        citation_count=0,
        rank_scores={"source_relevance": 0.5},
    )

    ranked = await PaperRanker(current_year=2026).rank(
        "same topic", [unknown, known_zero], limit=2
    )
    scores = {item.paper_id: item.rank_scores for item in ranked}

    assert scores["unknown"]["total"] > scores["zero"]["total"]


@pytest.mark.asyncio
async def test_equal_scores_keep_input_order_and_limit() -> None:
    first = paper("z", "Identical", rank_scores={"source_relevance": 0.5})
    second = paper("a", "Identical", rank_scores={"source_relevance": 0.5})

    ranked = await PaperRanker(current_year=2026).rank(
        "absent query", [first, second], limit=1
    )

    assert [item.paper_id for item in ranked] == ["z"]


@pytest.mark.asyncio
async def test_ranker_supports_cjk_query_tokens_and_nonpositive_limit() -> None:
    item = paper("paper", "蛋白质折叠与衰老")
    ranker = PaperRanker(current_year=2026)

    ranked = await ranker.rank("蛋白质折叠", [item], limit=1)

    assert ranked[0].rank_scores["query_relevance"] > 0
    assert await ranker.rank("蛋白质折叠", [item], limit=0) == []


def test_scout_rerank_keeps_relevant_directional_primary_evidence_in_early_window() -> (
    None
):
    reviews = [
        paper(
            f"review-{index}",
            f"Review of Hsp70 regulation {index}",
            abstract=f"Review context {index}.",
            rank_scores={"total": 0.90 - index * 0.01},
        )
        for index in range(4)
    ]
    supporting = paper(
        "supporting-primary",
        "Hsp70 ATPase assay supports the proposed mechanism",
        abstract="Hsp70 directly increased ATPase activity.",
        rank_scores={"total": 0.73},
    )
    contradicting = paper(
        "contradicting-primary",
        "Hsp70 ATPase experiment finds no association during aging",
        abstract="Hsp70 had no effect on ATPase activity during aging.",
        rank_scores={"total": 0.72},
    )
    papers = [*reviews, supporting, contradicting]
    notes = [
        ScoutNote(
            paper_id=item.paper_id,
            relevance_to_question=0.90,
            directness_to_question=0.25,
            evidence_buckets={EvidenceBucket.REVIEW},
            study_design="review",
        )
        for item in reviews
    ]
    notes.extend(
        [
            ScoutNote(
                paper_id=supporting.paper_id,
                relevance_to_question=0.80,
                directness_to_question=0.90,
                evidence_buckets={EvidenceBucket.SUPPORTING},
                supporting_evidence=["Hsp70 directly increased ATPase activity."],
                study_design="experimental",
            ),
            ScoutNote(
                paper_id=contradicting.paper_id,
                relevance_to_question=0.75,
                directness_to_question=0.90,
                evidence_buckets={EvidenceBucket.CONTRADICTING},
                contradicting_evidence=[
                    "Hsp70 had no effect on ATPase activity during aging."
                ],
                study_design="experimental",
            ),
        ]
    )

    reranked = rerank_with_scout(papers, notes)
    early_ids = {item.paper_id for item in reranked[:5]}

    assert supporting.paper_id in early_ids
    assert contradicting.paper_id in early_ids
    assert all("diversity_selection_score" in item.rank_scores for item in reranked[:5])


def test_scout_rerank_builds_a_credible_final_set_not_a_review_list() -> None:
    reviews = [
        paper(
            f"review-{index}",
            f"Review of Hsp70 ATPase activity during aging {index}",
            rank_scores={"total": 0.90 - index * 0.01},
        )
        for index in range(4)
    ]
    supporting = paper(
        "supporting-primary",
        "Hsp70 ATPase experiment supports the mechanism",
        abstract="An in vitro assay demonstrates the proposed Hsp70 mechanism.",
        rank_scores={"total": 0.72},
    )
    contradicting = paper(
        "contradicting-primary",
        "Hsp70 ATPase experiment finds no association",
        abstract="An in vivo experiment found no association during aging.",
        rank_scores={"total": 0.71},
    )
    additional_primary = paper(
        "additional-primary",
        "Hsp70 ATPase activity in an aging animal model",
        abstract="An animal study measures ATPase activity during aging.",
        rank_scores={"total": 0.70},
    )
    low_relevance = paper(
        "low-relevance-review",
        "Review of Hsp70 in unrelated disorders",
        rank_scores={"total": 0.85},
    )
    papers = [
        *reviews,
        supporting,
        contradicting,
        additional_primary,
        low_relevance,
    ]
    notes = [
        ScoutNote(
            paper_id=item.paper_id,
            relevance_to_question=0.90,
            directness_to_question=0.85,
            evidence_buckets={EvidenceBucket.REVIEW},
            study_design="review",
        )
        for item in reviews
    ]
    notes.extend(
        [
            ScoutNote(
                paper_id=supporting.paper_id,
                relevance_to_question=0.80,
                directness_to_question=0.90,
                evidence_buckets={EvidenceBucket.SUPPORTING},
                supporting_evidence=[
                    "An in vitro assay demonstrates the proposed Hsp70 mechanism."
                ],
                study_design="experimental",
            ),
            ScoutNote(
                paper_id=contradicting.paper_id,
                relevance_to_question=0.78,
                directness_to_question=0.90,
                evidence_buckets={EvidenceBucket.CONTRADICTING},
                contradicting_evidence=[
                    "An in vivo experiment found no association during aging."
                ],
                study_design="experimental",
            ),
            ScoutNote(
                paper_id=additional_primary.paper_id,
                relevance_to_question=0.75,
                directness_to_question=0.90,
                evidence_buckets={EvidenceBucket.METHODOLOGICAL},
                study_design="observational",
            ),
            ScoutNote(
                paper_id=low_relevance.paper_id,
                relevance_to_question=0.50,
                directness_to_question=0.80,
                evidence_buckets={EvidenceBucket.REVIEW},
                study_design="review",
            ),
        ]
    )

    reranked = rerank_with_scout(papers, notes)
    early = reranked[:5]
    early_ids = {item.paper_id for item in early}
    early_review_count = sum("review" in item.title.casefold() for item in early)

    assert early_review_count <= 2
    assert {
        supporting.paper_id,
        contradicting.paper_id,
        additional_primary.paper_id,
    }.issubset(early_ids)
    assert low_relevance.paper_id not in early_ids


@pytest.mark.parametrize("final_k", [3, 5, 10])
def test_scout_rerank_uses_the_requested_final_selection_window(final_k: int) -> None:
    papers = [
        paper(
            f"paper-{index}",
            f"Hsp70 study {index}",
            rank_scores={"total": 1.0 - index * 0.02},
        )
        for index in range(12)
    ]
    notes = [
        ScoutNote(
            paper_id=item.paper_id,
            relevance_to_question=0.8,
            directness_to_question=0.6,
            study_design="observational",
        )
        for item in papers
    ]

    reranked = rerank_with_scout(papers, notes, selection_limit=final_k)

    assert all(
        "diversity_selection_score" in item.rank_scores
        for item in reranked[:final_k]
    )
    assert all(
        "diversity_selection_score" not in item.rank_scores
        for item in reranked[final_k:]
    )
