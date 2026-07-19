from __future__ import annotations

from typing import Any

import pytest

from hypoforge.literature.models import (
    CoverageReport,
    EvidenceBucket,
    PaperRecord,
    ScoutNote,
    SearchState,
)
from hypoforge.literature.search.coverage import CoverageEvaluator


class FakeStructuredClient:
    def __init__(
        self,
        response: dict[str, Any] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.response = response or {
            "covered_topics": ["Hsp70 regulation"],
            "missing_topics": [],
            "rationale": "The essential topics are represented.",
        }
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def structured_chat(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


def paper(index: int, **kwargs: object) -> PaperRecord:
    return PaperRecord(
        paper_id=f"paper-{index}",
        title=kwargs.pop("title", f"Paper {index}"),
        abstract=kwargs.pop("abstract", "Abstract"),
        sources=["test"],
        **kwargs,
    )


def note(
    index: int,
    *buckets: EvidenceBucket,
    relevance: float = 0.9,
    **kwargs: object,
) -> ScoutNote:
    return ScoutNote(
        paper_id=f"paper-{index}",
        main_topic=kwargs.pop("main_topic", f"Topic {index}"),
        relevance_to_question=relevance,
        evidence_buckets=set(buckets),
        evidence_summary=kwargs.pop("evidence_summary", f"Evidence {index}"),
        **kwargs,
    )


def balanced_inputs() -> tuple[list[PaperRecord], list[ScoutNote]]:
    papers = [paper(index, year=2024) for index in range(5)]
    notes = [
        note(0, EvidenceBucket.SUPPORTING, EvidenceBucket.RECENT),
        note(1, EvidenceBucket.CONTRADICTING, EvidenceBucket.METHODOLOGICAL),
        note(2, EvidenceBucket.REVIEW),
        note(3, EvidenceBucket.SUPPORTING),
        note(4, EvidenceBucket.METHODOLOGICAL),
    ]
    return papers, notes


@pytest.mark.asyncio
async def test_balanced_coverage_satisfies_deterministic_gate() -> None:
    papers, notes = balanced_inputs()
    client = FakeStructuredClient()

    report = await CoverageEvaluator(client, current_year=2026).evaluate(
        "Does Hsp70 regulate folding?", papers, notes, SearchState()
    )

    assert report.sufficient is True
    assert len(report.covered_buckets) == 5
    assert EvidenceBucket.SUPPORTING in report.covered_buckets
    assert EvidenceBucket.CONTRADICTING in report.covered_buckets
    assert report.missing_buckets == {EvidenceBucket.CLASSIC}
    assert report.missing_topics == []
    assert report.covered_buckets.isdisjoint(report.missing_buckets)
    assert "sufficient" not in client.calls[0]["output_schema"]["properties"]
    assert client.calls[0]["disable_thinking"] is True


@pytest.mark.asyncio
async def test_llm_critical_topic_gap_can_block_but_not_bypass_hard_gate() -> None:
    papers, notes = balanced_inputs()
    client = FakeStructuredClient(
        {
            "covered_topics": ["Hsp70 regulation"],
            "missing_topics": ["direct ATPase measurements"],
            "rationale": "A critical measurement is absent.",
            "sufficient": True,
        }
    )

    report = await CoverageEvaluator(client, current_year=2026).evaluate(
        "Does Hsp70 regulate folding?", papers, notes, SearchState()
    )

    assert report.sufficient is False
    assert report.missing_topics == ["direct ATPase measurements"]

    sparse_report = await CoverageEvaluator(client, current_year=2026).evaluate(
        "Does Hsp70 regulate folding?",
        papers[:1],
        notes[:1],
        SearchState(),
    )
    assert sparse_report.sufficient is False
    assert any("contradicting" in topic for topic in sparse_report.missing_topics)


@pytest.mark.asyncio
async def test_later_round_ignores_new_model_gap_but_keeps_persistent_gap() -> None:
    papers, notes = balanced_inputs()
    previous_gap = "direct ATPase measurements"
    state = SearchState(round_index=2, missing_topics={previous_gap})

    new_gap_report = await CoverageEvaluator(
        FakeStructuredClient({
            "covered_topics": ["Hsp70 regulation"],
            "missing_topics": ["new speculative gap"],
            "rationale": "A new target appeared.",
        }),
        current_year=2026,
    ).evaluate("question", papers, notes, state)
    persistent_report = await CoverageEvaluator(
        FakeStructuredClient({
            "covered_topics": ["Hsp70 regulation"],
            "missing_topics": [previous_gap],
            "rationale": "The prior target remains absent.",
        }),
        current_year=2026,
    ).evaluate("question", papers, notes, state)

    assert new_gap_report.sufficient is True
    assert new_gap_report.missing_topics == []
    assert persistent_report.sufficient is False
    assert persistent_report.missing_topics == [previous_gap]


@pytest.mark.asyncio
async def test_metadata_supplements_review_recent_classic_and_method_buckets() -> None:
    papers = [
        paper(0, year=2026),
        paper(1, year=2026),
        paper(2, title="A systematic review of Hsp70", year=2020),
        paper(3, year=2010, citation_count=150),
        paper(4, publication_type="randomized trial", year=2020),
    ]
    notes = [
        note(0, EvidenceBucket.SUPPORTING),
        note(1, EvidenceBucket.CONTRADICTING),
        note(2),
        note(3),
        note(4, study_design="randomized controlled trial"),
    ]

    report = await CoverageEvaluator(current_year=2026).evaluate(
        "question", papers, notes, SearchState()
    )

    assert report.sufficient is True
    assert report.covered_buckets == set(EvidenceBucket)


@pytest.mark.asyncio
async def test_irrelevant_and_duplicate_notes_do_not_inflate_paper_count() -> None:
    papers, notes = balanced_inputs()
    notes = [notes[0], notes[0], *notes[1:4], note(4, relevance=0.2)]

    report = await CoverageEvaluator(current_year=2026).evaluate(
        "question", papers, notes, SearchState()
    )

    assert report.sufficient is False
    assert "1 more needed" in " ".join(report.missing_topics)


@pytest.mark.asyncio
async def test_llm_failure_uses_deterministic_coverage_fallback() -> None:
    papers, notes = balanced_inputs()
    client = FakeStructuredClient(error=RuntimeError("offline"))

    report = await CoverageEvaluator(client, current_year=2026).evaluate(
        "question", papers, notes, SearchState()
    )

    assert report.sufficient is True
    assert report.missing_topics == []
    assert "5 relevant papers" in report.rationale


@pytest.mark.asyncio
async def test_empty_input_reports_actionable_gaps_without_calling_llm() -> None:
    client = FakeStructuredClient()

    report = await CoverageEvaluator(client, current_year=2026).evaluate(
        "question", [], [], SearchState()
    )

    assert isinstance(report, CoverageReport)
    assert report.sufficient is False
    assert report.covered_buckets == set()
    assert report.missing_buckets == set(EvidenceBucket)
    assert any("supporting" in topic for topic in report.missing_topics)
    assert any("contradicting" in topic for topic in report.missing_topics)
    assert client.calls == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"relevance_threshold": -0.1},
        {"relevance_threshold": 1.1},
        {"min_relevant_papers": 0},
        {"min_bucket_count": 0},
        {"min_bucket_count": 7},
        {"max_tokens": 0},
    ],
)
def test_coverage_rejects_invalid_thresholds(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        CoverageEvaluator(**kwargs)
