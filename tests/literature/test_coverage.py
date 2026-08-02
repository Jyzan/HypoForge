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
            "facets": [
                {
                    "facet": "direct mechanism evidence",
                    "status": "covered",
                    "paper_ids": ["paper-0"],
                    "sentence_ids": ["paper-0:S1"],
                }
            ],
            "missing_topics": [],
            "sufficient": True,
            "rationale": "The necessary facet is grounded.",
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
        abstract=kwargs.pop(
            "abstract", f"Finding {index} directly addresses the question."
        ),
        sources=["test"],
        **kwargs,
    )


def note(
    index: int,
    *,
    support: bool = False,
    contradict: bool = False,
    relevance: float = 0.9,
    directness: float = 0.9,
    study_type: str = "experimental",
    methodological: bool = False,
) -> ScoutNote:
    buckets: set[EvidenceBucket] = set()
    supporting: list[str] = []
    contradicting: list[str] = []
    if support:
        buckets.add(EvidenceBucket.SUPPORTING)
        supporting = [f"Finding {index} directly addresses the question."]
    if contradict:
        buckets.add(EvidenceBucket.CONTRADICTING)
        contradicting = [f"Finding {index} directly addresses the question."]
    if methodological:
        buckets.add(EvidenceBucket.METHODOLOGICAL)
    return ScoutNote(
        paper_id=f"paper-{index}",
        main_topic=f"Topic {index}",
        relevance_to_question=relevance,
        directness_to_question=directness,
        evidence_buckets=buckets,
        supporting_evidence=supporting,
        contradicting_evidence=contradicting,
        study_design=study_type,
        evidence_summary=f"Evidence {index}",
    )


def mechanism_inputs(count: int = 3) -> tuple[list[PaperRecord], list[ScoutNote]]:
    papers = [paper(index, year=2026) for index in range(count)]
    notes = [note(0, support=True), *[note(index) for index in range(1, count)]]
    return papers, notes


@pytest.mark.asyncio
async def test_mechanism_coverage_requires_grounded_support_but_not_contradiction() -> None:
    papers, notes = mechanism_inputs()
    client = FakeStructuredClient()

    report = await CoverageEvaluator(client, current_year=2026).evaluate(
        "How does Hsp70 regulate folding?",
        papers,
        notes,
        SearchState(question_type="mechanism_explanation"),
    )

    assert report.sufficient is True
    assert EvidenceBucket.SUPPORTING in report.covered_buckets
    assert EvidenceBucket.CONTRADICTING not in report.covered_buckets
    assert client.calls[0]["disable_thinking"] is True
    assert "facets" in client.calls[0]["output_schema"]["properties"]
    prompt = client.calls[0]["user_prompt"]
    assert '"evidence_sentences"' in prompt
    assert '"sentence_id": "paper-0:S1"' in prompt
    assert "Finding 0 directly addresses the question." in prompt


@pytest.mark.asyncio
async def test_method_profile_uses_methodological_directness() -> None:
    papers = [paper(index, year=2024) for index in range(3)]
    notes = [
        note(
            index,
            study_type="method",
            methodological=True,
            directness=0.9,
        )
        for index in range(3)
    ]
    client = FakeStructuredClient(
        {
            "facets": [
                {
                    "facet": "ATPase assay design",
                    "status": "covered",
                    "paper_ids": ["paper-0"],
                    "sentence_ids": [],
                }
            ],
            "missing_topics": [],
            "sufficient": True,
            "rationale": "The methodological facet is covered.",
        }
    )
    report = await CoverageEvaluator(client, current_year=2026).evaluate(
        "Develop an ATPase assay",
        papers,
        notes,
        SearchState(question_type="method_development"),
    )

    assert report.sufficient is True
    assert EvidenceBucket.METHODOLOGICAL in report.covered_buckets


@pytest.mark.asyncio
async def test_discovery_profile_requires_support_recent_and_two_papers() -> None:
    papers = [paper(index, year=2026) for index in range(2)]
    notes = [note(0, support=True), note(1)]
    report = await CoverageEvaluator(
        FakeStructuredClient(), current_year=2026
    ).evaluate(
        "Was the phenomenon observed?",
        papers,
        notes,
        SearchState(question_type="phenomenon_discovery"),
    )
    assert report.sufficient is True

    sparse = await CoverageEvaluator(
        FakeStructuredClient(), current_year=2026
    ).evaluate(
        "Was the phenomenon observed?",
        papers[:1],
        notes[:1],
        SearchState(question_type="phenomenon_discovery"),
    )
    assert sparse.sufficient is False
    assert any("more needed" in gap for gap in sparse.missing_topics)


@pytest.mark.asyncio
async def test_unknown_question_type_uses_conservative_general_gate() -> None:
    papers, notes = mechanism_inputs()
    report = await CoverageEvaluator(
        FakeStructuredClient(), current_year=2026
    ).evaluate("question", papers, notes, SearchState())
    assert report.sufficient is True


@pytest.mark.asyncio
async def test_invalid_facet_paper_or_sentence_ids_cannot_cover_facet() -> None:
    papers, notes = mechanism_inputs()
    client = FakeStructuredClient(
        {
            "facets": [
                {
                    "facet": "mechanism",
                    "status": "covered",
                    "paper_ids": ["invented-paper"],
                    "sentence_ids": ["paper-99:S9"],
                }
            ],
            "missing_topics": [],
            "sufficient": True,
            "rationale": "Invented evidence.",
        }
    )

    report = await CoverageEvaluator(client, current_year=2026).evaluate(
        "question", papers, notes, SearchState()
    )

    assert report.sufficient is False
    assert "mechanism" in report.missing_topics


@pytest.mark.asyncio
async def test_invalidated_coverage_does_not_keep_a_sufficient_rationale() -> None:
    papers, notes = mechanism_inputs()
    client = FakeStructuredClient(
        {
            "facets": [
                {
                    "facet": "mechanism",
                    "status": "partial",
                    "paper_ids": ["paper-0"],
                    "sentence_ids": [],
                }
            ],
            "missing_topics": ["mechanism"],
            "sufficient": False,
            "rationale": "The evidence is sufficiently complete.",
        }
    )

    report = await CoverageEvaluator(client, current_year=2026).evaluate(
        "question", papers, notes, SearchState()
    )

    assert report.sufficient is False
    assert "sufficiently complete" not in report.rationale
    assert "remains insufficient" in report.rationale


@pytest.mark.asyncio
async def test_model_missing_topics_are_limited_and_preserve_previous_context() -> None:
    papers, notes = mechanism_inputs()
    state = SearchState(
        missing_topics={"prior mechanism gap"},
        key_entities={"HSP70"},
        domains={"protein homeostasis"},
    )
    client = FakeStructuredClient(
        {
            "facets": [
                {
                    "facet": "mechanism",
                    "status": "partial",
                    "paper_ids": ["paper-0"],
                    "sentence_ids": [],
                }
            ],
            "missing_topics": ["mechanism", "method", "result", "extra"],
            "sufficient": False,
            "rationale": "Gaps remain.",
        }
    )
    report = await CoverageEvaluator(client).evaluate(
        "How does the HSP70 mechanism work?", papers, notes, state
    )

    assert report.sufficient is False
    assert len(report.missing_topics) <= 3
    prompt = client.calls[0]["user_prompt"]
    assert "prior mechanism gap" in prompt
    assert "HSP70" in prompt
    assert "protein homeostasis" in prompt


@pytest.mark.asyncio
async def test_global_entities_not_present_in_subquestion_are_not_prompted() -> None:
    papers, notes = mechanism_inputs()
    state = SearchState(
        key_entities={
            "Hsp70",
            "ATPase activity",
            "J-domain proteins",
            "post-translational modifications",
        },
    )
    client = FakeStructuredClient()

    await CoverageEvaluator(client).evaluate(
        "How does Hsp70 ATPase activity affect protein aggregation?",
        papers,
        notes,
        state,
    )

    prompt = client.calls[0]["user_prompt"]
    assert "Hsp70" in prompt
    assert "ATPase activity" in prompt
    assert "J-domain proteins" not in prompt
    assert "post-translational modifications" not in prompt


@pytest.mark.asyncio
async def test_selection_limit_uses_only_final_window() -> None:
    papers = [paper(index, year=2026) for index in range(4)]
    notes = [note(index) for index in range(3)] + [note(3, support=True)]

    report = await CoverageEvaluator(
        FakeStructuredClient(), selection_limit=3, current_year=2026
    ).evaluate("question", papers, notes, SearchState())

    assert report.sufficient is False
    assert EvidenceBucket.SUPPORTING not in report.covered_buckets


@pytest.mark.asyncio
async def test_direction_bucket_without_exact_sentence_is_ignored() -> None:
    papers, notes = mechanism_inputs()
    notes[0] = notes[0].model_copy(
        update={"supporting_evidence": ["A fabricated sentence."]}
    )

    report = await CoverageEvaluator(
        FakeStructuredClient(), current_year=2026
    ).evaluate("question", papers, notes, SearchState())

    assert report.sufficient is False
    assert EvidenceBucket.SUPPORTING not in report.covered_buckets


@pytest.mark.asyncio
async def test_llm_failure_is_conservative_and_does_not_fabricate_coverage() -> None:
    papers, notes = mechanism_inputs()
    report = await CoverageEvaluator(
        FakeStructuredClient(error=RuntimeError("offline")), current_year=2026
    ).evaluate("question", papers, notes, SearchState())

    assert report.sufficient is False
    assert "Semantic coverage was not confirmed" in report.rationale


@pytest.mark.asyncio
async def test_empty_input_reports_actionable_gaps_without_calling_llm() -> None:
    client = FakeStructuredClient()
    report = await CoverageEvaluator(client).evaluate(
        "question", [], [], SearchState()
    )

    assert isinstance(report, CoverageReport)
    assert report.sufficient is False
    assert report.covered_buckets == set()
    assert client.calls == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"relevance_threshold": -0.1},
        {"relevance_threshold": 1.1},
        {"min_relevant_papers": 0},
        {"min_bucket_count": 0},
        {"min_bucket_count": 7},
        {"selection_limit": 0},
        {"max_tokens": 0},
    ],
)
def test_coverage_rejects_invalid_thresholds(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        CoverageEvaluator(**kwargs)
