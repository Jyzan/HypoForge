"""Pure, auditable aggregation for the modern M6 score."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from pydantic import BaseModel, Field

from ..state import M6ScoringSummary, ScoreCap, ScoreDimensionDetail


M6_SCORE_WEIGHTS: dict[str, float] = {
    "task_coverage": 0.10,
    "novelty": 0.15,
    "scientific_logic": 0.15,
    "evidence_reliability": 0.15,
    "testability": 0.10,
    "experimental_rigor": 0.15,
    "statistics_reproducibility": 0.10,
    "technical_feasibility": 0.10,
}


class SemanticScoreAssessment(BaseModel):
    """Reasoned score payload used by bounded M6 semantic audits."""

    score: float = Field(ge=1.0, le=5.0)
    confidence: float = Field(ge=0.0, le=1.0)
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    deductions: List[str] = Field(default_factory=list)
    blocking_issues: List[str] = Field(default_factory=list)


class PlanQualityAssessment(BaseModel):
    """One bounded call's independent assessment of the hypothesis-plan pair."""

    task_coverage: SemanticScoreAssessment
    evidence_reliability: SemanticScoreAssessment
    testability: SemanticScoreAssessment
    experimental_rigor: SemanticScoreAssessment
    statistics_reproducibility: SemanticScoreAssessment
    technical_feasibility: SemanticScoreAssessment


@dataclass(frozen=True)
class M6ScoreConditions:
    """Structured defects that may cap the score, independent of its value."""

    task_misaligned: bool = False
    core_fact_contradicted: bool = False
    core_untestable: bool = False
    experimental_validation_missing: bool = False
    multiple_major_design_defects: bool = False


def _normalise_dimensions(
    dimensions: Iterable[ScoreDimensionDetail],
) -> list[ScoreDimensionDetail]:
    rows = list(dimensions)
    expected = set(M6_SCORE_WEIGHTS)
    names = [row.dimension for row in rows]
    if set(names) != expected or len(names) != len(set(names)):
        raise ValueError(
            "M6 score must contain exactly one row for each dimension"
        )
    if any(abs(row.weight - M6_SCORE_WEIGHTS[row.dimension]) > 1e-9 for row in rows):
        raise ValueError("M6 score dimension weights do not match the contract")
    return [
        row.model_copy(update={
            "weighted_contribution": round(
                row.score * M6_SCORE_WEIGHTS[row.dimension], 4
            )
        })
        for row in sorted(
            rows,
            key=lambda item: list(M6_SCORE_WEIGHTS).index(item.dimension),
        )
    ]


def aggregate_m6_scoring(
    dimensions: Iterable[ScoreDimensionDetail],
    conditions: M6ScoreConditions,
) -> M6ScoringSummary:
    """Calculate weighted score, then apply the strictest triggered cap."""

    rows = _normalise_dimensions(dimensions)
    raw_score = round(sum(row.weighted_contribution for row in rows), 1)
    novelty = next(row.score for row in rows if row.dimension == "novelty")

    caps: list[ScoreCap] = []
    if conditions.task_misaligned:
        caps.append(ScoreCap(
            rule_id="task_misaligned",
            maximum=1.9,
            attribution="both",
            reason="The hypothesis or plan does not answer the original task contract.",
        ))
    if conditions.core_fact_contradicted:
        caps.append(ScoreCap(
            rule_id="core_fact_contradicted",
            maximum=2.9,
            attribution="hypothesis",
            reason="A required factual premise is contradicted by reliable evidence.",
        ))
    if conditions.core_untestable:
        caps.append(ScoreCap(
            rule_id="core_untestable",
            maximum=2.9,
            attribution="hypothesis",
            reason="The core hypothesis lacks an operational test or falsification condition.",
        ))
    if conditions.experimental_validation_missing:
        caps.append(ScoreCap(
            rule_id="experimental_validation_missing",
            maximum=2.9,
            attribution="plan",
            reason="The research plan does not test one or more required M4 targets.",
        ))
    if conditions.multiple_major_design_defects:
        caps.append(ScoreCap(
            rule_id="multiple_major_design_defects",
            maximum=3.4,
            attribution="plan",
            reason="Multiple major design weaknesses undermine causal interpretation.",
        ))
    if novelty < 2.0:
        caps.append(ScoreCap(
            rule_id="low_novelty",
            maximum=3.9,
            attribution="hypothesis",
            reason=f"Independent novelty score is {novelty:.1f}/5, below the 2.0 threshold.",
        ))

    final_score = min(
        raw_score,
        *(cap.maximum for cap in caps),
    ) if caps else raw_score
    rationale = (
        f"Weighted raw score {raw_score:.1f}/5; "
        f"final score {final_score:.1f}/5. "
        "This score is display-only; iteration routing uses structured hard gates "
        "and evidence/validation verdicts, never this numeric value."
    )
    if caps:
        rationale += " Applied caps: " + ", ".join(cap.rule_id for cap in caps) + "."
    return M6ScoringSummary(
        raw_score=raw_score,
        final_score=round(final_score, 1),
        dimensions=rows,
        applied_caps=caps,
        rationale=rationale,
        complete=all(row.source != "degraded" for row in rows),
    )
