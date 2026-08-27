import pytest
from types import SimpleNamespace

from hypoforge.evaluation.m6_scoring import (
    M6ScoreConditions,
    PlanQualityAssessment,
    SemanticScoreAssessment,
)
from hypoforge.graph_context import build_graph_context
from hypoforge.modules.m6_review_iteration import M6ReviewIteration
from hypoforge.state import (
    EvidenceSufficiencyVerdict,
    ExperimentalValidationVerdict,
    FactualPremiseAudit,
    HypothesisCard,
    PipelineState,
    ResearchPlan,
    ReviewResult,
    ReviewerDimension,
    ScoreDimensionDetail,
    ValidationCoverageItem,
)


def _rows() -> list[ScoreDimensionDetail]:
    weights = {
        "task_coverage": 0.10,
        "novelty": 0.15,
        "scientific_logic": 0.15,
        "evidence_reliability": 0.15,
        "testability": 0.10,
        "experimental_rigor": 0.15,
        "statistics_reproducibility": 0.10,
        "technical_feasibility": 0.10,
    }
    return [
        ScoreDimensionDetail(
            dimension=name,
            score=score,
            weight=weight,
            weighted_contribution=score * weight,
            source="hybrid",
            confidence=0.8,
        )
        for (name, score), weight in zip(
            [(name, 4.0) for name in weights], weights.values()
        )
    ]


def test_m6_build_scoring_summary_preserves_dimension_audit_rows():
    module = object.__new__(M6ReviewIteration)

    summary = module._build_scoring_summary(_rows(), M6ScoreConditions())

    assert summary.final_score == 4.0
    assert all(row.source == "hybrid" for row in summary.dimensions)
    assert all(row.confidence == 0.8 for row in summary.dimensions)


def test_low_novelty_is_a_score_cap_not_an_actionable_routing_issue():
    module = object.__new__(M6ReviewIteration)
    rows = _rows()
    rows = [row.model_copy(update={"score": 5.0}) for row in rows]
    rows[1] = rows[1].model_copy(update={"score": 1.0})

    summary = module._build_scoring_summary(rows, M6ScoreConditions())

    assert summary.final_score == 3.9
    assert any(cap.rule_id == "low_novelty" for cap in summary.applied_caps)


@pytest.mark.asyncio
async def test_semantic_scoring_audits_render_graph_context_with_method():
    class Client:
        async def structured_chat(self, **kwargs):
            assessment = {
                "score": 4.0,
                "confidence": 0.9,
                "strengths": ["graph context received"],
                "weaknesses": [],
                "deductions": [],
                "blocking_issues": [],
            }
            if "experimental_rigor" in kwargs.get("output_schema", {}).get("properties", {}):
                return {
                    "task_coverage": assessment,
                    "evidence_reliability": assessment,
                    "testability": assessment,
                    "experimental_rigor": assessment,
                    "statistics_reproducibility": assessment,
                    "technical_feasibility": assessment,
                }
            return assessment

    state = PipelineState(input_question="q")
    module = M6ReviewIteration()
    module.client = Client()
    module.detailed_scoring = True
    graph_context = build_graph_context(state)
    hypothesis = HypothesisCard(hypothesis_id="H1", statement="A causes B")
    plan = ResearchPlan(hypothesis_id="H1", study_subjects="A and B")

    novelty = await module._review_novelty(state, hypothesis, plan, graph_context)
    quality = await module._review_plan_quality(state, hypothesis, plan, graph_context)

    assert novelty.score == 4.0
    assert quality.experimental_rigor.score == 4.0


@pytest.mark.asyncio
async def test_prompt_scores_prevent_structural_gates_from_forcing_three_dimensions_to_five():
    module = M6ReviewIteration()
    module.detailed_scoring = False
    semantic = lambda score: SemanticScoreAssessment(
        score=score,
        confidence=0.9,
        weaknesses=[f"semantic weakness at {score}"],
    )

    async def review_novelty(*args, **kwargs):
        return semantic(3.0)

    async def review_plan_quality(*args, **kwargs):
        return PlanQualityAssessment(
            task_coverage=semantic(4.0),
            evidence_reliability=semantic(4.1),
            testability=semantic(4.3),
            experimental_rigor=semantic(3.5),
            statistics_reproducibility=semantic(3.2),
            technical_feasibility=semantic(4.0),
        )

    module._review_novelty = review_novelty
    module._review_plan_quality = review_plan_quality
    state = PipelineState(input_question="Explain all major mechanisms of Q")
    hypothesis = HypothesisCard(
        hypothesis_id="H1",
        statement="One mechanism of Q",
        observable_predictions=["P1"],
        falsification_conditions=["F1"],
    )
    plan = ResearchPlan(hypothesis_id="H1", study_subjects="Q")
    alignment = SimpleNamespace(passed=True, score=1.0, missing_anchors=())
    factual = EvidenceSufficiencyVerdict(
        sufficient=True,
        premise_audits=[FactualPremiseAudit(
            premise_id="P1",
            claim="fact",
            verdict="supported",
            evidence_ids=["E1"],
        )],
    )
    validation = ExperimentalValidationVerdict(
        sufficient=True,
        items=[ValidationCoverageItem(
            target_id="T1",
            target_kind="prediction",
            target_text="P1",
            verdict="covered",
        )],
    )
    logic_review = ReviewResult(
        dimension=ReviewerDimension.SCIENTIFIC_LOGIC,
        score=4.0,
        hard_gate_passed=True,
    )

    rows, _ = await module._build_modern_scoring_dimensions(
        state=state,
        hypothesis=hypothesis,
        plan=plan,
        graph_context=build_graph_context(state),
        hypothesis_alignment=alignment,
        plan_alignment=alignment,
        semantic_alignment_passed=True,
        deterministic_alignment_passed=True,
        factual_verdict=factual,
        experimental_validation_verdict=validation,
        gates={"evidence_coverage": 1.0, "source_quality": 1.0},
        reviews=[logic_review],
    )

    scores = {row.dimension: row.score for row in rows}
    assert scores["task_coverage"] == 4.0
    assert scores["evidence_reliability"] == 4.1
    assert scores["testability"] == 4.3
