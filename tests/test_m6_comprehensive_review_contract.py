from __future__ import annotations

import pytest

from hypoforge.state import (
    EvidenceGap,
    EvidenceGraph,
    ExperimentalValidationVerdict,
    HypothesisCard,
    HypothesisPremise,
    PipelineState,
    ResearchPlan,
    ValidationCoverageItem,
)
from hypoforge.evidence_audit import PremiseAuditResult


def test_old_checkpoint_loads_without_validation_verdict():
    state = PipelineState.model_validate({"input_question": "Q"})
    assert state.experimental_validation_verdict is None


def test_validation_item_links_one_m4_target_to_m5_plan():
    item = ValidationCoverageItem(
        target_id="H1:prediction:0",
        target_kind="prediction",
        target_text="Combined treatment lowers IL-6 more than either monotherapy.",
        verdict="covered",
        procedure_refs=["procedure:0"],
        measurement_refs=["metric:0"],
        analysis_refs=["analysis:0"],
        falsification_text="No interaction effect is detected.",
    )
    assert item.verdict == "covered"


def test_validation_verdict_defaults_to_empty_items():
    verdict = ExperimentalValidationVerdict(sufficient=True)
    assert verdict.items == []


def test_m6_audit_scope_excludes_mechanism_synergy_and_bridges():
    from hypoforge.modules.m6_review_iteration import M6ReviewIteration
    from hypoforge.state import ResearchPlan

    hypothesis = HypothesisCard(
        hypothesis_id="H1",
        statement="SASP and rapalogues may act synergistically.",
        mechanism="SASP may activate mTOR in neighbouring stem cells.",
        factual_premises=[HypothesisPremise(
            premise_id="P1",
            claim="Cellular senescence is an aging hallmark.",
            kind="evidence_backed",
            required=True,
        )],
        working_assumptions=[HypothesisPremise(
            premise_id="BRIDGE_HYP_G1",
            claim="The combination may preserve tissue function.",
            kind="unverified_bridge",
            bridge_hypothesis_node_id="HYP_G1",
        )],
    )
    plan = ResearchPlan(hypothesis_id="H1")
    scope = M6ReviewIteration._build_evidence_audit_scope(hypothesis, plan)
    assert [item.source_claim_id for item in scope] == ["P1"]
    rendered = " ".join(item.claim for item in scope)
    assert "SASP" not in rendered
    assert "synerg" not in rendered.lower()
    assert "bridge" not in rendered.lower()


def test_evidence_gap_keeps_source_claim_identity_for_m6_routing():
    gap = EvidenceGap(
        description="The factual premise lacks canonical support.",
        source_claim_type="factual_premise",
        source_claim_id="P1",
    )
    assert gap.source_claim_type == "factual_premise"
    assert gap.source_claim_id == "P1"


def test_public_result_keeps_experimental_validation_verdict():
    from hypoforge.webapp import RunManager

    verdict = ExperimentalValidationVerdict(items=[ValidationCoverageItem(
        target_id="H1:prediction:0",
        target_kind="prediction",
        target_text="Y increases.",
        verdict="partial",
        procedure_refs=["procedure:0"],
        rationale="No interaction analysis.",
    )], sufficient=False)
    payload = RunManager._public_state_payload(
        "run-m6", {
            "input_question": "Q",
            "experimental_validation_verdict": verdict.model_dump(mode="json"),
            "evidence_verdict": {"sufficient": True, "premise_audits": []},
        }, is_final=False,
    )
    assert payload["experimental_validation_verdict"]["items"][0]["target_id"] == "H1:prediction:0"
    assert payload["evidence_verdict"]["sufficient"] is True


@pytest.mark.asyncio
async def test_m6_fact_audit_only_sends_factual_premises_and_maps_outcomes(monkeypatch):
    import hypoforge.modules.m6_review_iteration as m6_module

    received = []

    class FakeAuditor:
        def __init__(self, client, **kwargs):
            pass

        async def audit_premises(self, premises, graph, **kwargs):
            received.extend(premises)
            return [
                PremiseAuditResult(
                    premise_id="P1", claim="A is present.", verdict="supported",
                    evidence_ids=["E1"], rationale="supported",
                ),
                PremiseAuditResult(
                    premise_id="P2", claim="B causes C.", verdict="unsupported",
                    rationale="missing canonical support",
                ),
                PremiseAuditResult(
                    premise_id="P3", claim="D increases E.", verdict="contradicted",
                    evidence_ids=["E9"], rationale="canonical evidence disagrees",
                ),
            ]

    monkeypatch.setattr(m6_module, "EvidenceAuditService", FakeAuditor)
    hypothesis = HypothesisCard(
        hypothesis_id="H1",
        statement="A may affect B.",
        mechanism="A activates an unverified bridge.",
        factual_premises=[
            HypothesisPremise(
                premise_id="P1", claim="A is present.", kind="evidence_backed",
                supporting_evidence_ids=["E1"],
            ),
            HypothesisPremise(
                premise_id="P2", claim="B causes C.", kind="evidence_backed",
            ),
            HypothesisPremise(
                premise_id="P3", claim="D increases E.", kind="evidence_backed",
            ),
        ],
        working_assumptions=[HypothesisPremise(
            premise_id="BRIDGE_HYP_G1", claim="A may activate B.",
            kind="unverified_bridge", bridge_hypothesis_node_id="HYP_G1",
        )],
    )
    state = PipelineState(
        input_question="Can A affect B?",
        top_hypotheses=[hypothesis],
        evidence_graph=EvidenceGraph(),
    )
    module = object.__new__(m6_module.M6ReviewIteration)
    module.client = object()
    module.reviewer_timeout_seconds = 1.0
    verdict = await module._audit_factual_premises(
        state, hypothesis, ResearchPlan(hypothesis_id="H1"), 2,
    )
    assert [premise.premise_id for premise in received] == ["P1", "P2", "P3"]
    assert all(p.kind == "evidence_backed" for p in received)
    assert verdict.sufficient is False
    assert [item.premise_id for item in verdict.premise_audits] == ["P1", "P2", "P3"]
    assert {gap.source_claim_id for gap in verdict.gaps} == {"P2", "P3"}
    assert next(gap for gap in verdict.gaps if gap.source_claim_id == "P3").gap_type == "conflict"
    assert next(gap for gap in verdict.gaps if gap.source_claim_id == "P3").resolution_evidence_ids == ["E9"]
