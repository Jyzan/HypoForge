import pytest

from hypoforge.graph_context import GraphContext, GraphContextItem
from hypoforge.epistemic_contract import (
    normalize_hypothesis_grounding,
    validate_epistemic_contract,
)
from hypoforge.state import HypothesisCard, HypothesisPremise, PipelineState
from hypoforge.modules.m4_hypothesis_generation import M4HypothesisGeneration


def context_with_evidence(
    *, evidence_id: str = "E1", paper_id: str = "P1"
) -> GraphContext:
    return GraphContext(
        available_evidence_ids=[evidence_id],
        available_paper_ids=[paper_id],
        evidence_to_paper={evidence_id: paper_id},
        established_facts=[GraphContextItem(
            entry_id="F1",
            kind="established_fact",
            text="X is present.",
            evidence_ids=[evidence_id],
            source_paper_id=paper_id,
        )],
    )


def context_with_bridge(bridge_id: str = "HYP_G1") -> GraphContext:
    return GraphContext(
        bridge_hypotheses=[GraphContextItem(
            entry_id=bridge_id,
            kind="bridge_hypothesis",
            text="The unresolved bridge is testable.",
        )]
    )


def supported_card(evidence_id: str = "E1") -> HypothesisCard:
    return HypothesisCard(
        hypothesis_id="H1",
        statement="X changes Y.",
        research_gap="The causal link remains unresolved.",
        factual_premises=[HypothesisPremise(
            premise_id="P1",
            claim="X is present.",
            kind="evidence_backed",
            supporting_evidence_ids=[evidence_id],
            audit_verdict="supported",
        )],
    )


def bridge_only_card(*, evidence_ids: list[str] | None = None) -> HypothesisCard:
    return HypothesisCard(
        hypothesis_id="H1",
        statement="The bridge changes Y.",
        research_gap="The bridge mechanism remains unresolved.",
        working_assumptions=[HypothesisPremise(
            premise_id="BRIDGE_HYP_G1",
            claim="The unresolved bridge is testable.",
            kind="unverified_bridge",
            bridge_hypothesis_node_id="HYP_G1",
        )],
        supporting_evidence=list(evidence_ids or []),
    )


def test_historical_hypothesis_defaults_to_legacy_unknown():
    card = HypothesisCard(hypothesis_id="H1", statement="X changes Y.")
    assert card.grounding_status == "legacy_unknown"


def test_grounding_status_round_trips_through_json():
    card = HypothesisCard(
        hypothesis_id="H1",
        statement="X changes Y.",
        grounding_status="bridge_only",
    )
    restored = HypothesisCard.model_validate_json(card.model_dump_json())
    assert restored.grounding_status == "bridge_only"


def test_new_card_cannot_have_no_grounding():
    result = validate_epistemic_contract(
        HypothesisCard(
            hypothesis_id="H1",
            statement="X changes Y.",
            research_gap="The causal link is unresolved.",
        ),
        context_with_evidence(),
    )
    assert result.valid is False
    assert "missing_grounding_structure" in result.codes


def test_normalizer_derives_card_evidence_from_supported_premises():
    normalized = normalize_hypothesis_grounding(
        supported_card("E1"), context_with_evidence(evidence_id="E1", paper_id="P1")
    )
    assert normalized.grounding_status == "evidence_backed"
    assert normalized.supporting_evidence == ["E1"]
    assert normalized.source_paper_ids == ["P1"]


def test_bridge_only_card_cannot_keep_card_level_evidence():
    result = validate_epistemic_contract(
        bridge_only_card(evidence_ids=["E1"]),
        context_with_bridge(),
    )
    assert result.valid is False
    assert "bridge_claims_evidence" in result.codes


class MalformedRepairClient:
    async def structured_chat(self, **kwargs):
        corrected = bridge_only_card().model_dump(mode="json")
        corrected["working_assumptions"] = [{
            "premise_id": "WA2",
            "claim": "A newly proposed bridge without an M3 node.",
            "kind": "unverified_bridge",
            "required": True,
            "bridge_hypothesis_node_id": "",
        }]
        return {"items": [{
            "hypothesis_id": "H1",
            "passed": False,
            "hidden_factual_claims": [],
            "overclaim_segments": [],
            "repair_summary": "Malformed model repair",
            "corrected_hypothesis": corrected,
        }]}


@pytest.mark.asyncio
async def test_malformed_epistemic_repair_keeps_original_valid_hypothesis():
    module = M4HypothesisGeneration()
    module.client = MalformedRepairClient()
    original = bridge_only_card()

    cards, diagnostics = await module._audit_and_repair_epistemic_structure(
        state=PipelineState(input_question="How does X affect Y?"),
        cards=[original],
        context=context_with_bridge(),
    )

    assert len(cards) == 1
    assert cards[0].hypothesis_id == "H1"
    assert cards[0].working_assumptions[0].bridge_hypothesis_node_id == "HYP_G1"
    assert diagnostics[0].hypothesis_id == "H1"
