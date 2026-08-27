from hypoforge.modules.m4_hypothesis_generation import M4HypothesisGeneration
from hypoforge.prompts.m4_prompts import M4_GENERATOR_SYSTEM_PROMPT
from hypoforge.graph_context import GraphContext, GraphContextItem
from hypoforge.state import HypothesisCard, HypothesisPremise, PipelineState
import pytest


def test_m4_generator_contract_separates_premises_gap_and_conjecture():
    prompt = M4_GENERATOR_SYSTEM_PROMPT.lower()
    assert "factual_premises" in prompt
    assert "working_assumptions" in prompt
    assert "research_gap" in prompt
    assert "observable_predictions" in prompt
    assert "proposed falsifiable relation" in prompt
    assert "epistemic boundary" in prompt


def test_m4_normalizer_keeps_mechanism_as_conjecture_not_fact():
    module = object.__new__(M4HypothesisGeneration)
    cards = module._normalise_hypotheses([{
        "hypothesis_id": "H1",
        "statement": "A may influence B under condition C.",
        "mechanism": "A activates X which changes B.",
        "factual_premises": [{
            "premise_id": "P1",
            "claim": "A is present in the system.",
            "kind": "evidence_backed",
            "supporting_evidence_ids": ["E1"],
        }],
        "working_assumptions": [],
        "research_gap": "Whether X mediates the change in B.",
    }])
    assert len(cards) == 1
    assert cards[0].factual_premises[0].claim == "A is present in the system."
    assert cards[0].mechanism == "A activates X which changes B."
    assert cards[0].research_gap.startswith("Whether X")


def test_m4_generation_schema_requires_epistemic_fields():
    schema = M4HypothesisGeneration._hypothesis_generation_schema()
    required = set(schema["items"]["required"])
    assert {
        "factual_premises",
        "working_assumptions",
        "research_gap",
        "supporting_evidence",
        "source_paper_ids",
    } <= required


@pytest.mark.asyncio
async def test_m4_epistemic_auditor_is_called_once_and_can_repair_card():
    class FakeClient:
        async def structured_chat(self, **kwargs):
            return {"items": [{
                "hypothesis_id": "H1",
                "passed": True,
                "hidden_factual_claims": [],
                "overclaim_segments": [],
                "repair_summary": "",
                "corrected_hypothesis": {
                    **card.model_dump(mode="json"),
                    "statement": "A may influence B through C under condition D.",
                },
            }]}

    card = HypothesisCard(
        hypothesis_id="H1",
        statement="A may influence B.",
        factual_premises=[HypothesisPremise(
            premise_id="P1", claim="A is present.", kind="evidence_backed",
            supporting_evidence_ids=["E1"], audit_verdict="supported",
        )],
        research_gap="Whether C mediates the effect.",
    )
    module = object.__new__(M4HypothesisGeneration)
    module.client = FakeClient()
    calls = []
    module._tool_call = lambda tool, operation, **kwargs: (calls.append(tool) or operation)
    context = GraphContext(
        established_facts=[GraphContextItem(
            entry_id="N1", kind="established_fact", text="A is present.",
            evidence_ids=["E1"],
        )],
        available_evidence_ids=["E1"],
    )
    output, diagnostics = await module._audit_and_repair_epistemic_structure(
        PipelineState(input_question="Does A influence B?"), [card], context,
    )

    assert calls == ["epistemic_boundary_auditor"]
    assert output[0].statement.startswith("A may influence B through C")
    assert diagnostics[0].valid
