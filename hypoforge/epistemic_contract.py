"""Deterministic validation and provenance normalization for M4 hypotheses."""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field

from .graph_context import GraphContext
from .state import (
    HypothesisCard,
    HypothesisGroundingStatus,
)


class EpistemicContractDiagnostic(BaseModel):
    hypothesis_id: str
    valid: bool
    codes: List[str] = Field(default_factory=list)
    messages: List[str] = Field(default_factory=list)
    removed_evidence_ids: List[str] = Field(default_factory=list)
    grounding_status: HypothesisGroundingStatus = "legacy_unknown"


def _expected_status(card: HypothesisCard) -> HypothesisGroundingStatus:
    has_facts = any(item.required for item in card.factual_premises)
    has_bridges = any(item.required for item in card.working_assumptions)
    if has_facts and has_bridges:
        return "mixed"
    if has_facts:
        return "evidence_backed"
    if has_bridges:
        return "bridge_only"
    return "legacy_unknown"


def validate_epistemic_contract(
    card: HypothesisCard,
    context: GraphContext,
    *,
    allow_legacy: bool = False,
) -> EpistemicContractDiagnostic:
    """Validate the M4 facts/assumptions boundary without calling an LLM."""

    codes: List[str] = []
    messages: List[str] = []
    valid_evidence = set(context.available_evidence_ids)
    valid_bridges = {item.entry_id for item in context.bridge_hypotheses}
    facts = [item for item in card.factual_premises if item.required]
    assumptions = [item for item in card.working_assumptions if item.required]
    expected_status = _expected_status(card)

    if card.grounding_status == "legacy_unknown" and not allow_legacy:
        codes.append("legacy_unknown")
        messages.append("new M4 output must declare a grounding status")
    if card.grounding_status != expected_status:
        codes.append("grounding_status_mismatch")
        messages.append("grounding status does not match facts and bridges")
    if not facts and not assumptions:
        codes.append("missing_grounding_structure")
        messages.append("no factual premise or M3 bridge assumption is present")
    if any(not premise.supporting_evidence_ids for premise in facts):
        codes.append("factual_premise_without_evidence")
        messages.append("a required factual premise has no canonical evidence")
    if any(
        evidence_id not in valid_evidence
        for premise in facts
        for evidence_id in premise.supporting_evidence_ids
    ):
        codes.append("invalid_evidence_id")
        messages.append("a factual premise cites evidence outside GraphContext")
    if assumptions and (
        card.supporting_evidence or card.source_paper_ids
    ):
        codes.append("bridge_claims_evidence")
        messages.append("an unverified bridge cannot claim paper provenance")
    if any(
        assumption.supporting_evidence_ids or assumption.source_paper_ids
        for assumption in assumptions
    ):
        codes.append("bridge_claims_evidence")
        messages.append("an unverified bridge cannot claim canonical evidence")
    if any(
        assumption.bridge_hypothesis_node_id not in valid_bridges
        for assumption in assumptions
    ):
        codes.append("unknown_bridge_node")
        messages.append("a working assumption references no supplied M3 node")
    if not card.research_gap.strip():
        codes.append("missing_research_gap")
        messages.append("the unresolved research gap is empty")

    return EpistemicContractDiagnostic(
        hypothesis_id=card.hypothesis_id,
        valid=not codes,
        codes=list(dict.fromkeys(codes)),
        messages=list(dict.fromkeys(messages)),
        grounding_status=card.grounding_status,
    )


def normalize_hypothesis_grounding(
    card: HypothesisCard,
    context: GraphContext,
) -> HypothesisCard:
    """Remove invalid provenance and derive card-level citations deterministically."""

    normalized = card.model_copy(deep=True)
    valid_evidence = set(context.available_evidence_ids)
    valid_bridges = {item.entry_id for item in context.bridge_hypotheses}

    normalized.factual_premises = [
        premise.model_copy(update={
            "supporting_evidence_ids": list(dict.fromkeys(
                evidence_id
                for evidence_id in premise.supporting_evidence_ids
                if evidence_id in valid_evidence
            )),
        })
        for premise in normalized.factual_premises
        if premise.kind == "evidence_backed"
    ]
    normalized.working_assumptions = [
        assumption.model_copy(update={
            "supporting_evidence_ids": [],
            "source_paper_ids": [],
        })
        for assumption in normalized.working_assumptions
        if assumption.kind == "unverified_bridge"
        and assumption.bridge_hypothesis_node_id in valid_bridges
    ]

    cited = list(dict.fromkeys(
        evidence_id
        for premise in normalized.factual_premises
        if premise.audit_verdict in {"supported", "partially_supported"}
        for evidence_id in premise.supporting_evidence_ids
    ))
    normalized.supporting_evidence = cited
    normalized.source_paper_ids = list(dict.fromkeys(
        context.evidence_to_paper[evidence_id]
        for evidence_id in cited
        if context.evidence_to_paper.get(evidence_id)
    ))
    normalized.grounding_status = _expected_status(normalized)
    return normalized
