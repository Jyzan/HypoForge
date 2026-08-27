"""Shared strict semantic audit for claims against canonical graph evidence."""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Literal, Sequence

from pydantic import BaseModel, Field

from .state import EvidenceGraph, EvidenceNodeType, HypothesisCard, HypothesisPremise


SupportStatus = Literal[
    "direct_support",
    "partial_support",
    "related_only",
    "unsupported",
    "contradicted",
]


class ClaimEvidenceVerdict(BaseModel):
    claim: str
    support_status: SupportStatus = "unsupported"
    evidence_ids: list[str] = Field(default_factory=list)
    rationale: str = ""


PremiseVerdict = Literal[
    "supported", "partially_supported", "unsupported",
    "contradicted", "invalid_citation", "not_applicable",
]


class PremiseAuditResult(BaseModel):
    """Independent audit result for one explicit M4 factual premise."""

    premise_id: str
    claim: str
    verdict: PremiseVerdict = "unsupported"
    evidence_ids: list[str] = Field(default_factory=list)
    corrected_claim: str = ""
    rationale: str = ""


class EvidenceAuditService:
    """Fail-closed semantic entailment service shared by M4/M3/M6 metrics."""

    _SEARCHABLE_TYPES = {
        EvidenceNodeType.CLAIM,
        EvidenceNodeType.EVIDENCE,
        EvidenceNodeType.LIMITATION,
        EvidenceNodeType.CONFLICT,
    }

    def __init__(
        self,
        client: object | None,
        *,
        timeout_seconds: float = 180.0,
        allow_legacy_node_ids: bool = False,
    ):
        self.client = client
        self.timeout_seconds = timeout_seconds
        self.allow_legacy_node_ids = bool(allow_legacy_node_ids)

    @staticmethod
    def decompose_hypothesis(hypothesis: HypothesisCard) -> list[str]:
        values = [
            hypothesis.statement,
            hypothesis.mechanism,
            *hypothesis.observable_predictions,
        ]
        return list(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {
            token.casefold()
            for token in re.findall(r"[A-Za-z0-9_\u4e00-\u9fff]+", text or "")
            if len(token) > 1 or token.isupper()
        }

    def _canonical_evidence_ids(self, graph: EvidenceGraph | None) -> set[str]:
        if graph is None:
            return set()
        evidence_ids: set[str] = set()
        for node in graph.nodes:
            if node.type not in self._SEARCHABLE_TYPES:
                continue
            metadata = node.metadata or {}
            evidence_ids.update(str(item) for item in metadata.get("evidence_ids", []) if str(item).strip())
            evidence_id = str(metadata.get("evidence_id") or "").strip()
            if evidence_id:
                evidence_ids.add(evidence_id)
            elif self.allow_legacy_node_ids and not metadata.get("evidence_ids"):
                evidence_ids.add(node.id)
        for edge in graph.edges:
            evidence_ids.update(str(item) for item in edge.evidence_ids if str(item).strip())
        return evidence_ids

    def _candidate_nodes(self, graph: EvidenceGraph | None, candidate_ids: set[str]):
        if graph is None:
            return []
        nodes = []
        for node in graph.nodes:
            if node.type not in self._SEARCHABLE_TYPES:
                continue
            metadata_ids = {
                str(item)
                for item in (node.metadata or {}).get("evidence_ids", [])
                if str(item).strip()
            }
            direct_id = str((node.metadata or {}).get("evidence_id") or "").strip()
            metadata_ids.update(filter(None, [direct_id]))
            if self.allow_legacy_node_ids and not metadata_ids:
                metadata_ids.add(node.id)
            ids = metadata_ids & candidate_ids if candidate_ids else metadata_ids
            if ids:
                nodes.append((node, sorted(ids)))
        return nodes

    @classmethod
    def _canonical_evidence_context(
        cls,
        graph: EvidenceGraph | None,
        allowed_ids: set[str],
    ) -> dict[str, dict[str, str]]:
        """Return exact text snippets keyed by canonical Evidence ID.

        This intentionally reads only graph/evidence provenance.  Generator
        rationale, candidate mechanisms and other hypothesis text never enter
        the premise-auditor context.
        """
        if graph is None:
            return {}
        context: dict[str, dict[str, str]] = {}

        def add_context(
            evidence_id: str,
            *,
            node_id: str,
            canonical_text: str,
            source_paper_id: str,
        ) -> None:
            current = context.get(evidence_id)
            if current is None:
                context[evidence_id] = {
                    "evidence_id": evidence_id,
                    "node_id": node_id,
                    "canonical_text": canonical_text,
                    "source_paper_id": source_paper_id,
                }
                return
            node_ids = list(dict.fromkeys(filter(None, [
                *current.get("node_id", "").split(" | "), node_id,
            ])))
            snippets = list(dict.fromkeys(filter(None, [
                *current.get("canonical_text", "").split("\n---\n"),
                canonical_text,
            ])))
            paper_ids = list(dict.fromkeys(filter(None, [
                *current.get("source_paper_id", "").split(" | "),
                source_paper_id,
            ])))
            current.update({
                "node_id": " | ".join(node_ids),
                "canonical_text": "\n---\n".join(snippets),
                "source_paper_id": " | ".join(paper_ids),
            })
        for node in graph.nodes:
            if node.type not in cls._SEARCHABLE_TYPES:
                continue
            metadata = node.metadata or {}
            ids = {
                str(item).strip()
                for item in metadata.get("evidence_ids", [])
                if str(item).strip()
            }
            direct_id = str(metadata.get("evidence_id") or "").strip()
            if direct_id:
                ids.add(direct_id)
            if cls is EvidenceAuditService and not ids:
                # Legacy node IDs are deliberately opt-in.  The strict M4
                # premise auditor receives only actual canonical Evidence IDs.
                ids = set()
            quote = str(metadata.get("quote") or "").strip()
            text = quote or str(node.label or "").strip()
            for evidence_id in ids & allowed_ids:
                add_context(
                    evidence_id,
                    node_id=node.id,
                    canonical_text=text,
                    source_paper_id=str(
                        metadata.get("paper_id") or metadata.get("source_paper_id") or ""
                    ).strip(),
                )
        for edge in graph.edges:
            for evidence_id in {
                str(item).strip() for item in edge.evidence_ids if str(item).strip()
            } & allowed_ids:
                add_context(
                    evidence_id,
                    node_id=f"{edge.source}->{edge.target}",
                    canonical_text=str(edge.rationale or "").strip(),
                    source_paper_id="",
                )
        return context

    async def audit_premises(
        self,
        premises: Sequence[HypothesisPremise],
        graph: EvidenceGraph | None,
        *,
        candidate_evidence_ids: Sequence[str] | None = None,
    ) -> list[PremiseAuditResult]:
        """Audit only factual premises against exact canonical evidence text.

        The call is intentionally separate from :meth:`audit_claims`: M4's
        generator rationale, mechanism and predictions are not supplied here.
        Missing/invalid provenance fails closed and becomes a searchable gap;
        it is never treated as support.
        """
        normalized = [
            premise for premise in premises
            if premise.kind == "evidence_backed" and str(premise.claim).strip()
        ]
        canonical_ids = self._canonical_evidence_ids(graph)
        allowed_ids = (
            canonical_ids & {str(item).strip() for item in candidate_evidence_ids or []}
            if candidate_evidence_ids is not None
            else canonical_ids
        )
        context_by_id = self._canonical_evidence_context(graph, allowed_ids)
        results: list[PremiseAuditResult | None] = [None] * len(normalized)
        pending: list[dict[str, Any]] = []

        for index, premise in enumerate(normalized):
            requested_ids = list(dict.fromkeys(
                str(item).strip()
                for item in premise.supporting_evidence_ids
                if str(item).strip()
            ))
            invalid_ids = [item for item in requested_ids if item not in context_by_id]
            valid_ids = [item for item in requested_ids if item in context_by_id]
            if invalid_ids:
                results[index] = PremiseAuditResult(
                    premise_id=premise.premise_id,
                    claim=premise.claim,
                    verdict="invalid_citation",
                    rationale="Premise cited non-canonical Evidence IDs: " + ", ".join(invalid_ids),
                )
                continue
            if not valid_ids:
                results[index] = PremiseAuditResult(
                    premise_id=premise.premise_id,
                    claim=premise.claim,
                    verdict="unsupported",
                    rationale="Factual premise has no canonical Evidence ID.",
                )
                continue
            pending.append({
                "premise_index": index,
                "premise_id": premise.premise_id,
                "claim": premise.claim,
                "evidence": [context_by_id[evidence_id] for evidence_id in valid_ids],
            })

        if pending:
            try:
                if self.client is None:
                    raise RuntimeError("premise auditor has no LLM client")
                operation = self.client.structured_chat(
                    system_prompt=(
                        "You are an independent factual-premise auditor. Review only "
                        "the supplied premise claim against the exact canonical evidence "
                        "text and IDs. Do not use generator rationale, mechanism, "
                        "predictions, topic similarity, or common knowledge. Return one "
                        "verdict per premise: supported, partially_supported, unsupported, "
                        "or contradicted. For partial support, provide a corrected_claim "
                        "that is fully entailed by the supplied evidence; otherwise leave "
                        "it empty. Use only supplied Evidence IDs."
                    ),
                    user_prompt=json.dumps({"premises": pending}, ensure_ascii=False, indent=2),
                    output_schema={
                        "type": "object",
                        "properties": {
                            "premises": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "premise_id": {"type": "string"},
                                        "verdict": {"type": "string", "enum": [
                                            "supported", "partially_supported", "unsupported", "contradicted",
                                        ]},
                                        "evidence_ids": {"type": "array", "items": {"type": "string"}},
                                        "corrected_claim": {"type": "string"},
                                        "rationale": {"type": "string"},
                                    },
                                    "required": ["premise_id", "verdict", "evidence_ids", "rationale"],
                                },
                            },
                        },
                        "required": ["premises"],
                    },
                    max_tokens=4096,
                    temperature=0.0,
                    disable_thinking=True,
                )
                payload = await asyncio.wait_for(operation, timeout=self.timeout_seconds)
                items = payload if isinstance(payload, list) else (payload or {}).get("premises", [])
                pending_by_id = {item["premise_id"]: item for item in pending}
                seen: set[str] = set()
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    premise_id = str(item.get("premise_id") or "").strip()
                    if premise_id not in pending_by_id or premise_id in seen:
                        continue
                    seen.add(premise_id)
                    source = pending_by_id[premise_id]
                    source_ids = {
                        str(entry["evidence_id"]) for entry in source["evidence"]
                    }
                    evidence_ids = list(dict.fromkeys(
                        str(value).strip()
                        for value in item.get("evidence_ids", [])
                        if str(value).strip() in source_ids
                    ))
                    verdict = str(item.get("verdict") or "unsupported")
                    if verdict not in {"supported", "partially_supported", "unsupported", "contradicted"}:
                        verdict = "unsupported"
                    if verdict in {"supported", "partially_supported"} and not evidence_ids:
                        verdict = "unsupported"
                    corrected_claim = str(item.get("corrected_claim") or "").strip()
                    if verdict != "partially_supported":
                        corrected_claim = ""
                    results[next(
                        index for index, entry in enumerate(normalized)
                        if entry.premise_id == premise_id
                    )] = PremiseAuditResult(
                        premise_id=premise_id,
                        claim=source["claim"],
                        verdict=verdict,
                        evidence_ids=evidence_ids,
                        corrected_claim=corrected_claim,
                        rationale=str(item.get("rationale") or "").strip(),
                    )
                for index, item in enumerate(results):
                    if item is None:
                        premise = normalized[index]
                        results[index] = PremiseAuditResult(
                            premise_id=premise.premise_id,
                            claim=premise.claim,
                            verdict="unsupported",
                            rationale="Premise auditor returned no verdict.",
                        )
            except Exception as exc:
                failure = f"Premise auditor failed closed: {type(exc).__name__}: {exc}"
                for index, item in enumerate(results):
                    if item is None:
                        premise = normalized[index]
                        results[index] = PremiseAuditResult(
                            premise_id=premise.premise_id,
                            claim=premise.claim,
                            verdict="unsupported",
                            rationale=failure,
                        )

        return [item for item in results if item is not None]

    def _anchor_nodes(self, claim: str, graph: EvidenceGraph | None, candidate_ids: set[str]):
        claim_tokens = self._tokens(claim)
        scored = []
        for node, evidence_ids in self._candidate_nodes(graph, candidate_ids):
            overlap = len(claim_tokens & self._tokens(node.label))
            if overlap:
                scored.append((overlap, node.label, evidence_ids))
        if not scored and self.allow_legacy_node_ids:
            # Compatibility for pre-M2 graphs whose metric caller supplied its
            # own anchor retriever but no evidence metadata. Strict M4/M3
            # callers never enable this branch.
            scored = [(0, node.label, ids) for node, ids in self._candidate_nodes(graph, candidate_ids)]
        scored.sort(key=lambda item: (-item[0], item[1]))
        return scored[:5]

    async def audit_claims(
        self,
        claims: Sequence[str],
        graph: EvidenceGraph | None,
        *,
        candidate_evidence_ids: Sequence[str] | None = None,
    ) -> list[ClaimEvidenceVerdict]:
        normalized_claims = [str(claim or "").strip() for claim in claims if str(claim or "").strip()]
        canonical_ids = self._canonical_evidence_ids(graph)
        allowed_ids = canonical_ids & set(candidate_evidence_ids) if candidate_evidence_ids is not None else canonical_ids
        verdicts: list[ClaimEvidenceVerdict | None] = [None] * len(normalized_claims)
        pending: list[dict[str, Any]] = []

        for index, claim in enumerate(normalized_claims):
            anchors = self._anchor_nodes(claim, graph, allowed_ids)
            if not anchors:
                verdicts[index] = ClaimEvidenceVerdict(
                    claim=claim,
                    support_status="unsupported",
                    rationale="No canonical evidence anchor matched this claim.",
                )
                continue
            pending.append({
                "claim_index": index,
                "target_claim": claim,
                "evidence_context": [label for _, label, _ in anchors],
                "candidate_evidence_ids": sorted({eid for _, _, ids in anchors for eid in ids}),
            })

        if pending:
            payload: Any = None
            pending_by_index = {item["claim_index"]: item for item in pending}
            try:
                if self.client is None:
                    raise RuntimeError("evidence auditor has no LLM client")
                payload = await self.client.structured_chat(
                    system_prompt=(
                        "You are a strict evidence-entailment reviewer. Topic overlap, "
                        "shared entities, an existing citation ID, and absence of contradiction "
                        "are not support. Return one verdict per claim: direct_support, "
                        "partial_support, related_only, unsupported, or contradicted. "
                        "Use only supplied evidence IDs."
                    ),
                    user_prompt=json.dumps({"claims": pending}, ensure_ascii=False, indent=2),
                    output_schema={
                        "type": "object",
                        "properties": {
                            "verdicts": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "claim_index": {"type": "integer"},
                                        "support_status": {"type": "string", "enum": [
                                            "direct_support", "partial_support", "related_only",
                                            "unsupported", "contradicted",
                                        ]},
                                        "evidence_ids": {"type": "array", "items": {"type": "string"}},
                                        "rationale": {"type": "string"},
                                    },
                                    "required": ["claim_index", "support_status", "evidence_ids", "rationale"],
                                },
                            }
                        },
                        "required": ["verdicts"],
                    },
                    max_tokens=4096,
                    temperature=0.0,
                    disable_thinking=True,
                )
                for item in (payload or {}).get("verdicts", []):
                    index = int(item.get("claim_index", -1))
                    if not 0 <= index < len(normalized_claims):
                        continue
                    status = item.get("support_status", "unsupported")
                    if status not in {"direct_support", "partial_support", "related_only", "unsupported", "contradicted"}:
                        status = "unsupported"
                    valid_ids = [
                        str(evidence_id)
                        for evidence_id in item.get("evidence_ids", [])
                        if str(evidence_id) in allowed_ids
                    ]
                    if (
                        status in {"direct_support", "partial_support"}
                        and not valid_ids
                        and self.allow_legacy_node_ids
                    ):
                        valid_ids = list(pending_by_index.get(index, {}).get("candidate_evidence_ids", []))
                    if status in {"direct_support", "partial_support"} and not valid_ids:
                        status = "unsupported"
                    verdicts[index] = ClaimEvidenceVerdict(
                        claim=normalized_claims[index],
                        support_status=status,
                        evidence_ids=list(dict.fromkeys(valid_ids)),
                        rationale=str(item.get("rationale") or "").strip(),
                    )
            except Exception as exc:
                failure = f"Evidence entailment judge failed closed: {type(exc).__name__}: {exc}"
                for item in pending:
                    index = item["claim_index"]
                    verdicts[index] = ClaimEvidenceVerdict(
                        claim=normalized_claims[index],
                        support_status="unsupported",
                        rationale=failure,
                    )

        return [item or ClaimEvidenceVerdict(
            claim=normalized_claims[index],
            support_status="unsupported",
            rationale="No verdict returned by the evidence judge.",
        ) for index, item in enumerate(verdicts)]
