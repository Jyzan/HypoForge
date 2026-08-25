"""Shared M4-to-M5 experimental validation coverage audit.

M5 uses this service for one bounded self-check before its result reaches M6.
M6 uses the same service again as the authoritative independent review.  The
service deliberately returns a fail-closed verdict together with a separate
error string so callers can distinguish an incomplete plan from an unavailable
auditor.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .prompts.m6_prompts import (
    M6_EXPERIMENTAL_VALIDATION_SYSTEM,
    M6_EXPERIMENTAL_VALIDATION_TEMPLATE,
)
from .state import (
    ExperimentalValidationVerdict,
    ResearchPlan,
    ValidationCoverageItem,
    ValidationTarget,
)


@dataclass(frozen=True)
class ExperimentalValidationAuditOutcome:
    """A coverage verdict plus transport/parse failure information."""

    verdict: ExperimentalValidationVerdict
    error: str = ""


class ExperimentalValidationAuditor:
    """Audit whether one M5 plan can test every non-empty M4 target."""

    def __init__(
        self,
        *,
        client: Any,
        llm_config: Optional[Any],
        timeout_seconds: float,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.client = client
        self.llm_config = llm_config
        self.timeout_seconds = float(timeout_seconds)

    @staticmethod
    def validation_targets(hypothesis: Any) -> List[ValidationTarget]:
        """Enumerate every non-empty M4 assertion that M5 must test."""
        hypothesis_id = str(getattr(hypothesis, "hypothesis_id", "") or "")
        targets: List[ValidationTarget] = []
        for target_kind, field_name in (("statement", "statement"), ("mechanism", "mechanism")):
            text = str(getattr(hypothesis, field_name, "") or "").strip()
            if text:
                targets.append(ValidationTarget(
                    target_id=f"{hypothesis_id}:{target_kind}",
                    hypothesis_id=hypothesis_id,
                    target_kind=target_kind,
                    target_text=text,
                ))
        for index, text in enumerate(getattr(hypothesis, "observable_predictions", []) or []):
            text = str(text or "").strip()
            if text:
                targets.append(ValidationTarget(
                    target_id=f"{hypothesis_id}:prediction:{index}",
                    hypothesis_id=hypothesis_id,
                    target_kind="prediction",
                    target_text=text,
                ))
        for index, text in enumerate(getattr(hypothesis, "falsification_conditions", []) or []):
            text = str(text or "").strip()
            if text:
                targets.append(ValidationTarget(
                    target_id=f"{hypothesis_id}:falsification:{index}",
                    hypothesis_id=hypothesis_id,
                    target_kind="falsification",
                    target_text=text,
                ))
        for premise in getattr(hypothesis, "working_assumptions", []) or []:
            bridge_id = str(getattr(premise, "bridge_hypothesis_node_id", "") or "").strip()
            premise_id = str(getattr(premise, "premise_id", "") or "").strip()
            text = str(getattr(premise, "claim", "") or "").strip()
            if bridge_id and text:
                target_identity = premise_id or bridge_id
                targets.append(ValidationTarget(
                    target_id=f"{hypothesis_id}:working_assumption:{target_identity}",
                    hypothesis_id=hypothesis_id,
                    target_kind="working_assumption",
                    target_text=text,
                    bridge_hypothesis_node_id=bridge_id,
                ))
        return targets

    @staticmethod
    def validation_plan_context(plan: ResearchPlan) -> Dict[str, Any]:
        """Expose indexed M5 fields so the judge can cite exact refs."""
        return {
            "procedures": {
                f"procedure:{index}": str(value or "").strip()
                for index, value in enumerate(plan.procedures)
                if str(value or "").strip()
            },
            "measurements": {
                f"metric:{index}": str(value or "").strip()
                for index, value in enumerate(plan.measurement_metrics)
                if str(value or "").strip()
            },
            "controls": {
                f"control:{index}": str(value or "").strip()
                for index, value in enumerate(plan.control_groups)
                if str(value or "").strip()
            },
            "analysis": {
                f"analysis:{index}": str(value or "").strip()
                for index, value in enumerate(plan.analysis_methods)
                if str(value or "").strip()
            },
            "expected_supported": plan.expected_results_if_supported,
            "expected_refuted": plan.expected_results_if_refuted,
            "bridge_validations": {
                f"bridge_validation:{item.bridge_hypothesis_node_id}": {
                    "procedure": item.procedure,
                    "measurement": item.measurement,
                    "falsification_condition": item.falsification_condition,
                }
                for item in plan.bridge_validations
                if item.bridge_hypothesis_node_id
            },
        }

    @staticmethod
    def validation_item_from_target(
        target: ValidationTarget,
        raw: Optional[Dict[str, Any]],
        valid_refs: Dict[str, set[str]],
        plan_context: Dict[str, Any],
    ) -> ValidationCoverageItem:
        raw = raw or {}

        def refs(name: str) -> List[str]:
            return list(dict.fromkeys(
                str(item).strip()
                for item in raw.get(name, [])
                if str(item).strip() in valid_refs[name]
            ))

        procedure_refs = refs("procedure_refs")
        measurement_refs = refs("measurement_refs")
        control_refs = refs("control_refs")
        analysis_refs = refs("analysis_refs")
        bridge_refs = refs("bridge_validation_refs")
        requested_verdict = str(raw.get("verdict") or "missing").strip().lower()
        if requested_verdict not in {"covered", "partial", "missing"}:
            requested_verdict = "missing"
        falsification_text = str(raw.get("falsification_text") or "").strip()
        rationale = str(raw.get("rationale") or "").strip()

        verdict = requested_verdict
        if target.target_kind == "working_assumption":
            bridge_id = target.bridge_hypothesis_node_id or target.target_id.rsplit(":", 1)[-1]
            bridge_candidates = [f"bridge_validation:{bridge_id}"]
            if bridge_id.startswith("BRIDGE_"):
                bridge_candidates.append(
                    f"bridge_validation:{bridge_id.removeprefix('BRIDGE_')}"
                )
            expected_bridge = next(
                (item for item in bridge_candidates
                 if item in plan_context["bridge_validations"] and item in bridge_refs),
                "",
            )
            if not expected_bridge:
                verdict = "missing"
                rationale = " ".join(filter(None, [
                    rationale,
                    f"Missing required bridge_validation for {bridge_id}.",
                ]))
            else:
                bridge_validation = plan_context["bridge_validations"][expected_bridge]
                missing_bridge_fields = [
                    field_name
                    for field_name in (
                        "procedure", "measurement", "falsification_condition",
                    )
                    if not str(bridge_validation.get(field_name) or "").strip()
                ]
                if missing_bridge_fields:
                    verdict = "missing"
                    rationale = " ".join(filter(None, [
                        rationale,
                        "Bridge validation is missing: "
                        + ", ".join(missing_bridge_fields)
                        + ".",
                    ]))
                elif not falsification_text:
                    falsification_text = str(
                        bridge_validation["falsification_condition"]
                    ).strip()
        else:
            # A non-bridge target must point into the generic M5 procedure,
            # measurement and analysis indexes.  Invalid or absent references
            # are downgraded code-side rather than trusted from the LLM's
            # boolean summary.
            has_core_refs = bool(
                procedure_refs and measurement_refs and analysis_refs
            )
            if not has_core_refs:
                verdict = (
                    "missing"
                    if not any((procedure_refs, measurement_refs, analysis_refs))
                    else "partial"
                )
                rationale = " ".join(filter(None, [
                    rationale,
                    "Missing one or more required procedure, measurement, or analysis references.",
                ]))
        if target.target_kind == "falsification" and not falsification_text:
            verdict = "partial" if verdict == "covered" else verdict
            rationale = " ".join(filter(None, [
                rationale,
                "No explicit falsification decision criterion was supplied.",
            ]))
        if target.target_kind in {"statement", "mechanism", "prediction"}:
            lower_target = target.target_text.casefold()
            analyses = " ".join(plan_context["analysis"].values()).casefold()
            if any(token in lower_target for token in ("synerg", "interaction", "协同", "交互")) and "interaction" not in analyses and "交互" not in analyses:
                verdict = "partial" if verdict == "covered" else verdict
                rationale = " ".join(filter(None, [
                    rationale,
                    "The plan does not specify an interaction-term analysis for a synergy target.",
                ]))
        return ValidationCoverageItem(
            target_id=target.target_id,
            target_kind=target.target_kind,
            target_text=target.target_text,
            verdict=verdict,
            procedure_refs=procedure_refs,
            measurement_refs=measurement_refs,
            control_refs=control_refs,
            analysis_refs=analysis_refs,
            bridge_validation_refs=bridge_refs,
            falsification_text=falsification_text,
            rationale=rationale,
        )

    async def audit(
        self,
        hypothesis: Any,
        plan: ResearchPlan,
        version: int,
    ) -> ExperimentalValidationAuditOutcome:
        """Run the LLM audit and apply deterministic reference validation."""
        targets = self.validation_targets(hypothesis)
        plan_context = self.validation_plan_context(plan)
        if not targets:
            return ExperimentalValidationAuditOutcome(
                verdict=ExperimentalValidationVerdict(
                    sufficient=True,
                    rationale="M4 produced no non-empty validation target.",
                )
            )
        valid_refs = {
            "procedure_refs": set(plan_context["procedures"]),
            "measurement_refs": set(plan_context["measurements"]),
            "control_refs": set(plan_context["controls"]),
            "analysis_refs": set(plan_context["analysis"]),
            "bridge_validation_refs": set(plan_context["bridge_validations"]),
        }
        user_payload = {
            "validation_targets": [target.model_dump(mode="json") for target in targets],
            "m5_plan_index": plan_context,
            "review_version": version,
        }
        try:
            response = await asyncio.wait_for(
                self.client.structured_chat(
                    system_prompt=M6_EXPERIMENTAL_VALIDATION_SYSTEM,
                    user_prompt=M6_EXPERIMENTAL_VALIDATION_TEMPLATE.format(
                        validation_payload=json.dumps(user_payload, ensure_ascii=False, indent=2),
                    ),
                    output_schema=ExperimentalValidationVerdict.model_json_schema(),
                    max_tokens=8192,
                    temperature=getattr(self.llm_config, "temperature", 0.1) if self.llm_config else 0.1,
                    disable_thinking=True,
                ),
                timeout=self.timeout_seconds,
            )
            parsed = ExperimentalValidationVerdict.model_validate(dict(response or {}))
            raw_by_id: Dict[str, Dict[str, Any]] = {}
            for item in parsed.items:
                raw_by_id.setdefault(item.target_id, item.model_dump(mode="json"))
            items = [
                self.validation_item_from_target(
                    target, raw_by_id.get(target.target_id), valid_refs, plan_context,
                )
                for target in targets
            ]
            sufficient = all(item.verdict == "covered" for item in items)
            rationale = " ".join(filter(None, [
                parsed.rationale,
                "All required M4 targets have explicit M5 coverage." if sufficient
                else "At least one required M4 target lacks complete M5 coverage.",
            ]))
            return ExperimentalValidationAuditOutcome(
                verdict=ExperimentalValidationVerdict(
                    sufficient=sufficient,
                    items=items,
                    rationale=rationale,
                )
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            items = [ValidationCoverageItem(
                target_id=target.target_id,
                target_kind=target.target_kind,
                target_text=target.target_text,
                verdict="missing",
                rationale=f"Experimental validation audit failed closed: {error}",
            ) for target in targets]
            return ExperimentalValidationAuditOutcome(
                verdict=ExperimentalValidationVerdict(
                    sufficient=False,
                    items=items,
                    rationale=f"Experimental validation audit failed closed: {error}",
                ),
                error=error,
            )
