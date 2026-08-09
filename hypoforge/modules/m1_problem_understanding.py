"""
M1: Problem Understanding & Decomposition.

Decomposes a frontier scientific question into structured sub-questions,
identifies domains, audits source-bounded key entities, and builds a task
contract. No question-category label is generated or consumed.

Iteration core: when ``followup_routing`` is enabled and the state carries a
``FollowupRequest``, M1 additionally decides ``skip_search`` — whether the
follow-up introduces new entities/mechanisms/domains (needs a fresh search)
or merely refines direction (existing evidence base is reusable).

Output: ``ProblemCard`` in ``state.problem_card`` (+ ``followup.skip_search``
in the followup path).
"""

from __future__ import annotations

import json
import logging
import re
import time
import unicodedata
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from ..observability import emit_event
from ..protocol import ModuleProtocol
from ..prompts.m1_prompts import (
    M1_COVERAGE_CHECK_SYSTEM_PROMPT,
    M1_COVERAGE_CHECK_USER_TEMPLATE,
    M1_COVERAGE_MERGE_SYSTEM_PROMPT,
    M1_COVERAGE_MERGE_USER_TEMPLATE,
    M1_COVERAGE_SUPPLEMENT_SYSTEM_PROMPT,
    M1_COVERAGE_SUPPLEMENT_USER_TEMPLATE,
    M1_ENTITY_AUDIT_SYSTEM_PROMPT,
    M1_ENTITY_AUDIT_USER_TEMPLATE,
    M1_ENTITY_EXTRACTION_SYSTEM_PROMPT,
    M1_ENTITY_EXTRACTION_USER_TEMPLATE,
    M1_ENTITY_REPAIR_NOTE_TEMPLATE,
    M1_REQUIREMENT_SYSTEM_PROMPT,
    M1_REQUIREMENT_USER_TEMPLATE,
    M1_FOLLOWUP_SYSTEM_PROMPT,
    M1_FOLLOWUP_USER_TEMPLATE,
    M1_SYSTEM_PROMPT,
    M1_USER_TEMPLATE,
)
from ..registry import ModuleRegistry
from ..state import (
    PipelineState,
    ProblemCard,
    TaskContract,
    TaskEntity,
    TaskRequirement,
)
from ..tools.qwen_client import QwenClient

logger = logging.getLogger(__name__)


class _CandidateDecomposition(BaseModel):
    """First-pass output intentionally excludes entities and contracts."""

    domain: List[str] = []
    sub_questions: List[str] = []


class _FollowupTriageDecision(BaseModel):
    """Routing-only output.  It can never rewrite the parent ProblemCard."""

    category: Literal[
        "presentation_adjustment",
        "evidence_reuse_refinement",
        "research_change",
    ]
    skip_search: bool
    rebuild_problem_card: bool
    rationale: str
    confidence: float


class _SubQuestionCoverage(BaseModel):
    sufficient: bool = False
    core_intent_covered: bool = False
    missing_aspects: List[str] = []
    over_fragmented: bool = False
    merge_instructions: List[str] = []
    reason: str = ""


class _SubQuestionSupplement(BaseModel):
    sub_questions: List[str] = []


class _CandidateEntity(BaseModel):
    name: str
    source_mention: str
    aliases: List[str] = Field(default_factory=list)
    role: Literal[
        "primary_object", "intervention", "outcome", "method",
        "context", "constraint", "other",
    ] = "other"
    required: bool = False
    extraction_reason: str = ""


class _CandidateEntityList(BaseModel):
    entities: List[_CandidateEntity] = Field(default_factory=list)


class _EntityAuditItem(BaseModel):
    candidate_name: str
    accepted: bool
    source_mention: str = ""
    reason: str = ""


class _EntitySourceAudit(BaseModel):
    items: List[_EntityAuditItem] = Field(default_factory=list)
    missing_explicit_entities: List[str] = Field(default_factory=list)
    complete: bool = False


class _RequirementList(BaseModel):
    requirements: List[TaskRequirement] = Field(default_factory=list)


@ModuleRegistry.register
class M1ProblemUnderstanding(ModuleProtocol):
    module_name = "m1"
    module_version = "0.3.0"
    description = "Problem decomposition: sub-questions, domain tagging, entity extraction"
    MAX_SUB_QUESTIONS = 5

    @staticmethod
    def _sub_question_violations(sub_questions: List[str]) -> List[str]:
        violations: List[str] = []
        if len(sub_questions) > M1ProblemUnderstanding.MAX_SUB_QUESTIONS:
            violations.append(
                "sub-question list must contain at most 5 items; "
                f"found {len(sub_questions)}"
            )
        for index, value in enumerate(sub_questions, start=1):
            text = " ".join(str(value or "").split())
            reasons: List[str] = []
            if len(text) > 240:
                reasons.append("longer than 240 characters")
            if text.count("?") + text.count("？") > 1:
                reasons.append("contains multiple question marks")
            if re.search(r"[;；]", text):
                reasons.append("contains a semicolon/parallel clause")
            if re.search(r"[（(][^）)]*[,，、;/；][^）)]*[）)]", text):
                reasons.append("contains a parenthesized enumeration")
            if reasons:
                violations.append(f"sub_question[{index}]: {', '.join(reasons)}")
        return violations

    @staticmethod
    def _contract_violations(card: ProblemCard) -> List[str]:
        """Validate structure/coverage without encoding any scientific domain."""

        contract = card.task_contract
        violations: List[str] = []
        if contract.source != "m1":
            violations.append("task_contract was omitted and only a legacy contract was derived")
        required_primary = [
            entity for entity in contract.entities
            if entity.role == "primary_object" and entity.required
        ]
        if not required_primary:
            violations.append("task_contract has no required primary_object")
        requirements_by_question: Dict[str, int] = {}
        for requirement in contract.requirements:
            key = " ".join(requirement.sub_question.split()).casefold()
            requirements_by_question[key] = requirements_by_question.get(key, 0) + 1
            if not requirement.primary_entity_id:
                violations.append(
                    f"{requirement.requirement_id}: primary_entity_id is empty"
                )
            if not requirement.relation:
                violations.append(f"{requirement.requirement_id}: relation is empty")
        for index, question in enumerate(card.sub_questions, start=1):
            key = " ".join(question.split()).casefold()
            count = requirements_by_question.get(key, 0)
            if count != 1:
                violations.append(
                    f"sub_question[{index}] must map to exactly one requirement; found {count}"
                )
        return violations

    # ------------------------------------------------------------------
    # ModuleProtocol implementation
    # ------------------------------------------------------------------

    def __init__(
        self,
        mode: str = "llm",
        llm_config: Optional[Any] = None,
        coverage_max_rounds: int = 2,
        entity_repair_attempts: int = 1,
        requirement_repair_attempts: int = 1,
        followup_routing: bool = False,
        followup_triage_confidence_threshold: float = 0.75,
        **kwargs,
    ):
        self.mode = mode
        self.llm_config = llm_config
        self.coverage_max_rounds = max(0, int(coverage_max_rounds))
        self.entity_repair_attempts = max(0, int(entity_repair_attempts))
        self.requirement_repair_attempts = max(0, int(requirement_repair_attempts))
        # Iteration-core switch: enables mandatory LLM follow-up triage.
        self.followup_routing = followup_routing
        self.followup_triage_confidence_threshold = max(
            0.0, min(1.0, float(followup_triage_confidence_threshold))
        )
        self.client = QwenClient.from_config(llm_config) if llm_config else None

    async def __call__(
        self,
        state: PipelineState,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if self.client is None:
            raise RuntimeError(
                "M1 requires an LLM client — pass llm_config / set OPENAI_API_KEY."
            )

        # --- iteration core: followup search-free triage ---
        if self.followup_routing and state.followup is not None:
            try:
                return await self._understand_followup(state)
            except Exception as exc:
                # Parse/LLM failure ⇒ prefer searching over missing evidence.
                logger.warning(
                    "Followup triage failed (%s: %s); falling back to standard "
                    "M1 with skip_search=False",
                    type(exc).__name__,
                    exc,
                )

        card = await self._understand_question(state.input_question)
        patch: Dict[str, Any] = {"problem_card": card}
        if self.followup_routing and state.followup is not None:
            patch["followup"] = state.followup.model_copy(update={"skip_search": False})
        return patch

    # ------------------------------------------------------------------
    # Standard question understanding (legacy behaviour, unchanged)
    # ------------------------------------------------------------------

    async def _understand_question(self, question: str) -> ProblemCard:
        started_at = time.monotonic()
        emit_event(
            "tool_started",
            module="m1",
            tool="qwen_problem_understanding",
            status="running",
            message="Qwen 开始拆分科学问题",
        )
        try:
            payload = await self.client.structured_chat(
                system_prompt=M1_SYSTEM_PROMPT,
                user_prompt=M1_USER_TEMPLATE.format(question=question),
                output_schema=_CandidateDecomposition.model_json_schema(),
                max_tokens=8192,
                temperature=getattr(self.llm_config, "temperature", 0.1),
            )
        except BaseException as exc:
            emit_event(
                "tool_failed",
                module="m1",
                tool="qwen_problem_understanding",
                status="failed",
                message=f"问题拆分失败：{type(exc).__name__}: {exc}",
                elapsed_seconds=time.monotonic() - started_at,
            )
            raise
        decomposition = _CandidateDecomposition.model_validate(payload)
        violations = self._sub_question_violations(decomposition.sub_questions)
        if violations:
            retry_prompt = "\n".join([
                M1_USER_TEMPLATE.format(question=question),
                "The previous decomposition violated the atomic-sub-question contract:",
                *[f"- {violation}" for violation in violations],
                "Return corrected domains and sub-questions only.",
            ])
            retry_payload = await self.client.structured_chat(
                system_prompt=M1_SYSTEM_PROMPT,
                user_prompt=retry_prompt,
                output_schema=_CandidateDecomposition.model_json_schema(),
                max_tokens=8192,
                temperature=0.0,
            )
            retried = _CandidateDecomposition.model_validate(retry_payload)
            retry_violations = self._sub_question_violations(
                retried.sub_questions
            )
            if retry_violations:
                raise ValueError(
                    "Decomposition still violates the atomic task contract "
                    "after deterministic repair: "
                    + "; ".join(retry_violations)
                )
            decomposition = retried
        sub_questions = await self._check_subquestion_coverage(
            question,
            decomposition.sub_questions,
        )
        final_sub_question_violations = self._sub_question_violations(
            sub_questions
        )
        if final_sub_question_violations:
            raise ValueError(
                "Final decomposition violates the sub-question contract: "
                + "; ".join(final_sub_question_violations)
            )
        entities = await self._extract_and_audit_entities(question)
        requirements = await self._build_requirements(sub_questions, entities)
        card = ProblemCard(
            original_question=question,
            domain=decomposition.domain,
            sub_questions=sub_questions,
            task_contract=TaskContract(
                source="m1",
                entities=entities,
                requirements=requirements,
            ),
        )
        final_violations = self._contract_violations(card)
        if final_violations:
            raise ValueError(
                "M1 final task contract validation failed: "
                + "; ".join(final_violations)
            )
        emit_event(
            "tool_completed",
            module="m1",
            tool="qwen_problem_understanding",
            status="completed",
            message=f"问题拆分完成：{len(card.sub_questions)} 个子问题",
            elapsed_seconds=time.monotonic() - started_at,
            details={
                "sub_questions": list(card.sub_questions),
                "domains": list(card.domain),
                "key_entities": list(card.key_entities),
                "task_entities": len(card.task_contract.entities),
                "task_requirements": len(card.task_contract.requirements),
            },
        )
        return card

    async def _check_subquestion_coverage(
        self,
        question: str,
        sub_questions: List[str],
    ) -> List[str]:
        """Audit, supplement, and merge sub-questions within a bounded loop."""

        questions = list(dict.fromkeys(
            " ".join(str(item or "").split())
            for item in sub_questions
            if str(item or "").strip()
        ))
        if not questions or self.coverage_max_rounds <= 0:
            return questions

        started_at = time.monotonic()

        def render(items: List[str]) -> str:
            return "\n".join(f"- {item}" for item in items)

        for round_index in range(1, self.coverage_max_rounds + 1):
            payload = await self.client.structured_chat(
                system_prompt=M1_COVERAGE_CHECK_SYSTEM_PROMPT,
                user_prompt=M1_COVERAGE_CHECK_USER_TEMPLATE.format(
                    question=question,
                    sub_questions_text=render(questions),
                ),
                output_schema=_SubQuestionCoverage.model_json_schema(),
                max_tokens=4096,
                temperature=0.0,
            )
            audit = _SubQuestionCoverage.model_validate(payload)
            emit_event(
                "m1_subquestion_coverage",
                module="m1",
                tool="qwen_subquestion_coverage",
                status="running",
                message=f"子问题覆盖审查第 {round_index} 轮完成",
                details={
                    "round": round_index,
                    "sufficient": audit.sufficient,
                    "core_intent_covered": audit.core_intent_covered,
                    "missing_aspects": list(audit.missing_aspects),
                    "over_fragmented": audit.over_fragmented,
                    "merge_instructions": list(audit.merge_instructions),
                },
            )

            if (
                audit.sufficient
                and audit.core_intent_covered
                and not audit.over_fragmented
            ):
                emit_event(
                    "m1_subquestion_coverage",
                    module="m1",
                    tool="qwen_subquestion_coverage",
                    status="completed",
                    message=f"子问题覆盖审查通过（第 {round_index} 轮）",
                    elapsed_seconds=time.monotonic() - started_at,
                    details={"sub_question_count": len(questions)},
                )
                return questions

            if audit.over_fragmented:
                merged_payload = await self.client.structured_chat(
                    system_prompt=M1_COVERAGE_MERGE_SYSTEM_PROMPT,
                    user_prompt=M1_COVERAGE_MERGE_USER_TEMPLATE.format(
                        question=question,
                        sub_questions_text=render(questions),
                        merge_instructions_text=render(audit.merge_instructions),
                    ),
                    output_schema=_SubQuestionSupplement.model_json_schema(),
                    max_tokens=4096,
                    temperature=0.0,
                )
                merged = _SubQuestionSupplement.model_validate(merged_payload)
                replacement = [
                    " ".join(item.split())
                    for item in merged.sub_questions
                    if item.strip()
                ]
                if replacement:
                    questions = list(dict.fromkeys(replacement))

            if not audit.sufficient:
                supplement_payload = await self.client.structured_chat(
                    system_prompt=M1_COVERAGE_SUPPLEMENT_SYSTEM_PROMPT,
                    user_prompt=M1_COVERAGE_SUPPLEMENT_USER_TEMPLATE.format(
                        question=question,
                        sub_questions_text=render(questions),
                        missing_aspects_text=render(audit.missing_aspects),
                    ),
                    output_schema=_SubQuestionSupplement.model_json_schema(),
                    max_tokens=4096,
                    temperature=0.0,
                )
                supplement = _SubQuestionSupplement.model_validate(
                    supplement_payload
                )
                questions = list(dict.fromkeys([
                    *questions,
                    *(
                        " ".join(item.split())
                        for item in supplement.sub_questions
                        if item.strip()
                    ),
                ]))

            if len(questions) > self.MAX_SUB_QUESTIONS:
                questions = await self._merge_subquestions_to_limit(
                    question,
                    questions,
                )

            violations = self._sub_question_violations(questions)
            if violations:
                raise ValueError(
                    "coverage repair produced non-atomic sub-questions: "
                    + "; ".join(violations)
                )

        final_payload = await self.client.structured_chat(
            system_prompt=M1_COVERAGE_CHECK_SYSTEM_PROMPT,
            user_prompt=M1_COVERAGE_CHECK_USER_TEMPLATE.format(
                question=question,
                sub_questions_text=render(questions),
            ),
            output_schema=_SubQuestionCoverage.model_json_schema(),
            max_tokens=4096,
            temperature=0.0,
        )
        final_audit = _SubQuestionCoverage.model_validate(final_payload)
        if not final_audit.core_intent_covered:
            raise ValueError("core user intent remains uncovered after coverage repair")
        if final_audit.over_fragmented:
            raise ValueError("sub-questions remain over-fragmented after coverage repair")
        emit_event(
            "m1_subquestion_coverage",
            module="m1",
            tool="qwen_subquestion_coverage",
            status="warning",
            message="覆盖轮次耗尽，保留已覆盖核心意图的子问题",
            elapsed_seconds=time.monotonic() - started_at,
            details={
                "missing_aspects": list(final_audit.missing_aspects),
                "sub_question_count": len(questions),
            },
        )
        return questions

    async def _merge_subquestions_to_limit(
        self,
        question: str,
        sub_questions: List[str],
    ) -> List[str]:
        """Semantically merge overflow instead of dropping questions by position."""

        rendered = "\n".join(f"- {item}" for item in sub_questions)
        payload = await self.client.structured_chat(
            system_prompt=M1_COVERAGE_MERGE_SYSTEM_PROMPT,
            user_prompt=M1_COVERAGE_MERGE_USER_TEMPLATE.format(
                question=question,
                sub_questions_text=rendered,
                merge_instructions_text=(
                    "Merge overlapping or adjacent aspects so the complete "
                    "list contains at most 5 atomic sub-questions. Preserve "
                    "the original core action and every indispensable aspect."
                ),
            ),
            output_schema=_SubQuestionSupplement.model_json_schema(),
            max_tokens=4096,
            temperature=0.0,
        )
        merged = _SubQuestionSupplement.model_validate(payload)
        result = list(dict.fromkeys(
            " ".join(item.split())
            for item in merged.sub_questions
            if item.strip()
        ))
        violations = self._sub_question_violations(result)
        if violations:
            raise ValueError(
                "sub-question limit merge produced an invalid decomposition: "
                + "; ".join(violations)
            )
        if not result:
            raise ValueError("sub-question limit merge returned no questions")
        return result

    @staticmethod
    def _normalize_source_text(value: str) -> str:
        """Normalize only for literal source checks, never for entity invention."""

        return " ".join(
            unicodedata.normalize("NFKC", str(value or "")).casefold().split()
        )

    async def _extract_and_audit_entities(self, question: str) -> List[TaskEntity]:
        """Extract entities from the original question and independently audit them.

        The LLM audit is intentionally followed by a deterministic literal-source
        gate.  Therefore even an overly agreeable auditor cannot legitimize a term
        introduced by a generated sub-question or by model background knowledge.
        """

        source = self._normalize_source_text(question)
        audit_feedback = ""
        last_missing: List[str] = []

        for attempt in range(self.entity_repair_attempts + 1):
            extraction_prompt = M1_ENTITY_EXTRACTION_USER_TEMPLATE.format(
                question=question,
            )
            if audit_feedback:
                extraction_prompt += M1_ENTITY_REPAIR_NOTE_TEMPLATE.format(
                    audit_feedback=audit_feedback,
                )
            candidate_payload = await self.client.structured_chat(
                system_prompt=M1_ENTITY_EXTRACTION_SYSTEM_PROMPT,
                user_prompt=extraction_prompt,
                output_schema=_CandidateEntityList.model_json_schema(),
                max_tokens=4096,
                temperature=0.0,
            )
            candidate_list = _CandidateEntityList.model_validate(candidate_payload)
            audit_payload = await self.client.structured_chat(
                system_prompt=M1_ENTITY_AUDIT_SYSTEM_PROMPT,
                user_prompt=M1_ENTITY_AUDIT_USER_TEMPLATE.format(
                    question=question,
                    candidate_entities_json=json.dumps(
                        [item.model_dump(mode="json") for item in candidate_list.entities],
                        ensure_ascii=False,
                        indent=2,
                    ),
                ),
                output_schema=_EntitySourceAudit.model_json_schema(),
                max_tokens=4096,
                temperature=0.0,
            )
            audit = _EntitySourceAudit.model_validate(audit_payload)
            audit_by_name = {
                self._normalize_source_text(item.candidate_name): item
                for item in audit.items
            }

            accepted: List[TaskEntity] = []
            seen: set[str] = set()
            for candidate in candidate_list.entities:
                name = self._normalize_source_text(candidate.name)
                mention = self._normalize_source_text(candidate.source_mention)
                audited = audit_by_name.get(name)
                audited_mention = self._normalize_source_text(
                    audited.source_mention if audited is not None else ""
                )
                literal_source_valid = bool(
                    name
                    and name == mention
                    and name in source
                    and audited_mention == name
                )
                if (
                    audited is None
                    or not audited.accepted
                    or not literal_source_valid
                    or name in seen
                ):
                    continue
                seen.add(name)
                accepted.append(TaskEntity(
                    entity_id=f"E{len(accepted) + 1}",
                    name=candidate.name,
                    source_mention=candidate.source_mention,
                    aliases=candidate.aliases,
                    role=candidate.role,
                    required=candidate.required,
                ))

            has_required_primary = any(
                item.role == "primary_object" and item.required
                for item in accepted
            )
            last_missing = list(audit.missing_explicit_entities)
            if audit.complete and not last_missing and has_required_primary:
                return accepted

            audit_feedback = "; ".join(filter(None, [
                (
                    "No accepted required primary object was grounded in the "
                    "original question."
                    if not has_required_primary
                    else ""
                ),
                (
                    "Missing explicit entities: " + ", ".join(last_missing)
                    if last_missing
                    else ""
                ),
                "The audit marked the extraction incomplete." if not audit.complete else "",
            ]))

            emit_event(
                "m1_entity_audit",
                module="m1",
                tool="qwen_entity_source_audit",
                status="warning" if attempt < self.entity_repair_attempts else "failed",
                message="关键实体来源复核未通过，正在修复" if attempt < self.entity_repair_attempts else "关键实体来源复核未通过",
                details={
                    "attempt": attempt + 1,
                    "accepted_entities": [item.name for item in accepted],
                    "missing_explicit_entities": last_missing,
                    "has_required_primary": has_required_primary,
                },
            )

        if not any(
            item.role == "primary_object" and item.required
            for item in accepted
        ):
            raise ValueError(
                "entity extraction has no required primary object grounded in "
                "the original user question"
            )
        raise ValueError(
            "entity source audit remains incomplete after bounded repair"
            + (f": missing {last_missing}" if last_missing else "")
        )

    @staticmethod
    def _requirement_violations(
        requirements: List[TaskRequirement],
        sub_questions: List[str],
        entities: List[TaskEntity],
    ) -> List[str]:
        """Validate requirement structure without changing audited entities."""

        violations: List[str] = []
        known_ids = {item.entity_id for item in entities}
        required_questions = [" ".join(item.split()) for item in sub_questions]
        actual_questions = [" ".join(item.sub_question.split()) for item in requirements]
        if actual_questions != required_questions:
            violations.append(
                "requirements must map one-to-one to final sub-questions in order"
            )
        requirement_ids = [item.requirement_id for item in requirements]
        expected_ids = [f"R{index}" for index in range(1, len(sub_questions) + 1)]
        if requirement_ids != expected_ids:
            violations.append("requirement IDs must be stable R1..Rn in order")
        for item in requirements:
            if not item.primary_entity_id:
                violations.append(f"{item.requirement_id}: missing primary_entity_id")
            elif item.primary_entity_id not in known_ids:
                violations.append(f"{item.requirement_id}: unknown primary entity")
            if any(entity_id not in known_ids for entity_id in item.related_entity_ids):
                violations.append(f"{item.requirement_id}: unknown related entity")
            if not item.relation.strip():
                violations.append(f"{item.requirement_id}: empty relation")
        return violations

    async def _build_requirements(
        self,
        sub_questions: List[str],
        entities: List[TaskEntity],
    ) -> List[TaskRequirement]:
        """Map final questions onto immutable, audited entity IDs."""

        questions_json = json.dumps(sub_questions, ensure_ascii=False, indent=2)
        entities_json = json.dumps(
            [item.model_dump(mode="json") for item in entities],
            ensure_ascii=False,
            indent=2,
        )
        last_violations: List[str] = []
        for attempt in range(self.requirement_repair_attempts + 1):
            repair_note = ""
            if attempt:
                repair_note = (
                    "\nThe previous mapping failed deterministic validation. "
                    "Rebuild it using exactly the listed questions and entity IDs."
                )
            payload = await self.client.structured_chat(
                system_prompt=M1_REQUIREMENT_SYSTEM_PROMPT,
                user_prompt=M1_REQUIREMENT_USER_TEMPLATE.format(
                    sub_questions_json=questions_json,
                    entities_json=entities_json,
                    repair_note=repair_note,
                ),
                output_schema=_RequirementList.model_json_schema(),
                max_tokens=4096,
                temperature=0.0,
            )
            result = _RequirementList.model_validate(payload)
            last_violations = self._requirement_violations(
                result.requirements,
                sub_questions,
                entities,
            )
            if not last_violations:
                return result.requirements

        raise ValueError(
            "requirement mapping remains invalid after bounded repair: "
            + "; ".join(last_violations)
        )

    # ------------------------------------------------------------------
    # Followup triage (iteration core)
    # ------------------------------------------------------------------

    async def _understand_followup(self, state: PipelineState) -> Dict[str, Any]:
        """Use an LLM for routing only; never let triage rewrite the parent card."""
        followup = state.followup
        assert followup is not None  # caller guarantees

        # Low-cost graph overview (counts only) when available.
        graph = state.evidence_graph
        if graph is not None:
            graph_overview = (
                "Evidence graph overview: "
                f"{len(graph.nodes)} nodes, {len(graph.edges)} edges, "
                f"{len(graph.established_facts)} established facts, "
                f"{len(graph.conflicts)} conflicts, "
                f"{len(graph.knowledge_gaps)} knowledge gaps."
            )
        else:
            graph_overview = "(no evidence graph available)"
        problem_card_json = (
            json.dumps(state.problem_card.model_dump(mode="json"), ensure_ascii=False, indent=2)
            if state.problem_card is not None
            else "(none)"
        )
        parent_artifacts_summary = self._parent_artifacts_summary(state)

        started_at = time.monotonic()
        emit_event(
            "tool_started",
            module="m1",
            tool="qwen_followup_triage",
            status="running",
            message="Qwen 开始判定追问是否免检索",
            details={"parent_run_id": followup.parent_run_id},
        )
        try:
            payload = await self.client.structured_chat(
                system_prompt=M1_FOLLOWUP_SYSTEM_PROMPT,
                user_prompt=M1_FOLLOWUP_USER_TEMPLATE.format(
                    problem_card_json=problem_card_json,
                    followup_text=followup.text,
                    graph_overview=graph_overview,
                    parent_artifacts_summary=parent_artifacts_summary,
                ),
                output_schema=_FollowupTriageDecision.model_json_schema(),
                max_tokens=8192,
                temperature=0.0,
            )
            decision = _FollowupTriageDecision.model_validate(payload)
        except BaseException as exc:
            emit_event(
                "tool_failed",
                module="m1",
                tool="qwen_followup_triage",
                status="failed",
                message=f"追问判定失败：{type(exc).__name__}: {exc}",
                elapsed_seconds=time.monotonic() - started_at,
            )
            raise

        expected = {
            "presentation_adjustment": (True, False),
            "evidence_reuse_refinement": (True, False),
            "research_change": (False, True),
        }[decision.category]
        graph_reusable = bool(
            state.evidence_graph is not None
            and (state.evidence_graph.nodes or state.evidence_graph.edges)
        )
        decision_consistent = (
            (decision.skip_search, decision.rebuild_problem_card) == expected
        )
        confident = decision.confidence >= self.followup_triage_confidence_threshold
        skip_search = bool(
            decision.skip_search
            and decision_consistent
            and confident
            and graph_reusable
            and state.problem_card is not None
        )
        card = (
            state.problem_card.model_copy(deep=True)
            if state.problem_card is not None
            else None
        )
        if decision.category == "research_change":
            parent_question = (
                state.problem_card.original_question
                if state.problem_card is not None
                else state.input_question
            )
            rebuild_input = (
                "Original research question:\n"
                f"{parent_question}\n\n"
                "User's research-changing follow-up:\n"
                f"{followup.text}"
            )
            card = await self._understand_question(rebuild_input)
        emit_event(
            "tool_completed",
            module="m1",
            tool="qwen_followup_triage",
            status="completed",
            message=(
                f"追问判定完成：{'免检索（skip_search）' if skip_search else '需要新检索'}"
            ),
            elapsed_seconds=time.monotonic() - started_at,
            details={
                "skip_search": skip_search,
                "rationale": decision.rationale,
                "decision_source": "llm_triage",
                "category": decision.category,
                "confidence": decision.confidence,
                "decision_consistent": decision_consistent,
                "reusable_graph": graph_reusable,
            },
        )
        patch: Dict[str, Any] = {
            "followup": followup.model_copy(update={"skip_search": skip_search}),
        }
        if card is not None:
            patch["problem_card"] = card
        return patch

    @staticmethod
    def _parent_artifacts_summary(state: PipelineState) -> str:
        """Summarise what the parent run already produced, so triage knows
        the knowledge base is complete unless the follow-up is genuinely new.
        """
        paper_total = sum(r.papers_retrieved for r in state.literature_results)
        lines: List[str] = ["Parent run artefacts:"]
        if state.literature_results:
            lines.append(
                f"- literature_results: {len(state.literature_results)} "
                f"sub-question result(s), {paper_total} paper(s) already retrieved"
            )
        else:
            lines.append("- literature_results: none")
        lines.append(
            "- best_hypotheses: "
            + (
                f"already produced ({len(state.best_hypotheses)} card(s))"
                if state.best_hypotheses
                else "not yet produced"
            )
        )
        lines.append(
            "- research_plans: "
            + (
                f"already produced ({len(state.research_plans)} plan(s))"
                if state.research_plans
                else "not yet produced"
            )
        )
        lines.append(
            "The parent run's knowledge base is as complete as summarised "
            "above; trigger a new literature search ONLY if the follow-up "
            "introduces a genuinely new scientific or engineering direction."
        )
        return "\n".join(lines)

    @classmethod
    def get_input_fields(cls) -> List[str]:
        return ["input_question", "followup", "problem_card", "evidence_graph"]

    @classmethod
    def get_output_fields(cls) -> List[str]:
        return ["problem_card"]
