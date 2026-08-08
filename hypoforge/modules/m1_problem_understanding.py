"""
M1: Problem Understanding & Decomposition.

Decomposes a frontier scientific question into structured sub-questions,
identifies domains, extracts key entities, and classifies the question type.

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
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel

from ..observability import emit_event
from ..protocol import ModuleProtocol
from ..prompts.m1_prompts import (
    M1_FOLLOWUP_SYSTEM_PROMPT,
    M1_FOLLOWUP_USER_TEMPLATE,
    M1_SYSTEM_PROMPT,
    M1_USER_TEMPLATE,
)
from ..registry import ModuleRegistry
from ..state import PipelineState, ProblemCard
from ..tools.qwen_client import QwenClient

logger = logging.getLogger(__name__)


class _FollowupUnderstanding(BaseModel):
    """Structured output schema for the followup triage call."""

    problem_card: ProblemCard
    skip_search: bool = False
    rationale: str = ""


# ---------------------------------------------------------------------------
# Deterministic shortcut for pure formatting / language / presentation
# follow-ups.
#
# A match short-circuits triage to ``skip_search=True`` with the parent
# ProblemCard reused verbatim, *before* any LLM call — so the triage model
# can never reinterpret e.g. "请你给我中文方案" as a new research direction
# and invent extra entities to justify a fresh search.
#
# The whitelist is deliberately conservative: every pattern is anchored
# (``fullmatch`` on whitespace-stripped, length-bounded text), so semantic
# follow-ups such as "聚焦阿尔茨海默方向" can never match.
# ---------------------------------------------------------------------------

_FO_PREFIX = r"(?:请(?:你)?给(?:我)?|请(?:你)?|麻烦(?:你)?|帮我|给我)?"
_FO_LANG = r"(?:中文|简体中文|繁體中文|繁体中文|英文|英语)"
_FO_ARTIFACT = r"(?:研究方案|方案|结果|内容|回答|报告)"

_FORMAT_SHORTCUT_PATTERNS: Tuple["re.Pattern[str]", ...] = tuple(
    re.compile(p) for p in (
        # Language-version requests: "中文方案" / "给我中文版" / "英文结果"
        rf"^{_FO_PREFIX}{_FO_LANG}(?:版|版本)?{_FO_ARTIFACT}?$",
        # Translation requests: "翻译成中文" / "把方案改成英文"
        rf"^{_FO_PREFIX}(?:把|将)?(?:它|这些|上面的|之前的)?{_FO_ARTIFACT}?"
        rf"(?:翻译成?|译为|改成|换成|转成){_FO_LANG}$",
        # Re-answer in a given language: "用中文重新回答" / "用中文给我"
        rf"^{_FO_PREFIX}用{_FO_LANG}(?:重新)?(?:回答|输出|描述|展示|给我|"
        rf"写一(?:遍|份)|说一(?:遍|下))$",
        # Re-layout / re-formatting: "重新排版" / "换个格式" / "改成表格"
        rf"^{_FO_PREFIX}(?:把|将)?(?:它|这些|上面的)?(?:重新)?(?:排版|"
        rf"换(?:一?个|一?种)格式|改格式)$",
        rf"^{_FO_PREFIX}(?:把|将)?(?:它|这些|上面的)?{_FO_ARTIFACT}?"
        rf"改(?:成|为)(?:表格|markdown|列表|要点|简洁版)$",
        # Re-emit the same deliverable: "再给我一遍方案" / "重新输出一份"
        rf"^{_FO_PREFIX}(?:再|重新)(?:给我|来|输出|发|写|生成|说)"
        rf"(?:一(?:遍|份|次|下))?{_FO_ARTIFACT}?$",
    )
)

# Upper bound for shortcut candidates — genuinely format-only requests are
# short; anything longer must go through LLM triage.
_FORMAT_SHORTCUT_MAX_CHARS = 40


@ModuleRegistry.register
class M1ProblemUnderstanding(ModuleProtocol):
    module_name = "m1"
    module_version = "0.2.0"
    description = "Problem decomposition: sub-questions, domain tagging, entity extraction"

    @staticmethod
    def _sub_question_violations(sub_questions: List[str]) -> List[str]:
        violations: List[str] = []
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
        max_sub_questions: Optional[int] = None,
        followup_routing: bool = False,
        **kwargs,
    ):
        self.mode = mode
        self.llm_config = llm_config
        # Optional cap on the number of sub-questions carried forward — mainly a
        # cost/speed knob for smoke tests (fewer sub-questions → less M2 search).
        self.max_sub_questions = max_sub_questions
        # Iteration-core switch: enables the search-free followup triage.
        # Off ⇒ behaviour is byte-for-byte identical to the legacy module.
        self.followup_routing = followup_routing
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
                output_schema=ProblemCard.model_json_schema(),
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
        card = ProblemCard.model_validate(payload)
        violations = [
            *self._sub_question_violations(card.sub_questions),
            *self._contract_violations(card),
        ]
        if violations:
            retry_prompt = "\n".join([
                M1_USER_TEMPLATE.format(question=question),
                "The previous decomposition violated the atomic-sub-question contract:",
                *[f"- {violation}" for violation in violations],
                "Return a corrected ProblemCard. Split parallel tasks into separate atomic questions.",
            ])
            retry_payload = await self.client.structured_chat(
                system_prompt=M1_SYSTEM_PROMPT,
                user_prompt=retry_prompt,
                output_schema=ProblemCard.model_json_schema(),
                max_tokens=8192,
                temperature=0.0,
            )
            retried = ProblemCard.model_validate(retry_payload)
            retry_violations = [
                *self._sub_question_violations(retried.sub_questions),
                *self._contract_violations(retried),
            ]
            if retry_violations:
                raise ValueError(
                    "ProblemCard still violates the atomic task contract "
                    "after deterministic repair: "
                    + "; ".join(retry_violations)
                )
            card = retried
        if not card.original_question:
            card.original_question = question
        if self.max_sub_questions and self.max_sub_questions > 0:
            card.sub_questions = card.sub_questions[: self.max_sub_questions]
            retained = {" ".join(item.split()).casefold() for item in card.sub_questions}
            card.task_contract.requirements = [
                requirement for requirement in card.task_contract.requirements
                if " ".join(requirement.sub_question.split()).casefold() in retained
            ]
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

    # ------------------------------------------------------------------
    # Followup triage (iteration core)
    # ------------------------------------------------------------------

    async def _understand_followup(self, state: PipelineState) -> Dict[str, Any]:
        """Update the ProblemCard with the follow-up and decide skip_search."""
        followup = state.followup
        assert followup is not None  # caller guarantees

        # --- deterministic shortcut: pure format/language/presentation ---
        # Only when the parent run already produced downstream artefacts
        # (research plans or best hypotheses), so a search-free rerun is
        # actually meaningful.
        if state.problem_card is not None and (
            state.research_plans or state.best_hypotheses
        ):
            matched = self._match_format_only_followup(followup.text)
            if matched is not None:
                return self._format_shortcut_result(state, followup, matched)

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
                output_schema=_FollowupUnderstanding.model_json_schema(),
                max_tokens=8192,
                temperature=getattr(self.llm_config, "temperature", 0.1),
            )
            decision = _FollowupUnderstanding.model_validate(payload)
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

        card = decision.problem_card
        violations = [
            *self._sub_question_violations(card.sub_questions),
            *self._contract_violations(card),
        ]
        if violations:
            retry_prompt = "\n".join([
                M1_FOLLOWUP_USER_TEMPLATE.format(
                    problem_card_json=problem_card_json,
                    followup_text=followup.text,
                    graph_overview=graph_overview,
                    parent_artifacts_summary=parent_artifacts_summary,
                ),
                "The proposed follow-up decomposition violated the atomic task contract:",
                *[f"- {violation}" for violation in violations],
                "Return a corrected follow-up decision with one atomic requirement per sub-question.",
            ])
            retry_payload = await self.client.structured_chat(
                system_prompt=M1_FOLLOWUP_SYSTEM_PROMPT,
                user_prompt=retry_prompt,
                output_schema=_FollowupUnderstanding.model_json_schema(),
                max_tokens=8192,
                temperature=0.0,
            )
            retried = _FollowupUnderstanding.model_validate(retry_payload)
            retry_violations = [
                *self._sub_question_violations(retried.problem_card.sub_questions),
                *self._contract_violations(retried.problem_card),
            ]
            if retry_violations:
                logger.warning(
                    "Follow-up ProblemCard still violates the atomic task "
                    "contract after deterministic repair: %s",
                    "; ".join(retry_violations),
                )
            decision = retried
            card = decision.problem_card
        if not card.original_question:
            card.original_question = (
                state.problem_card.original_question
                if state.problem_card is not None
                else followup.text
            )
        if self.max_sub_questions and self.max_sub_questions > 0:
            card.sub_questions = card.sub_questions[: self.max_sub_questions]
            retained = {" ".join(item.split()).casefold() for item in card.sub_questions}
            card.task_contract.requirements = [
                requirement for requirement in card.task_contract.requirements
                if " ".join(requirement.sub_question.split()).casefold() in retained
            ]

        skip_search = bool(decision.skip_search)
        emit_event(
            "tool_completed",
            module="m1",
            tool="qwen_followup_triage",
            status="completed",
            message=(
                f"追问判定完成：{'免检索（skip_search）' if skip_search else '需要新检索'}，"
                f"{len(card.sub_questions)} 个子问题"
            ),
            elapsed_seconds=time.monotonic() - started_at,
            details={
                "skip_search": skip_search,
                "rationale": decision.rationale,
                "decision_source": "llm_triage",
                "sub_questions": list(card.sub_questions),
                "domains": list(card.domain),
                "key_entities": list(card.key_entities),
            },
        )
        return {
            "problem_card": card,
            "followup": followup.model_copy(update={"skip_search": skip_search}),
        }

    # ------------------------------------------------------------------
    # Deterministic format-only shortcut helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _match_format_only_followup(text: str) -> Optional[str]:
        """Return the matched whitelist pattern if ``text`` is a pure
        format/language/presentation request, else ``None``.

        Matching is conservative: whitespace-stripped, length-bounded,
        anchored full-match against the phrase whitelist — semantic
        follow-ups (new entities/directions) can never match.
        """
        normalized = re.sub(r"\s+", "", text or "")
        normalized = normalized.strip("。！!？?，,.、；;：:~～ ")
        normalized = normalized.lower()
        if not (2 <= len(normalized) <= _FORMAT_SHORTCUT_MAX_CHARS):
            return None
        for pattern in _FORMAT_SHORTCUT_PATTERNS:
            if pattern.fullmatch(normalized):
                return pattern.pattern
        return None

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
            "introduces a genuinely new biomedical direction."
        )
        return "\n".join(lines)

    def _format_shortcut_result(
        self,
        state: PipelineState,
        followup: Any,
        matched_pattern: str,
    ) -> Dict[str, Any]:
        """Skip the LLM entirely for whitelisted format-only follow-ups:
        reuse the parent ProblemCard verbatim and set ``skip_search=True``.
        """
        started_at = time.monotonic()
        rationale = (
            "确定性短路：追问命中格式/语言/呈现类白名单短语，父运行已有完整产物，"
            "保持 ProblemCard 与父卡一致，免检索（skip_search=True）。"
        )
        emit_event(
            "tool_started",
            module="m1",
            tool="followup_format_shortcut",
            status="running",
            message="追问命中格式类白名单，执行确定性短路",
            details={"parent_run_id": followup.parent_run_id},
        )
        card = state.problem_card.model_copy(deep=True)
        if self.max_sub_questions and self.max_sub_questions > 0:
            card.sub_questions = card.sub_questions[: self.max_sub_questions]
        emit_event(
            "tool_completed",
            module="m1",
            tool="followup_format_shortcut",
            status="completed",
            message="格式类追问确定性短路：免检索（skip_search）",
            elapsed_seconds=time.monotonic() - started_at,
            details={
                "skip_search": True,
                "rationale": rationale,
                "decision_source": "deterministic_shortcut",
                "matched_pattern": matched_pattern,
                "followup_text": followup.text,
                "sub_questions": list(card.sub_questions),
                "domains": list(card.domain),
                "key_entities": list(card.key_entities),
            },
        )
        return {
            "problem_card": card,
            "followup": followup.model_copy(update={"skip_search": True}),
        }

    @classmethod
    def get_input_fields(cls) -> List[str]:
        return ["input_question", "followup", "problem_card", "evidence_graph"]

    @classmethod
    def get_output_fields(cls) -> List[str]:
        return ["problem_card"]
