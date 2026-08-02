"""
M1: Problem Understanding & Decomposition.

Decomposes a frontier scientific question into structured sub-questions,
identifies domains, extracts key entities, and classifies the question type.

Output: ``ProblemCard`` in ``state.problem_card``.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from ..observability import emit_event
from ..protocol import ModuleProtocol
from ..prompts.m1_prompts import M1_SYSTEM_PROMPT, M1_USER_TEMPLATE
from ..registry import ModuleRegistry
from ..state import PipelineState, ProblemCard
from ..tools.qwen_client import QwenClient

logger = logging.getLogger(__name__)


@ModuleRegistry.register
class M1ProblemUnderstanding(ModuleProtocol):
    module_name = "m1"
    module_version = "0.1.0"
    description = "Problem decomposition: sub-questions, domain tagging, entity extraction"

    # ------------------------------------------------------------------
    # ModuleProtocol implementation
    # ------------------------------------------------------------------

    def __init__(
        self,
        mode: str = "llm",
        llm_config: Optional[Any] = None,
        max_sub_questions: Optional[int] = None,
        **kwargs,
    ):
        self.mode = mode
        self.llm_config = llm_config
        # Optional cap on the number of sub-questions carried forward — mainly a
        # cost/speed knob for smoke tests (fewer sub-questions → less M2 search).
        self.max_sub_questions = max_sub_questions
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

        question = state.input_question
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
        if not card.original_question:
            card.original_question = question
        if self.max_sub_questions and self.max_sub_questions > 0:
            card.sub_questions = card.sub_questions[: self.max_sub_questions]
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
            },
        )
        return {"problem_card": card}

    @classmethod
    def get_input_fields(cls) -> List[str]:
        return ["input_question"]

    @classmethod
    def get_output_fields(cls) -> List[str]:
        return ["problem_card"]
