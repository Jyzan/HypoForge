"""
M5: Research Plan Design.

For each top hypothesis, generates a detailed experimental research plan
covering subjects, variables, controls, procedures, metrics, analysis,
and success/failure criteria.

Output: ``research_plans`` in state.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from ..observability import emit_event
from ..protocol import ModuleProtocol
from ..prompts.m5_prompts import M5_SYSTEM_PROMPT, M5_USER_TEMPLATE
from ..registry import ModuleRegistry
from ..state import PipelineState, ResearchPlan
from ..tools.qwen_client import QwenClient

logger = logging.getLogger(__name__)


def _followup_requirement_block(state: PipelineState) -> str:
    """Format ``state.followup.text`` as a priority requirement block.

    Returns ``""`` on non-followup runs, so prompt construction stays
    byte-for-byte identical to the legacy behaviour when no followup exists.
    """
    followup = getattr(state, "followup", None)
    text = (followup.text or "").strip() if followup is not None else ""
    if not text:
        return ""
    return (
        "\n--- 用户追问要求 (user follow-up instruction, highest priority, must be honoured) ---\n"
        f"{text}\n"
    )


@ModuleRegistry.register
class M5ResearchPlan(ModuleProtocol):
    module_name = "m5"
    module_version = "0.1.0"
    description = "Generate detailed experimental research plans for top hypotheses"

    # ------------------------------------------------------------------
    # ModuleProtocol implementation
    # ------------------------------------------------------------------

    def __init__(
        self,
        mode: str = "llm",
        llm_config: Optional[Any] = None,
        **kwargs,
    ):
        self.mode = mode
        self.llm_config = llm_config
        self.client = QwenClient.from_config(llm_config) if llm_config else None

    @staticmethod
    def _build_feedback_context(state: PipelineState) -> str:
        """Route the latest method-feasibility review to plan revision.

        Followup runs additionally inject ``state.followup.text`` so user
        instructions (e.g. "输出中文方案") shape plan generation; on
        non-followup runs the output is unchanged."""
        followup_block = _followup_requirement_block(state)
        if state.iteration_count <= 0 or not state.reviews:
            return followup_block
        latest = max(review.version for review in state.reviews)
        feedback = [
            review.suggestions or review.comments or ""
            for review in state.reviews
            if review.version == latest
            and review.dimension.value == "method_feasibility"
        ]
        feedback = [item.strip() for item in feedback if item.strip()]
        if not feedback:
            return followup_block
        lines = [
            "",
            "Research-plan revision guidance from the latest method-feasibility review:",
            *[f"- {item}" for item in feedback],
            "Apply only suggestions relevant to this hypothesis and keep them in the research plan, not the hypothesis statement.",
        ]
        return followup_block + "\n".join(lines)

    async def __call__(
        self,
        state: PipelineState,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if self.client is None:
            raise RuntimeError(
                "M5 requires an LLM client — pass llm_config / set OPENAI_API_KEY."
            )

        plans: List[ResearchPlan] = []
        for h in state.top_hypotheses:
            started_at = time.monotonic()
            emit_event(
                "tool_started",
                module="m5",
                tool="research_plan_designer",
                status="running",
                message=f"开始为假设 {h.hypothesis_id} 设计研究方案",
                details={"hypothesis_id": h.hypothesis_id},
            )
            payload = await self.client.structured_chat(
                system_prompt=M5_SYSTEM_PROMPT,
                user_prompt=M5_USER_TEMPLATE.format(
                    statement=h.statement,
                    mechanism=h.mechanism,
                    predictions="\n".join(h.observable_predictions),
                    falsification_conditions="\n".join(h.falsification_conditions),
                    feedback_context=self._build_feedback_context(state),
                ),
                output_schema=ResearchPlan.model_json_schema(),
                max_tokens=16384,  # ResearchPlan has 11 fields — needs headroom
                temperature=getattr(self.llm_config, "temperature", 0.1),
            )
            plan = ResearchPlan.model_validate(payload)
            if not plan.hypothesis_id:
                plan.hypothesis_id = h.hypothesis_id
            plans.append(plan)
            emit_event(
                "tool_completed",
                module="m5",
                tool="research_plan_designer",
                status="completed",
                message=f"假设 {h.hypothesis_id} 的研究方案完成",
                elapsed_seconds=time.monotonic() - started_at,
                details={
                    "hypothesis_id": h.hypothesis_id,
                    "procedures": len(plan.procedures),
                    "analysis_methods": len(plan.analysis_methods),
                },
            )

        return {"research_plans": plans}

    @classmethod
    def get_input_fields(cls) -> List[str]:
        return ["top_hypotheses", "reviews", "iteration_count"]

    @classmethod
    def get_output_fields(cls) -> List[str]:
        return ["research_plans"]
