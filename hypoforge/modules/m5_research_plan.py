"""
M5: Research Plan Design.

For each top hypothesis, generates a detailed experimental research plan
covering subjects, variables, controls, procedures, metrics, analysis,
and success/failure criteria.

Output: ``research_plans`` in state.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ..protocol import ModuleProtocol
from ..prompts.m5_prompts import M5_SYSTEM_PROMPT, M5_USER_TEMPLATE
from ..registry import ModuleRegistry
from ..state import PipelineState, ResearchPlan
from ..tools.qwen_client import QwenClient

logger = logging.getLogger(__name__)


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
            payload = await self.client.structured_chat(
                system_prompt=M5_SYSTEM_PROMPT,
                user_prompt=M5_USER_TEMPLATE.format(
                    statement=h.statement,
                    mechanism=h.mechanism,
                    predictions="\n".join(h.observable_predictions),
                    falsification_conditions="\n".join(h.falsification_conditions),
                ),
                output_schema=ResearchPlan.model_json_schema(),
                max_tokens=16384,  # ResearchPlan has 11 fields — needs headroom
                temperature=getattr(self.llm_config, "temperature", 0.1),
            )
            plan = ResearchPlan.model_validate(payload)
            if not plan.hypothesis_id:
                plan.hypothesis_id = h.hypothesis_id
            plans.append(plan)

        return {"research_plans": plans}

    @classmethod
    def get_input_fields(cls) -> List[str]:
        return ["top_hypotheses"]

    @classmethod
    def get_output_fields(cls) -> List[str]:
        return ["research_plans"]
