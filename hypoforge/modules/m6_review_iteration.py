"""
M6: Review & Iterative Refinement.

Four reviewer agents evaluate the hypothesis and research plan on different
dimensions.  A meta-reviewer synthesises the feedback and decides whether
to accept or iterate.

Output: ``reviews`` appended; ``iteration_count`` incremented.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from ..protocol import ModuleProtocol
from ..prompts.m6_prompts import M6_REVIEWER_PROMPTS, M6_USER_TEMPLATE
from ..registry import ModuleRegistry
from ..state import PipelineState, ReviewResult, ReviewerDimension
from ..tools.qwen_client import QwenClient

logger = logging.getLogger(__name__)


@ModuleRegistry.register
class M6ReviewIteration(ModuleProtocol):
    module_name = "m6"
    module_version = "0.1.0"
    description = "Multi-reviewer assessment + iterative refinement decision"

    # ------------------------------------------------------------------
    # Configurable
    # ------------------------------------------------------------------

    def __init__(
        self,
        reviewers: Optional[List[str]] = None,
        mode: str = "stub",
        llm_config: Optional[Any] = None,
        **kwargs,
    ):
        self.reviewer_dims = reviewers or [
            "scientific_logic",
            "evidence_consistency",
            "method_feasibility",
            "overall",
        ]
        self.mode = mode
        self.llm_config = llm_config
        self.client = QwenClient.from_config(llm_config) if llm_config else None

    # ------------------------------------------------------------------
    # ModuleProtocol implementation
    # ------------------------------------------------------------------

    async def __call__(
        self,
        state: PipelineState,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        version = state.iteration_count + 1

        if self.mode in {"llm", "direct", "api"} and self.client and state.top_hypotheses and state.research_plans:
            try:
                hypothesis = state.top_hypotheses[0]
                plan = next(
                    (p for p in state.research_plans if p.hypothesis_id == hypothesis.hypothesis_id),
                    state.research_plans[0],
                )
                graph = state.evidence_graph
                new_reviews: List[ReviewResult] = []
                for dim in self.reviewer_dims:
                    payload = await self.client.structured_chat(
                        system_prompt=M6_REVIEWER_PROMPTS[dim],
                        user_prompt=M6_USER_TEMPLATE.format(
                            original_question=state.input_question,
                            hypothesis_json=json.dumps(hypothesis.model_dump(mode="json"), ensure_ascii=False, indent=2),
                            plan_json=json.dumps(plan.model_dump(mode="json"), ensure_ascii=False, indent=2),
                            facts_count=len(graph.established_facts) if graph else 0,
                            conflicts_count=len(graph.conflicts) if graph else 0,
                            gaps_count=len(graph.knowledge_gaps) if graph else 0,
                        ),
                        output_schema=ReviewResult.model_json_schema(),
                        max_tokens=getattr(self.llm_config, "max_tokens", 4096),
                        temperature=getattr(self.llm_config, "temperature", 0.1),
                    )
                    payload = dict(payload)
                    payload["dimension"] = dim
                    payload["version"] = version
                    new_reviews.append(ReviewResult.model_validate(payload))

                return {
                    "reviews": state.reviews + new_reviews,
                    "iteration_count": version,
                }
            except Exception as exc:
                logger.warning("M6 LLM mode failed; falling back to stub: %s", exc)

        # ---- stub reviews for the top hypothesis ----
        if state.top_hypotheses:
            h = state.top_hypotheses[0]

            # Simulate improving scores across iterations
            base_scores: Dict[str, List[float]] = {
                "scientific_logic":    [3.2, 3.8, 4.2],
                "evidence_consistency": [3.5, 4.0, 4.3],
                "method_feasibility":  [2.8, 3.5, 3.9],
                "overall":             [3.1, 3.7, 4.1],
            }

            comments: Dict[str, str] = {
                "scientific_logic": (
                    "因果链基本完整，但机制解释中的NAD+→Hsp70调控环节需更多直接证据支持。"
                    if version < 3 else "因果链严密，逻辑连贯。"
                ),
                "evidence_consistency": (
                    "引用了关键文献，但忽略了部分关于Hsp70独立于NAD+功能的报道。"
                    if version < 3 else "证据引用充分，冲突文献已被合理回应。"
                ),
                "method_feasibility": (
                    "实验设计合理但样本量偏小，建议增加power analysis来论证n值。"
                    if version < 3 else "方法设计完善，统计方案合理，可执行。"
                ),
                "overall": (
                    "整体质量中等偏上，修订后可达到发表水平。"
                    if version < 3 else "高质量，建议接受。"
                ),
            }

            new_reviews: List[ReviewResult] = []
            for dim in self.reviewer_dims:
                scores = base_scores.get(dim, [3.0, 3.0, 3.0])
                idx = min(version - 1, len(scores) - 1)
                new_reviews.append(ReviewResult(
                    dimension=ReviewerDimension(dim),
                    score=scores[idx],
                    comments=comments.get(dim, "评审意见。"),
                    suggestions=(
                        "需要更充分的引用支持。"
                        if version == 1 else "可接受。"
                    ),
                    version=version,
                ))

            return {
                "reviews": state.reviews + new_reviews,
                "iteration_count": version,
            }

        # No hypotheses to review — just bump the counter
        return {
            "iteration_count": version,
        }

    @classmethod
    def get_input_fields(cls) -> List[str]:
        return ["top_hypotheses", "research_plans", "evidence_graph", "iteration_count"]

    @classmethod
    def get_output_fields(cls) -> List[str]:
        return ["reviews", "iteration_count"]
