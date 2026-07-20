"""
M6: Review & Iterative Refinement.

Three specialist reviewer agents (scientific_logic / evidence_consistency /
method_feasibility) score the top hypothesis + research plan on a 1–5 scale;
each reasons *before* it scores (rubric-anchored).  The ``overall`` score is
computed as the mean of the specialist scores.

Output: ``reviews`` appended; ``iteration_count`` incremented.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from ..protocol import ModuleProtocol
from ..prompts.m6_prompts import M6_REASON_FIRST, M6_REVIEWER_PROMPTS, M6_USER_TEMPLATE
from ..evaluation.rubric import review_rubric_line
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
        mode: str = "llm",
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

        if self.client is None:
            raise RuntimeError(
                "M6 requires an LLM client — pass llm_config / set OPENAI_API_KEY."
            )

        # Nothing to review yet — just advance the iteration counter.
        if not (state.top_hypotheses and state.research_plans):
            return {"iteration_count": version}

        hypothesis = state.top_hypotheses[0]
        plan = next(
            (p for p in state.research_plans if p.hypothesis_id == hypothesis.hypothesis_id),
            state.research_plans[0],
        )
        graph = state.evidence_graph

        # Specialist reviewers (LLM); "overall" is computed, not queried.
        specialist_dims = [d for d in self.reviewer_dims if d != "overall"]
        new_reviews: List[ReviewResult] = []

        for dim in specialist_dims:
            # Anchor the score (rubric) and force reason-before-score, both
            # sourced from the single rubric definition.
            system_prompt = "\n\n".join(
                p for p in (M6_REVIEWER_PROMPTS[dim], review_rubric_line(dim), M6_REASON_FIRST) if p
            )
            payload = await self.client.structured_chat(
                system_prompt=system_prompt,
                user_prompt=M6_USER_TEMPLATE.format(
                    original_question=state.input_question,
                    hypothesis_json=json.dumps(hypothesis.model_dump(mode="json"), ensure_ascii=False, indent=2),
                    plan_json=json.dumps(plan.model_dump(mode="json"), ensure_ascii=False, indent=2),
                    facts_count=len(graph.established_facts) if graph else 0,
                    conflicts_count=len(graph.conflicts) if graph else 0,
                    gaps_count=len(graph.knowledge_gaps) if graph else 0,
                ),
                output_schema=ReviewResult.model_json_schema(),
                max_tokens=8192,
                temperature=getattr(self.llm_config, "temperature", 0.1),
            )
            payload = dict(payload)
            payload["dimension"] = dim
            payload["version"] = version
            new_reviews.append(ReviewResult.model_validate(payload))

        # Compute overall as the mean of the specialist scores.
        if "overall" in self.reviewer_dims and new_reviews:
            specialist_scores = [r.score for r in new_reviews if r.score > 0]
            avg = sum(specialist_scores) / len(specialist_scores) if specialist_scores else 3.0
            new_reviews.append(ReviewResult(
                dimension=ReviewerDimension("overall"),
                reasoning="Computed as the mean of the specialist reviews.",
                score=round(avg, 1),
                comments="Computed from specialist reviews (scientific_logic, evidence_consistency, method_feasibility).",
                suggestions="See individual dimension reviews for detailed suggestions.",
                version=version,
            ))

        return {
            "reviews": state.reviews + new_reviews,
            "iteration_count": version,
        }

    @classmethod
    def get_input_fields(cls) -> List[str]:
        return ["top_hypotheses", "research_plans", "evidence_graph", "iteration_count"]

    @classmethod
    def get_output_fields(cls) -> List[str]:
        return ["reviews", "iteration_count"]
