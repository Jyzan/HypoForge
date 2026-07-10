"""
M6: Review & Iterative Refinement.

Four reviewer agents evaluate the hypothesis and research plan on different
dimensions.  A meta-reviewer synthesises the feedback and decides whether
to accept or iterate.

Output: ``reviews`` appended; ``iteration_count`` incremented.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..protocol import ModuleProtocol
from ..registry import ModuleRegistry
from ..state import PipelineState, ReviewResult, ReviewerDimension


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
        **kwargs,
    ):
        self.reviewer_dims = reviewers or [
            "scientific_logic",
            "evidence_consistency",
            "method_feasibility",
            "overall",
        ]

    # ------------------------------------------------------------------
    # ModuleProtocol implementation
    # ------------------------------------------------------------------

    async def __call__(
        self,
        state: PipelineState,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        version = state.iteration_count + 1

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
