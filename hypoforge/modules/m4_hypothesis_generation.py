"""
M4: Hypothesis Generation & Screening (core innovation module).

Multi-agent pipeline: Generator → Critic → Falsifiability Checker → Ranker.

In stub mode, returns pre-crafted hypotheses.  The full implementation will
orchestrate four separate Qwen calls.

Output: ``candidate_hypotheses`` + ``top_hypotheses`` in state.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..protocol import ModuleProtocol
from ..registry import ModuleRegistry
from ..state import HypothesisCard, PipelineState


@ModuleRegistry.register
class M4HypothesisGeneration(ModuleProtocol):
    module_name = "m4"
    module_version = "0.1.0"
    description = "Multi-agent hypothesis generation: Generator → Critic → Falsifiability → Ranker"

    # ------------------------------------------------------------------
    # Configurable parameters
    # ------------------------------------------------------------------

    def __init__(
        self,
        num_candidates: int = 7,
        top_k: int = 3,
        mode: str = "stub",  # "stub" | "direct" | "multi_agent"
        **kwargs,
    ):
        self.num_candidates = num_candidates
        self.top_k = top_k
        self.mode = mode

    # ------------------------------------------------------------------
    # Stub hypotheses
    # ------------------------------------------------------------------

    _STUB_HYPOTHESES = [
        HypothesisCard(
            hypothesis_id="H1",
            statement=(
                "Hsp70分子伴侣的ATPase循环速率受到细胞内NAD+/NADH比值的直接调控，"
                "这一代谢-折叠耦合机制在衰老细胞中因NAD+下降而失调，"
                "导致蛋白质错误折叠的积累。"
            ),
            mechanism="NAD+ → Hsp70 ATPase活性 → 底物释放效率 → 折叠质量控制",
            observable_predictions=[
                "补充NMN应能恢复衰老细胞中Hsp70的折叠辅助活性。",
                "NAD+/NADH比值与错误折叠蛋白积累呈负相关（r < -0.6）。",
                "Hsp70 ATPase活性在Sirt1敲除细胞中显著降低。",
            ],
            falsification_conditions=[
                "若NAD+补充后Hsp70活性不变，则假设不成立。",
                "若Hsp70突变体（ATPase活性不依赖NAD+）的细胞仍出现年龄依赖性折叠缺陷。",
            ],
            supporting_evidence=["KE001", "KE002", "KE005"],
            scores={
                "novelty": 0.82,
                "scientific_soundness": 0.78,
                "testability": 0.75,
                "evidence_consistency": 0.88,
                "composite": 0.81,
            },
        ),
        HypothesisCard(
            hypothesis_id="H2",
            statement=(
                "细胞内相分离形成的应激颗粒（stress granules）通过富集分子伴侣Hsp70，"
                "在蛋白质毒性应激下起到保护性缓冲作用，但这种保护机制在重复应激下"
                "因相分离动力学改变而失效。"
            ),
            mechanism="应激 → 相分离 → Hsp70富集于应激颗粒 → 局部折叠缓冲 → 慢性应激下失效",
            observable_predictions=[
                "应激颗粒中Hsp70浓度在急性应激1h内增加3倍以上。",
                "相分离缺陷突变体中错误折叠蛋白积累加速。",
                "重复应激后应激颗粒Hsp70富集效率递减。",
            ],
            falsification_conditions=[
                "若Hsp70在应激颗粒中无富集（超分辨显微镜检测），则假设核心前提不成立。",
                "若相分离抑制剂对折叠质量控制无影响。",
            ],
            supporting_evidence=["KE001", "KE007"],
            scores={
                "novelty": 0.91,
                "scientific_soundness": 0.72,
                "testability": 0.68,
                "evidence_consistency": 0.75,
                "composite": 0.77,
            },
        ),
        HypothesisCard(
            hypothesis_id="H3",
            statement=(
                "分子伴侣网络（Hsp70/Hsp90/CHIP）的协调活性由乙酰化修饰动态调控，"
                "去乙酰化酶Sirt1的年龄依赖性下降导致伴侣网络乙酰化失衡，"
                "是蛋白质稳态崩溃的上游驱动因素。"
            ),
            mechanism="Sirt1下降 → 伴侣蛋白超乙酰化 → Hsp70-Hsp90-CHIP协同失调 → 折叠质控崩溃",
            observable_predictions=[
                "Sirt1激动剂（如白藜芦醇）处理可降低伴侣蛋白乙酰化水平。",
                "衰老组织中Hsp70乙酰化水平与错误折叠蛋白量呈正相关。",
                "模拟乙酰化突变的Hsp70显示底物释放速率降低。",
            ],
            falsification_conditions=[
                "若伴侣蛋白乙酰化位点突变不影响其折叠功能。",
                "若Sirt1敲除小鼠在蛋白质稳态方面无显著表型。",
            ],
            supporting_evidence=["KE001", "KE002"],
            scores={
                "novelty": 0.78,
                "scientific_soundness": 0.85,
                "testability": 0.80,
                "evidence_consistency": 0.82,
                "composite": 0.81,
            },
        ),
    ]

    # ------------------------------------------------------------------
    # ModuleProtocol implementation
    # ------------------------------------------------------------------

    async def __call__(
        self,
        state: PipelineState,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        # In stub mode, return pre-crafted hypotheses
        candidates = self._STUB_HYPOTHESES[:self.num_candidates]
        top = sorted(
            candidates,
            key=lambda h: h.scores.get("composite", 0),
            reverse=True,
        )[:self.top_k]

        return {
            "candidate_hypotheses": candidates,
            "top_hypotheses": top,
        }

    @classmethod
    def get_input_fields(cls) -> List[str]:
        return ["evidence_graph", "problem_card", "literature_results"]

    @classmethod
    def get_output_fields(cls) -> List[str]:
        return ["candidate_hypotheses", "top_hypotheses"]
