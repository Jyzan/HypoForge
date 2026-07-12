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
    # Stub research plans
    # ------------------------------------------------------------------

    _STUB_PLANS = {
        "H1": ResearchPlan(
            hypothesis_id="H1",
            study_subjects=(
                "HEK293T细胞系（用于机制验证）、C2C12肌管细胞（代谢模型）、"
                "C. elegans衰老模型（体内验证）。每组n=6，独立重复3次。"
            ),
            independent_variables=[
                "NMN浓度（0, 100μM, 500μM, 1mM）",
                "NAD+合成抑制剂FK866处理",
                "Hsp70 siRNA敲低",
            ],
            dependent_variables=[
                "Hsp70 ATPase活性（孔雀绿比色法）",
                "错误折叠蛋白水平（ProteoStat染料）",
                "细胞内NAD+/NADH比值（比色法试剂盒）",
                "细胞活力（MTT assay）",
            ],
            control_groups=[
                "阴性对照：PBS处理",
                "阳性对照：17-AAG（Hsp90抑制剂，已知诱导错误折叠）",
                "载体对照：scrambled siRNA",
            ],
            procedures=[
                "Day 0: 细胞铺板，使密度达到70-80%。",
                "Day 1: 加药处理（NMN/FK866/siRNA），培养24-48h。",
                "Day 2: 收集细胞，裂解，测定NAD+/NADH比值。",
                "Day 2: 免疫沉淀Hsp70，测定ATPase活性。",
                "Day 2: ProteoStat染色 + 流式细胞术检测错误折叠蛋白。",
                "Day 3: MTT assay评估细胞活力。",
                "统计分析：双因素ANOVA + Tukey post-hoc。",
            ],
            measurement_metrics=[
                "NAD+/NADH: 比色法试剂盒（Sigma-Aldrich MAK037）",
                "Hsp70 ATPase: 孔雀绿比色法",
                "错误折叠蛋白: ProteoStat + 流式",
                "细胞活力: MTT",
            ],
            analysis_methods=[
                "双因素ANOVA（处理×浓度）",
                "Tukey HSD post-hoc",
                "Pearson相关性分析（NAD+/NADH vs 折叠指标）",
                "效应量：Cohen's d",
                "显著性水平：α=0.05，Bonferroni校正",
            ],
            expected_results_if_supported=(
                "NMN组（500μM, 1mM）的Hsp70 ATPase活性显著高于对照组（p<0.01），"
                "错误折叠蛋白水平显著低于对照组（p<0.01）。NAD+/NADH比值与错误折叠蛋白"
                "呈负相关（r<-0.6）。FK866组效果相反。"
            ),
            expected_results_if_refuted=(
                "若各浓度NMN处理组的Hsp70 ATPase活性与对照组无显著差异，"
                "且NAD+/NADH与折叠指标无显著相关性（p>0.05），则假设被驳斥。"
            ),
            timeline="4周：细胞培养优化（1周）、正式实验（2周）、数据分析与重复（1周）",
            risks_and_alternatives=(
                "技术风险：NMN不稳定需冷冻保存；备选：NR（nicotinamide riboside）。"
                "样本风险：C. elegans实验周期长；备选：先在细胞系充分验证后再启动。"
            ),
        ),
    }

    _DEFAULT_PLAN = ResearchPlan(
        hypothesis_id="",
        study_subjects="HEK293T细胞系，n=6，3次独立重复。",
        independent_variables=["药物处理（浓度梯度）"],
        dependent_variables=["蛋白质表达水平", "细胞活力"],
        control_groups=["阴性对照（PBS）", "阳性对照"],
        procedures=["铺板 → 加药 → 孵育 → 裂解 → 检测 → 分析"],
        measurement_metrics=["Western blot", "qPCR", "MTT"],
        analysis_methods=["Student's t-test", "one-way ANOVA"],
        expected_results_if_supported="实验组指标显著优于对照组（p<0.05）。",
        expected_results_if_refuted="实验组与对照组无显著差异（p>0.05）。",
        timeline="3周",
        risks_and_alternatives="标准风险。",
    )

    # ------------------------------------------------------------------
    # ModuleProtocol implementation
    # ------------------------------------------------------------------

    def __init__(
        self,
        mode: str = "stub",
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
        plans: List[ResearchPlan] = []
        for h in state.top_hypotheses:
            if self.mode in {"llm", "direct", "api"} and self.client:
                try:
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
                    continue
                except Exception as exc:
                    logger.warning("M5 LLM mode failed for %s; falling back to stub: %s", h.hypothesis_id, exc)

            stub = self._STUB_PLANS.get(h.hypothesis_id)
            if stub:
                plan = stub.model_copy()
            else:
                plan = self._DEFAULT_PLAN.model_copy()
                plan.hypothesis_id = h.hypothesis_id
            plans.append(plan)

        return {"research_plans": plans}

    @classmethod
    def get_input_fields(cls) -> List[str]:
        return ["top_hypotheses"]

    @classmethod
    def get_output_fields(cls) -> List[str]:
        return ["research_plans"]
