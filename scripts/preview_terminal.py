#!/usr/bin/env python
"""Offline preview for HypoForge terminal rendering.

This script exercises the display layer only. It does not call an LLM,
perform literature searches, or run the LangGraph pipeline.

Examples:
    python scripts/preview_terminal.py
    python scripts/preview_terminal.py --no-input
    python scripts/preview_terminal.py --width 80
    python scripts/preview_terminal.py --section m5
    python scripts/preview_terminal.py --section m6
    python scripts/preview_terminal.py --section m6 --no-input
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Dict

from rich.console import Console

# Make the script work when launched as `python scripts/preview_terminal.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hypoforge.display.panels as panels
from hypoforge.state import (
    ConfidenceLevel,
    EvidenceGraph,
    HypothesisCard,
    KnowledgeEntry,
    KnowledgeEntryType,
    LiteratureResult,
    PipelineState,
    ProblemCard,
    ResearchPlan,
    ReviewResult,
    ReviewerDimension,
    QuestionType,
)


def sample_problem_card() -> ProblemCard:
    return ProblemCard(
        original_question="Can an effective HIV vaccine be developed?",
        domain=["virology", "immunology", "vaccinology"],
        sub_questions=[
            "How do HIV mutation and Env glycan shielding limit broad neutralizing antibody induction?",
            "Which vaccine platforms can induce durable and broad anti-HIV immunity?",
        ],
        key_entities=["HIV-1", "Env", "bNAbs", "CD4+ T cells"],
        question_type=QuestionType.METHOD,
    )


def sample_literature() -> list[LiteratureResult]:
    # --- English-language sources (PubMed / Semantic Scholar) ---
    en_source = "A representative PubMed paper on HIV vaccine immunology"
    en_entries = [
        (KnowledgeEntryType.ESTABLISHED_FACT, "HIV infection progressively depletes CD4+ T cells.", "high"),
        (KnowledgeEntryType.MECHANISTIC_CONCLUSION, "Env glycan shielding limits antibody access to conserved epitopes.", "medium"),
        (KnowledgeEntryType.CONFLICTING_EVIDENCE, "Animal-model protection has not consistently translated to humans.", "medium"),
        (KnowledgeEntryType.KNOWLEDGE_GAP, "The best immunogen sequence for sequential bNAb maturation remains unknown.", "high"),
        (KnowledgeEntryType.METHOD, "Sequential immunization can be used to guide antibody-lineage maturation.", "high"),
        (KnowledgeEntryType.KEY_ENTITY, "Broadly neutralizing antibodies are central entities for vaccine design.", ""),
    ]
    en_knowledge = [
        KnowledgeEntry(
            id=f"KE_PMID_40060683_{i}",
            type=kind,
            content=content,
            confidence=ConfidenceLevel(conf) if conf else None,
            source_paper_id="PMID:40060683",
            source_paper_title=en_source,
            entities=["HIV-1", "Env"],
        )
        for i, (kind, content, conf) in enumerate(en_entries)
    ]

    # --- Chinese-language sources (万方 / 知网 / 中华医学会期刊) ---
    cn_papers = [
        {
            "id": "WFD_20240715_003",
            "title": "中国HIV-1流行株CRF01_AE包膜糖蛋白gp120的V3环氨基酸变异与中和抗体逃逸关系研究",
            "authors": ["张明华", "李红", "王建国", "陈思远", "刘洋"],
            "journal": "中华微生物学和免疫学杂志",
            "year": 2024,
            "entries": [
                (KnowledgeEntryType.ESTABLISHED_FACT,
                 "中国主要流行株CRF01_AE的V3环第308位氨基酸由甘氨酸突变为精氨酸后，可显著降低V3导向中和抗体的结合能力。",
                 "high"),
                (KnowledgeEntryType.MECHANISTIC_CONCLUSION,
                 "V3环顶端第312-315位残基构成的β-发夹结构通过空间位阻效应屏蔽了CD4结合位点的保守表位，导致广谱中和抗体2G12的结合效率下降约60%。",
                 "medium"),
                (KnowledgeEntryType.METHOD,
                 "采用假病毒中和实验（TZM-bl细胞系）结合深度测序技术，对全国7个省份328份HIV-1感染者血浆样本进行了包膜基因扩增和中和表型分析。",
                 "high"),
                (KnowledgeEntryType.KNOWLEDGE_GAP,
                 "目前尚不清楚V3环变异与我国感染者体内中和抗体谱系成熟轨迹之间的动态关系，缺乏纵向队列的长期随访数据。",
                 ""),
                (KnowledgeEntryType.CONFLICTING_EVIDENCE,
                 "不同研究对CRF01_AE的V3环糖基化位点数目报道存在差异：南方队列报道平均3.2个，北方队列报道平均4.1个，可能与采样人群的感染阶段和抗病毒治疗史有关。",
                 "medium"),
                (KnowledgeEntryType.ESTABLISHED_FACT,
                 "我国HIV-1新发感染中以CRF01_AE和CRF07_BC重组型占主导地位，合计超过总感染人数的75%，且CRF01_AE在男男性行为人群中占比持续上升。",
                 "high"),
            ],
        },
        {
            "id": "CNKI_SYXB202405008",
            "title": "基于mRNA-LNP平台的嵌合Env三聚体疫苗在中国恒河猴模型中的免疫原性评价",
            "authors": ["赵伟", "孙丽萍", "周大可", "吴敏霞"],
            "journal": "中国实验动物学报",
            "year": 2024,
            "entries": [
                (KnowledgeEntryType.MECHANISTIC_CONCLUSION,
                 "编码BG505 SOSIP.664三聚体的mRNA-LNP疫苗在中国恒河猴（Macaca mulatta）中诱导了针对自体Tier-2假病毒的中和抗体，几何平均滴度（GMT）在第8周达到峰值1:320。",
                 "high"),
                (KnowledgeEntryType.KNOWLEDGE_GAP,
                 "该疫苗诱导的异源中和广度有限，仅能中和约28%的测试毒株，提示需要进一步优化免疫原设计策略以提高交叉保护能力。",
                 ""),
                (KnowledgeEntryType.METHOD,
                 "采用脂质纳米颗粒（LNP）包裹化学修饰mRNA，通过肌肉注射途径免疫6只中国恒河猴（3♂/3♀），分别于第0、4、12周加强，使用TZM-bl假病毒中和实验评估血清中和活性。",
                 "high"),
                (KnowledgeEntryType.ESTABLISHED_FACT,
                 "mRNA-LNP疫苗平台在新冠疫情中已展现出快速研发、高效生产和良好安全性的优势，其技术路线可拓展至HIV疫苗研发领域。",
                 "high"),
            ],
        },
        {
            "id": "PMID:38476912",
            "title": "广谱中和抗体VRC01在中国HIV-1高风险人群中的IIb期临床保护效果评估：一项多中心随机双盲安慰剂对照试验",
            "authors": ["陈晓东", "黄丽华", "马强", "林雨桐", "何志伟", "谢芳"],
            "journal": "中华传染病杂志",
            "year": 2024,
            "entries": [
                (KnowledgeEntryType.ESTABLISHED_FACT,
                 "VRC01单克隆抗体在中国5个省份共纳入1,867名HIV-1高风险受试者的IIb期临床试验中未达到预设的保护效力终点（P=0.42），但对VRC01敏感毒株的亚组分析显示感染风险降低约75%（HR=0.25, 95%CI 0.09-0.69）。",
                 "high"),
                (KnowledgeEntryType.CONFLICTING_EVIDENCE,
                 "欧美人群中VRC01敏感毒株占比约65%，而本试验中国队列中仅约30%的流行株对VRC01敏感，提示基于欧美流行株设计的bNAb产品可能不完全适用于中国人群的流行病学特征。",
                 "high"),
                (KnowledgeEntryType.MECHANISTIC_CONCLUSION,
                 "VRC01的保护效果与毒株对其中和敏感性高度相关，IC80<1μg/mL的毒株可被有效阻断，而IC80>10μg/mL的毒株则几乎完全逃逸。",
                 "medium"),
                (KnowledgeEntryType.KNOWLEDGE_GAP,
                 "中国HIV-1流行株（以CRF01_AE和CRF07_BC为主）的Env蛋白结构特征尚未被系统纳入免疫原设计流程，缺乏针对中国人群流行株优化的下一代广谱中和抗体候选分子。",
                 ""),
            ],
        },
    ]

    cn_knowledge: list[KnowledgeEntry] = []
    for paper in cn_papers:
        for i, (kind, content, conf) in enumerate(paper["entries"]):
            cn_knowledge.append(KnowledgeEntry(
                id=f"KE_{paper['id']}_{i}",
                type=kind,
                content=content,
                confidence=ConfidenceLevel(conf) if conf else None,
                source_paper_id=paper["id"],
                source_paper_title=f"{paper['title']}  [{paper['journal']}, {paper['year']}, 作者: {', '.join(paper['authors'][:3])}{'等' if len(paper['authors']) > 3 else ''}]",
                entities=["HIV-1", "Env", "V3环", "CRF01_AE", "中和抗体"],
            ))

    return [
        LiteratureResult(
            sub_question="How can an HIV vaccine induce broad neutralization?",
            papers_retrieved=8,
            knowledge_entries=en_knowledge,
        ),
        LiteratureResult(
            sub_question="中国HIV-1流行株的免疫逃逸机制及疫苗设计的本土化策略是什么？",
            papers_retrieved=15,
            knowledge_entries=cn_knowledge,
        ),
    ]


def sample_graph() -> EvidenceGraph:
    return EvidenceGraph(
        nodes=[],
        edges=[],
        established_facts=[
            "KE_PMID_40060683_0",
            "KE_WFD_20240715_003_0",
            "KE_WFD_20240715_003_5",
            "KE_CNKI_SYXB202405008_3",
            "KE_PMID:38476912_0",
        ],
        conflicts=[
            "KE_PMID_40060683_2",
            "KE_WFD_20240715_003_4",
            "KE_PMID:38476912_1",
        ],
        knowledge_gaps=[
            "KE_PMID_40060683_3",
            "KE_WFD_20240715_003_3",
            "KE_CNKI_SYXB202405008_1",
            "KE_PMID:38476912_3",
        ],
    )


def sample_hypotheses() -> list[HypothesisCard]:
    return [
        HypothesisCard(
            hypothesis_id="HIV-PRIME-BOOST-001",
            statement="A sequential germline-targeting immunization regimen can mature bNAb precursors against conserved Env epitopes.",
            mechanism="Prime germline precursors -> guide affinity maturation -> expose conserved Env sites",
            observable_predictions=[
                "The regimen increases the frequency of Env-specific precursor B cells.",
                "Serum neutralization breadth increases after the final boost.",
            ],
            falsification_conditions=["No increase in precursor frequency or neutralization breadth is observed."],
            scores={
                "novelty": 0.90,
                "scientific_soundness": 0.82,
                "testability": 0.88,
                "evidence_consistency": 0.76,
                "composite": 0.84,
            },
        ),
    ]


def sample_plans() -> list[ResearchPlan]:
    return [
        ResearchPlan(
            hypothesis_id="HIV-PRIME-BOOST-001",
            study_subjects="Humanized mice and ex vivo human B-cell cultures with defined Env precursor frequencies",
            independent_variables=["Prime immunogen", "Boost interval", "Adjuvant"],
            dependent_variables=["Precursor frequency", "Binding affinity", "Neutralization breadth"],
            control_groups=["Naive control", "Conventional Env control", "Adjuvant-only control"],
            procedures=["Administer sequential immunogens", "Collect serum and lymphoid tissue", "Measure lineage and neutralization"],
            measurement_metrics=[
                "ELISA binding assay: measures Env-binding antibody titers using serial serum dilutions and standardized reference controls.",
                "Single-cell BCR sequencing: quantifies lineage expansion, somatic mutation, and inferred germline-targeting trajectories.",
                "Tier-2 neutralization panel: measures breadth and potency across representative heterologous HIV pseudoviruses.",
                "Flow cytometry: measures Env-specific precursor B-cell frequencies using fluorescent probe and tetramer staining.",
                "SPR binding kinetics: estimates antibody affinity and on/off rates against stabilized Env trimers.",
                "Lymph-node germinal-center profiling: measures Tfh and B-cell activation states at defined time points.",
                "Serum IgG purification: quantifies antigen-specific binding after removal of nonspecific serum components.",
                "Adverse-event monitoring: records weight, temperature, injection-site findings, and general health throughout study.",
            ],
            analysis_methods=[
                "Mixed-effects model with treatment, time, and animal as fixed and random effects.",
                "Clonal lineage analysis compares mutation trajectories and convergence across immunization groups.",
                "Neutralization breadth is compared using area-under-curve summaries and multiplicity-adjusted contrasts.",
                "Two-way ANOVA with Tukey correction tests precursor frequency across dose and time factors.",
                "SPR parameters are compared with nonlinear regression and bootstrap confidence intervals.",
                "False-discovery rate correction controls multiple testing across BCR lineage features.",
                "Power analysis defines the minimum detectable difference for the primary neutralization endpoint.",
                "Sensitivity analysis evaluates results after excluding samples below the assay quality threshold.",
            ],
            expected_results_if_supported="Sequential immunization increases the abundance and breadth of Env-specific bNAb lineages.",
            expected_results_if_refuted="No group shows improved precursor maturation or neutralization breadth over controls.",
            timeline="Months 1-3: immunogen design and pilot production; Months 4-9: immunization and sample collection; Months 10-12: neutralization assays and analysis",
            risks_and_alternatives=(
                "Risk: Precursor frequencies may be too low for reliable lineage tracking. "
                "Alternative: Enrich the starting B-cell population and validate in a second model. "
                "Risk: Env probes may show nonspecific binding. "
                "Alternative: Repeat validation with orthogonal probes and competition controls. "
                "Risk: Neutralization testing may exceed the available assay throughput. "
                "Alternative: Use a representative tier-2 panel before expanding the screen. "
                "Risk: mRNA-LNP optimization may delay the main study. "
                "Alternative: Extend the pilot phase and retain a validated protein-immunogen fallback."
            ),
        )
    ]


def sample_reviews() -> list[ReviewResult]:
    return [
        ReviewResult(
            dimension=ReviewerDimension.SCIENTIFIC_LOGIC,
            score=4.2,
            comments="The causal chain is coherent, but the relationship between precursor frequency and protection should be stated more explicitly.",
            suggestions="1. Add a direct protection endpoint and define the expected effect size. 2. Replace the OVA vehicle control with scaffold-only nanoparticles to control scaffold-specific effects. 3. Validate the FP-tetramer reagent with positive and negative antibody controls. 4. Reduce the neutralization panel if assay throughput becomes limiting. 5. Add a positive-control immunogen group for benchmarking. 6. Extend the timeline by 2-3 months for optimization delays.",
            version=1,
        ),
        ReviewResult(
            dimension=ReviewerDimension.SCIENTIFIC_LOGIC,
            score=3.5,
            comments="逻辑链总体合理，但关于V3环变异与中和逃逸之间的因果关系缺乏直接的生化和结构生物学证据。研究者需要明确区分“关联”与“因果”——目前的数据仅表明V3环第308位氨基酸多态性与中和敏感性之间存在统计学关联（Spearman ρ=-0.43, P<0.01），这一效应量在生物学意义上仅为中等程度。",
            suggestions="1. 建议补充V3环突变体的表面等离子共振（SPR）直接结合实验，测定G308R突变对2G12抗体结合速率常数（ka/kd）的影响。2. 在中国恒河猴模型中纳入CRF01_AE野生型和V3环突变型毒株的攻击实验，以验证V3环变异对体内保护效果的因果性。3. 建议将“广谱中和抗体”的操作性定义细化为“对至少70%的中国流行株中和效价IC50<50μg/mL”。",
            version=1,
        ),
        ReviewResult(
            dimension=ReviewerDimension.EVIDENCE_CONSISTENCY,
            score=3.8,
            comments="The proposal is compatible with the evidence base, although human translation remains uncertain.",
            suggestions="Include a human ex vivo validation step.",
            version=1,
        ),
        ReviewResult(
            dimension=ReviewerDimension.EVIDENCE_CONSISTENCY,
            score=3.2,
            comments="证据一致性方面存在两个关键问题。第一，欧美队列中VRC01敏感毒株占比约65%，而中国队列仅约30%，这一差异意味着基于欧美流行株设计的免疫原在中国人群中可能面临更严峻的抗原匹配挑战。第二，关于V3环糖基化位点数目，南方与北方队列报道的差异（3.2 vs 4.1）尚未得到独立实验室的验证，不能排除实验批次效应或PCR引物偏好性造成的系统误差。",
            suggestions="1. 在中国多中心队列中使用标准化操作流程（SOP）重新测定V3环糖基化位点数目，控制PCR引物批次、测序深度和生物信息学分析流程的一致性。2. 纳入东南亚（泰国、越南）CRF01_AE流行株的公开序列数据进行跨人群比较，以评估中国队列发现的普遍性。",
            version=1,
        ),
        ReviewResult(
            dimension=ReviewerDimension.METHOD_FEASIBILITY,
            score=4.0,
            comments="The proposed assays are available in a standard translational immunology laboratory.",
            suggestions="Predefine sample-size and batch-effect controls.",
            version=1,
        ),
        ReviewResult(
            dimension=ReviewerDimension.METHOD_FEASIBILITY,
            score=4.1,
            comments="方法设计较为扎实。假病毒中和实验（TZM-bl）是国际公认的金标准，七个省份的采样覆盖了我国主要HIV-1流行区域，具有较好的代表性。mRNA-LNP平台的恒河猴实验设计合理，但样本量（6只，3♂/3♀）可能不足以检测中等效应量（Cohen's d<0.8）的免疫原性差异。",
            suggestions="1. 基于预实验数据估算效应量和变异系数，使用G*Power软件进行正式的样本量论证（α=0.05, power≥0.80）。2. 明确说明恒河猴的MHC分型（Mamu-A*01等位基因频率），因为MHC限制性可能显著影响疫苗免疫原性的评估。3. 建议提前与CDE（国家药品监督管理局药品审评中心）沟通mRNA-LNP疫苗的CMC（化学、制造和控制）要求。",
            version=1,
        ),
        ReviewResult(
            dimension=ReviewerDimension.OVERALL,
            score=4.0,
            comments="The plan is testable and appropriately connects mechanistic and translational measurements.",
            suggestions="Prioritize the protection endpoint in the first iteration.",
            version=1,
        ),
        ReviewResult(
            dimension=ReviewerDimension.OVERALL,
            score=3.8,
            comments="综合三个维度，该研究计划的核心科学问题（中国HIV-1流行株Env变异→免疫逃逸→疫苗设计）具有明确的公共卫生意义和学术价值。主要不足在于：(1) 分子机制层面的因果证据链需要强化；(2) 南北队列间的数据不一致问题需在正式实验中先行排除；(3) 建议增加一项「基于中国流行株Env序列计算设计的嵌合免疫原」作为对比组，以验证本土化设计的优越性。",
            suggestions="在修订版中优先解决因果性证据和跨队列一致性问题。如果时间有限，建议将研究分为两个阶段：第一阶段（6个月）聚焦V3环变异的生化和结构验证，第二阶段（12个月）基于第一阶段结果优化免疫原设计并开展动物实验。",
            version=1,
        ),
    ]


def render_section(section: str) -> None:
    card = sample_problem_card()
    literature = sample_literature()
    graph = sample_graph()
    hypotheses = sample_hypotheses()
    plans = sample_plans()
    reviews = sample_reviews()

    renderers: Dict[str, Callable[[], None]] = {
        "m1": lambda: panels.render_problem_card(card),
        "m2": lambda: panels._render_m2_results(literature),
        "m3": lambda: panels.render_evidence_summary(graph, PipelineState(literature_results=literature)),
        "m4": lambda: panels.render_hypotheses_summary(len(hypotheses), len(hypotheses), len(hypotheses), hypotheses),
        "m5": lambda: panels.render_research_plans_summary(plans),
        "m6": lambda: panels._render_m6_reviews(reviews),
    }

    sections = [section] if section != "all" else list(renderers)
    for name in sections:
        panels.render_phase_header(name, f"Terminal preview for {name.upper()}")
        renderers[name]()
        panels.render_phase_done(name)


def preview_m6_guidance(*, no_input: bool = False) -> str:
    """Preview the real human-guidance prompt and optionally wait for input."""
    reviews = sample_reviews()
    latest_iteration = max((review.version for review in reviews), default=1)
    panels.render_guidance_prompt(latest_iteration)
    if no_input:
        muted = panels.COLORS["muted"]
        panels.console.print(f"  [{muted}](--no-input: skipping stdin)[/{muted}]")
        return ""

    try:
        guidance = input("  > ").strip()
    except (EOFError, KeyboardInterrupt):
        guidance = ""

    panels.render_guidance_result(guidance)
    return guidance


def preview_m4_revision() -> None:
    """Show the M4 phase that follows the guidance interaction."""
    hypotheses = sample_hypotheses()
    panels.render_phase_header("m4", "Terminal preview for M4 revision")
    panels.render_hypotheses_summary(
        len(hypotheses),
        len(hypotheses),
        len(hypotheses),
        hypotheses,
    )
    panels.render_phase_done("m4")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview HypoForge terminal output without running the pipeline.")
    parser.add_argument("--width", type=int, default=100, help="Simulated terminal width, default: 100")
    parser.add_argument(
        "--section",
        choices=["all", "m1", "m2", "m3", "m4", "m5", "m6"],
        default="all",
        help="Preview one module or all modules, default: all",
    )
    parser.add_argument(
        "--no-input",
        action="store_true",
        help="Show the guidance UI without blocking for input in all/M6 previews.",
    )
    args = parser.parse_args()

    # Rich score bars use Unicode block characters; keep redirected Windows
    # output on UTF-8 instead of the console's legacy GBK code page.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    panels.console = Console(
        width=args.width,
        force_terminal=True,
        legacy_windows=False,
    )
    render_section(args.section)
    if args.section in {"all", "m6"}:
        preview_m6_guidance(no_input=args.no_input)
    if args.section == "all":
        preview_m4_revision()


if __name__ == "__main__":
    main()
