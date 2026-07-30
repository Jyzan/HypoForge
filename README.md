# HypoForge

> 挑战杯 2026 · 赛题A：科学假设生成与研究计划设计

基于 LangGraph StateGraph 的六模块闭环 AI Scientist pipeline。从输入一个前沿科学问题开始，自动完成问题分解、文献检索、证据图谱构建、假设生成、研究计划设计和评审迭代。

## 快速开始

### 环境准备

```bash
# 1. 创建 conda 环境（Python 3.11–3.13）
conda create -n hypoforge python=3.11 -y
conda activate hypoforge

# 2. 安装依赖
pip install -r requirements.txt
```

### 配置 API Key

在项目根目录创建 `.env` 文件：

```bash
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

### 运行

```bash
# 完整 Pipeline（legacy M2 双后端检索）
python run_hypoforge.py -q "蛋白质如何折叠及错误折叠导致疾病的机制？" -c configs/full_pipeline.yaml

# Agentic M2 — 迭代搜索 + 全文阅读 + 知识导出（Track A）
python run_hypoforge.py -q "蛋白质如何折叠及错误折叠导致疾病的机制？" -c configs/m2_agentic.yaml

# Agentic M2 轻量版 — PubMed only
python run_hypoforge.py -q "蛋白质如何折叠？" -c configs/m2_minimal_pubmed.yaml

# M3 Grounding — 全文证据接地 + 关系抽取（Track B）
python run_hypoforge.py -q "CRISPR能治疗亨廷顿病吗？" -c configs/m3_grounding.yaml

# 消融基线
python run_hypoforge.py -q "衰老的生物学基础是什么？" -c configs/baseline_b0.yaml

# 断点续跑
python run_hypoforge.py -q "..." --run-id my_experiment --resume
```

### Web 仪表盘

```bash
# 启动 M1-M6 实时进度仪表盘（默认 http://127.0.0.1:7860）
python scripts/run_pipeline_ui.py

# 自定义端口和配置
python scripts/run_pipeline_ui.py --port 8080 --config configs/m2_agentic.yaml
```

### Smoke 测试

```bash
python scripts/smoke_pipeline.py
python scripts/smoke_pipeline.py -q "衰老的生物学基础是什么？"
```

## CLI 参考

```
python run_hypoforge.py [OPTIONS]
```

| 参数 | 简写 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--question` | `-q` | ✅ | — | 待分析的前沿科学问题 |
| `--config` | `-c` | ❌ | `configs/default.yaml` | YAML 配置文件路径 |
| `--output-dir` | `-o` | ❌ | `./output` | 输出目录 |
| `--run-id` | | ❌ | 自动生成 | 自定义运行标识符 |
| `--resume` | | ❌ | `false` | 从 checkpoint 断点续跑 |
| `--quiet` | | ❌ | `false` | 静默模式 |

## 输出文件

```
<output_dir>/
├── <run_id>.json           # PipelineState 完整序列化
├── <run_id>_checkpoint.json # 断点续跑 checkpoint
└── <run_id>_scores.json    # 自动评分报告
```

## Pipeline 架构

```
用户输入科学问题
       │
       ▼
  ┌──────────────┐
  │ M1: 问题理解   │ → ProblemCard: domain, sub_questions, key_entities, question_type
  └──────┬───────┘
         ▼
  ┌──────────────┐
  │ M2: 文献检索   │ → PubMed / SemanticScholar(OpenAlex) / ArXiv 多源搜索
  │   与全文获取   │ → 全文获取 → 分块 → 可引用证据导出
  └──────┬───────┘   → 输出: LiteratureResult + M2KnowledgeExport
         ▼
  ┌──────────────┐
  │ M3: 证据图谱   │ → 消费 M2KnowledgeExport.evidence，不做联网下载
  │   构建        │ → 规则构建 + 原子声明合成 + 关系检测
  └──────┬───────┘   → 输出: EvidenceGraph + GroundingReport
         ▼
  ┌──────────────┐
  │ M4: 假设生成   │ → Multi-Agent: Generator → Critic → Falsifiability Checker → Ranker
  │   与筛选      │ → 四维评分: novelty / scientific_soundness / testability / evidence_consistency
  └──────┬───────┘
         ▼
  ┌──────────────┐
  │ M5: 研究计划   │ → 11 要素结构化研究计划
  │   设计        │
  └──────┬───────┘
         ▼
  ┌──────────────┐
  │ M6: 评审与     │ → 三维 Specialist Reviewer + overall 计算值
  │   迭代更新    │ → 反馈回 M4/M5，循环 ≤ max_iterations
  └──────────────┘
```

### M2/M3 职责边界

```
M2 职责：
  检索论文 → 获取全文 → 分块 → 提取 M2EvidenceExport（可引用文本片段）
  输出: LiteratureResult + M2KnowledgeExport（含 evidence + provenance）

M3 职责：
  消费 M2KnowledgeExport.evidence → 原子声明合成 → 关系判断 → 证据图构建
  M3 不联网下载论文
```

## M2 搜索模式

| 模式 | 实现 | 说明 |
|------|------|------|
| `legacy` | `m2_literature_search.py` | PubMed + OpenAlex 双后端，直接检索 + 批量知识提取 |
| `agentic` | `literature/search/agent.py` | 迭代搜索代理：QueryPlanner → 多源搜索 → 去重 → Scout → 排序 → Coverage 检查 |
| `agentic` + reading | `literature/reading/workflow.py` | 完整管线：全文解析 → 分块 → BM25 检索 → Qwen LLM 阅读 → 证据导出 |

切换方式：在配置文件中设置 `search.implementation: agentic`，默认 `legacy`。

## M3 Grounding 模式

| 模式 | 配置 | 说明 |
|------|------|------|
| 规则图 | 默认 | 基于 KnowledgeEntry 类型和实体共现构建 EvidenceGraph |
| Direct Grounding | `grounding.enabled: true` | 消费 M2KnowledgeExport.evidence → 原子声明合成 → LLM 关系判断 |
| GAMS（实验性） | `grounding.enable_gams: true` | 蒙特卡洛关系搜索，默认关闭 |

## 评估指标

| 指标 | 状态 | 说明 |
|------|------|------|
| NoveltyMetric | ✅ 已实现 | LLM 声明分解 + 证据图 BFS 路径距离判定新颖性 |
| EvidenceConsistencyMetric | ✅ 已实现 | Embedding 匹配 + LLM-as-judge 冲突检测 |
| TestabilityMetric | ✅ 已实现 | LLM-as-judge 评估可预测性和可证伪性 |

消融矩阵脚本：
```bash
python scripts/run_ablation_matrix.py
```

## 运行测试

```bash
# 全部测试
python -m pytest tests -q

# 仅核心测试（无网络）
python -m pytest tests/test_pipeline.py tests/test_qwen_client.py -q

# 文献模块测试
python -m pytest tests/literature/ -q --timeout=30

# M3 Grounding 测试
python -m pytest tests/test_m3_*.py -q
```

## 配置文件

| 配置 | 说明 |
|------|------|
| `configs/default.yaml` | 默认（全模块 + 迭代） |
| `configs/full_pipeline.yaml` | 完整系统 + 持久化 KG |
| `configs/baseline_b0.yaml` | B0：千问直出 |
| `configs/baseline_b1.yaml` | B1：+ web search |
| `configs/baseline_b2.yaml` | B2：+ 结构化提取 |
| `configs/baseline_b3.yaml` | B3：+ 多 Agent |
| `configs/m2_agentic.yaml` | Agentic M2（迭代搜索 + 全文阅读） |
| `configs/m2_minimal_pubmed.yaml` | Agentic M2 轻量版（PubMed only） |
| `configs/m3_grounding.yaml` | M3 Direct Grounding |
| `configs/m3_evidence_gams.yaml` | M3 GAMS 实验性 |
| `configs/evaluation.yaml` | 评估 / 消融矩阵配置 |
| `configs/full_pipeline_m2agentic.yaml` | 完整系统 + Agentic M2 |

## 项目状态

当前已合并 **Track A + B + C + Web UI**，具备完整的 agentic 文献搜索、全文证据接地、自动评估和实时进度仪表盘能力。

| 模块 / 系统 | 状态 | 说明 |
|-------------|------|------|
| M1 问题理解 | ✅ | Qwen `structured_chat` 多尺度问题分解 |
| M2 文献检索 (legacy) | ✅ | PubMed + OpenAlex 双后端，中文→英文改写 |
| M2 文献检索 (agentic) | ✅ | 迭代搜索代理 + Scout + Coverage + 全文阅读管线 |
| M3 证据图谱 (规则) | ✅ | 规则构建 + LLM 跨条目语义边 |
| M3 Grounding | ✅ | 消费 M2KnowledgeExport → 原子声明 → 关系判断 |
| M4 假设生成 | 🟡 | direct / multi_agent 模式，四维加权评分 |
| M5 研究计划 | 🟡 | 11 要素结构化输出 |
| M6 评审迭代 | 🟡 | 三维 Specialist Reviewer + 迭代循环 |
| QwenClient | ✅ | async wrapper + 3 层 API 回退 + thinking 控制 |
| 持久化 KG | ✅ | JSONL CRUD + BM25 检索 + 增量更新 + entity linking |
| Pipeline 编排 | ✅ | LangGraph + 条件迭代 + checkpoint/resume + Skill hooks |
| 评估系统 | ✅ | Novelty / EvidenceConsistency / Testability + 消融矩阵 |
| Web 仪表盘 | ✅ | FastAPI + SSE M1-M6 实时进度 |
| 配置系统 | ✅ | YAML + 环境变量，14 套配置 |

详细开发计划见 [TODO.md](TODO.md)。

## 离线预览终端输出

```bash
python scripts/preview_terminal.py            # 预览全部模块
python scripts/preview_terminal.py --width 120 # 模拟不同终端宽度
python scripts/preview_terminal.py --section m5 # 只预览指定模块
```
