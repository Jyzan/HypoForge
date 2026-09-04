# HypoForge

> 基于 LangGraph StateGraph 的六模块闭环 AI Scientist Pipeline：
> 从科学问题出发，自动完成问题理解、文献检索、证据图谱构建、假设生成、研究计划设计与评审迭代。

## 核心流程

```text
M1 问题理解        Problem Understanding      -> ProblemCard + TaskContract
M2 文献检索        Agentic Literature Search  -> 论文、知识条目与可溯源证据
M3 证据图谱        Evidence Graph / Grounding -> EvidenceGraph + GroundingReport
M4 假设生成        Hypothesis Generation      -> 排序后的候选假设 / 澄清请求
M5 研究计划        Research Plan              -> 结构化实验/计算方案
M6 评审迭代        Review & Iteration         -> 八维显示评分 + 结构化路由
```

- 每个模块的输出都带可审计溯源（证据 ID、引用、任务契约追踪）。
- 模块间通过统一 observability 事件流同步到终端与 Web UI。
- 支持断点续传、失败重试、追问链、标准/快速双模式。
- 最终研究方案可通过“固定此方案并细化”进入对话式细化工作台。

## 最新视觉特效

本版本已包含两套最新的 Web 端视觉特效：

### 1. 水波特效（Water Ripple）

- `hypoforge/web/hero-fluid.js`：基于 Three.js / WebGL 的实时水波折射与交互涟漪。
- `hypoforge/web/index.html`：引入 `/assets/hero-fluid.js` 与 Three.js。
- `hypoforge/webapp.py`：新增 `/assets/hero-fluid.js` 静态路由。

### 2. 图标阴影 / 立体光照特效

- `hypoforge/web/index.html`：`hero-core` 根据鼠标位置实时计算光照方向、倾角、阴影偏移。
- `hypoforge/web/ui-polish.css`：`--hero-shadow-*`、`--hero-light-*`、多层级立体阴影与高光。

## M6 评审体系

M6 是“质量总结 + 迭代路由”分离的评审层：

- **结构化硬门禁**：任务对齐、客观证据一致性、证据覆盖率、答案完整性、实验验证覆盖等。
- **专家评审**：`scientific_logic`、`method_feasibility` 等 LLM 评审。
- **八维现代评分**：
  `task_coverage`、`novelty`、`scientific_logic`、`evidence_reliability`、
  `testability`、`experimental_rigor`、`statistics_reproducibility`、`technical_feasibility`。
- **显示分校准**：对 3–4 分区间做单调校准，使展示分更有区分度；异常结构会触发相应封顶。
- **路由规则**：M6 数值分仅用于展示和报告，不参与迭代路由；路由由硬门禁、证据充分性、实验验证覆盖和图修正请求决定。

## 快速开始

### 1. 创建环境

```powershell
python -m pip install -r requirements.txt
```

### 2. 配置环境变量

```powershell
Copy-Item .env_template .env
```

最少配置 LLM（推荐 Qwen OpenAI-compatible）：

```env
QWEN_API_KEY=your_api_key_here
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_MODEL=qwen3.7-plus
```

### 3. 运行完整 Pipeline

```powershell
python run_hypoforge.py -q "蛋白质如何折叠以及错误折叠导致疾病的机制是什么？"
```

常用参数：

```powershell
# 快速模式（约 5 分钟单轮）
python run_hypoforge.py -q "..." --mode fast

# 只运行部分模块
python run_hypoforge.py -q "..." --modules m1,m2

# 断点续跑
python run_hypoforge.py -q "..." --run-id hypoforge-xxxxxxxx --resume

# 静默运行
python run_hypoforge.py -q "..." --quiet
```

### 4. 启动 Web UI

```powershell
python run_hypoforge_ui.py --port 7862 --no-browser
```

Web UI 提供实时事件流、模块详情、历史运行管理、断点重试，以及最新的水波与立体图标特效。

### 5. 方案细化工作台

```powershell
cd refinement_assistant
pip install -r requirements.txt
python app.py        # http://127.0.0.1:5000
```

从 Web UI 方案卡片点击「固定此方案并细化」可自动载入最终方案并开始对话式细化。

> 细化工作台中的可微分渲染干实验（`refinement_assistant/scripts/`）使用生成式训练数据。
> 该数据未随源码打包，需要时先运行：`python refinement_assistant/scripts/generate_data.py`。

## 配置文件

| 文件 | 用途 |
|---|---|
| `configs/default.yaml` | 默认完整 M1-M6 Pipeline（CLI 默认） |
| `configs/web_ui.yaml` | Web UI 实跑配置（含快速/标准模式参数与超时） |
| `configs/evaluation.yaml` | 评估配置 |

## 输出文件

默认输出目录为 `./output`：

```text
output/
├── <run_id>.json            # 最终状态
├── <run_id>_checkpoint.json # 模块级断点
├── <run_id>_scores.json     # 独立评分
└── snapshots/               # 每轮迭代快照
```

## 测试

```powershell
# 完整测试集
python -m pytest -q tests

# 历史整合测试脚本
python -m pytest -q scripts/test_pipeline.py
```

## 项目结构

```text
hypoforge/
├── modules/                 # M1-M6 各模块
├── modules/m2_literature/   # Agentic M2 检索
├── modules/m3_grounding/    # M3 证据图谱/落地
├── evaluation/              # M6 八维评分、独立指标
├── literature/              # M2 检索基础设施
├── memory/                  # 持久化记忆与缓存
├── web/                     # Web UI 前端资源（含 hero-fluid.js）
├── webapp.py                # 本地 Web UI 后端
├── pipeline.py              # LangGraph 管线编排
├── state.py                 # 全量状态模型
├── strict_contracts.py      # 严格契约
└── config.py                # 配置加载

scripts/
├── test_pipeline.py         # 历史整合测试
└── smoke_pipeline.py        # 冒烟验证

tests/                       # 按模块/契约拆分的测试集

refinement_assistant/        # 方案细化工作台
├── app.py                   # Flask Web 入口
├── main.py                  # 对话主循环
├── core/                    # LLM、上下文、权限、RAG
├── tools/                   # 工具注册表
├── subagent/                # 子代理与安全审计
└── skills/                  # 技能与记忆
```

## 开发约定

- 新增输出字段时同步更新状态模型、严格契约与测试。
- 不要提交 `.env`、API Key、运行输出、缓存或临时目录。
- 修改完成后运行完整测试集。
