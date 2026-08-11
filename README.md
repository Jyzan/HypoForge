# HypoForge

> 基于 LangGraph StateGraph 的六模块闭环 AI Scientist Pipeline，用于科学问题分解、文献检索、证据图谱构建、假设生成、研究计划设计与评审迭代。

## 当前架构

HypoForge 当前采用 **Agentic-only M2** 架构。M2 的 Pipeline 入口和 Literature 实现分层如下：

```text
Pipeline
  -> ModuleRegistry
  -> hypoforge.modules.m2_literature_search.M2LiteratureSearch
  -> hypoforge.literature.adapter.AgenticM2Module
  -> AgenticM2Adapter
  -> plan -> search -> screen -> read evidence -> export to M3
```

`M2LiteratureSearch` 是 Pipeline-facing facade，只负责统一模块边界和 Registry 注册；实际的搜索、筛选、全文/摘要阅读、知识抽取和证据导出由 `hypoforge.literature` 提供。

M2 导出的结果包括：

- `literature_results`
- `m2_knowledge_export`
- 带有 `evidence_ids` 的知识条目
- 带有 `quote`、`chunk_id`、`page`、`paper_id` 等溯源信息的证据条目

M3 消费 M2 导出的证据，不负责重复下载论文。严格契约模式下，M2 仍然使用 `StrictM2LiteratureSearch` 和 `StrictAgenticM2Adapter`。

## 快速开始

### 1. 创建环境

推荐使用项目当前验证过的 Anaconda 环境：

```powershell
conda activate biodsa
D:/Programming/Anaconda/envs/biodsa/python.exe -m pip install -r requirements.txt
```

如果环境尚未创建，可以使用：

```powershell
conda env create -f environment.yml
conda activate biodsa
```

### 2. 配置环境变量

复制模板并编辑项目根目录下的 `.env`：

```powershell
Copy-Item .env_template .env
```

最少需要配置 OpenAI-compatible LLM endpoint 和 API Key：

```env
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
OPENAI_API_KEY=your_api_key_here
```

如果启用了实体规范化 embedding，建议显式配置独立 endpoint：

```env
ENTITY_EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
ENTITY_EMBEDDING_API_KEY=your_api_key_here
```

当前代码也支持以下 fallback：

```text
embedding endpoint: entity_embedding_base_url -> ENTITY_EMBEDDING_BASE_URL -> OPENAI_BASE_URL
API key:           ENTITY_EMBEDDING_API_KEY -> OPENAI_API_KEY
```

本地 `.env` 已被 `.gitignore` 忽略，不要将真实 API Key 提交到 Git。

### 3. 运行 Pipeline

完整 M1-M6 Pipeline（默认配置，无需 `-c`）：

```powershell
D:/Programming/Anaconda/envs/biodsa/python.exe run_hypoforge.py `
  -q "蛋白质如何折叠以及错误折叠导致疾病的机制是什么？"
```

只运行 M1 + M2（复用默认配置，裁剪模块）：

```powershell
D:/Programming/Anaconda/envs/biodsa/python.exe run_hypoforge.py `
  -q "强化学习相比于 Pass@K 是否真的有性能上的改进？" `
  --modules m1,m2
```

断点续跑：

```powershell
D:/Programming/Anaconda/envs/biodsa/python.exe run_hypoforge.py `
  -q "原始问题" `
  --run-id hypoforge-xxxxxxxx `
  --resume
```

静默运行：

```powershell
D:/Programming/Anaconda/envs/biodsa/python.exe run_hypoforge.py `
  -q "原始问题" `
  --quiet
```

M2 运行时会通过 Rich 输出紫色圆角进度框，并展示 Query planning、Source search、Screening、Reading、Evidence export 等阶段。

## 六模块 Pipeline

```text
M1 Problem Understanding
    -> ProblemCard
M2 Agentic Literature Search
    -> LiteratureResult + M2KnowledgeExport
M3 Evidence Graph / Grounding
    -> EvidenceGraph + GroundingReport
M4 Hypothesis Generation
    -> ranked hypotheses
M5 Research Plan
    -> structured research plan
M6 Review & Iteration
    -> review scores and optional feedback loop
```

### M2 Agentic 流程

```text
Query Planner
    -> multi-source search (PubMed / Semantic Scholar)
    -> deduplication
    -> ranking
    -> Scout screening
    -> coverage evaluation
    -> full-text / abstract reading
    -> evidence retrieval and knowledge extraction
    -> M2 -> M3 evidence export
```

M2 默认优先使用缓存和已有 Paper Store 数据，并在需要时执行增量补搜。每个阶段都会通过统一 observability event 同步到终端显示和 Web UI。

## 配置文件

| 文件 | 用途 |
|---|---|
| `configs/default.yaml` | 默认完整 M1-M6 Pipeline（CLI 无 `-c` 时使用；配合 `--modules m1,m2` 可裁剪模块） |
| `configs/web_ui.yaml` | Web UI 运行配置（Agentic M2 + 实时 API 参数） |
| `configs/evaluation.yaml` | 评估与消融矩阵配置 |

Embedding 模型通过 Pipeline 配置中的以下字段启用：

```yaml
entity_embedding_model: "text-embedding-v3"
```

设置为空字符串表示仅使用 lexical-only entity normalization：

```yaml
entity_embedding_model: ""
```

## 输出文件

默认输出目录为 `./output`，典型文件包括：

```text
output/
├── <run_id>.json
├── <run_id>_checkpoint.json
└── <run_id>_scores.json
```

如果配置了 `memory_cache_dir`，Paper Store 和实体规范化缓存会写入对应目录。

## Web UI

启动实时 Pipeline 进度界面：

```powershell
D:/Programming/Anaconda/envs/biodsa/python.exe scripts/run_pipeline_ui.py
```

指定端口：

```powershell
D:/Programming/Anaconda/envs/biodsa/python.exe scripts/run_pipeline_ui.py `
  --port 8080
```

## 测试与验证

运行完整测试集：

```powershell
D:/Programming/Anaconda/envs/biodsa/python.exe -m pytest -q tests
```

当前完整测试结果：

```text
637 passed
```

常用聚焦测试：

```powershell
D:/Programming/Anaconda/envs/biodsa/python.exe -m pytest -q `
  tests/test_pipeline.py `
  tests/test_phase0_contracts.py `
  tests/test_strict_contracts.py `
  tests/test_m2_progress.py
```

## 项目结构

```text
hypoforge/
├── modules/
│   ├── m1_problem_understanding.py
│   ├── m2_literature_search.py       # Pipeline-facing M2 facade
│   ├── m3_evidence_graph.py
│   ├── m4_hypothesis_generation.py
│   ├── m5_research_plan.py
│   └── m6_review_iteration.py
├── literature/
│   ├── adapter.py                    # Literature adapter and config wrapper
│   ├── search/                       # Query planning and multi-source retrieval
│   ├── reading/                      # Screening, reading and evidence extraction
│   └── export.py                     # M2 -> M3 evidence export
├── strict_contracts.py               # Strict module and evidence contracts
├── registry.py                       # Module registration and construction
├── pipeline.py                       # LangGraph pipeline orchestration
└── display/
    ├── __init__.py                   # Rich console and color palette
    └── m2_progress.py                # Agentic M2 terminal reporter
```

## 开发约定

- M2 的 Pipeline 入口统一使用 `hypoforge.modules.m2_literature_search.M2LiteratureSearch`。
- Literature 层负责检索和证据生产，但不直接向 `ModuleRegistry` 注册 Pipeline 模块。
- 新增输出字段时，应同步更新状态模型、严格契约和对应测试。
- 不要提交 `.env`、API Key、运行输出、缓存或临时测试目录。
- 修改完成后至少运行完整测试集：

```powershell
D:/Programming/Anaconda/envs/biodsa/python.exe -m pytest -q tests
```
