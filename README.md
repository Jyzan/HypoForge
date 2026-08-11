# HypoForge

> 基于 LangGraph StateGraph 的六模块闭环 AI Scientist Pipeline：从科学问题出发，自动完成问题理解、文献检索、证据图谱构建、假设生成、研究计划设计与评审迭代。

## 核心流程

```text
M1 问题理解        Problem Understanding      -> ProblemCard + TaskContract
M2 文献检索        Agentic Literature Search  -> 论文、知识条目与可溯源证据
M3 证据图谱        Evidence Graph / Grounding -> EvidenceGraph + GroundingReport
M4 假设生成        Hypothesis Generation      -> 排序后的候选假设
M5 研究计划        Research Plan              -> 结构化实验/计算方案
M6 评审迭代        Review & Iteration         -> 评分与迭代修订
```

- 每个模块的输出都带**可审计的溯源**（证据 ID、引用、任务契约追踪）
- 模块间通过统一的 observability 事件流同步到终端显示与 Web UI
- 支持断点续传（checkpoint）、失败重试（从断点继续）、追问链（基于上一轮结果迭代）

## 快速开始

### 1. 创建环境

```powershell
conda activate biodsa
python -m pip install -r requirements.txt
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

启用实体规范化 embedding 时，建议显式配置独立 endpoint：

```env
ENTITY_EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
ENTITY_EMBEDDING_API_KEY=your_api_key_here
```

Endpoint 与 Key 的 fallback 链：`entity_embedding_base_url` → `ENTITY_EMBEDDING_BASE_URL` → `OPENAI_BASE_URL`；`ENTITY_EMBEDDING_API_KEY` → `OPENAI_API_KEY`。

本地 `.env` 已被 `.gitignore` 忽略，不要将真实 API Key 提交到 Git。

### 3. 运行完整 Pipeline

```powershell
python run_hypoforge.py -q "蛋白质如何折叠以及错误折叠导致疾病的机制是什么？"
```

常用参数：

```powershell
# 只运行部分模块（复用默认配置，裁剪流程）
python run_hypoforge.py -q "..." --modules m1,m2

# 断点续跑（同一 --run-id）
python run_hypoforge.py -q "..." --run-id hypoforge-xxxxxxxx --resume

# 静默运行（关闭 Rich 终端输出）
python run_hypoforge.py -q "..." --quiet
```

### 4. 启动 Web UI

```powershell
python run_hypoforge_ui.py            # 默认 http://127.0.0.1:7860
python run_hypoforge_ui.py --port 8080
```

UI 提供实时事件流、模块级详情抽屉（检索源统计、证据、评审意见等）、历史运行管理与断点重试。

## 配置文件

| 文件 | 用途 |
|---|---|
| `configs/default.yaml` | 默认完整 M1-M6 Pipeline（CLI 无 `-c` 时使用；配合 `--modules` 可裁剪模块） |
| `configs/web_ui.yaml` | Web UI 运行配置 |
| `configs/evaluation.yaml` | 评估配置 |

实体规范化 embedding 通过 `entity_embedding_model` 字段启用（设为空字符串表示仅使用 lexical-only 匹配）。

## 输出文件

默认输出目录为 `./output`，典型文件包括：

```text
output/
├── <run_id>.json            # 最终状态
├── <run_id>_checkpoint.json # 模块级断点（续传/重试的依据）
├── <run_id>_scores.json     # 独立评分
└── snapshots/               # 每轮迭代快照
```

如果配置了 `memory_cache_dir`，Paper Store 与实体规范化缓存会写入对应目录。

## 测试与验证

整合测试集（单文件，覆盖 M1-M6 各流程与完整管线）：

```powershell
python -m pytest -q scripts/test_pipeline.py
```

按模块聚焦（文件内以 `====` 注释分段）：

```powershell
python -m pytest -q scripts/test_pipeline.py -k m2
python -m pytest -q scripts/test_pipeline.py -k m4
```

冒烟验证（真实 LLM，端到端最小流程）：

```powershell
python scripts/smoke_pipeline.py
```

## 项目结构

```text
hypoforge/
├── modules/                 # M1-M6 各模块（Pipeline-facing facade）
├── literature/              # M2 检索：规划、多源检索、阅读、证据导出
├── entity_graph_merge.py    # M3 最终图实体合并（带审计）
├── strict_contracts.py      # 严格契约（fail-closed 包装）
├── registry.py              # 模块注册与装配
├── pipeline.py              # LangGraph 管线编排（checkpoint/续传/取消）
├── observability.py         # 统一事件流
├── webapp.py                # 本地 Web UI 后端
└── display/                 # 终端渲染（Rich）

scripts/
├── test_pipeline.py         # 整合测试集
└── smoke_pipeline.py        # 冒烟验证脚本
```

## 开发约定

- 新增输出字段时，同步更新状态模型、严格契约与测试。
- 不要提交 `.env`、API Key、运行输出、缓存或临时目录。
- 修改完成后运行完整测试集：

```powershell
python -m pytest -q scripts/test_pipeline.py
```
