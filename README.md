# HypoForge

> 挑战杯 2026 · 赛题A：科学假设生成与研究计划设计

基于 LangGraph StateGraph 的六模块闭环 AI Scientist pipeline。从输入一个前沿科学问题开始，自动完成问题分解、文献检索、证据图谱构建、假设生成、研究计划设计和评审迭代。

## 快速开始

### Conda 环境（推荐）

```bash
# 1. 创建 conda 环境（Python 3.11–3.13）
conda create -n hypoforge python=3.11 -y

# 2. 激活环境
conda activate hypoforge

# 3. 安装依赖
pip install -r requirements.txt
```

如果希望从 `environment.yml` 一键重建：

```bash
conda env create -f environment.yml
conda activate hypoforge
```

### pip 安装（备选）

```bash
pip install -r requirements.txt
```

### 配置 API Key

在项目根目录创建 `.env` 文件（可参考 `.env_template`）：

```bash
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

### 运行

```bash
# 跑通示例（默认走 LLM mode，会调用 Qwen + PubMed + OpenAlex）
python run_hypoforge.py -q "蛋白质如何折叠及错误折叠导致疾病的机制？"

# 用 stub 模式快速验证（不调用 LLM / API）
python run_hypoforge.py -q "test" -c configs/baseline_b0.yaml --quiet

# 用不同配置跑消融实验
python run_hypoforge.py -q "..." -c configs/baseline_b0.yaml
python run_hypoforge.py -q "..." -c configs/full_pipeline.yaml

# 指定自定义 run-id
python run_hypoforge.py -q "..." --run-id my_experiment_01
```

运行后终端会输出 Rich 美化的六模块执行过程。

## CLI 参考

```
python run_hypoforge.py [OPTIONS]
```

| 参数 | 简写 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--question` | `-q` | ✅ | — | 待分析的前沿科学问题 |
| `--config` | `-c` | ❌ | `configs/default.yaml` | YAML 配置文件路径 |
| `--output-dir` | `-o` | ❌ | `./output` | 输出目录（会覆盖配置文件中的设置） |
| `--run-id` | | ❌ | 自动生成 | 自定义运行标识符，用于输出文件命名和 checkpoint |
| `--resume` | | ❌ | `false` | 从已有 `--run-id` 的 checkpoint 断点续跑 |
| `--quiet` | | ❌ | `false` | 静默模式，关闭 Rich 终端美化输出 |

### 使用示例

```bash
# 最简用法（使用默认配置）
python run_hypoforge.py -q "衰老的生物学基础是什么？"

# 指定配置和输出目录
python run_hypoforge.py -q "..." -c configs/full_pipeline.yaml -o output/experiment_01

# 指定自定义 run-id
python run_hypoforge.py -q "..." --run-id my_custom_run

# 静默模式
python run_hypoforge.py -q "..." --quiet
```

## 输出文件

每次运行会在 `<output_dir>/` 下生成一个 JSON 结果文件：

```
<output_dir>/
└── <run_id>.json
```

- **`output_dir`**：由 `--output-dir`（或配置文件中的 `output_dir`）指定，默认为 `./output`
- **`run_id`**：由 `--run-id` 指定；若未指定则自动生成（基于时间戳），例如 `20250711_143052`

JSON 文件包含 PipelineState 的完整序列化结果，涵盖所有六个模块的输出、迭代历史和错误信息。

## Checkpoint / 断点续跑

Pipeline 每完成一个模块就自动保存 checkpoint 到 `<output_dir>/<run_id>_checkpoint.json`。
如果运行中断（网络超时、API 限流等），可以用相同的 `--run-id` + `--resume` 从断点继续，
已完成的模块会被跳过：

```bash
# 首次运行（假设在 M4 时网络中断）
python run_hypoforge.py -q "..." --run-id my_experiment

# 从 M4 断点恢复（M1/M2/M3 自动跳过）
python run_hypoforge.py -q "..." --run-id my_experiment --resume
```

注意事项：
- M1–M3 有产出即跳过；M4/M5/M6 在迭代未完成时仍会重新运行
- 如果更改了 `--question` 或配置文件，建议使用新的 `--run-id` 而非 resume

## 持久化知识图谱

在配置文件中设置 `memory_cache_dir`（或通过 `configs/*.yaml` 的顶层字段）即可启用 JSONL 持久化知识图谱：

```yaml
# configs/full_pipeline.yaml
memory_cache_dir: "./kg_cache"
```

启用后，M3 模块执行时会自动将证据图谱（Entity + Relation）写入 `{memory_cache_dir}/memory-evidence_graph.jsonl`，支持：
- **跨运行复用**：后续运行可加载已持久化的图谱
- **BM25 语义搜索**：通过 `KnowledgeGraphManager.search_nodes()` 检索相关实体
- **增量更新**：追加新 Entity/Relation 无需全量重写

如果留空（默认），图谱仅在内存中保留，运行结束后丢弃。

## 模块结构

```
M1 → M2 → M3 → M4 → M5 → M6 → (loop to M4)
│     │     │     │     │     │
问题  文献  证据  假设  研究  评审
理解  检索  图谱  生成  计划  迭代
```

## 运行测试

```bash
PYTHONIOENCODING=utf-8 python tests/test_pipeline.py
```

## 配置文件

| 配置 | 说明 |
|------|------|
| `configs/default.yaml` | 默认（全模块 + 迭代） |
| `configs/baseline_b0.yaml` | B0：千问直出 |
| `configs/baseline_b1.yaml` | B1：+ web search |
| `configs/baseline_b2.yaml` | B2：+ 结构化提取 |
| `configs/baseline_b3.yaml` | B3：+ 多 Agent |
| `configs/full_pipeline.yaml` | 完整系统 + 持久化 KG |

> 在配置文件中设置 `memory_cache_dir` 可启用持久化知识图谱（跨运行复用 + BM25 搜索），详见上方「持久化知识图谱」章节。

## 项目状态

当前处于 **LLM 真实调用阶段**。M1–M6 端到端 pipeline 可跑通；全部模块均已接入 Qwen，检索层已对接 PubMed + OpenAlex 真实 API，证据图谱支持 LLM 语义关系抽取。配置文件默认开启 LLM mode。

| 模块 | 状态 | 说明 |
|------|------|------|
| M1 问题理解 | 🟡 LLM 就绪 | `mode="llm"` 时调 Qwen `structured_chat` 做问题分解；stub 为 fallback |
| M2 文献检索 | ✅ 已实现 | PubMed + OpenAlex 双后端检索 → DOI/title 去重 → Qwen 批量知识提取；stub 为 fallback |
| M3 证据图谱 | ✅ 已实现 | 规则构建节点 + `INVOLVES` 边；`mode="llm"` 时 Qwen 批量提取跨 Entry 语义边（SUPPORTS / CONTRADICTS / EXTENDS / LIMITS），含跨批 bridge 任务和 thinking 控制防止截断 |
| M4 假设生成 | 🟡 LLM 就绪 | 支持 `direct` / `multi_agent` 模式（Generator + Ranker 调 Qwen）；composite score 由四维加权重算；Critic / Falsifiability Checker 待补 |
| M5 研究计划 | 🟡 LLM 就绪 | `mode="llm"` 时调 Qwen `structured_chat` 生成含 11 项要素的结构化研究计划 |
| M6 评审迭代 | 🟡 LLM 就绪 | 三维 Reviewer（scientific_logic / evidence_consistency / method_feasibility）各调 Qwen；overall 由 specialist 分数平均计算 |
| Semantic Scholar | ✅ 已实现 | 双后端自动切换：有 key → Semantic Scholar；无 key → OpenAlex |
| PubMed | ✅ 已实现 | NCBI E-utilities（esearch + efetch 两步流程）|
| QwenClient | ✅ 完成 | 完整 async wrapper（`chat` / `structured_chat` / `list_models` / `disable_thinking` 推理控制 / 3 层 API 回退） |
| 持久化 KG | ✅ 完成 | JSONL-backed KnowledgeGraphManager（Entity/Relation CRUD + BM25 搜索） |
| Pipeline 编排 | ✅ 完成 | LangGraph StateGraph + 条件迭代 + checkpoint/resume + Rich 终端输出 |
| 配置系统 | ✅ 完成 | YAML + 环境变量 + 6 套 baseline 配置 |
| Prompt 模板 | ✅ 完成 | M1–M6 全部 prompt 已就绪，M3 含专用 batch edge-only prompt |

详细开发计划见 [TODO.md](TODO.md)。

## 离线预览终端输出

修改终端展示代码后，可以使用离线预览脚本检查布局，无需调用 LLM 或检索 API：

```bash
# 预览全部模块
python scripts/preview_terminal.py

# 模拟不同终端宽度
python scripts/preview_terminal.py --width 80
python scripts/preview_terminal.py --width 120

# 只预览指定模块
python scripts/preview_terminal.py --section m5
python scripts/preview_terminal.py --section m6
```

支持的模块选项为 `m1`–`m6` 和 `all`，默认宽度为 100 列。
