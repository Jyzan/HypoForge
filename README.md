# HypoForge

> 挑战杯 2026 · 赛题A：科学假设生成与研究计划设计

基于 LangGraph StateGraph 的六模块闭环 AI Scientist pipeline。从输入一个前沿科学问题开始，自动完成问题分解、文献检索、证据图谱构建、假设生成、研究计划设计和评审迭代。

## 快速开始

### Conda 环境（推荐）

```bash
# 1. 创建 conda 环境（Python 3.11+）
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
# 跑通示例（当前为 stub 数据）
python run_hypoforge.py -q "蛋白质如何折叠及错误折叠导致疾病的机制？"

# 用不同配置跑消融实验
python run_hypoforge.py -q "..." -c configs/baseline_b0.yaml
python run_hypoforge.py -q "..." -c configs/full_pipeline.yaml
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
| `--run-id` | | ❌ | 自动生成 | 自定义运行标识符，用于输出文件命名 |
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
| `configs/full_pipeline.yaml` | 完整系统 |

## 项目状态

当前为**骨架阶段**：所有六个模块均为 stub 实现，端到端可跑通。详细开发计划见 [TODO.md](TODO.md)。
