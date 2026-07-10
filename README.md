# HypoForge

> 挑战杯 2026 · 赛题A：科学假设生成与研究计划设计

基于 LangGraph StateGraph 的六模块闭环 AI Scientist pipeline。从输入一个前沿科学问题开始，自动完成问题分解、文献检索、证据图谱构建、假设生成、研究计划设计和评审迭代。

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 跑通示例（当前为 stub 数据）
python run_hypoforge.py -q "蛋白质如何折叠及错误折叠导致疾病的机制？"

# 3. 用不同配置跑消融实验
python run_hypoforge.py -q "..." -c configs/baseline_b0.yaml
python run_hypoforge.py -q "..." -c configs/full_pipeline.yaml
```

运行后终端会输出 Rich 美化的六模块执行过程，并在 `output/` 目录下生成 JSON 结果文件。

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
