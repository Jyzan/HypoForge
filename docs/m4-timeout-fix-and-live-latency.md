# M4 超时修复与真实调用耗时

## 本版说明

本版用于修复 M4 在大语言模型正常长耗时调用期间被过早终止的问题。历史运行中，M4 `hypothesis_generator` 的正常返回时间已经超过原先的 120 秒限制；在保持推理模式和原始输入上下文的无超时复测中，该调用曾在 282.35 秒后正常返回。因此，旧限制会把仍在正常生成的请求误判为超时。

本版没有改变 M4 的假设生成、审查、可证伪性检查和排序逻辑，只调整分层超时预算，并增加配置回归测试。

## 超时配置

`configs/web_ui.yaml` 当前采用以下配置：

| 层级 | 配置值 | 作用 |
|---|---:|---|
| Qwen Plus 单次 HTTP 请求 | 330 秒 | 容纳已观测到的模型长尾响应 |
| M4 单次 LLM Tool 调用 | 360 秒 | 略高于底层请求上限，为解析和收尾留出余量 |
| M4 本轮总时间预算 | 840 秒 | 容纳 Generator、Auditor、Critic、Falsifiability 和 Ranker 的连续调用 |
| M4 节点总超时 | 900 秒 | 高于 M4 内部总预算，避免外层先于内部预算终止 |

此外，M1 节点超时调整为 240 秒，以覆盖历史日志中约 172 秒的 P90 长尾。

## 真实问题复测

复测日期：2026-08-21。两个样本均从周报保存的 M3 检查点继续运行完整 M4，用于验证超时修复；这不是完整 M1–M6 重跑。

### 问题一：Why does life require chirality?

| Tool | 耗时 |
|---|---:|
| `hypothesis_generator` | 112.562 秒 |
| `hypothesis_contract_auditor` | 5.797 秒 |
| `hypothesis_critic` | 49.125 秒 |
| `falsifiability_checker` | 22.437 秒 |
| `hypothesis_ranker` | 29.547 秒 |
| **M4 总耗时** | **219.490 秒** |

结果：运行完成，得到 1 个最终假设。一个未覆盖必要任务实体 `chirality` 的候选被正常过滤，未触发超时或恢复流程。

### 问题二：Will the Navier–Stokes problem ever be solved?

该样本携带上一轮的 1 个假设和 10 条评审意见进入迭代 M4，因此上下文和推理负担更重。

| Tool | 耗时 |
|---|---:|
| `hypothesis_generator` | 262.094 秒 |
| `hypothesis_contract_auditor` | 8.860 秒 |
| `hypothesis_critic` | 67.953 秒 |
| `falsifiability_checker` | 26.000 秒 |
| `hypothesis_ranker` | 31.312 秒 |
| **M4 总耗时** | **396.246 秒** |

结果：运行完成，得到 1 个最终假设，未触发 Tool 报错、超时或候选不足重试。Generator 的 262.094 秒耗时表明，原先的 120 秒、180 秒或 240 秒限制均会错误终止这次正常调用；360 秒单次 Tool 上限可以覆盖该实测长尾。

## 回归保护

`scripts/test_pipeline.py` 增加了超时预算关系测试，确保：

- 底层 Qwen 请求时间能够覆盖已观测的长尾；
- M4 单次 Tool 上限高于底层请求上限；
- M4 本轮总预算高于单次 Tool 上限；
- M4 外层节点超时高于内部总预算。

真实运行日志和模型输出位于本地 `output/report_m4_retries/`，该目录属于运行产物，不纳入 Git 提交。
