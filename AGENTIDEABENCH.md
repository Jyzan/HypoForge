# GLM-5.1 + HypoForge 的 AgentIdeaBench 实验

本入口运行完整的 HypoForge M1–M6，然后使用 AgentIdeaBench 的 `lit8d`
提示词和三个外部评审独立评分。默认是 **HypoForge 原生资源设置**：候选生成、
内部审查、证据图、研究计划和迭代都保留，不限制为论文的 10 次工具调用。
与 Table 2 的 GLM-5.1 Active 行比较时，应报告资源、服务商及检索时间的差异。

## 环境与凭据

建议使用独立的 Python 3.11 或 3.12 环境，在 HypoForge 仓库根目录执行：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

本机已建立并验证的 `.venv` 使用 Python 3.12。默认实验配置为
`configs/agentideabench_glm51.yaml`，所有文本模型层级都是 `glm-5.1`，显式
`enable_thinking=false`，三次生成分别使用种子 42、43、44，避免支持 seed 的服务
因重复同一随机流而给出三个相同样本；导出和外部评分固定 seed=42。
生成温度 0.7，导出和外部评分温度 0.0。
`text-embedding-v3` 单独用于实体规范化。内部模块仍可按原有逻辑使用自己的调用温度。

将 AgentIdeaBench 放在 HypoForge 的同级目录，或在每次命令中传入
`--agentideabench-root /path/to/AgentIdeaBench`。没有该仓库时可以克隆：

```bash
git clone https://github.com/HKUST-KnowComp/AgentIdeaBench.git ../AgentIdeaBench
```

API Key 支持直接读取百炼导出的 CSV，无须复制密钥到仓库。
CSV 可为纵向键值表，也可为单记录表，字段为 `apiKey` / `openAiCompatible`
或 `api_key` / `base_url`。

```bash
export HF_API_KEY_CSV="/absolute/path/model-studio-key.csv"
python run_agentideabench.py --phase check --api-key-csv "$HF_API_KEY_CSV"
```

`check` 会实际发送少量请求，检查 GLM 的 JSON 输出及指定调用参数、另两个评审、
嵌入模型和 Semantic Scholar 的日期过滤检索。也可不传 CSV，使用原有 `.env`
或环境中的 `OPENAI_API_KEY` / `OPENAI_BASE_URL`。现有配置优先解析 `QWEN_API_KEY`
与 `QWEN_BASE_URL`；显式 CSV 优先于这些设置。

检索可使用 `SEMANTIC_SCHOLAR_API_KEY`，以及原生 HypoForge 启用的其他来源凭据。
Semantic Scholar 的匿名访问可能限流；外部评分检索收到 429、网络错误等失败时，
本入口会保留错误并停止该条评分，不将其当成“没有相关文献”。模型检查通过而
检索检查失败时，先处理网络、限流或检索凭据，再进行完整评分。

## 先做五学科试跑

试跑选用公开评分数据中每个学科按子领域名称排序后的第一个子领域，各运行一次。
该子集是本入口固定的工程试跑集，不是原论文的 20 子领域 pilot。

```bash
python run_agentideabench.py --sample pilot --repeats 1 --dry-run
python run_agentideabench.py --sample pilot --repeats 1 --api-key-csv "$HF_API_KEY_CSV"
python run_agentideabench.py --phase score --sample pilot --api-key-csv "$HF_API_KEY_CSV"
python run_agentideabench.py --phase analyze --sample pilot
```

默认输出在 `output/agentideabench/glm51-pilot/`。只检查一个子领域可在生成命令中
加 `--limit 1`，同时指定独立的 `--output-dir output/agentideabench/smoke`；后续评分
和分析使用相同的 `--output-dir`。试跑分数仅对照历史数据的相同子领域，不能直接
对照全体 40 子领域的 6.33。

## 正式 40 × 3 实验

```bash
python run_agentideabench.py --sample full --repeats 3 --dry-run
python run_agentideabench.py --sample full --repeats 3 --api-key-csv "$HF_API_KEY_CSV"
python run_agentideabench.py --phase score --sample full --api-key-csv "$HF_API_KEY_CSV"
python run_agentideabench.py --phase analyze --sample full
```

输出在 `output/agentideabench/glm51-full/`。正式生成是 120 次独立完整管线运行；
评分至少涉及 360 次评审调用，另有查询提取、检索和必要的格式重试。
执行是串行的，便于追踪错误与检索限流。可用 `--output-dir` 保存不同实验。
同一输出目录应只由一个命令执行生成或评分，避免同时写入。

## 恢复与结果

重复执行相同命令会跳过已完成项。失败项不会自动变成新的随机样本；查看对应的
`result.json` 或评分错误后，给生成或评分命令加 `--retry-failed` 明确重试。
成功的假说不会重新生成，成功的评审不会重新评分；仅导出失败时复用已完成的管线。
管线本身失败时重试使用新的 attempt 目录，保留原有检查点和错误。

输出包括：

- `manifest.json`：固定任务、配置、代码版本、公开数据摘要与实验规则，不包含 API Key。
- `cells/<item_id>/attempt-N/`：每次管线的完整状态、检查点、事件与独立缓存。
- `cells/<item_id>/result.json`：最终假说来源、导出文本、状态、用量、检索记录及重试历史。
- `submissions.jsonl`：所有成功导出的假说；遗漏项同时列在 `generation_summary.json`。
- `scores/<item_id>/`：冻结的检索证据、三位评审的原始输出与用量。
- `score_manifest.json`：评审接口、提示词摘要、截止日期与评分参数。
- `comparison.json`：总分、五维分数、相同子领域历史基线、差值、胜率及按子领域配对
  bootstrap 的 95% 区间；未完成时列出遗漏项，不产生完整总分。

配置或任务集变化时，需使用新输出目录。检查点恢复中的 prompt/API 版本变化不能
保证逐 token 再现。计量包含成功返回的文本模型调用；网络重试、嵌入请求等不属于
这些文本 token 计数，检索详情保留在管线状态和 M2 导出中。

## 对齐规则和解释边界

1. 从 `release_data/core/lit8d_scores_3seed.csv.gz` 读取实际评分的 40 个子领域，
   校验五学科各 8 个，避免误用全部 100 个子领域或早期 paper-centric 流程。
2. 每个子领域独立运行三次，每次提交最终 `top_hypotheses[0]` 及其对应计划。
   不使用外部评审选择赢家，也不将一次运行的 top-3 当作三个独立样本。
3. 一个固定的 GLM 导出调用将已有卡片与计划压缩/翻译为 80–150 个英文单词。
   它被明确要求保留原有内容，最多进行一次格式重试；原文和调用用量均保留。
   格式校验不能证明语义完全忠实，正式报告前应抽样核对导出内容。
4. 外部评审沿用上游 `LIT8D_SYSTEM`、`USER_TEMPLATE` 和解析器，模型为
   GLM-5.1、Qwen3.6-Plus、Kimi-K2.6；三位评审均包含在主结果中。
5. 每条假说按三个查询检索先前工作，按查询交错保留最多 8 篇论文、摘要截断
   450 字符，固定检索截止日期为 `2026-05-31`。检索证据生成后缓存，供所有评审共享。
6. 每个维度删除三位评审中的最高分，再平均其余两分。总分权重为
   O=2、I=1.5、F=1、C=0.5、S=0.5，除以 5.5；先平均三个独立输出，再平均子领域。
   新实验要求三个评审齐全。历史数据则保留作者发布数据的缺失评审处理规则。
7. 完整历史 Active 基线可还原为约 6.3326，四舍五入为 Table 2 的 6.33。
   百炼与论文的 OpenRouter 后端、当前检索索引、原生流程预算均有差异。
   本入口的独立生成种子为 42/43/44，论文报告 generation seed=42，也应注明。
   因此分数差值是跨条件的系统结果，尚不能单独归因于 HypoForge 的机制。
   正式主张改进时应补充同接口原生 Active 基线、资源控制及评审尺度校准。
8. 原生生成检索使用当前资料，外部评分沿用历史截止日期。这不是历史文献冻结实验；
   当前新文献可能影响生成及原创性解释。需要历史冻结版本时应作为单独实验实现。

上游项目：[AgentIdeaBench](https://github.com/HKUST-KnowComp/AgentIdeaBench)。
评分提示词由运行时的上游 checkout 加载，代码按上游 MIT 许可使用；公开生成数据
按其 CC BY 4.0 许可使用，结果报告应注明来源和 checkout 版本。

## 验证

```bash
python -m pytest tests/test_agentideabench.py tests/test_pipeline.py -q
```

测试涵盖 API 参数在 JSON 回退中的保持、CSV 解析、独立缓存、恢复、检索失败、
逐维度聚合和缺失评审处理。同级 AgentIdeaBench 存在时还验证历史 6.33 基线。
