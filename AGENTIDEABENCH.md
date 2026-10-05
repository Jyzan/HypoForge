# HypoForge 的 AgentIdeaBench 实验

本入口运行完整的 HypoForge M1–M6，然后使用 AgentIdeaBench 的 `lit8d`
提示词和三个外部评审独立评分。默认是 **HypoForge 原生资源设置**：候选生成、
内部审查、证据图、研究计划和迭代都保留，不限制为论文的 10 次工具调用。
与 Table 2 中同一模型的 Active 行比较时，应报告资源、服务商及检索时间的差异。

整体流程分四个阶段，均通过 `run_agentideabench.py` 的 `--phase` 选择：

| 阶段 | 作用 |
| --- | --- |
| `check` | 预检 PDF 解码、生成模型、嵌入模型、外部评审和文献检索 |
| `generate`（默认） | 对每个子领域运行完整 HypoForge 管线并导出假说 |
| `score` | 检索先前工作证据，由三位外部评审按 `lit8d` 打分 |
| `analyze` | 聚合分数，与历史基线比较，写出 `comparison.json` |

## 环境与凭据

### Python 环境

使用独立的 Python 3.11 或 3.12 环境，在 HypoForge 仓库根目录执行：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### AgentIdeaBench 仓库

将 AgentIdeaBench 放在 HypoForge 的同级目录，或在每次命令中传入
`--agentideabench-root /path/to/AgentIdeaBench`。没有该仓库时可以克隆：

```bash
git clone https://github.com/HKUST-KnowComp/AgentIdeaBench.git ../AgentIdeaBench
```

任务集和历史基线来自其中的 `release_data/core/lit8d_scores_3seed.csv.gz`，
评分提示词在运行时从该仓库加载。

### 选择生成模型

生成模型从 `.env` 的 `HYPOFORGE_MODEL` 读取，填写服务商
使用的模型 ID：

```bash
HYPOFORGE_MODEL=glm-5.1               # 百炼
# HYPOFORGE_MODEL=Pro/zai-org/GLM-5.1 # 硅基流动
```

未设置时 `configure` 直接报错，不会回退到其他模型。`.env` 以 `override=True`
加载，会覆盖同名的 shell 环境变量。模型 ID 进入实验指纹，换模型须使用新输出目录；
默认输出目录按模型名生成，例如 `Pro/zai-org/GLM-5.1` → `output/agentideabench/glm51-pilot`。

`analyze` 按模型 ID 的最后一段（不区分大小写）在 AgentIdeaBench 发布数据中匹配
历史 Active 基线，例如 `glm-5.1` → `z-ai/glm-5.1`。服务商命名不同时用
`AGENTIDEABENCH_BASELINE_MODEL` 显式指定；发布数据中没有该模型时只报告绝对分数，
不输出差值。三位外部评审属于基准协议，不随生成模型变化。

### 模型服务凭据

凭据有三种提供方式，优先级从高到低：

1. `--api-key-csv`：百炼导出的 CSV。可为纵向键值表，也可为单记录表，字段为
   `apiKey` / `openAiCompatible` 或 `api_key` / `base_url`。
2. `--api-key-file`：JSON，或包含一条 `sk-` Key 和一条 HTTPS 接口地址的文本，
   支持地址外围的中文引号。
3. `.env` 或环境变量中的 `QWEN_API_KEY` / `QWEN_BASE_URL`，其次是
   `OPENAI_API_KEY` / `OPENAI_BASE_URL`。

凭据只在运行时读取，不会写入配置、manifest 或输出目录。

外部评审（GLM-5.1、Qwen3.6-Plus、Kimi-K2.6）使用百炼的模型名。生成与评审不在同一
服务商时，用 `--critic-api-key-csv` / `--critic-api-key-file` 单独提供评审凭据。
生成使用硅基流动接口时，`score` 和完整 `check` 必须提供评审凭据，否则直接报错。

嵌入模型默认沿用生成凭据，可用 `--embedding-api-key-csv` /
`--embedding-api-key-file` 单独指定。`--embedding-model` 会同时替换实体规范化和
一致性评估所用的嵌入模型，并改变实验指纹。切换接口不会自动替换配置中的嵌入模型
或三位评审。

### 文献检索与网络

检索可使用 `SEMANTIC_SCHOLAR_API_KEY`，以及原生 HypoForge 启用的其他来源凭据。
Semantic Scholar 的匿名访问很容易限流，建议使用 Key：

```bash
export SEMANTIC_SCHOLAR_API_KEY="<your key>"
export SEMANTIC_SCHOLAR_MIN_INTERVAL_SECONDS=5
```

`SEMANTIC_SCHOLAR_MIN_INTERVAL_SECONDS` 同时控制生成侧和外部评分的请求间隔，默认
5 秒。搜索、论文详情和外部评分器共享按 Key 划分的文件锁，同一台机器的不同进程共用
请求间隔及 429 冷却；不同机器之间不共享。同一个检索 Key 建议只运行一个实验进程，
需要并行时通过本入口的 `--workers` 实现。

生成侧和外部评分都遵循 `http_proxy` / `https_proxy` 等代理环境变量，不强制直连。
需要代理时，不要为文献 API 设置绕过代理的 `NO_PROXY` / `no_proxy` 规则。

外部评分检索遇到 429、网络错误等失败时，会保留错误并停止该条评分，不会当成
“没有相关文献”。预检中的一次成功检索不能保证整轮实验不限流。

## 实验配置

两份配置都运行完整的标准 M1–M6，所有文本模型层级使用同一个生成模型，显式
`enable_thinking=false`。三次独立生成分别使用种子 42、43、44，避免支持 seed 的
服务因重复同一随机流而给出三个相同样本；导出和外部评分固定 seed=42。生成温度
0.7，导出和外部评分温度 0.0。内部模块仍可按原有逻辑使用自己的调用温度。
生成检索来源固定为 Semantic Scholar 和 PubMed，默认关闭学科路由；若增加其他来源，
应使用新的配置和输出目录，并报告资源条件的变化。

| 配置 | 适用接口 | 说明 |
| --- | --- | --- |
| `configs/agentideabench.yaml`（默认） | 百炼等 | 原生预算；嵌入模型 `text-embedding-v3` |
| `configs/agentideabench_siliconflow.yaml` | 硅基流动 | 接口固定为 `https://api.siliconflow.cn/v1`；嵌入模型 `Qwen/Qwen3-Embedding-8B` |

硅基流动配置保留完整 M1–M6、搜索预算、候选数、迭代数和种子，为适应接口的响应
时延调整了批次与时间预算：

- M4 七个候选按 2/2/2/1 小批次生成，边界审核同样分批、最多并发两批，仍逐一审核
  全部候选并保留前三选择；M4 共享预算 900 秒，节点上限 1200 秒。
- M5 验证审核每批 3 个目标、最多并发 2 批，每个假说的全部批次共享 120 秒预算。
- M6 的三个内部评审并发执行，每个评审输出上限 2048 tokens。
- Semantic Scholar 单次查询的等待与重试预算为 25 秒。

这些是新的生成条件，不是论文的资源匹配复现，结果须与默认配置分开报告。M4 没有
有效模型候选时该样本直接记为失败，确定性降级候选不能导出为测评结果。

## 运行步骤

以下命令均在 HypoForge 仓库根目录执行。示例使用 `--api-key-file`，换成
`--api-key-csv` 或省略（使用 `.env`）同样可以。

```bash
source .venv/bin/activate
export HF_API_KEY_FILE="/path/to/llm_api.txt"
export HF_CRITIC_KEY_CSV="/path/to/critic-key.csv"   # 评审与生成同服务商时可省略
export HF_CONFIG="configs/agentideabench.yaml"        # 或 configs/agentideabench_siliconflow.yaml
export HF_OUTPUT="output/agentideabench/<实验名>"
```

### 1. 预检

```bash
python run_agentideabench.py --phase check --config "$HF_CONFIG" \
  --api-key-file "$HF_API_KEY_FILE" --critic-api-key-csv "$HF_CRITIC_KEY_CSV"
```

`check` 先离线解析一个内嵌 CFF Type1 字体的测试 PDF，校验实际字符解码；随后发送
少量请求，检查生成模型的 JSON 与普通文本输出及调用参数、嵌入模型、三位外部评审和
Semantic Scholar 的日期过滤检索。只想先验证生成侧时加 `--check-scope generation`，
此时跳过外部评审。

预检通过不能代替 M1–M6 的真实执行，正式实验前务必先完成下面的五学科试跑。

### 2. 五学科试跑

试跑选用公开评分数据中每个学科按子领域名称排序后的第一个子领域，各运行一次。
该子集是本入口固定的工程试跑集，不是原论文的 20 子领域 pilot。

```bash
python run_agentideabench.py --sample pilot --repeats 1 --dry-run --output-dir "$HF_OUTPUT"
python -u run_agentideabench.py --sample pilot --repeats 1 --workers 2 --llm-concurrency 8 \
  --config "$HF_CONFIG" --api-key-file "$HF_API_KEY_FILE" --output-dir "$HF_OUTPUT"
python -u run_agentideabench.py --phase score --sample pilot --workers 3 \
  --config "$HF_CONFIG" --api-key-file "$HF_API_KEY_FILE" \
  --critic-api-key-csv "$HF_CRITIC_KEY_CSV" --output-dir "$HF_OUTPUT"
python run_agentideabench.py --phase analyze --sample pilot --output-dir "$HF_OUTPUT"
```

`--dry-run` 只打印任务计划，不调用 API、不需要凭据。生成结束后先检查
`generation_summary.json` 和各项 `cells/<item_id>/result.json`；M1 覆盖检查或 M2
证据获取失败时，先处理原因再扩大实验。只检查一个子领域可加 `--limit 1`，并使用
独立的 `--output-dir`。

在 macOS 上长时间运行时，可在生成命令前加 `caffeinate -i`，防止系统空闲休眠中断进程。

试跑分数仅对照历史数据的相同子领域，不能直接对照全体 40 子领域的 6.33。
不指定 `--output-dir` 时，默认输出在 `output/agentideabench/<模型名>-pilot/`。

### 3. 正式 40 × 3 实验

```bash
python run_agentideabench.py --sample full --repeats 3 --dry-run --output-dir "$HF_OUTPUT"
python -u run_agentideabench.py --sample full --repeats 3 --workers 2 --llm-concurrency 8 \
  --config "$HF_CONFIG" --api-key-file "$HF_API_KEY_FILE" --output-dir "$HF_OUTPUT"
python -u run_agentideabench.py --phase score --sample full --workers 3 \
  --config "$HF_CONFIG" --api-key-file "$HF_API_KEY_FILE" \
  --critic-api-key-csv "$HF_CRITIC_KEY_CSV" --output-dir "$HF_OUTPUT"
python run_agentideabench.py --phase analyze --sample full --output-dir "$HF_OUTPUT"
```

正式生成是 120 次独立完整管线运行；评分至少涉及 360 次评审调用，另有查询提取、
检索和必要的格式重试。不指定 `--output-dir` 时，默认输出在
`output/agentideabench/<模型名>-full/`。同一输出目录只允许一个生成或评分命令占用，
重复启动会被文件锁拒绝。

## 并行与耗时

生成阶段 `--workers 1–4`（默认 1）控制同时运行的独立管线数。建议先使用 2，检查
试跑中的耗时、token 用量、429、超时及各模块完成情况，再决定是否升到 4。
`--llm-concurrency`（默认 8）是所有生成管线和导出调用共享的文本请求并发上限；它
不是 RPM/TPM 限流器，SDK 内部仍可能因服务端限制而重试。出现限流时可降低至 4。

跨管线的 M2 按顺序执行，排队发生在模块计时开始之前，因此不消耗该次 M2 的时间
预算。M2 内部的来源并发和全局请求间隔保留，M1、M3–M6 和导出可与其他管线重叠。
每条假说有独立的 runner、状态、知识图、实体缓存、随机种子和用量计数；并行不会把
单次 top-3 当作三个独立实验。M2 是串行部分，4 条管线不能保证四倍提速，应先用
五学科试跑的吞吐量估计正式生成时间。

评分阶段 `--workers 3` 让三个评审同时读取该假说已经冻结的同一份证据，各有独立
客户端和用量计数；证据提取和 Semantic Scholar 查询保持串行。

`execution_history.jsonl` 记录每次执行的并发设置和实际总耗时。仅改变 worker 数或
文本并发上限可以继续同一实验。

## 断点、失败与输出

### 断点恢复与重试

重复执行相同命令会跳过已完成项。失败项不会自动变成新的随机样本；查看对应的
`result.json` 或评分错误后，给生成或评分命令加 `--retry-failed` 明确重试。
成功的假说不会重新生成，成功的评审不会重新评分；仅导出失败时复用已完成的管线。
管线本身失败时重试使用新的 attempt 目录，保留原有检查点和错误。

### 何时必须换新输出目录

`manifest.json` 记录实验指纹，包含配置、生成模型、任务集、导出提示词及相关源码
文件的哈希。指纹不一致时程序拒绝在原目录续跑。因此以下任一变化都要使用新的
`--output-dir`：

- 更换生成模型、配置文件或任务集；
- 修改 HypoForge 中进入指纹的源码或依赖（包括拉取了新代码）。

旧目录应保留，不要把不同指纹的样本混合统计。

### 常见失败

- **模型返回 401/403**：直接抛出错误，不通过切换 thinking 参数或普通文本模式重试。
  生成器保存当前失败记录、取消其他在途管线并停止批次，已完成结果保留。
  `AccessDenied.Unpurchased` 本身不能证明额度耗尽，应结合对应业务空间的模型开通
  状态、额度、账单及调用日志判断。
- **Semantic Scholar 429**：生成侧持 Key 的请求在 25 秒查询预算内最多重试一次；
  匿名访问不重试。本地排队或请求超时不会被标记为服务端限流。
- **M1 检查失败**：M1 的覆盖与拆分检查严格执行，预算耗尽后错误记录包含最终审核
  理由及子问题，可据此定位原因。
- **M2 无可用证据**：通常是检索限流或网络问题，先处理检索凭据和网络再重试。
- **M6 声明结构异常**：被异常 JSON 恢复为字符串的声明会尝试重新解析，整批无效时
  保留“证据图覆盖不足”的降级结果，不丢弃坏行来改变分母。

定位问题时，先用 `--sample pilot --limit 1 --workers 1` 运行单条样本，检查模块状态
和检索记录。每次尝试的终端会显示各模块开始/结束，详细事件与阶段快照保存在该次
尝试的 `telemetry/` 目录。

### 从中间状态调试

`run_agentideabench_diagnostic.py` 可以从已保存的成功 M3 或 M4 状态继续运行后续
模块，用于单独排查下游问题。该入口只接受 `--api-key-file` 形式的凭据。例如，从 M4
状态只验证 M5–M6：

```bash
python -u run_agentideabench_diagnostic.py \
  --after m4 --state /path/to/successful-m4-state.json \
  --config "$HF_CONFIG" --api-key-file "$HF_API_KEY_FILE" \
  --output-dir output/agentideabench/<诊断目录>
```

该入口保留标准模型和质量检查，给 M6 一次评审机会，关闭自动迭代。它使用新缓存和
新输出目录，记录源状态哈希及诊断条件，不能覆盖已有目录，不导出基准 submission，
也不运行外部评分。`completed` 只表示后续流程执行完成，假说质量需另看评审结果。
诊断结果不计入正式基准，正式评测须从头运行完整、指纹一致的标准配置。

### 输出文件

- `manifest.json`：固定任务、配置、代码版本、公开数据摘要与实验规则，不包含 API Key。
- `cells/<item_id>/attempt-N/`：每次管线的完整状态、检查点、事件与独立缓存。
- `cells/<item_id>/result.json`：最终假说来源、导出文本、状态、用量、检索记录及重试历史。
- `submissions.jsonl`：所有成功导出的假说；遗漏项同时列在 `generation_summary.json`。
- `scores/<item_id>/`：冻结的检索证据、三位评审的原始输出与用量。
- `score_manifest.json`：评审接口、提示词摘要、截止日期与评分参数。
- `execution_history.jsonl`：每次执行的并发设置、起止时间与实际总耗时。
- `comparison.json`：总分、五维分数、相同子领域历史基线、差值、胜率及按子领域配对
  bootstrap 的 95% 区间；未完成时列出遗漏项，不产生完整总分。

检查点恢复中的 prompt/API 版本变化不能保证逐 token 再现。计量包含成功返回的文本
模型调用；网络重试、嵌入请求等不属于这些文本 token 计数，检索详情保留在管线状态
和 M2 导出中。

## PDF 依赖与下载完整性

生成和 `check` 均在调用模型之前执行本地 CFF PDF 字体解码检查，正常输出为
`PDF CFF font decoding / text extraction: OK`。`requirements.txt` 显式包含
`fonttools` 和已验证支持该解码方式的 `pypdf>=6.19.0`。`pip check` 只验证已声明的
依赖关系，不能替代这个实际解码检查。安装新依赖后需启动新的 Python 进程。也可以
单独运行该检查：

```bash
python -c 'import asyncio; from hypoforge.benchmarks.agentideabench import check_pdf_environment; asyncio.run(check_pdf_environment())'
```

PDF 下载会核对响应的 Content-Length（存在有效值时）、PDF 文件头与结尾
`%%EOF` 标记。已缓存但缺少结尾标记的文件会保留为 `paper.invalid-*.pdf`，随后
重新下载；不完整下载最多重试一次，两次共用总下载时间预算。HTML、出版商拒绝访问
等不触发这一重试。持续失败不会写入正常 PDF 缓存，有摘要时降级使用摘要，并在 M2
导出中保留失败原因与实际内容层级。完整性检查只校验 PDF 外壳，其他结构问题由
解析器处理并记录。

## 对齐规则和解释边界

1. 从 `release_data/core/lit8d_scores_3seed.csv.gz` 读取实际评分的 40 个子领域，
   校验五学科各 8 个，避免误用全部 100 个子领域或早期 paper-centric 流程。
2. 每个子领域独立运行三次，每次提交最终 `top_hypotheses[0]` 及其对应计划。
   不使用外部评审选择赢家，也不将一次运行的 top-3 当作三个独立样本。
3. 一个使用生成模型的固定导出调用将已有卡片与计划压缩/翻译为 80–150 个英文单词。
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
   本入口与论文的 OpenRouter 后端、当前检索索引、原生流程预算均有差异。
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
python -m pytest tests -q
```

测试涵盖 API 参数在 JSON 回退中的保持、CSV 解析、生成模型从环境变量读取、并行
状态/用量隔离、M2 与文本并发上限、取消恢复、目录写锁、共享评分证据、失败评审
重试、检索失败、逐维度聚合和缺失评审处理，以及实际 CFF 字体解码、截断 PDF 的
有界重试与缓存隔离、摘要降级归因、取消和总下载时间预算。同级 AgentIdeaBench
存在时还验证历史 6.33 基线。失败回归还覆盖 401/403 的立即停止及在途任务取消、
导出文本预检、异常声明结构、有界 Semantic Scholar 429 重试、超时不触发限流熔断，
以及 M1 的严格检查与诊断保留。
