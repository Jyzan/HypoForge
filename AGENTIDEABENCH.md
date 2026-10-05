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

`check` 先离线解析一个内嵌 CFF Type1 字体的测试 PDF，校验实际字符解码；
随后发送少量请求，检查 GLM 的 JSON 与普通文本输出及指定调用参数、另两个评审、
嵌入模型和 Semantic Scholar 的日期过滤检索。也可不传 CSV，使用原有 `.env`
或环境中的 `OPENAI_API_KEY` / `OPENAI_BASE_URL`。现有配置优先解析 `QWEN_API_KEY`
与 `QWEN_BASE_URL`；显式 CSV 优先于这些设置。

检索可使用 `SEMANTIC_SCHOLAR_API_KEY`，以及原生 HypoForge 启用的其他来源凭据。
Semantic Scholar 的匿名访问可能限流；外部评分检索收到 429、网络错误等失败时，
本入口会保留错误并停止该条评分，不将其当成“没有相关文献”。模型检查通过而
检索检查失败时，先处理网络、限流或检索凭据，再进行完整评分。

生成侧 Semantic Scholar/OpenAlex 和外部评分均遵循代理环境变量，不强制直连。
本机使用 7890 端口代理，Semantic Scholar Key 文件是单行明文时，可在启动前执行：

```bash
export http_proxy="http://127.0.0.1:7890"
export https_proxy="$http_proxy"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"
export SEMANTIC_SCHOLAR_API_KEY="$(cat '../semantic_scholar_api.txt')"
export SEMANTIC_SCHOLAR_MIN_INTERVAL_SECONDS=5
```

不要为文献 API 设置绕过代理的 `NO_PROXY` / `no_proxy` 规则。生成侧默认将
Semantic Scholar 请求间隔限制为 5 秒。搜索、论文详情和独立评分器共享按 Key
划分的文件锁，同一台机器的不同进程也共用请求间隔及 429 冷却；不同机器不共享。
该变量现在同样控制外部评分请求。先通过 `check` 并完成五学科试跑，再启动
正式实验。独立 Key 不能保证永久没有 429；失败记录和断点恢复仍保留。

本配置的生成检索来源固定为 Semantic Scholar 和 PubMed，默认关闭学科路由。
Semantic Scholar 不可用时，PubMed 对计算机和物理学科的覆盖可能不足；预检中的
一次成功检索不能证明整轮实验不会限流。若增加其他来源，应使用新的配置和输出
目录，并报告资源条件的变化。

## 先做五学科试跑

### 硅基流动生成试跑，外部评分暂缓

`configs/agentideabench_glm51_siliconflow.yaml` 保留完整 M1–M6、搜索预算、候选数、
迭代数和种子，使用 `Pro/zai-org/GLM-5.1`，实体规范化和证据一致性嵌入均改为
`Qwen/Qwen3-Embedding-8B`。这是新的提供商及嵌入条件，必须使用独立输出目录。
原有百炼配置保留。`--api-key-file` 从 JSON 或包含一条 `sk-` Key 和一条 HTTPS
接口地址的文本读取凭据，支持地址外围的中文引号，不把密钥写入配置或 manifest。

2026-10-05 的首轮五学科试跑全部失败。修复版增加覆盖修复后的原子性重写和
独立复审；Semantic Scholar 仍使用原 `/paper/search` 接口，但按其文档发送纯文本
短关键词，避免强制 AND 拼接整条任务名与通用输出要求。M1 原始任务合同保留，
检索及生成结果仍接受后续任务、证据和科学性检查。

硅基流动配置将七候选生成拆为 2/2/2/1 小批次，保留全部候选的审查与前三选择；
三个独立内部评审并发执行，每个评审的输出上限改为 2048 tokens。阶段时间上限
及检索预算未改变，Semantic Scholar 单次查询等待与重试预算由 15 秒改为 25 秒，
仍小于源检索的 30 秒上限。这些是明确记录的新生成条件，不是论文资源匹配复现。
若 M4 恢复后仍没有有效模型候选，该配置直接记为失败；确定性降级候选不能导出
为有效测评结果。终端显示各模块开始/结束，详细事件与阶段快照保存在每次尝试的
`telemetry/` 目录。原失败目录保留，修复版必须使用新的输出目录。

本地 Mac 可执行以下命令。`caffeinate -i` 在生成进程运行期间阻止空闲休眠；
两条管线共享 8 个文本请求并发名额，M2 仍按顺序运行。

```bash
cd "/Users/jiangyi/Desktop/HypoForge课题文件/HypoForge"
source .venv/bin/activate
export http_proxy="http://127.0.0.1:7890"
export https_proxy="$http_proxy"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"
unset no_proxy NO_PROXY
export SEMANTIC_SCHOLAR_API_KEY="$(cat '../semantic_scholar_api.txt')"
export SEMANTIC_SCHOLAR_MIN_INTERVAL_SECONDS=5
export HF_API_KEY_FILE="../llm_api.txt"
export HF_CONFIG="configs/agentideabench_glm51_siliconflow.yaml"
export HF_OUTPUT="output/agentideabench/glm51-siliconflow-pilot-v2"

python run_agentideabench.py --phase check --check-scope generation \
  --config "$HF_CONFIG" --api-key-file "$HF_API_KEY_FILE" && \
caffeinate -i python -u run_agentideabench.py --sample pilot --repeats 1 \
  --workers 2 --llm-concurrency 8 --config "$HF_CONFIG" \
  --api-key-file "$HF_API_KEY_FILE" --output-dir "$HF_OUTPUT"
```

`--check-scope generation` 校验 PDF、GLM JSON/普通文本、嵌入和 Semantic Scholar，
明确跳过外部评审。请求预检通过后可以开始生成试跑，但不能代替 M1–M6 的真实
执行验证。先检查五项的 `generation_summary.json` 和各项 `result.json`；M1 覆盖
检查或 M2 证据获取仍失败时，保留记录并处理原因，再扩大实验。

外部评分保持原三评审 GLM-5.1、Qwen3.6-Plus、Kimi-K2.6，暂不运行 `score/analyze`。
原服务恢复后先做完整预检，并传入独立评审凭据：

```bash
python run_agentideabench.py --phase check --config "$HF_CONFIG" \
  --api-key-file "$HF_API_KEY_FILE" --critic-api-key-csv "../默认业务空间-apiKey-6225319.csv" && \
python run_agentideabench.py --phase score --sample pilot --workers 3 \
  --config "$HF_CONFIG" --api-key-file "$HF_API_KEY_FILE" \
  --critic-api-key-csv "../默认业务空间-apiKey-6225319.csv" --output-dir "$HF_OUTPUT"
python run_agentideabench.py --phase analyze --sample pilot --output-dir "$HF_OUTPUT"
```

`--embedding-api-key-csv` / `--embedding-api-key-file` 可指定独立嵌入凭据；默认沿用
生成凭据。`--embedding-model` 是显式模型替换，同时更新实体规范化和一致性评估的
模型名称，并改变实验指纹。切换接口不会自动替换原配置中的嵌入模型或三位评审。

### 原百炼配置

试跑选用公开评分数据中每个学科按子领域名称排序后的第一个子领域，各运行一次。
该子集是本入口固定的工程试跑集，不是原论文的 20 子领域 pilot。

```bash
python run_agentideabench.py --sample pilot --repeats 1 --dry-run
python run_agentideabench.py --sample pilot --repeats 1 --workers 2 --api-key-csv "$HF_API_KEY_CSV"
python run_agentideabench.py --phase score --sample pilot --workers 3 --api-key-csv "$HF_API_KEY_CSV"
python run_agentideabench.py --phase analyze --sample pilot
```

默认输出在 `output/agentideabench/glm51-pilot/`。只检查一个子领域可在生成命令中
加 `--limit 1`，同时指定独立的 `--output-dir output/agentideabench/smoke`；后续评分
和分析使用相同的 `--output-dir`。试跑分数仅对照历史数据的相同子领域，不能直接
对照全体 40 子领域的 6.33。

## 正式 40 × 3 实验

```bash
python run_agentideabench.py --sample full --repeats 3 --dry-run
python run_agentideabench.py --sample full --repeats 3 --workers 2 --api-key-csv "$HF_API_KEY_CSV"
python run_agentideabench.py --phase score --sample full --workers 3 --api-key-csv "$HF_API_KEY_CSV"
python run_agentideabench.py --phase analyze --sample full
```

输出在 `output/agentideabench/glm51-full/`。正式生成是 120 次独立完整管线运行；
评分至少涉及 360 次评审调用，另有查询提取、检索和必要的格式重试。
默认 `--workers 1`；上面的命令启用受控并行。可用 `--output-dir` 保存不同实验。
同一输出目录只允许一个生成或评分命令占用，重复启动会被文件锁拒绝。

## 并行与耗时

生成阶段 `--workers 1–4` 控制同时运行的独立管线数。建议先使用 2，检查试跑中
的耗时、token 用量、429、超时及各模块完成情况，再决定是否升到 4。
`--llm-concurrency 8` 是所有生成管线和导出调用共享的文本请求并发上限；它不是
RPM/TPM 限流器，SDK 内部仍可能因服务端限制而重试。出现限流时可降低至 4。

跨管线的 M2 按顺序执行，排队发生在模块计时开始之前，因此不消耗该次 M2 的
原有时间预算。M2 内部原有来源并发和全局请求间隔保留，M1、M3–M6 和导出可与
另一条管线重叠。每条假说仍有独立的 runner、状态、知识图、实体缓存、随机种子和
用量计数；并行不会把单次 top-3 当作三个独立实验。

评分阶段 `--workers 3` 让三个评审同时读取该假说已经冻结的同一份证据，各有独立
客户端和用量计数；证据提取和 Semantic Scholar 查询保持串行。失败评审单独保存，
`--retry-failed` 复用成功评审及既有证据。

`execution_history.jsonl` 记录每次调用的并发设置和实际总耗时。仅改变 worker 数或
文本并发上限可以继续同一实验；科学配置、任务集和已记录源码变更仍需新输出目录。
同一个检索 Key 建议只运行一个实验进程，通过本入口的 workers 增加并行。
120 次管线的耗时取决于真实样本、迭代次数和检索占比。M2 仍是串行部分，4 条管线
不能保证四倍提速，先用五学科试跑的吞吐量估计正式生成时间。

## PDF 依赖与下载完整性

生成和 `check` 均在调用模型之前执行本地 CFF PDF 字体解码检查，正常输出为
`PDF CFF font decoding / text extraction: OK`。`requirements.txt` 显式包含
`fonttools` 和已验证支持该解码方式的 `pypdf>=6.19.0`。`pip check` 仅验证已声明的
依赖关系，不能替代这个实际解码检查。安装新依赖后需启动新的 Python 进程。

PDF 下载会核对响应的 Content-Length（存在有效值时）、PDF 文件头与结尾
`%%EOF` 标记。已缓存但缺少结尾标记的文件会保留为 `paper.invalid-*.pdf`，
随后重新下载；不完整下载最多重试一次，两次共用原有总下载时间预算。
HTML、出版商拒绝访问等不触发这一重试。持续失败不会写入正常 PDF 缓存，
有摘要时仍沿用原生摘要降级，并在 M2 导出中保留失败原因与实际内容层级。
完整性检查只校验 PDF 外壳，其他结构问题仍由解析器处理并记录。

已停止的旧试跑应保留，用新的输出目录重跑。本次修复涉及 PDF 检索/解析源码及
依赖，实验指纹会变化，旧成功结果不能与修复后样本混合。服务器原有 `.venv`
可以继续使用，无需再次创建环境：

```bash
git pull --ff-only origin AgentIdeaBench
source .venv/bin/activate
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --timeout 60
python -m pip check
python -c 'import asyncio; from hypoforge.benchmarks.agentideabench import check_pdf_environment; asyncio.run(check_pdf_environment())'
```

重新设置上文的代理和凭据环境变量，再执行 `--phase check`。重跑五学科时，生成、
评分、分析均指定 `--output-dir output/agentideabench/glm51-pilot-pdf-fixed`：

```bash
python run_agentideabench.py --phase check --api-key-csv "$HF_API_KEY_CSV"
python -u run_agentideabench.py --sample pilot --repeats 1 --workers 2 --llm-concurrency 8 --output-dir output/agentideabench/glm51-pilot-pdf-fixed --api-key-csv "$HF_API_KEY_CSV"
python -u run_agentideabench.py --phase score --sample pilot --workers 3 --output-dir output/agentideabench/glm51-pilot-pdf-fixed --api-key-csv "$HF_API_KEY_CSV"
python -u run_agentideabench.py --phase analyze --sample pilot --output-dir output/agentideabench/glm51-pilot-pdf-fixed
```

## 恢复与结果

### API 与管线失败

模型返回 401/403 时，JSON 客户端会直接抛出错误，不再通过切换 thinking 参数或
普通文本模式重试。生成器收到该错误后保存当前失败记录、取消其他在途管线并停止
批次，保留已完成结果。`check` 同时检查 GLM 的普通文本调用，覆盖提交导出所用的
接口路径。`AccessDenied.Unpurchased` 本身不能证明免费额度耗尽，应结合对应业务
空间的模型开通状态、额度、账单及调用日志判断。

生成侧持 Key 的 Semantic Scholar 请求遇到 429 时，在原有 15 秒检索阶段预算内
最多重试一次；持续 429 仍开启 120 秒熔断。匿名访问不增加这一重试。本地排队或
请求超时不会被错误标记为服务端限流，也不会因此封锁后续查询。

M1 的覆盖与拆分检查仍严格执行。预算耗尽后，错误记录包含最终审核理由及子问题，
供定位失败原因。M6 对原子声明要求对象结构；被异常 JSON 恢复为字符串的声明会
尝试重新解析，整批无效时保留“证据图覆盖不足”的降级结果，不丢弃坏行来改变分母。

这类源码修复会改变实验指纹，保留旧失败目录，使用新的 `--output-dir`，例如
`output/agentideabench/glm51-pilot-repaired`。先通过 `--phase check`，再运行一条
独立样本（`--sample pilot --limit 1 --workers 1`）检查模块状态和检索记录。
M1 仍失败或 M2 无可用证据时，先处理对应原因；安装 PDF 依赖不能解决这些问题。

### 断点与导出

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
- `execution_history.jsonl`：每次执行的并发设置、起止时间与实际总耗时。
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
python -m pytest tests -q
```

测试涵盖 API 参数在 JSON 回退中的保持、CSV 解析、并行状态/用量隔离、M2 与文本
并发上限、取消恢复、目录写锁、共享评分证据、失败评审重试、检索失败、逐维度
聚合和缺失评审处理，以及实际 CFF 字体解码、截断 PDF 的有界重试与缓存隔离、
摘要降级归因、取消和总下载时间预算。同级 AgentIdeaBench 存在时还验证历史 6.33 基线。
失败回归还覆盖 401/403 的立即停止及在途任务取消、导出文本预检、异常声明结构、
有界 Semantic Scholar 429 重试、超时不触发限流熔断，以及 M1 的严格检查与诊断保留。
