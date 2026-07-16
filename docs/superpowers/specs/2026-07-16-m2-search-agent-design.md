# M2 Iterative Search Agent 设计

## 目标

在不删除、不改变 legacy M2 默认行为的前提下，实现一个可测试、可复现的迭代式论文搜索编排器。编排器接收一个具体子问题和 M1 提供的实体、领域、问题类型，调用组员实现的查询规划、单源检索、去重、排序、Scout Reading 和覆盖评估 Tool，最终返回 `SearchRunResult`。随后通过独立适配层与阅读提取流程共同转换为现有 Pipeline 使用的 `LiteratureResult`。

## 范围

本设计包含：

- Search Agent 主循环；
- 多数据源并发与单源失败降级；
- `SearchState`、预算消耗、低增益和无结果计数；
- 结构化停止原因；
- `SearchRunResult` 完整输出；
- legacy/agentic 显式配置边界；
- 离线单元测试和 Pipeline 接口测试。

本设计不包含：

- PubMed、Semantic Scholar/OpenAlex、Europe PMC 的真实 API 实现；
- Query Planner、Deduplicator、Ranker、Scout Reader、Coverage Evaluator 的具体算法；
- PDF 下载、全文解析、向量索引和论文级知识提取；
- 删除或重写 `hypoforge/modules/m2_literature_search.py` 中的 legacy 实现；
- 自动提交、推送或合并代码。

## 方案选择

采用纯 Python 异步编排器，不在 M2 内部再嵌套 LangGraph，也不使用自由 ReAct 循环。原因是它能够通过依赖注入独立测试每个分支，停止逻辑可复现，且不要求各组员共享私有实现。

核心公开入口：

```python
class IterativeSearchAgent:
    async def run(
        self,
        sub_question: str,
        key_entities: Sequence[str] = (),
        domains: Sequence[str] = (),
        question_type: str = "",
        budget: SearchBudget | None = None,
        existing_papers: Sequence[PaperRecord] = (),
    ) -> SearchRunResult:
        ...
```

构造函数注入以下协议实现：

- `QueryPlannerProtocol`；
- 一个或多个 `LiteratureSourceProtocol`；
- `PaperDeduplicatorProtocol`；
- `PaperRankerProtocol`；
- `ScoutReaderProtocol`；
- `CoverageEvaluatorProtocol`。

Agent 只依赖公共协议，不导入任何 Tool 的私有实现。

## 公共契约调整

### 查询规划

`QueryPlannerProtocol.plan()` 增加可选的 `state: SearchState | None` 参数。首轮收到初始空状态；后续轮次从状态中读取已使用查询、已知术语、已覆盖主题和缺口。这样同一个 Planner 可以同时承担初始规划与定向补充规划，不再引入第二套重规划协议。

`SearchQuery` 增加 `round_index`，由 Agent 在执行前统一写入，用于追踪各轮查询。

### SearchState

新增 `RemainingSearchBudget`，字段与 `SearchBudget` 对应但允许等于0。`SearchBudget` 继续表示创建任务时必须为正数的上限，`SearchState.remaining_budget` 改用 `RemainingSearchBudget`，避免用同一个模型同时表达“有效上限”和“已经耗尽”。

在 `SearchState` 现有字段上增加：

- `queries_executed`：实际发往数据源的查询数量；
- `unique_papers_seen`：去重后进入候选池的唯一论文数量；
- `estimated_tokens_used`：进入 Planner、Scout 和覆盖判断上下文的估算 Token 数；
- `elapsed_seconds`：Agent 墙钟耗时；
- `consecutive_low_gain_rounds`：连续低增益轮数；
- `consecutive_no_result_rounds`：连续无新增结果轮数。

`remaining_budget` 字段名继续保留，每轮执行后由 Agent 根据初始预算和上述累计用量生成，不允许出现负数。

Token预算定义为 Agent 上下文估算量，而不是供应商账单 Token。默认估算器使用稳定的字符近似算法，并允许构造函数注入替代估算器，保证离线测试可复现。

### SearchRunResult

在现有字段上增加：

- `source_result_counts: dict[str, int]`：各数据源累计成功返回数量；
- `scout_notes: list[ScoutNote]`：最终候选的轻量阅读结果；
- `reused_paper_ids: list[str]`：命中已有论文目录并复用的稳定 ID；
- `final_state: SearchState | None`：为兼容旧序列化数据允许缺省，但每次 Agent运行都必须填写停止时的压缩状态。

`coverage.missing_topics` 和 `coverage.missing_buckets` 表示未解决缺口；`queries` 中的 `round_index`、`purpose`、`target_gap` 和 `relation_to_question` 构成完整查询轨迹。

## 单轮执行流程

1. Planner 根据子问题和当前 `SearchState` 生成查询。
2. Agent 删除与历史查询重复的查询，并按剩余查询预算截断。
3. 根据 `SearchQuery.target_source` 路由到对应数据源。
4. 不同查询异步并发执行；每次调用有独立超时。
5. 单个来源异常或超时只写入 `failed_sources` 和 `errors`，不取消其他来源。
6. 将成功结果与当前候选池、`existing_papers` 一起送入 Deduplicator。
7. 对累计候选池排序并截取 Scout 候选。
8. Scout Reader 只处理轻量元数据和摘要。
9. Coverage Evaluator 输出结构化覆盖与缺口。
10. Agent 更新 `SearchState`、预算和连续计数，执行停止门。
11. 未停止时进入下一轮；停止时生成 `SearchRunResult`。

## 预算语义

- 查询预算按实际执行的 `SearchQuery` 数量计算。每条查询已经指定唯一目标数据源，因此一次查询对应一次数据源调用。
- 论文预算按跨来源、跨轮次去重后的唯一论文数量计算，重复命中不重复消耗。
- 时间预算使用单调时钟统计整个 `run()`。
- Token预算统计 Planner 输入状态、候选摘要、Scout 结果和覆盖评估输入的估算量。
- Planner生成的查询超过剩余查询预算时，按原顺序截断。
- 候选论文超过论文预算时，保留 Ranker 返回顺序靠前的论文。

## 停止门

每轮结束后按以下优先级选择唯一 `StopReason`：

1. `COVERAGE_SATISFIED`：`coverage.sufficient` 为真；
2. `ERROR`：没有任何可用候选且本轮所有已执行查询均失败；
3. `MAX_ROUNDS`：已完成最大轮数；
4. `QUERY_BUDGET`：查询预算耗尽；
5. `PAPER_BUDGET`：唯一论文预算耗尽；
6. `TOKEN_BUDGET`：估算 Token预算耗尽；
7. `TIME_BUDGET`：墙钟预算耗尽；
8. `NO_RESULTS`：连续两轮无新增唯一论文；
9. `LOW_MARGINAL_GAIN`：连续两轮新增唯一论文数低于阈值。

达到停止条件时不再调用 Planner。若首轮 Planner 没有生成可执行查询，结果以 `NO_RESULTS` 停止并记录说明。

## 并发与错误处理

- 使用 `asyncio.gather(..., return_exceptions=True)` 隔离数据源失败。
- 每个数据源调用使用 `asyncio.wait_for()` 限时。
- 未注册的 `target_source` 作为失败来源记录，不尝试猜测替代数据源。
- `failed_sources` 去重并保持首次出现顺序。
- `errors` 保存便于调试的短消息，不保存 API Key、完整响应或论文全文。
- 有部分来源成功时继续排序、Scout 和覆盖评估。
- Planner、Deduplicator、Ranker、Scout 或 Coverage 等核心编排 Tool 失败时，以 `ERROR` 返回已有候选和错误，不静默切换到 legacy 或 stub。

## Pipeline 接入

现有 `PipelineState` 仍只保存 `literature_results: list[LiteratureResult]`，M3仍消费 `KnowledgeEntry`。因此 Search Agent 不直接替换当前 M2节点。

接入边界分为两层：

1. `IterativeSearchAgent`：从每个 `sub_question` 产生 `SearchRunResult`；
2. `AgenticM2Adapter`：调用 Search Agent，再调用组员提供的 Reading Extraction Workflow，将 `EvidenceLinkedKnowledge` 统一转换为旧 `KnowledgeEntry` 和 `LiteratureResult`。

为避免 Adapter 依赖组员的私有类，公共协议增加：

```python
class ReadingExtractionWorkflowProtocol(ABC):
    async def run(
        self,
        sub_question: str,
        papers: Sequence[PaperRecord],
    ) -> list[PaperReadingResult]:
        ...
```

Adapter 根据每个 `PaperReadingResult.paper_id` 找回论文标题，并把其中的 `EvidenceLinkedKnowledge` 统一映射为现有 `KnowledgeEntry`。如果 Search Agent 或 Reading Workflow 发生不可恢复错误，Adapter抛出明确异常，交由现有 Pipeline 错误处理记录；不生成假知识条目。

`SearchConfig` 增加 `implementation: Literal["legacy", "agentic"] = "legacy"`。默认配置和现有测试继续运行 legacy。只有显式选择 agentic 且所需依赖已配置时才运行新路径；依赖缺失时给出明确错误，不回退假数据。

`PipelineConfig.get_module_kwargs("m2")` 将该值作为 M2构造参数的默认值，同时允许现有 `module_overrides.m2.kwargs` 显式覆盖。legacy M2入口只增加一个最小路由：`implementation="legacy"` 完整执行原逻辑；`implementation="agentic"` 委托给注入的 `AgenticM2Adapter`。没有注入 Adapter 时立即给出明确配置错误。这样配置开关真实生效，但不会要求当前公共分支提前猜测其他组员的具体实现类。

本阶段为 Adapter 提供依赖注入接口和离线假实现测试；真实 Reading Workflow 合入后，只替换注入对象，不改变 Search Agent。

## 文件边界

- `hypoforge/literature/search/agent.py`：异步主循环和依赖编排；
- `hypoforge/literature/search/budget.py`：预算计算、Token估算和停止门；
- `hypoforge/literature/models.py`：补充公共状态和结果字段；
- `hypoforge/literature/protocols.py`：Planner接收状态，并增加 Reading Workflow 公共协议；
- `hypoforge/literature/adapter.py`：agentic M2到现有 Pipeline 的适配边界；
- `hypoforge/config.py`：legacy/agentic 配置开关；
- `tests/literature/test_search_agent.py`：主循环、预算和失败降级；
- `tests/literature/test_adapter.py`：Pipeline输出转换和配置开关；
- `tests/literature/fakes.py`：离线协议假实现，不包含真实网络调用。

## 测试策略

所有生产行为先写失败测试，再写最小实现。测试覆盖：

- 首轮覆盖充分并停止；
- 缺口驱动第二轮查询；
- 跨来源和跨轮次去重；
- 查询预算截断；
- 最大轮数、论文、Token和时间预算；
- 连续无结果和连续低增益；
- 单一来源失败后降级继续；
- 所有来源失败；
- 未注册来源；
- 输出查询轨迹、来源计数、Scout结果、复用ID和最终状态；
- legacy为默认实现；
- Adapter将证据关联知识转换成现有 `LiteratureResult`；
- 完整现有测试套件无回归。

真实 API 测试不进入默认测试套件；测试 fixture 不包含密钥、PDF、全文XML、缓存和向量索引。

## 完成标准

- `IterativeSearchAgent` 能完全通过假 Tool 离线运行；
- 所有图片中列出的停止条件都有确定、可测试的语义；
- 一个数据源失败不会使其他来源结果丢失；
- `SearchRunResult` 可以完整解释查询、候选、覆盖、预算、失败和停止原因；
- legacy默认路径和现有 M1→M6测试保持通过；
- agentic适配边界能够使用假 Reading Workflow 完成 Pipeline格式转换；
- 没有修改真实数据源实现，没有提交大文件或秘密信息；
- 代码保持未提交状态，等待用户审阅。
