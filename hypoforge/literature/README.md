# M2 Agentic Literature Package

这个目录是新版 M2 的公共开发边界。它定义共享数据契约、Tool Protocol、迭代式 Search Agent 和 Pipeline适配边界，但不包含任何真实 API 实现，也不改变现有 M2 的默认 legacy行为。

## 目录边界

```text
hypoforge/literature/
├── models.py       所有 Tool 共享的 Pydantic 数据契约
├── protocols.py    异步 Tool 接口
├── search/         查询规划、去重、排序、Scout Reading、覆盖评估和 Search Agent
├── sources/        PubMed、Semantic Scholar/OpenAlex、Europe PMC 等单一数据源
└── reading/        全文获取、文档解析、分块、RAG 和论文阅读
```

## 开发规则

1. 每个功能分支只实现一个边界清晰的 Tool 或一组不可分割的测试。
2. Tool 通过 `hypoforge.literature` 导入公共契约，不导入其他 Tool 的私有实现。
3. 公共模型或 Protocol 需要变化时，先提交独立的小 PR 到 `develop/m2-agentic`，其他成员同步后再继续开发。
4. 每个 Tool 必须提供离线测试和脱敏 fixture；真实 API 测试默认不运行。
5. 不向 Git 提交 `.env`、API Key、PDF、全文 XML、向量索引、知识库缓存或 Pipeline 输出。
6. 新版 M2 稳定前不删除旧 M2，也不修改其默认运行行为。

## 建议的功能分支

```text
feat/m2-query-planner
feat/m2-pubmed-source
feat/m2-academic-source
feat/m2-paper-dedup
feat/m2-paper-ranking
feat/m2-scout-reader
feat/m2-coverage
feat/m2-search-agent
feat/m2-fulltext-resolver
feat/m2-document-parser
feat/m2-evidence-retrieval
feat/m2-paper-reader
```

这些分支的 PR 目标均为 `develop/m2-agentic`，不是 `main`。

## 数据流

```text
M1 ProblemCard
  → Iterative Search Agent
  → SearchRunResult / Final-K PaperRecord
  → Reading Extraction Workflow
  → EvidenceChunk + EvidenceLinkedKnowledge
  → M3 适配层
  → 现有 EvidenceGraph
```

`EvidenceLinkedKnowledge` 是新版 M2 的证据可追溯输出；接入旧 Pipeline 时应通过单独的适配层转换为现有 `hypoforge.state.KnowledgeEntry`，避免各 Tool 自行维护不同转换逻辑。

## Search Agent 接入契约

`IterativeSearchAgent` 通过构造函数接收 Tool，不导入任何组员的私有实现：

```python
from hypoforge.literature import IterativeSearchAgent, SearchBudget

agent = IterativeSearchAgent(
    query_planner=query_planner,
    sources=[pubmed_source, academic_source],
    deduplicator=deduplicator,
    ranker=ranker,
    scout_reader=scout_reader,
    coverage_evaluator=coverage_evaluator,
)

result = await agent.run(
    sub_question="Does the proposed mechanism hold?",
    key_entities=["target protein"],
    domains=["molecular biology"],
    question_type="mechanism",
    budget=SearchBudget(max_rounds=3, max_queries=12),
)
```

Query Planner 首轮收到空的 `SearchState`；后续轮次收到已使用查询、已知术语、覆盖方向和缺口。每个 `SearchQuery.target_source` 必须与某个 `LiteratureSourceProtocol.source_name` 对应。单一数据源可以抛出异常，Agent会记录失败并继续使用其他来源；去重、排序、Scout或覆盖评估等核心 Tool 异常会以 `StopReason.ERROR` 结束，不会生成 stub结果。

查询预算按实际调度的 `SearchQuery` 计算，论文预算按去重后的唯一论文计算，时间使用单调时钟，Token预算使用可替换的稳定估算器。所有 Tool 必须保证相同输入下返回顺序稳定，方便离线测试和实验复现。

## Pipeline 开关和适配

配置默认值为：

```yaml
search:
  implementation: legacy
```

显式选择 `agentic` 时，现有 M2入口会委托给注入的 `AgenticM2Adapter`。如果没有提供 Adapter，会立即报告配置错误，不会静默回退 legacy或假数据。当前功能分支提供了可离线测试的 Adapter边界；等全文阅读组的实现合入后，需要由最终集成代码创建真实 `ReadingExtractionWorkflowProtocol` 实例并注入。

`AgenticM2Adapter` 本身没有注册到 `ModuleRegistry`，单纯导入它不会替换当前 `M2LiteratureSearch`。

## 仍由其他功能分支提供的内容

- 任何真实数据库调用；
- Query Planner、去重、排序、Scout Reading和覆盖评估的真实算法；
- Reading Extraction Workflow的真实实现及依赖工厂；
- 全文存储和向量数据库选型；
- 在线集成测试。

这些内容应在对应功能分支中实现并通过 PR 合入。
