# M2 Agentic Literature Package

这个目录是新版 M2 的公共开发边界。当前分支只定义共享数据契约、Tool Protocol 和目录职责，不替换现有的 `hypoforge/modules/m2_literature_search.py`，也不包含任何真实 API 实现。

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

## 当前不在公共骨架中的内容

- Search Agent 编排实现；
- 任何真实数据库调用；
- legacy/agentic 配置切换；
- M2 到 M3 的适配器；
- 全文存储和向量数据库选型；
- 在线集成测试。

这些内容应在对应功能分支中实现并通过 PR 合入。
