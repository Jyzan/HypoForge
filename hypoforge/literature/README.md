# M2 Agentic Literature Package

这个目录是新版 M2 的公共开发边界。它定义共享数据契约、Tool Protocol、迭代式 Search Agent 和 Pipeline 适配边界。下文两条显式 smoke path 可能调用真实外部服务：Minimal 路径只访问 PubMed，Integrated 路径调用 Qwen 并检索 PubMed、Semantic Scholar/OpenAlex 与 arXiv。默认配置和 legacy M2 的运行边界保持不变，也不会隐式发起真实 API 请求。

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

当前包提供可直接注入的 `PaperDeduplicator`、`PaperRanker`、`ScoutReader`
和 `CoverageEvaluator`。去重与排序是确定性实现；Scout 与覆盖评估接受兼容
`QwenClient.structured_chat()` 的 client，并在 client 缺失、部分响应或调用失败时
使用保守的离线规则降级。

Query Planner 首轮收到空的 `SearchState`；后续轮次收到已使用查询、已知术语、覆盖方向和缺口。每个 `SearchQuery.target_source` 必须与某个 `LiteratureSourceProtocol.source_name` 对应。单一数据源可以抛出异常，Agent会记录失败并继续使用其他来源；去重、排序、Scout或覆盖评估等核心 Tool 异常会以 `StopReason.ERROR` 结束，不会生成 stub结果。

查询预算按实际调度的 `SearchQuery` 计算，论文预算按去重后的唯一论文计算，时间使用单调时钟，Token预算使用可替换的稳定估算器。所有 Tool 必须保证相同输入下返回顺序稳定，方便离线测试和实验复现。

## Pipeline 开关和适配

配置默认值为：

```yaml
search:
  implementation: legacy
```

显式选择 `agentic` 时，现有 M2入口会委托给注入的 `AgenticM2Adapter`。如果没有提供 Adapter，会立即报告配置错误，不会静默回退 legacy或假数据。`build_integrated_search_adapter()` 已创建并注入真实的 `FullTextReadingWorkflow`；最小 PubMed 路径仍保留 `AbstractReadingWorkflow`，仅用于独立冒烟测试和备份。

`AgenticM2Adapter` 本身没有注册到 `ModuleRegistry`，单纯导入它不会替换当前 `M2LiteratureSearch`。

## 当前边界

- 全文使用 PMC Open Access BioC JSON 或入选 arXiv 论文的公开 PDF；不可用时明确降级到论文真实摘要；
- 当前 RAG 使用本地分区感知 BM25、实体/章节加权、MMR 去冗余和相邻块扩展，不依赖向量数据库；
- 付费墙 PDF、OCR、通用 PDF 解析和默认启用的在线集成测试暂不在本分支范围内。

## Minimal PubMed smoke-test path

This explicit development path searches only real PubMed metadata and uses
deterministic rules for the remaining M2 Tools. It does not call an LLM, read
PDFs, perform RAG, fabricate papers, or change the default legacy M2 path.

```powershell
python scripts/run_m2_pubmed.py `
  --question "Hippo YAP TAZ organ size mechanotransduction" `
  --limit 5
```

PubMed network/HTTP/parse failures produce an error and a non-zero exit code.
Zero matches produce an empty literature result. There is no synthetic
fallback.

## Integrated multi-source search smoke path

The primary integration runner combines the available source implementations:

- Qwen Query Planner plus PubMed, Semantic Scholar/OpenAlex, and arXiv sources;
- canonical paper deduplication and metadata-aware ranking;
- model-assisted Scout Reading and balanced coverage evaluation;
- PMC Open Access or selected arXiv PDF resolution, section/page-aware parsing,
  local hybrid retrieval, and evidence-constrained Qwen extraction.

The integrated path now uses `FullTextReadingWorkflow`. It requests legally
reusable PMC BioC full text for PubMed-linked records. For selected arXiv
records it downloads the hosted PDF only after Final-K selection, caches it,
and parses page-attributed chunks. Both routes fall back explicitly to the real
abstract when full text is unavailable and return no knowledge when neither
source is readable.
Every accepted knowledge entry references retrieved evidence IDs. Whole papers
remain in the ignored runtime cache and are never placed in the model context.
The minimal PubMed-only runner above keeps `AbstractReadingWorkflow` as an
explicit backup and is never selected as an implicit fallback.

```powershell
python scripts/run_m2_integrated.py `
  --question "Hippo YAP TAZ organ size mechanotransduction" `
  --model qwen3.6-plus `
  --limit 5 `
  --timeout 30
```

## arXiv operational boundary

arXiv is an independent planner target named `arxiv`, not an alias for
Semantic Scholar or OpenAlex. It is intended for recent preprints in computer
science, mathematics, physics, statistics, electrical engineering,
quantitative biology, quantitative finance, and economics. An arXiv result is
a preprint and must not be treated as peer-reviewed merely because its PDF is
publicly accessible.

```text
PubMed -> PMC BioC when available -> abstract fallback
arXiv -> metadata/abstract search -> Final-K cached PDF -> page-aware RAG
Semantic Scholar/OpenAlex -> metadata/abstract -> abstract fallback
```

Search candidates carry only metadata and abstracts. PDF download happens
inside the reading workflow, so only Final-K papers consume bandwidth and
storage. Failed searches and PDF downloads report the real error; they do not
return fabricated papers. The PDF document record deliberately leaves the
licence field empty unless a future source supplies a paper-specific licence.

## Cross-source coverage and timeout policy

On the first search round, the query planner guarantees at least one query for
every configured and available backend: PubMed, Semantic Scholar/OpenAlex, and
arXiv. This applies to biomedical questions too; later rounds remain driven by
the coverage gaps found by the Search Agent. A failed or rate-limited source is
reported explicitly while the other sources continue independently.

arXiv PDF retrieval is streamed with both a byte cap and a 180-second monotonic
deadline. The reading workflow also applies a separate 210-second per-paper
resolver timeout. If one paper exceeds that limit, only that paper is cancelled
and degraded to its real source abstract; sibling papers continue through
parsing, retrieval, and reading. The 600-second workflow timeout remains the
global last-resort guard and does not replace these paper-local limits.

## M2 knowledge export boundary

Agentic M2 returns both literature_results and m2_knowledge_export.
m2_knowledge_export/v1 contains Final-K paper metadata, reading diagnostics,
all retrieved evidence excerpts, extracted knowledge entries, and search
provenance. Every knowledge evidence_id resolves inside the same run. The
package excludes full papers, local cache paths, and graph construction.
Downstream modules own their own adaptation to this M2 contract.
