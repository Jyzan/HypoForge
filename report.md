# Track A: M2 Agentic 文献搜索 — 移植报告

> 分支：`track/a-m2-agentic`
> 源分支：`feat/m2-integrated-search-v2`
> 依据：`TODO.md` Track A 任务清单

---

## 一、整体架构

按照 TODO.md Task A.6 的设计，`hypoforge/literature/adapter.py` 提供**两层结构**：

```
ModuleRegistry (search.implementation="agentic")
  │
  ▼
AgenticM2Module (配置驱动包装，ModuleProtocol)
  │  __init__(llm_config, variant="integrated", **kwargs)
  │  __call__() → self.adapter(state, config)
  │
  ▼
build_adapter_from_config()
  │  variant="integrated" → build_integrated_search_adapter()
  │  variant="minimal"   → build_minimal_pubmed_adapter()
  │
  ▼
AgenticM2Adapter (内部依赖注入类，ModuleProtocol)
  │  __init__(search_agent, reading_workflow, budget)
  │  __call__() → 执行搜索 + 阅读 + 知识导出
  │
  ▼
输出: LiteratureResult + M2KnowledgeExport
```

### 关键设计决策

| 类 | 定位 | 初始化方式 |
|---|---|---|
| `AgenticM2Adapter` | 内部类，依赖注入 | 传入预构建的 `search_agent`、`reading_workflow`、`budget` |
| `AgenticM2Module` | 公开类，配置驱动 | 传入 `llm_config` + `variant`，内部调用 `build_adapter_from_config()` |

这样 `integrated.py` / `minimal.py` 的工厂函数返回 `AgenticM2Adapter`（保持源分支设计），而 `ModuleRegistry` 通过 `dotted_path` 加载 `AgenticM2Module`（不需要手工依赖注入）。

---

## 二、文件变更清单

### 修改的文件（Track A 范围内）

| 文件 | 变更 |
|---|---|
| `hypoforge/literature/adapter.py` | 重构为两层：`AgenticM2Adapter` + `AgenticM2Module` + `build_adapter_from_config()` |
| `hypoforge/literature/integrated.py` | 返回类型改为 `AgenticM2Adapter` |
| `hypoforge/literature/minimal.py` | 返回类型改为 `AgenticM2Adapter` |
| `hypoforge/literature/__init__.py` | 导出 `AgenticM2Adapter` 和 `AgenticM2Module` |
| `tests/literature/test_adapter.py` | DI 测试用 `AgenticM2Adapter`，配置测试用 `AgenticM2Module` |
| `tests/literature/test_integrated_factory.py` | 去掉 M2LiteratureSearch 包装，直接用适配器 |
| `tests/literature/test_minimal_tools.py` | 同上 |
| `tests/literature/test_knowledge_export_models.py` | 修复 error message regex 匹配 |

### 新增的文件

| 文件 | 说明 |
|---|---|
| `configs/m2_agentic.yaml` | 完整 agentic M2 配置（`variant: integrated`），含 5 个 sub-question、25 篇论文、52 条知识条目验证通过 |
| `configs/m2_minimal_pubmed.yaml` | PubMed-only 最小配置（`variant: minimal`），无需 LLM |

### 未修改的文件（Phase 0 相关，Track A 未做修改）

| 文件 | 说明 |
|---|---|
| `hypoforge/state.py` | M2KnowledgeExport 模型（Phase 0 Task 0.1） |
| `hypoforge/config.py` | SearchConfig.implementation 开关（Phase 0 Task 0.3） |
| `hypoforge/registry.py` | dotted_path 加载逻辑（Phase 0 Task 0.4） |
| `hypoforge/pipeline.py` | Skill hook（Phase 0 Task 0.4） |

---

## 三、配置使用

### 完整 agentic 模式

```bash
python run_hypoforge.py \
  --question "Hippo信号通路如何调控器官大小？" \
  --config configs/m2_agentic.yaml
```

`configs/m2_agentic.yaml` 关键配置：

```yaml
search:
  implementation: agentic

module_overrides:
  m2:
    kwargs:
      variant: integrated   # 路由到 build_integrated_search_adapter()
      final_k: 5
      max_rounds: 3
```

### 最小 PubMed 模式（无 LLM）

```bash
python run_hypoforge.py \
  --question "Hippo信号通路如何调控器官大小？" \
  --config configs/m2_minimal_pubmed.yaml
```

```yaml
search:
  implementation: agentic

module_overrides:
  m2:
    kwargs:
      variant: minimal      # 路由到 build_minimal_pubmed_adapter()
      final_k: 5
```

---

## 四、Registry 集成路径

当 `config.search.implementation == "agentic"` 时，`ModuleRegistry.build_all()` 的流程：

1. `dotted_path = "hypoforge.literature.adapter.AgenticM2Module"`
2. `module_cls = _load_module_class(dotted_path, "m2")` → 获取 `AgenticM2Module` 类
3. 注入 `llm_config`（turbo tier）+ `query_llm_config`（plus tier）
4. `module_cls(**kwargs)` → `AgenticM2Module.__init__`
5. 内部调用 `build_adapter_from_config(llm_config=..., variant="integrated", **kwargs)`
6. 返回完整的 `AgenticM2Adapter`（含 `IterativeSearchAgent` + `FullTextReadingWorkflow`）

---

## 五、测试结果

```bash
python -m pytest tests/literature/ -q
# 244 passed in 7.23s
```

覆盖范围：

| 测试文件 | 数量 | 覆盖内容 |
|---|---|---|
| `test_adapter.py` | 7 | AgenticM2Adapter DI + AgenticM2Module 配置 |
| `test_models.py` | ~15 | PaperRecord, SearchQuery, SearchBudget 等 |
| `test_protocols.py` | ~5 | 协议接口合规 |
| `test_search_agent.py` | ~30 | 迭代搜索代理全流程 |
| `test_query_planner_source_coverage.py` | ~15 | QueryPlanner 多源覆盖 |
| `test_scout.py` | ~20 | ScoutReader 标题/摘要筛选 |
| `test_coverage.py` | ~15 | CoverageEvaluator 证据覆盖 |
| `test_dedup.py` | ~10 | 论文去重 |
| `test_ranking.py` | ~15 | 论文排序 |
| `test_budget.py` | ~10 | 预算跟踪 |
| `test_integrated_factory.py` | 8 | build_integrated_search_adapter |
| `test_minimal_tools.py` | 11 | build_minimal_pubmed_adapter |
| `test_knowledge_export_*.py` | ~20 | M2KnowledgeExport 序列化/校验 |
| `test_teammate_search_tools.py` | ~30 | 源适配器 + LiteratureSearchTool |
| `test_arxiv_*.py` | ~20 | ArXiv 源和解析器 |
| `test_reading/` | ~30 | 全文阅读管线（resolver/parser/reader/retriever） |

---

## 六、数据流

```
M1 ProblemCard
  │  sub_questions, key_entities, domain, question_type
  ▼
AgenticM2Module.__call__(state, config)
  │  委托给 self.adapter
  ▼
AgenticM2Adapter.__call__(state, config)
  │
  ├─► IterativeSearchAgent.run(sub_question, ...)
  │     │  每轮: QueryPlanner → Sources(PubMed/S2/ArXiv) → Dedup → Rank → Scout → Coverage
  │     └─► SearchRunResult (final_papers, coverage, provenance)
  │
  ├─► ReadingExtractionWorkflow.run(sub_question, papers)
  │     │  Resolver → Parser → Retriever → QwenPaperReader
  │     └─► PaperReadingResult (evidence, knowledge_entries)
  │
  ├─► build_m2_knowledge_export_run()
  │     └─► M2KnowledgeRun (papers + evidence + knowledge + provenance)
  │
  └─► 输出:
        literature_results: List[LiteratureResult]    (legacy 兼容)
        m2_knowledge_export: M2KnowledgeExport        (M3 消费)
```
