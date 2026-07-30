# HypoForge — 开发路线图

> 挑战杯 2026 · 赛题A：科学假设生成与研究计划设计
> 当前状态：**LLM 全模块端到端可跑 + 4 条 feature branch 待移植**
> 最后更新：2026-07-21

---

## 一、项目概览

HypoForge 是一个基于 LangGraph StateGraph 的六模块闭环 AI Scientist pipeline：

```
用户输入科学问题
       │
       ▼
  ┌──────────────┐
  │ M1: 问题理解   │ → ProblemCard: domain, sub_questions, key_entities, question_type
  └──────┬───────┘
         ▼
  ┌──────────────┐
  │ M2: 文献检索   │ → PubMed / SemanticScholar(OpenAlex) / ArXiv 多源搜索
  │   与全文获取   │ → 全文获取 → 分块 → 可引用证据导出
  └──────┬───────┘   → 输出: LiteratureResult + M2KnowledgeExport
         ▼
  ┌──────────────┐
  │ M3: 证据图谱   │ → 消费 M2KnowledgeExport.evidence，不做联网下载
  │   构建        │ → 规则构建 + 原子声明合成 + 关系检测
  └──────┬───────┘   → 输出: EvidenceGraph + GroundingReport
         ▼
  ┌──────────────┐
  │ M4: 假设生成   │ → Multi-Agent: Generator → Critic → Falsifiability Checker → Ranker
  │   与筛选      │ → 四维评分: novelty / scientific_soundness / testability / evidence_consistency
  └──────┬───────┘
         ▼
  ┌──────────────┐
  │ M5: 研究计划   │ → 11 要素结构化研究计划
  │   设计        │
  └──────┬───────┘
         ▼
  ┌──────────────┐
  │ M6: 评审与     │ → 三维 Specialist Reviewer + overall 计算值
  │   迭代更新    │ → 反馈回 M4/M5，循环 ≤ max_iterations
  └──────────────┘
```

### 技术栈

| 组件 | 选择 | 说明 |
|------|------|------|
| 编排引擎 | LangGraph StateGraph | 基于 `PipelineState` 的增量状态合并 |
| 状态管理 | Pydantic BaseModel | `PipelineState` 贯穿全流程 |
| 模块接口 | `ModuleProtocol` | async callable，装饰器注册 |
| 注册机制 | `@ModuleRegistry.register` | 一行装饰器注册，热插拔 |
| 配置系统 | YAML + 环境变量 | 6 套 baseline 配置已就绪 |
| LLM 调用 | 千问百炼 (OpenAI 兼容) | 4 tier (base/max/plus/turbo) |
| 终端输出 | Rich (Panel/Rule/Table) | ~940 行渲染代码 |
| 证据图谱 | Pydantic + JSONL | 持久化知识图谱 + BM25 搜索 |
| 评分系统 | rubric.py 单一真相源 | reason-before-score + self_reported/independent 分离 |

---

## 二、当前状态：main 分支

M1-M6 全部 LLM 就绪，端到端可跑：

| 模块 | 文件 | 状态 | LLM Tier |
|------|------|------|----------|
| M1 | `modules/m1_problem_understanding.py` | ✅ structured_chat | `base` |
| M2 | `modules/m2_literature_search.py` | ✅ 双后端搜索 + 批量知识提取 | `turbo`(提取) `plus`(查词) |
| M3 | `modules/m3_evidence_graph.py` | ✅ 规则 + LLM 批量边提取 | `plus` |
| M4 | `modules/m4_hypothesis_generation.py` | ✅ Generator→Critic→Falsifiability→Ranker | `base`(生成) `plus`(排序) |
| M5 | `modules/m5_research_plan.py` | ✅ 11 要素 structured_chat | `plus` |
| M6 | `modules/m6_review_iteration.py` | ✅ 三维 Specialist + overall 计算值 | `plus` |

**附加系统：** scoring (rubric + scorer + TestabilityMetric) ✅ · 持久化 KG 写入 (JSONL) ✅ / 读取与 BM25 检索路径未接入主流程 ⚠️（详见 Track B.9-B.11） · Checkpoint/Resume ✅ · 6 套 config ✅

---

## 三、Feature Branches（待移植到 main）

> ⚠️ 所有分支基于旧 merge-base `88ad23d`（落后 main 9 个提交）。**必须从 main 开新分支做移植，不能直接 merge。**

| 分支 | 提交 | 改动 | 核心内容 |
|------|------|------|---------|
| `feat/m2-integrated-search-v2` | 20 | +20K lines, 97 files | M2 agentic 搜索 + 全文阅读管线 + 多源集成 + 知识导出 |
| `M3_EvidenceGraph` | 3 | +1.9K lines, 16 files | PaperQA 风格全文 evidence grounding + PDF/thinking 修复 |
| `feature/m3-evidence-gams` | 2 | +3.4K lines, 21 files | Evidence-GAMS 蒙特卡洛关系搜索（实验性） |
| `develop/m2-agentic` | 8 | +5.7K lines, 38 files | M2 核心代理框架（已被 v2 完全包含，忽略） |

### 移植时需解决的架构问题

**M2 和 M3 在当前设计中存在全文阅读职责重叠。** 两个分支各自实现了 PDF 下载→解析→分块→检索→证据抽取。如果直接合入，同一篇论文会被下载和解析两次，产生两套不兼容的 chunk/evidence 数据模型，且 M2 和 M3 的证据无法对齐。

**解决方案（本 TODO 采用）：明确 M2/M3 边界。**

```
M2 职责（Track A）：
  检索论文 → 获取全文 → 分块 → 提取 M2EvidenceExport（可引用文本片段）
  输出: LiteratureResult + M2KnowledgeExport（含 evidence + provenance）

M3 职责（Track B）：
  消费 M2KnowledgeExport.evidence → 原子声明合成 → 关系判断 → 证据图构建
  M3 不联网下载论文
```

---

## 四、模块化任务分配

### 核心理念

Phase 0 先冻结共享接口；之后按文件设置主负责人并行开发。Phase 0 与后续 Track 可能顺序编辑同一文件（例如 `skills/base.py`），但同一时间只允许一个负责人，避免并行写冲突。

> **注意**："零文件冲突"保证的是 git merge 不会产生 `<<<<<<<` 标记。**语义/接口冲突**（如 M2 输出格式变化影响 M3）仍需通过 code review 和集成测试发现。各 Track 在开发期间应保持沟通，特别是 Track A（M2 输出）和 Track B（M3 输入）之间的数据契约。

---

### Phase 0: 共享基础设施（协调员，P0，预计 2-3 天）

> **角色**：架构/协调员
> **文件**：`state.py`、`config.py`、`protocol.py`、`pipeline.py`、`registry.py`、`skills/base.py`、依赖文件及 Phase 0 contract tests
> **目标**：为所有 Track 准备好 canonical 数据模型、配置入口和 Skill hook。完成后合入 main，各 Track 从此分支。

#### Task 0.1: 冻结 M2KnowledgeExport canonical schema

- **文件**：`hypoforge/state.py`
- **原则**：直接采用 `origin/feat/m2-integrated-search-v2` 已由 exporter 和测试使用的契约，不再新增平行的 `EvidenceSpan` 模型。
- **必须同步的变更**：
  - `KnowledgeEntry` 新增 `evidence_ids: List[str]`，用于追溯具体证据。
  - `M2KnowledgeRun` 必须包含独立的 `evidence: List[M2EvidenceExport]` 字段。
  - `M2KnowledgeExport` 保留 `schema_version="m2-knowledge-export/v1"`。
  - 所有 list/dict 使用 `Field(default_factory=...)`，禁止在模型中使用 `=[]` 或 `={}`。

  ```python
  class KnowledgeEntry(BaseModel):
      # 保留现有字段
      evidence_ids: List[str] = Field(default_factory=list)

  class M2SearchQueryExport(BaseModel):
      query_id: str
      text: str
      round_index: int = 0
      intent: str = "core"
      target_source: str
      purpose: str
      target_gap: str = ""
      relation_to_question: str = ""

  class M2SearchProvenance(BaseModel):
      queries: List[M2SearchQueryExport] = Field(default_factory=list)
      coverage: M2CoverageExport = Field(default_factory=M2CoverageExport)
      source_result_counts: Dict[str, int] = Field(default_factory=dict)
      failed_sources: List[str] = Field(default_factory=list)
      iterations: int = 0
      stop_reason: Optional[str] = None
      errors: List[str] = Field(default_factory=list)
      stage_elapsed_seconds: Dict[str, float] = Field(default_factory=dict)
      papers_found: int = 0
      papers_after_dedup: int = 0

  class M2PaperExport(BaseModel):
      paper_id: str
      title: str
      abstract: str = ""
      authors: List[str] = Field(default_factory=list)
      year: Optional[int] = None
      journal: str = ""
      doi: str = ""
      pmid: str = ""
      pmcid: str = ""
      external_ids: Dict[str, str] = Field(default_factory=dict)
      citation_count: Optional[int] = None
      publication_type: str = ""
      sources: List[str] = Field(default_factory=list)
      is_open_access: Optional[bool] = None
      fulltext_status: str = "unknown"
      rank_scores: Dict[str, float] = Field(default_factory=dict)
      reading_summary: str = ""
      content_level: str = "metadata"
      document_id: str = ""
      document_source_uri: str = ""
      document_license: str = ""
      degraded_to_abstract: bool = False
      chunks_parsed: int = 0
      chunks_retrieved: int = 0
      stage_elapsed_seconds: Dict[str, float] = Field(default_factory=dict)
      errors: List[str] = Field(default_factory=list)

  class M2EvidenceExport(BaseModel):
      evidence_id: str
      paper_id: str
      chunk_id: str
      section: str = ""
      page: Optional[int] = None
      quote: str
      normalized_claim: str
      relevance_score: float
      citable: bool = True

  class M2KnowledgeRun(BaseModel):
      sub_question: str
      papers: List[M2PaperExport] = Field(default_factory=list)
      evidence: List[M2EvidenceExport] = Field(default_factory=list)
      knowledge_entries: List[KnowledgeEntry] = Field(default_factory=list)
      search_provenance: M2SearchProvenance = Field(default_factory=M2SearchProvenance)

  class M2KnowledgeExport(BaseModel):
      schema_version: Literal["m2-knowledge-export/v1"] = "m2-knowledge-export/v1"
      runs: List[M2KnowledgeRun] = Field(default_factory=list)
  ```

- **校验器**：移植源分支的 `M2KnowledgeRun.validate_provenance()`，检查 paper/evidence/knowledge ID 唯一性及跨引用有效性。
- **PipelineState 新增字段**：`m2_knowledge_export: Optional[M2KnowledgeExport] = None`
- **契约测试**：直接运行源分支 `build_m2_knowledge_export_run()` fixture，确认 `evidence` 和 `KnowledgeEntry.evidence_ids` 序列化后不会丢失。

#### Task 0.2: 冻结 M3 Grounding 输出契约 + 扩展 EvidenceEdge

- **文件**：`hypoforge/state.py`
- **原则**：保留 `feature/m3-evidence-gams` 的关系元数据，但将输入引用统一指向 Task 0.1 的 `M2EvidenceExport.evidence_id`。
- **新增模型**：

  ```python
  class AtomicClaim(BaseModel):
      """M3 从 M2EvidenceExport 合成的原子声明。"""
      id: str
      statement: str
      evidence_ids: List[str] = Field(default_factory=list)
      paper_ids: List[str] = Field(default_factory=list)
      entities: List[str] = Field(default_factory=list)
      confidence: float = Field(default=0.5, ge=0.0, le=1.0)

  class EvidenceRelation(BaseModel):
      id: str
      source: str
      target: str
      relation: Literal["supports", "contradicts", "extends", "limits", "same_as", "refines"]
      confidence: float = Field(default=0.5, ge=0.0, le=1.0)
      rationale: str = ""
      evidence_ids: List[str] = Field(default_factory=list)
      source_paper_ids: List[str] = Field(default_factory=list)
      target_paper_ids: List[str] = Field(default_factory=list)

  class GroundingReport(BaseModel):
      """M3 grounding 产出报告。evidence_records/relations 是数量字段。"""
      papers_total: int = 0
      full_text_papers: int = 0
      abstract_only_papers: int = 0
      fallback_papers: int = 0
      chunks_total: int = 0
      queries_total: int = 0
      evidence_records: int = 0
      claims_total: int = 0
      relations_total: int = 0
      relation_pairs_recalled: int = 0
      relation_candidates_judged: int = 0
      relation_candidates_selected: int = 0
      relation_selection_mode: str = "direct"
      warnings: List[str] = Field(default_factory=list)

  class GroundingResult(BaseModel):
      claims: List[AtomicClaim] = Field(default_factory=list)
      relations: List[EvidenceRelation] = Field(default_factory=list)
      report: GroundingReport = Field(default_factory=GroundingReport)
  ```

- **扩展 EvidenceEdge**（增强版关系需存储 confidence + rationale + evidence 引用）：
  ```python
  class EvidenceEdge(BaseModel):
      source: str
      target: str
      relation: EvidenceEdgeRelation
      # 新增字段（可选，M3 grounding 变体使用）:
      confidence: Optional[float] = None
      rationale: Optional[str] = None
      evidence_ids: List[str] = Field(default_factory=list)
  ```

- **PipelineState 新增字段**：`grounding_report: Optional[GroundingReport] = None`
- **验证**：构造 `GroundingResult`，确认 claim/relation/report 均可 JSON round-trip；EvidenceRelation 转换为 EvidenceEdge 后不丢 confidence/rationale/evidence_ids。

#### Task 0.3: 扩展 PipelineConfig，并只保留一个 M2 实现开关

- **文件**：`hypoforge/config.py`
- **新增配置类**：

  ```python
  class SearchConfig(BaseModel):
      implementation: Literal["legacy", "agentic"] = "legacy"
      tools: List[str] = Field(default_factory=lambda: ["semantic_scholar", "pubmed"])
      papers_per_sub_question: int = 15            # 保留 legacy 兼容字段
      max_papers_total: int = 80                   # 保留 legacy 兼容字段
      max_rounds: int = 3
      max_queries: int = 12
      max_papers: int = 100
      max_tokens: int = 100000
      max_seconds: int = 900

  class GroundingConfig(BaseModel):
      enabled: bool = False
      max_papers: int = 10
      max_evidence_items: int = 200
      max_claims: int = 100
      enable_gams: bool = False                    # 实验性开关，默认关闭
      gams_iterations: int = 100
      gams_min_confidence: float = 0.35
  ```

- **PipelineConfig 字段**：
  - 扩展现有 `search: SearchConfig`，使用 `search.implementation` 选择 legacy/agentic。
  - `grounding: GroundingConfig = Field(default_factory=GroundingConfig)`。
  - 不再新增 `use_agentic_m2`；避免 `use_agentic_m2`、`search.implementation`、`class_name` 三套开关相互冲突。
- **补齐现有配置 wiring**：legacy M2 也必须实际接收 `search.tools`、`papers_per_sub_question` 和 `max_papers_total`；新增测试证明 YAML 修改会改变 M2 行为，不能只完成 Pydantic 解析。
- **配置约束**：`grounding.enabled=true` 时要求 `search.implementation="agentic"`，否则在配置加载阶段给出明确错误。

- **验证**：加载全部 YAML config 不报错

#### Task 0.4: 扩展 PipelineRunner — Skill Hook 与 Agentic M2 接入

- **文件**：`hypoforge/pipeline.py`、`hypoforge/registry.py`

- **模块替换规则**：
  - `ModuleOverride.class_name` 保留为通用扩展点；它位于 `ModuleOverride` 顶层，不在 `kwargs` 中。
  - `search.implementation="agentic"` 是内置 Agentic M2 的唯一用户开关；Registry 将其映射到 Track A 提供的**配置驱动包装类** `AgenticM2Module`，而不是源分支中需要手工依赖注入的 `AgenticM2Adapter`。
  - `ModuleOverride.class_name` 仅用于高级自定义实现；设置后显式覆盖内置映射，不要求普通 agentic config 重复填写类路径。
  - `AgenticM2Module.__init__()` 接受普通配置（`llm_config`、`variant`、预算、超时等），内部调用 `build_integrated_search_adapter()` 或 `build_minimal_pubmed_adapter()`。
  - 动态加载后验证目标类是 `ModuleProtocol` 子类，并验证 `module_name` 与被替换模块一致；配置错误必须立即失败，禁止静默退回 legacy M2。

  ```yaml
  search:
    implementation: agentic
  module_overrides:
    m2:
      kwargs:
        variant: integrated
        final_k: 5
        enable_fulltext_reading: true
  ```

- **SkillRegistry 生命周期**：
  - Registry 保存 skill class；`PipelineRunner` 按本次 config 的 `enabled_skills` 创建实例。
  - 禁止把有状态 skill 实例缓存为进程级单例，避免多个 run 之间日志、token 差值和计时状态串扰。
  - 未知 skill 名称在构图阶段报错。

- **pipeline.py Skill hook 规则**：
  1. `before()` 返回的 patch 必须先合并成 `state_for_module`，模块实际接收该 state。
  2. `after()` 接收 `(module_name, state_before, result, state_after)`，其中 `state_before` 是已应用 before patch 的模块输入。
  3. Skill patch 只能更新 `PipelineState` 已声明字段；未知字段应报错或仅写 sidecar 文件，不能依赖 Pydantic 静默忽略。
  4. 默认禁止 Skill 覆盖模块自身的 output field；若确有需要，应显式声明覆盖权限。
  5. Skill 异常默认记录 warning 和 skill 名称；增加可选 `skill_fail_fast`，禁止无日志吞掉异常。

  ```python
  before_patch = await run_before_hooks(name, state)
  state_for_module = PipelineState.model_validate({
      **state.model_dump(mode="python"),
      **before_patch,
  })
  result = await mod(state_for_module)
  state_after = PipelineState.model_validate({
      **state_for_module.model_dump(mode="python"),
      **result,
  })
  after_patch = await run_after_hooks(name, state_for_module, result, state_after)
  validate_skill_patch(after_patch, protected_fields=mod.get_output_fields())
  return {**before_patch, **result, **after_patch}
  ```

- **验证**：增加测试确认 before patch 对模块可见、不同 PipelineRunner 不共享 skill 状态、skill 冲突/异常行为符合规则。

#### Task 0.5: 更新依赖

- **文件**：`requirements.txt`、`environment.yml`
- **单文件策略**：暂时只维护一个 `requirements.txt`，避免项目早期出现过多依赖入口。
  - 核心 Pipeline 运行依赖保持启用。
  - Agentic M2 / 全文阅读依赖 `arxiv`、`httpx`、`pypdf`、`tiktoken`、`rank-bm25` 暂时注释，Track A 合入时按实际使用情况启用。
  - Gradio 暂时注释，Track D 开始开发 Web Demo 时启用。
  - pytest 系列作为开发依赖保留在同一文件中并注释；开发者需要运行测试时自行取消注释或单独安装。
  - `environment.yml` 只安装核心依赖，可选项以注释说明，避免默认环境过重。
- **原则**：注释必须写明所属 Track 和启用时机；版本以对应分支测试通过的 API 为准，合入时锁定兼容范围。
- **验证**：核心环境必须可安装；各 Track 启用其注释依赖后，再执行对应测试和 smoke test。

---

### Track A: M2 Agentic 文献搜索（1-2 人，2 周）

> **源分支**：`origin/feat/m2-integrated-search-v2`
> **分支名**：`track/a-m2-agentic`（从 main 含 Phase 0 开出）
> **职责**：检索论文 → 获取全文 → 分块 → 提取 KnowledgeEntry + M2EvidenceExport
> **不负责**：原子声明合成、关系判断（留给 M3/Track B）

#### Track A 文件清单

```
hypoforge/literature/__init__.py           [NEW]
hypoforge/literature/protocols.py          [NEW] — LiteratureSourceProtocol 等 11 个协议
hypoforge/literature/models.py             [NEW] — SearchQuery, PaperRecord 等（使用源分支版本）
hypoforge/literature/adapter.py            [NEW] — AgenticM2Module（配置包装）+ 内部 AgenticM2Adapter
hypoforge/literature/export.py             [NEW] — build_m2_knowledge_export_run()
hypoforge/literature/integrated.py         [NEW] — build_integrated_search_adapter()
hypoforge/literature/minimal.py            [NEW] — build_minimal_pubmed_adapter()
hypoforge/literature/search/__init__.py    [NEW]
hypoforge/literature/search/agent.py       [NEW] — IterativeSearchAgent
hypoforge/literature/search/query_planner.py [NEW]
hypoforge/literature/search/scout.py       [NEW]
hypoforge/literature/search/coverage.py    [NEW]
hypoforge/literature/search/dedup.py       [NEW]
hypoforge/literature/search/ranking.py     [NEW]
hypoforge/literature/search/search_tool.py [NEW]
hypoforge/literature/search/budget.py      [NEW]
hypoforge/literature/search/_text.py       [NEW]
hypoforge/literature/sources/__init__.py   [NEW]
hypoforge/literature/sources/pubmed.py     [NEW]
hypoforge/literature/sources/pubmed_source.py [NEW]
hypoforge/literature/sources/academic_source.py [NEW]
hypoforge/literature/sources/arxiv_source.py [NEW]
hypoforge/literature/reading/__init__.py   [NEW]
hypoforge/literature/reading/workflow.py   [NEW] — FullTextReadingWorkflow（M2 全文阅读）
hypoforge/literature/reading/resolver.py   [NEW]
hypoforge/literature/reading/arxiv_resolver.py [NEW]
hypoforge/literature/reading/parser.py     [NEW]
hypoforge/literature/reading/reader.py     [NEW] — QwenPaperReader（输出 PaperReadingResult）
hypoforge/literature/reading/retriever.py  [NEW]
hypoforge/literature/reading/store.py      [NEW]
hypoforge/literature/reading/routing.py    [NEW]
hypoforge/tools/arxiv_search.py            [NEW]
configs/m2_agentic.yaml                    [NEW]
configs/m2_minimal_pubmed.yaml             [NEW]
tests/literature/                          [NEW]
```

#### Task A.1: 移植 Literature 协议和模型

- **文件**：`literature/__init__.py`、`protocols.py`、`models.py`
- **源**：`origin/feat/m2-integrated-search-v2`
- **操作**：从源分支复制，使用源分支的字段名（`text` 不是 `query_text`，`authors: List[str]` 不是 `str`，`search_provenance` 不是 `provenance`）。确认与 Phase 0 `state.py` 中的模型无冲突（literature/models.py 定义搜索域模型，state.py 定义管线级模型）。
- **验证**：`python -c "from hypoforge.literature import SearchQuery, PaperRecord; print('OK')"`

#### Task A.2: 移植文献源适配器

- **文件**：`literature/sources/` 下全部 5 个文件
- **实现**：每个 adapter 封装现有 `ToolProtocol` 实现（`PubMedTool` / `SemanticScholarTool`），转 `dict` → `PaperRecord`
- **异步要求**：现有 PubMed/OpenAlex 底层同步 HTTP 不得直接阻塞事件循环；使用 `httpx.AsyncClient` 或明确的 `asyncio.to_thread()` 边界，并设置连接/读取超时和并发上限
- **验证**：`python -m pytest tests/literature/test_arxiv_source.py tests/literature/test_pubmed_source*.py -q`

#### Task A.3: 移植迭代搜索代理

- **文件**：`literature/search/` 下全部 9 个文件
- **核心接口**：`IterativeSearchAgent.run(sub_question, entities, domains, question_type, budget) -> SearchRunResult`
- **验证**：Mock source 测试 agent 迭代/去重/排序/侦察/覆盖率停止逻辑

#### Task A.4: 移植全文阅读管线（M2 产出可引用证据）

- **文件**：`literature/reading/` 下全部 9 个文件
- **关键区别**：沿用源分支 `PaperReadingResult.evidence`，由 `export.py` 转换成 canonical `M2EvidenceExport`。M3 只消费导出，不接触 reader 内部模型。
- **子组件**：
  - `RoutingFulltextResolver` — 自动路由 PMC/ArXiv/出版商 PDF
  - `RoutingDocumentParser` — BioC XML / PDF 解析为章节感知块
  - `QwenPaperReader` — LLM 阅读检索块，产出带 quote/chunk/page/citable 的 evidence item
  - `InMemoryChunkStore` — 块存储与 BM25 索引
- **验证**：用已知 PMC 开放获取论文测试全文解析；确认导出的 `M2EvidenceExport.quote` 可回溯至 chunk，且 KnowledgeEntry.evidence_ids 有效

#### Task A.5: 新增 ArXiv 搜索工具

- **文件**：`hypoforge/tools/arxiv_search.py` (NEW)
- **接口**：`@ToolRegistry.register`，实现 `ToolProtocol`，返回标准化 `dict`
- **验证**：异步搜索返回 5 条结果

#### Task A.6: 构建配置驱动的 AgenticM2Module 和知识导出

- **文件**：`literature/adapter.py`、`export.py`
- **核心接口**：内部 `AgenticM2Adapter` 保留源分支的依赖注入设计；对 Pipeline 暴露可由 Registry 普通实例化的包装类：
  ```python
  class AgenticM2Module(ModuleProtocol):
      module_name = "m2"

      def __init__(self, llm_config, variant="integrated", **kwargs):
          self.adapter = build_adapter_from_config(
              llm_config=llm_config,
              variant=variant,
              **kwargs,
          )

      async def __call__(self, state, config=None) -> Dict[str, Any]:
          return await self.adapter(state, config)
  ```
- **导出结构**：`M2KnowledgeRun(papers=[...], evidence=[M2EvidenceExport(...)], knowledge_entries=[...], search_provenance=...)`。
- **关键**：`build_m2_knowledge_export_run()` 必须通过 Task 0.1 的 canonical contract test，禁止额外字段被 Pydantic 静默丢弃。
- **验证**：通过 `build_minimal_pubmed_adapter()` 运行，验证 LiteratureResult + M2KnowledgeExport 产出；随后通过 `AgenticM2Module` 和 Registry 再运行一次，覆盖真实构造路径。

#### Task A.7: 构建集成工厂和配置文件

- **文件**：`literature/integrated.py`、`minimal.py`、`configs/m2_agentic.yaml`、`configs/m2_minimal_pubmed.yaml`
- **配置**：使用 `search.implementation: agentic` 和 `module_overrides.m2.kwargs.variant`；普通用户不填写 `class_name`
- **兼容性**：`configs/default.yaml` 保持 legacy，确保新增包未安装时核心 Pipeline 仍可运行
- **验证**：`python run_hypoforge.py -q "test" -c configs/m2_minimal_pubmed.yaml` 不报错

#### Task A.8: 移植文献测试

- **文件**：`tests/literature/` 下全部（从源分支复制，修复 import 路径）
- **验证**：`python -m pytest tests/literature/ -q --timeout=60`

---

### Track B: M3 Evidence Grounding（1 人，1 周）

> **源分支**：`origin/M3_EvidenceGraph`（修复）+ `origin/feature/m3-evidence-gams`（GAMS 实验）
> **分支名**：`track/b-m3-grounding`（从 main 含 Phase 0 开出）
> **职责**：消费 M2KnowledgeExport.evidence → 原子声明合成 → 关系判断 → 构建证据图
> **不负责**：全文获取、分块（由 M2/Track A 完成）

#### Track B 文件清单

```
hypoforge/modules/m3_grounding/__init__.py      [NEW]
hypoforge/modules/m3_grounding/models.py        [NEW] — 本地模型（引用 state.py 的 canonical 模型）
hypoforge/modules/m3_grounding/workflow.py      [NEW] — GroundingWorkflow（消费 M2EvidenceExport）
hypoforge/modules/m3_grounding/evidence_gams.py [NEW] — 实验性 GAMS 搜索
hypoforge/modules/m3_grounding/relation_retrieval.py [NEW] — 关系候选检索
hypoforge/tools/qwen_client.py                  [EDIT] — thinking toggle 修复
hypoforge/modules/m3_evidence_graph.py          [EDIT] — 接入 grounding
configs/m3_grounding.yaml                       [NEW]
configs/m3_evidence_gams.yaml                   [NEW] — 实验性 GAMS 配置
tests/test_m3_fulltext_grounding.py             [NEW]
tests/test_evidence_gams.py                     [NEW]
tests/test_m3_relation_retrieval.py             [NEW]
tests/test_m3_gams_integration.py               [NEW]
```

> ⚠️ `m3_evidence_graph.py` 和 `qwen_client.py` 仅 Track B 编辑，其他 Track 不碰。

#### Task B.1: 移植 M3 Grounding 本地模型

- **文件**：`m3_grounding/__init__.py`、`models.py`
- **操作**：从源分支复制后，**删除与 Phase 0 state.py 重复的模型定义**，改为导入 `AtomicClaim`、`EvidenceRelation`、`GroundingReport`、`GroundingResult`。保留 M3 独有中间类型（如 `RelationCandidate`、`RelationPair`）。
- **验证**：`python -c "from hypoforge.modules.m3_grounding.models import RelationCandidate; print('OK')"`

#### Task B.2: 修复 QwenClient Thinking Toggle

- **文件**：`hypoforge/tools/qwen_client.py`
- **操作**：main 已包含 thinking toggle 和部分 JSON salvage。逐项比较源分支，只移植缺失用例；禁止用旧分支文件覆盖 main 当前的三层回退、token 统计和已有修复
- **验证**：`python -m pytest tests/test_qwen_client.py -q`

#### Task B.3: 实现 GroundingWorkflow（消费 M2EvidenceExport，不联网）

- **文件**：`m3_grounding/workflow.py`
- **核心接口**：
  ```python
  class GroundingWorkflow:
      async def run(
          self,
          m2_knowledge_export: M2KnowledgeExport,
          problem_card: ProblemCard,
          config: GroundingConfig,
      ) -> GroundingResult: ...
  ```
- **内部流程（7 步）**：
  1. **Collect Evidence**：从 `M2KnowledgeExport.runs[].evidence` 收集 citable evidence，并建立 paper/knowledge 引用映射
  2. **Index**：对 `quote + normalized_claim` 建立 BM25 索引
  3. **Query**：从子问题 + 知识空白生成子查询
  4. **Retrieve**：每个查询检索 top-k M2EvidenceExport
  5. **Synthesize Claims**：Qwen 从相关 evidence 合成 AtomicClaim（含 evidence_ids 引用）
  6. **Detect Relations**：Qwen 对 claim 对做关系分类（SUPPORTS/CONTRADICTS/EXTENDS/LIMITS），输出 confidence + rationale + evidence_ids
  7. **Merge**：将 AtomicClaim 转换为 EvidenceNode，关系转换为 EvidenceEdge（含 confidence/rationale/evidence_ids 到扩展字段），合并入 EvidenceGraph 的 buckets

- **关键**：不使用 `LiteratureResult`；M3 不读取全文路径、不重新解析 PDF，只消费 M2 已导出的 evidence/provenance。
- **降级规则**：若 `grounding.enabled=true` 但 `m2_knowledge_export` 缺失或 schema_version 不兼容，应产生明确配置/运行错误，不得返回一个看似成功的空 GroundingReport。
- **验证**：对小型 M2KnowledgeExport fixture 运行，验证 GroundingResult 中 claims/relations/report 完整且引用均有效

#### Task B.4: 实现关系候选检索

- **文件**：`m3_grounding/relation_retrieval.py`
- **接口**：`RelationCandidateRetriever.retrieve(claims, evidence, top_k) -> List[RelationPair]`
- **验证**：`python -m pytest tests/test_m3_relation_retrieval.py -q`

#### Task B.5: 实现 Evidence-GAMS（实验性，默认关闭）

- **文件**：`m3_grounding/evidence_gams.py`
- **状态**：实验性功能。GAMS 优化的是一套人工设计的奖励函数，不代表关系更接近科学事实。**必须通过实验对比验证后才可设为默认。**
- **对比实验要求**（在 Task B.8 测试中）：
  - 基线 A：简单 confidence 阈值（threshold=0.5）
  - 基线 B：规则去重 + 冲突消解
  - 实验 C：GAMS（gams_iterations=100）
  - 在人工标注关系集上对比 precision/recall
- **开关**：通过 `GroundingConfig.enable_gams = False`（默认）控制
- **接口**：`EvidenceGraphGAMS.search(candidates, iterations) -> List[EvidenceRelation]`
- **验证**：`python -m pytest tests/test_evidence_gams.py -q`

#### Task B.6: 将 Grounding 接入 M3 Evidence Graph 模块

- **文件**：`hypoforge/modules/m3_evidence_graph.py`
- **修改**：
  - `__init__` 新增：`grounding_config: Optional[GroundingConfig] = None`
  - `__call__` 新增：若 `grounding_config.enabled`，运行 `GroundingWorkflow.run(...) -> GroundingResult`，将 result.claims 和 result.relations 并入 `EvidenceGraph`
- **配置注入**：在 `ModuleRegistry.build_all()` 中，从 `config.grounding` 读取并传递给 M3 构造参数（在 Phase 0 Task 0.4 的 `build_all` 中实现）
- **输出**：`{"evidence_graph": ..., "grounding_report": grounding_result.report}`
- **验证**：使用 `configs/m3_grounding.yaml` 运行 pipeline

#### Task B.7: 创建 M3 配置文件

- **文件**：`configs/m3_grounding.yaml`、`configs/m3_evidence_gams.yaml`
- **配置约束**：两份配置都显式设置 `search.implementation: agentic`；GAMS 配置仅额外打开 `grounding.enable_gams`
- **验证**：`PipelineConfig.from_yaml()` 加载成功

#### Task B.8: 移植 M3 测试 + GAMS 对比实验

- **文件**：`tests/test_m3_*.py`（从源分支移植）
- **GAMS 对比测试**（人工标注关系集）：验证 precision/recall 对比 baselines
- **验证**：`python -m pytest tests/test_m3_*.py -q`

#### Task B.9: 持久化知识图谱的增量更新 + 读取接入

> 来源：`m3_evidence_graph.py:58` 类文档字符串内联 TODO（长期未被排期，本次补录）。
> **现状**：`M3EvidenceGraph._persist_graph()` 每次运行都会把当次构建的整张图整体覆盖写入
> `{memory_cache_dir}/memory-evidence_graph.jsonl`；`KnowledgeGraphManager.load_to_evidence_graph()` /
> `search_nodes()`（BM25 检索）已经实现，但**没有任何模块调用它们**——README 中
> "跨运行复用 + BM25 语义搜索" 的描述目前只是能力具备，未接通。

- **文件**：`hypoforge/modules/m3_evidence_graph.py`（唯一编辑者）
- **操作**：
  1. `__call__` 开头，当 `state.memory_cache_dir` 非空时，先调用
     `KnowledgeGraphManager.load_to_evidence_graph()` 加载历史图谱。
  2. 本次新构建的节点/边与历史图谱合并（按节点 `id` 去重；跨 run 的 `SRC_*`/`N_KE_*` id
     需保持稳定，可复用已有的 paper-id/entry-id 哈希方案，避免同一篇论文重复建节点）。
  3. `_persist_graph` 改为增量落盘（复用 `KnowledgeGraphManager.create_entities` /
     `create_relations` 的流式追加路径），不再是整图覆盖重写。
  4. 明确合并策略：新证据与历史证据冲突时如何处理（例如同一 claim 的置信度更新）——
     必须在实现前用一两句话写清楚规则，不能隐式覆盖。
- **配置开关**：新增 `M3EvidenceGraph.__init__(reuse_persisted_graph: bool = False, ...)`，
  默认关闭，避免默认行为突变影响现有 baseline 配置的可重复性。
- **验证**：连续两次以不同问题运行同一 `memory_cache_dir`，确认第二次运行的图中包含
  第一次的节点，且 `KnowledgeGraphManager.search_nodes()` 能检索到第一次运行写入的实体。

#### Task B.10: Entity Linking（实体归一化）

> 来源：`m3_evidence_graph.py:58` 内联 TODO。

- **文件**：`hypoforge/modules/m3_evidence_graph.py`
- **问题**：同一实体的不同写法（如 "Hsp70" / "HSP70" / "HSPA1A"）当前被视为不同节点，
  导致图谱碎片化、跨论文关系检测漏检。
- **实现范围（最小可行）**：不要求接入 UMLS/Gene Ontology 完整本体，先做基于字符串
  规范化 + 别名表（大小写/连字符归一化 + 手工维护的常见生物医学同义词表）的轻量归一化；
  完整本体链接作为后续可选项标注在代码注释中，不阻塞本任务验收。
- **验证**：构造含同一实体三种写法的 fixture，确认建图后只产生一个规范节点，
  三处引用都指向该节点。

#### Task B.11: Iterative Graph Refinement（M6 反馈回补）

> 来源：`m3_evidence_graph.py:58` 内联 TODO。

- **文件**：`hypoforge/modules/m3_evidence_graph.py`、`hypoforge/pipeline.py`（如需新增回边）
- **现状**：`EvidenceGraph` 在 M3 首次构建后保持不变，M6 的评审反馈只影响 M4/M5，
  不会回补或修正图中的关系边（例如评审指出某条 `supports` 边方向或强度有问题时，
  图谱本身不会被修正）。
- **实现范围**：定义 M3 的一个可选 `refine(graph, review_feedback) -> EvidenceGraph` 入口，
  在迭代循环（M6→M4）中，如果 reviewer 反馈明确指向某条边的问题，允许对该边打标记
  （例如 `EvidenceEdge.metadata` 记录 "disputed_by_reviewer"）或调整 `confidence`；
  不要求自动重新跑一遍 LLM 关系抽取。
- **依赖**：需要 Phase 0 `PipelineState`/`pipeline.py` 的迭代边保持不变，本任务只在
  M3 内部新增方法，不改变图的调用时序。
- **验证**：模拟一轮 M6 评审反馈含边修正建议，确认下一轮 `evidence_graph` 中对应边已更新。

---

### Track C: 评估与指标（1 人，1 周）

> **分支名**：`track/c-evaluation`
> **依赖**：Track A + Track B 完成后才能做完整的端到端指标测试。但指标代码本身可提前开发（用 fixture 测试）。

#### Track C 文件清单

```
hypoforge/evaluation/metrics.py    [EDIT] — 实现 NoveltyMetric + EvidenceConsistencyMetric
hypoforge/evaluation/scorer.py     [EDIT] — 传递 evidence_graph + knowledge_entries
configs/evaluation.yaml            [NEW]
scripts/run_ablation_matrix.py     [NEW]
```

#### Task C.1: 实现 NoveltyMetric（声明级 LLM 判断）

- **文件**：`hypoforge/evaluation/metrics.py`
- **当前状态**：`NoveltyMetric` 标记 `implemented=False`
- **问题**：简单的 "BM25 匹配少 = 新颖" 不可靠。匹配少可能意味着检索不完整或措辞不同。
- **正确流程**：
  1. 将假设拆成原子声明（call Qwen to decompose）
  2. 对每个声明，用 BM25 检索最相关的 knowledge_entries（top-5）
  3. LLM-as-judge 逐声明判断："Is claim X novel compared to these retrieved findings? Cite specific entries."
  4. 总分 = LLM 判定新颖的声明数 / 总声明数
- **接口**：`async def compute(hypothesis, knowledge_entries, llm_config=None, **kwargs) -> float`
- **验证**：先建立带人工标签的重复/组合创新/机制创新样本集，再据此校准阈值；不得仅凭手写两个样例宣称 `<0.3` / `>0.7` 有效

#### Task C.2: 实现 EvidenceConsistencyMetric（声明级 + 图谱引用）

- **文件**：`hypoforge/evaluation/metrics.py`
- **当前状态**：`EvidenceConsistencyMetric` 标记 `implemented=False`
- **问题**：不能直接遍历 CONTRADICTS 边——假设的声明尚未链接到图谱节点。
- **正确流程**：
  1. 将假设拆成原子声明
  2. 对每个声明，在 evidence_graph 中检索最相关节点（BM25 或其他方式）
  3. 遍历相关节点之间的 CONTRADICTS/SUPPORTS 边
  4. LLM-as-judge 逐声明判断："Does claim X conflict with evidence node Y? Output: consistent/conflicting/uncertain with citation."
  5. 总分 = LLM 判定一致的声明数 / 总声明数
- **接口**：`async def compute(hypothesis, knowledge_entries, evidence_graph=None, llm_config=None, **kwargs) -> float`
- **验证**：对已知矛盾假设做测试

#### Task C.3: 扩展 Scorer

- **文件**：`hypoforge/evaluation/scorer.py`
- **修改**：`score_pipeline_state_async()` 中传递 `evidence_graph` 和 `knowledge_entries` 给 metric
- **验证**：现有 scorer 测试仍通过

#### Task C.4: 构建消融矩阵脚本

- **文件**：`scripts/run_ablation_matrix.py` (NEW)
- **功能**：对 5 个问题 × 6 个 config 跑全矩阵
- **输出 CSV 列**：`config`, `question`, `top1_composite`, `mean_composite`, `mean_plan_completeness`, `latest_overall_review`, `iterations`, `errors`, `cost_usd`, `elapsed_seconds`, `failed_modules`
- **可靠性要求**：每个 (config, question) 组合至少重复 3 次（记录 mean±std），否则 LLM 随机生成差异可能大于模块改进
- **成本控制**：支持最大预算、最大运行数、resume、失败重试和中间结果增量落盘；优先缓存不受实验变量影响的检索/全文产物
- **可复现信息**：记录模型 ID、配置 hash、prompt hash、代码 commit、开始时间、token 用量和运行耗时
- **成本字段**：只有在配置了明确的模型单价表时才输出 `cost_usd`；否则输出 token 数，禁止伪造成本
- **验证**：运行脚本，验证 CSV 输出

#### Task C.5: 添加评估配置

- **文件**：`configs/evaluation.yaml` (NEW)
- **实现**：新增独立 `EvaluationConfig`/`AblationConfig`，由 `run_ablation_matrix.py` 加载。不要把 `evaluation:` 塞入 `PipelineConfig` 后依赖 Pydantic 静默忽略未知字段。
- **验证**：配置中未知字段、重复 config、空问题集和非法预算均产生明确校验错误

---

### Track D: Prompt 工程与质量优化（1 人，1 周）

> **分支名**：`track/d-prompts`
> **文件清单**：`hypoforge/prompts/m{1-6}_prompts.py` [EDIT] — 仅 Track D 编辑 prompts
> **前置门槛**：Track C 的固定评估集和基线报告已生成。Prompt 修改必须与冻结基线对比，不能只靠主观观察。

#### Task D.1-D.6: 逐模块优化 Prompt

| Task | 文件 | 主要改进 |
|------|------|---------|
| D.1 | `m1_prompts.py` | few-shot 示例 + 多尺度分解指导 |
| D.2 | `m2_prompts.py` | MeSH 术语指导 + 置信度校准 |
| D.3 | `m3_prompts.py` | 生物医学关系定义 + 证据引用要求 + confidence 字段 |
| D.4 | `m4_prompts.py` | 跨学科假设 + 隐性假设检查 + 优先 novelty |
| D.5 | `m5_prompts.py` | 具体仪器 + 样本量 + 时间线模板 |
| D.6 | `m6_prompts.py` | 可操作建议 + 评分校准 |

#### Task D.7: Prompt 回归测试

- **文件**：`scripts/test_prompts.py` (NEW)
- **功能**：验证模板变量可解析、结构化 schema 不漂移，并在固定 fixture 上做最小质量回归
- **验证**：除“不抛 KeyError”外，还应检查必需字段、引用 ID、候选数量及分数范围

---

### Track E: Skills 中间件（1 人，1 周）

> **分支名**：`track/e-skills`
> **依赖**：Phase 0 Task 0.4（Skill hook 在 pipeline 中）
> **文件清单**：`hypoforge/skills/` 下全部 — 仅 Track E 编辑

#### Task E.1-E.5: 实现 5 个 Skill

| Task | 文件 | skill_name | 功能 |
|------|------|-----------|------|
| E.1 | `skills/base.py` [EDIT] | `logging` | before/after 记录时间戳和字段变化 → `_execution_log.json` |
| E.2 | `skills/citation_formatter.py` [NEW] | `citation_formatter` | GB/T 7714 格式化 |
| E.3 | `skills/token_tracker.py` [NEW] | `token_tracker` | Token 用量追踪 → `_token_usage.json` |
| E.4 | `skills/result_archiver.py` [NEW] | `result_archiver` | 中间产物存档 |
| E.5 | `skills/output_translator.py` [NEW] | `output_translator` | 中英翻译（选做） |

- **数据边界**：
  - CitationFormatter 生成独立 citation/report 字段或 sidecar 文件，禁止把 GB/T 7714 字符串拼接进 `source_paper_title`。
  - OutputTranslator 只生成展示层翻译产物，不覆盖英文科学事实、引用原文或结构化主状态。
  - LoggingSkill 默认只记录字段是否填充、数量、耗时和错误摘要，不完整复制可能含敏感信息的大型 state。
  - TokenTracker 按模块记录累计计数差值，并增加多 run 隔离测试。

- **Skill 接口**（Phase 0 已定义）：
  ```python
  class SkillProtocol(ABC):
      skill_name: str
      async def before(self, module_name: str, state: PipelineState) -> Dict[str, Any]: ...
      async def after(self, module_name: str, state_before: PipelineState, result: Dict[str, Any], state_after: PipelineState) -> Dict[str, Any]: ...
  ```
- **验证**：启用 `enabled_skills: ["logging"]` 运行 pipeline，验证日志生成

---

### Track F: 测试与 QA（1 人，1 周）

> **分支名**：`track/f-testing`
> **依赖说明**：Track F 的测试文件不与其他 Track 冲突，但部分测试（F.4 集成测试、F.6 烟雾测试）**需要 Track A/B/C 合入后才能通过**。开发期间可用 fixture/mock 编写测试框架，待其他 Track 合入后再验证。

#### Track F 文件清单

```
tests/conftest.py                    [NEW]
tests/fixtures/                      [NEW]
tests/test_config_validation.py      [NEW]
tests/test_checkpoint_resume.py      [NEW]
tests/test_error_handling.py         [NEW]
tests/test_pipeline_integration.py   [NEW] — 依赖 Track A/B/C
scripts/smoke_pipeline.py            [EDIT]
```

#### Task F.1: 创建测试 Fixtures

- **文件**：`tests/__init__.py`、`conftest.py`、`fixtures/`
- **Fixtures**：`mock_pipeline_state`（完整 PipelineState）、`mock_literature_results`、`mock_evidence_graph`、`mock_hypothesis_cards`、`mock_research_plans`
- **验证**：`python -c "from tests.conftest import mock_pipeline_state; print('OK')"`

#### Task F.2: 配置验证测试

- **文件**：`tests/test_config_validation.py` (NEW)
- **验证**：`python -m pytest tests/test_config_validation.py -q`

#### Task F.3: 断点续跑测试

- **文件**：`tests/test_checkpoint_resume.py` (NEW)
- **验证**：`python -m pytest tests/test_checkpoint_resume.py -q`

#### Task F.4: 集成测试

- **文件**：`tests/test_pipeline_integration.py` (NEW)
- **测试说明**：
  - 使用 **B3 config**（含 M1-M5 + multi_agent M4 + 无 M6）验证全流程输出字段填充 — 不用 B0（B0 无 M6，无法测 iteration）
  - 使用 **full_pipeline config** 验证 iteration_count 递增（需要 M6）和 errors 列表
- **实现要求**：默认使用 fake LLM、fake literature source 和临时目录，CI 不依赖真实 API；另设显式标记的 live smoke tests
- **依赖**：Track A/B/C 合入后启用对应测试；依赖未满足前使用 `pytest.skip/xfail(strict=True)` 并写明解除条件，禁止向 main 合入无标记的失败测试
- **验证**：`python -m pytest tests/test_pipeline_integration.py -q --timeout=120`

#### Task F.5: 错误处理测试

- **文件**：`tests/test_error_handling.py` (NEW)
- **覆盖**：缺少 API key、搜索源部分超时、LLM 非法 JSON、Skill hook 异常、Agentic M2 配置错误、grounding 缺失 M2 export、下游模块是否应停止/降级
- **验证**：`python -m pytest tests/test_error_handling.py -q`

#### Task F.6: 扩展烟雾测试

- **文件**：`scripts/smoke_pipeline.py`
- **新增**：agentic M2 变体、M3 grounding 变体（默认关闭 GAMS）
- **依赖**：Track A/B 合入后，新增测试路径才能通过
- **验证**：`python scripts/smoke_pipeline.py` 退出码 0

---

### Track G: 前端 Demo（1 人，Phase 2，1 周）

> **分支名**：`track/g-frontend`
> **文件清单**：`hypoforge/app.py` [NEW]、`hypoforge/display/demo_utils.py` [NEW]

#### Task G.1: Gradio Web 界面

- **文件**：`hypoforge/app.py` (NEW)
- **功能**：问题输入 + config 选择 + 进度条 + Tab 页（Hypothesis Cards / Plans / Reviews / Evidence Graph Mermaid / Scores）+ JSON 下载
- **验证**：`python hypoforge/app.py` 启动正常

#### Task G.2: 证据图谱 Mermaid 可视化

- **文件**：`hypoforge/display/demo_utils.py` (NEW)
- **接口**：`evidence_graph_to_mermaid(graph, max_nodes=80) -> str`
- **验证**：在 Gradio 中渲染正确

---

### Track I: M2 检索质量与成本优化（1 人，3-5 天）

> **分支名**：`track/i-m2-search-quality`
> **来源**：`m2_literature_search.py:154` 类文档字符串内联 TODO，长期未被排期，本次补录。
> **依赖**：与 Track A（agentic M2）互斥——两者是同一个 `search.implementation` 开关下的
> 不同实现路径。本 Track 只优化 **legacy M2**（`search.implementation="legacy"`），
> 不影响 Track A 的 agentic 实现。若 Track A 合入后 legacy 路径被判定为不再维护，
> 本 Track 可降级为可选项或直接关闭。

#### Track I 文件清单

```
hypoforge/modules/m2_literature_search.py   [EDIT] — 唯一编辑者（与 Track A 不冲突，A 只加新文件）
hypoforge/prompts/m2_prompts.py             [EDIT] — 若涉及查询扩写 prompt，需与 Track D 协调排期先后
tests/test_m2_search_quality.py             [NEW]
```

#### Task I.1: Query Expansion（查询扩写）

- **文件**：`hypoforge/modules/m2_literature_search.py`
- **现状**：直接用 `sub_question` 原文搜索，未生成变体查询。
- **实现**：为每个 `sub_question` 调 Qwen 生成 2-3 个变体查询（MeSH 术语、同义词），
  合并多路检索结果后去重。
- **成本控制**：变体查询数量可配置（默认 0 = 关闭，向后兼容现有 baseline）。
- **验证**：对同一 sub_question 开启/关闭 query expansion 对比召回的论文数量变化。

#### Task I.2: Citation-aware Relevance（引用感知排序）

- **文件**：`hypoforge/modules/m2_literature_search.py`
- **现状**：仅按 `citation_count` 降序排列。
- **实现**：加入与查询的语义相似度（复用 `memory/bm25_index.py` 的 BM25 或轻量 embedding），
  与引用数做加权组合排序；权重可配置。
- **验证**：构造高引用但低相关 + 低引用但高相关的 fixture，确认排序结果符合权重预期。

#### Task I.3: Result Caching（检索结果缓存）

- **文件**：`hypoforge/modules/m2_literature_search.py`
- **现状**：相同查询会重复调用 PubMed / OpenAlex，两者均有速率限制，重复调用有失败风险。
- **实现**：按 query 文本 hash 做本地缓存（文件或 sqlite，落盘位置遵循现有 `output_dir` /
  `memory_cache_dir` 约定），带 TTL；测试/CI 环境默认关闭真实网络调用不受影响。
- **验证**：同一查询连续调用两次，第二次命中缓存不发起网络请求（mock 网络层验证）。

#### Task I.4: Query Type Routing（按问题类型路由检索源）

- **文件**：`hypoforge/modules/m2_literature_search.py`
- **现状**：无差别调用 PubMed + OpenAlex 两个后端。
- **实现**：依据 `problem_card.question_type` 或 `domain` 做简单路由（临床问题优先
  PubMed，工程/CS 问题优先 OpenAlex），路由规则需可配置、可关闭（默认双跑，保持向后兼容）。
- **验证**：分别构造临床类和工程类 `problem_card` fixture，确认路由结果符合预期且
  默认配置下行为不变（回归测试防止破坏现有 baseline）。

---

### Track H: 文档与提交（全员，Phase 3，2 周）

> **分支名**：`track/h-docs`
> **文件清单**：`docs/` 下全部 [NEW] + `README.md` [EDIT]

| Task | 文件 | 内容 |
|------|------|------|
| H.1 | `docs/technical_report.md` [NEW] | 技术报告（≤20 页 PDF）|
| H.2 | `docs/user_guide.md` [NEW] | 安装/配置/运行/输出/排错 |
| H.3 | `docs/demo_script.md` [NEW] | 演示视频脚本 |
| H.4 | `README.md` [EDIT] | 最终更新架构图/链接/状态 |

---

## 五、交付门槛（Quality Gates）

| Gate | 必须满足的条件 | 未通过时 |
|------|----------------|----------|
| Gate 0：契约 | 源分支 exporter fixture 可无损构造/序列化 M2KnowledgeExport；Skill before patch 对模块可见；全部 config 校验通过 | 不允许其他 Track 基于 Phase 0 开分支 |
| Gate 1：M2 | legacy 与 agentic M2 均通过；证据 quote 可追溯到 chunk；KnowledgeEntry.evidence_ids 全部有效；搜索超时可降级 | 不合入 Track A |
| Gate 2：M3 | M3 不发起全文下载；claim/edge 均引用有效 evidence ID；direct baseline 在人工样本上达到约定 precision | 不合入 Track B |
| Gate 2b：持久化 KG（B.9-B.11） | 增量更新/读取路径接入且有跨 run 测试证明可检索历史节点；entity linking 至少覆盖 fixture 场景；两者均可选关闭不影响默认 baseline | 不合入对应子任务，但不阻塞 Gate 2 主体 |
| Gate 3：评估 | 独立指标有人工标注校准集；消融脚本支持重复、resume、预算和版本记录 | 不开始 Prompt 优化结论和最终实验 |
| Gate 4：GAMS | 在同一候选集上显著优于 confidence threshold 与规则基线，并报告 precision/recall/成本 | 保持实验代码或不合入默认流程 |
| Gate 4b：M2 检索优化（Track I） | 每项优化默认关闭时行为与现有 baseline 完全一致（回归测试通过）；开启后有对比数据支撑（召回/排序/缓存命中率） | 保持默认关闭，不合入 baseline 配置 |
| Gate 5：发布 | 离线测试、live smoke、checkpoint/resume、文档安装流程均通过 | 不制作最终演示/提交包 |

---

## 六、时间线与合并顺序（修正版）

```
  [Phase 0] 冻结 M2/M3 schema、模块工厂、Skill 生命周期、单文件依赖清单
  [Track F] 同步编写 schema contract tests、config tests、Skill hook tests
  Gate 0: 源分支 exporter fixture 可无损通过 canonical M2KnowledgeExport

  [Track A] Tasks A.1-A.5  (模型+搜索代理+唯一全文阅读管线)
  [Track C] 先完成评估 fixture、配置 loader 和消融脚本骨架
  [Track F] Agentic M2 fake-source tests

  [Track A] Tasks A.6-A.8  (AgenticM2Module+factory+配置+完整测试)
  Gate 1: legacy/agentic 两条 M2 路径均通过；evidence/provenance 无损
  [Track B] 可基于冻结 fixture 开发 B.1-B.4

  [Track B] 完成 direct relation baseline、M3 接入和测试（GAMS 默认关闭）
  Gate 2: M3 全程不联网；所有 claim/edge 可追溯到 M2 evidence ID
  [Track B] Tasks B.9-B.11（持久化 KG 增量/读取接入、entity linking、迭代图修正，均默认关闭）
  [Track I] 可与 Track B 并行开发（M2 legacy 检索优化，默认关闭，不阻塞其他 Track）

  [Track C] 完成独立指标、冻结基线和第一轮消融
  [Track D] 基于冻结基线优化 prompts
  [Track E] 实现 logging/token/archive，翻译器保持选做

  全员: 集成测试、错误恢复、成本/延迟优化、人工评审
  [Track B] 单独进行 GAMS vs 简单基线对比；未通过 Gate 4 则不合入默认流程

  [Track G] 前端 Demo
  [Track H] 技术报告、用户指南、演示视频和最终 README
```

### 合并顺序

```
1. Phase 0 + Track F contract tests → main
2. Track A Agentic M2              → main
3. Track B M3 direct grounding     → main（GAMS 关闭）
3b. Track B.9-B.11 持久化 KG 增量/读取 + entity linking → main（默认关闭，随 Track B 主体或单独 PR）
4. Track C evaluation baseline     → main
4b. Track I M2 检索质量优化         → main（默认关闭，不早于 Track A 稳定，可晚至此步之后任意时点）
5. Track D prompt improvements     → main（必须附基线对比）
6. Track E observability skills    → main
7. Track F remaining integration tests 随对应实现 PR 一同合入
8. Track B GAMS experiment         → 仅在人工标注集显著优于简单基线时合入
9. Track G Demo                    → main
10. Track H docs                   → main
```

> **CI 规则**：main 必须始终保持绿色。未实现依赖的测试使用带解除条件的 skip/strict-xfail，禁止合入“预期直接失败”的测试。

---

## 七、文件冲突矩阵

| 文件路径 | 所属 Track | 备注 |
|----------|-----------|------|
| `hypoforge/state.py` | Phase 0 | 协调员独占 |
| `hypoforge/config.py` | Phase 0 | 协调员独占 |
| `hypoforge/protocol.py` | Phase 0 | SkillProtocol 契约 |
| `hypoforge/pipeline.py` | Phase 0 | 协调员独占 |
| `hypoforge/registry.py` | Phase 0 | 协调员独占 |
| `hypoforge/skills/base.py` | Phase 0 → Track E | Phase 0 先适配接口，Track E 后续实现 |
| `requirements.txt` + `environment.yml` | Phase 0 | 核心依赖启用、可选依赖注释并标注启用时机 |
| `hypoforge/literature/**` (30+ files) | Track A | |
| `hypoforge/tools/arxiv_search.py` | Track A | |
| `configs/m2_*.yaml` (2 files) | Track A | |
| `tests/literature/**` | Track A | |
| `hypoforge/modules/m3_grounding/**` (5 files) | Track B | |
| `hypoforge/modules/m3_evidence_graph.py` | Track B | ⚠️ 唯一编辑者（含 B.9-B.11：增量图更新/entity linking/迭代修正） |
| `hypoforge/tools/qwen_client.py` | Track B | ⚠️ 唯一编辑者 |
| `configs/m3_*.yaml` (2 files) | Track B | |
| `tests/test_m3_*.py` (4 files) | Track B | |
| `hypoforge/modules/m2_literature_search.py` | Track I | ⚠️ 唯一编辑者（仅 legacy 路径优化，不与 Track A 新增文件冲突） |
| `tests/test_m2_search_quality.py` | Track I | |
| `hypoforge/evaluation/metrics.py` | Track C | ⚠️ 唯一编辑者 |
| `hypoforge/evaluation/scorer.py` | Track C | ⚠️ 唯一编辑者 |
| `configs/evaluation.yaml` | Track C | |
| `scripts/run_ablation_matrix.py` | Track C | |
| `hypoforge/prompts/m{1-6}_prompts.py` (6 files) | Track D | ⚠️ 唯一编辑者 |
| `scripts/test_prompts.py` | Track D | |
| `hypoforge/skills/citation_formatter.py` | Track E | |
| `hypoforge/skills/token_tracker.py` | Track E | |
| `hypoforge/skills/result_archiver.py` | Track E | |
| `hypoforge/skills/output_translator.py` | Track E | |
| `tests/conftest.py` + `tests/fixtures/**` | Track F | |
| `tests/test_config_validation.py` | Track F | |
| `tests/test_checkpoint_resume.py` | Track F | |
| `tests/test_error_handling.py` | Track F | |
| `tests/test_pipeline_integration.py` | Track F | |
| `scripts/smoke_pipeline.py` | Track F | |
| `hypoforge/app.py` | Track G | |
| `hypoforge/display/demo_utils.py` | Track G | |
| `docs/*.md` (4 files) | Track H | |
| `README.md` | Track H | |

### 语义冲突风险提示

虽然文件层面零冲突，但以下接口变更需要跨 Track 沟通：

| 接口 | 生产者 | 消费者 | 风险 |
|------|--------|--------|------|
| `M2KnowledgeExport.runs[].evidence` | Track A | Track B | M2 输出格式变化影响 M3 解析 |
| `EvidenceEdge` 扩展字段 (confidence/rationale/evidence_ids) | Phase 0 | Track B + Track C | 下游消费者需适配新字段 |
| `SkillProtocol.before/after` 签名 | Phase 0 | Track E | 签名变更影响所有 Skill 实现 |
| `PipelineState.m2_knowledge_export` | Phase 0 | Track A + Track B | 字段存在性需要两边确认 |
| `search.implementation` → AgenticM2Module 映射 | Phase 0 | Track A | 包装类路径和构造参数必须稳定 |

**缓解措施**：Phase 0 完成后立即发布状态模型和接口签名的 changelog；Track A 和 Track B 只允许通过 canonical `M2EvidenceExport` schema 交换证据。

---

## 八、接口速查表

| 你想做的事 | 继承/装饰 | 实现方法 | 注册到哪里 |
|-----------|----------|---------|-----------|
| 写一个新模块 | `ModuleProtocol` | `async def __call__(state, config) -> dict` | `@ModuleRegistry.register` |
| 加一个搜索工具 | `ToolProtocol` | `async def search(query, limit) -> list[dict]` + `fetch(id)` | `@ToolRegistry.register` |
| 加一个中间件 | `SkillProtocol` | `async def before(name, state) -> patch_dict` + `async def after(name, state_before, result, state_after) -> patch_dict` | `@SkillRegistry.register` |
| 加一个评估指标 | `MetricProtocol` | `async def compute(hypothesis, entries, **kwargs) -> float` | `@MetricRegistry.register` |
| 加一个文献源适配器 | `LiteratureSourceProtocol` | `async def search(query, limit) -> List[PaperRecord]` | 无需注册 |
| 调用千问 | `QwenClient` | `await client.chat(system, user)` | — |

---

## 九、State 数据流（更新版）

```
input_question          ← CLI 输入
problem_card            ← M1 写入
literature_results      ← M2 写入 (原有兼容格式)
m2_knowledge_export     ← M2 写入 (增强导出，含 papers + evidence + provenance)
evidence_graph          ← M3 写入 (含扩展 EvidenceEdge)
grounding_report        ← M3 写入 (consumes m2_knowledge_export)
candidate_hypotheses    ← M4 写入
top_hypotheses          ← M4 写入
best_hypotheses         ← M4 写入 (跨迭代最优)
research_plans          ← M5 写入
reviews                 ← M6 追加
iteration_count         ← M6 递增
errors                  ← 任何模块追加
```

---

## 十、运行命令

```bash
conda env create -f environment.yml
conda activate hypoforge
cp .env_template .env   # 编辑填入 API key

# 可选能力
# 在 requirements.txt 中取消对应分组的注释后重新安装：
pip install -r requirements.txt  # Agentic M2、Gradio 或测试依赖按当前 Track 按需启用

# 运行
python run_hypoforge.py -q "蛋白质如何折叠及错误折叠导致疾病的机制？" -c configs/full_pipeline.yaml
python run_hypoforge.py -q "..." -c configs/baseline_b0.yaml   # 消融
python run_hypoforge.py -q "..." -c configs/m2_agentic.yaml    # Agentic M2
python run_hypoforge.py -q "..." -c configs/m3_grounding.yaml  # M3 Grounding

# 测试
python -m pytest tests -q

# 工具
python scripts/smoke_pipeline.py
python scripts/run_ablation_matrix.py
python hypoforge/app.py
```

---

## 十一、关键文件速查

| 需求 | 文件 |
|------|------|
| 修改 State 结构 | `hypoforge/state.py` |
| 修改配置项 | `hypoforge/config.py` + `configs/*.yaml` |
| 查看 canonical 状态模型 | Phase 0 → `hypoforge/state.py` |
| M2 搜索代理逻辑 | Track A → `hypoforge/literature/search/agent.py` |
| M2 全文阅读管线 | Track A → `hypoforge/literature/reading/workflow.py` |
| M2→M3 数据契约 | Phase 0 → `hypoforge/state.py` → `M2KnowledgeExport` / `M2EvidenceExport` |
| M3 Grounding 工作流 | Track B → `hypoforge/modules/m3_grounding/workflow.py` |
| M3 GAMS 实验 | Track B → `hypoforge/modules/m3_grounding/evidence_gams.py` |
| LLM 客户端修复 | Track B → `hypoforge/tools/qwen_client.py` |
| 评估指标实现 | Track C → `hypoforge/evaluation/metrics.py` |
| 改 Prompt | Track D → `hypoforge/prompts/m*_prompts.py` |
| 实现 Skill | Track E → `hypoforge/skills/` (注意新接口签名) |
| 编写测试 | Track F → `tests/` |
| 构建前端 | Track G → `hypoforge/app.py` |
| 写文档 | Track H → `docs/` |
| 改终端颜色 | `hypoforge/display/__init__.py` → `COLORS` |
| 改迭代逻辑 | `hypoforge/pipeline.py` → `_should_continue_iterating()` |
| 操作持久化 KG | `hypoforge/memory/graph_manager.py` |
| 持久化 KG 增量更新/读取接入 | Track B.9 → `hypoforge/modules/m3_evidence_graph.py`（当前只写不读，见 Gate 2b） |
| M3 实体归一化 / 迭代图修正 | Track B.10-B.11 → `hypoforge/modules/m3_evidence_graph.py` |
| M2 检索优化（query expansion/缓存/路由/引用排序） | Track I → `hypoforge/modules/m2_literature_search.py`（legacy 路径，见 Gate 4b） |

---

## 附录：分支开发惯例

```bash
# 从最新 main（含 Phase 0）开始
git checkout main && git pull origin main

# 按 Track 开分支
git checkout -b track/a-m2-agentic   # 或 track/b-m3-grounding 等

# 开发完成后
git fetch origin main && git rebase origin/main
git push -u origin track/<xxx>

# PR 合入后，其他 Track 更新
git checkout main && git pull origin main
git checkout track/<yyy> && git rebase main
```
