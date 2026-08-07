# Agentic M2（adapter.py）— 类函数伪代码

> 对应代码文件：`literature/adapter.py`（766 行）
> 相关类：`AgenticM2Module`（注册入口）→ `AgenticM2Adapter`（核心执行体）
> 工厂函数：`build_adapter_from_config`、`_build_or_explain`
>
> **这是 Web 界面实际运行的 M2**（`search.implementation: agentic` 时由
> ModuleRegistry 加载），区别于 legacy 的 `m2_literature_search.py`。
>
> **模块职责一句话**：对每个子问题做"迭代搜索 → 全文阅读提取证据 → 组装知识导出"，
> 输出 `literature_results` + `m2_knowledge_export`（后者是 M3 grounding 的输入契约）。

---

## 类结构与调用关系

```
ModuleRegistry 加载
   │
   ▼
AgenticM2Module（薄包装，adapter.py:647）
   ├─ __init__ ─→ build_adapter_from_config()（adapter.py:701）
   │                └─ variant="integrated" → build_integrated_search_adapter()
   │                     （integrated.py，组装全部组件）
   └─ __call__ ─→ AgenticM2Adapter.__call__（adapter.py:131，核心执行体）
                    ├─ 首轮：for 每个子问题 → search_agent.run → reading_workflow.run → export
                    ├─ 补搜轮：_run_supplement（缓存优先增量补搜）
                    └─ 辅助：_cache_lookup / _cache_hit_run / _gap_queries /
                             _supplement_search / _merge_increment / _persist_full_run
```

---

## 1. `AgenticM2Module` — 公开包装类（被 Registry 加载）

### 1.1 `__init__` — 构造（adapter.py:659）

```python
def __init__(self, llm_config=None, variant="integrated", **kwargs):
    # Registry 注入的编排参数直接丢掉（这层不管它们）：
    kwargs 弹出 "implementation"
    kwargs 弹出 "query_llm_config"
    supplement_paper_budget = kwargs 弹出 "supplement_paper_budget"（可空）

    # 核心动作：把组装工作全权交给工厂
    self.adapter = build_adapter_from_config(
        llm_config=llm_config,
        variant=variant,          # "integrated" = 完整版 / "minimal" = PubMed-only
        **kwargs,                 # 剩余参数（final_k、超时等）透传给工厂
    )

    if supplement_paper_budget 不为空:
        self.adapter.supplement_paper_budget = max(1, int(它))
```

**一句话**：它自己啥活都不干，只是工厂 + 参数搬运工。

### 1.2 `__call__` — 入口（adapter.py:680）

```python
async def __call__(self, state, config=None):
    return await self.adapter(state, config)   # 直接转给内部 adapter
```

### 1.3 `get_input_fields` / `get_output_fields`

```python
get_input_fields()  → ["problem_card"]              # 读什么
get_output_fields() → ["literature_results", "m2_knowledge_export"]  # 写什么
```

---

## 2. `build_adapter_from_config` — 工厂函数（adapter.py:701）

```python
def build_adapter_from_config(*, llm_config=None, variant="integrated", **kwargs):
    """按 variant 构建 AgenticM2Adapter。"""

    if variant == "integrated":          # 完整版：多源搜索 + 全文阅读
        if llm_config 是 None:
            报错: "variant='integrated' 必须有 llm_config"
        延迟导入避免循环依赖
        client = QwenClient.from_config(llm_config)
        return _build_or_explain(build_integrated_search_adapter, variant, client=client, **kwargs)

    if variant == "minimal":             # 轻量版：纯 PubMed 规则搜索，不需要 LLM
        延迟导入 minimal 模块
        return _build_or_explain(build_minimal_pubmed_adapter, variant, **kwargs)

    其他 → 报错: "未知 variant，只支持 'integrated' 或 'minimal'"
```

**一句话**：按 `variant` 选装配方案，把活交给 integrated.py / minimal.py 的工厂。

---

## 3. `AgenticM2Adapter` — 核心执行体（adapter.py:107）

### 3.1 `__init__` — 构造（adapter.py:118）

```python
def __init__(self, *, search_agent, reading_workflow, budget=None, supplement_paper_budget=6):
    self.search_agent = search_agent              # IterativeSearchAgent（迭代搜索代理）
    self.reading_workflow = reading_workflow      # FullTextReadingWorkflow（全文阅读）
    self.budget = budget                          # SearchBudget（搜索预算）
    self.supplement_paper_budget = max(1, int(supplement_paper_budget))  # 补搜预算
```

### 3.2 `__call__` — 总调度（adapter.py:131）★ 必看

```python
async def __call__(self, state, config=None):
    """Pipeline 只调用这一个函数。"""

    # ---- 判断：是不是补搜轮（M6 回跳）----
    open_gaps = 从 state 里挑出状态为 "open" 的证据缺口
    if open_gaps 非空 且 state.search_round > 0:
        return await self._run_supplement(state, open_gaps)   # 补搜轮 → 增量补搜

    # ---- 首轮：完整搜索流程 ----
    problem_card = state.problem_card
    sub_questions = card 的子问题（没有就用原始问题兜底）
    key_entities / domains / question_type = 从 card 取

    literature_results = []
    export_runs = []

    for sub_question in sub_questions:        # 循环：每个子问题
        # ① 迭代搜索代理（内部自带多轮迭代）
        search_result = await self.search_agent.run(
            sub_question, key_entities=..., domains=..., question_type=..., budget=self.budget)
        if search_result.stop_reason 是 ERROR:
            报错: "Agentic M2 搜索失败: <错误明细>"

        # ② 全文阅读工作流：对入选论文做 RAG 证据提取
        reading_results = await self.reading_workflow.run(
            sub_question, search_result.final_papers, search_context=search_result)

        # ③ 组装知识导出（papers + evidence + provenance）
        export_run = build_m2_knowledge_export_run(sub_question, search_result, reading_results)
        export_runs.append(export_run)
        literature_results.append(LiteratureResult(
            sub_question=sub_question,
            papers_retrieved=len(export_run.papers),
            knowledge_entries=list(export_run.knowledge_entries),
        ))

    # ④ 把论文写进持久化缓存（副作用，不影响返回结果，供以后补搜命中）
    if state 配置了 memory_cache_dir:
        self._persist_full_run(state, export_runs)

    return {
        literature_results: literature_results,
        m2_knowledge_export: M2KnowledgeExport(runs=export_runs),   # M3 的输入
    }
```

**一句话**：`for 子问题 → 搜索 → 阅读 → 导出`，补搜轮单独走 `_run_supplement`。

### 3.3 `_run_supplement` — 补搜执行者（adapter.py:209）★ 必看

```python
async def _run_supplement(self, state, open_gaps):
    """缓存优先的增量补搜：命中缓存不重搜，没命中才实时搜索，全程受预算限制。"""

    key_entities / domains / question_type = 从 problem_card 取（可能为空）

    # ---- 缓存：能用就用，用不了降级为实时搜索 ----
    store = None
    if state 配置了 memory_cache_dir:
        try: store = PaperStore(它)
        except OSError: 记 warning，store 保持 None（优雅降级）

    existing_runs = 已有的 M2KnowledgeExport 运行列表
    known_keys = 所有已见论文 key（来自旧 runs + search_ledger）
    issued_norms = 所有已发检索词的规范化集合

    merged_results = 深拷贝现有 literature_results（绝不清空！）
    new_runs = []          # 本轮的增量导出
    new_query_texts = []   # 新发出的检索词
    new_paper_keys = []    # 新见论文
    attempted_gap_ids = set()   # 处理过的缺口
    remaining_budget = supplement_paper_budget   # 本轮预算

    for gap in open_gaps:     # 循环：逐个缺口补

        sub_question = _resolve_sub_question(gap, 已有子问题)   # 缺口属于哪个子问题

        # ---- 第一步：先翻缓存 ----
        if store 非空:
            cached_papers, cache_queries = _cache_lookup(store, gap, known_keys)
            if cached_papers 非空:
                hit_keys = 每篇缓存论文的 key
                new_runs.append(_cache_hit_run(...))     # 只含论文元数据，不伪造知识
                _merge_increment(merged_results, sub_question, len(cached_papers), 无新条目)
                known_keys 更新
                记 "memory_hit" 事件（前端显示：缺口命中缓存 X 篇）
                store 记录 query→keys 映射
                continue   # 这个缺口完了

        # ---- 第二步：缓存没命中 → 实时搜索 ----
        attempted_gap_ids.add(gap.gap_id)
        fresh_queries = _gap_queries(gap, issued_norms)   # 缺口的候选检索词（去重）
        if 没有新检索词 或 预算用完:
            continue

        search_result = await _supplement_search(         # 单轮、受预算限制的搜索
            gap, sub_question, fresh_queries,
            key_entities=..., domains=..., question_type=..., paper_limit=remaining_budget)

        new_query_texts += fresh_queries
        new_papers = 从 search_result.final_papers 里挑出 key 没见过的
        if store 非空: store 记录 query / upsert 新论文
        if 没有新论文: continue

        # 对新论文做全文阅读提取（复用同一个 reading_workflow）
        reading_results = await self.reading_workflow.run(sub_question, new_papers, ...)
        export_run = build_m2_knowledge_export_run(sub_question, 过滤后的结果, reading_results)
        new_runs.append(export_run)
        _merge_increment(merged_results, sub_question, len(new_papers), export_run 的知识条目)
        remaining_budget -= len(new_papers)   # 扣预算

    # ---- 第三步：更新缺口状态 + 台账，全量返回 ----
    updated_gaps = []
    for gap in state.evidence_gaps:
        if gap 被处理过 且 状态是 "open":
            状态改为 "pending_grounding"    # 补完了，交给 M3 接地
        else:
            原样保留

    ledger = SearchLedger(旧检索词 + 新检索词, 旧论文key + 新论文key)

    return {
        literature_results: merged_results,                        # 旧 + 增量
        m2_knowledge_export: M2KnowledgeExport(runs=旧runs + 新runs),  # 追加新 run，不覆盖
        evidence_gaps: updated_gaps,                               # 状态更新
        search_ledger: ledger,                                     # 台账更新
    }
```

**一句话**：缓存优先、预算封顶、旧数据绝不清空、缺口状态推进到 `pending_grounding`。

---

## 4. 补搜辅助函数（被 `_run_supplement` 调用）

### 4.1 `_cache_lookup` — 翻缓存（adapter.py:390）

```python
def _cache_lookup(store, gap, known_keys):
    # 候选检索词 = 缺口的 suggested_queries + 缺口描述，去空
    candidate_queries = [缺口的建议检索词..., 缺口描述]

    hit_keys = []
    for query in candidate_queries:
        for key in store.lookup_query(query):      # 用检索词查缓存索引
            if key 没被 seen 过 且 不在 known_keys:   # 只收"没见过"的
                hit_keys.append(key)

    papers = store.lookup_by_keys(hit_keys) if hit_keys 非空 else []
    return papers, candidate_queries    # 返回缓存论文 + 用过的检索词
```

### 4.2 `_cache_hit_run` — 造缓存命中导出（adapter.py:413）

```python
def _cache_hit_run(sub_question, gap, cache_queries, cached_papers, state):
    # 把缓存元数据转成 M2PaperExport（只保留合法字段，坏数据跳过）
    papers = []
    for meta in cached_papers:
        papers.append(M2PaperExport(**过滤出合法字段))
        # 字段不合法 → debug 日志跳过，不崩溃

    # 构造"检索来源=缓存"的 provenance（记录用了哪些 query、命中了几篇）
    provenance = M2SearchProvenance(
        queries=[对每个 query 生成 M2SearchQueryExport(
            query_id=f"cache-{gap.gap_id}-{序号}",
            text=query, target_source="paper_store",
            purpose="cache lookup for evidence gap", ...)],
        iterations=0, stop_reason="cache_hit",
        papers_found=len(papers), papers_after_dedup=len(papers),
    )

    # 关键：knowledge_entries 为空 —— 缓存没全文证据，绝不伪造知识
    return M2KnowledgeRun(sub_question, papers, evidence=[], knowledge_entries=[], provenance)
```

### 4.3 `_gap_queries` — 缺口的候选检索词（adapter.py:472）

```python
def _gap_queries(gap, issued_norms):
    # 优先用缺口的 suggested_queries（最多取 3 个）
    candidates = gap.suggested_queries 去空 [:3]

    # 没有建议 → 规则派生（不调 LLM）：
    if candidates 为空:
        if 缺口描述非空: 加描述前 200 字符
        if 缺口有 canonical_entities: 加实体拼接串

    # 与已发过的检索词去重（规范化后比较）
    fresh = []
    for query in candidates:
        if 规范化后为空 或 已在 issued_norms 里: 跳过
        fresh.append(query)
    return fresh
```

### 4.4 `_supplement_search` — 单轮受限搜索（adapter.py:507）

```python
async def _supplement_search(gap, sub_question, query_texts, *, key_entities, domains, question_type, paper_limit):
    # ① 每个检索词 × 每个可用数据源，造出 SearchQuery 列表
    source_names = 从 search_agent 拿数据源名（pubmed/semantic_scholar/arxiv）
    queries = []
    for 每个检索词, for 每个源:
        queries.append(SearchQuery(text=检索词, target_source=源,
                                   purpose="supplement evidence gap", target_gap=缺口描述, ...))

    # ② 预算：只允许 1 轮、queries 数量上限、论文上限
    budget = SearchBudget(max_rounds=1, max_queries=len(queries), max_papers=paper_limit)

    # ③ 复用原搜索代理的组件，但换上"一次性规划器"（只服务这组 query，跑完即停）
    base = self.search_agent
    if base 是 IterativeSearchAgent:
        agent = IterativeSearchAgent(
            query_planner=_GapQueryPlanner(queries),   # 换掉规划器
            sources=原 sources, deduplicator=原去重器, ranker=原排序器,
            scout_reader=原快速阅读器, coverage_evaluator=原覆盖检查器,
            final_k=min(原 final_k, paper_limit), ...)
    else:
        agent = base   # 测试用的假对象直接跑

    result = await agent.run(sub_question, key_entities=..., budget=budget)
    if result.stop_reason 是 ERROR:
        报错: "补搜失败: <明细>"
    return result
```

### 4.5 `_resolve_sub_question` — 缺口归属判断（adapter.py:577）

```python
def _resolve_sub_question(gap, existing):
    # 缺口明确指定了 target_sub_question → 直接用
    if gap.target_sub_question 非空: return 它

    # 没有子问题列表 → 用缺口描述兜底
    if existing 为空: return 缺口描述 或 "supplement search"

    # 否则：算缺口描述与每个子问题的单词重叠数，选最像的
    best = existing[0], 分数 = -1
    for candidate in existing:
        分数 = |缺口描述的词 ∩ 候选子问题的词|
        if 更大: 更新 best
    return best
```

### 4.6 `_merge_increment` — 合并增量进结果（adapter.py:594）

```python
def _merge_increment(merged_results, sub_question, paper_count, entries):
    # 找到同名子问题的结果 → 累加论文数 + 追加不重复的知识条目（按 ID 去重）
    for result in merged_results:
        if result.sub_question == sub_question:
            result.papers_retrieved += paper_count
            新条目 = 过滤掉 ID 已在 result 里的
            result.knowledge_entries.extend(新条目)
            return
    # 找不到 → 新建一个 LiteratureResult 追加
    merged_results.append(LiteratureResult(sub_question, paper_count, entries))
```

### 4.7 `_persist_full_run` — 首轮后种缓存（adapter.py:618）

```python
def _persist_full_run(state, export_runs):
    # 尽力而为：缓存写失败绝不能拖垮主流程
    try:
        store = PaperStore(state.memory_cache_dir)
        for run in export_runs:
            keys = store.upsert_papers(run.papers, run_id, round)
            for query in run.search_provenance.queries:
                store.record_query(query.text, keys, run_id, round)
    except Exception:
        记 warning "缓存写入失败"
```

---

## 5. 辅助类

### 5.1 `_GapQueryPlanner` — 一次性规划器（adapter.py:77）

```python
class _GapQueryPlanner:
    """补搜专用：把预制的检索词原样发出去一次，之后返回空（不迭代）。"""

    def __init__(self, queries):
        self._queries = 预制检索词列表
        self._served = False          # 是否已服务过

    async def plan(self, sub_question, key_entities=(), domains=(), question_type="", state=None):
        if self._served:              # 已经服务过一次
            return []                 # 不再产生新检索词（避免无限迭代）
        self._served = True
        return list(self._queries)    # 一次性把检索词全给出
```

### 5.2 `_build_or_explain` — 报错美化（adapter.py:743）

```python
def _build_or_explain(factory, variant, **kwargs):
    # 调工厂，把"参数不认识"这种隐晦报错变成人话
    try:
        return factory(**kwargs)
    except TypeError as exc:
        if "unexpected keyword argument" in 报错文本:
            报错: "Agentic M2 收到了不支持的配置参数: <原文>。
                  检查 module_overrides.m2.kwargs —— legacy 的键
                  (mode/max_papers_per_query/batch_size) 和搜索预算键
                  (max_rounds/max_queries/max_papers) 都不收；
                  预算走顶层 search: 配置块。"
        raise
```

---

## 6. 与 legacy 的关键差异（快速对照）

| 维度 | legacy（m2_literature_search.py） | agentic（本文件） |
|------|-----------------------------------|-------------------|
| 搜索方式 | 一次性：query → 双后端 → 提取 | **迭代代理**：规划→搜索→去重→排序→Scout→覆盖检查，不足再迭代 |
| 数据源 | PubMed + OpenAlex | 三源（+arXiv） |
| 知识来源 | 直接抽摘要 | **全文 RAG 阅读**（获取全文→分块→检索→Qwen 读） |
| 输出 | 仅 `literature_results` | `literature_results` + **`m2_knowledge_export`**（M3 必需） |
| 补搜 | 缓存优先（PaperStore） | 缓存优先 + 一次性规划器（同思路更完善） |

---

*本文件仅为代码阅读笔记（伪代码），不参与程序执行，原代码未做任何改动。*
