# HypoForge — 开发路线图

> 挑战杯 2026 · 赛题A：科学假设生成与研究计划设计
> 当前状态：**LLM 真实调用阶段 — M1–M6 端到端可跑；M3 含批量 LLM 语义边提取 + thinking 控制；M6 的 overall 维度改为计算值；Pipeline 支持 checkpoint/resume**
> 最后更新：2026-07-12

---

## 一、项目概览

HypoForge 是一个基于 LangGraph StateGraph 的六模块闭环 pipeline：

```
用户输入科学问题
       │
       ▼
  ┌──────────────┐
  │ M1: 问题理解   │ → 问题分解、子问题生成、领域识别
  └──────┬───────┘
         ▼
  ┌──────────────┐
  │ M2: 文献检索   │ → Semantic Scholar API / PubMed / 千问web search
  │   与知识提取   │ → 结构化抽取: Facts / Conflicts / Gaps / Entities
  └──────┬───────┘
         ▼
  ┌──────────────┐
  │ M3: 证据图谱   │ → Claim-Evidence-Source-Limitation-Conflict 图
  │   构建        │ → 输出: 事实库 + 争议点列表 + 知识空白列表
  └──────┬───────┘
         ▼
  ┌──────────────┐
  │ M4: 假设生成   │ → 多 Agent 协作:
  │   与筛选      │    Generator → Critic → Falsifiability Checker → Ranker
  └──────┬───────┘
         ▼
  ┌──────────────┐
  │ M5: 研究计划   │ → 实验对象/变量/对照/指标/分析方法/预期结果
  │   设计        │ → 含支持/反驳判别标准
  └──────┬───────┘
         ▼
  ┌──────────────┐
  │ M6: 评审与     │ → 三维 Specialist Reviewer（LLM）+ overall 计算
  │   迭代更新    │ → 反馈回 M4/M5，循环 ≤3 轮
  └──────────────┘
```

### 技术栈

| 组件 | 选择 | 说明 |
|------|------|------|
| 编排引擎 | LangGraph StateGraph | 与 BioDSA / open_deep_research 一致 |
| 状态管理 | Pydantic BaseModel | `PipelineState` 贯穿全流程 |
| 模块接口 | `ModuleProtocol` | 每个模块是一个 async callable |
| 注册机制 | `@ModuleRegistry.register` | 一行装饰器注册，零配置 |
| 配置系统 | YAML + 环境变量 | 6 套 baseline 配置已就绪 |
| LLM 调用 | 千问百炼 (OpenAI 兼容) | `tools/qwen_client.py` 封装 |
| 终端输出 | Rich (Panel/Rule/Table) | 借鉴 Flash-Searcher |
| 证据图谱 | NetworkX 兼容 JSON | 当前纯 Python dict/NetworkX |

---

## 二、已完成：骨架文件清单

```
HypoForge/
├── run_hypoforge.py                    ✅ CLI 入口 (argparse + asyncio)
├── README.md                           ✅ 项目说明 + CLI 参考 + Conda 部署
├── requirements.txt                    ✅ pip 依赖清单
├── environment.yml                     ✅ Conda 环境一键部署
├── .env_template                       ✅ API key 配置模板
├── TODO.md                             ✅ 开发路线图
├── configs/
│   ├── default.yaml                    ✅ 默认配置（全模块+迭代）
│   ├── full_pipeline.yaml              ✅ 完整系统
│   ├── baseline_b0.yaml                ✅ 千问直出（M1→M4→M5）
│   ├── baseline_b1.yaml                ✅ +web search（M1→M2→M4→M5）
│   ├── baseline_b2.yaml                ✅ +结构化提取（M1→M2→M3→M4→M5）
│   └── baseline_b3.yaml                ✅ +多Agent（无迭代 M6）
├── hypoforge/
│   ├── __init__.py                     ✅ 包入口，版本号 0.1.0
│   ├── protocol.py                     ✅ 4 个抽象接口 (Module/Tool/Skill/Metric)
│   ├── state.py                        ✅ 15+ Pydantic 数据模型
│   ├── config.py                       ✅ PipelineConfig + LLMConfig
│   ├── registry.py                     ✅ 3 个注册表 (装饰器模式)
│   ├── pipeline.py                     ✅ PipelineRunner + LangGraph 编排
│   ├── modules/
│   │   ├── __init__.py                 ✅
│   │   ├── m1_problem_understanding.py ✅ LLM 就绪（stub fallback）
│   │   ├── m2_literature_search.py     ✅ 已实现（真实搜索 + LLM 知识提取，stub fallback）
│   │   ├── m3_evidence_graph.py        ✅ 已实现（规则构建 + LLM 语义边增强，stub fallback）
│   │   ├── m4_hypothesis_generation.py ✅ LLM 就绪（Generator + Ranker）
│   │   ├── m5_research_plan.py         ✅ LLM 就绪（structured output）
│   │   └── m6_review_iteration.py      ✅ LLM 就绪（四维 Reviewer，stub fallback）
│   ├── prompts/
│   │   ├── __init__.py                 ✅
│   │   ├── m1_prompts.py               ✅ 问题分解 prompt
│   │   ├── m2_prompts.py               ✅ 六类知识提取 prompt
│   │   ├── m3_prompts.py               ✅ 证据图谱构建 prompt
│   │   ├── m4_prompts.py               ✅ Generator/Critic/Falsifiability/Ranker prompts
│   │   ├── m5_prompts.py               ✅ 研究计划生成 prompt
│   │   └── m6_prompts.py               ✅ 4 Reviewer + Meta-reviewer prompts
│   ├── display/
│   │   ├── __init__.py                 ✅ 全局 COLORS + Rich 导入
│   │   ├── panels.py                   ✅ 9 个渲染函数
│   │   └── progress.py                 ✅ spinner + progress_bar
│   ├── tools/
│   │   ├── __init__.py                 ✅
│   │   ├── semantic_scholar.py         ✅ 已实现（Semantic Scholar / OpenAlex 双后端自动切换）
│   │   ├── pubmed_search.py            ✅ 已实现（NCBI E-utilities esearch + efetch）
│   │   ├── qwen_web_search.py          ❌ 新建 (Task 1.2c, 选做)
│   │   └── qwen_client.py              ✅ 已完成（async chat + structured_chat + list_models）
│   ├── skills/
│   │   ├── __init__.py                 ✅
│   │   ├── base.py                     ⬜ LoggingSkill stub → 真实实现 (Task S.1)
│   │   ├── citation_formatter.py       ❌ 新建 (Task S.2)
│   │   ├── token_tracker.py            ❌ 新建 (Task S.3)
│   │   ├── result_archiver.py          ❌ 新建 (Task S.4)
│   │   └── output_translator.py        ❌ 新建 (Task S.5, 选做)
│   ├── memory/
│   │   ├── __init__.py                 ✅
│   │   ├── schema.py                   ✅ Entity/Relation/KnowledgeGraph 数据模型
│   │   ├── graph_manager.py            ✅ JSONL 持久化 + BM25 搜索 + CRUD
│   │   ├── bm25_index.py               ✅ BM25 全文检索索引
│   │   └── tools.py                    ✅ 知识图谱辅助工具
│   └── evaluation/
│       ├── __init__.py                 ✅
│       ├── metrics.py                  ✅ 4 个 Metric stub + MetricRegistry
│       └── scorer.py                   ✅ 评分 pipeline (sync + async)
└── tests/
    ├── __init__.py                     ✅
    └── test_pipeline.py                ✅ 4 个端到端测试 (全部通过)
```

### 验证状态

```
[PASS] Default pipeline test    — M1→M6 全流程 + 3 轮迭代
[PASS] B0 baseline test         — M1→M4→M5 最少模块
[PASS] Iteration convergence    — 迭代在 max_iterations 内终止
[PASS] Scoring test             — 评分报告生成正确

CLI 端到端:
  python run_hypoforge.py -q "蛋白质如何折叠及错误折叠导致疾病的机制？"
  输出: Rich 美化终端 + JSON 持久化 + 3 轮迭代分数递进
```

---

## 三、TODO：按 Phase 分配

### Phase 0.5 — 基础设施补完（本周，1-2 天）

> 负责人：架构/全员

- [x] **0.1** 在项目根目录添加 `requirements.txt`（已完成）和 `environment.yml`（Conda 一键重建）
  - 依赖: `langgraph`, `pydantic>=2`, `pyyaml`, `rich`, `langchain-openai`
- [x] **0.1b** 新建 `.env_template` 模板文件，供组员复制为 `.env` 填写 API key
- [x] **0.1c** 新建 `environment.yml`，支持 `conda env create -f environment.yml` 一键部署
- [ ] **0.2** 确认所有组员能跑通`python run_hypoforge.py -q "test"` 应无报错
- [x] **0.3** `.env` 管理 API key：复制 `.env_template` → `.env`，填入 `OPENAI_API_KEY` 和 `OPENAI_BASE_URL`
- [ ] **0.4** 确认 Semantic Scholar API 无需 key，PubMed E-utilities 注册 API key（可选）

---

### Phase 1 — 核心 Pipeline 真实实现（7/14 - 8/3，3 周）

> **提示**：`QwenClient` 已完整实现（`chat` + `structured_chat` + `list_models`）。
> `structured_chat()` 接受 JSON Schema 后自动注入 response_format + markdown fence 剥离，
> 可直接配合 Pydantic 模型的 `model_json_schema()` 使用。M1/M4/M5/M6 的 LLM 调用路径均
> 已在 stub 中预留（`mode="llm"` 时生效），替换时只需确认 prompt 和 schema 对齐即可。

#### 第 1 周（7/14-7/20）：M1 + M2 真实实现

> Team A（M2 文献检索 + 知识提取）+ Team C（M1 问题理解）

- [x] **1.1 M1 真实实现**
  - 文件：`hypoforge/modules/m1_problem_understanding.py`
  - 当前实现：`mode="llm"` 时调用 `QwenClient.structured_chat()` 进行问题分解，stub 作为 fallback
  - 使用 prompt：`hypoforge/prompts/m1_prompts.py`
  - 输入：`state.input_question`
  - 输出：`ProblemCard`（domain, sub_questions, key_entities, question_type）
  - 接口：`@ModuleRegistry.register`，`module_name = "m1"`

- [x] **1.2 M2 检索 pipeline 真实实现**
  - 当前实现：`mode="llm"` 时按 sub-question 调用搜索后端并聚合结果，stub 作为 fallback
  - 流程：对每个 sub_question 生成 query → 调用 search tools → 收集 top-N 论文
  - 每个子问题检索 10-20 篇文献

- [x] **1.2a Tool: Semantic Scholar 真实 API**
  - 文件：`hypoforge/tools/semantic_scholar.py`
  - 接口：`async def search(query, limit) -> list[dict]` + `async def fetch(id) -> dict`
  - 注册：已通过 `@ToolRegistry.register` 装饰
  - API：`https://api.semanticscholar.org/graph/v1/paper/search`（无需 key，限速 1req/s）
  - 返回标准化字段：title, abstract, year, doi, authors, journal, citationCount
  - 注意：Semantic Scholar 不返回完整 abstract，需额外调用 fetch 获取

- [x] **1.2b Tool: PubMed E-utilities 真实 API**
  - 文件：`hypoforge/tools/pubmed_search.py`
  - 接口：同上 `ToolProtocol`
  - API：`https://eutils.ncbi.nlm.nih.gov/entrez/eutils/`（建议注册 NCBI API key 提限速到 10req/s）
  - 流程：esearch → efetch（两步获取摘要）
  - 返回标准化字段：title, abstract, year, pmid, authors, journal

- [ ] **1.2c Tool: Qwen Web Search（选做）**
  - 文件：`hypoforge/tools/qwen_web_search.py`（新建）
  - 接口：同上 `ToolProtocol`，`@ToolRegistry.register`
  - 作为 Semantic Scholar / PubMed 的补充数据源
  - 如果百炼平台支持 web search plugin，直接走千问 channel

- [x] **1.3 M2 知识提取真实实现**
  - 文件：`hypoforge/modules/m2_literature_search.py`（同一个文件）
  - 当前实现：调用 Qwen 对论文摘要做六类知识提取，解析失败时回退到 stub 数据
  - 使用 prompt：`hypoforge/prompts/m2_prompts.py`
  - 输出：`List[LiteratureResult]`，每篇论文 → 多个 `KnowledgeEntry`

- [ ] **1.4 M2 中的 Qwen web search 集成**（选做）
  - 文件：`hypoforge/tools/qwen_client.py` — 如果千问支持 web search plugin
  - 作为 Semantic Scholar / PubMed 的补充数据源

- [ ] **1.5 B1 baseline 跑通**
  - `python run_hypoforge.py -c configs/baseline_b1.yaml -q "..."`

- [ ] **1.6 M2 优化（组员可分工）**
  - 文件：`hypoforge/modules/m2_literature_search.py`
  - 当前实现已可用（搜索 + 去重 + Qwen 批量知识提取），以下为增强方向：
    - **Query expansion**：用 `M2_SEARCH_QUERY_TEMPLATE` + Qwen 对每个 sub_question 生成 2-3 个变体查询（MeSH 术语、同义词），提高召回率
    - [x] **Batch extraction**：将多篇论文摘要合并到一次 LLM 调用中批量提取知识条目（已通过 `batch_size` 参数实现，默认 5 篇/批）
    - **Cross-paper relation detection**：提取完所有论文条目后，检测跨论文的 supports/contradicts 关系（或留给 M3 做）
    - **Citation-aware ranking**：加入语义相似度（query vs abstract embedding）作为排序因子，替代纯引用数排序
    - **Result caching**：基于 query hash 缓存搜索结果，避免重复 PubMed/OpenAlex API 调用
    - **Query type routing**：临床问题优先 PubMed，工程/CS 问题优先 OpenAlex

#### 第 2 周（7/21-7/27）：M3 + M4 真实实现

> Team A（M3 证据图谱）+ Team B（M4 多Agent假设生成）

- [x] **2.1 M3 批量语义边提取**（已完成）
  - 文件：`hypoforge/modules/m3_evidence_graph.py`
  - 当前实现：规则构建 + `mode="llm"` 时 Qwen 批量语义边增强（含跨批 bridge 任务）
  - QwenClient 已支持 `disable_thinking` 参数，防止推理 token 抢占输出预算
  - Prompt 已拆分为 edge-only batch prompt（`M3_BATCH_RELATION_SYSTEM_PROMPT`），杜绝模型生成冗余 nodes
  - 以下为增强方向：
    - **Incremental graph update**：改为增量更新——仅处理 M2 新产出的条目
    - **Entity linking**：用 UMLS / MeSH / GO 做实体归一化（"Hsp70" ↔ "HSPA1A"）
    - **Confidence-weighted edges**：Qwen 提取关系时同步输出置信度分数
    - **Graph visualisation export**：导出 Cytoscape.js / Mermaid 格式供前端 Demo
    - **Iterative graph refinement**：M6 反馈后回补或修正图中关系边

- [ ] **2.2 M4 Generator Agent**
  - 文件：`hypoforge/modules/m4_hypothesis_generation.py`
  - 输入：M3 的 knowledge_gaps + established_facts + conflicts
  - 使用 prompt：`hypoforge/prompts/m4_prompts.py` → `M4_GENERATOR_SYSTEM_PROMPT`
  - 调用 Qwen-Max → 生成 5-8 个候选假设
  - 输出：`List[HypothesisCard]`（每个含 statement + mechanism + predictions + falsification）

- [ ] **2.3 M4 Critic Agent**
  - 文件：同上
  - 输入：Generator 产生的候选假设 + established_facts
  - 使用 prompt：`M4_CRITIC_SYSTEM_PROMPT`
  - 调用 Qwen-Max → 对每个假设做逻辑/一致性检查
  - 输出：pass/fail + critique + issues

- [ ] **2.4 M4 Falsifiability Checker**
  - 文件：同上
  - 输入：通过 Critic 的假设
  - 使用 prompt：`M4_FALSIFIABILITY_SYSTEM_PROMPT`
  - 调用 Qwen-Plus → 检查可反驳性
  - 输出：is_falsifiable + specificity_score

- [ ] **2.5 M4 Ranker**
  - 文件：同上
  - 输入：通过所有检查的假设
  - 使用 prompt：`M4_RANKER_SYSTEM_PROMPT`
  - 调用 Qwen-Max → 四维评分 + 加权排序
  - 输出：top-K 假设（含 novelty/soundness/testability/consistency 分数）

#### 第 3 周（7/28-8/3）：M5 + M6 + 端到端联调

> Team C（M5 研究计划）+ Team B（M6 评审迭代）

- [x] **3.1 M5 研究计划生成**
  - 文件：`hypoforge/modules/m5_research_plan.py`
  - 输入：M4 的 top_hypotheses
  - 使用 prompt：`hypoforge/prompts/m5_prompts.py`
  - 调用 Qwen-Max → 生成结构化研究计划
  - 输出：`List[ResearchPlan]`（含全部 11 项要素）

- [x] **3.2 M6 三维 Specialist Reviewer + 计算 overall**
  - 文件：`hypoforge/modules/m6_review_iteration.py`
  - 三个 Specialist Reviewer 各用独立 Qwen 调用：
    - `scientific_logic` — 科学逻辑
    - `evidence_consistency` — 证据一致性
    - `method_feasibility` — 方法可行性
  - `overall` 维度改为 **计算值**（三个 specialist 分数平均），不再单独调用 LLM
  - 调用 Qwen-Plus（轻量评审任务）
  - 输出：`List[ReviewResult]`（含 score 1-5 + comments + suggestions）

- [ ] **3.3 M6 Meta-Reviewer 与迭代决策**
  - 文件：同上
  - 使用 prompt：`M6_SYNTHESIS_SYSTEM_PROMPT`
  - 综合四个 Reviewer 意见 → accept or revise
  - 迭代逻辑已在 `pipeline.py:_should_continue_iterating()` 中实现

- [ ] **3.4 端到端联调**
  - 在 1 个深挖问题上跑通 M1→M6 完整真实流程
  - 验证：证据图谱 + 候选假设 + 研究计划 + 评审意见 + 迭代改善

- [ ] **3.5 评估脚本 v0.1**
  - 文件：`hypoforge/evaluation/metrics.py` — 替换 stub
  - 文件：`hypoforge/evaluation/scorer.py` — 已就绪
  - 实现真实的新颖性/可验证性/证据一致性/计划完备性计算

---

#### Skill 开发（Phase 1 期间穿插，按需）

> Skill 是**中间件**，在模块执行前后自动介入。与 Tool 的区别：
> - **Tool** = 被模块调用的外部能力（数据源），如 PubMed API
> - **Skill** = 包裹模块的横切逻辑（中间件），如日志、格式化

- [ ] **S.1 LoggingSkill 真实实现**
  - 文件：`hypoforge/skills/base.py`（替换 stub）
  - 接口：`async def before(state) -> state` + `async def after(state) -> state`
  - 功能：记录每个模块的输入/输出字段、执行耗时、token 消耗
  - 输出：`output/{run_id}_execution_log.json`

- [ ] **S.2 CitationFormatter Skill**
  - 文件：`hypoforge/skills/citation_formatter.py`（新建）
  - 接口：同上 `SkillProtocol`，`@SkillRegistry.register`
  - 功能：在 M2/M5 执行后，统一将论文引用格式化为 GB/T 7714 标准
  - 可配置：`enabled_skills: ["citation_formatter"]`

- [ ] **S.3 TokenUsageTracker Skill**
  - 文件：`hypoforge/skills/token_tracker.py`（新建）
  - 功能：在每个模块执行后统计 LLM token 消耗
  - 输出：每次运行的 token 用量汇总表（便于控制成本）
  - 可配合 config 设置 token 预算上限

- [ ] **S.4 ResultArchiver Skill**
  - 文件：`hypoforge/skills/result_archiver.py`（新建）
  - 功能：每个模块执行后自动将中间产物存盘（JSON/YAML）
  - 目的：方便断点续跑 + 调试 + 消融实验对比

- [ ] **S.5 OutputTranslator Skill（选做）**
  - 文件：`hypoforge/skills/output_translator.py`（新建）
  - 功能：将英文输出翻译为中文（或反过来），适用于双语提交场景

---

### Phase 2 — 优化与深挖（8/4 - 8/24，3 周）

> 全员

- [ ] **4.1** 基于第一个 case 优化所有模块的 prompt
- [ ] **4.2** 在 5 个深挖问题上运行完整 pipeline
- [ ] **4.3** Human review：对假设和研究计划做专业评估
- [ ] **4.4** 跑 B2、B3 baseline 对比实验（config 已就绪，只需跑）
- [ ] **4.5** 生成评分矩阵：每个方法 × 每个问题的评分
- [ ] **4.6** 前端 Demo：Gradio 或 Streamlit（`hypoforge/` 下新建 `app.py`）
- [ ] **4.7** 证据图谱可视化：ECharts / D3.js 嵌入 Demo
- [ ] **4.8** 在第二梯队 5 个问题上跑系统，展示通用性

---

### Phase 3 — 文档与提交（8/25 - 9/4，10 天）

> 全员

- [ ] **5.1** 技术文档初稿（≤20 页 PDF）
- [ ] **5.2** 指导老师审阅 + 修改
- [ ] **5.3** 代码整理 + 文档化
- [ ] **5.4** 录制演示视频（3-5 分钟）
- [ ] **5.5** 最终检查 + 提交

---

## 四、组员开发指南

### 如何替换一个 Stub 模块

1. 打开对应文件（如 `hypoforge/modules/m2_literature_search.py`）
2. 保留类定义和装饰器 `@ModuleRegistry.register`
3. 替换 `__init__` 方法：接收真实的配置参数（LLM client、API keys 等）
4. 替换 `__call__` 方法：调用真实 API 而非返回 stub 数据
5. 使用 `hypoforge/prompts/` 中的 prompt 模板
6. 运行测试确认：`python tests/test_pipeline.py`

### 接口速查

| 你想做的事 | 继承/装饰 | 实现方法 | 注册到哪里 | 谁调用它 |
|-----------|----------|---------|-----------|---------|
| 写一个新模块 | `ModuleProtocol` | `async def __call__(state, config) -> dict` | `@ModuleRegistry.register` | Pipeline runner |
| 加一个搜索工具/外部API | `ToolProtocol` | `async def search(query, limit) -> list[dict]` + `fetch(id)` | `@ToolRegistry.register` | 模块内部调用 |
| 加一个中间件/横切逻辑 | `SkillProtocol` | `async def before/after(state) -> state` | `@SkillRegistry.register` | Pipeline runner 自动 |
| 加一个评估指标 | `MetricProtocol` | `async def compute(hypothesis, entries) -> float` | `@MetricRegistry.register` | Scorer 调用 |
| 调用千问 | `QwenClient` | `await client.chat(system, user)` | — | 模块内部调用 |

**Tool vs Skill 一句话**：Tool 是「数据从哪来」（外部 API），Skill 是「流程上额外做什么」（日志/格式化/翻译）。Tool 被模块调用，Skill 被 pipeline runner 自动调用。

### State 数据流

```
PipelineState 字段:
  input_question          ← CLI 输入
  problem_card            ← M1 写入
  literature_results      ← M2 写入
  evidence_graph          ← M3 写入
  candidate_hypotheses    ← M4 写入
  top_hypotheses          ← M4 写入
  research_plans          ← M5 写入
  reviews                 ← M6 追加
  iteration_count         ← M6 递增
  errors                  ← 任何模块追加
```

### 运行命令

```bash
# 方式一：从 environment.yml 一键创建（推荐）
conda env create -f environment.yml
conda activate hypoforge

# 方式二：手动创建
conda create -n hypoforge python=3.11 -y
conda activate hypoforge
pip install -r requirements.txt

# 配置 API key（参考 .env_template）
cp .env_template .env   # 然后编辑 .env 填入真实 key

# 运行
python run_hypoforge.py \
  --question "蛋白质如何折叠及错误折叠导致疾病的机制？" \
  --config configs/full_pipeline.yaml

# 快速测试（用 stub 数据）
python run_hypoforge.py \
  -q "test" -c configs/default.yaml --quiet

# 消融实验
python run_hypoforge.py \
  -q "..." -c configs/baseline_b0.yaml
python run_hypoforge.py \
  -q "..." -c configs/baseline_b3.yaml

# 运行测试
PYTHONIOENCODING=utf-8 python tests/test_pipeline.py
```

---

## 五、关键文件速查

| 需求 | 文件 |
|------|------|
| 修改 State 结构 | `hypoforge/state.py` |
| 修改配置项 | `hypoforge/config.py` + `configs/*.yaml` |
| 改 M1 prompt | `hypoforge/prompts/m1_prompts.py` |
| 改 M4 Generator 行为 | `hypoforge/prompts/m4_prompts.py` + `modules/m4_hypothesis_generation.py` |
| 改终端颜色 | `hypoforge/display/__init__.py` → `COLORS` 字典 |
| 加新的渲染面板 | `hypoforge/display/panels.py` |
| 改迭代逻辑（何时停止） | `hypoforge/pipeline.py` → `_should_continue_iterating()` |
| 加新的 baseline 配置 | `configs/` 下新建 yaml |
| 加新的搜索数据源 | `hypoforge/tools/` + `@ToolRegistry.register` |
| 加新的中间件/横切逻辑 | `hypoforge/skills/` + `@SkillRegistry.register` |
| 加新的评估指标 | `hypoforge/evaluation/metrics.py` |
| 改 Skill 启用列表 | `configs/*.yaml` → `enabled_skills` 字段 |
| 操作持久化知识图谱 | `hypoforge/memory/graph_manager.py` → `KnowledgeGraphManager` |
| 查看 KG 数据模型 | `hypoforge/memory/schema.py` |
