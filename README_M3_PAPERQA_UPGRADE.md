# HypoForge M3 PaperQA 风格全文证据落地升级

本文档面向后续接手 HypoForge 的开发者，说明本分支对 M3（Evidence Graph）所做的改造、设计取舍、运行配置、测试方法、已验证结果和已知边界。

> 本次业务逻辑改造仅发生在 M3。M1、M2、M4、M5、M6 的业务实现没有修改；`state.py`、依赖文件和配置文件的变化用于支持新的 M3 数据结构与运行路径。

## 1. 改造目标

原 M3 直接根据 M2 生成的 `KnowledgeEntry` 建图。M2 中的内容主要来自论文标题、元数据和摘要，因此图中的“证据”缺少可审计的全文来源和精确位置。

本次升级将 PaperQA / PaperQA2 的核心思想嵌入 M3 内部：

1. 把 M2 结果视为候选论文和检索种子，而不是最终全文证据。
2. 尝试从 PMC / OpenAlex 获取合法开放全文。
3. 解析全文并按章节、页码或字符区间分块。
4. 使用 BM25、可选 embedding 和 MMR 检索相关且不重复的片段。
5. 对候选片段生成 RCS（retrieval-contextual summary）。
6. 保留逐字原文摘录、定位信息、相关性、方法和局限性。
7. 将证据原子化为 `AtomicClaim`，再判断证据/Claim 关系。
8. 构建可从 Claim 追溯到 EvidenceRecord、Chunk 和 PaperSource 的证据图。
9. 把增强证据回写到 `literature_results`，让未修改的 M4 可以继续消费。

顶层 workflow 没有新增 M2.5：

```text
M1 -> M2 -> M3 -> M4 -> M5 -> M6
            |
            +-- FullTextEvidenceGrounding（M3 内部 LangGraph 子图）
```

## 2. 当前内部工作流

`FullTextEvidenceGrounding` 使用 LangGraph `StateGraph` 编排以下节点：

```text
resolve_sources
  -> acquire_fulltext
  -> parse_and_chunk
  -> plan_queries
  -> retrieve
  -> rcs
  -> atomize_claims
  -> judge_relations
  -> finalize
```

代码位置：

- `hypoforge/modules/m3_grounding/workflow.py`
- `hypoforge/modules/m3_grounding/models.py`
- `hypoforge/modules/m3_evidence_graph.py`

### 2.1 来源解析

M3 从 M2 的 `source_paper_id`、`source_paper_title` 和 KnowledgeEntry 内容构造 `PaperSource`。

当前识别：

- `PMID:<id>`
- `DOI:<doi>` 或以 `10.` 开头的 DOI
- OpenAlex URL / ID
- 标题回退

### 2.2 全文获取顺序

获取顺序如下：

1. 可选的本地 PDF / TXT / XML。
2. M3 缓存。
3. PubMed PMID -> PMC OA BioC XML。
4. DOI / OpenAlex ID -> OpenAlex `best_oa_location`。
5. 都失败时使用 `m2_seed_fallback`。

项目不会使用绕过版权限制的来源。无法合法获取全文时，会在 `GroundingReport` 和 `PaperSource.acquisition_status` 中明确标记回退，不把摘要或 M2 条目伪装成全文。

PMC / OpenAlex 获取逻辑包含以下防护：

- 拒绝 PMC 返回的 `[Error]`、HTML 和 `No result can be found` 页面。
- 读取旧缓存时再次检查错误页。
- PMC 失败后继续尝试 OpenAlex，而不是提前终止。

### 2.3 解析和分块

支持：

- PDF：`pypdf.PdfReader`
- PMC / BioC XML：保留 passage 的 section
- TXT：UTF-8 容错读取

`FullTextChunk` 保存：

- `paper_id`
- `section`
- `page`
- `start_char` / `end_char`
- `source_path`
- `text`

默认分块大小约 9000 字符，重叠 1000 字符，可通过 YAML 调整。

### 2.4 查询规划

检索查询来自：

- 原始问题；
- M1 `sub_questions`；
- M1 `key_entities` 组合出的机制、方法、冲突和局限性查询；
- M2 的 `knowledge_gap` 和 `conflicting_evidence` 条目。

当前查询规划是规则驱动的，并有 `max_queries` 上限。它不是独立的 LLM query expansion 模块。

### 2.5 检索与去冗余

默认后端：

```text
BM25 sparse retrieval + MMR diversity selection
```

实现额外排除了 `REF`、`REFERENCES`、`BIBLIOGRAPHY` 等参考文献段。这是联网测试中发现并修复的问题：参考文献标题会重复查询术语，容易获得虚高 BM25 分数，但不是论文自身的结果证据。

如设置 `embedding_model`，会通过当前 OpenAI-compatible 服务调用 `/embeddings`，采用 sparse + dense 混合分数。embedding 请求失败时自动回退 BM25 + MMR，并在 report 中记录 warning。

当前没有下载或调用本地 embedding 模型。

### 2.6 RCS 证据对象

LLM 模式下，RCS 输出包括：

- `summary`
- `excerpt`：必须能够在原 chunk 中逐字找到
- `relevance_score`：0–10，表示与 query 的相关性，不是真实性评分
- `claims`
- `entities`
- `methods`
- `limitations`
- `epistemic_status`
- `context`：population、intervention、outcome、section、page 等

模型返回的 `excerpt` 如果不能在原 chunk 中找到，会自动退回抽取式 excerpt，防止生成不存在的原文。

`mode="rule"` 时不调用 LLM，而使用抽取式 summary / excerpt 和检索分数生成 EvidenceRecord。

### 2.7 Claim 和关系判断

每个 EvidenceRecord 中的候选 Claim 会被拆成 `AtomicClaim`。

确定性关系：

```text
EvidenceRecord --SUPPORTS--> AtomicClaim
```

LLM 模式还会保守判断 Claim 间：

- `CONTRADICTS`
- `EXTENDS`
- `LIMITS`
- `SAME_AS`
- `REFINES`

Prompt 明确要求：只有 population、intervention、outcome、剂量、时间和方法上下文可比，且结论互不相容时，才标为 contradiction。方法差异通常应标为 `LIMITS`，而不是冲突。

关系保存 `confidence` 和 `rationale`，低于 0.65 的跨 Claim 关系不会进入图。

## 3. 数据模型

新增 Pydantic 模型位于 `hypoforge/modules/m3_grounding/models.py`。

### PaperSource

论文身份、标题、DOI、PMID、URL、全文路径、获取状态和 M2 seed。

### FullTextChunk

全文的一段可定位文本，保留论文、章节、页码和字符范围。

### EvidenceRecord

一个 query 与一个 chunk 之间的证据对象，包含 RCS、逐字摘录、相关性、Claim、实体、方法、局限性和上下文。

### AtomicClaim

单一、尽量不可再拆分的科学陈述，记录其 EvidenceRecord 来源和置信度。

### EvidenceRelation

关系对象，包含 source、target、关系类型、confidence 和 rationale。

### GroundingReport

一次 M3 grounding 的运行诊断：论文数、全文数、fallback 数、chunk 数、query 数、证据数、Claim 数、关系数、检索后端、warning 和逐论文状态。

## 4. 图结构

当前实际生成：

```text
PaperSource
    | CONTAINS
    v
FullTextChunk
    | GROUNDS
    v
EvidenceRecord
    | SUPPORTS
    v
AtomicClaim
```

Claim 间可以有 `CONTRADICTS / LIMITS / EXTENDS / SAME_AS / REFINES`。

`state.py` 新增：

- 节点类型：`CHUNK`、`EVIDENCE_RECORD`
- 边类型：`CONTAINS`、`GROUNDS`、`CITES`、`SAME_AS`、`REFINES`
- 边属性：`confidence`、`rationale`
- `EvidenceGraph.grounding_report`

### Entity 的当前状态

`EvidenceRecord.entities` 和 `AtomicClaim.entities` 已保存实体字符串，但当前尚未为它们创建独立 Entity 节点，也尚未生成：

```text
AtomicClaim --INVOLVES--> Entity
```

`EvidenceNodeType.ENTITY` 和 `INVOLVES` 已存在，后续可以在此基础上加入 UMLS / MeSH / UniProt 等实体归一化。

## 5. M3 与 M4 的兼容方式

M4 当前只会：

1. 从 `EvidenceGraph.established_facts / conflicts / knowledge_gaps` 取得 ID；
2. 使用这些 ID 在 `state.literature_results` 中查找正文。

因此 M3 在建图后同时返回：

```python
{
    "evidence_graph": graph,
    "literature_results": enriched_literature_results,
}
```

EvidenceRecord 和 AtomicClaim 会转换成兼容的 `KnowledgeEntry` 并追加到 `literature_results`。相关性至少为 5 的全文证据会进入 M4 可读取的 legacy bucket。

这个设计避免修改 M4，但要注意：`established_facts` 这个 legacy 名称并不完全等同于“已证明真理”；其中也承担了向 M4 传递已通过 M3 相关性阈值的全文证据的兼容职责。

## 6. 配置

完整配置示例位于 `configs/full_pipeline.yaml`：

```yaml
module_overrides:
  m3:
    kwargs:
      mode: "llm"
      llm_tier: "plus"
      local_paper_dir: "papers"
      cache_dir: ".hypoforge_cache/m3_grounding"
      enable_network_fulltext: true
      embedding_model: ""
      retrieve_k: 30
      evidence_k: 12
      min_relevance: 5.0
      grounding_max_concurrency: 4
```

主要参数：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `mode` | `rule` | `rule` 不调用 LLM；`llm` 使用 RCS 和 Claim 关系判断 |
| `enable_network_fulltext` | `false` | 是否访问 PMC / OpenAlex；完整配置显式设为 `true` |
| `local_paper_dir` | `papers` | 可选本地论文路径；目录不存在会直接跳过 |
| `cache_dir` | `.hypoforge_cache/m3_grounding` | 全文、RCS 和报告缓存 |
| `embedding_model` | 空 | 空表示 BM25 + MMR；非空调用 `/embeddings` |
| `chunk_chars` | 9000 | chunk 字符数 |
| `chunk_overlap_chars` | 1000 | chunk 重叠字符数 |
| `retrieve_k` | 30 | 每个 query 的初始候选数 |
| `evidence_k` | 12 | MMR 后送入 RCS 的最大候选数 |
| `min_relevance` | 5.0 | EvidenceRecord 最低相关性 |
| `max_queries` | 16 | 最大 query 数 |
| `max_sources` | 60 | 最大论文来源数 |
| `grounding_max_concurrency` | 4 | 全文和 RCS 并发数 |

M3 默认恢复为 `rule` 且 `enable_network_fulltext=false`，确保 `PipelineConfig.from_defaults()` 和原离线测试不会意外访问 API。真实完整配置需要显式打开联网和 LLM。

## 7. 缓存和输出

默认缓存：

```text
.hypoforge_cache/m3_grounding/
  papers/                 # PMC XML / OA PDF
  rcs/                    # 按 model + query + chunk hash 缓存的 EvidenceRecord
  last_grounding_report.json
```

测试配置使用独立目录：

```text
.hypoforge_cache/m3_network_smoke/
.hypoforge_cache/m3_network_integration/
```

`.hypoforge_cache/` 已加入 `.gitignore`。

## 8. 安装

已新增依赖：

```text
pypdf>=5.0
rank-bm25>=0.2.2
```

使用现有环境：

```powershell
conda activate hypoforge
cd C:\Users\30881\tzb_agent\workflow\HypoForge
pip install -r requirements.txt
```

当前开发环境中实际安装过：

- `pypdf 6.14.2`
- `rank-bm25 0.2.2`
- `numpy 2.4.6`（rank-bm25 依赖）

## 9. 测试方法

### 9.1 完整离线测试

```powershell
conda activate hypoforge
cd C:\Users\30881\tzb_agent\workflow\HypoForge
python -m pytest -q
```

本次开发实际结果：

```text
8 passed in 98.36s
```

最后的缓存错误页防护变更后，M3 专项测试再次运行：

```text
2 passed in 2.17s
```

### 9.2 M3 专项离线测试

```powershell
python -m pytest tests\test_m3_fulltext_grounding.py -q
```

覆盖：

- 本地全文匹配、解析、分块；
- CHUNK / EVIDENCE_RECORD / GROUNDS 图结构；
- M4-compatible `literature_results`；
- 无全文时明确 `m2_seed_fallback`。

测试中的本地 TXT 由 pytest 在临时目录创建，不要求开发者建立项目级 `papers/`。

### 9.3 真实 PubMed / PMC / OpenAlex 测试，不调用 Qwen

```powershell
python scripts\test_m3_network_smoke.py
```

此脚本：

- 不使用项目本地 `papers/`；
- 从 PubMed 搜索真实论文；
- 通过 PMC / OpenAlex 获取合法开放全文；
- 使用 rule 模式执行分块、BM25/MMR、抽取式 EvidenceRecord、Claim 和图构建；
- 断言 M4 能读取 `Exact excerpt`。

本次实际联网结果之一：

```json
{
  "pmid": "40541208",
  "full_text_papers": 1,
  "fallback_papers": 0,
  "chunks_total": 69,
  "evidence_records": 2,
  "claims_total": 2,
  "relations_total": 2,
  "retrieval_backend": "bm25+mmr"
}
```

### 9.4 真实全文 + Qwen RCS 测试

```powershell
python scripts\test_m3_network_smoke.py --llm
```

本次实际结果：

```json
{
  "full_text_papers": 1,
  "chunks_total": 69,
  "evidence_records": 1,
  "claims_total": 4,
  "relations_total": 8,
  "warnings": []
}
```

实际 RCS 相关性为 6/10，并成功提取：

- summary；
- 可在原 chunk 中找到的 exact excerpt；
- 4 条 Claim；
- entities / methods / limitations / epistemic status；
- M4 bucket 中的全文证据。

增加两个候选片段：

```powershell
python scripts\test_m3_network_smoke.py --llm --relations
```

### 9.5 README 风格 M1 -> M2 -> M3 -> M4 联网集成

轻量测试配置：`configs/m3_network_integration.yaml`。

```powershell
python run_hypoforge.py `
  -q "α-synuclein 错误折叠通过哪些机制促进帕金森病进展？现有证据的冲突和局限性是什么？" `
  -c configs\m3_network_integration.yaml `
  --run-id m3_network_live_test
```

该配置强制：

```yaml
local_paper_dir: "__no_local_papers__"
enable_network_fulltext: true
max_sources: 2
max_queries: 2
evidence_k: 2
```

本次开发中，这条完整 CLI 曾两次在 5 分钟限制内没有完成，主要表现为模型阶段长期等待。服务 `/models` 可访问，且 `qwen3.7-max` / `qwen3.7-plus` 均存在；单独 M3 RCS 测试可以成功。因此不能把完整真实 CLI 标记为已通过，后续应增加阶段级超时和日志后再复测。

## 10. 已验证内容与验收状态

| 功能 | 状态 | 证据 |
|---|---|---|
| 默认离线 workflow | 通过 | 完整 pytest 8 passed |
| PubMed 搜索 | 通过 | network smoke PMID 40541208 |
| PMC / OpenAlex 开放全文 | 通过 | `open_access_fulltext`，69 chunks |
| 错误 OA 页面识别 | 通过修复 | 下载和缓存读取双重验证 |
| XML / PDF / TXT 解析 | XML/TXT 已测试；PDF 单测依赖可用 | pypdf 已安装 |
| BM25 + MMR | 通过 | network smoke |
| 排除参考文献段 | 通过修复 | RCS 从 REF 0 分切换到 ABSTRACT 6 分 |
| Qwen RCS | 通过 | summary/excerpt/claims/context 已产出 |
| 原文 excerpt 校验 | 通过 | excerpt 必须存在于 chunk，否则 fallback |
| AtomicClaim | 通过 | 实际产出 4 claims |
| SUPPORTS 和跨 Claim 关系 | 通过 | report 8 relations |
| M3 -> M4 Evidence 传递 | 通过 | M4 bucket 包含 Exact excerpt |
| 完整真实 M1 -> M4 CLI | 未完成 | 两次超过 5 分钟，模型阶段需继续诊断 |

## 11. 已知边界和下一步

### 11.1 尚未实现一跳引文扩展

`CITES` 枚举已预留，但当前没有完成 PaperQA2 风格的：

```text
高分证据 -> 解析参考文献 -> 一跳获取新论文 -> 重新检索/RCS
```

后续应加入来源预算、OA 状态、去重和循环防护，避免引用扩展无限增长。

### 11.2 Entity 尚未独立图化

实体目前是字符串列表，没有：

- canonical ID；
- alias 合并；
- UMLS / MeSH / UniProt 链接；
- Claim -> Entity `INVOLVES` 边。

### 11.3 Query expansion 仍较基础

目前使用 M1/M2 信息规则扩展，不是 PaperQA2 完整的 agentic query planning。

### 11.4 Dense embedding 仅支持 API

当前 `embedding_model` 走 OpenAI-compatible `/embeddings`，没有本地 BGE-M3 adapter。空值时默认 BM25 + MMR。

### 11.5 完整 CLI 需要超时和阶段日志

当前 QwenClient 没有为每次 M1/M2/M3/M4 请求提供统一的可配置超时。完整 CLI 在外部调用慢时难以判断阶段。建议下一步：

- 给 QwenClient 添加 request timeout；
- 每个 LangGraph 子节点记录起止时间；
- 在 `GroundingReport` 增加每阶段耗时；
- 给 RCS 与 relation judge 添加最大总调用数。

### 11.6 持久图的边扩展属性

运行时 `EvidenceEdge` 有 `confidence` 和 `rationale`，但原有持久 KnowledgeGraph 的 Relation schema 只保存 from/to/type。跨进程持久化时这些属性尚未完整保留。

## 12. 改动文件

主要改动：

```text
.gitignore
README.md
README_M3_PAPERQA_UPGRADE.md
configs/full_pipeline.yaml
configs/m3_network_integration.yaml
environment.yml
requirements.txt
hypoforge/state.py
hypoforge/modules/m3_evidence_graph.py
hypoforge/modules/m3_grounding/__init__.py
hypoforge/modules/m3_grounding/models.py
hypoforge/modules/m3_grounding/workflow.py
scripts/test_m3_network_smoke.py
tests/test_m3_fulltext_grounding.py
```

`references/` 是本地研究资料，不属于本次代码 commit 范围，除非项目维护者明确决定提交这些 PDF。

## 13. 建议的协作顺序

建议下一位开发者按以下顺序继续：

1. 先运行离线 pytest，确认环境和基本 workflow。
2. 运行无 LLM 的 network smoke，确认 NCBI / PMC / OpenAlex。
3. 运行单次 `--llm`，确认 RCS endpoint。
4. 添加 QwenClient timeout 和阶段耗时日志。
5. 再跑 `m3_network_integration.yaml` 完整 M1 -> M4。
6. 实现独立 Entity 节点与规范化。
7. 实现受预算限制的一跳 citation traversal。
