# M2 全文阅读与证据提取工作流设计

## 1. 目标

用真实的 `FullTextReadingWorkflow` 替换集成路径中的
`AbstractReadingWorkflow` 占位实现，使 Search Agent 选出的 Final-K 论文能够经过
合法全文获取、章节化解析、问题相关证据检索和 Qwen 结构化提取，最终输出可追溯的
`PaperReadingResult`。

本功能只负责“读论文并提取证据”，不负责检索、去重、排序或覆盖评估。

## 2. 设计边界

### 2.1 本次实现包含

- 优先通过 NCBI BioC PMC API 获取 Open Access 全文；
- 支持使用 PMID 或 PMCID 请求开放全文；
- 全文不可用时降级到 `PaperRecord.abstract`；
- 将 BioC JSON 或本地摘要转换成带章节和稳定 ID 的 `DocumentChunk`；
- 使用问题感知的本地 RAG 检索相关片段；
- 使用 Qwen 从片段中提取六类结构化知识；
- 校验每条知识与真实证据片段的绑定关系；
- 将网络失败、解析失败、全文缺失和模型失败记录到单篇结果；
- 对全文、解析结果和检索结果做本地缓存；
- 保留 `AbstractReadingWorkflow`，但不再作为集成路径的默认实现。

### 2.2 本次不包含

- 绕过付费墙或登录限制；
- 从任意出版社网页抓取受限全文；
- 通用 PDF 下载、版面分析和 OCR；
- 补充材料解析；
- 外部向量数据库；
- embedding API 或额外的 LLM reranker；
- 删除 legacy M2 或最小 PubMed 备份实现。

## 3. 公共工作流

新增 `FullTextReadingWorkflow`，实现现有
`ReadingExtractionWorkflowProtocol`。它仍然接收一个子问题和 Final-K
`PaperRecord`，返回与输入论文一一对应、顺序稳定的
`list[PaperReadingResult]`。

工作流内部组合四个已有协议对应的组件：

1. `PMCFulltextResolver`：获取合法开放全文或构造摘要降级文档；
2. `BioCDocumentParser`：将 BioC JSON 解析成章节化片段；
3. `HybridEvidenceRetriever`：从当前论文片段中检索问题相关证据；
4. `QwenPaperReader`：生成论文摘要和证据绑定的结构化知识。

工作流负责并发、缓存、降级、错误隔离和结果顺序；各子组件保持独立可测试。

## 4. 数据流

```text
Final-K PaperRecord
    -> PMCFulltextResolver
        -> BioC PMC 全文
        -> 或摘要降级文档
        -> 或无可读内容
    -> BioCDocumentParser
        -> section-aware DocumentChunk
    -> HybridEvidenceRetriever
        -> ranked EvidenceChunk candidates
    -> QwenPaperReader
        -> summary
        -> evidence-linked knowledge entries
    -> PaperReadingResult
```

完整论文永远不直接进入 Qwen 上下文。Qwen 只接收检索后的少量证据片段。

## 5. PMC 全文解析

### 5.1 内容来源

`PMCFulltextResolver` 调用 NCBI BioC PMC Open Access API。请求优先级为：

1. `paper.pmcid`；
2. `paper.pmid`；
3. 如果两者都不存在，则跳过全文请求并考虑摘要降级。

只有 API 实际返回正文时才标记为全文成功。HTTP 404、空响应、API
错误对象、超时和非预期数据结构都不得伪装成全文成功。

### 5.2 摘要降级

当全文不可用但论文存在摘要时，Resolver 构造
`ContentLevel.ABSTRACT` 文档。最终结果必须设置：

```text
degraded_to_abstract = true
```

并在 `errors` 中保留简短、可诊断的全文失败原因。

当全文和摘要均不可用时，返回没有证据和知识条目的
`PaperReadingResult`，同时记录 `no readable content`，不得生成假内容。

### 5.3 缓存

下载的 BioC JSON 和摘要降级文档写入运行时缓存目录，而不是仓库：

```text
.cache/hypoforge/literature/documents/<safe-paper-id>/
```

缓存文件使用内容哈希和格式版本，损坏或格式不兼容时重新获取。缓存目录加入
`.gitignore`，不会提交论文全文。

## 6. 章节化切块

`BioCDocumentParser` 保留 BioC passage 的章节信息，并将正文规范化为
`DocumentChunk`。优先识别：

- abstract；
- introduction/background；
- methods/materials；
- results；
- discussion；
- limitations；
- conclusion；
- other。

切块规则：

- 先按原始 passage 和段落边界切分；
- 超长段落再按句子切分；
- 目标块长 600 至 1200 个字符；
- 相邻块保留约 120 个字符重叠；
- 不把不同章节强行合并；
- `chunk_id` 由文档 ID、章节和顺序稳定生成；
- 原文不得由模型改写后再作为证据保存。

## 7. RAG 策略

### 7.1 多意图查询

对每个子问题构造四类检索意图：

1. 核心事实与支持证据；
2. 因果机制和作用通路；
3. 冲突、负面结果和研究局限；
4. 实验方法、模型与测量指标。

为了减少额外模型调用，Adapter 将当前 `SearchRunResult` 作为可选
`search_context` 传给全文工作流。工作流复用已经生成的英文检索式、Scout
关键词、机制词和缺失主题。这个参数保持可选，因此旧假实现和备份实现可以继续只使用
`sub_question` 与 `papers`。

### 7.2 初筛评分

`HybridEvidenceRetriever` 首先使用无外部依赖的 BM25 词法分数召回片段，再组合：

- 关键实体和精确短语匹配奖励；
- 标题与摘要关键词匹配；
- 与检索意图相符的章节先验；
- 对过短、参考文献和纯声明性片段的惩罚。

各分量归一化后形成可解释的最终检索分数。若问题为中文，英文 Search Agent
查询和 Scout 术语作为主要召回词，避免单纯中英文词法不匹配。

### 7.3 多样性和上下文扩展

每类意图先保留少量高分候选，合并后使用 MMR 去除语义近似的重复片段。MMR
的相似度第一版使用词项重叠，不引入 embedding。

对最终命中片段可以补充同章节的前后相邻块，但所有发送给 Qwen 的文字必须满足单篇
字符预算。默认每篇最终保留 6 至 10 个片段。

### 7.4 检索失败

如果存在解析后的正文但没有任何片段获得正相关分数，工作流使用带明确标记的有限回退：

- 摘要；
- Results 或 Discussion 的首个非空片段；
- Methods 的首个非空片段。

回退只是提供原文上下文，不得直接生成固定知识结论。

## 8. Qwen 论文阅读

`QwenPaperReader` 对每篇论文执行一次结构化调用。输入包含：

- 子问题；
- 论文元数据；
- 检索到的证据片段及稳定 ID；
- 允许的六类 `KnowledgeEntryType`；
- 严格的证据引用要求。

模型返回：

- `summary`；
- 零个或多个知识条目；
- 每条知识的类型、内容、实体、置信度和 `evidence_ids`。

模型不得返回思维链，也不得引用未提供的论文内容。模型只选择证据 ID，不生成或
改写作为引用保存的原文；`EvidenceChunk.quote` 始终由 Retriever 从解析文档中直接
复制。

## 9. 证据验证

模型输出在进入 `PaperReadingResult` 前必须通过确定性校验：

- `paper_id` 必须匹配当前论文；
- `evidence_id` 必须来自当前调用提供的候选片段；
- 每条知识至少绑定一个有效证据 ID；
- 每个 `EvidenceChunk.quote` 必须是原始 `DocumentChunk.text` 的真实子串；
- 知识类型必须属于现有枚举；
- 内容和实体经过空白清理与长度限制；
- 重复知识条目按类型、规范化内容和证据集合去重。

未知证据 ID、伪造引文和无法解析的条目直接丢弃，并在结果错误中记录计数，不自动替换为
其他证据。

## 10. 并发、预算和错误隔离

- 全文解析为本地操作，不额外限流；
- 网络获取和 Qwen 阅读分别使用可配置的 Semaphore；
- 默认最多同时获取 3 篇全文、同时阅读 2 篇论文；
- 每个网络请求、每个模型调用和整个阅读工作流都有超时；
- 单篇论文失败生成该论文的错误结果，不中断其他论文；
- 任务取消和全局超时必须继续向上抛出，不能被当作普通单篇错误吞掉；
- 输出顺序必须与输入 Final-K 论文顺序一致。

## 11. 集成方式

`build_integrated_search_adapter()` 将注入 `FullTextReadingWorkflow`，而不是
`AbstractReadingWorkflow`。最小 PubMed 工厂仍保留原有摘要工作流，作为离线备份和消融
对照。

运行脚本的 trace 继续输出 `reading_results`，并增加足以诊断但不包含完整论文正文的统计：

- 内容级别；
- 解析片段数；
- 检索片段数；
- 知识条目数；
- 是否摘要降级；
- 各阶段耗时和错误。

## 12. 测试策略

### 12.1 Resolver

- 使用 PMCID 成功获取 BioC 全文；
- 使用 PMID 成功获取 BioC 全文；
- 404、429、超时、错误 JSON 和空正文；
- 全文失败时摘要降级；
- 全文和摘要都没有时不生成内容；
- 缓存命中不重复请求网络。

### 12.2 Parser

- 使用固定 BioC fixture 验证章节识别；
- 超长段落切分和重叠；
- 稳定 chunk ID；
- 空 passage、表格、参考文献和异常字段；
- 不改变原文内容。

### 12.3 Retriever

- 核心机制、方法、冲突和局限分别能够召回对应章节；
- 实体和短语匹配提高排序；
- MMR 减少重复片段；
- 相邻上下文不突破预算；
- 中文问题可以利用英文搜索上下文召回正文。

### 12.4 Reader

- 正常结构化输出；
- 未知证据 ID 被拒绝；
- 非原文引用被拒绝；
- 非法类型和空内容被拒绝；
- 模型网络错误形成单篇错误结果；
- 不生成没有证据的知识条目。

### 12.5 集成

- 集成工厂不再注入占位工作流；
- Final-K 输入与阅读结果一一对应且顺序稳定；
- 一篇全文成功、一篇摘要降级、一篇完全失败时能够部分成功；
- Adapter 输出仍可被现有 M3 消费；
- legacy M2、最小 PubMed 备份和现有搜索测试不回归；
- 使用一篇已知 PMC Open Access 论文和真实 Qwen 完成可选 live smoke test。

## 13. 验收标准

- 集成链路中不存在运行时占位 Tool；
- 有 PMC 开放全文时使用正文，而不是只用摘要；
- 全文不可用时明确降级，不编造全文；
- 每条输出知识都能追溯到真实论文片段；
- 完整论文不会进入模型上下文或最终 pipeline state；
- 单篇失败不破坏其他论文结果；
- 离线单元测试不依赖网络和真实 API Key；
- live test 能清楚报告真实网络、模型和降级结果；
- 所有已有测试和新增测试通过；
- 实现保持未提交，直到用户明确批准提交。
