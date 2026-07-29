## 合并时可能需要注意的具体问题

### 1. Track C 的 NoveltyMetric 用 BFS 在图里找路径

Track C 把 NoveltyMetric 从简单 BM25 改成了 **networkx BFS 最短路径**：
- 把假设拆成 subject 和 object 两个概念
- 在 EvidenceGraph 里分别搜 subject 和 object 对应的节点
- 建一个 networkx 图（去掉 CONTRADICTS 和 LIMITS 边）
- BFS 找 subject 节点 → object 节点的最短距离
- 距离 0/1 判不新颖，越远越新颖
**问题**：我们的 EvidenceGraph 有两部分——规则图（`N_xxx`）和接地图（`GCLM_xxx`/`GEV_xxx`），两部分之间没有边。如果 subject 节点在规则图里、object 节点在接地图里，BFS 永远找不到路径，`min_dist = inf`，全部判为"完全新颖"。

不过这一部分怎么调整可以按照需求适度改一下，没必要说一定怎么调整，好像可以把图连在一块，但好像也没这个必要。

### 2. Track C 只搜索 `n.label`

不管是 NoveltyMetric 的 `_keyword_search` 还是 EvidenceConsistencyMetric 的 embedding 匹配，都只读 `n.label`，不读 `metadata`。我们的节点类型和 label 内容：

| 节点类型 | ID 前缀 | label 是什么 | 搜索好用吗 |
|---------|---------|-------------|----------|
| CLAIM | `N_xxx` | `KnowledgeEntry.content[:120]` | 还行，但截断了 |
| CLAIM | `GCLM_xxx` | `claim.statement[:200]` | 可以 |
| EVIDENCE | `N_xxx` | `KnowledgeEntry.content[:120]` | 还行 |
| EVIDENCE | `GEV_xxx` | `record.summary[:200]`（LLM 摘要） | 摘要可能跟查询词不一致 |
| ENTITY | `ENT_xxx` | `"hsp70"`（就一个词） | 基本没用 |
| SOURCE | `SRC_xxx` | 论文标题 | 不包含证据内容 |

Track C 可以优先读 `metadata.searchable_text` 或 `metadata.quote` 等字段。