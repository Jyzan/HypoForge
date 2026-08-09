# M2 OpenAlex 独立检索恢复设计

## 目标

在保留当前综合版 UI、M1、M3-M6 和 M2 文献门面的前提下，恢复旧 UI 版已经实现过的 OpenAlex 独立检索源。一次 M2 首轮检索应能把 PubMed、Semantic Scholar、OpenAlex、arXiv 作为四个彼此独立的来源调度、计时和记录。

## 根因

`d0d5c6e` 后的 M2 重构删除了 `OpenAlexSource`，并把 OpenAlex 仅保留为 `SemanticScholarTool` 的失败回退。结果是 Semantic Scholar 成功时 OpenAlex 永远不会运行；Semantic Scholar 耗尽 15 秒阶段 deadline 时，相同的过期 deadline 又被传给 OpenAlex；网页传入的 OpenAlex 凭据也没有被 integrated factory 接收。

## 设计

1. 恢复 `OpenAlexSource`，通过 `LiteratureSourceProtocol` 独立返回 `PaperRecord`。
2. `LiteratureSearchTool` 暴露四个 Tool，并分别持有 Semantic Scholar 与 OpenAlex Source。
3. integrated factory 接收并转发 `openalex_api_key`、`openalex_mailto`；实时配置重新启用 `openalex`。
4. Semantic Scholar 的 OpenAlex 回退保留为容错，但 OpenAlex 获得新的阶段 deadline，不能继承 S2 已耗尽的 deadline。
5. 所有进入 OpenAlex 的检索词统一清除 PubMed 字段标签、布尔操作符、括号和引号，避免 OpenAlex 对复杂布尔表达式返回 HTTP 400。
6. 不修改 UI，不记录或提交任何 API Key，不覆盖当前工作区的其他未提交修改。

## 验收标准

- 默认 `LiteratureSearchTool` 暴露四个来源。
- integrated factory 能构造包含四个 Source 的搜索代理，并能传入 OpenAlex 凭据。
- OpenAlex 查询由独立 Source 执行并标记为 `openalex`。
- S2 超时或失败后，回退到 OpenAlex 时仍获得新的有效 deadline。
- 复杂查询可转换成 OpenAlex 可接受的自由文本。
- M2 定向测试、全量测试和真实 OpenAlex 冒烟测试通过。
