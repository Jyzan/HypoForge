# HypoForge 持续优化设计

## 目标

在不破坏 M1–M6 现有输入输出契约的前提下完成四项改造：

1. 修复 M4 证据缺口返回 M2 时的状态错配，并用真实执行顺序证明回环可达。
2. 将 M2 实现移动到 `hypoforge/modules` 命名空间，与 M1–M6 并列，同时保留旧导入兼容层。
3. 为 M3–M6 建立可追溯、按用途裁剪的上下文管理，降低重复输入 token。
4. 拆分并美化 Web UI，明确显示模块详情、回环、当前 Tool 与 token 使用。

所有行为修改必须先有失败测试，修改后运行局部测试和完整测试；涉及运行时链路的修改还必须执行一次最小端到端运行。未经用户再次允许，不推送远端。

## 当前问题

### M4→M2 状态机

M4 使用 `evidence_gap_requests`，M6 使用 `evidence_gaps`。两类缺口在 M2 中按优先级处理：存在 M6 缺口时，M2 只把 M6 缺口转成子问题，却会把所有 M4 pending 请求一并标成 searched。随后 M3 将这些未实际搜索的请求标成 indexed，M4 再入时把它们耗尽。表面上发生了回环，实际证据问题没有被搜索。

### M2 文件位置

`hypoforge/modules/m2_literature_search.py` 只是入口，主体位于 `hypoforge/literature`。注册器还通过精确模块路径判断是否启用严格适配器，直接移动文件会静默改变行为；`strict_contracts.py` 也依赖旧模块全局并进行兼容绑定。

### 上下文和 token

`build_graph_context(state)` 生成通用的最多 24,000 字符上下文。M4 generator、critic、falsifiability、ranker，M5 每个假设以及 M6 各评审都会重复发送它。M4 generator 还同时发送相同的摘要桶。当前只统计进程级累计 token，没有每次调用的模块、用途、上下文组成和裁剪清单。

### UI

`hypoforge/web/index.html` 同时包含 HTML、CSS 和 JavaScript，约三千行。页面已有丰富详情能力，但首页密钥区域过大、信息层级偏弱，补搜回环和 token 消耗不够显眼，维护时容易牵动无关逻辑。

## 设计

### 1. 统一可执行搜索任务

在 M2 适配器入口构建内部 `SearchTask` 列表。每项包含：

- `question`
- `origin`: `initial_subquestion | m4_gap | m6_gap`
- `origin_id`

M4 和 M6 请求可以同时进入同一轮任务列表，按规范化问题文本去重，但保留一对多的来源 ID。只有成功进入本轮 `SearchRun` 的任务来源才允许推进状态：

- M4：`pending -> searched`
- M6：`open -> pending_grounding`

M3 仍负责根据新证据完成 grounding/indexing。回环上限继续由现有 `max_evidence_gap_rounds` 和 `max_search_rounds` 控制。

必须增加 runner 级测试，验证实际调用顺序为：

```text
M1 → M2 → M3 → M4 → M2 → M3 → M4 → M5 → M6
```

并增加 M4、M6 缺口并存的回归测试，确保没有被执行的请求状态不变化。

### 2. M2 兼容迁移

目标结构：

```text
hypoforge/modules/
├── m2_literature_search.py
└── m2_literature/
    ├── adapter.py
    ├── integrated.py
    ├── models.py
    ├── search/
    ├── sources/
    └── reading/
```

迁移规则：

1. 新代码的规范导入路径是 `hypoforge.modules.m2_literature.*`。
2. `m2_literature_search.py` 继续作为注册器稳定入口。
3. `hypoforge/literature/*` 在一个兼容周期内只做显式 re-export，不复制实现。
4. 注册器不再用 `__module__` 字符串判断 M2，而使用类属性 `strict_contract_family = "agentic_m2"`。
5. 状态 DTO 继续放在 `state.py`，防止 M2/M3 环形依赖。

先迁移叶子模块，再迁移聚合器和 adapter；每一步运行导入兼容、严格注册、取消、补搜、M2→M3 测试。最后用 `rg` 确认旧目录不再包含业务实现。

### 3. 上下文管理

增加 `hypoforge/context/`：

- `catalog.py`: `EvidenceCatalog`，保存完整证据、论文元数据、原文位置、图关系和稳定 hash。
- `planner.py`: `ContextPlanner`，根据调用用途和 token 预算选择证据。
- `models.py`: `ContextRequest`、`ContextPack`、`ContextManifest`。
- `token_budget.py`: 优先使用模型 tokenizer，缺失时采用保守字符估算。

`ContextPack` 必须包含：用途、预算、included IDs、dropped IDs 及原因、render hash。不得对最终字符串直接切片；只能按完整 evidence card 装包。完整原文仍保留在 catalog 中，裁剪只影响发送给模型的视图。

用途分别为：

- `m4_generate`: 多样化支持证据、冲突、缺口和少量关系。
- `m4_critic`: 当前候选引用的证据及最近的反证/限制。
- `m4_rank`: 候选卡和紧凑覆盖表，不发送原文引句。
- `m5_plan`: 当前假设支持/反对证据、方法约束和任务要求。
- `m6_logic`: 核心主张及正反证据。
- `m6_feasibility`: 研究计划、方法约束和相关证据。
- `m6_sufficiency`: 证据覆盖表和稳定 ID。

初始每次调用输入预算为 6,000 tokens，其中至少 20% 留给指令、Schema 和输出空间。先消除 M4 重复摘要桶，再逐模块切换。每次 LLM 调用记录模块、工具、用途、估算/实际输入输出 token、included/dropped 数量、上下文 hash 和耗时；计数按 run 隔离，不再只使用类级全局累计量。

验收门槛：固定的三候选、双评审测试夹具中，M4–M6 提示词 token 比当前基线降低至少 50%，所有被引用 evidence ID 仍能解析到 catalog，验证结果不因裁剪而丢失强制字段。

### 4. Web UI

新增静态资源路由并拆分：

```text
hypoforge/web/index.html
hypoforge/web/styles.css
hypoforge/web/app.js
```

不改变 API 契约。视觉改造采用渐进方式：

- 首页把问题输入作为主视觉，密钥配置折叠为次级面板。
- 统一模块卡片的间距、对比度、错误文本高度和响应式布局。
- 在 M4/M6 卡片旁显示补搜原因、轮次和返回路径动画。
- 运行区显示当前 Tool、耗时、输入/输出 token 和上下文裁剪数量。
- 保留 M1–M6 详情及 M3 全屏图谱；已完成模块摘要不会因轮询自动关闭。

拆分后对内联 JavaScript 做语法检查，并使用浏览器分别检查桌面宽屏、窄屏、空状态、运行中、失败和回环状态。

## 实施顺序与隔离

1. M4→M2 修复：独立提交，完整测试和最小回环运行。
2. M2 迁移：分批兼容提交，完整测试和独立 M2 运行。
3. 上下文管理：先观测，再装包，再逐模块接入；每阶段保留 token 对照。
4. UI：最后拆分和美化，避免业务调试与前端调试互相干扰。

所有工作在 `feat/integrated-fast-mode-20260823` 本地分支继续。每项通过后允许本地提交；未经授权不执行 `git push`。

## 非目标

- 不重写 M1、M3、M4、M5、M6 的科研业务提示词。
- 不更改 M3/M4/M5/M6 对外状态 DTO。
- 不引入新的远程数据库或必须联网的新依赖。
- 不以减少检索或删除证据来换取 token 降低。
