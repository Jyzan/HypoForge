# HypoForge 新协作基线整合设计

## 目标

以同学上传的 `d0d5c6e`（`fix/generalized-evidence-contracts`）作为新的代码基础，整合：

1. 当前主目录中已经验证过的 M1 逻辑；
2. `HypoForge-backups/ui-core-before-generalized-evidence-20260808-220001` 中的新 UI、追问迭代与凭据输入功能；
3. 同学最新版的 M2 literature facade、严格证据契约和终端进度事件。

最终产物成为双方共同使用的新 GitHub 基线分支。旧版完整保存在本机，不覆盖、不删除。

## 基线与备份

- 新版基础提交：`d0d5c6e refactor(m2): route pipeline through literature facade`。
- 当前旧版主目录在任何换位前复制到带时间戳的备份目录。
- 备份包含：所有 tracked/untracked 代码、文档、测试、配置和本地 `.env`。
- 备份同时保存：当前提交号、分支名、`git status`、未暂存差异和暂存差异。
- `.venv`、`.pytest_cache`、`__pycache__`、运行输出和可重新生成的论文/模型缓存不重复复制。
- `.env` 只保存在本机备份和新的本地工作目录，继续由 `.gitignore` 排除，任何提交前都检查其未被暂存。

## 主目录更新策略

不采用“把若干目录直接覆盖到一起”的方式。主目录先切换到远程提交 `d0d5c6e` 的干净基线，再按功能语义移植旧版修改。

这样可以保证新版删除的 legacy M2 不会被旧文件重新带回来，也不会破坏 `modules/m2_literature_search.py -> literature.adapter` 的单一入口。

## M1 移植范围

移植已经验证通过的完整 M1 行为：

- 首次提问先生成领域和候选子问题，不在第一次调用中生成实体；
- 每个子问题必须原子、单一关系、长度受控，总数最多 5 个；
- 使用 LLM 进行覆盖检查，必要时补充或合并子问题；
- 关键实体只能来自原问题明确出现的专业名词或角色；
- 实体提取后必须再调用一次 LLM 做 source-bound audit；
- 实体审核失败时只做有限次数修复，不允许从子问题引入新实体；
- 根据审核后的实体与子问题生成 `TaskContract`；
- 不再生成或消费“问题类别”；历史快照中的 `question_type` 只作为兼容字段存在；
- 所有追问先调用 LLM 分类；语言、排版、呈现调整复用父运行并跳过检索，研究目标变化则重建 ProblemCard；
- 低置信度、矛盾路由或没有可复用图谱时，默认重新搜索，避免错误跳过 M2/M3。

移植不仅包括 `m1_problem_understanding.py` 和 `m1_prompts.py`，也包括为了取消问题类别依赖而修改的 M2 query planner、coverage、adapter 兼容逻辑、状态模型、显示层及相应测试。

## 新 UI 整合范围

以备份中的新 UI 为用户界面基础，保留：

- 首页问题输入与本地凭据输入；
- M1-M6 实时状态、Tool 事件和模块详情；
- 追问迭代入口、父运行关系、轮次时间线；
- M6 回到 M2/M4 的可视化；
- 错误卡片限高、摘要弹窗稳定显示及历史版本查看。

但以下后端代码不能由备份版本整文件覆盖：

- `pipeline.py`：以 `d0d5c6e` 为主体，保留其 M2 event sink 和专用进度面板，再接入新 UI 需要的迭代/事件字段；
- `observability.py`：保留最新版的多事件接收器能力；
- M2 模块和 literature 子包：完全以最新版为主体；
- `webapp.py`：按 API 路由逐项合并，不恢复已经被最新版废弃的 legacy M2 参数。

## 已知新版缺陷处理

整合过程中同时修复 `d0d5c6e` 已确认的两个确定性缺陷：

1. M2 retention judge 事件文本中的 `????` 乱码；
2. Scout 实体校验中 `text.count("?") != text.count("?")` 永远为假的中文括号检查。

网络代理和 S2/OpenAlex 共用截止时间属于运行策略风险，本轮先写入验证结果；只有真实搜索测试证明阻塞主流程时才改动，避免扩大本次整合范围。

## 测试与验收

按以下顺序验证：

1. M1 单元测试：原子性、最多 5 个、覆盖补充、实体来源审核、TaskContract、follow-up LLM 路由；
2. M2 受影响测试：query planner、coverage、adapter、Scout 实体校验、retention judge；
3. UI/API 测试：新建运行、SSE 事件、模块详情、追问运行、凭据不持久化到 Git；
4. 全量 `pytest`；
5. 启动本地网页，进行一次离线/模拟完整流程；条件允许时再进行一次真实 Qwen 流程；
6. 检查 `git diff --check`、`.env` 未跟踪/未暂存、无旧版 legacy M2 文件复活。

验收标准：

- 新版 M2 facade 和 637 项原始测试能力不回退；
- 两个已验证问题能够生成合格的 M1 子问题和原问题来源实体；
- 新 UI 能显示首次运行与 follow-up 迭代；
- M1-M6 主流程接口保持兼容；
- 旧版能从备份目录完整恢复；
- 提交中不含任何 API key、运行输出或缓存。

## Git 协作策略

- 综合版本使用独立分支 `integration/generalized-evidence-ui-v2`。
- 只在整合与测试完成后创建提交；不把中间失败状态推送为团队基线。
- 推送前展示最终变更文件、测试结果和提交内容摘要。
- 推送后同学应从该分支重新建立修复分支，避免继续基于旧 UI 提交。
