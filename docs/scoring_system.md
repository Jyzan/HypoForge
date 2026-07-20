# HypoForge 评分系统设计文档

> 对应代码：`hypoforge/evaluation/`（`rubric.py` / `metrics.py` / `scorer.py`）、`hypoforge/config.py::ScoringConfig`、M4 / M6 模块。

---

## 1. 旧实现的问题

在本次重构之前，HypoForge 内部存在 **三套各自独立的打分逻辑**，且随着迭代逐渐漂移，导致评分结果不可信、不可比。

### 1.1 三套打分系统互不统一

| 子系统 | 打分对象 | 维度 | 量纲 | 权重来源 |
|---|---|---|---|---|
| M4 排序器（假设生成） | 候选假设 | novelty / scientific_soundness / testability / evidence_consistency | [0,1] | prompt 里写一份、代码里写一份，**易漂移** |
| M6 评审器（方案审核） | 假设 + 研究方案 | scientific_logic / evidence_consistency / method_feasibility | 1–5 | 硬编码在模块内，与 M4 无关 |
| `evaluation/` 打分器 | 最终假设 | 另一套维度 | [0,1] | 无独立权重，**直接抄 M4 自评分** |

核心矛盾：同一个 `novelty` 在 M4 和 evaluation 里是两个数，同一个 `evidence_consistency` 在 M4 和 M6 里是两套标准。无法横向对比同一假设在不同环节的表现，也无法做消融实验。

### 1.2 循环自证（Circular Self-Validation）

这个问题在两个层面都存在：

**层面一：`evaluation/` 层照抄 M4 自评分。** 重构前，最终评分报告直接读取 M4 输出的 `hypothesis.scores`，原样写入报告——生成器声称自己的假设有多新颖，评分器就照抄，没有任何独立验证。

**层面二：M4 内部 Generator 和 Ranker 是同一个模型。** Generator 用 `qwen3.7-max` 生成假设，Ranker 也用同一个 `qwen3.7-max` 打分排序。即使 prompt 不同，底层模型的偏好和盲区是共享的——Generator 认为有意义的方向，Ranker 大概率也会给高分。而且 prompt 里定义了 Critic 和 Falsifiability Checker 两个把关 agent，但代码中**从未调用过它们**，实际只跑了 Generator → Ranker 两步。

```
Generator (qwen3.7-max) → Ranker (qwen3.7-max) → 自评分写入报告
    ↑ 同一模型自产自评 ↑                    ↑ 循环回到 evaluation ↑
```

### 1.3 缺少 Rubric 锚定

M4 和 M6 的 prompt 中虽然要求打分，但**没有给出每档分数对应什么标准**（rubric anchor）。模型是在"凭感觉打分"——同一个假设，0.7 和 0.85 之间的差别没有语义依据，只是 token 概率的随机波动。

### 1.4 缺少迭代收敛机制

M6 评审完会给出意见和建议，但**没有量化门控来决定是否继续迭代**。系统不知道当前方案是否已经"足够好"，也不知道什么时候该停止打磨。

---

## 2. 当前设计的思路

重构的核心原则只有一条：**所有打分逻辑收敛，且严格区分"模型自评"与"独立评估"。**

### 2.1 单一真源：`rubric.py`

`hypoforge/evaluation/rubric.py` 是整个评分系统的唯一权威定义源。所有其他子系统（M4、M6、scorer）**只引用、不重复定义**：

- **维度名称与默认权重**：`novelty 0.30 / scientific_soundness 0.25 / testability 0.25 / evidence_consistency 0.20`
- **复合分公式** `composite_score()`：加权求和，且只对实际出现的维度做权重重归一化
- **Rubric 锚点**：每个维度在 0.0 / 0.5 / 1.0（或 1 / 3 / 5）分别代表什么，既写进评分报告，也可注入 M4/M6 的 prompt
- **跨量纲归一化** `normalise_review_score()`：把 M6 的 1–5 映射到 [0, 1]，使两类分数可对齐比较

`PipelineConfig.scoring.hypothesis_weights` 可以覆盖默认权重，但覆盖后的值**同时喂给 M4 排序器公式和事后打分器**——两个环节永远用同一组权重，不会漂移。

### 2.2 Reason-before-score：先论证，再打分

通过 **prompt 中的字段顺序**强制"先推理后评分"，提升打分校准度：

- **M4 排序器**：先逐维写出 `ranking_rationale`，再给出 4 个 [0, 1] 分数
- **M6 评审器**：先写 `reasoning`（引用假设/方案/证据图的具体内容指出问题），再给 1–5 分和 `suggestions`

这个设计的依据是大量 prompt-engineering 研究表明：让模型先组织论据、再做判断，比直接让它给一个数字，分数的一致性和可解释性都更好。

### 2.3 自评 vs 独立评：物理隔离

评分报告中将两类分数**显式分开**：

- **`self_reported`**：M4 给自身假设打的分。保留在报告里做透明性参考，但**明确标注"非独立评估"**
- **`independent`**：打分器用客观信号**独立重新计算**的分，与 M4 的自评完全解耦

| 指标 | 状态 | 计算方式 |
|---|---|---|
| `testability` | ✅ 已实现 | LLM-as-judge（turbo 模型），对预测的具体性/方向性/可测量性 + 证伪条件是否真正能否证假设，按 5 级 rubric 打分，而非简单检查字段是否非空 |
| `novelty` | ⏳ 声明未实现 | 需要检索/嵌入 grounding（bm25），未实现前直接跳过 |
| `evidence_consistency` | ⏳ 声明未实现 | 需要证据图 CONTRADICTS 边 grounding，未实现前跳过 |

设计取舍：**宁可暂时少一个独立分，也不把自评数字包装成客观评估**。

### 2.4 复合分重归一化

```python
composite = Σ (w_normalized[d] × score[d])   # d 只取实际出现的维度
```

关键点：**只对实际出现的维度做权重归一化**。若某维度缺失（如模型没输出），权重在剩余维度间重新分配，不会让缺失维度把复合分悄悄拖向 0。

### 2.5 迭代收敛门控

M6 完成后，`overall` 分驱动迭代决策：

```
若 iteration_count ≥ max_iterations（默认 3）→ 结束
若 overall ≥ review_threshold（默认 4.0）     → 提前达标，结束
否则                                           → 带着评审意见回流 M4 修订
```

其中 `overall` 不是单独调一次模型，而是**取三位专家 reviewer 的均值**，避免第四个主观打分带来新的偏差。

---

## 3. Pipeline 流程

### 3.1 整体架构

```
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐    ┌──────────┐
│  M1      │   │  M2      │   │  M3      │   │  M4      │   │  M5      │    │  M6      │
│  问题理解│──▶│  文献检索│─▶│  证据图谱│──▶│  假设生成│─▶│  研究方案│──▶│  评审迭代 │
│          │   │          │   │          │   │  + 打分   │   │          │   │  + 打分  │
└──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘    └─────┬────┘
                                                                                  │
                                                              ┌───────────────────┘
                                                              │  overall < 4.0 且未达 max_iter
                                                              ▼
                                                         M4 修订假设（带入评审意见）
```

### 3.2 三重打分解析

**一：M4 假设评分（生成侧自评）**

M4 内部是 4 步 pipeline，而非简单的"生成 → 打分"：

```
Generator (qwen3.7-max)        ← 生成 N 个候选假设
    ↓
Critic (qwen3.7-max)           ← 检查逻辑一致性 + 外部一致性
    │                             pass=false → 淘汰
    │                             suggested_revision → 自动修正 statement
    ↓
Falsifiability Checker         ← 检查是否可证伪
    (qwen3.7-max)                 is_falsifiable=false → 淘汰
    ↓
Ranker (qwen3.7-plus)          ← 不同模型！打破自评闭环
                                 对幸存候选做 4 维 [0,1] 打分
                                 → 加权复合分排序 → 返回 top-k
```

关键设计决策：

- **Generator 和 Ranker 使用不同模型**（`base` vs `plus` tier），避免同一模型的偏好盲区导致"自己产出的假设自己给高分"。
- Critic 和 Falsifiability Checker 作为两道质量过滤器，在两个维度上独立把关。如果过滤后幸存候选不足 2 个，回退到原始列表，确保 pipeline 不会空转。
- 权重来自 `rubric.py`，通过 `weights_summary()` 注入 ranker 的 system prompt，同时用于代码层的 `composite_score()` 复算。如果 ranker 没输出 `composite` 或数值不一致，代码层会用 `rubric.py` 的公式**强制重算**。

**二：M6 评审评分（评审侧）**

- 流程：3 位 specialist reviewer（scientific_logic / evidence_consistency / method_feasibility）各打 1–5 分 → `overall` = 三者的均值
- 每位 reviewer 先写 reasoning（引用假设、方案、证据图），再给分数和建议
- Rubric 锚点（1/3/5 各代表什么）通过 `review_rubric_line()` 注入 prompt

**三：运行后独立打分器（`scorer.py`）**

在整个 pipeline 结束后自动运行（由 `ScoringConfig.auto_score` 控制），生成 `{run_id}_scores.json`。内容包括：

- 每个假设的 `self_reported` + `independent` + `composite`
- 每个方案的完整度
- `aggregate` 汇总

**顶层 `composite` 的计算。**

注意报告中 `self_reported.composite`（示例中为 0.81）和顶层 `composite`（0.8045）是**两个不同的数字**：

- `self_reported.composite` 是 M4 Ranker 在 LLM 调用中自己算的
- 顶层 `composite` 是 scorer 用 `rubric.py` 的 `composite_score()` **重新算**的

具体计算过程（以示例数据为例）：

```
composite = 0.30 × 0.82  (novelty)
          + 0.25 × 0.78  (scientific_soundness)
          + 0.25 × 0.75  (testability)
          + 0.20 × 0.88  (evidence_consistency)
        = 0.246 + 0.195 + 0.1875 + 0.176
        = 0.8045
```

两者差 0.0055，原因是 Ranker 在 LLM 里手工算加权分时有浮点舍入偏差。**顶层 `composite` 才是权威值**——它保证无论 Ranker 怎么算，最终分数用的是 Python 端的精确公式。

**`aggregate` 四个字段的含义。**

| 字段 | 示例值 | 含义 | 用途 |
|---|---|---|---|
| `top1_composite` | 0.8045 | 排名第一的假设的复合分 | 单次运行的最佳质量——"这次最好的假设有多好" |
| `mean_composite` | 0.796 | 所有 top-k 假设复合分的均值 | 衡量整批假设的平均水平；0.796 < 0.8045 说明 H2/H3 比 H1 稍弱 |
| `mean_plan_completeness` | 0.9394 | 所有研究方案完整度的均值 | 0.9394 ≈ 10.33/11，方案平均填了约 10 个有效字段 |
| `latest_overall_review` | 4.1 | 最后一轮 M6 overall 评审分（1–5） | 4.1 ≥ 阈值 4.0 → 迭代提前达标停止；若此值低（如 2.5）说明即使跑满 3 轮也没打磨好 |

四个数字合在一起，一次运行的质量被完整刻画：**假设本身好不好**（composite）、**方案写得全不全**（plan completeness）、**评审是否认可**（overall review）。做消融实验时，改一个参数跑两次，对比这组数字就知道方向对不对。

### 3.3 方案完整度（Plan Completeness）

独立于假设评分之外，对研究方案的 11 个要素做清单式检查：

- 占位符识别：`TBD`、`待定`、`暂无`、`?` 等**不算已填写**
- 过短内容（< 3 字符）也不算已填写
- 返回 filled / 11，范围 [0, 1]

### 3.4 配置入口

所有评分相关参数集中在 `ScoringConfig`：

| 字段 | 默认值 | 说明 |
|---|---|---|
| `hypothesis_weights` | `0.30/0.25/0.25/0.20` | 同时喂给 M4 公式和事后打分器 |
| `review_threshold` | `4.0` | M6 overall 达标线，触发提前停止 |
| `auto_score` | `true` | 跑完自动写评分报告 |

启动时从 YAML 加载，环境变量可覆盖。

---

## 4. 后续优化方向

### 4.1 Novelty 独立打分

当前 `NoveltyMetric` 标记为 `implemented = False`。实现思路：

- 用 bm25 或 embedding 检索，衡量假设核心声明与已检索文献的语义重叠度
- 重叠度越低 → novelty 越高（真正的"新"，而非模型自认为的新）
- 也可引入 time-cutoff 回溯评测：用截止日期之前的文献生成假设，看它能否预测截止日期之后才发表的真实发现

### 4.2 Evidence Consistency 独立打分

当前 `EvidenceConsistencyMetric` 标记为 `implemented = False`。实现思路：

- 基于 M3 证据图中的 `CONTRADICTS` 边做 grounding
- 若假设声明与已确立事实存在冲突边 → 扣分
- 可用 LLM-as-judge + citation 的方式，但要求 judge 的输入是证据图的内容（而非 M4 的自评结论）

### 4.3 其他改进点

- **M6 overall 的加权逻辑**：当前是简单均值，可根据假设类型（机制型 / 方法型 / 临床型）动态调整 reviewer 权重
- **人机协同评分**：在交互模式下，允许用户在 M6 评审后手动输入评分，与模型评分合并计算
- **独立指标扩展**：novelty 和 evidence_consistency 的独立打分实现后，`independent` 字段将从当前的一项（testability）扩展到三项，报告的信息量会显著提升

---

## 附录：输出报告示例

```json
{
  "run_id": "test-001",
  "question": "蛋白质如何折叠及错误折叠导致疾病的机制？",
  "iterations": 3,
  "hypothesis_scores": [
    {
      "hypothesis_id": "H1",
      "self_reported": {
        "novelty": 0.82,
        "scientific_soundness": 0.78,
        "testability": 0.75,
        "evidence_consistency": 0.88,
        "composite": 0.81
      },
      "independent": {
        "testability": 0.65
      },
      "composite": 0.8045
    }
  ],
  "plan_scores": [
    { "hypothesis_id": "H1", "plan_completeness": 1.0 }
  ],
  "aggregate": {
    "top1_composite": 0.8045,
    "mean_composite": 0.796,
    "mean_plan_completeness": 0.9394,
    "latest_overall_review": 4.1
  },
  "rubric": {
    "weights": { "novelty": 0.30, "scientific_soundness": 0.25, "testability": 0.25, "evidence_consistency": 0.20 },
    "note": "self_reported = M4 生成器自评（非独立评估）; independent = 仅客观复算"
  }
}
```

> 注意：`self_reported.composite`（0.81）是 M4 自报的；顶层 `composite`（0.8045）是打分器用配置权重复算的。**后者才是权威值**——两者可能因 ranker 输出精度与代码端复算的舍入差异而略有不同。
