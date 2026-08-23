# Purpose-Aware Context Management Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace repeated generic 24k-character graph prompts with deterministic, traceable context packs tailored to each M4–M6 LLM call, while preserving canonical evidence IDs.

**Architecture:** `build_graph_context` remains the authoritative resolver. A new pure `ContextPlanner` selects complete evidence cards and relations into a `ContextPack` according to a purpose profile and token budget; every pack carries an inclusion/drop manifest and render hash. M4–M6 consume packs but keep the full GraphContext for validation.

**Tech Stack:** Python dataclasses/Pydantic, hashlib, deterministic selection, existing observability events, pytest.

**Spec:** `docs/superpowers/specs/2026-08-23-hypoforge-continuous-optimization-design.md`

## Global Constraints

- Never truncate in the middle of an evidence card or relation.
- Every included or dropped entry retains its canonical ID in the manifest.
- Purpose selection must be deterministic for identical input.
- Context reduction must not alter public state DTOs or evidence-ID validation.
- M4 generator must not repeat graph buckets outside the context pack.
- M5 context must be hypothesis-specific; M6 context must be reviewer-purpose-specific.
- Do not push the branch.

---

### Task 1: Build token budget and context-pack contracts

**Files:**
- Create: `hypoforge/context/__init__.py`
- Create: `hypoforge/context/models.py`
- Create: `hypoforge/context/token_budget.py`
- Create: `hypoforge/context/planner.py`
- Modify: `scripts/test_pipeline.py`

**Interfaces:**
- Produces: `ContextRequest(purpose, max_input_tokens=6000, reserve_tokens=1200, focus_evidence_ids=(), focus_entry_ids=())`.
- Produces: `ContextManifest(purpose, budget_tokens, estimated_tokens, included_ids, dropped_ids, drop_reasons, render_hash)`.
- Produces: `ContextPack(rendered, manifest)`.
- Produces: `ContextPlanner.plan(graph_context, request) -> ContextPack`.

- [ ] **Step 1: Add failing deterministic/budget tests**

Build a synthetic `GraphContext` with long complete cards in all three buckets. Assert the same request yields the same rendering/hash, `estimated_tokens <= max_input_tokens - reserve_tokens`, every rendered card ends before the next card, and `included_ids ∪ dropped_ids` equals all source entry IDs.

- [ ] **Step 2: Verify RED**

Expected: import failure because `hypoforge.context` does not exist.

- [ ] **Step 3: Implement conservative token estimation**

Estimate ASCII runs at four characters/token and every non-ASCII code point at one token; add a fixed per-message overhead. The estimator must never return zero for non-empty text.

- [ ] **Step 4: Implement purpose profiles and complete-card packing**

Use bounded profiles for `m4_generate`, `m4_critic`, `m4_rank`, `m5_plan`, `m6_logic`, `m6_feasibility`, and `m6_sufficiency`. Prioritize focus IDs, then source order. Add a complete card only when its estimated tokens fit. Render compact task-contract JSON and an explicit omitted-ID footer. Hash the exact rendered text with SHA-256.

- [ ] **Step 5: Run planner tests GREEN**

Expected: all context tests PASS.

### Task 2: Remove M4 prompt duplication and tailor quality-gate contexts

**Files:**
- Modify: `hypoforge/modules/m4_hypothesis_generation.py`
- Modify: `scripts/test_pipeline.py`

**Interfaces:**
- Consumes: `ContextPlanner.plan` purposes `m4_generate`, `m4_critic`, `m4_rank`.
- Produces: `llm_context_built` event details from each pack manifest.

- [ ] **Step 1: Add a failing generator prompt-shape test**

Use a recording client and a graph containing a unique fact marker. Assert the generator user prompt contains that marker exactly once and its emitted context event names `m4_generate` with included/dropped IDs.

- [ ] **Step 2: Verify RED**

Expected: FAIL because the marker appears in `graph_context` and the separate established-facts bucket.

- [ ] **Step 3: Integrate generator pack**

Build one `m4_generate` pack per M4 pass, pass its rendering as `graph_context`, and supply only short “included in context pack” placeholders for the three legacy bucket template fields.

- [ ] **Step 4: Integrate critic/falsifiability/ranker packs**

Critic and falsifiability use `m4_critic`, focused on candidate supporting evidence. Ranker uses `m4_rank` with no quotes. Contract checks continue using the full GraphContext object.

- [ ] **Step 5: Run M4 focused tests**

Expected: prompt-shape and existing M4 tests PASS.

### Task 3: Scope M5 and M6 contexts

**Files:**
- Modify: `hypoforge/modules/m5_research_plan.py`
- Modify: `hypoforge/modules/m6_review_iteration.py`
- Modify: `scripts/test_pipeline.py`

**Interfaces:**
- M5 consumes one `m5_plan` pack per hypothesis, focused on `supporting_evidence`.
- M6 consumes `m6_logic` for scientific/testability/novelty reviewers, `m6_feasibility` for method reviewers, and `m6_sufficiency` for the verdict.

- [ ] **Step 1: Add failing M5 differentiation test**

Use two hypotheses citing disjoint evidence. Capture both plan prompts and assert the context manifests differ and each focused evidence card appears before non-focused cards.

- [ ] **Step 2: Add failing M6 purpose test**

Capture reviewer prompts/events and assert scientific and method reviewers receive different purpose/hash pairs; assert the sufficiency judge uses `m6_sufficiency`.

- [ ] **Step 3: Integrate M5 per-hypothesis packs**

Move context planning inside the hypothesis loop. Preserve the full `available_evidence_ids` set for post-generation validation.

- [ ] **Step 4: Integrate M6 per-reviewer and sufficiency packs**

Build the full graph context once, select a purpose per reviewer dimension, and emit bounded manifest events. Keep deterministic hard gates unchanged.

- [ ] **Step 5: Run M5/M6 focused tests GREEN**

Expected: all selected tests PASS.

### Task 4: Measure and enforce reduction

**Files:**
- Modify: `scripts/test_pipeline.py`
- Create: `scripts/measure_context_budget.py`

**Interfaces:**
- Produces: deterministic baseline/current prompt-character and estimated-token report for a large synthetic graph.

- [ ] **Step 1: Add a fixed large-graph fixture and measurement script**

Construct 12 facts, 12 conflicts, 12 gaps, quotes and 20 relations. Compare ten legacy full renders with the exact mix of purpose packs used by M4–M6.

- [ ] **Step 2: Add the reduction gate**

Assert total purpose-pack estimated input tokens are at most 50% of ten legacy full renders and all focused/cited IDs remain included or explicitly listed as dropped.

- [ ] **Step 3: Run compile, focused and full tests**

```powershell
python -m compileall -q hypoforge
python scripts/measure_context_budget.py
python -m pytest -q
```

Expected: measurement reports at least 50% reduction and all tests PASS.

- [ ] **Step 4: Commit locally**

```powershell
git add hypoforge/context hypoforge/modules/m4_hypothesis_generation.py hypoforge/modules/m5_research_plan.py hypoforge/modules/m6_review_iteration.py scripts
git commit -m "feat(context): add purpose-aware evidence budgets"
```
