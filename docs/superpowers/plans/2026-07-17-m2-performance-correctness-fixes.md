# M2 Performance and Correctness Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove repeated Scout work, enforce the real search deadline, degrade failed academic sources, and make search/coverage decisions stable and source-correct.

**Architecture:** Keep orchestration policy in `IterativeSearchAgent`, provider configuration in `SemanticScholarTool`, and model-specific normalization in Query Planner/Coverage. Reuse immutable Scout notes by paper ID, pass cached notes in ranked order to Coverage, and wrap every awaited stage in the remaining global budget.

**Tech Stack:** Python, asyncio, Pydantic, LangChain, pytest, PubMed, Semantic Scholar/OpenAlex.

## Global Constraints

- Do not commit or push without explicit user approval.
- Do not fabricate papers or hide network failures.
- Do not persist API keys or live model output.
- Keep `AbstractReadingWorkflow` unchanged as the documented placeholder.
- Every production change must first have a failing regression test.

---

### Task 1: Incremental Scout and bounded candidates

**Files:**
- Modify: `hypoforge/literature/search/agent.py`
- Modify: `hypoforge/literature/integrated.py`
- Test: `tests/literature/test_search_agent.py`
- Test: `tests/literature/test_integrated_factory.py`

- [ ] Add a two-round test asserting Scout receives `[\"paper-1\"]` then only `[\"paper-2\"]`, while Coverage receives cached notes for both ranked papers.
- [ ] Verify the test fails because the second call currently contains both papers.
- [ ] Read only paper IDs absent from `scout_by_paper`; assemble `scout_notes` from the cache in ranked order before Coverage.
- [ ] Change the integrated candidate limit from `max(50, final_k)` to `max(20, final_k)` and assert it in the factory test.
- [ ] Run the focused tests and expect them to pass.

### Task 2: Hard global time budget

**Files:**
- Modify: `hypoforge/literature/search/agent.py`
- Test: `tests/literature/test_search_agent.py`

- [ ] Add an async test with a slow Planner and `max_seconds=1`, asserting wall time stays close to one second and `StopReason.TIME_BUDGET` is returned.
- [ ] Verify RED: the current Planner completes after the deadline.
- [ ] Make the stage wrapper calculate remaining wall time and use `asyncio.wait_for`; translate expiry into `TIME_BUDGET` while preserving stage timing and final state.
- [ ] Run all Search Agent tests and expect them to pass.

### Task 3: Academic backend configuration and circuit breaking

**Files:**
- Modify: `hypoforge/tools/semantic_scholar.py`
- Modify: `hypoforge/literature/sources/academic_source.py`
- Modify: `hypoforge/literature/models.py`
- Modify: `hypoforge/literature/search/agent.py`
- Modify: `hypoforge/literature/search/query_planner.py`
- Test: `tests/literature/test_teammate_search_tools.py`
- Test: `tests/literature/test_search_agent.py`

- [ ] Add tests proving a constructor Semantic Scholar key selects/authenticates Semantic Scholar, OpenAlex key is added to OpenAlex requests, and propagated errors name the actual backend.
- [ ] Add a two-round Agent test proving a failed source is marked unavailable and is not called again.
- [ ] Verify all tests fail against the current global backend and repeated dispatch behavior.
- [ ] Make backend/key selection instance-specific, expose `backend_name`, and preserve the planner routing alias.
- [ ] Store unavailable routing sources in `SearchState`, show them in Planner prompts, and filter them before dispatch.
- [ ] Run focused academic and Agent tests.

### Task 4: Stable queries and Coverage gaps

**Files:**
- Modify: `hypoforge/literature/search/query_planner.py`
- Modify: `hypoforge/literature/search/coverage.py`
- Test: `tests/literature/test_teammate_search_tools.py`
- Test: `tests/literature/test_coverage.py`

- [ ] Add tests asserting Academic queries contain no PubMed field tags and Planner temperature is zero.
- [ ] Add a later-round Coverage test asserting a newly invented model gap cannot replace the previous target gap, while a persistent previous gap can still block.
- [ ] Verify RED.
- [ ] Sanitize backend-specific query text, make Planner deterministic, and only accept new model gaps in round one; later rounds retain only previously tracked gaps.
- [ ] Run Planner and Coverage tests.

### Task 5: Unknown citation metadata

**Files:**
- Modify: `hypoforge/literature/sources/pubmed_source.py`
- Modify: `hypoforge/literature/search/ranking.py`
- Test: `tests/literature/test_teammate_search_tools.py`
- Test: `tests/literature/test_ranking.py`

- [ ] Add tests asserting absent PubMed citation data stays `None` and Ranker renormalizes weights instead of treating it as a known zero.
- [ ] Verify RED.
- [ ] Preserve `None` in the source adapter and exclude citation weight for records without citation metadata.
- [ ] Run ranking/source tests.

### Task 6: Verification

- [ ] Run all focused literature tests.
- [ ] Run the full suite, `compileall`, `git diff --check`, and a secret scan.
- [ ] Run one real integrated flow and compare stage timing, Scout paper slots, source failures, and stop reason with the diagnostic baseline.
- [ ] Confirm no commit or push occurred.
