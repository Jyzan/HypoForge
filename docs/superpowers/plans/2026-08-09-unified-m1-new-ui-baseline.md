# Unified M1 and New UI Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the main local HypoForge working tree with teammate commit `d0d5c6e`, then integrate the validated M1 behavior and the backed-up new UI into one tested collaboration branch.

**Architecture:** Keep `d0d5c6e` as the authoritative M2/literature implementation and restore only M1 behavior plus UI/iteration deltas. Shared files such as `pipeline.py`, `strict_contracts.py`, and the M2 query/coverage path are merged semantically so the new literature facade is not replaced by legacy code.

**Tech Stack:** Python 3, Pydantic, LangGraph, FastAPI/SSE, static HTML/CSS/JavaScript, pytest, Git worktrees.

## Global Constraints

- Preserve the current working tree, untracked files, and local `.env` in a timestamped local backup before switching commits.
- Never add `.env`, API keys, output runs, virtual environments, or caches to Git.
- Base the collaboration branch on `origin/fix/generalized-evidence-contracts` at `d0d5c6e`.
- Preserve the new M2 literature facade; do not restore the deleted legacy M2 implementation.
- M1 generates no operational question category; historical `question_type` remains compatibility-only.
- Sub-question count is capped at five without a configurable `max_sub_questions` truncation knob.
- Follow-up routing must call the LLM and fail open to search when uncertain.
- Do not push until targeted tests, full tests, diff checks, and secret checks pass.

---

### Task 1: Create a Recoverable Old-Version Snapshot

**Files:**
- Create: `C:/Users/LIU/Desktop/挑战杯/核心代码/HypoForge-backups/pre-d0d5c6e-unified-<timestamp>/`
- Preserve: current tracked/untracked source, tests, docs, configuration, and `.env`

**Interfaces:**
- Consumes: current working tree at commit `4b70878` plus uncommitted changes
- Produces: a timestamped source snapshot and Git diagnostic files sufficient for recovery

- [ ] Record current branch, HEAD, status, staged diff, unstaged diff, and untracked file list.
- [ ] Copy source/config/test/doc files and `.env`, excluding `.git`, `.venv`, caches, output, and generated knowledge stores.
- [ ] Verify hashes for `.env`, M1 implementation, M1 prompt, UI HTML, and web backend between source and backup.
- [ ] Create a named Git stash including untracked files as a second recovery layer.

### Task 2: Promote the Teammate Version to the Main Working Directory

**Files:**
- Update working tree root: `C:/Users/LIU/Desktop/挑战杯/核心代码/HypoForge`
- Restore ignored local file: `.env`

**Interfaces:**
- Consumes: `origin/fix/generalized-evidence-contracts@d0d5c6e`
- Produces: branch `integration/generalized-evidence-ui-v2` rooted at `d0d5c6e`

- [ ] Switch the main working tree to a new branch from `d0d5c6e`.
- [ ] Verify the working tree is clean and `modules/m2_literature_search.py` is the thin literature facade.
- [ ] Restore `.env` from the verified backup and confirm Git ignores it.
- [ ] Run the untouched teammate test suite; expected result is `637 passed`.

### Task 3: Port the Validated M1 Contract with Tests First

**Files:**
- Modify: `hypoforge/modules/m1_problem_understanding.py`
- Modify: `hypoforge/prompts/m1_prompts.py`
- Modify: `hypoforge/state.py`
- Modify: `hypoforge/display/panels.py`
- Modify: `configs/full_pipeline_m2agentic_qwen37plus_live.yaml`
- Modify: `scripts/smoke_pipeline.py`
- Test: `tests/test_m1_atomicity.py`
- Test: `tests/test_m1_combined.py`
- Test: `tests/test_m1_followup_triage.py`

**Interfaces:**
- Consumes: `PipelineState.input_question`, optional parent `ProblemCard`, graph and follow-up artefacts
- Produces: audited `ProblemCard.task_contract`, at most five atomic sub-questions, and LLM-routed `FollowupRequest.skip_search`

- [ ] Restore the previously passing M1 tests before production files and run them to verify failure against clean `d0d5c6e`.
- [ ] Port candidate decomposition, coverage audit/supplement/merge, source-bound entity extraction/audit, and TaskContract construction.
- [ ] Port mandatory LLM follow-up triage, confidence gate, parent-card reuse, and research-change rebuild.
- [ ] Remove the configurable `max_sub_questions` path while enforcing the constant upper bound of five.
- [ ] Run all M1 tests and verify they pass.

### Task 4: Remove Operational Question-Type Coupling from M2

**Files:**
- Modify: `hypoforge/literature/adapter.py`
- Modify: `hypoforge/literature/models.py`
- Modify: `hypoforge/literature/search/agent.py`
- Modify: `hypoforge/literature/search/coverage.py`
- Modify: `hypoforge/literature/search/query_planner.py`
- Modify: `hypoforge/strict_contracts.py`
- Test: `tests/literature/test_coverage.py`

**Interfaces:**
- Consumes: atomic sub-question, scoped task entities, and domains
- Produces: query/coverage decisions independent of historical `question_type`

- [ ] Restore the question-type independence tests and verify failure on clean teammate behavior.
- [ ] Remove `question_type` from live adapter/agent/planner flow while keeping deserialization compatibility in `SearchState`.
- [ ] Make coverage use the same evidence gate for every scientific domain.
- [ ] Reconcile changes with the new `term_importance`, source-coverage, retention-judge, and strict-facade code from `d0d5c6e`.
- [ ] Run query planner, coverage, adapter, strict-contract, Scout, and ranking tests.

### Task 5: Integrate the New UI and Iteration Backend

**Files:**
- Modify: `hypoforge/web/index.html`
- Modify: `hypoforge/webapp.py`
- Modify: `hypoforge/pipeline.py`
- Preserve: `hypoforge/observability.py`
- Preserve: `hypoforge/display/m2_progress.py`
- Test: `tests/test_observability_webapp.py`
- Test: `tests/test_iteration_routing.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: run snapshots, SSE events, parent run ID, follow-up request, local credentials
- Produces: new-question UI, M1-M6 live progress, details, iteration timeline, follow-up submission, and result history

- [ ] Restore UI/API tests from the new-UI backup and run them against the teammate base to expose missing behavior.
- [ ] Restore the new HTML UI and merge API endpoints without reintroducing legacy M2 configuration.
- [ ] Merge pipeline iteration changes while preserving `M2ProgressReporter`, `bind_event_sink`, and the M2-specific event lifecycle.
- [ ] Confirm credential fields affect only the local run environment and are not serialized into snapshots or Git-tracked configuration.
- [ ] Run UI/API/pipeline tests and perform an inline JavaScript syntax check.

### Task 6: Fix Confirmed Regressions in the New Teammate Commit

**Files:**
- Modify: `hypoforge/literature/search/scout.py`
- Modify: `hypoforge/literature/search/agent.py`
- Test: `tests/literature/test_scout.py`
- Test: `tests/test_m2_progress.py`

**Interfaces:**
- Consumes: LLM entity strings and retention-judge events
- Produces: valid domain-neutral entities and readable event messages

- [ ] Add tests proving unmatched Chinese parentheses and sentence-like Chinese entities are rejected.
- [ ] Verify those tests fail because the current code compares `?` with itself.
- [ ] Restore proper Chinese punctuation/parenthesis checks and readable retention-judge messages.
- [ ] Run Scout and M2 progress tests.

### Task 7: Verify, Commit, and Publish the Shared Baseline

**Files:**
- Review all modified files
- Do not add: `.env`, `output/`, caches, knowledge stores, or `.venv/`

**Interfaces:**
- Consumes: integrated working tree
- Produces: tested remote branch `integration/generalized-evidence-ui-v2`

- [ ] Run focused M1, M2, pipeline, webapp, and iteration tests.
- [ ] Run full `pytest -q` and record the exact result.
- [ ] Run Python compilation, JavaScript syntax validation, `git diff --check`, and secret/path scans.
- [ ] Start the local web service and verify new-run and follow-up endpoints without exposing credentials.
- [ ] Review final `git status`, diff summary, and staged file list.
- [ ] Commit the integrated code with one collaboration-baseline commit and push `integration/generalized-evidence-ui-v2`.
- [ ] Report the backup path, commit ID, branch URL, test result, and instructions for the teammate to branch from it.
