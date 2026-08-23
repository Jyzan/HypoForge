# M4→M2 Evidence-Gap Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ensure every M4 and M6 evidence-gap request selected for an M2 revisit is actually searched before its status advances, including when both gap families coexist.

**Architecture:** Build a deterministic internal `_SearchTask` list in `AgenticM2Adapter`. A task carries one normalized question and all M4/M6 origin IDs mapped to it; the adapter updates only origin IDs whose task produced a real search outcome. Pipeline topology remains bounded by the existing round limits.

**Tech Stack:** Python 3.12, asyncio, dataclasses, Pydantic state models, pytest.

**Spec:** `docs/superpowers/specs/2026-08-23-hypoforge-continuous-optimization-design.md`

## Global Constraints

- Preserve all public M1–M6 state DTOs and the `AgenticM2Adapter.__call__` return contract.
- Do not advance any gap whose mapped search task failed or was not executed.
- Search both M4 and M6 gap families when they coexist; deduplicate identical question text without losing origin IDs.
- Preserve existing M4/M6 loop bounds and cancellation behavior.
- Run each focused test after RED and GREEN, then run the full suite.
- Do not push the branch.

---

### Task 1: Reproduce coexisting-gap state corruption

**Files:**
- Modify: `scripts/test_pipeline.py`
- Test: `scripts/test_pipeline.py`

**Interfaces:**
- Consumes: `AgenticM2Adapter(state) -> dict[str, Any]`, `EvidenceGap`, `EvidenceGapRequest`.
- Produces: regression expectations for simultaneous M4/M6 tasks and failed-task state preservation.

- [ ] **Step 1: Add a failing coexistence test**

Create a later-round `PipelineState` with one open M6 `EvidenceGap` and one pending M4 `EvidenceGapRequest` using different questions. Use `_ConcurrentFreshSearchAgent` and `EchoReadingWorkflow(with_evidence=True)`. Assert that both questions occur in the search export, the M4 request becomes `searched`, and the M6 gap becomes `pending_grounding`.

- [ ] **Step 2: Run the coexistence test and verify RED**

Run:

```powershell
python -m pytest -q scripts/test_pipeline.py -k "coexisting_m4_and_m6_gaps_are_both_searched"
```

Expected: FAIL because current code selects M6 questions with `if revisit_gaps` and never searches the M4 question.

- [ ] **Step 3: Add a failing partial-failure test**

Use two distinct questions and configure `_ConcurrentFreshSearchAgent` to fail only the M4 question. Assert that the successful M6 gap advances to `pending_grounding`, while the failed M4 request remains `pending` with unchanged attempts.

- [ ] **Step 4: Run the partial-failure test and verify RED**

Run:

```powershell
python -m pytest -q scripts/test_pipeline.py -k "failed_m4_gap_task_stays_pending_when_m6_task_succeeds"
```

Expected: FAIL because current code marks every pending M4 request `searched` even though its question was never run.

### Task 2: Build origin-aware M2 search tasks

**Files:**
- Modify: `hypoforge/literature/adapter.py`
- Test: `scripts/test_pipeline.py`

**Interfaces:**
- Produces: private immutable `_SearchTask(question: str, m4_gap_ids: tuple[str, ...], m6_gap_ids: tuple[str, ...])`.
- Produces: `AgenticM2Adapter._build_search_tasks(state, pending_gaps, revisit_gaps) -> list[_SearchTask]`.
- Consumes: existing `_gap_as_sub_question` and ProblemCard subquestions.

- [ ] **Step 1: Implement `_SearchTask` and deterministic task construction**

Normalize question identity with collapsed whitespace plus `casefold()`. Add M6 tasks in state order, then merge/append M4 tasks, aggregating every origin ID. When neither gap family is active, create one origin-free task per ProblemCard subquestion or the original question.

- [ ] **Step 2: Execute tasks instead of the mutually exclusive question branches**

Set `sub_questions = [task.question for task in search_tasks]` and preserve the existing concurrent worker, ordering, timeout, cancellation and partial-failure behavior.

- [ ] **Step 3: Advance only executed M4 origins**

Build `searched_m4_ids` from `_SearchTask.m4_gap_ids` only for task indices present in `outcomes`. Update matching requests to `searched`, increment attempts, and append only that task's executed queries. Leave all other requests byte-for-byte equivalent via `model_copy(deep=True)`. Increment `evidence_gap_search_rounds` only when `searched_m4_ids` is non-empty.

- [ ] **Step 4: Advance only grounded M6 origins**

Build `grounded_m6_ids` from tasks whose `export_run` contains evidence or knowledge entries. Change only matching open gaps to `pending_grounding`.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run:

```powershell
python -m pytest -q scripts/test_pipeline.py -k "coexisting_m4_and_m6_gaps_are_both_searched or failed_m4_gap_task_stays_pending_when_m6_task_succeeds or revisit_turns_open_gap_into_subquestion or no_open_gap_runs_full_flow"
```

Expected: all selected tests PASS.

### Task 3: Verify one task can satisfy both origin families

**Files:**
- Modify: `scripts/test_pipeline.py`
- Test: `scripts/test_pipeline.py`

**Interfaces:**
- Consumes: `_SearchTask` deduplication through public `AgenticM2Adapter.__call__` behavior.
- Produces: regression guarantee that one identical question is searched once and advances both matching origin records.

- [ ] **Step 1: Add the shared-question test**

Create one M4 request and one M6 gap that resolve to exactly the same question. Assert `len(agent.calls) == 1`, M4 becomes `searched`, M6 becomes `pending_grounding`, and the export appends one run.

- [ ] **Step 2: Run the test**

Run:

```powershell
python -m pytest -q scripts/test_pipeline.py -k "identical_m4_and_m6_gap_question_runs_once"
```

Expected: PASS after Task 2.

### Task 4: Prove the runner-level M4 loop

**Files:**
- Modify: `scripts/test_pipeline.py`
- Test: `scripts/test_pipeline.py`

**Interfaces:**
- Consumes: `PipelineRunner`, `ModuleRegistry.build_all`, conditional `_route_after_m4` graph edge.
- Produces: an async fake-module end-to-end test asserting module call order and final gap state.

- [ ] **Step 1: Add stateful fake modules**

Construct M1–M6 fake modules. First M4 call returns one pending `EvidenceGapRequest`; M2 converts it to searched; M3 converts it to indexed; second M4 returns it exhausted plus a hypothesis; M5 returns a minimal plan; M6 returns a terminal patch. Append every call name to a shared list.

- [ ] **Step 2: Run the runner test**

Run:

```powershell
python -m pytest -q scripts/test_pipeline.py -k "runner_executes_m4_m2_m3_m4_gap_loop"
```

Expected call order:

```python
["m1", "m2", "m3", "m4", "m2", "m3", "m4", "m5", "m6"]
```

- [ ] **Step 3: Run the full test suite**

Run:

```powershell
python -m pytest -q
```

Expected: all tests PASS with no new warnings or unhandled tasks.

- [ ] **Step 4: Commit locally**

```powershell
git add hypoforge/literature/adapter.py scripts/test_pipeline.py
git commit -m "fix(m2): preserve origin state across evidence-gap revisits"
```
