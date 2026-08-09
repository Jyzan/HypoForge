# M4 Cancellable Contract Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent M4 semantic contract checks from hanging indefinitely and make the web stop action interrupt an active module promptly.

**Architecture:** Split M4 contract validation into deterministic synchronous validation followed by one observable asynchronous batch semantic audit with a 60-second deadline. Wrap every pipeline module coroutine in a cancellation-aware waiter that polls the existing `threading.Event` and cancels the active task while preserving the last completed state.

**Tech Stack:** Python 3.11+, asyncio, Pydantic, pytest, existing HypoForge observability and PipelineRunner.

## Global Constraints

- Do not commit or push this repair before the user completes another real web test.
- Preserve completed M1-M3 state when a running M4 is cancelled.
- Semantic contract failures remain fail-closed; timeout must never become silent acceptance.
- Cancellation should be observed within 1 second.

---

### Task 1: Async observable M4 semantic contract audit

**Files:**
- Modify: `hypoforge/modules/m4_hypothesis_generation.py`
- Test: `tests/test_m4_hypothesis_generation.py`
- Test: `tests/test_strict_contracts.py`

**Interfaces:**
- Produces: `M4HypothesisGeneration._audit_context_contract_semantics(state, candidates) -> tuple[list[HypothesisCard], list[dict[str, Any]]]`
- Produces: constructor option `semantic_alignment_timeout_seconds: float = 60.0`

- [ ] Write a failing test using a blocking fake client and assert the audit raises a timeout error rather than hanging.
- [ ] Run the focused test and verify it fails because M4 still uses the synchronous semantic checker.
- [ ] Write a failing test asserting three candidates are judged by one `hypothesis_contract_auditor` structured call and all candidate IDs require one valid verdict.
- [ ] Implement the async batch audit, observable event wrapper, strict verdict validation and timeout.
- [ ] Route both initial candidates and contract-repair candidates through deterministic validation followed by the async audit.
- [ ] Run focused M4 and strict-contract tests.

### Task 2: Cancel a currently running module

**Files:**
- Modify: `hypoforge/pipeline.py`
- Test: `tests/test_run_cancel.py`

**Interfaces:**
- Produces: `PipelineRunner._await_module_or_cancel(awaitable, module_name)` used by the module node wrapper.

- [ ] Write a failing test with an M4 coroutine waiting on an event; set the cancellation flag after M4 starts and assert the task is cancelled within one second.
- [ ] Run the focused test and verify the existing between-module-only cancellation behavior fails it.
- [ ] Implement cancellation-aware module awaiting with a 0.2-second poll interval, task cancellation, `module_cancelled` event and `PipelineCancelled` propagation.
- [ ] Run cancellation endpoint and pipeline cancellation tests.

### Task 3: Verify integration and restore the local test page

**Files:**
- Runtime data only: `output/ui_runs/ui-20260809-165216-261ce3/manifest.json`

**Interfaces:**
- Consumes: the repaired M4 and PipelineRunner cancellation behavior.
- Produces: a local web service on `http://127.0.0.1:7860/` ready for another user test.

- [ ] Mark the force-terminated historical run as cancelled so it no longer appears live.
- [ ] Run focused M4, cancellation, webapp and observability tests.
- [ ] Run the full pytest suite, Python compile check and inline frontend JavaScript syntax check.
- [ ] Start the local web service and verify `/` plus `/api/runs` return successfully.
- [ ] Leave all source changes uncommitted for user validation.
