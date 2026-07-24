# M2 Qwen Structured Call Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Query Planner, Scout Reader, and Coverage Evaluator use Qwen3.6 structured output without the OpenAI SDK `thinking` TypeError, while making stage latency observable.

**Architecture:** Keep provider-specific request construction inside `QwenClient`. Send DashScope's non-standard `enable_thinking` through LangChain's `extra_body`, leave standard `response_format` in `model_kwargs`, and request non-thinking mode from all three deterministic structured-generation stages. Record stage elapsed times in the search result without changing the existing per-source timeout contract.

**Tech Stack:** Python 3.11+, asyncio, LangChain `ChatOpenAI`, Pydantic, pytest, pytest-asyncio.

## Global Constraints

- Do not commit or push these changes without explicit user approval.
- Do not print, persist, or copy API keys into the repository.
- Preserve real network failures such as HTTP 429; do not fabricate papers.
- Keep `--timeout` as a per-source timeout in this change.
- Preserve `AbstractReadingWorkflow` as the explicitly documented placeholder.

---

### Task 1: Correct Qwen provider request parameters

**Files:**
- Modify: `hypoforge/tools/qwen_client.py`
- Test: `tests/test_qwen_client.py`

**Interfaces:**
- Consumes: `QwenClient._build_llm(..., disable_thinking: bool)` and `QwenClient.structured_chat(..., disable_thinking: bool)`.
- Produces: `ChatOpenAI` instances with `extra_body={"enable_thinking": False}` when thinking is disabled.

- [ ] **Step 1: Write failing tests**

Patch `ChatOpenAI` with a capturing fake and assert that disabled thinking is sent in `extra_body`, never inside `model_kwargs`, both for `_build_llm` and the formatted structured call.

- [ ] **Step 2: Run tests to verify RED**

Run:

```powershell
uv run --with pytest --with pytest-asyncio --with langchain-openai --with python-dotenv pytest tests/test_qwen_client.py -q
```

Expected: FAIL because current code emits `thinking={"type": "disabled"}` via `model_kwargs`.

- [ ] **Step 3: Implement the minimal fix**

Change provider extras to:

```python
extra_body = {"enable_thinking": False} if disable_thinking else None
```

Pass `extra_body=extra_body` to `ChatOpenAI`. Keep `response_format` in `model_kwargs`.

- [ ] **Step 4: Run tests to verify GREEN**

Run the Task 1 command and expect all selected tests to pass.

### Task 2: Disable thinking for all M2 structured selection stages

**Files:**
- Modify: `hypoforge/literature/search/query_planner.py`
- Verify: `hypoforge/literature/search/scout.py`
- Verify: `hypoforge/literature/search/coverage.py`
- Test: `tests/literature/test_query_planner.py`
- Test: `tests/literature/test_scout.py`
- Test: `tests/literature/test_coverage.py`

**Interfaces:**
- Consumes: each component's existing `client.structured_chat(**kwargs)`.
- Produces: every M2 structured selection stage supplies `disable_thinking=True`.

- [ ] **Step 1: Write or strengthen failing assertions**

Assert `disable_thinking is True` for Query Planner, Scout Reader, and Coverage Evaluator calls. Scout already covers this behavior; add missing Query Planner and Coverage assertions.

- [ ] **Step 2: Run tests to verify RED**

Run:

```powershell
uv run --with pytest --with pytest-asyncio --with langchain-openai --with python-dotenv pytest tests/literature/test_query_planner.py tests/literature/test_scout.py tests/literature/test_coverage.py -q
```

Expected: Query Planner and Coverage assertions fail against current code.

- [ ] **Step 3: Implement minimal call-site changes**

Add `disable_thinking=True` only to the missing structured calls.

- [ ] **Step 4: Run tests to verify GREEN**

Run the Task 2 command and expect all selected tests to pass.

### Task 3: Record stage elapsed time

**Files:**
- Modify: `hypoforge/literature/models.py`
- Modify: `hypoforge/literature/search/agent.py`
- Test: `tests/literature/test_search_agent.py`

**Interfaces:**
- Consumes: the agent's injected monotonic `clock`.
- Produces: `SearchRunResult.stage_elapsed_seconds: dict[str, float]` with cumulative `query_planner`, `source_search`, `paper_deduplicator`, `paper_ranker`, `scout_reader`, and `coverage_evaluator` durations.

- [ ] **Step 1: Write a failing result-contract test**

Use a deterministic advancing clock and assert that all six stage keys exist with non-negative finite values.

- [ ] **Step 2: Run test to verify RED**

Run:

```powershell
uv run --with pytest --with pytest-asyncio --with langchain-openai --with python-dotenv pytest tests/literature/test_search_agent.py -q
```

Expected: FAIL because `SearchRunResult` does not yet expose stage timing.

- [ ] **Step 3: Implement cumulative timing**

Wrap each awaited stage at the agent orchestration boundary with the injected monotonic clock and accumulate elapsed seconds in a local dictionary returned through `SearchRunResult`.

- [ ] **Step 4: Run tests to verify GREEN**

Run the Task 3 command and expect all selected tests to pass.

### Task 4: Verify the real failure path and the complete suite

**Files:**
- No additional production files.

**Interfaces:**
- Consumes: the fixed Qwen client and integrated M2 factory.
- Produces: verification evidence only; no persisted live output.

- [ ] **Step 1: Run a one-paper real Qwen Scout check**

Load credentials process-locally from `C:\Users\LIU\Desktop\code\saber\.env`, call one Scout batch with `qwen3.6-plus`, and print only success metadata.

- [ ] **Step 2: Run focused integration tests**

```powershell
uv run --with pytest --with pytest-asyncio --with langchain-openai --with python-dotenv pytest tests/literature tests/test_qwen_client.py tests/test_run_m2_integrated.py -q
```

- [ ] **Step 3: Run the full suite and static checks**

```powershell
uv run --with pytest --with pytest-asyncio --with langchain-openai --with python-dotenv pytest -q
uv run --with langchain-openai --with python-dotenv python -m compileall -q hypoforge scripts
git diff --check
git diff --cached --check
```

- [ ] **Step 4: Inspect the final diff**

Confirm no secret, unrelated file, commit, or push was introduced.
