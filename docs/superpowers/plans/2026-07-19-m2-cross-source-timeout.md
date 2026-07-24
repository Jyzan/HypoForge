# M2 Cross-Source Coverage and Full-Text Timeout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Search every configured source in the first round and degrade individual slow arXiv PDFs to their real abstracts within a hard 180-second download deadline.

**Architecture:** `QueryPlanner` deterministically adds one query for any configured backend omitted by the model on round one. `ArxivPDFResolver` streams the built-in HTTP response under byte and monotonic-time limits, while `FullTextReadingWorkflow` adds a per-paper resolver deadline and uses source-aware abstract fallback instead of cancelling sibling papers.

**Tech Stack:** Python 3.12, asyncio, urllib, Pydantic v2, pytest, pytest-asyncio.

## Global Constraints

- First-round coverage uses the current `tool_definitions`; do not hard-code a second source list.
- Force all available sources only on round one; later rounds remain gap-driven.
- Preserve query, paper, token, time, and round budgets.
- Default arXiv total download deadline is exactly 180 seconds.
- Default maximum PDF size remains exactly 52,428,800 bytes.
- Ordinary paper-local failure returns real abstract evidence when available and never fabricates content.
- Preserve existing injected backends and PubMed/PMC behavior.
- Do not commit or push.

---

### Task 1: Guarantee all configured sources in the first-round plan

**Files:**
- Modify: `hypoforge/literature/search/query_planner.py`
- Create: `tests/literature/test_query_planner_source_coverage.py`

**Interfaces:**
- Produces: `QueryPlanner._ensure_first_round_coverage(sub_question, queries, unavailable_sources) -> list[SearchQuery]`.
- Consumes: `_tool_definitions`, `_valid_backends`, and existing `_sanitize_query`.

- [ ] **Step 1: Write failing source-coverage tests**

```python
@pytest.mark.asyncio
async def test_first_round_adds_every_missing_configured_source():
    client = FakeClient({"queries": [{
        "text": "cancer immunotherapy", "tool": "pubmed",
        "purpose": "core", "reasoning": "clinical evidence",
    }]})
    planner = QueryPlanner(client, LiteratureSearchTool.TOOL_DEFINITIONS)
    queries = await planner.plan("How does cancer immunotherapy work?",
                                 state=SearchState())
    assert {item.target_source for item in queries} == {
        "pubmed", "semantic_scholar", "arxiv"
    }
    added = [item for item in queries if item.purpose == "cross_source_coverage"]
    assert {item.target_source for item in added} == {"semantic_scholar", "arxiv"}

@pytest.mark.asyncio
async def test_later_round_does_not_force_all_sources():
    state = SearchState(round_index=1, missing_topics={"review evidence"})
    queries = await planner.plan("question", state=state)
    assert [item.target_source for item in queries] == ["arxiv"]

@pytest.mark.asyncio
async def test_unavailable_source_is_not_reintroduced():
    state = SearchState(unavailable_sources={"semantic_scholar"})
    queries = await planner.plan("question", state=state)
    assert "semantic_scholar" not in {item.target_source for item in queries}
```

- [ ] **Step 2: Run tests and confirm RED**

Run: `python -m pytest tests/literature/test_query_planner_source_coverage.py -q`

Expected: first-round target set lacks model-omitted sources.

- [ ] **Step 3: Add deterministic round-one augmentation**

```python
def _ensure_first_round_coverage(self, sub_question, queries,
                                 unavailable_sources):
    output = list(queries)
    selected = {item.target_source for item in output}
    unavailable = set(unavailable_sources)
    for backend in sorted(self._valid_backends - selected - unavailable):
        text = _sanitize_query(sub_question, backend)
        output.append(SearchQuery(
            query_id=f"q-1-coverage-{backend}",
            text=text,
            round_index=0,
            intent=QueryIntent.CORE,
            target_source=backend,
            purpose="cross_source_coverage",
            relation_to_question=(
                f"Guarantees first-round coverage of the {backend} source."
            ),
        ))
    return output
```

Call this method after model rows are validated only when `round_num == 1`. If model output is empty in non-strict mode, existing all-backend fallback remains sufficient and must not be duplicated.

- [ ] **Step 4: Verify focused planner tests**

Run: `python -m pytest tests/literature/test_query_planner_source_coverage.py tests/literature/test_teammate_search_tools.py tests/literature/test_arxiv_search_tool.py -q`

Expected: all tests pass; existing sanitizer and fallback ordering remain unchanged.

---

### Task 2: Enforce a real total deadline while streaming arXiv PDFs

**Files:**
- Modify: `hypoforge/literature/reading/arxiv_resolver.py`
- Modify: `tests/literature/reading/test_arxiv_resolver.py`

**Interfaces:**
- Produces: `ArxivDownloadTimeoutError(TimeoutError)`.
- Produces: `_read_bounded_response(response, *, max_bytes, deadline, clock, chunk_size=65_536) -> bytes`.
- Extends: `ArxivPDFResolver(..., download_timeout_seconds=180.0)`.

- [ ] **Step 1: Write failing bounded-stream tests**

```python
def test_stream_reader_enforces_total_deadline_between_chunks():
    clock = FakeClock([0.0, 40.0, 100.0, 181.0])
    response = FakeResponse([b"%PDF-", b"a" * 10, b"b" * 10])
    with pytest.raises(ArxivDownloadTimeoutError, match="180"):
        _read_bounded_response(response, max_bytes=1000, deadline=180.0,
                               clock=clock)

def test_stream_reader_rejects_content_length_before_reading():
    response = FakeResponse([], headers={"Content-Length": "1001"})
    with pytest.raises(ValueError, match="size limit"):
        _read_bounded_response(response, max_bytes=1000, deadline=180.0,
                               clock=lambda: 0.0)
    assert response.read_calls == 0

@pytest.mark.asyncio
async def test_default_download_timeout_degrades_without_partial_cache(...):
    document = await resolver.resolve(paper)
    assert document.content_level is ContentLevel.ABSTRACT
    assert "download deadline" in document.retrieval_error
    assert not list(tmp_path.rglob("*.tmp"))
    assert not list(tmp_path.rglob("paper.pdf"))
```

- [ ] **Step 2: Run resolver tests and confirm RED**

Run: `python -m pytest tests/literature/reading/test_arxiv_resolver.py -q`

Expected: bounded reader and timeout exception imports fail.

- [ ] **Step 3: Implement chunked deadline and size enforcement**

```python
class ArxivDownloadTimeoutError(TimeoutError):
    pass

def _read_bounded_response(response, *, max_bytes, deadline, clock,
                           chunk_size=65_536):
    length = response.headers.get("Content-Length")
    if length and int(length) > max_bytes:
        raise ValueError("arXiv PDF exceeds configured size limit")
    chunks, used = [], 0
    while True:
        if clock() >= deadline:
            raise ArxivDownloadTimeoutError("arXiv PDF exceeded 180 second download deadline")
        chunk = response.read(chunk_size)
        if not chunk:
            break
        used += len(chunk)
        if used > max_bytes:
            raise ValueError("arXiv PDF exceeds configured size limit")
        chunks.append(chunk)
    return b"".join(chunks)
```

`_default_fetch` records `started = time.monotonic()`, opens with `min(socket_timeout, 30.0, download_timeout_seconds)`, and passes `started + download_timeout_seconds` into the bounded reader. `ArxivPDFResolver.resolve` passes its new total deadline to the default backend; injected two-argument async backends remain unchanged.

- [ ] **Step 4: Verify resolver GREEN and cache regressions**

Run: `python -m pytest tests/literature/reading/test_arxiv_resolver.py tests/literature/reading/test_resolver.py -q`

Expected: all tests pass; valid PDFs are still cached atomically.

---

### Task 3: Isolate resolver timeout per Final-K paper

**Files:**
- Modify: `hypoforge/literature/reading/resolver.py`
- Modify: `hypoforge/literature/reading/routing.py`
- Modify: `hypoforge/literature/reading/workflow.py`
- Modify: `hypoforge/literature/integrated.py`
- Modify: `tests/literature/reading/test_workflow.py`
- Modify: `tests/literature/test_integrated_factory.py`

**Interfaces:**
- Adds: `PMCFulltextResolver.resolve_abstract(paper) -> DocumentRecord`.
- Extends: `RoutingFulltextResolver.resolve_abstract` to route by paper source.
- Extends: `FullTextReadingWorkflow(..., resolver_timeout_seconds=210.0)`.
- Extends: `build_integrated_search_adapter(..., arxiv_download_timeout_seconds=180.0, resolver_timeout_seconds=210.0)`.

- [ ] **Step 1: Write failing sibling-isolation test**

```python
@pytest.mark.asyncio
async def test_one_resolver_timeout_degrades_without_cancelling_siblings():
    workflow = FullTextReadingWorkflow(
        resolver=SlowOneResolver(), parser=FakeParser(),
        retriever=FakeRetriever(), reader=FakeReader(),
        store=InMemoryChunkStore(), resolver_timeout_seconds=0.02,
    )
    results = await workflow.run("question", [slow_paper, fast_paper])
    assert [item.paper_id for item in results] == ["slow", "fast"]
    assert results[0].degraded_to_abstract is True
    assert "timed out" in " ".join(results[0].errors)
    assert results[1].summary == "Summary for fast"
```

Add a routing test proving PubMed timeout fallback uses PMC's abstract helper and arXiv timeout fallback uses arXiv's helper.

- [ ] **Step 2: Run workflow tests and confirm RED**

Run: `python -m pytest tests/literature/reading/test_workflow.py tests/literature/reading/test_arxiv_resolver.py -q`

Expected: constructor rejects the new timeout argument or the slow paper cancels/returns no abstract.

- [ ] **Step 3: Add source-aware abstract fallback and per-paper wait**

```python
document = await self._measure(
    timings,
    "fulltext_resolver",
    lambda: asyncio.wait_for(
        self.resolver.resolve(paper),
        timeout=self.resolver_timeout_seconds,
    ),
)
```

Catch `asyncio.TimeoutError` inside `_read_one`, call `resolver.resolve_abstract(paper)`, append `"fulltext_resolver timed out"`, then continue through parsing, retrieval, and reading. Do not propagate this paper-local timeout. Keep `asyncio.CancelledError` propagation for the 600-second global workflow deadline.

`RoutingFulltextResolver.resolve_abstract` must choose the same source branch as `resolve`; `PMCFulltextResolver.resolve_abstract` exposes its existing abstract materialization without a network call.

- [ ] **Step 4: Wire independent timeout settings**

```python
arxiv_resolver = ArxivPDFResolver(
    ..., timeout_seconds=source_timeout_seconds,
    download_timeout_seconds=arxiv_download_timeout_seconds,
)
reading_workflow = FullTextReadingWorkflow(
    ..., resolver_timeout_seconds=resolver_timeout_seconds,
)
```

Validate both new settings are positive. Defaults must be 180 and 210 seconds respectively.

- [ ] **Step 5: Verify reading and factory GREEN**

Run: `python -m pytest tests/literature/reading tests/literature/test_integrated_factory.py -q`

Expected: all tests pass; slow sibling degrades and fast sibling completes.

---

### Task 4: Full regression and bounded live validation

**Files:**
- Modify: `hypoforge/literature/README.md`

- [ ] **Step 1: Document guaranteed source coverage and timeout behavior**

Add an operational note stating that every first round searches all configured sources, arXiv downloads have a 180-second total deadline, and a slow PDF degrades independently to its real abstract.

- [ ] **Step 2: Run deterministic verification**

Run: `python -m pytest -q`

Expected: zero failures.

Run: `python -m compileall -q hypoforge` and `git diff --check`.

Expected: exit code zero; CRLF notices are allowed, whitespace errors are not.

- [ ] **Step 3: Run bounded live search verification**

Use Qwen `qwen3.6-plus` with the prior technical question. Assert the first-round trace contains `pubmed`, `semantic_scholar`, and `arxiv`, even if a provider returns a real network error.

- [ ] **Step 4: Run bounded Final-K reading verification**

Run the integrated pipeline with `final_k=3`. Record total time, each paper's content level, `degraded_to_abstract`, errors, chunks, and knowledge entries. Expected: the 5.07 MB slow PDF becomes an abstract result around 180 seconds while smaller or cached PDFs continue; the workflow returns a result list rather than raising global `TimeoutError`.

- [ ] **Step 5: Preserve working tree for user review**

Run: `git status --short`.

Do not commit, push, merge, or clean the worktree.
