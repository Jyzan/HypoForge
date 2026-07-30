# M2 Iterative Search Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic, dependency-injected iterative literature Search Agent with budgets, graceful source degradation, structured results, and a tested adapter boundary for the existing M2 Pipeline.

**Architecture:** A pure-Python asynchronous orchestrator calls public Tool protocols in a bounded loop. Budget and stopping rules live in a small pure module, while an unregistered `AgenticM2Adapter` converts search-plus-reading outputs to existing `LiteratureResult` objects without changing the legacy default path.

**Tech Stack:** Python 3.12, asyncio, Pydantic 2, pytest, pytest-asyncio.

## Global Constraints

- Keep `hypoforge/modules/m2_literature_search.py` legacy behavior unchanged by default.
- Do not call real literature APIs in the default test suite.
- Do not import one Tool's private implementation from another component.
- Do not store full paper text in Agent state or Git.
- Do not commit, push, merge, rebase, or force-push during implementation; stop with an uncommitted reviewable diff.
- Every production behavior follows RED → GREEN → REFACTOR with a focused test command.

---

### Task 1: Extend shared contracts for iterative execution

**Files:**
- Modify: `hypoforge/literature/models.py`
- Modify: `hypoforge/literature/protocols.py`
- Modify: `hypoforge/literature/__init__.py`
- Modify: `tests/literature/test_models.py`
- Modify: `tests/literature/test_protocols.py`

**Interfaces:**
- Consumes: existing `SearchBudget`, `SearchState`, `SearchRunResult`, `QueryPlannerProtocol`.
- Produces: `RemainingSearchBudget`; state-aware `QueryPlannerProtocol.plan()`; `ReadingExtractionWorkflowProtocol.run()`; trace fields required by the Agent.

- [ ] **Step 1: Write failing model tests**

Add tests asserting that:

```python
def test_remaining_budget_allows_exhausted_dimensions() -> None:
    remaining = RemainingSearchBudget(
        max_rounds=0,
        max_queries=0,
        max_papers=0,
        max_tokens=0,
        max_seconds=0,
    )
    assert remaining.max_queries == 0


def test_search_state_tracks_iterative_usage() -> None:
    state = SearchState(
        queries_executed=2,
        unique_papers_seen=7,
        estimated_tokens_used=120,
        elapsed_seconds=1.5,
        consecutive_low_gain_rounds=1,
        consecutive_no_result_rounds=0,
    )
    assert state.queries_executed == 2
    assert state.elapsed_seconds == 1.5


def test_search_result_carries_agent_trace() -> None:
    result = SearchRunResult(
        sub_question="question",
        source_result_counts={"pubmed": 3},
        reused_paper_ids=["paper-1"],
        final_state=SearchState(),
    )
    assert result.source_result_counts == {"pubmed": 3}
    assert result.final_state is not None
```

- [ ] **Step 2: Run focused models tests and verify RED**

Run: `.\.venv\Scripts\python.exe -m pytest tests/literature/test_models.py -q`

Expected: collection or assertion failure because `RemainingSearchBudget` and the new trace fields do not exist.

- [ ] **Step 3: Implement minimal model changes**

Add:

```python
class RemainingSearchBudget(LiteratureModel):
    max_rounds: int = Field(default=0, ge=0)
    max_queries: int = Field(default=0, ge=0)
    max_papers: int = Field(default=0, ge=0)
    max_tokens: int = Field(default=0, ge=0)
    max_seconds: float = Field(default=0.0, ge=0.0)
```

Add `round_index: int = Field(default=0, ge=0)` to `SearchQuery`. Change `SearchState.remaining_budget` to `RemainingSearchBudget` and add the six non-negative usage/counter fields from the design. Add the following compatible defaults to `SearchRunResult`:

```python
source_result_counts: Dict[str, int] = Field(default_factory=dict)
scout_notes: List[ScoutNote] = Field(default_factory=list)
reused_paper_ids: List[str] = Field(default_factory=list)
final_state: Optional[SearchState] = None
```

The field is optional for backward-compatible deserialization, but every `IterativeSearchAgent.run()` result must populate it.

- [ ] **Step 4: Run focused models tests and verify GREEN**

Run: `.\.venv\Scripts\python.exe -m pytest tests/literature/test_models.py -q`

Expected: all model tests pass.

- [ ] **Step 5: Write failing protocol tests**

Define a concrete planner accepting `state: SearchState | None = None` and a concrete reading workflow returning `list[PaperReadingResult]`. Add `ReadingExtractionWorkflowProtocol` to the abstract-protocol list.

- [ ] **Step 6: Run focused protocol tests and verify RED**

Run: `.\.venv\Scripts\python.exe -m pytest tests/literature/test_protocols.py -q`

Expected: import/signature failure because the reading workflow protocol and state argument do not exist.

- [ ] **Step 7: Implement protocol changes and public exports**

Use:

```python
class ReadingExtractionWorkflowProtocol(ABC):
    tool_name = "reading_extraction_workflow"

    @abstractmethod
    async def run(
        self,
        sub_question: str,
        papers: Sequence[PaperRecord],
    ) -> List[PaperReadingResult]:
        ...
```

Extend `QueryPlannerProtocol.plan()` with `state: SearchState | None = None`. Export both new public types from `hypoforge.literature`.

- [ ] **Step 8: Verify Task 1 and inspect the diff**

Run: `.\.venv\Scripts\python.exe -m pytest tests/literature/test_models.py tests/literature/test_protocols.py -q`

Run: `git diff --check`

Expected: focused tests pass and diff check exits 0. Do not commit.

---

### Task 2: Implement deterministic budget and stopping rules

**Files:**
- Create: `hypoforge/literature/search/budget.py`
- Create: `tests/literature/test_budget.py`

**Interfaces:**
- Consumes: `SearchBudget`, `SearchState`, `CoverageReport`, `StopReason`.
- Produces: `estimate_tokens(text)`, `calculate_remaining(budget, state)`, `choose_stop_reason(...)`.

- [ ] **Step 1: Write failing pure-function tests**

Cover these exact rules:

```python
def test_calculate_remaining_reaches_zero_without_validation_error() -> None:
    budget = SearchBudget(max_rounds=2, max_queries=3, max_papers=4, max_tokens=5, max_seconds=6)
    state = SearchState(
        round_index=2,
        queries_executed=3,
        unique_papers_seen=4,
        estimated_tokens_used=5,
        elapsed_seconds=6,
    )
    assert calculate_remaining(budget, state).model_dump() == {
        "max_rounds": 0,
        "max_queries": 0,
        "max_papers": 0,
        "max_tokens": 0,
        "max_seconds": 0.0,
    }


def test_coverage_has_highest_stop_priority() -> None:
    reason = choose_stop_reason(
        coverage=CoverageReport(sufficient=True),
        budget=SearchBudget(max_rounds=1),
        state=SearchState(round_index=1),
        all_queries_failed=True,
        has_candidates=False,
    )
    assert reason is StopReason.COVERAGE_SATISFIED


def test_all_failed_without_candidates_is_error() -> None:
    reason = choose_stop_reason(
        coverage=CoverageReport(),
        budget=SearchBudget(max_rounds=1),
        state=SearchState(round_index=1),
        all_queries_failed=True,
        has_candidates=False,
    )
    assert reason is StopReason.ERROR
```

Add one focused test for each remaining stop reason and a deterministic token-estimator test.

- [ ] **Step 2: Run budget tests and verify RED**

Run: `.\.venv\Scripts\python.exe -m pytest tests/literature/test_budget.py -q`

Expected: import failure because `budget.py` does not exist.

- [ ] **Step 3: Implement pure functions**

`estimate_tokens` returns `0` for empty text and otherwise `max(1, math.ceil(len(text) / 4))`. `calculate_remaining` clamps every dimension at zero. `choose_stop_reason` implements the approved priority and returns `None` when another round is permitted.

Use keyword-only arguments for `choose_stop_reason`, including configurable `no_result_round_limit=2` and `low_gain_round_limit=2`.

- [ ] **Step 4: Run budget tests and verify GREEN**

Run: `.\.venv\Scripts\python.exe -m pytest tests/literature/test_budget.py -q`

Expected: all tests pass.

- [ ] **Step 5: Refactor without behavior change**

Keep each stop test in a small private predicate only if it makes priority order clearer. Re-run the focused suite and `git diff --check`. Do not commit.

---

### Task 3: Implement the happy-path iterative Agent

**Files:**
- Create: `tests/literature/fakes.py`
- Create: `tests/literature/test_search_agent.py`
- Create: `hypoforge/literature/search/agent.py`
- Modify: `hypoforge/literature/search/__init__.py`
- Modify: `hypoforge/literature/__init__.py`

**Interfaces:**
- Consumes: all search Tool protocols, shared models, budget helpers.
- Produces: public `IterativeSearchAgent.run(...) -> SearchRunResult`.

- [ ] **Step 1: Build protocol-conforming offline fakes**

Create deterministic fakes that record received arguments and return configured values. Fakes must subclass the actual abstract protocols; do not use `unittest.mock` for core behavior.

- [ ] **Step 2: Write a failing one-round success test**

The test supplies two sources, overlapping paper IDs, a deduplicator, ranker, scout reader and sufficient coverage evaluator. Assert:

- both sources were called;
- duplicate papers appear once;
- result stops with `COVERAGE_SATISFIED`;
- `source_result_counts`, `queries`, `scout_notes`, candidates and final papers are populated;
- `final_state.round_index == 1`.

- [ ] **Step 3: Run the focused test and verify RED**

Run: `.\.venv\Scripts\python.exe -m pytest tests/literature/test_search_agent.py::test_agent_returns_complete_result_when_first_round_is_sufficient -q`

Expected: import failure because `IterativeSearchAgent` does not exist.

- [ ] **Step 4: Implement the minimal one-round Agent**

Constructor signature:

```python
def __init__(
    self,
    *,
    query_planner: QueryPlannerProtocol,
    sources: Sequence[LiteratureSourceProtocol],
    deduplicator: PaperDeduplicatorProtocol,
    ranker: PaperRankerProtocol,
    scout_reader: ScoutReaderProtocol,
    coverage_evaluator: CoverageEvaluatorProtocol,
    final_k: int = 10,
    candidate_limit: int = 30,
    per_query_limit: int = 20,
    source_timeout_seconds: float = 30.0,
    min_new_papers: int = 1,
    no_result_round_limit: int = 2,
    low_gain_round_limit: int = 2,
    clock: Callable[[], float] = time.monotonic,
    token_estimator: Callable[[str], int] = estimate_tokens,
) -> None:
```

Validate positive limits and unique non-empty `source_name` values. `run()` creates an empty `SearchState`, calls the Planner with that state, annotates queries with round 1, dispatches searches concurrently, deduplicates, ranks, Scouts, evaluates coverage and returns a populated result.

- [ ] **Step 5: Run the one-round test and verify GREEN**

Run the same focused command; expected PASS.

- [ ] **Step 6: Write a failing two-round gap test**

The first coverage result contains `missing_topics=["negative evidence"]`; assert that the Planner's second call receives a state containing that gap, the second-round query has `round_index == 2`, and final output contains queries from both rounds.

- [ ] **Step 7: Run the two-round test and verify RED**

Expected: assertion failure because the Agent currently exits after one round.

- [ ] **Step 8: Implement the bounded loop**

Loop until `choose_stop_reason` returns a reason. Normalize query history keys as `(target_source.casefold(), " ".join(text.casefold().split()))`; remove repeated queries before dispatch. Maintain candidates by stable `paper_id`, pass the existing catalog plus accumulated candidates to the Deduplicator, and use returned canonical records for incoming hits.

Update known terms from all Scout notes, covered/missing topics from Coverage, and set covered bucket counts to at least one. Recompute remaining budget after every round.

- [ ] **Step 9: Verify happy path and iteration**

Run: `.\.venv\Scripts\python.exe -m pytest tests/literature/test_search_agent.py -q`

Expected: current Agent tests pass. Run `git diff --check`. Do not commit.

---

### Task 4: Add degradation, reuse and every budget boundary

**Files:**
- Modify: `tests/literature/test_search_agent.py`
- Modify: `tests/literature/fakes.py`
- Modify: `hypoforge/literature/search/agent.py`

**Interfaces:**
- Consumes: Task 3 Agent.
- Produces: complete error and stop behavior required by the design.

- [ ] **Step 1: Write failing degradation tests**

Add separate tests for:

- one source raises while a second succeeds;
- every scheduled source raises;
- unknown `target_source`;
- one source exceeds `source_timeout_seconds`;
- Planner, Deduplicator, Ranker, Scout and Coverage core failures.

Assert partial success remains usable, while unrecoverable core failures return `StopReason.ERROR`, preserve accumulated candidates, and include short sanitized messages.

- [ ] **Step 2: Run degradation tests and verify RED**

Run the named tests with `pytest -k "source or planner or deduplicator or ranker or scout or coverage"`; expected failures for missing handling.

- [ ] **Step 3: Implement isolated search failures**

Wrap source calls with `asyncio.wait_for`, gather with `return_exceptions=True`, and append failures through an order-preserving helper. Wrap each core orchestration stage and return an error result without invoking legacy/stub fallback.

- [ ] **Step 4: Run degradation tests and verify GREEN**

Expected: all degradation tests pass.

- [ ] **Step 5: Write failing budget and reuse tests**

Add separate tests for query truncation, max rounds, unique-paper budget, estimated Token budget, deterministic fake-clock time budget, two empty rounds, two low-gain rounds, duplicate query suppression, cross-round paper deduplication, and reuse of an `existing_papers` canonical record.

- [ ] **Step 6: Run budget/reuse tests and verify RED**

Expected: each new test fails for its missing boundary behavior rather than fixture errors.

- [ ] **Step 7: Implement budget/reuse behavior**

Truncate planned queries before dispatch, count unique canonical papers, update Token usage before budget evaluation, and use the injected clock. Reused IDs are the intersection between returned canonical candidate IDs and the supplied catalog IDs. Preserve Ranker order when enforcing paper and Final-K limits.

- [ ] **Step 8: Verify Task 4**

Run: `.\.venv\Scripts\python.exe -m pytest tests/literature/test_search_agent.py tests/literature/test_budget.py -q`

Expected: all tests pass. Run `git diff --check`. Do not commit.

---

### Task 5: Implement the agentic M2 adapter and explicit configuration seam

**Files:**
- Create: `hypoforge/literature/adapter.py`
- Create: `tests/literature/test_adapter.py`
- Modify: `hypoforge/config.py`
- Modify: `hypoforge/modules/m2_literature_search.py`
- Modify: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `IterativeSearchAgent`, `ReadingExtractionWorkflowProtocol`, `ProblemCard`, `EvidenceLinkedKnowledge`.
- Produces: `AgenticM2Adapter.__call__(PipelineState) -> dict[str, Any]` with existing `literature_results` output.

- [ ] **Step 1: Write failing config tests**

Assert `PipelineConfig.from_defaults().search.implementation == "legacy"`, valid `agentic` is accepted, and any other value raises `ValidationError`.

- [ ] **Step 2: Run config tests and verify RED**

Run the focused tests; expected failure because `SearchConfig.implementation` does not exist.

- [ ] **Step 3: Implement the config field and propagation**

Add:

```python
implementation: Literal["legacy", "agentic"] = "legacy"
```

Change `PipelineConfig.get_module_kwargs("m2")` so it starts from a copy of `module_overrides.m2.kwargs` and then calls `kwargs.setdefault("implementation", self.search.implementation)`. Other module names retain the current behavior. Do not change existing YAML files, so every checked-in configuration continues to select legacy.

- [ ] **Step 4: Run config tests and verify GREEN**

Expected: focused config tests pass and legacy remains default.

- [ ] **Step 5: Write a failing Adapter test**

Use a fake Search Agent that returns Final-K papers and a fake Reading Workflow that returns `PaperReadingResult` containing evidence-linked entries. Assert the Adapter emits one existing `LiteratureResult` per sub-question, maps IDs/types/content/confidence/entities, and fills `source_paper_id` and title from Final-K.

- [ ] **Step 6: Run Adapter test and verify RED**

Run: `.\.venv\Scripts\python.exe -m pytest tests/literature/test_adapter.py -q`

Expected: import failure because `AgenticM2Adapter` does not exist.

- [ ] **Step 7: Implement the unregistered Adapter**

`AgenticM2Adapter` implements `ModuleProtocol` but has no `@ModuleRegistry.register` decorator, so importing it cannot replace legacy M2. It accepts `search_agent`, `reading_workflow`, and optional `SearchBudget`. Its `__call__` reads `ProblemCard`, runs both workflows for each sub-question, maps entries, and returns `{"literature_results": results}`. Its input/output fields remain `problem_card` and `literature_results`.

If a search result stops with `ERROR`, or the Reading Workflow raises, propagate a descriptive exception to the existing Pipeline wrapper. Empty but non-error search results create an empty `LiteratureResult`, not fabricated knowledge.

- [ ] **Step 8: Write failing M2 routing tests**

Test these three cases directly on `M2LiteratureSearch`:

```python
async def test_m2_legacy_remains_default() -> None:
    module = M2LiteratureSearch()
    assert module.implementation == "legacy"


async def test_m2_agentic_delegates_to_injected_adapter() -> None:
    adapter = FakeAgenticAdapter()
    module = M2LiteratureSearch(implementation="agentic", agentic_adapter=adapter)
    result = await module(PipelineState(input_question="question"))
    assert adapter.calls == 1
    assert result == {"literature_results": []}


async def test_m2_agentic_requires_adapter() -> None:
    module = M2LiteratureSearch(implementation="agentic")
    with pytest.raises(RuntimeError, match="AgenticM2Adapter"):
        await module(PipelineState(input_question="question"))
```

Run the focused tests and verify they fail because the routing fields do not exist.

- [ ] **Step 9: Implement the minimal M2 route**

Add `implementation: str = "legacy"` and `agentic_adapter: ModuleProtocol | None = None` to the constructor, validate the implementation value, and place this guard at the beginning of `__call__` before the existing legacy code:

```python
if self.implementation == "agentic":
    if self.agentic_adapter is None:
        raise RuntimeError(
            "search.implementation='agentic' requires an injected AgenticM2Adapter"
        )
    return await self.agentic_adapter(state, config)
```

Do not change the remainder of the legacy method.

- [ ] **Step 10: Verify Adapter, routing and legacy isolation**

Run Adapter tests and existing Pipeline tests. Assert `ModuleRegistry.get("m2")` remains `M2LiteratureSearch` after importing the Adapter. Run `git diff --check`. Do not commit.

---

### Task 6: Documentation and full verification

**Files:**
- Modify: `hypoforge/literature/README.md`
- Modify: `tests/literature/fixtures/README.md` only if fake-data guidance needs clarification.

**Interfaces:**
- Consumes: all implemented public APIs.
- Produces: collaboration instructions for Tool authors and the future real dependency factory.

- [ ] **Step 1: Document the concrete integration contract**

Document constructor injection, Planner state semantics, source exception behavior, budget counting, Adapter isolation, and a minimal fake/offline usage example. State explicitly that selecting `search.implementation: agentic` requires an injected Adapter and does not fabricate missing Tool implementations; the actual dependency factory must be provided when teammate branches merge.

- [ ] **Step 2: Run focused tests**

Run: `.\.venv\Scripts\python.exe -m pytest tests/literature -q`

Expected: all literature tests pass.

- [ ] **Step 3: Run the full regression suite**

Run: `.\.venv\Scripts\python.exe -m pytest -q`

Expected: all existing and new tests pass with zero failures.

- [ ] **Step 4: Run static repository checks**

Run:

```powershell
git diff --check
git status --short --branch
git diff --stat
```

Inspect the full diff for secrets, generated artifacts, PDF/XML content, unrelated edits and accidental legacy changes.

- [ ] **Step 5: Present the review checkpoint**

Report the feature worktree path, branch name, changed files, focused/full test counts, remaining dependency on teammate Tool implementations, and explicit confirmation that no commit or push occurred.
