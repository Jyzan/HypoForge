# M2 Paper Selection Tool Integration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the integrated M2 placeholder deduplication, ranking, Scout Reading, and coverage evaluation components with Carl's reviewed implementations while keeping the existing Query Planner and source adapters primary.

**Architecture:** `build_integrated_search_adapter()` remains the only composition root. It will construct `PaperDeduplicator`, `PaperRanker`, `ScoutReader(client)`, and `CoverageEvaluator(client)` from `hypoforge.literature.search`; the existing abstract-only reading workflow remains injected because no teammate full reading implementation exists. The minimal PubMed-only adapter remains an independent backup.

**Tech Stack:** Python 3, asyncio, Pydantic, pytest, existing HypoForge literature protocols.

## Global Constraints

- Do not commit, push, or finalize the pending merge before user review.
- Do not modify teammate Tool algorithms in this task.
- Do not add a primary-to-backup fallback.
- Keep offline tests network-free and preserve the explicit minimal PubMed backup.

---

### Task 1: Wire the real paper-selection Tools

**Files:**
- Modify: `hypoforge/literature/integrated.py`
- Modify: `tests/literature/test_integrated_factory.py`
- Modify: `hypoforge/literature/README.md`

**Interfaces:**
- Consumes: `PaperDeduplicator()`, `PaperRanker()`, `ScoutReader(client)`, `CoverageEvaluator(client)`.
- Produces: `build_integrated_search_adapter(...) -> AgenticM2Adapter` whose `search_agent` uses both teammates' primary Tools.

- [ ] **Step 1: Write the failing factory wiring test**

```python
def test_factory_wires_real_paper_selection_tools() -> None:
    adapter = build_integrated_search_adapter(
        client=FakeClient(), search_tool=make_search_tool(), final_k=5
    )
    agent = adapter.search_agent
    assert isinstance(agent.deduplicator, PaperDeduplicator)
    assert isinstance(agent.ranker, PaperRanker)
    assert isinstance(agent.scout_reader, ScoutReader)
    assert agent.scout_reader.client is not None
    assert isinstance(agent.coverage_evaluator, CoverageEvaluator)
    assert agent.coverage_evaluator.client is not None
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```powershell
& 'C:\Users\LIU\Desktop\挑战杯\核心代码\HypoForge\.venv\Scripts\python.exe' `
  -m pytest tests/literature/test_integrated_factory.py::test_factory_wires_real_paper_selection_tools -q
```

Expected: fail because the factory still constructs `ExactPaperDeduplicator`, `MetadataPaperRanker`, `AbstractScoutReader`, and `SingleSourceCoverageEvaluator`.

- [ ] **Step 3: Replace only the four placeholder constructors**

```python
from .search import (
    CoverageEvaluator,
    IterativeSearchAgent,
    PaperDeduplicator,
    PaperRanker,
    ScoutReader,
)

agent = IterativeSearchAgent(
    query_planner=planner,
    sources=tool.as_source_list(),
    deduplicator=PaperDeduplicator(),
    ranker=PaperRanker(),
    scout_reader=ScoutReader(client),
    coverage_evaluator=CoverageEvaluator(client),
    ...
)
```

Keep `AbstractReadingWorkflow()` unchanged.

- [ ] **Step 4: Update the integration documentation**

Document that Query Planner and both sources come from the first teammate, paper selection comes from the second teammate, and only abstract reading remains a deterministic placeholder in the integrated path.

- [ ] **Step 5: Run GREEN and regressions**

```powershell
& 'C:\Users\LIU\Desktop\挑战杯\核心代码\HypoForge\.venv\Scripts\python.exe' `
  -m pytest tests/literature/test_integrated_factory.py `
  tests/literature/test_paper_selection_integration.py `
  tests/test_run_m2_integrated.py tests/test_run_m2_pubmed.py -q
& 'C:\Users\LIU\Desktop\挑战杯\核心代码\HypoForge\.venv\Scripts\python.exe' -m pytest -q
```

Expected: all focused and full offline tests pass.

- [ ] **Step 6: Audit remaining placeholders**

Inspect the final integrated factory and report every injected Tool. Confirm whether any placeholder remains and why. Run `git diff --check` and leave all changes uncommitted.
