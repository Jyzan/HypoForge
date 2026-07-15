# M2 Agentic Scaffold Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development to implement this plan task-by-task.

**Goal:** Create a reviewable, uncommitted public scaffold for parallel development of the M2 iterative search and reading-extraction tools.

**Architecture:** Keep the existing `M2LiteratureSearch` entry point untouched. Add a new `hypoforge.literature` package containing shared Pydantic data contracts and asynchronous Tool protocols; individual feature branches will implement those protocols later.

**Tech Stack:** Python 3.12, Pydantic 2, pytest, pytest-asyncio.

## Global Constraints

- Do not commit or push any change until the user explicitly approves it.
- Do not replace or modify the legacy M2 implementation in this scaffold.
- Do not add live API calls or implementation placeholders that are imported at runtime.
- Keep downloaded papers, indexes, caches, outputs, and API keys outside Git.
- Every shared contract must have an offline test.

---

### Task 1: Shared literature data contracts

**Files:**
- Create: `hypoforge/literature/models.py`
- Create: `tests/literature/test_models.py`

**Interfaces:**
- Produces: `SearchQuery`, `PaperRecord`, `ScoutNote`, `CoverageReport`, `SearchBudget`, `SearchState`, `SearchRunResult`, `DocumentRecord`, `DocumentChunk`, and `EvidenceChunk`.

- [ ] Write model validation and serialization tests first.
- [ ] Run the focused test and verify it fails because the package does not exist.
- [ ] Implement the minimal Pydantic models and enums.
- [ ] Run the focused test and verify it passes.
- [ ] Review the uncommitted diff; do not commit.

### Task 2: Tool protocols

**Files:**
- Create: `hypoforge/literature/protocols.py`
- Create: `tests/literature/test_protocols.py`

**Interfaces:**
- Consumes: Task 1 data contracts.
- Produces: asynchronous protocols for query planning, source search, deduplication, ranking, scout reading, coverage evaluation, full-text resolution, document parsing, evidence retrieval, and paper reading.

- [ ] Write tests using small concrete test implementations.
- [ ] Run the focused test and verify it fails because protocols do not exist.
- [ ] Implement minimal abstract protocols with typed signatures.
- [ ] Run the focused test and verify it passes.
- [ ] Review the uncommitted diff; do not commit.

### Task 3: Package surface and collaboration guide

**Files:**
- Create: `hypoforge/literature/__init__.py`
- Create: `hypoforge/literature/README.md`
- Create: `hypoforge/literature/sources/__init__.py`
- Create: `hypoforge/literature/search/__init__.py`
- Create: `hypoforge/literature/reading/__init__.py`
- Create: `tests/literature/__init__.py`
- Create: `tests/literature/fixtures/README.md`

**Interfaces:**
- Consumes: Tasks 1 and 2.
- Produces: stable import surface and ownership boundaries for future feature branches.

- [ ] Export only the reviewed shared contracts and protocols.
- [ ] Document package boundaries, branch ownership, and fixture rules.
- [ ] Run the new literature tests.
- [ ] Run the full existing test suite.
- [ ] Present branch name, status, diff, and test evidence to the user; do not commit or push.
