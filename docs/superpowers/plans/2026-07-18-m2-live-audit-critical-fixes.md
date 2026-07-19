# M2 Live-Audit Critical Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct the four highest-impact quality defects reproduced by the live M2 audit.

**Architecture:** Keep all existing Tool boundaries. Add a deterministic post-Scout ranking function, distinguish citable primary evidence from context-only neighbors, extend the legacy knowledge contract with optional evidence IDs, and make the parser's ASCII overlap boundary safe.

**Tech Stack:** Python 3.14, Pydantic 2, asyncio, pytest, pytest-asyncio.

## Global Constraints

- Do not change the default legacy M2 implementation.
- Do not remove teammate Tools or the minimal PubMed backup.
- Do not commit or push without explicit user approval.
- Use RED-GREEN TDD for every production behavior change.

---

### Task 1: Post-Scout Re-ranking

**Files:**
- Modify: `tests/literature/test_search_agent.py`
- Modify: `hypoforge/literature/search/ranking.py`
- Modify: `hypoforge/literature/search/agent.py`

**Interfaces:**
- Consumes: ordered `PaperRecord` candidates and `ScoutNote` results.
- Produces: `rerank_with_scout(papers, notes) -> list[PaperRecord]`.

- [ ] Add a test where metadata order prefers paper A but Scout relevance prefers paper B.
- [ ] Run that test and confirm the final paper remains A.
- [ ] Implement a deterministic post-Scout blend and call it before coverage.
- [ ] Run the focused search tests and confirm paper B becomes Final-1.

### Task 2: Non-citable Neighbor Context

**Files:**
- Modify: `tests/literature/reading/test_retriever.py`
- Modify: `tests/literature/reading/test_reader.py`
- Modify: `hypoforge/literature/models.py`
- Modify: `hypoforge/literature/reading/retriever.py`
- Modify: `hypoforge/literature/reading/reader.py`

**Interfaces:**
- Extends: `EvidenceChunk.citable: bool = True`.
- Produces: primary hits that are citable and neighbor context that is not.

- [ ] Add a retriever test asserting a zero-score neighbor is non-citable.
- [ ] Add a reader test asserting knowledge that cites only a non-citable ID is rejected.
- [ ] Run both tests and confirm the new behavior is missing.
- [ ] Add the field, prompt label, and citable-ID validation.
- [ ] Run all reading tests.

### Task 3: Word-safe Chunk Overlap

**Files:**
- Modify: `tests/literature/reading/test_parser.py`
- Modify: `hypoforge/literature/reading/parser.py`

**Interfaces:**
- Keeps: `BioCDocumentParser.parse(document) -> list[DocumentChunk]`.

- [ ] Add a passage whose overlap begins inside `mitogen-activated`.
- [ ] Run the test and confirm a chunk starts with a broken token.
- [ ] Advance ASCII overlap starts to the next word boundary.
- [ ] Run parser tests and confirm every chunk starts on a readable boundary.

### Task 4: Preserve Evidence IDs at the Adapter Boundary

**Files:**
- Modify: `tests/literature/test_adapter.py`
- Modify: `hypoforge/state.py`
- Modify: `hypoforge/literature/adapter.py`

**Interfaces:**
- Extends: `KnowledgeEntry.evidence_ids: list[str] = []`.

- [ ] Assert the adapter output contains the reading entry's evidence IDs.
- [ ] Run the test and confirm the field is missing.
- [ ] Add the backward-compatible field and map it in the adapter.
- [ ] Run adapter and CLI serialization tests.

### Task 5: Exclude Reference-list Passages

**Files:**
- Modify: `tests/literature/reading/test_parser.py`
- Modify: `hypoforge/literature/reading/parser.py`

**Interfaces:**
- Keeps: `BioCDocumentParser.parse(document) -> list[DocumentChunk]`.

- [ ] Add a BioC `section_type=REF`, `type=ref` passage to the parser fixture.
- [ ] Run the test and confirm the citation title becomes a chunk.
- [ ] Filter reference-list passage types before chunking.
- [ ] Run parser and cached real-document checks.

### Task 6: Verification

**Files:**
- No production changes.

- [ ] Run all focused literature tests.
- [ ] Run the complete pytest suite.
- [ ] Run `compileall`, `git diff --check`, secret scan, and cache-ignore check.
- [ ] Run a cached real-paper reading check and inspect citable scores and chunk starts.
