# M2 OpenAlex Restoration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore OpenAlex as an independent fourth M2 literature source while keeping Semantic Scholar fallback bounded and query-compatible.

**Architecture:** Add a protocol adapter dedicated to OpenAlex, register it beside PubMed, Semantic Scholar and arXiv, and forward per-run credentials through the existing integrated factory. Keep the legacy S2 fallback for resilience, but give the fallback its own deadline and centralize OpenAlex query normalization at the transport boundary.

**Tech Stack:** Python 3.12, asyncio, urllib, Pydantic models, pytest/pytest-asyncio.

## Global Constraints

- Preserve all unrelated uncommitted UI and M4 changes.
- Do not commit or push.
- Never persist or print API keys.
- Use tests before production changes.

---

### Task 1: Independent OpenAlex source contract

**Files:**
- Create: `hypoforge/literature/sources/openalex_source.py`
- Modify: `hypoforge/literature/sources/__init__.py`
- Modify: `hypoforge/literature/search/search_tool.py`
- Test: `tests/literature/test_openalex_source.py`

**Interfaces:**
- Consumes: `search_openalex_strict(query, limit, api_key, mailto)` and `_dict_to_record`.
- Produces: `OpenAlexSource.search(SearchQuery, limit) -> list[PaperRecord]`.

- [x] Write tests requiring an independently injectable `OpenAlexSource`, four Tool definitions, direct dispatch, and credential plumbing.
- [x] Run the new test file and verify failure because the source module/registration is missing.
- [x] Implement the source and four-source registration with the smallest compatible change.
- [x] Run the new test file and verify it passes.

### Task 2: Factory and live configuration wiring

**Files:**
- Modify: `hypoforge/literature/integrated.py`
- Modify: `configs/full_pipeline_m2agentic_qwen37plus_live.yaml`
- Modify: `tests/literature/test_integrated_factory.py`

**Interfaces:**
- Consumes: `LiteratureSearchTool(openalex_api_key=..., openalex_mailto=...)`.
- Produces: four-source `IterativeSearchAgent` construction from the live pipeline configuration.

- [x] Change the factory expectation from three sources to four and add a credential-forwarding assertion.
- [x] Run the targeted factory tests and verify the old implementation fails.
- [x] Add the factory parameters and restore `openalex` in the live source list.
- [x] Run the targeted factory tests and verify they pass.

### Task 3: OpenAlex transport safety

**Files:**
- Modify: `hypoforge/tools/semantic_scholar.py`
- Test: `tests/literature/test_openalex_source.py`
- Test: `tests/test_semantic_scholar_fallback.py`

**Interfaces:**
- Produces: `search_openalex_strict(...)`, normalized OpenAlex free-text queries, and a fresh OpenAlex fallback deadline.

- [x] Add tests proving Boolean/field-tag queries are normalized before URL construction and fallback does not reuse the expired S2 deadline.
- [x] Run the tests and verify both regressions fail.
- [x] Add centralized OpenAlex normalization, strict one-shot API, mailto support, and a fresh fallback deadline.
- [x] Run the targeted tests and verify they pass.

### Task 4: Verification

**Files:**
- No production changes expected.

- [x] Run all literature/M2 tests.
- [x] Run the full test suite.
- [x] Run a real OpenAlex-only search using existing environment configuration without displaying credentials.
- [x] Inspect `git diff --check` and the final scoped diff.
