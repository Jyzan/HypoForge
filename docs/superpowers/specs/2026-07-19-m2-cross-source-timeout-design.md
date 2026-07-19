# M2 Cross-Source Coverage and Full-Text Timeout Design

**Date:** 2026-07-19
**Status:** Approved design, pending implementation
**Scope:** Guarantee first-round coverage of every configured literature source and prevent a slow arXiv PDF from cancelling the entire Final-K reading batch.

## Goals

1. Every question, including biomedical questions, searches PubMed, Semantic Scholar/OpenAlex, and arXiv in the first round.
2. A single slow, unavailable, invalid, or oversized arXiv PDF degrades to its real abstract without failing other Final-K papers.
3. PDF timeout means a real end-to-end download deadline, not only a socket inactivity timeout.

## Non-goals

- Do not force unavailable sources in later rounds after the search agent opens their circuit.
- Do not fabricate search results or paper content.
- Do not add OCR, publisher scraping, or additional search providers.
- Do not replace the existing global search, token, or workflow budgets.
- Do not commit or push without a separate user instruction.

## First-round source coverage

The query planner remains model-driven, but its first-round output receives a deterministic coverage check before returning to the search agent.

```text
Model-planned first-round queries
  -> validate and sanitize
  -> collect selected backends
  -> for each configured backend not selected:
       create one deterministic query from the sub-question
  -> return model queries plus missing-source queries
```

The configured backends are taken from `tool_definitions`; no source names are duplicated in a second configuration list. With the current search tool, the required first-round set is:

- `pubmed`
- `semantic_scholar`
- `arxiv`

The added query uses the normalized sub-question text. PubMed receives ordinary keyword text accepted by PubMed; non-PubMed backends remove PubMed field tags. Each added query records `purpose="cross_source_coverage"` and an explicit relation explaining that it guarantees first-round source coverage.

Only the first round is forced. Later rounds remain gap-driven and honor `SearchState.unavailable_sources`. The existing `SearchBudget.max_queries` remains authoritative if model output plus coverage queries reaches the query limit.

## Bounded arXiv PDF download

The default arXiv downloader changes from one unbounded `response.read()` call to a chunked loop.

```text
open response with a bounded socket inactivity timeout
  -> validate Content-Length when present
  -> read fixed-size chunks
  -> after every chunk:
       check total elapsed monotonic time
       check accumulated byte count
  -> validate PDF signature
  -> atomically cache completed payload
```

Defaults:

- total download deadline: 180 seconds per PDF;
- socket inactivity timeout: at most 30 seconds per blocking operation;
- maximum PDF size: 52,428,800 bytes;
- read chunk size: 65,536 bytes.

The resolver constructor keeps its injected async backend interface for existing deterministic tests. The built-in network backend enforces the byte and time limits while reading. Injected backends remain subject to a per-paper workflow timeout so a misbehaving test or alternative backend cannot hold the whole batch indefinitely.

## Failure isolation and abstract fallback

Each Final-K paper is processed independently.

- Successful PDF: cache, parse pages, retrieve evidence, and read with Qwen.
- Download timeout, socket timeout, HTTP error, invalid signature, or size violation: create the existing real-abstract document and continue RAG at `ContentLevel.ABSTRACT`.
- PDF parsing failure or scanned/empty PDF: use the existing explicit `resolve_abstract()` path.
- No abstract: return a metadata-level result with the real error.
- Cancellation caused by the global workflow timeout still propagates; ordinary per-paper timeout is converted to a paper-local degraded result.

The global reading workflow still has a 600-second deadline. With three concurrent downloads bounded at 180 seconds and Qwen reads bounded at 120 seconds with concurrency two, one slow PDF no longer consumes the whole workflow budget by itself.

## Error reporting and timing

Degraded reading results must contain:

- `degraded_to_abstract=true` when a real abstract is used;
- the original timeout or download failure in `errors`;
- normal `stage_elapsed_seconds` entries for resolution, parsing, retrieval, and reading;
- successful results from other Final-K papers in original order.

The downloader never returns a partial PDF and never promotes a `.tmp` file to the cache unless all validation succeeds.

## Test strategy

Implementation follows red-green-refactor cycles.

1. Planner tests prove that model output missing two backends is augmented to all three on round one.
2. Planner tests prove later rounds do not force every source and do not restore unavailable sources.
3. Downloader tests use a fake streaming response and monotonic clock to prove the 180-second total deadline.
4. Downloader tests cover Content-Length rejection, accumulated-size rejection, valid cache write, and partial-file safety.
5. Workflow tests prove one timed-out paper degrades to its real abstract while sibling papers complete.
6. Existing PubMed, PMC, arXiv, parser, retriever, and full repository tests must remain green.
7. A bounded live run repeats the prior technical question and reports search coverage, Final-K content levels, errors, and total time.

## Acceptance criteria

- Every first-round plan targets all configured search sources exactly at least once.
- Later rounds remain coverage-driven and skip unavailable sources.
- The built-in arXiv downloader cannot spend more than 180 seconds reading one PDF, apart from small scheduling overhead.
- A timed-out PDF returns an abstract reading result when an abstract exists.
- One slow PDF does not cancel successful sibling paper results.
- No incomplete PDF is cached.
- Deterministic and full repository test suites pass.
- Live network failures are reported, never replaced by synthetic data.
