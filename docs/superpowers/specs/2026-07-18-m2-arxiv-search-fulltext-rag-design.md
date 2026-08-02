# M2 arXiv Search and Full-Text RAG Design

**Date:** 2026-07-18
**Status:** Approved design, pending implementation
**Scope:** Add arXiv as an independent search source and enable PDF-based RAG for selected arXiv papers.

## Goal

Extend the existing M2 literature pipeline beyond biomedical literature without disrupting the working PubMed and PMC path. The search agent must be able to plan and execute arXiv searches, then download and read only the final selected arXiv papers.

## Non-goals

- Do not replace PubMed, PMC, Semantic Scholar, or OpenAlex.
- Do not download PDFs for every search result.
- Do not implement citation-graph traversal or source-archive parsing in this change.
- Do not claim that every arXiv submission has completed peer review.
- Do not commit or push the changes without a separate user instruction.

## Architecture

```text
Query Planner
  |-- PubMedSource
  |-- AcademicSource (Semantic Scholar/OpenAlex)
  `-- ArxivSource
          |
          v
Deduplicate -> Rank -> Scout -> Coverage iteration
          |
          v
        Final-K
          |
          v
Multi-source full-text resolver
  |-- PubMed/PMC record -> PMC BioC JSON
  `-- arXiv record      -> cached PDF
          |
          v
Multi-format parser
  |-- BioC JSON
  |-- abstract JSON
  `-- PDF pages
          |
          v
Chunk store -> Hybrid retrieval -> Qwen paper reader
```

## Components

### 1. arXiv search backend

Add a small wrapper around the third-party `arxiv` Python package. It will:

- execute searches through a reusable `arxiv.Client`;
- enforce a configured page size, request delay, retry count, and result limit;
- run the blocking client outside the asyncio event loop;
- return normalized dictionaries for the source adapter;
- surface network and API failures rather than fabricating results.

The backend will be injectable so unit tests do not require the public arXiv service.

### 2. `ArxivSource`

Add an implementation of `LiteratureSourceProtocol` with `source_name = "arxiv"`. Each result will map to `PaperRecord` as follows:

- `paper_id`: stable `ARXIV:<id-without-version>` identifier;
- `title`, `abstract`, `authors`, `year`, and optional DOI;
- `external_ids`: arXiv identifier, entry URL, PDF URL, and primary category when available;
- `publication_type`: `preprint`;
- `sources`: `["arxiv"]`;
- `fulltext_status`: `PDF_AVAILABLE` when a PDF URL exists, otherwise `ABSTRACT_ONLY` or `UNKNOWN`.

The search tool will expose arXiv to the query planner as the preferred source for computer science, mathematics, physics, statistics, electrical engineering, quantitative biology, quantitative finance, and related preprints.

### 3. Full-text routing and PDF cache

Keep `PMCFulltextResolver` unchanged for PubMed/PMC records. Add an arXiv PDF resolver and a routing resolver:

- arXiv records are recognized by their source or arXiv external ID;
- only records passed to the existing Final-K reading workflow are resolved;
- downloaded PDFs are stored under the existing document cache using a stable paper-ID-derived directory;
- valid cached files are reused;
- downloads use a descriptive user agent, timeout, and maximum response size;
- HTTP errors, timeouts, invalid content, and oversized files return an abstract fallback with an explicit retrieval error;
- the document records preserve the arXiv PDF URL and avoid asserting a reuse licence that was not supplied by the record.

### 4. Multi-format document parser

Keep the existing BioC and abstract JSON parser behavior. Add PDF parsing through `pypdf` and route by `DocumentRecord.content_level`:

- extract text page by page;
- normalize whitespace without merging page boundaries;
- split long pages into bounded chunks using the existing chunk-size conventions;
- attach one-based page numbers and stable chunk IDs;
- reject unreadable or empty PDFs so the workflow can report a clear failure.

Scanned PDFs requiring OCR are outside this change and will degrade to the abstract.

## Data flow

1. The query planner receives `arxiv` in its tool definitions.
2. It may emit source-targeted arXiv queries in any search round.
3. The iterative agent calls arXiv concurrently with the other selected sources.
4. arXiv metadata participates in the existing deduplication, ranking, scout reading, and coverage evaluation.
5. Only Final-K papers enter the reading workflow.
6. PMC-backed papers continue through BioC; arXiv-backed papers use cached PDF resolution.
7. Parsed PDF chunks enter the existing hybrid retriever and Qwen reader.
8. A failed arXiv search is recorded as a source failure. A failed PDF resolution or parsing attempt degrades that paper to its abstract when an abstract exists.

## Deduplication and identity

arXiv versions such as `2401.12345v1` and `2401.12345v2` share the canonical identity `ARXIV:2401.12345`. The original versioned identifier may remain in `external_ids`. DOI normalization remains the cross-source bridge, allowing an arXiv preprint and a published record to merge when both provide the same DOI.

## Dependencies

- `arxiv`: client wrapper for the public arXiv API.
- `pypdf`: local PDF text extraction.

Both dependencies will be recorded in `requirements.txt`. Network calls remain runtime operations; tests use injected deterministic backends.

## Error handling

- Search failures raise a source-specific error and are visible in `SearchRunResult.failed_sources` and `errors`.
- No fake or cached unrelated search results are returned after a network failure.
- PDF downloads validate status, content signature, non-empty body, and size.
- Parser failures do not silently produce empty evidence.
- Cancellation propagates instead of being converted into an ordinary failure.
- Existing source timeout and reading workflow timeout remain authoritative.

## Test strategy

Implementation follows red-green-refactor cycles.

1. Source mapping tests: canonical IDs, metadata, PDF status, malformed rows, and version handling.
2. Search-tool tests: arXiv appears in planner definitions, direct dispatch, and source list.
3. Resolver tests: successful PDF cache, cache reuse, invalid PDF, size limit, network failure, and abstract fallback.
4. Parser tests: page-aware extraction, stable chunks, and empty/unreadable PDF errors.
5. Integration factory tests: PMC records still use PMC; arXiv records use PDF; existing injected backends remain supported.
6. Full regression suite.
7. One bounded live arXiv smoke search and one Final-K PDF reading run, reported separately from deterministic tests.

## Acceptance criteria

- The planner can select `arxiv` as an independent source.
- A successful arXiv search returns normalized `PaperRecord` values without blocking the event loop.
- Existing PubMed and academic sources remain available.
- Only Final-K arXiv PDFs are downloaded and cached.
- PDF text enters the existing chunk retrieval and reading path with page attribution.
- PDF failure degrades to the paper abstract and records the real error.
- Unit and integration tests pass without requiring network access.
- A live smoke test reports real success or the real network failure; it never fabricates papers.
