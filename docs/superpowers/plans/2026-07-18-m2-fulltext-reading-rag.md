# M2 Full-Text Reading RAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the integrated M2 abstract placeholder with a PMC-open-full-text-first, abstract-fallback RAG workflow that produces evidence-linked knowledge entries for M3.

**Architecture:** Keep `ReadingExtractionWorkflowProtocol` as the public boundary and compose a resolver, parser, in-memory chunk store, hybrid lexical retriever, and Qwen paper reader behind it. Store raw documents only in a runtime cache, send only retrieved passages to Qwen, validate every returned evidence ID deterministically, and isolate ordinary failures per paper.

**Tech Stack:** Python 3 standard library (`asyncio`, `urllib`, `json`, `pathlib`, `hashlib`, `math`, `re`), Pydantic v2, existing `QwenClient`, pytest, pytest-asyncio.

## Global Constraints

- Use only the NCBI BioC PMC Open Access endpoint for full-text network retrieval.
- Never bypass a paywall, scrape publisher pages, download arbitrary PDFs, or add OCR.
- Do not add an embedding service, vector database, external BM25 dependency, or LLM reranker.
- Keep `AbstractReadingWorkflow`, legacy M2, and the minimal PubMed adapter intact as backups.
- Do not put whole papers in prompts or pipeline state; cache raw documents under `.cache/` only.
- Do not fabricate content on network, parse, retrieval, or model failure.
- Preserve input paper order and isolate ordinary failures per paper.
- Propagate cancellation and workflow-wide timeout rather than swallowing them as paper errors.
- Do not stage, commit, or push. The user will review the complete working tree first.

---

## File Map

- `hypoforge/literature/models.py`: reading trace fields on `PaperReadingResult`.
- `hypoforge/literature/protocols.py`: optional `SearchRunResult` context for reading.
- `hypoforge/literature/adapter.py`: pass completed search context into reading.
- `hypoforge/literature/reading/store.py`: in-memory chunk catalog shared by parser/retriever.
- `hypoforge/literature/reading/resolver.py`: BioC PMC fetch, cache, and abstract fallback.
- `hypoforge/literature/reading/parser.py`: BioC/abstract parsing and stable section-aware chunks.
- `hypoforge/literature/reading/retriever.py`: pure-Python BM25, section/entity bonuses, MMR, neighboring context.
- `hypoforge/literature/reading/reader.py`: Qwen structured extraction and evidence validation.
- `hypoforge/literature/reading/workflow.py`: concurrency, timeout, timing, error isolation, output order.
- `hypoforge/literature/reading/__init__.py`: public reading exports.
- `hypoforge/literature/integrated.py`: inject the real workflow in the integrated factory.
- `hypoforge/literature/minimal.py`: accept optional context while retaining deterministic abstract behavior.
- `hypoforge/literature/README.md`: document the finished reading path and limits.
- `tests/literature/reading/`: focused offline unit and workflow tests.
- `tests/literature/test_adapter.py`: search-context handoff test.
- `tests/literature/test_integrated_factory.py`: real workflow wiring test.
- `tests/test_run_m2_integrated.py`: trace compatibility test.

---

### Task 1: Extend the public reading contract without breaking backups

**Files:**
- Modify: `hypoforge/literature/models.py`
- Modify: `hypoforge/literature/protocols.py`
- Modify: `hypoforge/literature/adapter.py`
- Modify: `hypoforge/literature/minimal.py`
- Modify: `tests/literature/fakes.py`
- Modify: `tests/literature/test_adapter.py`
- Modify: `tests/literature/test_models.py`

**Interfaces:**
- Consumes: existing `SearchRunResult`, `ContentLevel`, and `PaperReadingResult`.
- Produces: `ReadingExtractionWorkflowProtocol.run(sub_question, papers, search_context=None)` and trace-capable reading results.

- [ ] **Step 1: Add failing model and Adapter tests**

Add assertions equivalent to:

```python
result = PaperReadingResult(
    paper_id="PMID:1",
    content_level=ContentLevel.STRUCTURED_FULLTEXT,
    document_id="PMID_1-pmc",
    chunks_parsed=12,
    chunks_retrieved=7,
    stage_elapsed_seconds={"resolver": 0.2},
)
assert result.chunks_parsed == 12

await adapter(state)
assert reading.search_contexts[0] is search_result
```

The fake workflow records both paper IDs and `search_context` values.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
uv run --with-requirements requirements.txt --with pytest --with pytest-asyncio pytest tests/literature/test_models.py tests/literature/test_adapter.py -q
```

Expected: failures for unknown `PaperReadingResult` fields and missing `search_context`.

- [ ] **Step 3: Implement the compatible contract**

Add these fields with safe defaults:

```python
class PaperReadingResult(LiteratureModel):
    paper_id: NonEmptyStr
    summary: str = ""
    evidence: List[EvidenceChunk] = Field(default_factory=list)
    knowledge_entries: List[EvidenceLinkedKnowledge] = Field(default_factory=list)
    content_level: ContentLevel = ContentLevel.METADATA
    document_id: str = ""
    chunks_parsed: int = Field(default=0, ge=0)
    chunks_retrieved: int = Field(default=0, ge=0)
    stage_elapsed_seconds: Dict[str, float] = Field(default_factory=dict)
    degraded_to_abstract: bool = False
    errors: List[str] = Field(default_factory=list)
```

Extend the workflow protocol and every current implementation/fake:

```python
async def run(
    self,
    sub_question: str,
    papers: Sequence[PaperRecord],
    search_context: SearchRunResult | None = None,
) -> List[PaperReadingResult]:
    ...
```

Pass `search_context=search_result` from `AgenticM2Adapter`.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the command from Step 2. Expected: all selected tests pass.

- [ ] **Step 5: Record an uncommitted checkpoint**

Run `git diff --check` and keep all changes uncommitted.

---

### Task 2: Implement PMC resolution, legal cache, and abstract fallback

**Files:**
- Create: `hypoforge/literature/reading/resolver.py`
- Create: `tests/literature/reading/__init__.py`
- Create: `tests/literature/reading/test_resolver.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: `PaperRecord` and `DocumentRecord`.
- Produces: `PMCFulltextResolver(cache_dir, timeout_seconds, backend)` implementing `resolve(paper)`.

- [ ] **Step 1: Write failing resolver tests**

Cover these exact cases with an injected async backend:

```python
async def backend(url: str, timeout: float) -> bytes:
    calls.append(url)
    return json.dumps(BIOC_FIXTURE).encode("utf-8")

document = await resolver.resolve(paper(pmcid="PMC123", abstract="fallback"))
assert document.content_level is ContentLevel.STRUCTURED_FULLTEXT
assert document.source_uri.endswith("/PMC123/unicode")
assert Path(document.local_path).read_bytes()

second = await resolver.resolve(the_same_paper)
assert len(calls) == 1
```

Also test PMID fallback, no identifier, HTTP failure with abstract, HTTP failure without abstract, invalid JSON, and cancellation propagation.

- [ ] **Step 2: Run resolver tests and verify RED**

Run:

```powershell
uv run --with-requirements requirements.txt --with pytest --with pytest-asyncio pytest tests/literature/reading/test_resolver.py -q
```

Expected: import failure for `PMCFulltextResolver`.

- [ ] **Step 3: Implement the resolver**

Use this concrete boundary:

```python
FetchBackend = Callable[[str, float], Awaitable[bytes]]

class PMCFulltextResolver(FulltextResolverProtocol):
    API_ROOT = (
        "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/"
        "pmcoa.cgi/BioC_json"
    )

    def __init__(
        self,
        cache_dir: str | Path = ".cache/hypoforge/literature/documents",
        timeout_seconds: float = 30.0,
        backend: FetchBackend | None = None,
    ) -> None:
        ...

    async def resolve(self, paper: PaperRecord) -> DocumentRecord:
        ...
```

The default backend uses `urllib.request.urlopen` inside `asyncio.to_thread`.
Validate that parsed JSON contains at least one non-empty passage before accepting
it as full text. Store full text as `bioc.json`; store fallback as
`abstract.json` with `{"format": "hypoforge_abstract_v1", "text": ...}`.
Use a sanitized SHA-256-based directory name, temporary file plus
`Path.replace`, and return a `retrieval_error` when fallback occurred.
Add `.cache/` to `.gitignore`.

- [ ] **Step 4: Run resolver tests and verify GREEN**

Run the command from Step 2. Expected: all resolver tests pass without network.

- [ ] **Step 5: Record an uncommitted checkpoint**

Run `git diff --check`; do not stage or commit.

---

### Task 3: Parse BioC and abstract documents into stable section-aware chunks

**Files:**
- Create: `hypoforge/literature/reading/parser.py`
- Create: `tests/literature/reading/test_parser.py`
- Create: `tests/literature/reading/fixtures/pmc_bioc.json`

**Interfaces:**
- Consumes: local `DocumentRecord` from Task 2.
- Produces: `BioCDocumentParser(target_chars=900, overlap_chars=120).parse(document)`.

- [ ] **Step 1: Add a compact BioC fixture and failing parser tests**

The fixture contains at least Abstract, Methods, Results, Discussion, and
Limitations passages. Assert:

```python
chunks = await parser.parse(document)
assert {chunk.section for chunk in chunks} >= {
    "abstract", "methods", "results", "discussion", "limitations"
}
assert all(chunk.text.strip() for chunk in chunks)
assert chunks == await parser.parse(document)
assert all(chunk.end_offset >= chunk.start_offset for chunk in chunks)
```

Test an overlong Results passage, overlap bounded by 120 characters, abstract
fallback JSON, malformed JSON, missing local file, and empty passages.

- [ ] **Step 2: Run parser tests and verify RED**

Run:

```powershell
uv run --with-requirements requirements.txt --with pytest --with pytest-asyncio pytest tests/literature/reading/test_parser.py -q
```

Expected: import failure for `BioCDocumentParser`.

- [ ] **Step 3: Implement deterministic parsing and chunking**

Normalize section labels through one mapping function, preserve original text,
split at paragraph/sentence boundaries, and generate IDs with:

```python
digest = hashlib.sha256(
    f"{document.document_id}\0{section}\0{ordinal}\0{text}".encode("utf-8")
).hexdigest()[:16]
chunk_id = f"{document.document_id}:chunk:{digest}"
```

Support both BioC shapes: a top-level list of documents and
`{"documents": [...]}`. Do not silently treat malformed content as empty;
raise `DocumentParseError` with the document ID.

- [ ] **Step 4: Run parser tests and verify GREEN**

Run the command from Step 2. Expected: all parser tests pass.

- [ ] **Step 5: Record an uncommitted checkpoint**

Run `git diff --check`; do not stage or commit.

---

### Task 4: Implement local multi-intent hybrid retrieval

**Files:**
- Create: `hypoforge/literature/reading/store.py`
- Create: `hypoforge/literature/reading/retriever.py`
- Create: `tests/literature/reading/test_retriever.py`

**Interfaces:**
- Produces: `InMemoryChunkStore.replace(paper_id, chunks)` and
  `InMemoryChunkStore.get(paper_ids)`.
- Produces: `HybridEvidenceRetriever(store).retrieve(query, paper_ids, top_k)`.

- [ ] **Step 1: Write failing store and retriever tests**

Create chunks whose sections separately contain a YAP mechanism, CRISPR method,
negative result, and limitation. Assert:

```python
results = await retriever.retrieve(
    "YAP TAZ mechanotransduction organ size", ["PMID:1"], top_k=6
)
assert {item.section for item in results} >= {"results", "methods", "limitations"}
assert len({item.chunk_id for item in results}) == len(results)
assert all(0.0 <= item.relevance_score <= 1.0 for item in results)
assert all(item.quote in chunk_text_by_id[item.chunk_id] for item in results)
```

Also test exact entity bonus, duplicate suppression, neighboring-block expansion,
paper filtering, deterministic order, and empty-query/empty-store behavior.

- [ ] **Step 2: Run retriever tests and verify RED**

Run:

```powershell
uv run --with-requirements requirements.txt --with pytest --with pytest-asyncio pytest tests/literature/reading/test_retriever.py -q
```

Expected: import failures for the store and retriever.

- [ ] **Step 3: Implement pure-Python BM25 and multi-intent reranking**

Tokenize Latin biomedical terms, hyphenated identifiers, numbers, and individual
CJK characters. Compute BM25 with `k1=1.5`, `b=0.75`, then score four intent
views:

```python
INTENTS = {
    "core": ("evidence association effect", {}),
    "mechanism": ("mechanism pathway causal activation inhibition", {
        "results": 0.18, "discussion": 0.12,
    }),
    "conflict": ("contradict negative no effect limitation uncertainty", {
        "limitations": 0.24, "discussion": 0.12, "results": 0.08,
    }),
    "method": ("method experiment assay model measurement", {
        "methods": 0.25,
    }),
}
```

Merge per-intent candidates, add exact multi-character entity/phrase bonuses,
normalize scores, and apply MMR with lexical Jaccard similarity and
`lambda=0.75`. Expand one same-section neighbor only while the final character
budget remains at or below 12,000 characters. Create `EvidenceChunk` directly
from the original `DocumentChunk.text`; `normalized_claim` is the first complete
sentence or the first 300 characters, never an LLM rewrite.

- [ ] **Step 4: Run retriever tests and verify GREEN**

Run the command from Step 2. Expected: all retrieval tests pass.

- [ ] **Step 5: Record an uncommitted checkpoint**

Run `git diff --check`; do not stage or commit.

---

### Task 5: Implement evidence-constrained Qwen paper reading

**Files:**
- Create: `hypoforge/literature/reading/reader.py`
- Create: `tests/literature/reading/test_reader.py`

**Interfaces:**
- Consumes: `QwenClient`, one `PaperRecord`, and retrieved `EvidenceChunk` values.
- Produces: `QwenPaperReader(client).read(sub_question, paper, evidence)`.

- [ ] **Step 1: Write failing reader tests with a fake structured client**

The successful fake response has this shape:

```python
{
    "summary": "Mechanical tension promotes nuclear YAP.",
    "summary_evidence_ids": ["PMID:1:evidence:1"],
    "entries": [{
        "type": "mechanistic_conclusion",
        "content": "Mechanical tension promotes nuclear YAP.",
        "confidence": "high",
        "entities": ["YAP", "mechanical tension"],
        "evidence_ids": ["PMID:1:evidence:1"],
    }],
}
```

Assert valid output, `temperature=0.0`, `disable_thinking=True`, and a bounded
prompt. Add tests for unknown evidence IDs, mixed valid/invalid IDs, invalid
types, empty content, duplicate entries, parse error payload, client exception,
and no evidence (which must skip the model call).

- [ ] **Step 2: Run reader tests and verify RED**

Run:

```powershell
uv run --with-requirements requirements.txt --with pytest --with pytest-asyncio pytest tests/literature/reading/test_reader.py -q
```

Expected: import failure for `QwenPaperReader`.

- [ ] **Step 3: Implement schema, prompt, and deterministic validation**

Use one object schema containing `summary`, `summary_evidence_ids`, and
`entries`. Call:

```python
raw = await client.structured_chat(
    system_prompt=SYSTEM_PROMPT,
    user_prompt=user_prompt,
    output_schema=OUTPUT_SCHEMA,
    max_tokens=4096,
    temperature=0.0,
    disable_thinking=True,
)
```

Discard an entry unless every evidence ID belongs to the supplied paper and
candidate set. Deduplicate by `(entry_type, normalized_content,
sorted_evidence_ids)`. Generate stable IDs from SHA-256 of the same tuple.
Return only evidence referenced by an accepted entry or the validated summary.
Map invalid model output to a `PaperReadingResult` error; never replace it with
fixed facts.

- [ ] **Step 4: Run reader tests and verify GREEN**

Run the command from Step 2. Expected: all reader tests pass.

- [ ] **Step 5: Record an uncommitted checkpoint**

Run `git diff --check`; do not stage or commit.

---

### Task 6: Compose the full reading workflow with concurrency and diagnostics

**Files:**
- Create: `hypoforge/literature/reading/workflow.py`
- Modify: `hypoforge/literature/reading/__init__.py`
- Modify: `hypoforge/literature/__init__.py`
- Create: `tests/literature/reading/test_workflow.py`

**Interfaces:**
- Consumes: resolver, parser, chunk store, retriever, and reader from Tasks 2-5.
- Produces: `FullTextReadingWorkflow.run(sub_question, papers, search_context=None)`.

- [ ] **Step 1: Write failing workflow tests**

Use fakes to simulate one full-text paper, one abstract fallback, and one paper
without readable content. Assert:

```python
results = await workflow.run("中文子问题", papers, search_context=search_result)
assert [item.paper_id for item in results] == [p.paper_id for p in papers]
assert results[0].content_level is ContentLevel.STRUCTURED_FULLTEXT
assert results[1].degraded_to_abstract is True
assert results[2].knowledge_entries == []
assert "no readable content" in " ".join(results[2].errors)
assert retrieval_queries[0] contains an English search query from search_result
```

Also test maximum fetch/read concurrency, per-paper reader timeout, workflow
timeout propagation, cancellation propagation, parser failure isolation, and
stage timing keys.

- [ ] **Step 2: Run workflow tests and verify RED**

Run:

```powershell
uv run --with-requirements requirements.txt --with pytest --with pytest-asyncio pytest tests/literature/reading/test_workflow.py -q
```

Expected: import failure for `FullTextReadingWorkflow`.

- [ ] **Step 3: Implement orchestration**

Constructor defaults:

```python
def __init__(
    self,
    *,
    resolver: FulltextResolverProtocol,
    parser: DocumentParserProtocol,
    retriever: EvidenceRetrieverProtocol,
    reader: PaperReaderProtocol,
    store: InMemoryChunkStore,
    fetch_concurrency: int = 3,
    read_concurrency: int = 2,
    reader_timeout_seconds: float = 120.0,
    workflow_timeout_seconds: float = 600.0,
    top_k: int = 8,
) -> None:
    ...
```

Build the retrieval query from the sub-question plus unique executed search
queries, Scout terms/mechanisms, and coverage missing topics. Limit this query
context to 4,000 characters. Measure `resolver`, `parser`, `retriever`, and
`paper_reader` with `time.monotonic`. Use `asyncio.gather` so output order is
stable. Catch ordinary per-paper exceptions, but re-raise `CancelledError` and
the outer `asyncio.TimeoutError`.

- [ ] **Step 4: Run workflow tests and verify GREEN**

Run the command from Step 2. Expected: all workflow tests pass.

- [ ] **Step 5: Run the complete reading package tests**

Run:

```powershell
uv run --with-requirements requirements.txt --with pytest --with pytest-asyncio pytest tests/literature/reading -q
```

Expected: all reading tests pass.

- [ ] **Step 6: Record an uncommitted checkpoint**

Run `git diff --check`; do not stage or commit.

---

### Task 7: Replace the integrated placeholder and preserve observable traces

**Files:**
- Modify: `hypoforge/literature/integrated.py`
- Modify: `tests/literature/test_integrated_factory.py`
- Modify: `tests/test_run_m2_integrated.py`
- Modify: `hypoforge/literature/README.md`

**Interfaces:**
- Consumes: `QwenClient` and the new reading components.
- Produces: an integrated `AgenticM2Adapter` whose `reading_workflow` is `FullTextReadingWorkflow`.

- [ ] **Step 1: Write failing factory and trace tests**

Assert:

```python
adapter = build_integrated_search_adapter(client=client, search_tool=tool)
assert isinstance(adapter.reading_workflow, FullTextReadingWorkflow)
assert not isinstance(adapter.reading_workflow, AbstractReadingWorkflow)
```

Inject a temporary cache directory and fake PMC backend so the factory test
does not use the network. Verify serialized CLI trace includes
`content_level`, `chunks_parsed`, `chunks_retrieved`, and
`stage_elapsed_seconds` while still omitting full document text.

- [ ] **Step 2: Run integration tests and verify RED**

Run:

```powershell
uv run --with-requirements requirements.txt --with pytest --with pytest-asyncio pytest tests/literature/test_integrated_factory.py tests/test_run_m2_integrated.py -q
```

Expected: factory still injects `AbstractReadingWorkflow`.

- [ ] **Step 3: Wire the real workflow**

Extend `build_integrated_search_adapter` with optional dependency-injection
arguments for `reading_cache_dir`, `pmc_backend`, and reading timeouts. Build
one shared store, resolver, parser, retriever, reader, and workflow. Do not
change `build_minimal_pubmed_adapter`.

Update README status text from “placeholder remains” to the exact PMC-first,
abstract-fallback behavior and limitations.

- [ ] **Step 4: Run integration tests and verify GREEN**

Run the command from Step 2. Expected: all selected tests pass.

- [ ] **Step 5: Run all M2 regression tests**

Run:

```powershell
uv run --with-requirements requirements.txt --with pytest --with pytest-asyncio pytest tests/literature tests/test_qwen_client.py tests/test_run_m2_integrated.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Record an uncommitted checkpoint**

Run `git diff --check`; do not stage or commit.

---

### Task 8: Final verification and one transparent live smoke test

**Files:**
- Modify only if a verified defect is found: files owned by Tasks 1-7.
- Do not persist real API output or full paper text in tracked files.

**Interfaces:**
- Consumes: complete integrated M2.
- Produces: verification evidence and a user-facing report; no Git commit.

- [ ] **Step 1: Run the complete offline suite**

Run:

```powershell
uv run --with-requirements requirements.txt --with pytest --with pytest-asyncio pytest -q
```

Expected: all tests pass.

- [ ] **Step 2: Compile and inspect diffs**

Run:

```powershell
uv run --with-requirements requirements.txt python -m compileall -q hypoforge tests
git diff --check
git diff --cached --check
```

Expected: exit code 0. Existing line-ending warnings are acceptable; errors are not.

- [ ] **Step 3: Scan for persisted secrets and paper text**

Search tracked/untracked source files for API-key assignments and make sure
`.cache/` is ignored. Expected: no secret literals and no PMC document payload
under a tracked path.

- [ ] **Step 4: Run a live PMC plus Qwen smoke test**

Use process-local environment variables from the user-authorized external
`.env`; do not print their values. Select one known PMC Open Access paper and
run one-paper reading with `qwen3.6-plus`. Report:

- resolver status and content level;
- parsed/retrieved chunk counts;
- evidence and knowledge counts;
- evidence ID validity;
- per-stage and wall time;
- any real network/model failure without fabricated fallback.

- [ ] **Step 5: Run the original full M2 question if the smoke test succeeds**

Use the same Hippo/YAP/TAZ sub-question and report Search plus Reading traces.
Do not claim full-text success for papers that actually degraded to abstracts.

- [ ] **Step 6: Present the working tree for review**

Report branch, HEAD, exact test counts, live results, changed files, known
limitations, and confirm that no commit or push occurred.
