# M2 arXiv Search and Full-Text RAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add arXiv as an independent iterative-search source and feed only selected arXiv PDFs into the existing evidence RAG workflow.

**Architecture:** A small `ArxivTool` wraps the blocking third-party client and an `ArxivSource` normalizes results. A routing resolver preserves PMC behavior while resolving arXiv records to cached PDFs, and a routing parser sends PDFs to a page-aware `pypdf` parser while keeping BioC and abstract JSON parsing unchanged.

**Tech Stack:** Python 3.10+, asyncio, Pydantic v2, `arxiv`, `pypdf`, pytest, pytest-asyncio.

## Global Constraints

- Preserve PubMed, PMC, Semantic Scholar, OpenAlex, legacy M2, and all existing injected-test seams.
- Download PDFs only for papers passed to the Final-K reading workflow.
- Return real provider failures; never fabricate results.
- Canonicalize arXiv versions to `ARXIV:<id-without-version>` while preserving the versioned ID in metadata.
- Fall back to the real paper abstract when PDF resolution fails and record the failure.
- Do not add OCR, source-tarball parsing, or citation-graph traversal.
- Do not commit or push; the user will review the working tree first.

## File map

- Create `hypoforge/tools/arxiv_search.py`: blocking arXiv client wrapper with an async strict interface.
- Create `hypoforge/literature/sources/arxiv_source.py`: `PaperRecord` normalization and source-specific errors.
- Modify `hypoforge/literature/search/search_tool.py`: planner definition, dependency injection, dispatch, and source listing.
- Modify `hypoforge/literature/sources/__init__.py`: public source export.
- Create `hypoforge/literature/reading/arxiv_resolver.py`: bounded PDF fetch, cache, validation, and abstract fallback.
- Create `hypoforge/literature/reading/routing.py`: source-aware resolver and content-aware parser routing.
- Modify `hypoforge/literature/reading/parser.py`: page-aware PDF parsing.
- Modify `hypoforge/literature/reading/__init__.py`: public reading exports.
- Modify `hypoforge/literature/integrated.py`: wire the routing resolver and parser.
- Modify `requirements.txt`: add `arxiv` and `pypdf`.
- Modify `hypoforge/literature/README.md`: document source coverage and Final-K PDF behavior.
- Create focused tests under `tests/literature/` and `tests/literature/reading/`.

---

### Task 1: arXiv backend and normalized literature source

**Files:**
- Create: `hypoforge/tools/arxiv_search.py`
- Create: `hypoforge/literature/sources/arxiv_source.py`
- Create: `tests/literature/test_arxiv_source.py`
- Modify: `hypoforge/literature/sources/__init__.py`
- Modify: `requirements.txt`

**Interfaces:**
- Produces: `ArxivTool.search_strict(query: str, limit: int = 20) -> list[dict]`.
- Produces: `ArxivSource.search(query: SearchQuery, limit: int = 20) -> list[PaperRecord]`.
- Produces: `ArxivBackendError(OSError)`.

- [ ] **Step 1: Write failing source-mapping tests**

```python
@pytest.mark.asyncio
async def test_arxiv_source_maps_versioned_result_to_canonical_record():
    source = ArxivSource(tool=FakeTool([{
        "arxiv_id": "2401.12345v2",
        "entry_url": "https://arxiv.org/abs/2401.12345v2",
        "pdf_url": "https://arxiv.org/pdf/2401.12345v2",
        "title": "  A useful preprint  ",
        "abstract": "Evidence.",
        "authors": ["Ada Lovelace"],
        "year": 2025,
        "doi": "https://doi.org/10.1000/ABC",
        "primary_category": "cs.AI",
    }]))
    papers = await source.search(query("arxiv"), limit=3)
    assert papers[0].paper_id == "ARXIV:2401.12345"
    assert papers[0].external_ids["arxiv"] == "2401.12345v2"
    assert papers[0].external_ids["pdf_url"].endswith("2401.12345v2")
    assert papers[0].doi == "10.1000/abc"
    assert papers[0].publication_type == "preprint"
    assert papers[0].fulltext_status is FulltextStatus.PDF_AVAILABLE

@pytest.mark.asyncio
async def test_arxiv_source_propagates_backend_failure():
    with pytest.raises(ArxivBackendError, match="arxiv backend failed"):
        await ArxivSource(tool=FailingTool()).search(query("arxiv"))
```

- [ ] **Step 2: Run the focused tests and confirm RED**

Run: `python -m pytest tests/literature/test_arxiv_source.py -q`

Expected: collection fails because `hypoforge.literature.sources.arxiv_source` does not exist.

- [ ] **Step 3: Implement the minimal backend**

```python
# hypoforge/tools/arxiv_search.py
async def search_strict(self, query: str, limit: int = 20) -> list[dict]:
    if limit <= 0:
        return []
    return await asyncio.to_thread(_search_arxiv_sync, query, limit)

def _search_arxiv_sync(query: str, limit: int) -> list[dict]:
    client = arxiv.Client(page_size=min(limit, 100), delay_seconds=3.0, num_retries=3)
    search = arxiv.Search(query=query, max_results=limit,
                          sort_by=arxiv.SortCriterion.Relevance)
    return [_result_to_dict(item) for item in client.results(search)]
```

`_result_to_dict` must extract `get_short_id()`, `entry_id`, `pdf_url`, normalized title/summary whitespace, author names, `published.year`, DOI, primary category, categories, journal reference, and comment. `ArxivTool.search` delegates to `search_strict` for protocol compatibility and does not swallow errors.

- [ ] **Step 4: Implement normalization and exports**

```python
# hypoforge/literature/sources/arxiv_source.py
_VERSION = re.compile(r"v\d+$", re.IGNORECASE)

def canonical_arxiv_id(value: str) -> str:
    clean = value.rsplit("/", 1)[-1].strip()
    return _VERSION.sub("", clean)

class ArxivSource(LiteratureSourceProtocol):
    source_name = "arxiv"

    async def search(self, query: SearchQuery, limit: int = 20) -> list[PaperRecord]:
        try:
            rows = await self._tool.search_strict(query.text, limit=limit)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise ArxivBackendError(f"arxiv backend failed: {type(exc).__name__}: {exc}") from exc
        return [_dict_to_record(row) for row in rows if _identifiable(row)]
```

Map PDF availability to `FulltextStatus.PDF_AVAILABLE`, abstracts to `ABSTRACT_ONLY`, leave citation count as `None`, and add `ArxivSource` to `sources.__all__`.

- [ ] **Step 5: Record dependencies and verify GREEN**

Add these lines under core dependencies:

```text
arxiv>=4.0.0
pypdf>=5.0
```

Run: `python -m pytest tests/literature/test_arxiv_source.py -q`

Expected: all focused tests pass.

---

### Task 2: Expose arXiv to the query planner and iterative agent

**Files:**
- Modify: `hypoforge/literature/search/search_tool.py`
- Modify: `tests/literature/test_teammate_search_tools.py`
- Modify: `tests/literature/test_integrated_factory.py`

**Interfaces:**
- Consumes: `ArxivSource` from Task 1.
- Produces: `LiteratureSearchTool(..., arxiv_source: ArxivSource | None = None)`.
- Produces: direct backend name `arxiv` and three-source `as_source_list()`.

- [ ] **Step 1: Write failing planner and dispatch tests**

```python
def test_search_tool_exposes_three_independent_sources():
    tool = LiteratureSearchTool(pubmed=fake_pubmed, academic=fake_academic,
                                arxiv_source=fake_arxiv)
    assert [item["name"] for item in tool.tool_definitions] == [
        "pubmed", "semantic_scholar", "arxiv"
    ]
    assert [item.source_name for item in tool.as_source_list()] == [
        "pubmed", "semantic_scholar", "arxiv"
    ]

@pytest.mark.asyncio
async def test_direct_search_dispatches_arxiv_query():
    records = await tool.search("graph neural networks", backend="arxiv", limit=4)
    assert fake_arxiv.calls[0].target_source == "arxiv"
    assert len(records) == 1
```

- [ ] **Step 2: Run the tests and confirm RED**

Run: `python -m pytest tests/literature/test_teammate_search_tools.py tests/literature/test_integrated_factory.py -q`

Expected: assertions fail because only PubMed and academic sources are exposed.

- [ ] **Step 3: Add arXiv definition, injection, and dispatch**

```python
{
    "name": "arxiv",
    "display_name": "arXiv",
    "description": (
        "Search preprints in computer science, mathematics, physics, statistics, "
        "electrical engineering, quantitative biology, quantitative finance, and "
        "economics. Prefer for recent technical work; results may not be peer reviewed."
    ),
}
```

Initialize `self._arxiv_source`, return it after the existing sources, accept `backend == "arxiv"`, and include its name in `_TOOL_NAME_SET` through the existing class derivation.

- [ ] **Step 4: Verify planner integration**

Run: `python -m pytest tests/literature/test_teammate_search_tools.py tests/literature/test_integrated_factory.py -q`

Expected: all tests pass and the factory source assertion expects exactly `pubmed`, `semantic_scholar`, and `arxiv`.

---

### Task 3: Resolve and cache only Final-K arXiv PDFs

**Files:**
- Create: `hypoforge/literature/reading/arxiv_resolver.py`
- Create: `hypoforge/literature/reading/routing.py`
- Create: `tests/literature/reading/test_arxiv_resolver.py`
- Modify: `hypoforge/literature/reading/__init__.py`

**Interfaces:**
- Produces: `ArxivPDFResolver(cache_dir, timeout_seconds=30.0, max_pdf_bytes=52_428_800, backend=None)`.
- Produces: `RoutingFulltextResolver(pmc_resolver, arxiv_resolver)`.
- Consumes: arXiv ID and PDF URL from `PaperRecord.external_ids`.

- [ ] **Step 1: Write failing resolver tests**

```python
@pytest.mark.asyncio
async def test_arxiv_resolver_downloads_valid_pdf_once_and_reuses_cache(tmp_path):
    calls = []
    async def fetch(url, timeout):
        calls.append(url)
        return b"%PDF-1.4\nvalid-test-payload"
    resolver = ArxivPDFResolver(tmp_path, backend=fetch)
    first = await resolver.resolve(arxiv_paper())
    second = await resolver.resolve(arxiv_paper())
    assert first.content_level is ContentLevel.PDF
    assert first.local_path == second.local_path
    assert len(calls) == 1

@pytest.mark.asyncio
async def test_arxiv_resolver_invalid_pdf_degrades_to_abstract(tmp_path):
    async def fetch(url, timeout):
        return b"<html>rate limited</html>"
    document = await ArxivPDFResolver(tmp_path, backend=fetch).resolve(arxiv_paper())
    assert document.content_level is ContentLevel.ABSTRACT
    assert "invalid PDF" in document.retrieval_error
```

Also test oversized payloads, missing IDs, cancellation, and `RoutingFulltextResolver` selecting arXiv only for arXiv records.

- [ ] **Step 2: Run tests and confirm RED**

Run: `python -m pytest tests/literature/reading/test_arxiv_resolver.py -q`

Expected: collection fails because the resolver modules do not exist.

- [ ] **Step 3: Implement bounded PDF resolution**

```python
async def resolve(self, paper: PaperRecord) -> DocumentRecord:
    arxiv_id = str(paper.external_ids.get("arxiv") or "").strip()
    if not arxiv_id:
        return self._abstract_document(paper, "arXiv identifier unavailable")
    source_uri = f"https://arxiv.org/pdf/{quote(arxiv_id, safe='.') }"
    path = self._paper_dir(paper) / "paper.pdf"
    if self._valid_cached_pdf(path):
        return self._pdf_document(paper, path, source_uri)
    try:
        payload = await self.backend(source_uri, self.timeout_seconds)
        if len(payload) > self.max_pdf_bytes:
            raise ValueError("arXiv PDF exceeds configured size limit")
        if not payload.lstrip().startswith(b"%PDF-"):
            raise ValueError("invalid PDF response from arXiv")
        self._write_atomic(path, payload)
        return self._pdf_document(paper, path, source_uri)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        return self._abstract_document(paper, _safe_error(exc))
```

The default backend reads at most `max_pdf_bytes + 1` bytes instead of reading an unbounded response. Abstract cache files use the existing `hypoforge_abstract_v1` shape.

- [ ] **Step 4: Implement source routing and exports**

```python
class RoutingFulltextResolver(FulltextResolverProtocol):
    async def resolve(self, paper: PaperRecord) -> DocumentRecord:
        if "arxiv" in paper.sources or paper.external_ids.get("arxiv"):
            return await self.arxiv_resolver.resolve(paper)
        return await self.pmc_resolver.resolve(paper)
```

Export `ArxivPDFResolver`, `ArxivPDFFetchBackend`, and `RoutingFulltextResolver` from `reading.__init__`.

- [ ] **Step 5: Verify GREEN and PMC regression**

Run: `python -m pytest tests/literature/reading/test_arxiv_resolver.py tests/literature/reading/test_resolver.py -q`

Expected: all tests pass; PMC resolver tests remain unchanged.

---

### Task 4: Parse PDF pages into attributable RAG chunks

**Files:**
- Modify: `hypoforge/literature/reading/parser.py`
- Modify: `hypoforge/literature/reading/routing.py`
- Create: `tests/literature/reading/test_pdf_parser.py`
- Modify: `hypoforge/literature/reading/__init__.py`

**Interfaces:**
- Produces: `PDFDocumentParser(target_chars=900, overlap_chars=120)`.
- Produces: `RoutingDocumentParser(document_parser, pdf_parser)`.
- Produces: `DocumentChunk.page` with one-based source page numbers.

- [ ] **Step 1: Write failing page-attribution tests**

```python
@pytest.mark.asyncio
async def test_pdf_parser_emits_page_aware_chunks(monkeypatch, tmp_path):
    monkeypatch.setattr(parser_module, "PdfReader", lambda path: FakeReader([
        "First page evidence.", "Second page evidence."
    ]))
    path = tmp_path / "paper.pdf"
    path.write_bytes(b"%PDF-1.4\nfixture")
    chunks = await PDFDocumentParser(target_chars=100).parse(pdf_document(path))
    assert [chunk.page for chunk in chunks] == [1, 2]
    assert [chunk.section for chunk in chunks] == ["page_1", "page_2"]
    assert all(chunk.start_offset is not None for chunk in chunks)

@pytest.mark.asyncio
async def test_pdf_parser_rejects_empty_extraction(monkeypatch, tmp_path):
    monkeypatch.setattr(parser_module, "PdfReader", lambda path: FakeReader(["", "  "]))
    with pytest.raises(DocumentParseError, match="no readable PDF pages"):
        await PDFDocumentParser().parse(pdf_document(tmp_path / "empty.pdf"))
```

- [ ] **Step 2: Run tests and confirm RED**

Run: `python -m pytest tests/literature/reading/test_pdf_parser.py -q`

Expected: import fails because `PDFDocumentParser` is missing.

- [ ] **Step 3: Implement async page extraction and stable chunks**

```python
class PDFDocumentParser(DocumentParserProtocol):
    async def parse(self, document: DocumentRecord) -> list[DocumentChunk]:
        if document.content_level is not ContentLevel.PDF:
            raise DocumentParseError(f"{document.document_id}: expected PDF content")
        pages = await asyncio.to_thread(self._extract_pages, Path(document.local_path))
        chunks = []
        ordinal = 0
        for page_number, text in pages:
            for start, end, chunk_text in _window_text(text, self.target_chars,
                                                        self.overlap_chars):
                ordinal += 1
                chunks.append(DocumentChunk(
                    chunk_id=_chunk_id(document, f"page_{page_number}", ordinal, chunk_text),
                    document_id=document.document_id, paper_id=document.paper_id,
                    section=f"page_{page_number}", page=page_number,
                    start_offset=start, end_offset=end, text=chunk_text,
                ))
        if not chunks:
            raise DocumentParseError(f"{document.document_id}: no readable PDF pages")
        return chunks
```

`_extract_pages` catches `pypdf` and filesystem exceptions and re-raises `DocumentParseError` with the document ID. Keep BioC and abstract behavior byte-for-byte compatible except for sharing the stable `_chunk_id` helper.

- [ ] **Step 4: Implement parser routing and exports**

```python
class RoutingDocumentParser(DocumentParserProtocol):
    async def parse(self, document: DocumentRecord) -> list[DocumentChunk]:
        if document.content_level is ContentLevel.PDF:
            return await self.pdf_parser.parse(document)
        return await self.document_parser.parse(document)
```

- [ ] **Step 5: Verify parser regression suite**

Run: `python -m pytest tests/literature/reading/test_pdf_parser.py tests/literature/reading/test_parser.py -q`

Expected: all tests pass, including existing BioC and abstract parsing tests.

---

### Task 5: Wire integrated pipeline, document behavior, and verify end to end

**Files:**
- Modify: `hypoforge/literature/integrated.py`
- Modify: `tests/literature/test_integrated_factory.py`
- Modify: `tests/literature/reading/test_workflow.py`
- Modify: `hypoforge/literature/README.md`

**Interfaces:**
- Consumes: all Task 1-4 interfaces.
- Produces: `build_integrated_search_adapter(..., arxiv_pdf_backend=None, arxiv_max_pdf_bytes=52_428_800)`.

- [ ] **Step 1: Write failing factory-routing test**

```python
def test_factory_wires_arxiv_search_pdf_resolution_and_parser(tmp_path):
    adapter = build_integrated_search_adapter(client=FakeClient(),
                                               reading_cache_dir=tmp_path)
    assert set(adapter.search_agent.sources) == {
        "pubmed", "semantic_scholar", "arxiv"
    }
    workflow = adapter.reading_workflow
    assert isinstance(workflow.resolver, RoutingFulltextResolver)
    assert isinstance(workflow.parser, RoutingDocumentParser)
```

Add a workflow-level test with one PDF `DocumentRecord` that asserts retrieved `EvidenceChunk.page == 1` is preserved by the reader input.

- [ ] **Step 2: Run integration tests and confirm RED**

Run: `python -m pytest tests/literature/test_integrated_factory.py tests/literature/reading/test_workflow.py -q`

Expected: factory assertions fail because it still wires only PMC and BioC.

- [ ] **Step 3: Wire resolver and parser routing**

```python
pmc_resolver = PMCFulltextResolver(cache_dir=reading_cache_dir,
                                   timeout_seconds=source_timeout_seconds,
                                   backend=pmc_backend)
arxiv_resolver = ArxivPDFResolver(cache_dir=reading_cache_dir,
                                  timeout_seconds=source_timeout_seconds,
                                  max_pdf_bytes=arxiv_max_pdf_bytes,
                                  backend=arxiv_pdf_backend)
reading_workflow = FullTextReadingWorkflow(
    resolver=RoutingFulltextResolver(pmc_resolver, arxiv_resolver),
    parser=RoutingDocumentParser(BioCDocumentParser(), PDFDocumentParser()),
    retriever=HybridEvidenceRetriever(store), reader=QwenPaperReader(client),
    store=store, reader_timeout_seconds=reader_timeout_seconds,
    workflow_timeout_seconds=reading_workflow_timeout_seconds,
)
```

- [ ] **Step 4: Document the operational boundary**

Update the literature README source table and flow so it states:

```text
PubMed -> PMC BioC when available -> abstract fallback
arXiv -> metadata/abstract search -> Final-K cached PDF -> page-aware RAG
Semantic Scholar/OpenAlex -> metadata/abstract -> current abstract fallback
```

Also state that arXiv papers are preprints and PDF retrieval does not imply a universal reuse licence.

- [ ] **Step 5: Run deterministic verification**

Run: `python -m pytest tests/literature tests/test_run_m2_integrated.py -q`

Expected: zero failures.

Run: `python -m pytest -q`

Expected: zero failures across the repository.

- [ ] **Step 6: Run bounded live smoke tests**

Run a direct arXiv search with `limit=3` for the prior real sub-question, record elapsed time, returned IDs, titles, abstract presence, and PDF URLs. Then pass one returned paper to `ArxivPDFResolver` and `PDFDocumentParser`, recording download time, byte size, page/chunk counts, and any real provider error.

Expected: either a real result with a valid cached PDF and non-empty page-attributed chunks, or an explicit real network/provider failure. No fabricated fallback papers are acceptable.

- [ ] **Step 7: Review the working tree without committing**

Run: `git status --short` and `git diff --check`.

Expected: no whitespace errors; only intended arXiv files plus the user's pre-existing changes are present. Do not run `git commit` or `git push`.
