# M2 Knowledge Export Interface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one typed M2 output that exposes every Final-K paper, retrieved evidence passage, extracted knowledge entry, and the provenance required for another team to construct an evidence graph.

**Architecture:** Cross-module export models live in `hypoforge/state.py`, alongside the existing M2 pipeline contract, to avoid a circular import from shared state back into `hypoforge.literature`. A pure builder maps `SearchRunResult` plus `PaperReadingResult` objects into one validated run; `AgenticM2Adapter` collects those runs into a versioned package while preserving the legacy `literature_results` output.

**Tech Stack:** Python 3.12, Pydantic v2, asyncio, pytest, pytest-asyncio.

## Global Constraints

- The public state field is named `m2_knowledge_export`.
- The schema version is exactly `m2-knowledge-export/v1`.
- Existing `literature_results` output remains unchanged.
- Every knowledge `evidence_id` must resolve to exactly one exported evidence passage in the same sub-question run.
- Export all retrieved evidence supplied to the reader, including evidence not cited by a final knowledge entry.
- Export Final-K paper metadata and reading/search provenance, but never full PDFs, full-text XML, local cache paths, or every parsed document chunk.
- The agentic integrated M2 path populates the package; legacy/minimal backup paths may leave it unset.
- Do not modify `hypoforge/modules/m3_evidence_graph.py` or any other M3 implementation file.
- Do not commit, push, merge, clean, or reset the shared worktree. Leave changes for user review.

---

### Task 1: Define the versioned pipeline contract and reference validation

**Files:**
- Modify: `hypoforge/state.py:12-15,98-123,228-232`
- Create: `tests/literature/test_knowledge_export_models.py`

**Interfaces:**
- Produces: `M2SearchQueryExport`, `M2CoverageExport`, `M2SearchProvenance`, `M2PaperExport`, `M2EvidenceExport`, `M2KnowledgeRun`, and `M2KnowledgeExport`.
- Extends: `PipelineState.m2_knowledge_export: Optional[M2KnowledgeExport]`.
- Consumes later: `build_m2_knowledge_export_run(...)` and `AgenticM2Adapter`.

- [ ] **Step 1: Write failing contract and JSON round-trip tests**

Create `tests/literature/test_knowledge_export_models.py` with a valid package fixture and assertions for both direct construction and pipeline-state serialization:

```python
from __future__ import annotations

import pytest
from pydantic import ValidationError

from hypoforge.state import (
    ConfidenceLevel,
    KnowledgeEntry,
    KnowledgeEntryType,
    M2CoverageExport,
    M2EvidenceExport,
    M2KnowledgeExport,
    M2KnowledgeRun,
    M2PaperExport,
    M2SearchProvenance,
    PipelineState,
)


def valid_run() -> M2KnowledgeRun:
    return M2KnowledgeRun(
        sub_question="How does force regulate YAP?",
        papers=[M2PaperExport(
            paper_id="PMID:1",
            title="Force and YAP",
            sources=["pubmed"],
            reading_summary="Force promotes YAP import.",
            content_level="structured_fulltext",
            document_id="document:1",
            document_source_uri="https://example.test/pmc/1",
            document_license="PMC Open Access subset",
        )],
        evidence=[M2EvidenceExport(
            evidence_id="ev-1",
            paper_id="PMID:1",
            chunk_id="chunk-1",
            section="results",
            page=3,
            quote="Force promoted YAP nuclear import.",
            normalized_claim="Force promotes YAP import.",
            relevance_score=0.95,
            citable=True,
        )],
        knowledge_entries=[KnowledgeEntry(
            id="knowledge-1",
            type=KnowledgeEntryType.MECHANISTIC_CONCLUSION,
            content="Force promotes YAP nuclear import.",
            confidence=ConfidenceLevel.HIGH,
            source_paper_id="PMID:1",
            source_paper_title="Force and YAP",
            entities=["YAP"],
            evidence_ids=["ev-1"],
        )],
        search_provenance=M2SearchProvenance(
            coverage=M2CoverageExport(sufficient=True),
            stop_reason="coverage_satisfied",
        ),
    )


def test_package_survives_pipeline_state_json_round_trip() -> None:
    package = M2KnowledgeExport(runs=[valid_run()])
    state = PipelineState(m2_knowledge_export=package)

    restored = PipelineState.model_validate_json(state.model_dump_json())

    assert restored.m2_knowledge_export == package
    assert package.schema_version == "m2-knowledge-export/v1"
    assert package.runs[0].evidence[0].quote.startswith("Force")
```

Add parameterized tests that mutate the fixture and expect `ValidationError` messages for each contract violation:

```python
@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda data: data["papers"].append(dict(data["papers"][0])), "duplicate paper_id"),
        (lambda data: data["evidence"].append(dict(data["evidence"][0])), "duplicate evidence_id"),
        (lambda data: data["knowledge_entries"].append(dict(data["knowledge_entries"][0])), "duplicate knowledge id"),
        (lambda data: data["evidence"][0].__setitem__("paper_id", "missing"), "unknown paper"),
        (lambda data: data["knowledge_entries"][0].__setitem__("source_paper_id", "missing"), "unknown paper"),
        (lambda data: data["knowledge_entries"][0].__setitem__("evidence_ids", ["missing"]), "unknown evidence"),
        (lambda data: data["knowledge_entries"][0].__setitem__("evidence_ids", []), "no evidence_ids"),
    ],
)
def test_run_rejects_broken_provenance(mutation, message) -> None:
    payload = valid_run().model_dump(mode="python")
    mutation(payload)

    with pytest.raises(ValidationError, match=message):
        M2KnowledgeRun.model_validate(payload)
```

- [ ] **Step 2: Run the new tests and confirm RED**

Run:

```powershell
& 'C:\Users\LIU\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' `
  -m pytest tests/literature/test_knowledge_export_models.py -q
```

Expected: collection fails because the `M2*Export` models do not exist.

- [ ] **Step 3: Implement the typed state models**

Extend the imports in `hypoforge/state.py`:

```python
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator, field_validator
```

Add these models after `LiteratureResult` and before the M3 models:

```python
class M2SearchQueryExport(BaseModel):
    query_id: str
    text: str
    round_index: int = 0
    intent: str = "core"
    target_source: str
    purpose: str
    target_gap: str = ""
    relation_to_question: str = ""


class M2CoverageExport(BaseModel):
    covered_buckets: List[str] = Field(default_factory=list)
    missing_buckets: List[str] = Field(default_factory=list)
    covered_topics: List[str] = Field(default_factory=list)
    missing_topics: List[str] = Field(default_factory=list)
    sufficient: bool = False
    rationale: str = ""


class M2SearchProvenance(BaseModel):
    queries: List[M2SearchQueryExport] = Field(default_factory=list)
    coverage: M2CoverageExport = Field(default_factory=M2CoverageExport)
    source_result_counts: Dict[str, int] = Field(default_factory=dict)
    failed_sources: List[str] = Field(default_factory=list)
    iterations: int = 0
    stop_reason: Optional[str] = None
    errors: List[str] = Field(default_factory=list)
    stage_elapsed_seconds: Dict[str, float] = Field(default_factory=dict)
    papers_found: int = 0
    papers_after_dedup: int = 0


class M2PaperExport(BaseModel):
    paper_id: str
    title: str
    abstract: str = ""
    authors: List[str] = Field(default_factory=list)
    year: Optional[int] = None
    journal: str = ""
    doi: str = ""
    pmid: str = ""
    pmcid: str = ""
    external_ids: Dict[str, str] = Field(default_factory=dict)
    citation_count: Optional[int] = None
    publication_type: str = ""
    sources: List[str] = Field(default_factory=list)
    is_open_access: Optional[bool] = None
    fulltext_status: str = "unknown"
    rank_scores: Dict[str, float] = Field(default_factory=dict)
    reading_summary: str = ""
    content_level: str = "metadata"
    document_id: str = ""
    document_source_uri: str = ""
    document_license: str = ""
    degraded_to_abstract: bool = False
    chunks_parsed: int = 0
    chunks_retrieved: int = 0
    stage_elapsed_seconds: Dict[str, float] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)


class M2EvidenceExport(BaseModel):
    evidence_id: str
    paper_id: str
    chunk_id: str
    section: str = ""
    page: Optional[int] = None
    quote: str
    normalized_claim: str
    relevance_score: float
    citable: bool = True


class M2KnowledgeRun(BaseModel):
    sub_question: str
    papers: List[M2PaperExport] = Field(default_factory=list)
    evidence: List[M2EvidenceExport] = Field(default_factory=list)
    knowledge_entries: List[KnowledgeEntry] = Field(default_factory=list)
    search_provenance: M2SearchProvenance = Field(
        default_factory=M2SearchProvenance
    )

    @model_validator(mode="after")
    def validate_provenance(self) -> "M2KnowledgeRun":
        paper_ids = [item.paper_id for item in self.papers]
        if len(paper_ids) != len(set(paper_ids)):
            raise ValueError("duplicate paper_id in M2 knowledge run")

        evidence_ids = [item.evidence_id for item in self.evidence]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("duplicate evidence_id in M2 knowledge run")

        knowledge_ids = [item.id for item in self.knowledge_entries]
        if len(knowledge_ids) != len(set(knowledge_ids)):
            raise ValueError("duplicate knowledge id in M2 knowledge run")

        paper_set = set(paper_ids)
        evidence_set = set(evidence_ids)
        for item in self.evidence:
            if item.paper_id not in paper_set:
                raise ValueError(
                    f"evidence {item.evidence_id!r} references unknown paper"
                )
        for item in self.knowledge_entries:
            if item.source_paper_id not in paper_set:
                raise ValueError(
                    f"knowledge {item.id!r} references unknown paper"
                )
            if not item.evidence_ids:
                raise ValueError(f"knowledge {item.id!r} has no evidence_ids")
            unknown = [
                evidence_id
                for evidence_id in item.evidence_ids
                if evidence_id not in evidence_set
            ]
            if unknown:
                raise ValueError(
                    f"knowledge {item.id!r} references unknown evidence: {unknown}"
                )
        return self


class M2KnowledgeExport(BaseModel):
    schema_version: Literal["m2-knowledge-export/v1"] = (
        "m2-knowledge-export/v1"
    )
    runs: List[M2KnowledgeRun] = Field(default_factory=list)
```

Add the optional state field immediately after `literature_results`:

```python
m2_knowledge_export: Optional[M2KnowledgeExport] = None
```

- [ ] **Step 4: Run contract tests and existing state tests**

Run:

```powershell
& 'C:\Users\LIU\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' `
  -m pytest tests/literature/test_knowledge_export_models.py `
  tests/test_pipeline.py tests/literature/test_models.py -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Review checkpoint without committing**

Run `git diff --check -- hypoforge/state.py tests/literature/test_knowledge_export_models.py` and inspect only these Task 1 changes. Do not commit.

---

### Task 2: Preserve portable document provenance through the reading workflow

**Files:**
- Modify: `hypoforge/literature/models.py:238-252`
- Modify: `hypoforge/literature/reading/workflow.py:153-363`
- Modify: `tests/literature/reading/test_workflow.py`

**Interfaces:**
- Extends: `PaperReadingResult.document_source_uri: str` and `PaperReadingResult.document_license: str`.
- Consumes: `DocumentRecord.source_uri` and `DocumentRecord.license`.
- Produces later: portable provenance consumed by `build_m2_knowledge_export_run(...)`.

- [ ] **Step 1: Write failing workflow provenance tests**

Add a resolver that returns a source URI and licence to `tests/literature/reading/test_workflow.py`, then assert both successful and metadata-only results preserve them:

```python
@pytest.mark.asyncio
async def test_workflow_preserves_document_source_and_license() -> None:
    class ProvenanceResolver:
        async def resolve(self, item: PaperRecord) -> DocumentRecord:
            return DocumentRecord(
                document_id="doc:licensed",
                paper_id=item.paper_id,
                content_level=ContentLevel.STRUCTURED_FULLTEXT,
                source_uri="https://example.test/fulltext/1",
                local_path="fulltext.json",
                license="CC BY 4.0",
            )

    workflow = FullTextReadingWorkflow(
        resolver=ProvenanceResolver(),
        parser=FakeParser(),
        retriever=FakeRetriever(),
        reader=FakeReader(),
        store=InMemoryChunkStore(),
    )

    result = (await workflow.run("question", [paper("p1")]))[0]

    assert result.document_source_uri == "https://example.test/fulltext/1"
    assert result.document_license == "CC BY 4.0"
```

Add a second test whose resolver returns `ContentLevel.METADATA` with the same
two fields and assert they are retained even though no parsing occurs.

- [ ] **Step 2: Run the focused tests and confirm RED**

Run:

```powershell
& 'C:\Users\LIU\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' `
  -m pytest tests/literature/reading/test_workflow.py `
  -k 'document_source or document_license' -q
```

Expected: failures because `PaperReadingResult` has no provenance fields.

- [ ] **Step 3: Add fields and one workflow mapping helper**

Add to `PaperReadingResult` in `hypoforge/literature/models.py`:

```python
document_source_uri: str = ""
document_license: str = ""
```

Add to `FullTextReadingWorkflow`:

```python
@staticmethod
def _document_result_fields(document: DocumentRecord) -> dict[str, str]:
    return {
        "document_id": document.document_id,
        "document_source_uri": document.source_uri,
        "document_license": document.license,
    }
```

Import `DocumentRecord` if it is not already imported. In every
`PaperReadingResult(...)` created after a `document` exists, replace the lone
`document_id=document.document_id` argument with:

```python
**self._document_result_fields(document)
```

When a parser fallback assigns `document = fallback_document`, subsequent
results must use the fallback document provenance. Extend the final
`reading.model_copy(update={...})` mapping with:

```python
**self._document_result_fields(document),
```

Branches that fail before any `DocumentRecord` exists retain empty defaults.

- [ ] **Step 4: Verify all reading tests**

Run:

```powershell
& 'C:\Users\LIU\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' `
  -m pytest tests/literature/reading -q
```

Expected: all reading tests pass, including timeout and abstract-degradation paths.

- [ ] **Step 5: Review checkpoint without committing**

Run `git diff --check -- hypoforge/literature/models.py hypoforge/literature/reading/workflow.py tests/literature/reading/test_workflow.py`. Do not commit.

---

### Task 3: Build a deterministic M2 export run from search and reading results

**Files:**
- Create: `hypoforge/literature/export.py`
- Modify: `hypoforge/literature/__init__.py`
- Create: `tests/literature/test_knowledge_export_builder.py`

**Interfaces:**
- Consumes: `SearchRunResult`, `PaperReadingResult`, and the Task 1 state models.
- Produces: `build_m2_knowledge_export_run(sub_question, search_result, reading_results) -> M2KnowledgeRun`.
- Preserves: Final-K order, per-paper evidence order, and per-paper knowledge order.

- [ ] **Step 1: Write a failing complete-mapping test**

Create a Final-K paper with full metadata, a reading result with one evidence
passage and one linked knowledge entry, and a search result with query,
coverage, errors, and timings. Assert every field survives:

```python
def test_builder_exports_papers_evidence_knowledge_and_search_provenance() -> None:
    paper = PaperRecord(
        paper_id="PMID:1",
        title="Force and YAP",
        abstract="Force changes YAP transport.",
        authors=["A. Author"],
        year=2025,
        doi="10.1/example",
        pmid="1",
        sources=["pubmed"],
        is_open_access=True,
        rank_scores={"relevance": 0.9},
    )
    evidence = EvidenceChunk(
        evidence_id="ev-1",
        paper_id="PMID:1",
        chunk_id="chunk-1",
        section="results",
        page=4,
        quote="Force changed YAP transport.",
        normalized_claim="Force changes YAP transport.",
        relevance_score=0.95,
    )
    reading = PaperReadingResult(
        paper_id="PMID:1",
        summary="Force promotes transport.",
        evidence=[evidence],
        knowledge_entries=[EvidenceLinkedKnowledge(
            entry_id="knowledge-1",
            entry_type=KnowledgeEntryType.MECHANISTIC_CONCLUSION,
            content="Force changes YAP transport.",
            confidence=ConfidenceLevel.HIGH,
            entities=["YAP"],
            evidence_ids=["ev-1"],
        )],
        content_level=ContentLevel.STRUCTURED_FULLTEXT,
        document_id="doc-1",
        document_source_uri="https://example.test/fulltext/1",
        document_license="CC BY 4.0",
        chunks_parsed=8,
        chunks_retrieved=1,
    )
    search = SearchRunResult(
        sub_question="question",
        queries=[SearchQuery(
            query_id="q-1",
            text="YAP force",
            round_index=1,
            target_source="pubmed",
            purpose="mechanism",
            relation_to_question="Direct mechanism evidence.",
        )],
        final_papers=[paper],
        coverage=CoverageReport(sufficient=True),
        source_result_counts={"pubmed": 1},
        stop_reason=StopReason.COVERAGE_SATISFIED,
        stage_elapsed_seconds={"source_search": 0.2},
    )

    run = build_m2_knowledge_export_run("question", search, [reading])

    assert [item.paper_id for item in run.papers] == ["PMID:1"]
    assert run.papers[0].document_license == "CC BY 4.0"
    assert run.evidence[0].quote == "Force changed YAP transport."
    assert run.knowledge_entries[0].evidence_ids == ["ev-1"]
    assert run.search_provenance.queries[0].target_source == "pubmed"
    assert run.search_provenance.stop_reason == "coverage_satisfied"
```

Add tests for stable Final-K ordering, a zero-paper run, duplicate reading
results, a missing reading result, and a reading result for an unknown paper.
The last three must raise `ValueError` with the relevant paper ID.

- [ ] **Step 2: Run builder tests and confirm RED**

Run:

```powershell
& 'C:\Users\LIU\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' `
  -m pytest tests/literature/test_knowledge_export_builder.py -q
```

Expected: import failure because `hypoforge.literature.export` does not exist.

- [ ] **Step 3: Implement the pure builder**

Create `hypoforge/literature/export.py`. Use small private mapping functions
and this public orchestration shape:

```python
from __future__ import annotations

from collections.abc import Sequence

from ..state import (
    KnowledgeEntry,
    M2CoverageExport,
    M2EvidenceExport,
    M2KnowledgeRun,
    M2PaperExport,
    M2SearchProvenance,
    M2SearchQueryExport,
)
from .models import PaperReadingResult, PaperRecord, SearchRunResult


def _enum_value(value: object) -> str:
    return str(getattr(value, "value", value or ""))


def _paper_export(
    paper: PaperRecord,
    reading: PaperReadingResult,
) -> M2PaperExport:
    return M2PaperExport(
        paper_id=paper.paper_id,
        title=paper.title,
        abstract=paper.abstract,
        authors=list(paper.authors),
        year=paper.year,
        journal=paper.journal,
        doi=paper.doi,
        pmid=paper.pmid,
        pmcid=paper.pmcid,
        external_ids=dict(paper.external_ids),
        citation_count=paper.citation_count,
        publication_type=paper.publication_type,
        sources=list(paper.sources),
        is_open_access=paper.is_open_access,
        fulltext_status=_enum_value(paper.fulltext_status),
        rank_scores=dict(paper.rank_scores),
        reading_summary=reading.summary,
        content_level=_enum_value(reading.content_level),
        document_id=reading.document_id,
        document_source_uri=reading.document_source_uri,
        document_license=reading.document_license,
        degraded_to_abstract=reading.degraded_to_abstract,
        chunks_parsed=reading.chunks_parsed,
        chunks_retrieved=reading.chunks_retrieved,
        stage_elapsed_seconds=dict(reading.stage_elapsed_seconds),
        errors=list(reading.errors),
    )
```

Map evidence and knowledge without changing content:

```python
def _evidence_exports(reading: PaperReadingResult) -> list[M2EvidenceExport]:
    return [M2EvidenceExport(**item.model_dump(mode="python")) for item in reading.evidence]


def _knowledge_exports(
    paper: PaperRecord,
    reading: PaperReadingResult,
) -> list[KnowledgeEntry]:
    return [
        KnowledgeEntry(
            id=item.entry_id,
            type=item.entry_type,
            content=item.content,
            confidence=item.confidence,
            source_paper_id=paper.paper_id,
            source_paper_title=paper.title,
            entities=list(item.entities),
            evidence_ids=list(item.evidence_ids),
        )
        for item in reading.knowledge_entries
    ]
```

In `build_m2_knowledge_export_run`, reject duplicate, missing, and unknown
reading IDs before mapping. Iterate `search_result.final_papers`, not the input
reading order. Map query enums and coverage bucket sets to strings, sorting
sets for deterministic JSON. Let `M2KnowledgeRun` perform final referential
validation.

- [ ] **Step 4: Export the builder from the package boundary**

Add to `hypoforge/literature/__init__.py`:

```python
from .export import build_m2_knowledge_export_run
```

and include `"build_m2_knowledge_export_run"` in `__all__`.

- [ ] **Step 5: Verify builder and model tests together**

Run:

```powershell
& 'C:\Users\LIU\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' `
  -m pytest tests/literature/test_knowledge_export_models.py `
  tests/literature/test_knowledge_export_builder.py -q
```

Expected: all tests pass and the builder performs no I/O or model calls.

- [ ] **Step 6: Review checkpoint without committing**

Run `git diff --check -- hypoforge/literature/export.py hypoforge/literature/__init__.py tests/literature/test_knowledge_export_builder.py`. Do not commit.

---

### Task 4: Wire the export into M2 without changing M3

**Files:**
- Modify: `hypoforge/literature/adapter.py:1-103`
- Modify: `tests/literature/test_adapter.py`
- Modify: `tests/literature/test_integrated_factory.py`
- Modify: `hypoforge/literature/README.md`

**Interfaces:**
- Consumes: `build_m2_knowledge_export_run(...)` and `M2KnowledgeExport`.
- Produces: `AgenticM2Adapter.__call__ -> {"literature_results": ..., "m2_knowledge_export": M2KnowledgeExport}`.
- Extends: `AgenticM2Adapter.get_output_fields() -> ["literature_results", "m2_knowledge_export"]`.

- [ ] **Step 1: Update the adapter test fixture to contain real linked evidence**

The existing adapter test creates knowledge with `evidence_ids=["evidence-1"]`
but no evidence object. Add this evidence to its `PaperReadingResult`:

```python
evidence=[EvidenceChunk(
    evidence_id="evidence-1",
    paper_id="paper-1",
    chunk_id="chunk-1",
    section="results",
    page=2,
    quote="The mechanism depends on ATP.",
    normalized_claim="The mechanism depends on ATP.",
    relevance_score=0.9,
)],
```

Import `EvidenceChunk` from `hypoforge.literature.models`.

- [ ] **Step 2: Write failing adapter-output assertions**

Extend `test_adapter_maps_evidence_linked_entries_to_legacy_literature_results`:

```python
package = output["m2_knowledge_export"]
assert package.schema_version == "m2-knowledge-export/v1"
assert len(package.runs) == 1
assert package.runs[0].papers[0].paper_id == "paper-1"
assert package.runs[0].evidence[0].evidence_id == "evidence-1"
assert package.runs[0].knowledge_entries[0] == entry
assert set(adapter.get_output_fields()) == {
    "literature_results",
    "m2_knowledge_export",
}
```

Add a two-sub-question test asserting `package.runs` follows the original
sub-question order. Add an integrated-factory assertion proving its real
adapter returns a package whose evidence IDs resolve.

- [ ] **Step 3: Run adapter tests and confirm RED**

Run:

```powershell
& 'C:\Users\LIU\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' `
  -m pytest tests/literature/test_adapter.py `
  tests/literature/test_integrated_factory.py -q
```

Expected: `m2_knowledge_export` is absent.

- [ ] **Step 4: Refactor the adapter to use the export builder**

Import:

```python
from ..state import LiteratureResult, M2KnowledgeExport
from .export import build_m2_knowledge_export_run
```

Initialize both output collections:

```python
literature_results: list[LiteratureResult] = []
export_runs = []
```

After each `reading_workflow.run(...)`, build one run:

```python
export_run = build_m2_knowledge_export_run(
    sub_question,
    search_result,
    reading_results,
)
export_runs.append(export_run)
literature_results.append(LiteratureResult(
    sub_question=sub_question,
    papers_retrieved=len(export_run.papers),
    knowledge_entries=list(export_run.knowledge_entries),
))
```

Delete the old duplicate mapping loop. Return:

```python
return {
    "literature_results": literature_results,
    "m2_knowledge_export": M2KnowledgeExport(runs=export_runs),
}
```

Change `get_output_fields()` to return both field names. Do not edit M3.

- [ ] **Step 5: Verify adapter, integrated path, state, and pipeline behavior**

Run:

```powershell
& 'C:\Users\LIU\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' `
  -m pytest tests/literature/test_adapter.py `
  tests/literature/test_integrated_factory.py `
  tests/test_pipeline.py tests/test_run_m2_integrated.py -q
```

Expected: all selected tests pass. Existing fake adapters that deliberately
return only `literature_results` remain valid test doubles; only the real
`AgenticM2Adapter` promises the new package.

- [ ] **Step 6: Document the M2-owned contract**

Add an `M2 knowledge export boundary` section to
`hypoforge/literature/README.md` containing:

```text
Agentic M2 returns both literature_results and m2_knowledge_export.
m2_knowledge_export/v1 contains Final-K paper metadata, reading diagnostics,
all retrieved evidence excerpts, extracted knowledge entries, and search
provenance. Every knowledge evidence_id resolves inside the same run. The
package excludes full papers, local cache paths, and graph construction.
Downstream modules own their own adaptation to this M2 contract.
```

- [ ] **Step 7: Run complete deterministic verification**

Run:

```powershell
& 'C:\Users\LIU\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m pytest -q
& 'C:\Users\LIU\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m compileall -q hypoforge scripts tests
git diff --check
```

Expected: the full suite passes, compilation exits zero, and `git diff --check`
reports no whitespace errors. CRLF conversion warnings are acceptable.

- [ ] **Step 8: Confirm scope and preserve the worktree**

Run:

```powershell
git diff --name-only
git status --branch --short
```

Confirm no M3 file changed, no API key or runtime cache is tracked, and no
commit/push/merge occurred. Present the uncommitted diff and test evidence to
the user.
