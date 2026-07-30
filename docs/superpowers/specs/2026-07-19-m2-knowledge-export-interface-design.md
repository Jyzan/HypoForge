# M2 Knowledge Export Interface Design

## Objective

M2 owns literature retrieval, document reading, RAG evidence selection, and
knowledge extraction. It must expose one complete, stable, machine-readable
package from which another module can build an evidence graph without reading
M2 internals or reopening papers.

M3 is not modified by this work. The M3 team is responsible for adapting its
consumer to the M2 contract.

## Current Gap

The existing adapter returns `literature_results`, whose nested
`KnowledgeEntry` objects contain `evidence_ids`. It discards the corresponding
`EvidenceChunk` objects, most paper metadata, reading diagnostics, and search
provenance. A consumer can therefore see a claim and an evidence identifier but
cannot resolve that identifier to the quoted passage, section, page, or
relevance score.

The new interface must preserve the existing `literature_results` output for
compatibility while adding a complete typed export.

## Public Output

`AgenticM2Adapter.__call__` returns both fields:

```python
{
    "literature_results": list[LiteratureResult],
    "m2_knowledge_export": M2KnowledgeExport,
}
```

`PipelineState` adds:

```python
m2_knowledge_export: M2KnowledgeExport | None = None
```

The package is JSON-serializable and checkpoint-safe. Its schema version is
`m2-knowledge-export/v1`.

## Contract Models

The cross-module contract models live with the other pipeline-state models in
`hypoforge/state.py`. This avoids importing `hypoforge.literature` from the
shared state module and creating a circular dependency.

```text
M2KnowledgeExport
├── schema_version
└── runs: list[M2KnowledgeRun]
    ├── sub_question
    ├── papers: list[M2PaperExport]
    ├── evidence: list[M2EvidenceExport]
    ├── knowledge_entries: list[KnowledgeEntry]
    └── search_provenance: M2SearchProvenance
```

### `M2PaperExport`

Each Final-K paper exports:

- bibliographic identity: `paper_id`, title, abstract, authors, year, journal,
  DOI, PMID, PMCID, external identifiers, publication type, and sources;
- availability and ranking: citation count, open-access state, full-text
  status, and rank scores;
- reading outcome: summary, content level, document ID, document source URI,
  document licence, abstract-degradation flag, parsed/retrieved chunk counts,
  stage timings, and sanitized errors.

Local cache paths are deliberately excluded because they are machine-specific
implementation details and are not required to construct a graph.

### `M2EvidenceExport`

Every evidence passage returned by the RAG retriever exports:

- `evidence_id`, `paper_id`, and `chunk_id`;
- section and optional page number;
- verbatim quote and normalized claim;
- relevance score and `citable` status.

This list contains all retrieved evidence supplied to the paper reader, not
only evidence eventually cited by a knowledge entry. M3 may therefore make its
own relation and filtering decisions.

### `KnowledgeEntry`

The existing pipeline `KnowledgeEntry` remains the canonical claim model. Each
entry includes:

- stable ID, type, content, and confidence;
- source paper ID and title;
- extracted entities;
- one or more `evidence_ids`.

No second claim representation is introduced.

### `M2SearchProvenance`

Each sub-question exports the information needed to audit how its evidence set
was assembled:

- executed queries, including round, source, purpose, gap, and relation;
- coverage report;
- source result counts and failed sources;
- iteration count, stop reason, sanitized errors, and stage timings;
- papers found before and after deduplication.

Intermediate candidate papers and full Scout notes are excluded. They are
search-control state, not validated knowledge or evidence, and would make the
contract unnecessarily large.

## Export Builder

M2 exposes a public, deterministic builder in its package boundary:

```python
build_m2_knowledge_export_run(
    sub_question: str,
    search_result: SearchRunResult,
    reading_results: Sequence[PaperReadingResult],
) -> M2KnowledgeRun
```

The adapter invokes this function after reading each sub-question and appends
the returned run to `M2KnowledgeExport.runs`. Ordering is stable:

1. sub-question input order;
2. Final-K paper order;
3. evidence order returned for each paper;
4. knowledge-entry order returned for each paper.

The builder performs mapping only. It does not search, call a model, build a
graph, persist files, or mutate its inputs.

## Provenance Integrity

The export must satisfy all of these invariants:

1. Every exported knowledge entry has a source paper present in `papers`.
2. Every exported evidence item has a source paper present in `papers`.
3. Every `KnowledgeEntry.evidence_ids` value resolves to exactly one item in
   the same run's `evidence` list.
4. Evidence IDs are unique within a run.
5. Knowledge IDs are unique within a run. The same stable claim may appear in
   more than one sub-question run, allowing a consumer to merge it explicitly.
6. No fabricated placeholder is added for a missing paper or missing evidence.

`M2KnowledgeRun` validates the run-local invariants when constructed. An
invariant violation is a programming/contract error and must raise a clear
exception rather than emit an untraceable claim. Normal source failures and
abstract degradation remain data in the paper and provenance error fields.

## Document Provenance Extension

`PaperReadingResult` currently preserves `document_id` and content level but
drops the resolved document's source URI and licence. It is extended with:

```python
document_source_uri: str = ""
document_license: str = ""
```

`FullTextReadingWorkflow` copies these values from `DocumentRecord` into every
normal and degraded reading result. This provides portable provenance without
exposing the local cache path.

## Compatibility Boundary

- Existing `literature_results` remains unchanged.
- The new output is additive and does not change legacy callers.
- The agentic integrated M2 path must populate `m2_knowledge_export`.
- Legacy or minimal backup implementations may leave the optional export field
  unset until they can supply evidence-resolvable output.
- M3 code, graph schemas, relation extraction, and graph persistence are out of
  scope.
- The export contains evidence excerpts, not complete PDFs, full-text XML, or
  every parsed document chunk.

## Error and Empty-Result Semantics

- A sub-question with zero Final-K papers still produces an `M2KnowledgeRun`
  with provenance and empty paper/evidence/knowledge lists.
- A metadata-only or unreadable paper remains in `papers` with its errors and
  contributes no evidence or knowledge.
- Source HTTP failures remain in `search_provenance.errors` and
  `failed_sources`.
- A knowledge-to-evidence reference mismatch raises a contract error.
- Export construction performs no silent fallback and fabricates no content.

## Tests

Implementation follows red-green-refactor cycles. Tests must prove:

1. The adapter returns both the legacy result and the new export package.
2. Paper metadata, reading summary, content level, document provenance,
   degradation state, timings, and errors survive mapping.
3. Evidence quote, normalized claim, section, page, relevance, and citable
   state survive mapping.
4. Knowledge entries keep source-paper and evidence links.
5. Multiple papers and sub-questions have stable ordering and unique IDs.
6. Missing evidence references, duplicate evidence IDs, and unknown paper IDs
   are rejected.
7. Empty and failed readings still produce valid audit records.
8. `PipelineState.model_dump(mode="json")` and checkpoint reconstruction retain
   the complete package.
9. Existing adapter, M2, pipeline, and full repository tests remain green.

## Acceptance Criteria

- One typed M2 output contains every Final-K paper, every retrieved evidence
  passage, every extracted knowledge entry, and sufficient search/reading
  provenance to audit them.
- Every claim is resolvable to its evidence quote and source paper without
  accessing M2 internals or reopening a document.
- The package is JSON serializable and survives pipeline checkpointing.
- Existing `literature_results` consumers remain compatible.
- No M3 implementation file is changed.
- No full paper or unbounded document content is placed in pipeline state.
