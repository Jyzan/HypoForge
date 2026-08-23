# Agentic M2 adapter

`hypoforge/literature/adapter.py` contains the only M2 implementation used by
this repository.

## Registration and construction

- `M2LiteratureSearch` in `hypoforge.modules` is the `ModuleRegistry` entry
  point and a thin Pipeline facade.
- `AgenticM2Module` is the Literature-layer configuration wrapper delegated to
  by the facade.
- `AgenticM2Adapter` is the dependency-injection executor used by the wrapper
  and by the integrated runner.
- `build_adapter_from_config()` selects the `integrated` or `minimal` Agentic
  variant. There is no legacy M2 implementation or compatibility shim.

## Runtime flow

```text
M1 ProblemCard
  -> QueryPlanner
  -> IterativeSearchAgent
  -> Scout / PaperRanker
  -> ReadingWorkflow
  -> M2KnowledgeExport
  -> M3 Grounding
```

The search agent plans queries, searches enabled sources, deduplicates papers,
uses semantic Scout notes, applies the LLM retain/reject judge when available,
and checks coverage. Retained papers then pass through the reading workflow,
which produces citable evidence and evidence-linked knowledge.

## M2 -> M3 contract

`literature_results` is the pipeline-facing summary. `m2_knowledge_export` is
the authoritative provenance export consumed by M3.

Each evidence item retains:

- `evidence_id`
- `paper_id`
- `chunk_id`
- `page`
- `quote`

Each knowledge entry retains one or more `evidence_ids`, and every ID must
resolve to an evidence item in the same export run. M3 must consume this export
directly rather than reconstructing provenance from paper metadata.

A paper metadata cache hit is not grounding evidence. It stays open until the
reading workflow produces citable evidence and evidence-linked knowledge.

## Configuration

```yaml
search:
  implementation: agentic
```

`agentic` is the only accepted implementation value.

## Supplement rounds

Supplement searches are cache-first and gap-driven. Cached metadata may be
reused for discovery, but it cannot close an evidence gap by itself. New
citable evidence is appended as a new export run so historical M3 references
remain stable.
