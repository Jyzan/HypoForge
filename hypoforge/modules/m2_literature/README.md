# Agentic literature package

The canonical `hypoforge.modules.m2_literature` package defines the Agentic M2
contracts, source adapters, query planning, iterative search, ranking,
full-text reading, and provenance export.  The historical
`hypoforge.literature` namespace is a temporary compatibility alias: old deep
imports resolve to the same canonical module objects, but new code must use
the modules namespace.

## M2 data flow

```text
ProblemCard -> SearchQuery -> PaperRecord -> DocumentChunk
           -> EvidenceChunk -> EvidenceLinkedKnowledge
           -> M2KnowledgeExport -> M3 Grounding
```

`EvidenceChunk` is the citable unit. It retains `evidence_id`, `paper_id`,
`chunk_id`, `page`, and the original `quote`. `EvidenceLinkedKnowledge` keeps
`evidence_ids` that must resolve within the same M2 export run.

`M2LiteratureSearch` in `hypoforge.modules` is the Pipeline-facing M2 entry
point. It delegates to `m2_literature.AgenticM2Module`, while the
lower-level `AgenticM2Adapter` remains the dependency-injection executor used
by the module and integrated runner. The Literature package itself does not
register a Pipeline module.

## Variants

- `integrated`: Qwen query planning plus the configured academic sources and
  full-text reading workflow.
- `minimal`: a small PubMed-oriented setup for isolated smoke tests.

Both variants emit the same evidence-linked contracts. Neither returns a
metadata-only claim as grounded knowledge.

## Grounding rule

M3 consumes `state.m2_knowledge_export`, not paper metadata from
`literature_results`. A metadata cache hit is useful for discovery but cannot
close a grounding gap until the reader emits citable evidence.

## Progress output

Long M2 runs emit structured, human-readable progress events. The display
helper in `hypoforge/display/m2_progress.py` renders stage, sub-question,
source, paper counts, and elapsed time without changing the JSON contracts.
