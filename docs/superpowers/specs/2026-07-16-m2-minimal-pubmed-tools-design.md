# M2 Minimal PubMed Tools Design

## Goal

Provide a small, explicit implementation set that can run the agentic M2 path
from one sub-question to a reasonable `LiteratureResult` while teammates are
still implementing the production Tools.

Only PubMed retrieval is real. Query planning, deduplication, ranking, Scout
Reading, coverage evaluation, and abstract reading use deterministic rules.
The implementation must never fabricate papers, silently fall back to stub
data, or change the default legacy M2 behavior.

## Scope

Included:

- one deterministic PubMed query per sub-question;
- one strict PubMed source implementing `LiteratureSourceProtocol`;
- exact metadata deduplication;
- deterministic metadata ranking;
- lightweight title/abstract Scout notes;
- one-source coverage evaluation;
- abstract-only evidence and knowledge extraction;
- an explicit factory that assembles `AgenticM2Adapter`;
- an M2-only CLI that does not run M1;
- offline tests for success and failure paths, plus a manual real-network run.

Excluded:

- multiple literature websites;
- Qwen or any other LLM;
- synonym expansion, MeSH generation, citation chaining, or iterative gap
  discovery;
- PDF/full-text download, document parsing, chunk storage, vector retrieval,
  or RAG;
- fabricated papers or synthetic fallback results;
- automatic activation through the default `legacy` or unconfigured
  `agentic` path.

## Architecture

```text
one M2 sub-question
  -> RuleBasedQueryPlanner
  -> PubMedLiteratureSource
  -> ExactPaperDeduplicator
  -> MetadataPaperRanker
  -> AbstractScoutReader
  -> SingleSourceCoverageEvaluator
  -> IterativeSearchAgent
  -> AbstractReadingWorkflow
  -> AgenticM2Adapter
  -> LiteratureResult
```

`build_minimal_pubmed_adapter()` is the only assembly point. It creates every
minimal Tool and injects them into the existing `IterativeSearchAgent` and
`AgenticM2Adapter`. The factory is explicit: importing it does not register a
module, replace legacy M2, or enable fallback behavior.

## Components

### RuleBasedQueryPlanner

Implements `QueryPlannerProtocol`.

- On the first round, collect Latin biomedical tokens from `key_entities` and
  the sub-question, case-insensitively deduplicate them, keep at most six, and
  join them with `AND`.
- For the accepted example, `Hippo`, `YAP`, and `TAZ` yield a usable PubMed
  query even though the surrounding question is Chinese.
- If no Latin token exists, use the original sub-question unchanged.
- Emit one deterministic `SearchQuery` targeting `pubmed`.
- After a query has already been used, emit no additional query. This minimal
  implementation does not perform iterative refinement.

### PubMedLiteratureSource

Implements `LiteratureSourceProtocol` with `source_name = "pubmed"`.

- Reuse the existing NCBI E-utilities parsing code.
- Add a public strict PubMed search helper that runs the blocking E-utilities
  work in `asyncio.to_thread` and lets network, timeout, HTTP, XML, and parsing
  exceptions propagate.
- Preserve the old `PubMedTool.search()` behavior for legacy callers.
- Convert returned dictionaries into `PaperRecord` objects.
- Prefer `PMID:<pmid>` as `paper_id`, then `DOI:<doi>`, then a deterministic
  normalized-title hash.
- Skip only a record that has no PMID, no DOI, and no non-empty title.
- Never create a paper when PubMed returns none or fails.

The search backend is constructor-injected so offline tests can supply known
responses and exceptions without network access.

### ExactPaperDeduplicator

Implements `PaperDeduplicatorProtocol`.

- Match PMID first, DOI second, and normalized title third.
- Reuse a matching object from `existing_papers` as the canonical record.
- Preserve first-seen ordering.

### MetadataPaperRanker

Implements `PaperRankerProtocol`.

- Papers with a non-empty abstract rank before papers without one.
- Within that grouping, newer publication years rank first.
- Original order is the stable final tie-breaker.
- Respect the requested limit and record only simple deterministic rank
  scores.

### AbstractScoutReader

Implements `ScoutReaderProtocol`.

- Extract a bounded set of Latin terms from title and abstract.
- Set the title as the main topic and emit a deterministic relevance score.
- Do not infer authors, mechanisms, or controversies that are not present in
  metadata.

### SingleSourceCoverageEvaluator

Implements `CoverageEvaluatorProtocol`.

- At least one valid candidate means the minimal search coverage is
  sufficient.
- No candidate means insufficient coverage with an explicit missing topic.
- This is a workflow smoke-test criterion, not a scientific completeness
  claim.

### AbstractReadingWorkflow

Implements `ReadingExtractionWorkflowProtocol`.

- Use abstracts only and set `degraded_to_abstract = True`.
- For a non-empty abstract, take the first non-empty sentence (bounded in
  length) as an attributable `EvidenceChunk`.
- Create one medium-confidence, source-reported `EvidenceLinkedKnowledge`
  entry linked to that evidence ID.
- If the abstract is empty, return the paper result with an explanatory error
  and no evidence or knowledge entry.
- Never invent missing mechanisms, claims, entities, or citations.

## M2-only Entry Point

Add:

```text
python scripts/run_m2_pubmed.py --question "..." --limit 5
```

The script:

1. creates a `PipelineState` containing the question but no M1
   `ProblemCard`;
2. builds the minimal PubMed adapter explicitly;
3. creates `M2LiteratureSearch(implementation="agentic",
   agentic_adapter=adapter)`;
4. invokes M2 directly;
5. prints UTF-8 JSON containing `literature_results`;
6. prints structured JSON error information and exits non-zero on an
   unrecoverable search error.

No config default changes are required. `search.implementation="agentic"`
without an injected adapter must continue to fail explicitly.

## Outcomes and Error Semantics

### Successful PubMed search

- Return real PubMed metadata.
- Finish after one successful coverage evaluation.
- Produce abstract-linked knowledge entries when abstracts are present.

### Zero PubMed results

- Return zero papers and zero knowledge entries.
- Search terminates with the existing no-results semantics.
- Do not return stub or synthetic content.

### Network, timeout, HTTP, or parse failure

- The strict source raises the original exception.
- `IterativeSearchAgent` records the sanitized source failure and ends with
  `StopReason.ERROR` when no source succeeded.
- `AgenticM2Adapter` raises an unrecoverable M2 error.
- The CLI emits an error object and exits non-zero.

### Duplicate records

- Count and retain only canonical papers after PMID/DOI/title matching.

### Missing abstract

- Keep the real paper in the returned paper count.
- Produce no fake evidence or knowledge for that paper.
- Record the degraded reading condition.

## Testing

All default automated tests remain offline. The PubMed source accepts an
injected backend, allowing tests to cover:

- successful real-shaped PubMed records;
- zero results;
- network exception propagation;
- malformed record filtering;
- PMID, DOI, and title deduplication;
- deterministic ranking;
- Scout and coverage outputs;
- evidence linkage from an abstract;
- missing-abstract degradation;
- final conversion through `AgenticM2Adapter` and `M2LiteratureSearch`;
- CLI success and error exit behavior where practical.

After offline tests pass, manually run the Hippo-YAP/TAZ sub-question against
the real PubMed endpoint. This network run is evidence for current integration
only and is not part of the deterministic default test suite.

## Planned Files

- `hypoforge/tools/pubmed_search.py`: add a public strict asynchronous helper
  without changing legacy defaults.
- `hypoforge/literature/sources/pubmed.py`: new protocol adapter and metadata
  conversion.
- `hypoforge/literature/minimal.py`: deterministic Tool implementations and
  `build_minimal_pubmed_adapter()`.
- `scripts/run_m2_pubmed.py`: explicit M2-only runner.
- `tests/literature/test_pubmed_source.py`: source conversion and failure
  tests.
- `tests/literature/test_minimal_tools.py`: rule Tool and end-to-end M2 tests.
- `hypoforge/literature/README.md`: usage and non-production limitations.

## Compatibility and Safety

- Legacy M2 remains the default and keeps its existing error behavior.
- The unconfigured agentic path still refuses to fabricate dependencies.
- No API key is required for PubMed; `NCBI_API_KEY` remains optional.
- No `.env`, API response cache, output JSON, PDF, or full text is committed.
- All results are either real PubMed metadata or explicit errors/empty
  results.
