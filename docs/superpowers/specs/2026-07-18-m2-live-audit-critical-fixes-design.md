# M2 Live-Audit Critical Fixes Design

## Goal

Fix the four highest-impact correctness issues observed during the real
Hippo–YAP/TAZ run without changing the legacy M2 default or replacing teammate
Tools.

## Scope

1. Re-rank candidates after Scout Reading using a deterministic blend of the
   metadata rank and `ScoutNote.relevance_to_question`.
2. Keep adjacent RAG chunks as model context, but mark them non-citable and
   reject model claims that cite them without a primary retrieved chunk.
3. Preserve `evidence_ids` when the agentic adapter maps reading knowledge into
   the existing pipeline `KnowledgeEntry` model.
4. Align overlapping English chunk starts to a word boundary.
5. Exclude BioC reference-list passages so cited-paper titles cannot be treated
   as evidence from the current paper.

The changes are backward-compatible: new fields have defaults, the minimal
PubMed backup remains unchanged, and no vector database, PDF parser, source
retry policy, or legacy module registration is added.

## Data Flow

`PaperRanker -> ScoutReader -> deterministic scout rerank -> CoverageEvaluator`

`BM25 primary hits + non-citable neighbor context -> QwenPaperReader -> only
citable evidence IDs accepted -> AgenticM2Adapter preserves evidence IDs`

## Error and Determinism Rules

- A missing Scout note preserves the original metadata order and score.
- Ties use the incoming order and paper ID.
- Context-only chunks may appear in the model prompt but never in accepted
  evidence-linked output.
- Word-boundary alignment applies only to ASCII word/hyphen runs so Chinese
  text is not skipped looking for spaces.

## Verification

Each behavior receives a regression test that must fail before production code
changes. Related tests and the complete suite must pass afterward. A cached
real-paper reading check verifies that no citable evidence has zero relevance
and that chunk starts are readable.
