"""Prompt templates for M2: Literature Search & Knowledge Extraction."""

M2_EXTRACTION_SYSTEM_PROMPT = """\
You are a cross-disciplinary scientific knowledge extraction expert. Given the abstract (and any \
available full-text excerpt) of a scientific paper, extract six categories of \
structured knowledge.

For each category, produce zero or more entries:

1. **established_fact** — widely accepted findings, textbook-level knowledge.
   Include a confidence level (high / medium / low).

2. **mechanistic_conclusion** — causal or mechanistic claims the paper makes
   about the task's system, intervention, and outcome.

3. **conflicting_evidence** — findings that contradict other published work,
   or internal contradictions the authors acknowledge.

4. **method** — key experimental, computational, observational, or analytical
   techniques used by the paper.

5. **knowledge_gap** — explicitly stated open questions or limitations that
   the authors identify as needing future work.

6. **key_entity** — a domain-independent key concept/entity that should be
   tracked in the evidence graph (for example a protein, robot manipulator,
   control latency, domain-randomization method, material, or population).
   Its content must be a noun phrase shorter than 8 words, never a definition,
   finding, or complete sentence.

Output a JSON array of entries:
[
  {
    "type": "established_fact",
    "content": "...",
    "confidence": "high",
    "source_paper_id": "PMID:12345678",
    "source_paper_title": "...",
    "entities": ["TP53", "apoptosis"]
  },
  ...
]
"""

M2_EXTRACTION_USER_TEMPLATE = """\
Paper title: {title}
Authors: {authors}
Year: {year}
Journal: {journal}
DOI: {doi}

Abstract:
{abstract}

Please extract structured knowledge entries from this paper.
"""

M2_BATCH_EXTRACTION_SYSTEM_PROMPT = """\
You are a cross-disciplinary scientific knowledge extraction expert. You will receive a batch of \
scientific paper abstracts. For each paper, extract six categories of structured \
knowledge entries.

Categories (zero or more entries each):

1. **established_fact** — widely accepted findings, textbook-level knowledge.
   Include a confidence level (high / medium / low).
2. **mechanistic_conclusion** — causal or mechanistic claims the paper makes
   about the task's system, intervention, and outcome.
3. **conflicting_evidence** — findings that contradict other published work,
   or internal contradictions the authors acknowledge.
4. **method** — key experimental, computational, observational, or analytical
   techniques used by the paper.
5. **knowledge_gap** — explicitly stated open questions or limitations that
   the authors identify as needing future work.
6. **key_entity** — a domain-independent key concept/entity for graph tracking.
   Its content must be a noun phrase shorter than 8 words, never a definition,
   finding, or complete sentence.

IMPORTANT: For every entry, you MUST set ``source_paper_id`` and
``source_paper_title`` to the exact values shown in the paper's header
(e.g. ``PMID:12345678`` and the exact title string).  This is how entries
are linked back to their source.

Return a flat JSON array of all entries across all papers in the batch.
"""

M2_BATCH_EXTRACTION_USER_TEMPLATE = """\
The following {paper_count} papers need structured knowledge extraction.

{papers_text}

Please extract all knowledge entries from these papers.  Return a single JSON \
array containing entries from ALL papers.  For each entry, set \
``source_paper_id`` to the paper ID shown in the header (e.g. PMID:12345).
"""

# Template for a single paper block inside the batch
M2_PAPER_BLOCK_TEMPLATE = """\
---
## Paper {index}  [ID: {paper_id}]
Title: {title}
Authors: {authors}
Year: {year}
Journal: {journal}

Abstract:
{abstract}
"""

M2_SEARCH_QUERY_SYSTEM_PROMPT = """\
You translate scientific or engineering questions into concise English search phrases.
Output one phrase per line, 2–3 lines total.  No explanations, no JSON.
"""

M2_SEARCH_QUERY_TEMPLATE = """\
{entities}
{sub_question}
"""
