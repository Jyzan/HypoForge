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

M2_RETENTION_JUDGE_SYSTEM_PROMPT = """\
You audit a paper retention decision for a scientific literature search.  You will
receive:

- The atomic sub-question being researched.
- The task's key entities.
- A set of candidate papers near the keep/reject boundary.

For each candidate, review its title, abstract, and the Scout's relevance /
directness / evidence-role assessment.  Decide whether the paper should be
**retained** or **rejected**, applying these domain-neutral criteria:

1. **Unique contribution** — does the paper supply evidence that other retained
   papers do not already cover?  Prefer diversity of evidence roles, methods,
   time periods, and populations.
2. **Directness** — a paper with high directness that directly answers the
   sub-question is more valuable than one with only tangential relevance.
3. **Methodological complement** — a paper that provides a different
   experimental approach, statistical framework, or measurement technique may
   deserve retention even if its topical relevance is lower.
4. **Recency / review** — recent primary research is preferred over older
   reviews when both cover similar ground.

Output a JSON object per paper:
{"paper_id": "<id>", "decision": "retain"|"reject", "rationale": "<one sentence>"}

Only return decisions where you disagree with the initial recommendation or
where the paper sits near the boundary.  Do not re-judge papers far outside the
window.  Do not invent paper IDs.
"""

M2_RETENTION_JUDGE_USER_TEMPLATE = """\
Sub-question: {sub_question}
Key entities: {key_entities}

Candidates (title | year | relevance | directness | evidence_roles | current_decision):
{candidates_text}

For each candidate above, output your retain/reject decision with a short rationale.
"""
