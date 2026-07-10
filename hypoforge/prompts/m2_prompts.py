"""Prompt templates for M2: Literature Search & Knowledge Extraction."""

M2_EXTRACTION_SYSTEM_PROMPT = """\
You are a biomedical knowledge extraction expert. Given the abstract (and any \
available full-text excerpt) of a scientific paper, extract six categories of \
structured knowledge.

For each category, produce zero or more entries:

1. **established_fact** — widely accepted findings, textbook-level knowledge.
   Include a confidence level (high / medium / low).

2. **mechanistic_conclusion** — causal or mechanistic claims the paper makes
   (e.g. "Protein X activates pathway Y via phosphorylation of Z").

3. **conflicting_evidence** — findings that contradict other published work,
   or internal contradictions the authors acknowledge.

4. **method** — key experimental techniques used (e.g. CRISPR-Cas9 knockout,
   RNA-seq, SPR, X-ray crystallography, …).

5. **knowledge_gap** — explicitly stated open questions or limitations that
   the authors identify as needing future work.

6. **key_entity** — important proteins, genes, pathways, drugs, diseases that
   should be tracked in the evidence graph.

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

M2_SEARCH_QUERY_TEMPLATE = """\
Generate 2–3 focused PubMed / Semantic Scholar search queries for the following \
sub-question in biomedicine.  Use MeSH terms where appropriate.  Return only \
the queries, one per line.

Sub-question: {sub_question}
Key entities: {entities}
"""
