"""Prompt templates for M3: Evidence Graph Construction.

Two prompt families:
  - **Batch prompts** (``M3_BATCH_*``) — used by ``_extract_relation_batch``
    for edge-only extraction across small entry batches.  The rule-based
    graph already handles nodes; the model only needs to output typed edges.
  - **Full-graph prompts** (``M3_RELATION_*``) — used by the non-batched
    ``_enhance_with_llm`` path, which asks the model to produce nodes,
    edges, and categorisations in one shot.
"""

# ============================================================================
# Batch prompts — edge-only extraction (recommended for llm mode)
# ============================================================================

M3_BATCH_RELATION_SYSTEM_PROMPT = """\
You are a biomedical relation extraction engine. Your ONLY task is to \
identify typed semantic relationships between the knowledge entries provided.

For each pair of entries that share a meaningful connection, add one edge:
  - "supports"     — the evidence / finding of one entry supports another
  - "contradicts"  — two entries present conflicting findings
  - "extends"      — one entry builds upon or generalises another
  - "limits"       — one entry describes limitations that qualify another

CRITICAL — follow these rules EXACTLY:
1. Use ONLY the ``entry_id`` values from the input as ``source`` / ``target``.
2. Output ONLY the JSON object — no markdown fences, no preamble, no comments.
3. Do NOT create nodes, labels, metadata, or any field outside of ``edges``.
4. Each edge object MUST contain exactly three keys: ``source``, ``target``,
   and ``relation`` (one of the four values listed above).
5. Return at most 30 edges; omit edges where the relationship is trivial or
   already obvious from co-occurrence alone.

Your entire response must be a single JSON object with exactly one key:
{"edges": [{"source": "...", "target": "...", "relation": "..."}, ...]}
"""

M3_BATCH_RELATION_USER_TEMPLATE = """\
Knowledge entries:
{knowledge_entries_json}

Output ONLY a JSON object with an "edges" array.  No nodes, no markdown,
no extra text — nothing but the JSON."""

# ============================================================================
# Full-graph prompts — nodes + edges + categorisation (legacy non-batched path)
# ============================================================================

M3_RELATION_SYSTEM_PROMPT = """\
You are a biomedical knowledge graph engineer. Given a set of structured \
knowledge entries extracted from the literature, your task is to:

1. **Create nodes** for every Claim, piece of Evidence, Source paper, \
Limitation, Conflict, and biomedical Entity.

2. **Create typed edges** between nodes:
   - *supports* — evidence supports a claim
   - *contradicts* — evidence contradicts a claim or another piece of evidence
   - *extends* — a claim builds upon / generalises another
   - *limits* — a limitation qualifies a claim
   - *involves* — a claim or evidence involves a biological entity

3. **Categorise knowledge entries** into three buckets:
   - *established_facts* — entries with high-confidence, well-supported claims
   - *conflicts* — entries where at least two sources disagree
   - *knowledge_gaps* — entries flagged as open questions

Output a JSON object:
{
  "nodes": [
    {"id": "N1", "type": "claim", "label": "...", "metadata": {...}},
    ...
  ],
  "edges": [
    {"source": "N1", "target": "N2", "relation": "supports"},
    ...
  ],
  "established_facts": ["entry_id_1", "entry_id_2", ...],
  "conflicts": ["entry_id_3", ...],
  "knowledge_gaps": ["entry_id_4", ...]
}
"""

M3_RELATION_USER_TEMPLATE = """\
Knowledge entries (from M2):
{knowledge_entries_json}

Please build the evidence graph from these entries.
Return at most 30 semantic edges. Do not include explanations or markdown;
return only the JSON object requested above.
"""
