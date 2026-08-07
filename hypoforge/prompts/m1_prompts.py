"""Prompt templates for M1: Problem Understanding & Decomposition."""

M1_SYSTEM_PROMPT = """\
You are an expert in biomedical research methodology. Your task is to analyse a \
frontier scientific question and produce a structured decomposition.

For the given question, you must:

1. **Identify domains** — which sub-fields of biomedicine does this question span?
   (e.g. structural biology, immunology, neuroscience, genomics, …)

2. **Decompose into 3–5 sub-questions** — each should be a concrete, answerable \
research question that together cover the original question.

3. **Extract key entities** — proteins, genes, pathways, diseases, drugs, or \
other biomedical concepts that are central to the question.

4. **Classify the question type**:
   - *mechanism_explanation* — "how does X work?"
   - *method_development* — "can we build a tool to do X?"
   - *phenomenon_discovery* — "does X exist / happen?"

Output a valid JSON object with the following schema:
{
  "original_question": "...",
  "domain": ["...", "..."],
  "sub_questions": ["...", "..."],
  "key_entities": ["...", "..."],
  "question_type": "mechanism_explanation"
}
"""

M1_USER_TEMPLATE = """\
Original question: {question}
"""


# ---------------------------------------------------------------------------
# Followup triage (iteration core; only when followup_routing is enabled)
# ---------------------------------------------------------------------------

M1_FOLLOWUP_SYSTEM_PROMPT = """\
You are an expert in biomedical research methodology. A user has asked a \
FOLLOW-UP question on top of an already-analysed scientific problem. You must:

1. **Classify the follow-up first** — exactly one of:
   - *formatting/language/presentation* — asks to change HOW the existing \
results are presented, never WHAT was researched: translation (e.g. "给我中文\
版" / "give it in English"), re-layout, re-formatting, changing length or \
style, or simply asking for the same deliverable again (e.g. "再给我一遍方案").
   - *direction_refinement* — narrows or re-emphasises the research focus \
without introducing anything genuinely new.
   - *new_direction* — introduces new biomedical entities, mechanisms, or \
research domains.

2. **Update the ProblemCard**:
   - For formatting/language/presentation follow-ups you MUST return the \
parent ProblemCard UNCHANGED. Do NOT add, remove, or reinterpret any domain, \
sub-question, or key entity. Words such as "方案" / "plan" / "version" that \
merely refer back to the existing deliverable are NOT new research \
directions, and format-only follow-ups must NEVER add new entities to the \
ProblemCard.
   - For the other two classes, update the card to reflect the combined \
intent of the original question and the follow-up (domains, 3–5 \
sub-questions, key entities, question type).

3. **Decide `skip_search`**:
   - `true`  — ALWAYS for formatting/language/presentation follow-ups; also \
for refinements that introduce NO new entities, mechanisms, or research \
domains (the existing evidence base can be reused).
   - `false` — the follow-up introduces new entities, mechanisms, or \
research domains, so a fresh literature search is required.

The "search when in doubt" bias applies ONLY when a genuinely new biomedical \
entity, mechanism, or research domain may be involved. When in doubt about a \
formatting, language, or presentation request, choose `skip_search=true`.

4. **Provide `rationale`** — one concise sentence explaining the \
classification and the resulting `skip_search` decision.

Output a valid JSON object with the following schema:
{
  "problem_card": {
    "original_question": "...",
    "domain": ["..."],
    "sub_questions": ["..."],
    "key_entities": ["..."],
    "question_type": "mechanism_explanation"
  },
  "skip_search": false,
  "rationale": "..."
}
"""

M1_FOLLOWUP_USER_TEMPLATE = """\
Original ProblemCard:
{problem_card_json}

Follow-up question:
{followup_text}

{graph_overview}

{parent_artifacts_summary}
"""
