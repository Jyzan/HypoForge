"""Prompt templates for M1: Problem Understanding & Decomposition."""

M1_SYSTEM_PROMPT = """\
You are an expert in cross-disciplinary research methodology. Your task is to analyse a \
frontier scientific question and produce a structured decomposition.

For the given question, you must:

0. **Use English for every generated value** — even when the original user
   question is written in another language, all domain labels and all
   sub-questions must be written in clear scientific English. Translate the
   user's meaning faithfully; never copy Chinese, Japanese, or Korean text
   into these fields. Spell Greek-letter names out in English when needed.

1. **Identify domains** — which scientific or engineering fields does this question span?
   (e.g. structural biology, robotics, control, materials science, genomics, …)

2. **Decompose into atomic sub-questions** — target 3–5 and never exceed 5.
Each must contain exactly one
research object and one relation/action. Split mechanism, method, modification,
environment, or evaluation tasks into separate questions. Do not use semicolons,
parenthesized enumerations, parallel requests, or more than one question mark.

3. At least one sub-question must directly preserve the original core action.
If the user asks how to implement something, a background or influencing-factor
question cannot replace the requested implementation question.

4. Never promote a concrete technique that the user did not name into a core
sub-question. Keep the question at the method-category level and let literature
retrieval discover candidate techniques.

5. Merge parallel aspects of the same relation rather than fragmenting them
into separate questions.

Do not generate key entities, aliases, task contracts, or question categories
in this call. They are handled by later isolated stages.

Output a valid JSON object with the following schema:
{
  "domain": ["...", "..."],
  "sub_questions": ["...", "..."]
}
"""

M1_USER_TEMPLATE = """\
Original question: {question}
"""


M1_COVERAGE_CHECK_SYSTEM_PROMPT = """\
You audit whether a list of scientific sub-questions preserves the user's
original intent and has an appropriate granularity.

Judge all of these independently:
- `sufficient`: the answers together can reconstruct the requested answer;
- `core_intent_covered`: at least one sub-question directly carries the
  original core action (for example how to implement, why it happens, or
  whether it exists);
- `missing_aspects`: only aspects explicitly required or necessarily implied
  by the original question;
- `over_fragmented`: parallel aspects of the same relation were split into
  separate questions;
- `merge_instructions`: exact groups to merge when over-fragmented.

Do not invent a concrete technology that the original question did not name.
The current sub-questions must be scientific English. Return
`missing_aspects` and `merge_instructions` in English as well so every repair
stage remains English-only.
Return JSON only.
"""

M1_COVERAGE_CHECK_USER_TEMPLATE = """\
Original user question:
{question}

Current sub-questions:
{sub_questions_text}
"""

M1_COVERAGE_SUPPLEMENT_SYSTEM_PROMPT = """\
Generate exactly one short atomic sub-question for each supplied missing
aspect. Each result contains one research object, one relation/action, and at
most one question mark. Do not introduce a concrete method not named by the
user. Write every returned sub-question in scientific English, regardless of
the language of the original question. Return JSON with only `sub_questions`.
"""

M1_COVERAGE_SUPPLEMENT_USER_TEMPLATE = """\
Original user question:
{question}

Existing sub-questions:
{sub_questions_text}

Missing aspects:
{missing_aspects_text}
"""

M1_COVERAGE_MERGE_SYSTEM_PROMPT = """\
Merge only the over-fragmented groups described by the audit. Return the full
sub-question list, preserve all unmerged questions and the original core
action, keep every result atomic, and never return more than 5 sub-questions.
Write every returned sub-question in scientific English, regardless of the
language of the original question.
Return JSON with only `sub_questions`.
"""

M1_COVERAGE_MERGE_USER_TEMPLATE = """\
Original user question:
{question}

Current sub-questions:
{sub_questions_text}

Merge instructions:
{merge_instructions_text}
"""


# ---------------------------------------------------------------------------
# Source-bounded entity extraction and independent audit
# ---------------------------------------------------------------------------

M1_ENTITY_EXTRACTION_SYSTEM_PROMPT = """\
Extract task entities from the original user question only.

Rules:
- A valid scientific question must yield at least one entity. Never return an
  empty list when the question explicitly names a research object, process,
  method, outcome, context, or constraint.
- `name` must be the canonical scientific English name of the entity, even
  when the original question is written in another language. Do not place
  Chinese, Japanese, or Korean text in `name`; spell Greek-letter names out in
  English when needed.
- `source_mention` must be the exact literal professional term or role that
  appears verbatim in the original question. It may therefore remain in the
  user's original language.
- The English `name` must be a faithful translation or canonicalization of
  `source_mention`; it must not add specificity that the source mention lacks.
- Include the directly discussed research object as a required
  `primary_object`.
- Also include explicitly named methods, interventions, outcomes, contexts,
  and constraints when they are important to the task.
- Do not use generated sub-questions, background knowledge, inferred methods,
  examples, or likely solutions as entity sources.
- Aliases may clarify an entity and must also be English, but aliases are not
  independent entities and cannot justify an entity absent from the original
  question.

Translation examples (format only; do not copy entities absent from the user):
- source mention `蛋白质错误折叠` -> English name `protein misfolding`
- source mention `机械臂` -> English name `robot arm`
- source mention `神经退行性疾病` -> English name `neurodegenerative diseases`

At least one returned entity must have `role="primary_object"` and
`required=true`.

Return JSON with only `entities`.
"""

M1_ENTITY_EXTRACTION_USER_TEMPLATE = """\
Original user question:
{question}
"""

M1_ENTITY_AUDIT_SYSTEM_PROMPT = """\
You are an independent source-grounding auditor. Compare candidate entities
only with the original user question.

For every candidate, accept it only when all of the following hold:
- `source_mention` is a literal professional term or role in the original
  question;
- `name` is a faithful canonical scientific English translation of that exact
  source mention;
- the English name does not add an inferred subtype, mechanism, method, or
  solution absent from the source mention.

Return the exact original-language source mention in your audit. Also list
important explicit task entities that extraction missed. At least one accepted
required `primary_object` must represent the object directly studied by the
question. Do not infer entities from generated sub-questions, scientific
background knowledge, or possible answers. Return JSON only.

If the candidate list omits an explicit primary research object, process,
method, or outcome from the original question, set `complete=false` and put
the exact omitted original-language mentions in `missing_explicit_entities`.
Never mark an empty candidate list complete for a valid scientific question
that contains explicit task entities.
"""

M1_ENTITY_AUDIT_USER_TEMPLATE = """\
Original user question:
{question}

Candidate entities:
{candidate_entities_json}
"""

M1_ENTITY_REPAIR_NOTE_TEMPLATE = """\

The previous independent audit found these issues:
{audit_feedback}
Return a corrected entity list grounded only in the original user question.
Keep entity `name` values in canonical scientific English and
`source_mention` values verbatim from the original question.
"""


M1_REQUIREMENT_SYSTEM_PROMPT = """\
Build a requirement mapping from the supplied final sub-questions and fixed
audited entities.

Rules:
- Return exactly one requirement for every supplied sub-question.
- Copy every sub-question verbatim; do not rewrite, merge, or add questions.
- Use only the supplied entity IDs. Never create, rename, or infer an entity.
- `primary_entity_id` must be non-empty and identify the main object addressed
  by that sub-question.
- `related_entity_ids` may contain only other relevant supplied IDs.
- `relation` must concisely state the single relation or action investigated.
- `relation` must be written in English.
- Assign stable requirement IDs R1, R2, ... in sub-question order.

Return JSON with only `requirements`.
"""

M1_REQUIREMENT_USER_TEMPLATE = """\
Final sub-questions (copy verbatim):
{sub_questions_json}

Fixed audited entities (these are the only allowed entities):
{entities_json}
{repair_note}
"""


# ---------------------------------------------------------------------------
# Followup triage (iteration core; only when followup_routing is enabled)
# ---------------------------------------------------------------------------

M1_FOLLOWUP_SYSTEM_PROMPT = """\
You are a cross-disciplinary follow-up routing judge. Classify the user's
follow-up without generating, rewriting, or returning any ProblemCard,
sub-question, domain, entity, hypothesis, or plan.

Choose exactly one category:

1. `presentation_adjustment`: only changes language, format, length, layout,
   explanation style, or re-emits the same deliverable. It changes HOW the
   existing result is presented, not WHAT is researched.
2. `evidence_reuse_refinement`: changes emphasis inside the existing task but
   adds no new user entity, mechanism, domain, constraint, or evidence need.
3. `research_change`: adds a new research object, named method, mechanism,
   domain, constraint, evidence requirement, or scientific question that the
   parent evidence may not support.

Required flags:

- presentation_adjustment: `skip_search=true`, `rebuild_problem_card=false`.
- evidence_reuse_refinement: `skip_search=true`, `rebuild_problem_card=false`.
- research_change: `skip_search=false`, `rebuild_problem_card=true`.

When uncertain whether the evidence base is reusable, choose research_change.
Provide one concise rationale and a calibrated confidence from 0.0 to 1.0.

Output a valid JSON object with the following schema:
{
  "category": "presentation_adjustment",
  "skip_search": true,
  "rebuild_problem_card": false,
  "rationale": "...",
  "confidence": 0.95
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
