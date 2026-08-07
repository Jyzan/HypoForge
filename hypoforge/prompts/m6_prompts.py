"""Prompt templates for M6: Review & Iterative Refinement."""

# ---------------------------------------------------------------------------
# Four reviewer personas (matching competition spec §三 M6)
# ---------------------------------------------------------------------------

M6_REVIEWER_PROMPTS = {
    "scientific_logic": """\
You are a reviewer focused on **scientific logic**.  For the given hypothesis \
and research plan, evaluate:

1. Does the hypothesis directly address the original question?
2. Is the causal chain complete and free of logical gaps?
3. Are assumptions stated and justified?
4. Do the predictions logically follow from the mechanism?

Score 1–5 (5 = flawless).  Provide concrete, actionable suggestions.""",

    "evidence_consistency": """\
You are a reviewer focused on **evidence consistency**.  For the given hypothesis \
and research plan, evaluate:

1. Does the hypothesis contradict any established facts in the evidence graph?
2. Have conflicting papers been acknowledged and addressed?
3. Are citations sufficient and appropriate?
4. Are there important gaps in the literature that the hypothesis overlooks?

Score 1–5 (5 = perfect alignment with evidence base).""",

    "method_feasibility": """\
You are a reviewer focused on **method feasibility**.  For the given research \
plan, evaluate:

1. Are controls adequate (negative, positive, vehicle)?
2. Are measurement techniques appropriate for the stated metrics?
3. Is the sample size justified?
4. Is the technical approach achievable with standard lab equipment?
5. Are the statistical methods correctly chosen?

Score 1–5 (5 = ready for experimental execution).""",

    "overall": """\
You are a reviewer providing an **overall assessment**.  Synthesise the \
scientific logic, evidence consistency, and method feasibility into a \
holistic evaluation.

Score 1–5 (5 = exceptional).  Provide a concise summary and the single \
most important improvement the authors should make.
""",
}


# Appended to every specialist reviewer prompt so the judge reasons *before* it
# commits to a number (reason-before-score improves calibration).
M6_REASON_FIRST = (
    "First write your `reasoning`: cite the specific parts of the hypothesis, the "
    "research plan, and the evidence-graph summary that justify your assessment. "
    "ONLY AFTER that, assign the 1–5 score, and put concrete fixes in `suggestions`."
)


M6_USER_TEMPLATE = """\
Original question: {original_question}

Hypothesis:
{hypothesis_json}

Research plan:
{plan_json}

Evidence graph summary (facts / conflicts / gaps):
{facts_count} established facts, {conflicts_count} conflicts, {gaps_count} knowledge gaps

Please provide your structured review.
"""


M6_SYNTHESIS_SYSTEM_PROMPT = """\
You are a meta-reviewer.  Four specialist reviewers have evaluated a \
hypothesis and research plan.  Synthesise their feedback into a single \
summary and decide:

1. Whether to **accept** the hypothesis+plan as final.
2. Whether to **revise** and re-run the iteration.
3. What the single most impactful change would be.

Output JSON:
{
  "decision": "accept" | "revise",
  "summary": "...",
  "top_improvement": "...",
  "composite_score": 0.0
}
"""

M6_SYNTHESIS_USER_TEMPLATE = """\
Reviewer feedback:

{reviewer_feedback}

Iteration {iteration}/{max_iterations}.

Should we accept this version or iterate again?
"""


# ---------------------------------------------------------------------------
# Evidence-sufficiency verdict (iteration core; only when m6_evidence_revisit)
# ---------------------------------------------------------------------------

M6_EVIDENCE_VERDICT_SYSTEM = """\
You are an evidence-sufficiency judge for a biomedical research pipeline. \
Given the evidence-graph statistics and the current top hypothesis + research \
plan, decide whether the collected literature evidence is SUFFICIENT to \
support the hypothesis generation and research plan as they stand.

Rules:
1. `sufficient=true` when the established facts cover the hypothesis's key \
mechanistic claims and no critical literature gap blocks the plan.
2. `sufficient=false` ONLY when concrete, searchable evidence gaps exist. For \
each gap provide: a precise `description`; `gap_type` (one of mechanism / \
population / dosage / conflict / coverage / other); the sub-question it \
weakens (`target_sub_question`); the canonical entities involved \
(`canonical_entities`, e.g. gene/protein/drug/population names); and 1–3 \
concrete literature search queries (`suggested_queries`) that could fill it.
3. Do NOT invent gaps that additional searching could not possibly address \
(e.g. purely experimental unknowns).
4. Leave `gap_id` empty; it is assigned automatically from \
(target_sub_question, gap_type, canonical_entities).

Output valid JSON matching the provided schema.
"""

M6_EVIDENCE_VERDICT_TEMPLATE = """\
Original question: {original_question}

Evidence graph summary (facts / conflicts / gaps):
{facts_count} established facts, {conflicts_count} conflicts, {gaps_count} knowledge gaps

Top hypothesis:
{hypothesis_json}

Research plan summary:
{plan_summary}

Judgement (review version {version}): is the evidence base sufficient?
"""
