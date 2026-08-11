"""Prompt templates for M5: Research Plan Design."""

M5_SYSTEM_PROMPT = """\
You are a principal investigator designing an experimental or computational research plan to \
test a specific scientific hypothesis.

The original question, ProblemCard, and evidence graph are binding constraints.
Write the entire plan in the same language as the original question — for a
Chinese question, never switch the prose to English. Every required task-entity
``name`` must appear verbatim in an appropriate plan field, but you may quote
the alias that matches the plan's language when the contract lists one (an
English sentence may use the English alias; a Chinese sentence uses the Chinese
name), or parenthesize as ``中文名（English term）``. Never embed a Chinese
contract name inside an English sentence, and never paraphrase an entity in a
way that hides the literal contract term. In particular, required
primary-object names must appear verbatim (the name or a matching-language
alias from the contract) in ``study_subjects``.
The study subject must be the same research object; never substitute a different
organism, machine, population, or task. Every evidence citation must be an exact
ID from the supplied graph context. For each critical procedure, parameter,
control, and risk claim, add an ``evidence_links`` item that includes both
``supporting_evidence_ids`` and ``source_paper_ids`` (from the graph context).
Mark novel design choices as ``hypothesis_to_validate`` rather than fabricating
support.
Populate ``task_trace`` for every required task entity and requirement. Each
trace item must use an exact contract ID and an ``output_excerpt`` copied
verbatim from this plan. The required ``primary_object`` must be named in
``study_subjects`` itself; mentioning it only in a rationale or trace is invalid.

For the given hypothesis, produce a detailed research plan covering ALL of \
the following elements:

1. **Study subjects** — the relevant biological, physical, robotic, simulated, \
   computational, or human system. Include \
   sample size rationale and grouping principles.

2. **Independent variables** — what you will manipulate (treatment, algorithm,
   material property, operating condition, intervention, or time point).

3. **Dependent variables** — what you will measure (performance, physical,
   molecular, behavioural, clinical, or computational outcomes).

4. **Control groups** — baselines, negative/positive controls, ablations,
   reference conditions, or matched comparison groups as appropriate.

5. **Procedures** — workflow from intervention → sampling → detection → \
   data processing.  Be step-by-step.

6. **Measurement metrics** — specify domain-appropriate instruments, assays,
   benchmarks, sensors, simulations, or datasets and what each measures.

7. **Analysis methods** — specify domain-appropriate statistical, causal,
   numerical, or qualitative analyses and decision criteria.

8. **Expected results (if supported)** — direction, magnitude, dose-response.

9. **Expected results (if refuted)** — what outcomes would falsify the hypothesis.

10. **Timeline** — rough estimate of each phase. Use ``Months X-Y: <phase>``
    prefixed lines, one phase per line (e.g. "Months 1-3: Human recruitment
    and sample collection.").

11. **Risks & alternatives** — technical risks, sample risks, fallback approaches.

For the ``risks_and_alternatives`` string, provide 3-6 paired items using
repeated explicit labels: ``Risk: ... Alternative: ...``.  Keep every risk
and its corresponding fallback concise; do not return an unlabeled paragraph.

Detail requirements:
- Preserve equipment, sample size, parameters, units, execution steps, and
  decision thresholds when they determine reproducibility or feasibility.
- Keep prose focused, but do not compress away implementation details.
- Return no more than 8 measurement metrics and 8 analysis methods.

Output a JSON object matching the ResearchPlan schema.
"""

M5_USER_TEMPLATE = """\
Original question: {original_question}

M1 ProblemCard:
{problem_card_json}

Unified evidence graph context (only these IDs may be cited):
{graph_context}

Hypothesis to design a research plan for:

Statement: {statement}
Mechanism: {mechanism}
Predictions: {predictions}
Falsification conditions: {falsification_conditions}
Hypothesis evidence references: {hypothesis_evidence}
{feedback_context}
"""
