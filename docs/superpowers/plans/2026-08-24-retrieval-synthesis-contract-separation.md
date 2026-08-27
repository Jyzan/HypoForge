# Retrieval / Synthesis Contract Separation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep M1 sub-questions private to M2/M3 retrieval provenance while requiring every M4 hypothesis, M5 plan, and M6 review to answer the original question as one whole task.

**Architecture:** Preserve the persisted M1 retrieval contract unchanged and derive a deterministic downstream `Q0` synthesis-contract view from the original question and original entities. Pass that view explicitly through graph rendering and alignment checks so M4–M6 never serialize R1…Rn or their sub-question text, while M2/M3 continue using the original retrieval contract.

**Tech Stack:** Python 3.11+, Pydantic v2 models, asyncio, pytest, existing Qwen structured-chat client.

**Spec:** `docs/superpowers/specs/2026-08-24-retrieval-synthesis-contract-separation-design.md`

## Global Constraints

- Do not modify or migrate persisted `ProblemCard.sub_questions` or `ProblemCard.task_contract.requirements`.
- M2 and M3 must continue using R1…Rn and the M1 sub-questions for retrieval/provenance.
- M4–M6 prompts must contain the original question and must not contain any exact M1 sub-question or its derived relation.
- Every M4 candidate must independently answer `Q0`; no portfolio-level division of the original task.
- Do not fabricate requirement coverage: a Q0 trace must quote literal output text and pass the existing object/semantic gates.
- Preserve all evidence facts, conflicts, graph relations, canonical IDs, and quotes.
- Do not upload or push any branch, commit, environment file, or credential.

---

### Task 1: Deterministic synthesis-contract view

**Files:**
- Create: `hypoforge/synthesis_contract.py`
- Modify: `scripts/test_pipeline.py`

**Interfaces:**
- Consumes: `ProblemCard`, `PipelineState`, `TaskContract`, `TaskRequirement`.
- Produces: `SYNTHESIS_REQUIREMENT_ID: str = "Q0"`, `build_synthesis_contract(card: ProblemCard | None) -> TaskContract`, and `synthesis_contract_for_state(state: PipelineState) -> TaskContract`.

- [ ] **Step 1: Write the failing contract-view test**

```python
def test_synthesis_contract_contains_only_original_question() -> None:
    state = robot_state()
    state.problem_card.sub_questions = ["retrieval-only question"]
    state.problem_card.task_contract.requirements[0].sub_question = "retrieval-only question"
    contract = synthesis_contract_for_state(state)
    assert [item.requirement_id for item in contract.requirements] == ["Q0"]
    assert contract.requirements[0].sub_question == state.problem_card.original_question
    assert "retrieval-only question" not in contract.model_dump_json()
    assert contract.entities == state.problem_card.task_contract.entities
```

- [ ] **Step 2: Run the test and verify RED**

Run: `python -m pytest scripts/test_pipeline.py -q -k synthesis_contract_contains_only_original_question`

Expected: collection/import failure because `hypoforge.synthesis_contract` does not exist.

- [ ] **Step 3: Implement the pure view functions**

```python
SYNTHESIS_REQUIREMENT_ID = "Q0"

def build_synthesis_contract(card: ProblemCard | None) -> TaskContract:
    if card is None:
        return TaskContract(source="derived")
    entities = [entity.model_copy(deep=True) for entity in card.task_contract.entities]
    primary = next((e.entity_id for e in entities if e.role == "primary_object"), "")
    related = [e.entity_id for e in entities if e.entity_id != primary and e.required]
    return TaskContract(
        source=card.task_contract.source,
        entities=entities,
        requirements=[TaskRequirement(
            requirement_id=SYNTHESIS_REQUIREMENT_ID,
            sub_question=card.original_question,
            primary_entity_id=primary,
            related_entity_ids=related,
            relation="answer the original question as a whole",
            required=True,
        )],
    )
```

`synthesis_contract_for_state()` delegates to `build_synthesis_contract(state.problem_card)`.

- [ ] **Step 4: Run the focused test and verify GREEN**

Run: `python -m pytest scripts/test_pipeline.py -q -k synthesis_contract_contains_only_original_question`

Expected: PASS.

- [ ] **Step 5: Commit locally**

```bash
git add hypoforge/synthesis_contract.py scripts/test_pipeline.py
git commit -m "feat(contract): separate synthesis scope from retrieval questions"
```

### Task 2: Remove retrieval questions from M4 context and generation contract

**Files:**
- Modify: `hypoforge/graph_context.py`
- Modify: `hypoforge/modules/m4_hypothesis_generation.py`
- Modify: `hypoforge/prompts/m4_prompts.py`
- Modify: `scripts/test_pipeline.py`

**Interfaces:**
- Consumes: `synthesis_contract_for_state(state)` from Task 1.
- Produces: M4 graph/prompt content containing only Q0, the original question, original entities, and evidence-graph content.

- [ ] **Step 1: Write failing prompt-privacy and candidate-scope tests**

Capture the first `structured_chat` call from `_generate_hypothesis_batch()` and assert:

```python
assert state.problem_card.original_question in call["user_prompt"]
for question in state.problem_card.sub_questions:
    assert question not in call["user_prompt"]
for requirement in state.problem_card.task_contract.requirements:
    assert requirement.relation not in call["user_prompt"]
assert "Q0" in call["user_prompt"]
```

Also assert that `_render_task_contract_block(state)` says every candidate must answer the original question as a whole and contains no R1…Rn.

- [ ] **Step 2: Run the tests and verify RED**

Run: `python -m pytest scripts/test_pipeline.py -q -k 'm4_prompt_hides_retrieval_subquestions or m4_contract_requires_original_question'`

Expected: FAIL because GraphContext and M4 currently serialize R1…Rn.

- [ ] **Step 3: Switch GraphContext and every M4 contract path to Q0**

- `build_graph_context()` sets `task_contract=synthesis_contract_for_state(state)`.
- M4 contract rendering, canonical trace creation, deterministic validation, contract repair, fallback generation, and iteration reminder all resolve the synthesis contract through the helper.
- `M4_GENERATOR_SYSTEM_PROMPT` replaces portfolio wording with: every candidate independently answers the original question and traces Q0.
- Remove standard/fast divergence for requirement scope; both modes require Q0.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run: `python -m pytest scripts/test_pipeline.py -q -k 'synthesis_contract or m4_prompt_hides_retrieval_subquestions or m4_contract_requires_original_question or graph_context'`

Expected: all selected tests PASS.

- [ ] **Step 5: Commit locally**

```bash
git add hypoforge/graph_context.py hypoforge/modules/m4_hypothesis_generation.py hypoforge/prompts/m4_prompts.py scripts/test_pipeline.py
git commit -m "fix(m4): generate only against the original question"
```

### Task 3: Make alignment checks accept an explicit synthesis contract

**Files:**
- Modify: `hypoforge/task_alignment.py`
- Modify: `hypoforge/modules/m4_hypothesis_generation.py`
- Modify: `hypoforge/modules/m5_research_plan.py`
- Modify: `hypoforge/modules/m6_review_iteration.py`
- Modify: `hypoforge/evaluation/scorer.py`
- Modify: `scripts/test_pipeline.py`

**Interfaces:**
- Consumes: `contract_override: TaskContract | None` added to `assess_task_alignment()`.
- Produces: M4/M5/M6/posthoc alignment against the same Q0 contract.

- [ ] **Step 1: Write failing cross-module alignment tests**

Create a state with retrieval requirements R1–R3 and a hypothesis/plan whose literal trace is only Q0. Assert:

```python
contract = synthesis_contract_for_state(state)
assessment = assess_task_alignment(
    state, output_text, trace=trace, contract_override=contract,
)
assert assessment.passed
assert assessment.missing_requirement_ids == ()
```

Exercise M5 trace canonicalization and M6 review to confirm neither reports missing R1/R2/R3.

- [ ] **Step 2: Run the tests and verify RED**

Run: `python -m pytest scripts/test_pipeline.py -q -k 'synthesis_alignment or m5_uses_q0 or m6_uses_q0'`

Expected: FAIL because `assess_task_alignment()` still reads the retrieval contract from `state.problem_card`.

- [ ] **Step 3: Implement explicit contract injection**

Add `contract_override: TaskContract | None = None` to `assess_task_alignment()` and use it instead of `card.task_contract` when supplied. Pass `synthesis_contract_for_state(state)` from M4, M5, M6 and `evaluation/scorer.py`. M5 canonicalizes Q0 from literal plan segments and inherits the hypothesis Q0 scope.

- [ ] **Step 4: Remove M6 sub-question fallback leakage**

In evidence-gap normalization, replace the fallback `card.sub_questions[0]` with `card.original_question`. The resulting M6 evidence gap is converted into a fresh supplement search question by M2; it does not reuse an M1 retrieval sub-question.

- [ ] **Step 5: Run the focused tests and verify GREEN**

Run: `python -m pytest scripts/test_pipeline.py -q -k 'synthesis_alignment or m5_uses_q0 or m6_uses_q0 or task_alignment or evidence_gap'`

Expected: all selected tests PASS.

- [ ] **Step 6: Commit locally**

```bash
git add hypoforge/task_alignment.py hypoforge/modules/m4_hypothesis_generation.py hypoforge/modules/m5_research_plan.py hypoforge/modules/m6_review_iteration.py hypoforge/evaluation/scorer.py scripts/test_pipeline.py
git commit -m "fix(pipeline): review final outputs against the whole question"
```

### Task 4: Backward-compatible Q0 normalization

**Files:**
- Modify: `hypoforge/modules/m6_review_iteration.py`
- Modify: `scripts/test_pipeline.py`

**Interfaces:**
- Consumes: old M5 snapshots whose traces contain R1…Rn.
- Produces: a Q0 trace only when a literal excerpt from the hypothesis/plan genuinely contains the original entities; otherwise preserves a failing alignment result.

- [ ] **Step 1: Write failing old-snapshot tests**

Test one old output that contains the original task object and can be normalized to Q0, plus one off-topic output that cannot. Assert the first reaches normal M6 review and the second remains a task-alignment failure.

- [ ] **Step 2: Run the tests and verify RED**

Run: `python -m pytest scripts/test_pipeline.py -q -k 'm6_normalizes_legacy_trace_to_q0'`

Expected: FAIL because M6 has no synthesis-trace normalization at its boundary.

- [ ] **Step 3: Implement literal-only normalization**

Reuse the existing M4/M5 trace canonicalization helpers; never generate new prose and never attach Q0 when no literal output segment contains the required original entities.

- [ ] **Step 4: Run the tests and verify GREEN**

Run: `python -m pytest scripts/test_pipeline.py -q -k 'm6_normalizes_legacy_trace_to_q0 or m6_task_alignment'`

Expected: PASS.

- [ ] **Step 5: Commit locally**

```bash
git add hypoforge/modules/m6_review_iteration.py scripts/test_pipeline.py
git commit -m "fix(m6): normalize legacy traces to whole-question scope"
```

### Task 5: Full and real-chain verification

**Files:**
- Modify: `docs/optimization-audit-2026-08-23.md`
- Test: `scripts/test_pipeline.py`

**Interfaces:**
- Consumes: completed Tasks 1–4.
- Produces: reproducible automated and real-model evidence for the new boundary.

- [ ] **Step 1: Run static and full automated verification**

```powershell
python -m pytest -q
python -m compileall -q hypoforge scripts
git diff --check
```

Expected: zero failures and zero compile/diff errors.

- [ ] **Step 2: Restart only the isolated 7862 service**

Run `run_hypoforge_ui.py` with `configs/web_ui.yaml`, `output/ui_runs_7862`, and the existing ignored `.env`. Do not touch port 7860.

- [ ] **Step 3: Run the real standard pipeline**

Submit `肠道微生物组如何影响人体免疫系统？` and wait for M1–M6 completion. Persist events and final result under the normal run directory.

- [ ] **Step 4: Audit the real prompt and result**

Verify from the captured M4 call/test instrumentation and final state:

- no exact M1 sub-question appears in M4 Generator input;
- M4 and M5 traces contain Q0, not R1…Rn;
- M6 task alignment does not report missing R1/R2/R3;
- the hypothesis and plan address the original question as a whole;
- any remaining low score is attributable to scientific/method quality, not retrieval-subquestion bookkeeping.

- [ ] **Step 5: Record evidence and run the full suite once more**

Append run ID, elapsed time, M4/M6 traces, score dimensions, error count and token usage to `docs/optimization-audit-2026-08-23.md`, then rerun `python -m pytest -q` and `python -m compileall -q hypoforge scripts`.

- [ ] **Step 6: Commit locally without pushing**

```bash
git add docs/optimization-audit-2026-08-23.md
git commit -m "docs: verify whole-question synthesis flow"
```

Report the local commit IDs and verification evidence. Do not run `git push`.
