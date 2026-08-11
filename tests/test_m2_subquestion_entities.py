"""Tests for M2-local sub-question entity generation (task #38).

Covers:

* generator happy path (``needed=true`` → new entities returned);
* ``needed=false`` → no entities;
* deterministic dedup against the sub-question's existing entities;
* fail-safe degradation on LLM errors (warning event, no exception);
* adapter integration — supplements reach the search agent / planner input
  but never leak back into ``problem_card`` / ``task_contract`` / outputs;
* prompt constraint assertions (existing entities visible, short noun
  phrases only — no full-sentence entities);
* planner prompt carries the supplementary concepts.
"""

from __future__ import annotations

import pytest

from hypoforge.literature.adapter import AgenticM2Adapter
from hypoforge.literature.models import (
    EvidenceChunk,
    EvidenceLinkedKnowledge,
    PaperReadingResult,
    PaperRecord,
    SearchRunResult,
    StopReason,
)
from hypoforge.literature.search.query_planner import QueryPlanner
from hypoforge.literature.subquestion_entities import (
    _EntityProposal,
    dedupe_new_entities,
    generate_subquestion_entities,
)
from hypoforge.observability import RunEventRecorder, bind_recorder
from hypoforge.state import (
    ConfidenceLevel,
    KnowledgeEntryType,
    PipelineState,
    ProblemCard,
    TaskContract,
    TaskEntity,
    TaskRequirement,
)

from tests.literature.fakes import FakeReadingWorkflow


SUB_QUESTION = "How does Hsp70 refold misfolded proteins?"


# ---------------------------------------------------------------------------
# Fakes / builders
# ---------------------------------------------------------------------------


class FakeEntityLLM:
    """Records structured_chat calls; returns a canned payload or raises."""

    def __init__(self, payload=None, error: Exception | None = None) -> None:
        self.payload = payload
        self.error = error
        self.calls: list[dict] = []

    async def structured_chat(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.payload


class RecordingSearchAgent:
    def __init__(self, result: SearchRunResult) -> None:
        self.result = result
        self.calls: list[tuple[str, dict]] = []

    async def run(self, sub_question, **kwargs):
        self.calls.append((sub_question, kwargs))
        return self.result.model_copy(deep=True)


def make_problem_card() -> ProblemCard:
    contract = TaskContract(
        entities=[
            TaskEntity(
                entity_id="ENT_hsp70",
                name="Hsp70",
                aliases=["HSPA1A", "heat shock protein 70"],
                role="primary_object",
                required=True,
            ),
            TaskEntity(
                entity_id="ENT_misfold",
                name="protein misfolding",
                role="context",
            ),
        ],
        requirements=[
            TaskRequirement(
                requirement_id="REQ_1",
                sub_question=SUB_QUESTION,
                primary_entity_id="ENT_hsp70",
                related_entity_ids=["ENT_misfold"],
                relation="mechanism",
            ),
        ],
    )
    return ProblemCard(
        original_question="Q",
        sub_questions=[SUB_QUESTION],
        key_entities=["Hsp70", "protein misfolding"],
        domain=["biology"],
        task_contract=contract,
    )


def make_search_result() -> SearchRunResult:
    paper = PaperRecord(paper_id="p1", title="T", sources=["pubmed"])
    return SearchRunResult(
        sub_question=SUB_QUESTION,
        final_papers=[paper],
        stop_reason=StopReason.COVERAGE_SATISFIED,
    )


def make_reading_results() -> list[PaperReadingResult]:
    return [
        PaperReadingResult(
            paper_id="p1",
            evidence=[
                EvidenceChunk(
                    evidence_id="ev-p1",
                    paper_id="p1",
                    chunk_id="chunk-p1",
                    quote="Observed binding.",
                    normalized_claim="Observed binding.",
                    relevance_score=0.9,
                )
            ],
            knowledge_entries=[
                EvidenceLinkedKnowledge(
                    entry_id="ke-p1",
                    entry_type=KnowledgeEntryType.MECHANISTIC_CONCLUSION,
                    content="Observed binding.",
                    confidence=ConfidenceLevel.HIGH,
                    evidence_ids=["ev-p1"],
                )
            ],
        )
    ]


def make_adapter(
    llm: FakeEntityLLM | None,
    agent: RecordingSearchAgent,
) -> AgenticM2Adapter:
    return AgenticM2Adapter(
        search_agent=agent,
        reading_workflow=FakeReadingWorkflow(make_reading_results()),
        subquestion_entity_client=llm,
    )


# ---------------------------------------------------------------------------
# Generator unit tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generator_returns_new_entities_when_needed() -> None:
    llm = FakeEntityLLM(payload={
        "needed": True,
        "new_entities": [
            {"name": "co-chaperone", "aliases": [], "rationale": "Hsp70 cycle"},
        ],
    })

    result = await generate_subquestion_entities(
        llm, SUB_QUESTION, existing_terms=["Hsp70"], domains=["biology"],
    )

    assert result == ["co-chaperone"]


@pytest.mark.asyncio
async def test_generator_returns_nothing_when_not_needed() -> None:
    llm = FakeEntityLLM(payload={"needed": False, "new_entities": []})

    result = await generate_subquestion_entities(
        llm, SUB_QUESTION, existing_terms=["Hsp70"],
    )

    assert result == []


@pytest.mark.asyncio
async def test_generator_skips_entirely_without_client() -> None:
    assert await generate_subquestion_entities(
        None, SUB_QUESTION, existing_terms=["Hsp70"],
    ) == []


@pytest.mark.asyncio
async def test_generator_llm_failure_degrades_silently(tmp_path) -> None:
    llm = FakeEntityLLM(error=RuntimeError("llm down"))
    recorder = RunEventRecorder(tmp_path / "run", "run-1")

    with bind_recorder(recorder):
        result = await generate_subquestion_entities(
            llm, SUB_QUESTION, existing_terms=["Hsp70"],
        )

    assert result == []  # no exception, search can proceed
    warnings = [
        event
        for event in recorder.read_events()
        if event["tool"] == "m2_subquestion_entities"
        and event["status"] == "warning"
    ]
    assert len(warnings) == 1
    assert warnings[0]["details"]["new_entities"] == []
    assert warnings[0]["details"]["existing_entity_count"] == 1


@pytest.mark.asyncio
async def test_generator_emits_started_and_completed_events(tmp_path) -> None:
    llm = FakeEntityLLM(payload={
        "needed": True,
        "new_entities": [{"name": "co-chaperone"}],
    })
    recorder = RunEventRecorder(tmp_path / "run", "run-1")

    with bind_recorder(recorder):
        await generate_subquestion_entities(
            llm, SUB_QUESTION, existing_terms=["Hsp70", "HSPA1A"],
        )

    events = [
        event
        for event in recorder.read_events()
        if event["tool"] == "m2_subquestion_entities"
    ]
    assert [event["event_type"] for event in events] == [
        "tool_started",
        "tool_completed",
    ]
    completed = events[-1]
    assert completed["details"]["sub_question"] == SUB_QUESTION
    assert completed["details"]["existing_entity_count"] == 2
    assert completed["details"]["new_entities"] == ["co-chaperone"]


@pytest.mark.asyncio
async def test_prompt_shows_existing_entities_and_bans_sentences() -> None:
    llm = FakeEntityLLM(payload={"needed": False, "new_entities": []})

    await generate_subquestion_entities(
        llm,
        SUB_QUESTION,
        existing_terms=["Hsp70", "heat shock protein 70"],
        domains=["biology"],
    )

    call = llm.calls[0]
    user_prompt = call["user_prompt"]
    system_prompt = call["system_prompt"]
    # The LLM must see every existing term (names + aliases).
    assert "Hsp70" in user_prompt
    assert "heat shock protein 70" in user_prompt
    assert SUB_QUESTION in user_prompt
    # Guard against the historical "full-sentence entity" bug.
    assert "short noun phrase" in system_prompt
    assert "NEVER" in system_prompt and "sentence" in system_prompt
    # Structured output contract follows the project pydantic convention.
    schema = call["output_schema"]
    assert schema["properties"].keys() >= {"needed", "new_entities"}


# ---------------------------------------------------------------------------
# Deterministic dedup fallback
# ---------------------------------------------------------------------------


def test_dedupe_drops_case_and_whitespace_duplicates() -> None:
    proposal = _EntityProposal.model_validate({
        "needed": True,
        "new_entities": [
            {"name": "  hsp70 "},            # duplicate of "Hsp70"
            {"name": "Protein Misfolding"},  # duplicate of "protein misfolding"
            {"name": "co-chaperone"},        # genuinely new
        ],
    })

    kept = dedupe_new_entities(
        proposal, ["Hsp70", "protein misfolding"],
    )

    assert kept == ["co-chaperone"]


def test_dedupe_caps_entity_count_and_internal_duplicates() -> None:
    proposal = _EntityProposal.model_validate({
        "needed": True,
        "new_entities": [
            {"name": "co-chaperone"},
            {"name": "Co-Chaperone"},
            {"name": "Hsp40"},
            {"name": "BAG-1"},
            {"name": "nucleotide exchange factor"},
        ],
    })

    kept = dedupe_new_entities(proposal, [])

    assert kept == ["co-chaperone", "Hsp40", "BAG-1"]


@pytest.mark.asyncio
async def test_generator_applies_dedupe_end_to_end() -> None:
    llm = FakeEntityLLM(payload={
        "needed": True,
        "new_entities": [
            {"name": "HSPA1A"},           # alias of an existing entity
            {"name": "co-chaperone"},
        ],
    })

    result = await generate_subquestion_entities(
        llm, SUB_QUESTION, existing_terms=["Hsp70", "HSPA1A"],
    )

    assert result == ["co-chaperone"]


# ---------------------------------------------------------------------------
# Adapter integration — planner input & scope isolation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_adapter_feeds_new_entities_to_search_agent() -> None:
    llm = FakeEntityLLM(payload={
        "needed": True,
        "new_entities": [{"name": "co-chaperone"}],
    })
    agent = RecordingSearchAgent(make_search_result())
    adapter = make_adapter(llm, agent)
    state = PipelineState(input_question="Q", problem_card=make_problem_card())
    card_before = state.problem_card.model_dump_json()
    contract_before = state.problem_card.task_contract.model_dump_json()

    output = await adapter(state)

    assert len(agent.calls) == 1
    sub_question, kwargs = agent.calls[0]
    assert sub_question == SUB_QUESTION
    # M1 entities untouched, supplements handed over separately.
    assert kwargs["key_entities"] == ["Hsp70", "protein misfolding"]
    assert kwargs["supplement_entities"] == ["co-chaperone"]
    # The LLM saw the sub-question's full existing term surface.
    user_prompt = llm.calls[0]["user_prompt"]
    assert "Hsp70" in user_prompt and "HSPA1A" in user_prompt

    # --- Scope isolation: nothing written back, nothing exported ---------
    assert state.problem_card.model_dump_json() == card_before
    assert state.problem_card.task_contract.model_dump_json() == contract_before
    assert set(output) == {"literature_results", "m2_knowledge_export"}
    exported = output["m2_knowledge_export"].model_dump_json()
    assert "co-chaperone" not in exported


@pytest.mark.asyncio
async def test_adapter_without_new_entities_keeps_original_input() -> None:
    llm = FakeEntityLLM(payload={"needed": False, "new_entities": []})
    agent = RecordingSearchAgent(make_search_result())
    adapter = make_adapter(llm, agent)
    state = PipelineState(input_question="Q", problem_card=make_problem_card())

    await adapter(state)

    _, kwargs = agent.calls[0]
    assert kwargs["key_entities"] == ["Hsp70", "protein misfolding"]
    assert kwargs["supplement_entities"] == []


@pytest.mark.asyncio
async def test_adapter_survives_llm_failure_and_keeps_scope(tmp_path) -> None:
    llm = FakeEntityLLM(error=RuntimeError("boom"))
    agent = RecordingSearchAgent(make_search_result())
    adapter = make_adapter(llm, agent)
    state = PipelineState(input_question="Q", problem_card=make_problem_card())
    contract_before = state.problem_card.task_contract.model_dump_json()
    recorder = RunEventRecorder(tmp_path / "run", "run-1")

    with bind_recorder(recorder):
        output = await adapter(state)

    _, kwargs = agent.calls[0]
    assert kwargs["supplement_entities"] == []
    assert kwargs["key_entities"] == ["Hsp70", "protein misfolding"]
    assert set(output) == {"literature_results", "m2_knowledge_export"}
    assert state.problem_card.task_contract.model_dump_json() == contract_before


@pytest.mark.asyncio
async def test_adapter_without_client_skips_generation() -> None:
    agent = RecordingSearchAgent(make_search_result())
    adapter = AgenticM2Adapter(
        search_agent=agent,
        reading_workflow=FakeReadingWorkflow(make_reading_results()),
    )
    state = PipelineState(input_question="Q", problem_card=make_problem_card())

    await adapter(state)

    _, kwargs = agent.calls[0]
    assert kwargs["supplement_entities"] == []


# ---------------------------------------------------------------------------
# Planner prompt integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_planner_prompt_lists_supplementary_concepts() -> None:
    llm = FakeEntityLLM(payload={
        "queries": [
            {
                "text": "Hsp70[tiab] AND co-chaperone[tiab]",
                "tool": "pubmed",
                "purpose": "core_mechanism",
            }
        ],
        "reasoning": "uses the supplementary concept",
    })
    planner = QueryPlanner(
        client=llm,
        tool_definitions=[
            {"name": "pubmed", "display_name": "PubMed", "description": "d"},
        ],
        max_rounds=3,
        strict=True,
    )

    queries = await planner.plan(
        SUB_QUESTION,
        key_entities=["Hsp70"],
        domains=["biology"],
        supplement_entities=["co-chaperone"],
    )

    assert queries
    user_prompt = llm.calls[0]["user_prompt"]
    assert "Supplementary Search Concepts" in user_prompt
    assert "co-chaperone" in user_prompt
    assert "Key Entities: Hsp70" in user_prompt


@pytest.mark.asyncio
async def test_planner_prompt_omits_supplement_section_when_empty() -> None:
    llm = FakeEntityLLM(payload={
        "queries": [
            {"text": "Hsp70[tiab]", "tool": "pubmed", "purpose": "core_mechanism"},
        ],
        "reasoning": "r",
    })
    planner = QueryPlanner(
        client=llm,
        tool_definitions=[
            {"name": "pubmed", "display_name": "PubMed", "description": "d"},
        ],
        max_rounds=3,
        strict=True,
    )

    await planner.plan(SUB_QUESTION, key_entities=["Hsp70"], domains=["biology"])

    assert "Supplementary Search Concepts" not in llm.calls[0]["user_prompt"]
