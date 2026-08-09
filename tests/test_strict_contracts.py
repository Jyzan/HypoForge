from __future__ import annotations

import json

import pytest

from hypoforge.graph_context import GraphContext
from hypoforge.config import PipelineConfig, SearchConfig
from hypoforge.registry import ModuleRegistry
from hypoforge.literature.models import (
    EvidenceChunk,
    EvidenceLinkedKnowledge,
    PaperReadingResult,
    PaperRecord,
)
from hypoforge.memory import PaperStore
from hypoforge.modules.m3_grounding.models import EvidenceRecord
from hypoforge.state import (
    ConfidenceLevel,
    EvidenceGap,
    EvidenceGraph,
    EvidenceNode,
    EvidenceNodeType,
    HypothesisCard,
    KnowledgeEntry,
    KnowledgeEntryType,
    LiteratureResult,
    M2EvidenceExport,
    M2KnowledgeExport,
    M2KnowledgeRun,
    M2PaperExport,
    PipelineState,
    ResearchPlan,
    ResearchPlanEvidenceLink,
    SearchLedger,
)
from hypoforge.strict_contracts import (
    StrictAgenticM2Adapter,
    StrictAgenticM2Module,
    StrictEntityNormalizationService,
    StrictM3EvidenceGraph,
    StrictM4HypothesisGeneration,
    StrictM5ResearchPlan,
    StrictM6ReviewIteration,
)


class EqualEmbeddings:
    async def aembed_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]


class PairDecisionClient:
    async def structured_chat(self, **kwargs):
        requests = json.loads(kwargs["user_prompt"])
        request = requests[0]
        decisions = []
        for candidate in request["candidates"]:
            decisions.append({
                "surface": request["surface"],
                "canonical_name": candidate,
                "same_concept": "regional" in candidate.casefold(),
                "rationale": "pair-level test decision",
            })
        return {"decisions": decisions}


@pytest.mark.asyncio
async def test_entity_pair_decisions_do_not_overwrite_each_other(tmp_path) -> None:
    service = StrictEntityNormalizationService(
        cache_dir=tmp_path,
        client=PairDecisionClient(),
        embedding_backend=EqualEmbeddings(),
    )
    service.register_alias_group("Regional Climate Model", [])
    service.register_alias_group("Global Climate Model", [])

    result = await service.resolve_batch(["RCM"])

    assert result["RCM"].canonical_name == "Regional Climate Model"
    positive = service.pair_decisions[
        service._pair_key("RCM", "Regional Climate Model")
    ]
    negative = service.pair_decisions[
        service._pair_key("RCM", "Global Climate Model")
    ]
    assert positive.same_concept is True
    assert negative.same_concept is False


class PartialDecisionClient:
    async def structured_chat(self, **kwargs):
        request = json.loads(kwargs["user_prompt"])[0]
        global_candidate = next(
            candidate for candidate in request["candidates"]
            if "global" in candidate.casefold()
        )
        return {"decisions": [{
            "surface": request["surface"],
            "canonical_name": global_candidate,
            "same_concept": False,
            "rationale": "global model is explicitly different",
        }]}


@pytest.mark.asyncio
async def test_unresolved_entity_is_not_persisted_as_confirmed_new_entity(tmp_path) -> None:
    first = StrictEntityNormalizationService(
        cache_dir=tmp_path,
        client=PartialDecisionClient(),
        embedding_backend=EqualEmbeddings(),
    )
    first.register_alias_group("Regional Climate Model", [])
    first.register_alias_group("Global Climate Model", [])

    unresolved = await first.resolve_batch(["RCM"])

    assert unresolved["RCM"].decision_source == "unresolved_identity"
    assert "rcm" not in first.alias_to_id

    second = StrictEntityNormalizationService(
        cache_dir=tmp_path,
        client=PairDecisionClient(),
        embedding_backend=EqualEmbeddings(),
    )
    resolved = await second.resolve_batch(["RCM"])

    assert resolved["RCM"].canonical_name == "Regional Climate Model"


class FailingRelationClient:
    async def structured_chat(self, **kwargs):
        raise ConnectionError("relation extractor unavailable")


@pytest.mark.asyncio
async def test_m3_all_relation_jobs_failed_is_hard_failure() -> None:
    module = StrictM3EvidenceGraph(
        mode="llm",
        relation_batch_size=1,
        enable_cross_batch=False,
    )
    module.client = FailingRelationClient()
    entries = [
        KnowledgeEntry(
            id="k1",
            type=KnowledgeEntryType.ESTABLISHED_FACT,
            content="fact one",
            source_paper_id="p1",
        ),
        KnowledgeEntry(
            id="k2",
            type=KnowledgeEntryType.ESTABLISHED_FACT,
            content="fact two",
            source_paper_id="p2",
        ),
    ]
    graph = module._build_rule_graph(entries)

    with pytest.raises(RuntimeError, match="Every configured M3 relation-extraction"):
        await module._enhance_with_llm_batched(graph, entries)


class RecordingGrounder:
    def __init__(self) -> None:
        self.calls = 0

    async def run(self, state):
        self.calls += 1
        evidence = state.m2_knowledge_export.runs[0].evidence[0]
        return {
            "evidence_records": [EvidenceRecord(
                id="rec-1",
                evidence_id=evidence.evidence_id,
                paper_id=evidence.paper_id,
                chunk_id=evidence.chunk_id,
                quote=evidence.quote,
                normalized_claim=evidence.normalized_claim,
                summary=evidence.normalized_claim,
                relevance_score=9.0,
            )],
            "claims": [],
            "relations": [],
            "report": None,
        }


@pytest.mark.asyncio
async def test_m3_resume_historical_graph_still_runs_grounding() -> None:
    module = StrictM3EvidenceGraph(grounding_enabled=False)
    module.grounding_enabled = True
    grounder = RecordingGrounder()
    module._grounder = grounder
    state = PipelineState(
        input_question="Q",
        literature_results=[],
        evidence_graph=EvidenceGraph(nodes=[
            EvidenceNode(
                id="N_old",
                type=EvidenceNodeType.EVIDENCE,
                label="historical evidence",
            )
        ]),
        m2_knowledge_export=M2KnowledgeExport(runs=[
            M2KnowledgeRun(
                sub_question="SQ",
                papers=[M2PaperExport(paper_id="p1", title="Paper")],
                evidence=[M2EvidenceExport(
                    evidence_id="e1",
                    paper_id="p1",
                    chunk_id="c1",
                    quote="evidence",
                    normalized_claim="claim",
                    relevance_score=0.9,
                )],
            )
        ]),
    )

    result = await module(state)

    assert result["evidence_graph"].nodes
    assert grounder.calls == 1


@pytest.mark.asyncio
async def test_m3_does_not_reground_already_grounded_evidence() -> None:
    module = StrictM3EvidenceGraph(grounding_enabled=False)
    module.grounding_enabled = True
    grounder = RecordingGrounder()
    module._grounder = grounder
    state = PipelineState(
        input_question="Q",
        literature_results=[],
        evidence_graph=EvidenceGraph(nodes=[
            EvidenceNode(
                id="GEV_e1",
                type=EvidenceNodeType.EVIDENCE,
                label="already grounded",
                metadata={
                    "evidence_id": "e1",
                    "provenance_level": "m2_fulltext_grounded",
                },
            )
        ]),
        m2_knowledge_export=M2KnowledgeExport(runs=[
            M2KnowledgeRun(
                sub_question="SQ",
                papers=[M2PaperExport(paper_id="p1", title="Paper")],
                evidence=[M2EvidenceExport(
                    evidence_id="e1",
                    paper_id="p1",
                    chunk_id="c1",
                    quote="evidence",
                    normalized_claim="claim",
                    relevance_score=0.9,
                )],
            )
        ]),
    )

    result = await module(state)

    assert result["evidence_graph"].nodes
    assert grounder.calls == 0


@pytest.mark.asyncio
async def test_m3_grounding_enabled_without_citable_export_fails() -> None:
    module = StrictM3EvidenceGraph(grounding_enabled=False)
    module.grounding_enabled = True
    module._grounder = RecordingGrounder()
    state = PipelineState(
        input_question="Q",
        literature_results=[LiteratureResult(
            sub_question="SQ",
            papers_retrieved=1,
            knowledge_entries=[KnowledgeEntry(
                id="k1",
                type=KnowledgeEntryType.ESTABLISHED_FACT,
                content="fact",
                source_paper_id="p1",
            )],
        )],
    )

    with pytest.raises(RuntimeError, match="no citable evidence"):
        await module(state)


def test_m5_requires_every_evidence_item_to_match_declared_paper() -> None:
    context = GraphContext(
        available_evidence_ids=["e1", "e2"],
        available_paper_ids=["p1", "p2", "p3"],
        evidence_to_paper={"e1": "p1", "e2": "p2"},
    )
    plan = ResearchPlan(
        hypothesis_id="H1",
        evidence_links=[ResearchPlanEvidenceLink(
            plan_element="procedure:1",
            supporting_evidence_ids=["e1", "e2"],
            source_paper_ids=["p1", "p3"],
            support_status="supported",
        )],
    )

    cleaned = StrictM5ResearchPlan._sanitize_evidence_links(plan, context)

    assert cleaned.evidence_links[0].support_status == "hypothesis_to_validate"
    assert cleaned.supporting_evidence_ids == []
    assert cleaned.source_paper_ids == []


class CacheReadingWorkflow:
    def __init__(self) -> None:
        self.calls = 0

    async def run(self, sub_question, papers, search_context=None):
        self.calls += 1
        output = []
        for paper in papers:
            output.append(PaperReadingResult(
                paper_id=paper.paper_id,
                evidence=[EvidenceChunk(
                    evidence_id=f"ev-{paper.paper_id}",
                    paper_id=paper.paper_id,
                    chunk_id=f"chunk-{paper.paper_id}",
                    quote="cached abstract evidence",
                    normalized_claim="cached candidate yields evidence",
                    relevance_score=0.9,
                )],
                knowledge_entries=[EvidenceLinkedKnowledge(
                    entry_id=f"ke-{paper.paper_id}",
                    entry_type=KnowledgeEntryType.ESTABLISHED_FACT,
                    content="cached candidate yields evidence",
                    confidence=ConfidenceLevel.HIGH,
                    evidence_ids=[f"ev-{paper.paper_id}"],
                )],
            ))
        return output


class MustNotSearch:
    async def run(self, *args, **kwargs):
        raise AssertionError("live search should not run after a usable cache hit")


@pytest.mark.asyncio
async def test_agentic_cache_hit_is_read_before_pending_grounding(tmp_path) -> None:
    store = PaperStore(tmp_path)
    cached = M2PaperExport(
        paper_id="cached-1",
        title="Cached paper",
        abstract="A real cached abstract.",
        sources=["paper_store"],
    )
    keys = store.upsert_papers([cached], run_id="old", round=0)
    store.record_query("gap query", keys, run_id="old", round=0)
    gap = EvidenceGap(
        description="missing evidence",
        suggested_queries=["gap query"],
        target_sub_question="SQ",
    )
    state = PipelineState(
        input_question="Q",
        search_round=1,
        run_id="run",
        memory_cache_dir=str(tmp_path),
        literature_results=[LiteratureResult(sub_question="SQ")],
        evidence_gaps=[gap],
        search_ledger=SearchLedger(),
    )
    reader = CacheReadingWorkflow()
    adapter = StrictAgenticM2Adapter(
        search_agent=MustNotSearch(),
        reading_workflow=reader,
    )

    result = await adapter(state)

    assert reader.calls == 1
    assert result["evidence_gaps"][0].status == "pending_grounding"
    assert result["m2_knowledge_export"].runs[0].evidence
    assert result["m2_knowledge_export"].runs[0].knowledge_entries


def test_registry_selects_strict_builtin_contracts() -> None:
    config = PipelineConfig(
        search=SearchConfig(implementation="agentic"),
        entity_embedding_model="text-embedding-v3",
    )
    instances = ModuleRegistry.build_all(config)

    assert isinstance(instances["m2"], StrictAgenticM2Module)
    assert isinstance(instances["m3"], StrictM3EvidenceGraph)
    assert isinstance(instances["m4"], StrictM4HypothesisGeneration)
    assert isinstance(instances["m5"], StrictM5ResearchPlan)
    assert isinstance(instances["m6"], StrictM6ReviewIteration)
    assert instances["m2"].adapter.entity_embedding_model == "text-embedding-v3"
    assert instances["m2"].adapter.entity_embedding_base_url == (
        config.evaluation.embedding.base_url
    )
    assert instances["m2"].adapter.entity_embedding_key_env == (
        config.evaluation.embedding.api_key_env_var
    )
    assert instances["m3"].entity_embedding_base_url == (
        config.evaluation.embedding.base_url
    )
    assert instances["m3"].entity_embedding_key_env == (
        config.evaluation.embedding.api_key_env_var
    )

    # Default config (no entity_embedding_model) → empty string, no embedding.
    config_nonembed = PipelineConfig(search=SearchConfig(implementation="agentic"))
    instances_nonembed = ModuleRegistry.build_all(config_nonembed)
    assert instances_nonembed["m2"].adapter.entity_embedding_model == ""


def test_registry_respects_explicit_empty_entity_embedding_model() -> None:
    config = PipelineConfig(
        search=SearchConfig(implementation="agentic"),
        entity_embedding_model="",
    )
    instances = ModuleRegistry.build_all(config)

    assert instances["m2"].adapter.entity_embedding_model == ""
    assert instances["m3"].entity_embedding_model == ""


class EmptyGrounder:
    async def run(self, state):
        return {
            "evidence_records": [],
            "claims": [],
            "relations": [],
            "report": None,
        }


@pytest.mark.asyncio
async def test_m3_grounding_success_without_current_provenance_fails() -> None:
    module = StrictM3EvidenceGraph(grounding_enabled=False)
    module.grounding_enabled = True
    module._grounder = EmptyGrounder()
    state = PipelineState(
        input_question="Q",
        literature_results=[],
        evidence_graph=EvidenceGraph(nodes=[
            EvidenceNode(
                id="N_old",
                type=EvidenceNodeType.EVIDENCE,
                label="historical evidence",
            )
        ]),
        m2_knowledge_export=M2KnowledgeExport(runs=[
            M2KnowledgeRun(
                sub_question="SQ",
                papers=[M2PaperExport(paper_id="p1", title="Paper")],
                evidence=[M2EvidenceExport(
                    evidence_id="e1",
                    paper_id="p1",
                    chunk_id="c1",
                    quote="evidence",
                    normalized_claim="claim",
                    relevance_score=0.9,
                )],
            )
        ]),
    )

    with pytest.raises(RuntimeError, match="grounding failed"):
        await module(state)


def test_registry_preserves_custom_registered_m3() -> None:
    from hypoforge.modules.m3_evidence_graph import M3EvidenceGraph

    class CustomM3(M3EvidenceGraph):
        pass

    original = ModuleRegistry._modules.get("m3")
    ModuleRegistry._modules["m3"] = CustomM3
    try:
        config = PipelineConfig(search=SearchConfig(implementation="agentic"))
        instances = ModuleRegistry.build_all(config)
        assert isinstance(instances["m3"], CustomM3)
    finally:
        if original is None:
            ModuleRegistry._modules.pop("m3", None)
        else:
            ModuleRegistry._modules["m3"] = original


def test_semantic_alignment_chat_uses_user_prompt_for_qwen_style_client() -> None:
    from hypoforge import task_alignment

    class QwenStyleClient:
        def __init__(self) -> None:
            self.system_prompt = None
            self.user_prompt = None

        async def chat(self, system_prompt="", user_prompt=""):
            self.system_prompt = system_prompt
            self.user_prompt = user_prompt
            return "yes — aligned"

    client = QwenStyleClient()
    response = task_alignment._call_chat(client, "alignment question")

    assert response.startswith("yes")
    assert client.system_prompt == ""
    assert client.user_prompt == "alignment question"


class FailingGroundingSynthesisClient:
    model = "fake-model"

    async def structured_chat(self, **kwargs):
        raise ConnectionError("synthetic synthesis outage")


@pytest.mark.asyncio
async def test_grounding_llm_synthesis_failure_cannot_return_fallback(tmp_path) -> None:
    import hypoforge.strict_contracts as strict
    from hypoforge.modules.m3_grounding.workflow import GroundingWorkflow

    workflow = GroundingWorkflow(
        client=FailingGroundingSynthesisClient(),
        mode="llm",
        cache_dir=str(tmp_path),
    )
    evidence = M2EvidenceExport(
        evidence_id="e-synth",
        paper_id="p1",
        chunk_id="c1",
        quote="quoted evidence",
        normalized_claim="atomic claim",
        relevance_score=0.9,
    )
    item = {"query": "Q", "score": 0.9, "evidence_item": evidence}

    with pytest.raises(RuntimeError, match="Grounding evidence synthesis failed"):
        await strict._strict_grounding_synthesize_one(workflow, item)


class EmptyRelationDecisionClient:
    model = "fake-model"

    async def structured_chat(self, **kwargs):
        return {"relations": []}


@pytest.mark.asyncio
async def test_grounding_relation_batch_requires_one_decision_per_pair(tmp_path) -> None:
    import hypoforge.strict_contracts as strict
    from hypoforge.modules.m3_grounding.models import RelationPair
    from hypoforge.modules.m3_grounding.workflow import GroundingWorkflow
    from hypoforge.state import AtomicClaim

    workflow = GroundingWorkflow(
        client=EmptyRelationDecisionClient(),
        mode="llm",
        cache_dir=str(tmp_path),
    )
    record = EvidenceRecord(
        id="er1",
        evidence_id="e1",
        paper_id="p1",
        quote="evidence",
        normalized_claim="claim",
        summary="evidence summary",
        relevance_score=9.0,
    )
    claim = AtomicClaim(
        id="c1",
        statement="claim",
        evidence_ids=["e1"],
        paper_ids=["p1"],
    )
    pair = RelationPair(
        id="pair1",
        source="er1",
        target="c1",
        source_type="evidence_record",
        target_type="claim",
        source_evidence_ids=["e1"],
        target_evidence_ids=["e1"],
        source_paper_ids=["p1"],
        target_paper_ids=["p1"],
    )

    with pytest.raises(RuntimeError, match="Partial relation judgment"):
        await strict._strict_grounding_judge_relation_batch(
            workflow,
            [pair],
            {claim.id: claim},
            {record.id: record},
        )


@pytest.mark.asyncio
async def test_configured_grounding_embedding_empty_result_fails(monkeypatch) -> None:
    import hypoforge.strict_contracts as strict
    from hypoforge.modules.m3_grounding.workflow import GroundingWorkflow

    async def empty_embedding(self, texts):
        return []

    monkeypatch.setattr(strict, "_ORIGINAL_GROUNDING_EMBED", empty_embedding)
    workflow = GroundingWorkflow(embedding_model="configured-embedding")

    with pytest.raises(RuntimeError, match="returned no vectors"):
        await strict._strict_grounding_embed(workflow, ["document", "query"])


@pytest.mark.asyncio
async def test_configured_grounding_embedding_vector_count_must_match(monkeypatch) -> None:
    import hypoforge.strict_contracts as strict
    from hypoforge.modules.m3_grounding.workflow import GroundingWorkflow

    async def short_embedding(self, texts):
        return [[1.0, 0.0]]

    monkeypatch.setattr(strict, "_ORIGINAL_GROUNDING_EMBED", short_embedding)
    workflow = GroundingWorkflow(embedding_model="configured-embedding")

    with pytest.raises(RuntimeError, match="1 vector"):
        await strict._strict_grounding_embed(workflow, ["document", "query"])


class MalformedScoutClient:
    async def structured_chat(self, **kwargs):
        return {"notes": []}


@pytest.mark.asyncio
async def test_configured_scout_cannot_emit_fallback_notes() -> None:
    import hypoforge.strict_contracts as strict
    from hypoforge.literature.search.scout import ScoutReader

    reader = ScoutReader(client=MalformedScoutClient())
    paper = PaperRecord(
        paper_id="p-scout",
        title="Relevant paper",
        abstract="Relevant evidence about the research question.",
        sources=["test"],
    )

    with pytest.raises(RuntimeError, match="fallback notes"):
        await strict._strict_scout_read(reader, "research question", [paper])


@pytest.mark.asyncio
async def test_unconfigured_scout_may_use_explicit_conservative_mode() -> None:
    import hypoforge.strict_contracts as strict
    from hypoforge.literature.search.scout import ScoutReader

    reader = ScoutReader(client=None)
    paper = PaperRecord(
        paper_id="p-scout-rule",
        title="Rule screened paper",
        abstract="Some abstract text.",
        sources=["test"],
    )

    notes = await strict._strict_scout_read(reader, "question", [paper])
    assert len(notes) == 1
    assert notes[0].paper_id == paper.paper_id


@pytest.mark.asyncio
async def test_m3_llm_mode_without_client_fails_closed() -> None:
    module = StrictM3EvidenceGraph(mode="llm")
    module.client = None
    state = PipelineState(
        input_question="Q",
        literature_results=[LiteratureResult(
            sub_question="SQ",
            papers_retrieved=1,
            knowledge_entries=[KnowledgeEntry(
                id="k1",
                type=KnowledgeEntryType.ESTABLISHED_FACT,
                content="fact",
                source_paper_id="p1",
            )],
        )],
    )

    with pytest.raises(RuntimeError, match="explicitly requires an LLM client"):
        await module(state)


@pytest.mark.asyncio
async def test_agentic_stop_reason_error_is_raised(monkeypatch) -> None:
    import hypoforge.strict_contracts as strict
    from hypoforge.literature.models import SearchRunResult, StopReason

    async def errored_run(self, *args, **kwargs):
        return SearchRunResult(
            sub_question="SQ",
            stop_reason=StopReason.ERROR,
            errors=["scout_reader: simulated failure"],
        )

    monkeypatch.setattr(strict, "_ORIGINAL_SEARCH_AGENT_RUN", errored_run)

    with pytest.raises(RuntimeError, match="StopReason.ERROR"):
        await strict._strict_search_agent_run(object())


@pytest.mark.asyncio
async def test_agentic_non_error_stop_reason_remains_explicit_recovery(monkeypatch) -> None:
    import hypoforge.strict_contracts as strict
    from hypoforge.literature.models import SearchRunResult, StopReason

    async def bounded_run(self, *args, **kwargs):
        return SearchRunResult(
            sub_question="SQ",
            stop_reason=StopReason.TIME_BUDGET,
        )

    monkeypatch.setattr(strict, "_ORIGINAL_SEARCH_AGENT_RUN", bounded_run)
    result = await strict._strict_search_agent_run(object())
    assert result.stop_reason is StopReason.TIME_BUDGET


class M4VerdictClient:
    def __init__(self, payload):
        self.payload = payload

    async def structured_chat(self, **kwargs):
        return self.payload


@pytest.mark.asyncio
async def test_m4_critic_cannot_readmit_all_rejected_candidates() -> None:
    module = StrictM4HypothesisGeneration(mode="multi_agent")
    candidates = [
        HypothesisCard(hypothesis_id="H1", statement="hypothesis one"),
        HypothesisCard(hypothesis_id="H2", statement="hypothesis two"),
    ]
    module.client = M4VerdictClient([
        {"hypothesis_id": "H1", "pass": False, "critique": "bad", "issues": []},
        {"hypothesis_id": "H2", "pass": False, "critique": "bad", "issues": []},
    ])

    with pytest.raises(RuntimeError, match="rejected every candidate"):
        await module._run_critic(PipelineState(input_question="Q"), candidates)


@pytest.mark.asyncio
async def test_m4_falsifiability_requires_verdict_for_every_candidate() -> None:
    module = StrictM4HypothesisGeneration(mode="multi_agent")
    candidates = [
        HypothesisCard(hypothesis_id="H1", statement="hypothesis one"),
        HypothesisCard(hypothesis_id="H2", statement="hypothesis two"),
    ]
    module.client = M4VerdictClient([
        {
            "hypothesis_id": "H1",
            "is_falsifiable": True,
            "specificity_score": 0.8,
            "assessment": "testable",
        }
    ])

    with pytest.raises(RuntimeError, match="candidate IDs"):
        await module._run_falsifiability(
            PipelineState(input_question="Q"), candidates
        )


def test_m4_ranker_incomplete_top_k_cannot_fallback() -> None:
    module = StrictM4HypothesisGeneration(top_k=2, mode="multi_agent")
    candidates = [
        HypothesisCard(hypothesis_id="H1", statement="hypothesis one"),
        HypothesisCard(hypothesis_id="H2", statement="hypothesis two"),
    ]
    dimensions = {
        "novelty": 0.8,
        "scientific_soundness": 0.8,
        "testability": 0.8,
        "evidence_consistency": 0.8,
    }

    with pytest.raises(RuntimeError, match="incomplete valid ranking"):
        module._attach_rankings(
            candidates,
            [{
                "hypothesis_id": "H1",
                "ranking_rationale": "only one result",
                "scores": dimensions,
            }],
        )


@pytest.mark.asyncio
async def test_m5_empty_top_hypotheses_is_not_success() -> None:
    module = StrictM5ResearchPlan()
    state = PipelineState(input_question="Q", top_hypotheses=[])

    with pytest.raises(RuntimeError, match="at least one top hypothesis"):
        await module(state)
