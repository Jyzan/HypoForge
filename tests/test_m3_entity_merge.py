from __future__ import annotations

import pytest

from hypoforge.entity_graph_merge import EntityGraphMerger
from hypoforge.modules.m3_evidence_graph import M3EvidenceGraph
from hypoforge.observability import RunEventRecorder, bind_recorder
from hypoforge.state import (
    EvidenceEdge,
    EvidenceEdgeRelation,
    EvidenceGraph,
    EvidenceNode,
    EvidenceNodeType,
    EntityMergeMember,
    EntityMergeRecord,
    KnowledgeEntry,
    KnowledgeEntryType,
    M2EvidenceExport,
    M2KnowledgeExport,
    M2KnowledgeRun,
    M2PaperExport,
    PipelineState,
    ProblemCard,
    TaskContract,
    TaskEntity,
)


class FixedEmbeddingBackend:
    def __init__(self, vectors_by_text: dict[str, list[float]]) -> None:
        self.vectors_by_text = vectors_by_text

    async def __call__(self, texts: list[str], model: str) -> list[list[float]]:
        assert model == "text-embedding-v3"
        return [self.vectors_by_text[text] for text in texts]


def _entity(node_id: str, label: str) -> EvidenceNode:
    return EvidenceNode(
        id=node_id,
        type=EvidenceNodeType.ENTITY,
        label=label,
    )


def test_entity_merge_log_round_trips_through_evidence_graph() -> None:
    """Removing the typed audit field must break lossless graph snapshots."""
    record = EntityMergeRecord(
        merge_id="merge-001",
        round_index=0,
        threshold=0.92,
        embedding_model="text-embedding-v3",
        method="embedding_cosine",
        canonical_node_id="ENT_llm",
        canonical_name="large language model",
        members=[
            EntityMergeMember(
                node_id="ENT_llm",
                name="large language model",
                similarity_to_canonical=1.0,
                metadata={"source": "task_contract"},
            )
        ],
    )

    restored = EvidenceGraph(entity_merge_log=[record]).model_dump()

    assert restored["entity_merge_log"][0]["canonical_name"] == (
        "large language model"
    )
    assert restored["entity_merge_log"][0]["members"][0]["metadata"] == {
        "source": "task_contract"
    }


@pytest.mark.asyncio
async def test_exact_entity_merge_redirects_and_deduplicates_edges() -> None:
    """Missing endpoint rewrite or edge aggregation loses graph evidence."""
    graph = EvidenceGraph(
        version=3,
        nodes=[
            EvidenceNode(
                id="ENT_llm",
                type=EvidenceNodeType.ENTITY,
                label="Large Language Model",
                metadata={"normalised_from": ["LLM"]},
            ),
            EvidenceNode(
                id="N_reading_llm",
                type=EvidenceNodeType.ENTITY,
                label=" large language model. ",
                metadata={"source": "reading"},
            ),
            EvidenceNode(
                id="EV_1",
                type=EvidenceNodeType.EVIDENCE,
                label="evaluation result",
            ),
        ],
        edges=[
            EvidenceEdge(
                source="EV_1",
                target="ENT_llm",
                relation=EvidenceEdgeRelation.INVOLVES,
                confidence=0.7,
                rationale="first rationale",
                evidence_ids=["paper-a"],
            ),
            EvidenceEdge(
                source="EV_1",
                target="N_reading_llm",
                relation=EvidenceEdgeRelation.INVOLVES,
                confidence=0.9,
                rationale="second rationale",
                evidence_ids=["paper-b"],
            ),
            EvidenceEdge(
                source="ENT_llm",
                target="N_reading_llm",
                relation=EvidenceEdgeRelation.SAME_AS,
                evidence_ids=["paper-c"],
            ),
        ],
    )

    outcome = await EntityGraphMerger(
        threshold=0.92,
        embedding_model="text-embedding-v3",
    ).merge(graph, task_entity_names=[], round_index=2)

    assert [node.id for node in outcome.graph.nodes] == ["ENT_llm", "EV_1"]
    assert outcome.graph.version == 4
    assert outcome.redirected_edge_count == 1
    assert outcome.deduplicated_edge_count == 1
    assert outcome.removed_self_loop_count == 1
    assert len(outcome.graph.edges) == 1
    merged_edge = outcome.graph.edges[0]
    assert (merged_edge.source, merged_edge.target) == ("EV_1", "ENT_llm")
    assert merged_edge.confidence == 0.9
    assert merged_edge.evidence_ids == ["paper-a", "paper-b"]
    assert merged_edge.rationale == "first rationale | second rationale"

    canonical = outcome.graph.nodes[0]
    assert canonical.metadata["aliases"] == [
        "Large Language Model",
        "large language model.",
    ]
    assert canonical.metadata["merged_node_ids"] == ["N_reading_llm"]
    assert canonical.metadata["normalised_from"] == ["LLM"]
    assert outcome.graph.entity_merge_log[0].method == "exact"
    assert [member.node_id for member in outcome.graph.entity_merge_log[0].members] == [
        "ENT_llm",
        "N_reading_llm",
    ]


@pytest.mark.asyncio
async def test_entity_merge_does_not_mutate_input_graph() -> None:
    """In-place merging would corrupt the state retained for safe fallback."""
    graph = EvidenceGraph(
        nodes=[
            EvidenceNode(
                id="ENT_a",
                type=EvidenceNodeType.ENTITY,
                label="RAG",
            ),
            EvidenceNode(
                id="ENT_b",
                type=EvidenceNodeType.ENTITY,
                label="rag",
            ),
        ]
    )
    before = graph.model_dump()

    outcome = await EntityGraphMerger().merge(
        graph,
        task_entity_names=[],
        round_index=0,
    )

    assert outcome.graph is not graph
    assert graph.model_dump() == before


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("cosine", "expected_count"),
    [(0.92, 1), (0.9199, 2)],
)
async def test_semantic_merge_honours_inclusive_threshold(
    cosine: float,
    expected_count: int,
) -> None:
    """Changing >= to > or rounding a sub-threshold score would mismerge."""
    perpendicular = (1.0 - cosine**2) ** 0.5
    backend = FixedEmbeddingBackend(
        {
            "retrieval augmented generation": [1.0, 0.0],
            "rag": [cosine, perpendicular],
        }
    )
    graph = EvidenceGraph(
        nodes=[
            _entity("ENT_long", "retrieval augmented generation"),
            _entity("ENT_short", "RAG"),
        ]
    )

    outcome = await EntityGraphMerger(
        threshold=0.92,
        embedding_model="text-embedding-v3",
        embed_texts=backend,
    ).merge(graph, task_entity_names=[], round_index=0)

    entity_nodes = [
        node for node in outcome.graph.nodes if node.type == EvidenceNodeType.ENTITY
    ]
    assert len(entity_nodes) == expected_count


@pytest.mark.asyncio
async def test_complete_linkage_blocks_chain_merge() -> None:
    """Single-link clustering would incorrectly absorb C through B."""
    backend = FixedEmbeddingBackend(
        {
            "entity a": [1.0, 0.0],
            "entity b": [0.95, 0.3122498999],
            "entity c": [0.805, 0.5932748098],
        }
    )
    graph = EvidenceGraph(
        nodes=[
            _entity("ENT_a", "entity a"),
            _entity("ENT_b", "entity b"),
            _entity("ENT_c", "entity c"),
        ]
    )

    outcome = await EntityGraphMerger(
        threshold=0.92,
        embedding_model="text-embedding-v3",
        embed_texts=backend,
    ).merge(graph, task_entity_names=[], round_index=0)

    remaining_ids = {node.id for node in outcome.graph.nodes}
    assert remaining_ids == {"ENT_a", "ENT_c"}
    assert [member.node_id for member in outcome.graph.entity_merge_log[0].members] == [
        "ENT_a",
        "ENT_b",
    ]


@pytest.mark.asyncio
async def test_task_contract_name_wins_over_ent_prefix_for_semantic_merge() -> None:
    """Ignoring M1 canonical names would choose an implementation ID instead."""
    backend = FixedEmbeddingBackend(
        {
            "large language model": [1.0, 0.0],
            "llms": [0.97, 0.2431049156],
        }
    )
    graph = EvidenceGraph(
        nodes=[
            _entity("ENT_long", "large language model"),
            _entity("N_reading_llms", "LLMs"),
        ]
    )

    outcome = await EntityGraphMerger(
        threshold=0.92,
        embedding_model="text-embedding-v3",
        embed_texts=backend,
    ).merge(graph, task_entity_names=["LLMs"], round_index=0)

    assert [node.id for node in outcome.graph.nodes] == ["N_reading_llms"]
    assert outcome.graph.entity_merge_log[0].canonical_name == "LLMs"
    assert outcome.graph.entity_merge_log[0].method == "embedding_cosine"


@pytest.mark.asyncio
async def test_non_entity_nodes_are_never_merged() -> None:
    """An over-broad node filter would delete distinct evidence records."""
    graph = EvidenceGraph(
        nodes=[
            EvidenceNode(
                id="EV_1",
                type=EvidenceNodeType.EVIDENCE,
                label="same label",
            ),
            EvidenceNode(
                id="EV_2",
                type=EvidenceNodeType.EVIDENCE,
                label="same label",
            ),
        ]
    )

    outcome = await EntityGraphMerger().merge(
        graph,
        task_entity_names=[],
        round_index=0,
    )

    assert [node.id for node in outcome.graph.nodes] == ["EV_1", "EV_2"]
    assert outcome.merge_count == 0


@pytest.mark.asyncio
async def test_embedding_failure_keeps_exact_merge_and_reports_degradation() -> None:
    """A network outage must not discard safe exact merges or fail M3."""
    async def fail_embeddings(texts: list[str], model: str) -> list[list[float]]:
        raise ConnectionError("embedding service unavailable")

    graph = EvidenceGraph(
        nodes=[
            _entity("ENT_rag", "RAG"),
            _entity("N_rag", "rag"),
            _entity("ENT_retrieval", "retrieval augmented generation"),
        ]
    )

    outcome = await EntityGraphMerger(
        embedding_model="text-embedding-v3",
        embed_texts=fail_embeddings,
    ).merge(graph, task_entity_names=[], round_index=0)

    assert outcome.merge_count == 1
    assert outcome.entity_count_after == 2
    assert "embedding service unavailable" in outcome.degraded_reason
    assert outcome.graph.entity_merge_log[0].method == "exact"


@pytest.mark.asyncio
async def test_m3_integration_merges_reused_graph_and_emits_events(tmp_path) -> None:
    """The no-new-entry return path must not bypass final entity merging."""
    recorder = RunEventRecorder(tmp_path / "run", "run")
    state = PipelineState(
        input_question="How can LLMs reduce hallucinations?",
        run_id="run",
        problem_card=ProblemCard(
            original_question="How can LLMs reduce hallucinations?",
            key_entities=["LLMs"],
            task_contract=TaskContract(
                entities=[
                    TaskEntity(
                        entity_id="TE_1",
                        name="LLMs",
                        source_mention="LLMs",
                    )
                ],
                source="m1",
            ),
        ),
        evidence_graph=EvidenceGraph(
            nodes=[
                _entity("ENT_llm", "LLMs"),
                _entity("N_reading_llm", "llms"),
            ]
        ),
    )
    module = M3EvidenceGraph(
        mode="rule",
        entity_merge_enabled=True,
        entity_merge_similarity_threshold=0.92,
    )

    with bind_recorder(recorder):
        result = await module(state)

    graph = result["evidence_graph"]
    assert len(graph.nodes) == 1
    assert graph.entity_merge_log[0].canonical_name == "LLMs"
    event_types = [
        event["event_type"]
        for event in recorder.read_events()
        if event["event_type"].startswith("entity_merge")
    ]
    assert event_types == ["entity_merge_started", "entity_merge_completed"]


@pytest.mark.asyncio
async def test_m3_integration_embedding_failure_emits_degraded_event(
    tmp_path,
    monkeypatch,
) -> None:
    """Embedding transport failure must be observable without failing M3."""
    async def fail_embeddings(texts: list[str], model: str) -> list[list[float]]:
        raise ConnectionError("temporary embedding outage")

    # The production strict normalizer validates the shared embedding config
    # before delegating, even when this fixture has no M2 entity surfaces.
    # Keep that independent precondition satisfied; the injected merger
    # backend below is still the component deliberately made to fail.
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    recorder = RunEventRecorder(tmp_path / "run", "run")
    state = PipelineState(
        input_question="How can RAG reduce hallucinations?",
        run_id="run",
        evidence_graph=EvidenceGraph(
            nodes=[
                _entity("ENT_rag", "RAG"),
                _entity("N_rag", "rag"),
                _entity("ENT_full", "retrieval augmented generation"),
            ]
        ),
    )
    module = M3EvidenceGraph(
        mode="rule",
        entity_embedding_model="text-embedding-v3",
        entity_merge_enabled=True,
        entity_merge_embedding_backend=fail_embeddings,
    )

    with bind_recorder(recorder):
        result = await module(state)

    assert len(result["evidence_graph"].nodes) == 2
    events = [
        event
        for event in recorder.read_events()
        if event["event_type"].startswith("entity_merge")
    ]
    assert [event["event_type"] for event in events] == [
        "entity_merge_started",
        "entity_merge_degraded",
        "entity_merge_completed",
    ]
    assert "temporary embedding outage" in events[1]["message"]


@pytest.mark.asyncio
async def test_merge_preserves_absorbed_entry_ids_for_incremental_dedup() -> None:
    """Dropping an absorbed N_<entry_id> must not recreate it next round."""
    graph = EvidenceGraph(
        nodes=[
            _entity("ENT_llm", "large language model"),
            EvidenceNode(
                id="N_entry-a",
                type=EvidenceNodeType.ENTITY,
                label="large language model",
                metadata={"entry_id": "entry-a"},
            ),
        ]
    )
    entry = KnowledgeEntry(
        id="entry-a",
        type=KnowledgeEntryType.KEY_ENTITY,
        content="large language model",
    )

    outcome = await EntityGraphMerger().merge(
        graph,
        task_entity_names=[],
        round_index=0,
    )

    assert outcome.graph.nodes[0].metadata["entry_ids"] == ["entry-a"]
    assert M3EvidenceGraph._find_new_entries([entry], outcome.graph) == []


@pytest.mark.asyncio
async def test_each_merge_record_has_its_own_edge_statistics() -> None:
    """Copying global edge counts into every group makes the audit misleading."""
    graph = EvidenceGraph(
        nodes=[
            _entity("ENT_a", "entity a"),
            _entity("N_a", "ENTITY A"),
            _entity("ENT_b", "entity b"),
            _entity("N_b", "ENTITY B"),
            EvidenceNode(id="EV_1", type=EvidenceNodeType.EVIDENCE, label="evidence"),
        ],
        edges=[
            EvidenceEdge(
                source="EV_1",
                target="N_a",
                relation=EvidenceEdgeRelation.INVOLVES,
            )
        ],
    )

    outcome = await EntityGraphMerger().merge(
        graph,
        task_entity_names=[],
        round_index=0,
    )

    records = {record.canonical_name: record for record in outcome.graph.entity_merge_log}
    assert records["entity a"].redirected_edge_count == 1
    assert records["entity b"].redirected_edge_count == 0


@pytest.mark.asyncio
async def test_joint_cross_group_edge_dedup_is_attributed_to_both_groups() -> None:
    """A duplicate caused jointly by two merges must remain visible in audit."""
    graph = EvidenceGraph(
        nodes=[
            _entity("ENT_a", "entity a"),
            _entity("N_a", "ENTITY A"),
            _entity("ENT_b", "entity b"),
            _entity("N_b", "ENTITY B"),
        ],
        edges=[
            EvidenceEdge(
                source="ENT_a",
                target="ENT_b",
                relation=EvidenceEdgeRelation.INVOLVES,
            ),
            EvidenceEdge(
                source="N_a",
                target="N_b",
                relation=EvidenceEdgeRelation.INVOLVES,
            ),
        ],
    )

    outcome = await EntityGraphMerger().merge(
        graph,
        task_entity_names=[],
        round_index=0,
    )

    records = {record.canonical_name: record for record in outcome.graph.entity_merge_log}
    assert outcome.deduplicated_edge_count == 1
    assert records["entity a"].deduplicated_edge_count == 1
    assert records["entity b"].deduplicated_edge_count == 1


@pytest.mark.asyncio
async def test_original_canonical_self_loop_survives_merge() -> None:
    """Only self-loops newly created by endpoint rewriting may be removed."""
    graph = EvidenceGraph(
        nodes=[
            _entity("ENT_a", "entity a"),
            _entity("N_a", "ENTITY A"),
        ],
        edges=[
            EvidenceEdge(
                source="ENT_a",
                target="ENT_a",
                relation=EvidenceEdgeRelation.SAME_AS,
                evidence_ids=["original-loop"],
            ),
            EvidenceEdge(
                source="ENT_a",
                target="N_a",
                relation=EvidenceEdgeRelation.SAME_AS,
                evidence_ids=["merge-loop"],
            ),
        ],
    )

    outcome = await EntityGraphMerger().merge(
        graph,
        task_entity_names=[],
        round_index=0,
    )

    assert outcome.removed_self_loop_count == 1
    assert len(outcome.graph.edges) == 1
    assert outcome.graph.edges[0].evidence_ids == ["original-loop"]


@pytest.mark.asyncio
async def test_similarity_below_threshold_by_tiny_margin_does_not_merge() -> None:
    """A numerical tolerance must not weaken the configured merge boundary."""
    score = 0.92 - 5e-13
    backend = FixedEmbeddingBackend(
        {
            "entity a": [1.0, 0.0],
            "entity b": [score, (1.0 - score**2) ** 0.5],
        }
    )
    graph = EvidenceGraph(
        nodes=[_entity("ENT_a", "entity a"), _entity("ENT_b", "entity b")]
    )

    outcome = await EntityGraphMerger(
        threshold=0.92,
        embedding_model="text-embedding-v3",
        embed_texts=backend,
    ).merge(graph, task_entity_names=[], round_index=0)

    assert outcome.entity_count_after == 2


@pytest.mark.asyncio
async def test_reused_graph_path_runs_grounding_before_final_merge() -> None:
    """Checkpoint resume with no new entries must still consume M2 evidence."""
    class Grounder:
        async def run(self, state: PipelineState) -> dict:
            from hypoforge.modules.m3_grounding.models import EvidenceRecord

            return {
                "evidence_records": [
                    EvidenceRecord(
                        id="REC_1",
                        evidence_id="E1",
                        paper_id="P1",
                        summary="new grounded evidence",
                    )
                ],
                "claims": [],
                "relations": [],
                "report": None,
            }

    export = M2KnowledgeExport(
        runs=[
            M2KnowledgeRun(
                sub_question="question",
                papers=[M2PaperExport(paper_id="P1", title="Paper 1")],
                evidence=[
                    M2EvidenceExport(
                        evidence_id="E1",
                        paper_id="P1",
                        chunk_id="CHK_E1",
                        quote="quote",
                        normalized_claim="claim",
                        relevance_score=0.9,
                    )
                ],
            )
        ]
    )
    state = PipelineState(
        input_question="question",
        evidence_graph=EvidenceGraph(nodes=[_entity("ENT_a", "entity a")]),
        m2_knowledge_export=export,
    )
    module = M3EvidenceGraph(mode="rule", grounding_enabled=False)
    module.grounding_enabled = True
    module._grounder = Grounder()

    result = await module(state)

    assert any(node.id == "GEV_E1" for node in result["evidence_graph"].nodes)
