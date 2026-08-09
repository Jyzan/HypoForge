"""Tests for the knowledge-retention layer:

* ``KnowledgeGraphManager.merge_from_evidence_graph`` (merge semantics)
* overwrite semantics of ``save_from_evidence_graph`` / ``clear_graph``
  remain intact (regression protection)
* lossless ``graph-round-{N}.json`` snapshots + ``load_latest_graph_round``
* M3 incremental fixes: empty entries never wipe the graph, no-new-entry
  rounds still persist, pending_grounding gap confirmation.
"""

import json

import pytest

from hypoforge.memory import (
    KnowledgeGraphManager,
    load_latest_graph_round,
    save_graph_round_snapshot,
    stable_entry_id,
)
from hypoforge.modules.m3_evidence_graph import M3EvidenceGraph
from hypoforge.state import (
    EvidenceEdge,
    EvidenceEdgeRelation,
    EvidenceGap,
    EvidenceGraph,
    EvidenceNode,
    EvidenceNodeType,
    KnowledgeEntry,
    KnowledgeEntryType,
    LiteratureResult,
    PipelineState,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _entry(entry_id: str, content: str, paper_id: str = "PMID:1") -> KnowledgeEntry:
    return KnowledgeEntry(
        id=entry_id,
        type=KnowledgeEntryType.ESTABLISHED_FACT,
        content=content,
        source_paper_id=paper_id,
        source_paper_title=f"Paper {paper_id}",
        entities=["hsp70"],
    )


def _graph_with(node_id: str, label: str = "node") -> EvidenceGraph:
    return EvidenceGraph(
        nodes=[EvidenceNode(id=node_id, type=EvidenceNodeType.CLAIM, label=label)],
        edges=[],
        established_facts=[],
        conflicts=[],
        knowledge_gaps=[],
    )


# ---------------------------------------------------------------------------
# Merge API
# ---------------------------------------------------------------------------


def test_merge_entities_by_normalised_name_and_union_observations(tmp_path):
    mgr = KnowledgeGraphManager(cache_dir=tmp_path / "kg")

    eg1 = EvidenceGraph(nodes=[
        EvidenceNode(id="ENT_HSP70", type=EvidenceNodeType.ENTITY, label="HSP70 node"),
    ])
    eg2 = EvidenceGraph(nodes=[
        # Same entity after normalisation (case / whitespace variants)
        EvidenceNode(id=" ent_hsp70 ", type=EvidenceNodeType.ENTITY, label="alias obs"),
        EvidenceNode(id="ENT_P53", type=EvidenceNodeType.ENTITY, label="p53 node"),
    ])

    mgr.merge_from_evidence_graph(eg1)
    merged = mgr.merge_from_evidence_graph(eg2)

    names = {e.name for e in merged.entities}
    assert names == {"ENT_HSP70", "ENT_P53"}
    hsp70 = next(e for e in merged.entities if e.name == "ENT_HSP70")
    # Observations are unioned (deduplicated)
    assert "HSP70 node" in hsp70.observations
    assert "alias obs" in hsp70.observations

    # Round-trip through disk matches the returned object
    on_disk = mgr.read_graph()
    assert {e.name for e in on_disk.entities} == names


def test_merge_relations_keep_higher_confidence_and_merge_rationale(tmp_path):
    mgr = KnowledgeGraphManager(cache_dir=tmp_path / "kg")

    def eg_with_edge(confidence, rationale):
        return EvidenceGraph(
            nodes=[
                EvidenceNode(id="N_a", type=EvidenceNodeType.CLAIM, label="a"),
                EvidenceNode(id="N_b", type=EvidenceNodeType.CLAIM, label="b"),
            ],
            edges=[EvidenceEdge(
                source="N_a", target="N_b",
                relation=EvidenceEdgeRelation.SUPPORTS,
                confidence=confidence, rationale=rationale,
            )],
        )

    mgr.merge_from_evidence_graph(eg_with_edge(0.6, "first rationale"))
    merged = mgr.merge_from_evidence_graph(eg_with_edge(0.9, "second rationale"))

    supports = [r for r in merged.relations if r.relation_type == "supports"]
    assert len(supports) == 1  # dedup by (from, relation_type, to)
    assert supports[0].confidence == pytest.approx(0.9)
    assert "first rationale" in supports[0].rationale
    assert "second rationale" in supports[0].rationale

    # Merging a lower-confidence duplicate afterwards keeps the higher one
    merged2 = mgr.merge_from_evidence_graph(eg_with_edge(0.2, "first rationale"))
    supports2 = [r for r in merged2.relations if r.relation_type == "supports"]
    assert len(supports2) == 1
    assert supports2[0].confidence == pytest.approx(0.9)


def test_overwrite_semantics_not_broken(tmp_path):
    """save_from_evidence_graph / clear_graph / delete_entities keep
    overwrite semantics (the merge API must not interfere)."""
    mgr = KnowledgeGraphManager(cache_dir=tmp_path / "kg")

    big = EvidenceGraph(nodes=[
        EvidenceNode(id="N_keep_old", type=EvidenceNodeType.CLAIM, label="old"),
        EvidenceNode(id="N_gone", type=EvidenceNodeType.CLAIM, label="gone"),
    ])
    small = EvidenceGraph(nodes=[
        EvidenceNode(id="N_keep_new", type=EvidenceNodeType.CLAIM, label="new"),
    ])

    # save_from_evidence_graph overwrites completely
    mgr.save_from_evidence_graph(big)
    mgr.save_from_evidence_graph(small)
    assert {e.name for e in mgr.read_graph().entities} == {"N_keep_new"}

    # clear_graph empties the store
    mgr.clear_graph()
    assert mgr.read_graph().is_empty

    # merge after clear must NOT resurrect cleared data
    mgr.merge_from_evidence_graph(small)
    assert {e.name for e in mgr.read_graph().entities} == {"N_keep_new"}

    # delete_entities still removes entities
    mgr.delete_entities(["N_keep_new"])
    assert mgr.read_graph().is_empty


# ---------------------------------------------------------------------------
# Round snapshots
# ---------------------------------------------------------------------------


def test_round_snapshot_write_and_load_latest(tmp_path):
    out = tmp_path / "run-output"

    eg0 = _graph_with("N_round0", "round zero")
    eg2 = _graph_with("N_round2", "round two")

    assert save_graph_round_snapshot(eg0, out, 0) is not None
    assert save_graph_round_snapshot(eg2, out, 2) is not None
    # noise file must be ignored by the loader
    (out / "graph-round-notes.json").write_text("{}", encoding="utf-8")

    latest = load_latest_graph_round(out)
    assert latest is not None
    assert [n["id"] for n in latest["nodes"]] == ["N_round2"]

    # Same round overwrites its own file
    eg2b = _graph_with("N_round2_updated", "round two v2")
    save_graph_round_snapshot(eg2b, out, 2)
    latest2 = load_latest_graph_round(out)
    assert [n["id"] for n in latest2["nodes"]] == ["N_round2_updated"]

    # Snapshot is a lossless EvidenceGraph dump (re-parseable)
    parsed = EvidenceGraph(**latest2)
    assert parsed.nodes[0].id == "N_round2_updated"


def test_load_latest_graph_round_missing(tmp_path):
    assert load_latest_graph_round(tmp_path / "does-not-exist") is None
    empty = tmp_path / "empty"
    empty.mkdir()
    assert load_latest_graph_round(empty) is None


# ---------------------------------------------------------------------------
# M3 — empty entries must not destroy the graph
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_m3_empty_entries_preserves_existing_graph():
    existing = _graph_with("N_old", "old claim")
    state = PipelineState(
        input_question="q",
        literature_results=[],
        evidence_graph=existing,
    )
    module = M3EvidenceGraph(mode="rule")
    result = await module(state)
    graph = result["evidence_graph"]
    assert [n.id for n in graph.nodes] == ["N_old"]


@pytest.mark.asyncio
async def test_m3_empty_entries_and_no_graph_returns_empty():
    state = PipelineState(input_question="q", literature_results=[])
    module = M3EvidenceGraph(mode="rule")
    with pytest.raises(RuntimeError, match="no usable knowledge entries"):
        await module(state)


@pytest.mark.asyncio
async def test_m3_empty_entries_loads_persisted_graph_from_memory(tmp_path):
    """No entries + no in-state graph → restore from the JSONL store."""
    cache_dir = tmp_path / "kg-cache"

    # Round 1: build + persist a graph
    entries = [_entry("KE_persist", "Persisted finding about AMPK.")]
    first = await M3EvidenceGraph(mode="rule")(PipelineState(
        input_question="q",
        literature_results=[LiteratureResult(
            sub_question="sq", papers_retrieved=1, knowledge_entries=entries,
        )],
        memory_cache_dir=str(cache_dir),
    ))
    assert any(n.id == "N_KE_persist" for n in first["evidence_graph"].nodes)

    # Round 2: fresh state with NO entries and NO graph → must recover
    # the persisted graph instead of returning an empty one.
    result = await M3EvidenceGraph(mode="rule")(PipelineState(
        input_question="q",
        literature_results=[],
        memory_cache_dir=str(cache_dir),
    ))
    graph = result["evidence_graph"]
    assert len(graph.nodes) > 0
    names = {n.label for n in graph.nodes} | {n.id for n in graph.nodes}
    assert any("KE_persist" in str(n) for n in names)


@pytest.mark.asyncio
async def test_m3_empty_entries_prefers_lossless_round_snapshot(tmp_path):
    """When both a snapshot and the JSONL store exist, the lossless
    round snapshot wins."""
    out_dir = tmp_path / "run-out"
    original = _graph_with("N_snapshot_only", "snapshot claim")
    assert save_graph_round_snapshot(original, out_dir, 3) is not None

    result = await M3EvidenceGraph(mode="rule", output_dir=str(out_dir))(
        PipelineState(input_question="q", literature_results=[])
    )
    assert [n.id for n in result["evidence_graph"].nodes] == ["N_snapshot_only"]


# ---------------------------------------------------------------------------
# M3 — no-new-entry rounds still persist + snapshot
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_m3_no_new_entries_still_persists_and_snapshots(tmp_path):
    entries = [_entry("KE_a", "Hsp70 assists protein folding.")]
    lr = LiteratureResult(
        sub_question="How does Hsp70 work?",
        papers_retrieved=1,
        knowledge_entries=entries,
    )

    module = M3EvidenceGraph(mode="rule")
    first = await module(PipelineState(
        input_question="q", literature_results=[lr],
    ))
    graph = first["evidence_graph"]
    assert any(n.id == "N_KE_a" for n in graph.nodes)

    cache_dir = tmp_path / "kg-cache"
    out_dir = tmp_path / "run-out"
    state2 = PipelineState(
        input_question="q",
        literature_results=[lr],
        evidence_graph=graph,
        memory_cache_dir=str(cache_dir),
        iteration_count=1,
    )
    module2 = M3EvidenceGraph(mode="rule", output_dir=str(out_dir))
    second = await module2(state2)

    # Graph unchanged (no new entries → no re-construction)
    assert {n.id for n in second["evidence_graph"].nodes} == {
        n.id for n in graph.nodes
    }
    # Persistent JSONL store written via merge
    assert (cache_dir / "memory-evidence_graph.jsonl").exists()
    stored = KnowledgeGraphManager(cache_dir=cache_dir).read_graph()
    assert any(e.name == "N_KE_a" for e in stored.entities)
    # Lossless round snapshot written to the run output dir
    snapshot = out_dir / "graph-round-1.json"
    assert snapshot.exists()
    payload = json.loads(snapshot.read_text(encoding="utf-8"))
    assert any(n["id"] == "N_KE_a" for n in payload["nodes"])


@pytest.mark.asyncio
async def test_m3_snapshot_round_prefers_search_round(tmp_path):
    """Supplement rounds number snapshots by search_round."""
    entries = [_entry("KE_c", "Supplement-round evidence about ROS.")]
    out_dir = tmp_path / "run-out"
    module = M3EvidenceGraph(mode="rule", output_dir=str(out_dir))
    await module(PipelineState(
        input_question="q",
        literature_results=[LiteratureResult(
            sub_question="sq", papers_retrieved=1, knowledge_entries=entries,
        )],
        search_round=2,
        iteration_count=1,
    ))
    assert (out_dir / "graph-round-2.json").exists()


@pytest.mark.asyncio
async def test_m3_snapshot_falls_back_to_memory_cache_dir(tmp_path):
    entries = [_entry("KE_b", "NAD+ boosts mitochondrial function.")]
    lr = LiteratureResult(
        sub_question="What does NAD+ do?",
        papers_retrieved=1,
        knowledge_entries=entries,
    )
    cache_dir = tmp_path / "cache"
    module = M3EvidenceGraph(mode="rule")  # no output_dir
    await module(PipelineState(
        input_question="q",
        literature_results=[lr],
        memory_cache_dir=str(cache_dir),
        iteration_count=0,
    ))
    assert (cache_dir / "graph-round-0.json").exists()


# ---------------------------------------------------------------------------
# M3 — incremental append across rounds (dedup via graph, not instance state)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_m3_incremental_appends_new_entries_across_instances(tmp_path):
    old_entry = _entry("KE_old", "Hsp70 is a heat shock protein.")
    new_entry = _entry("KE_new", "Hsp70 inhibition sensitises tumours.", "PMID:2")

    module1 = M3EvidenceGraph(mode="rule")
    first = await module1(PipelineState(
        input_question="q",
        literature_results=[LiteratureResult(
            sub_question="sq", papers_retrieved=1, knowledge_entries=[old_entry],
        )],
    ))
    graph = first["evidence_graph"]

    # A *fresh* module instance (no shared instance state) must still dedup
    module2 = M3EvidenceGraph(mode="rule")
    second = await module2(PipelineState(
        input_question="q",
        literature_results=[LiteratureResult(
            sub_question="sq", papers_retrieved=2,
            knowledge_entries=[old_entry, new_entry],
        )],
        evidence_graph=graph,
    ))
    ids = {n.id for n in second["evidence_graph"].nodes}
    assert "N_KE_old" in ids
    assert "N_KE_new" in ids
    # old entry not duplicated
    assert sum(1 for i in ids if i == "N_KE_old") == 1


# ---------------------------------------------------------------------------
# M3 — pending_grounding gap confirmation
# ---------------------------------------------------------------------------


def _pending_gap(target: str) -> EvidenceGap:
    return EvidenceGap(
        description=f"Missing evidence for: {target}",
        target_sub_question=target,
        status="pending_grounding",
    )


@pytest.mark.asyncio
async def test_gap_with_gain_is_closed():
    old_entry = _entry("KE_old2", "Baseline finding about p53.")
    new_entry = _entry("KE_new2", "New p53 mechanism found.", "PMID:9")

    module = M3EvidenceGraph(mode="rule")
    first = await module(PipelineState(
        input_question="q",
        literature_results=[LiteratureResult(
            sub_question="How does p53 work?",
            papers_retrieved=1,
            knowledge_entries=[old_entry],
        )],
    ))
    graph = first["evidence_graph"]

    gap = _pending_gap("How does p53 work?")
    state = PipelineState(
        input_question="q",
        literature_results=[
            LiteratureResult(
                sub_question="How does p53 work?",
                papers_retrieved=1,
                knowledge_entries=[old_entry],
            ),
            LiteratureResult(
                sub_question="How does p53 work?",
                papers_retrieved=1,
                knowledge_entries=[new_entry],
            ),
        ],
        evidence_graph=graph,
        evidence_gaps=[gap],
    )
    result = await module(state)

    assert "evidence_gaps" in result
    updated = result["evidence_gaps"]
    assert len(updated) == 1
    assert updated[0].status == "closed"
    # state object untouched (copy-on-write patch)
    assert state.evidence_gaps[0].status == "pending_grounding"
    # CONTRACT (P2): per-gap gain counters travel via metrics["m3_gap_gain"]
    # (PipelineState has no dedicated gap_gain field; pipeline.py rejects
    # unknown patch keys).
    assert result["metrics"]["m3_gap_gain"] == {gap.gap_id: 1}


@pytest.mark.asyncio
async def test_gap_without_gain_becomes_unimprovable_at_limit():
    entry = _entry("KE_same", "No change in evidence base.")
    module = M3EvidenceGraph(mode="rule")
    first = await module(PipelineState(
        input_question="q",
        literature_results=[LiteratureResult(
            sub_question="unrelated sub-question",
            papers_retrieved=1,
            knowledge_entries=[entry],
        )],
    ))
    graph = first["evidence_graph"]

    gap = _pending_gap("target that matches nothing new")
    state = PipelineState(
        input_question="q",
        literature_results=[LiteratureResult(
            sub_question="unrelated sub-question",
            papers_retrieved=1,
            knowledge_entries=[entry],
        )],
        evidence_graph=graph,
        evidence_gaps=[gap],
    )
    # default gap_no_improvement_limit == 1 → first no-gain round closes it
    result = await module(state)
    updated = result["evidence_gaps"]
    assert updated[0].attempts == 1
    assert updated[0].status == "unimprovable"


@pytest.mark.asyncio
async def test_gap_without_gain_increments_attempts_below_limit():
    entry = _entry("KE_same3", "Still no relevant new papers.")
    module = M3EvidenceGraph(mode="rule", gap_no_improvement_limit=3)
    first = await module(PipelineState(
        input_question="q",
        literature_results=[LiteratureResult(
            sub_question="something else",
            papers_retrieved=1,
            knowledge_entries=[entry],
        )],
    ))
    graph = first["evidence_graph"]

    state = PipelineState(
        input_question="q",
        literature_results=[LiteratureResult(
            sub_question="something else",
            papers_retrieved=1,
            knowledge_entries=[entry],
        )],
        evidence_graph=graph,
        evidence_gaps=[_pending_gap("still unmatched target")],
    )
    result = await module(state)
    updated = result["evidence_gaps"]
    assert updated[0].attempts == 1
    assert updated[0].status == "pending_grounding"


@pytest.mark.asyncio
async def test_no_pending_gaps_means_no_gap_patch():
    entry = _entry("KE_x", "Some content about mTOR signalling.")
    module = M3EvidenceGraph(mode="rule")
    result = await module(PipelineState(
        input_question="q",
        literature_results=[LiteratureResult(
            sub_question="sq", papers_retrieved=1, knowledge_entries=[entry],
        )],
        evidence_gaps=[EvidenceGap(
            description="an open gap", target_sub_question="sq", status="open",
        )],
    ))
    assert "evidence_gaps" not in result


# ---------------------------------------------------------------------------
# Stable KnowledgeEntry ids
# ---------------------------------------------------------------------------


def test_stable_entry_id_is_content_based():
    a = stable_entry_id("Hsp70   folds proteins.", "PMID:1")
    b = stable_entry_id("hsp70 folds proteins.", "PMID:1")  # case/whitespace
    c = stable_entry_id("Hsp70 folds proteins.", "PMID:2")  # other paper
    assert a == b
    assert a != c
    assert a.startswith("KE_")
    assert len(a) == len("KE_") + 12


def test_stable_entry_id_matches_canonical_memory_helper():
    """M2's local helper and the canonical memory-layer helper must agree
    (same evidence → same id regardless of producer)."""
    assert stable_entry_id("  Hsp70 folds\tproteins. ", "PMID:7") == \
        stable_entry_id("Hsp70 folds proteins.", "PMID:7")


@pytest.mark.asyncio
async def test_m3_blank_entry_ids_get_stable_content_based_node_ids():
    """Entries with empty ids must get stable content-based node ids, so
    two separate runs (fresh module instances) produce identical graphs
    and incremental dedup still works."""
    def blank_entry() -> KnowledgeEntry:
        return KnowledgeEntry(
            id="",
            type=KnowledgeEntryType.ESTABLISHED_FACT,
            content="Hsp90 stabilises client kinases.",
            source_paper_id="PMID:42",
            entities=["hsp90"],
        )

    first = await M3EvidenceGraph(mode="rule")(PipelineState(
        input_question="q",
        literature_results=[LiteratureResult(
            sub_question="sq", papers_retrieved=1,
            knowledge_entries=[blank_entry()],
        )],
    ))
    second = await M3EvidenceGraph(mode="rule")(PipelineState(
        input_question="q",
        literature_results=[LiteratureResult(
            sub_question="sq", papers_retrieved=1,
            knowledge_entries=[blank_entry()],
        )],
    ))
    ids_first = sorted(n.id for n in first["evidence_graph"].nodes)
    ids_second = sorted(n.id for n in second["evidence_graph"].nodes)
    assert ids_first == ids_second  # deterministic across runs

    expected = "N_" + stable_entry_id(
        "Hsp90 stabilises client kinases.", "PMID:42"
    )
    assert expected in ids_first

    # Incremental dedup: feeding the same blank-id entry back together with
    # the existing graph adds nothing.
    third = await M3EvidenceGraph(mode="rule")(PipelineState(
        input_question="q",
        literature_results=[LiteratureResult(
            sub_question="sq", papers_retrieved=1,
            knowledge_entries=[blank_entry()],
        )],
        evidence_graph=first["evidence_graph"],
    ))
    assert len(third["evidence_graph"].nodes) == len(first["evidence_graph"].nodes)


@pytest.mark.asyncio
async def test_gap_gain_present_for_all_gaps_including_zero():
    """metrics['m3_gap_gain'] covers every gap, zeros included."""
    entry = _entry("KE_g1", "Evidence only for the first gap.")
    matched = _pending_gap("matched sub-question")
    unmatched = _pending_gap("never matched sub-question")

    module = M3EvidenceGraph(mode="rule", gap_no_improvement_limit=3)
    first = await module(PipelineState(
        input_question="q",
        literature_results=[LiteratureResult(
            sub_question="matched sub-question",
            papers_retrieved=1,
            knowledge_entries=[entry],
        )],
    ))
    result = await module(PipelineState(
        input_question="q",
        literature_results=[LiteratureResult(
            sub_question="matched sub-question",
            papers_retrieved=1,
            knowledge_entries=[entry],
        )],
        evidence_graph=first["evidence_graph"],
        evidence_gaps=[matched, unmatched],
    ))
    gain = result["metrics"]["m3_gap_gain"]
    assert set(gain) == {matched.gap_id, unmatched.gap_id}
    # entry already in graph → not new → zero gain for both gaps
    assert gain[matched.gap_id] == 0
    assert gain[unmatched.gap_id] == 0
