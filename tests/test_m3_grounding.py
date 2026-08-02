"""Tests for M3 GroundingWorkflow — Track B refactored data flow.

Verifies that the grounding workflow correctly consumes M2KnowledgeExport
and does NOT attempt to download or parse papers.
"""

import pytest

from hypoforge.modules.m3_grounding.workflow import GroundingWorkflow
from hypoforge.modules.m3_grounding.models import (
    AtomicClaim,
    EvidenceRecord,
    GroundingReport,
    RelationCandidate,
)
from hypoforge.modules.m3_evidence_graph import (
    M3EvidenceGraph,
    add_entity_synonym,
    evidence_graph_to_mermaid,
    normalize_entity,
)
from hypoforge.state import (
    EvidenceEdge,
    EvidenceEdgeRelation,
    EvidenceGraph,
    EvidenceNode,
    EvidenceNodeType,
    M2EvidenceExport,
    M2KnowledgeExport,
    M2KnowledgeRun,
    M2PaperExport,
    PipelineState,
    ProblemCard,
)


def _make_m2_export(
    evidence_items: list[tuple[str, str, str, str]],
) -> M2KnowledgeExport:
    """Build a minimal M2KnowledgeExport with given evidence items.

    Each tuple: (evidence_id, paper_id, quote, normalized_claim)
    """
    paper_ids = sorted({pid for (_, pid, _, _) in evidence_items})
    return M2KnowledgeExport(
        runs=[
            M2KnowledgeRun(
                sub_question="test question",
                papers=[
                    M2PaperExport(paper_id=pid, title=f"Paper {pid}")
                    for pid in paper_ids
                ],
                evidence=[
                    M2EvidenceExport(
                        evidence_id=eid,
                        paper_id=pid,
                        chunk_id=f"CHK_{eid}",
                        quote=quote,
                        normalized_claim=nclaim,
                        relevance_score=0.9,
                        citable=True,
                    )
                    for (eid, pid, quote, nclaim) in evidence_items
                ],
            )
        ]
    )


def test_collect_m2_evidence_filters_citable_only():
    """Evidence items with citable=False should be excluded."""
    from hypoforge.state import M2PaperExport as Paper
    export = M2KnowledgeExport(
        runs=[
            M2KnowledgeRun(
                sub_question="q",
                papers=[
                    Paper(paper_id="P1", title="Test Paper"),
                ],
                evidence=[
                    M2EvidenceExport(
                        evidence_id="E1", paper_id="P1",
                        chunk_id="C1", quote="good", normalized_claim="good",
                        relevance_score=0.8, citable=True,
                    ),
                    M2EvidenceExport(
                        evidence_id="E2", paper_id="P1",
                        chunk_id="C2", quote="bad", normalized_claim="bad",
                        relevance_score=0.5, citable=False,
                    ),
                ],
            )
        ]
    )
    state = PipelineState(
        input_question="test",
        m2_knowledge_export=export,
    )
    wf = GroundingWorkflow(mode="rule")
    gs = {"pipeline_state": state, "report": GroundingReport()}

    import asyncio
    result = asyncio.run(wf._collect_m2_evidence(gs))  # type: ignore[arg-type]

    items = result["m2_evidence_items"]
    assert len(items) == 1
    assert items[0].evidence_id == "E1"


def test_collect_m2_evidence_warns_on_empty_export():
    """When m2_knowledge_export is None, should warn and return empty."""
    state = PipelineState(input_question="test")
    wf = GroundingWorkflow(mode="rule")
    gs = {"pipeline_state": state, "report": GroundingReport()}

    import asyncio
    result = asyncio.run(wf._collect_m2_evidence(gs))  # type: ignore[arg-type]

    assert result["m2_evidence_items"] == []
    assert any("empty" in w.lower() or "none" in w.lower() for w in result["report"].warnings)


def test_plan_queries_harvests_m2_knowledge_gaps():
    """Gaps and conflicts from M2KnowledgeExport should become queries."""
    from hypoforge.state import KnowledgeEntry, KnowledgeEntryType, M2PaperExport as Paper
    export = M2KnowledgeExport(
        runs=[
            M2KnowledgeRun(
                sub_question="q",
                papers=[
                    Paper(paper_id="P1", title="Test Paper"),
                ],
                knowledge_entries=[
                    KnowledgeEntry(
                        id="GAP1",
                        type=KnowledgeEntryType.KNOWLEDGE_GAP,
                        content="Unknown interaction between X and Y remains to be explored.",
                        source_paper_id="P1",
                        evidence_ids=["E1"],
                    ),
                ],
                evidence=[
                    M2EvidenceExport(
                        evidence_id="E1", paper_id="P1",
                        chunk_id="C1", quote="X interacts with Y is unknown.",
                        normalized_claim="X-Y interaction unknown",
                        relevance_score=0.8,
                    ),
                ],
            )
        ]
    )
    state = PipelineState(
        input_question="What is the role of X?",
        m2_knowledge_export=export,
        problem_card=ProblemCard(
            original_question="What is the role of X?",
            key_entities=["X", "Y"],
        ),
    )
    wf = GroundingWorkflow(mode="rule")
    gs = {"pipeline_state": state, "report": GroundingReport()}

    import asyncio
    result = asyncio.run(wf._plan_queries(gs))  # type: ignore[arg-type]

    queries = result["queries"]
    # Should include the gap content
    gap_queries = [q for q in queries if "Unknown interaction" in q]
    assert len(gap_queries) >= 1


def test_evidence_record_has_evidence_id():
    """EvidenceRecord should bridge to M2EvidenceExport via evidence_id."""
    record = EvidenceRecord(
        id="ER_test",
        evidence_id="E1",
        paper_id="P1",
        query="test",
        quote="Hsp70 binds ATP.",
        normalized_claim="Hsp70 has ATP binding activity.",
        relevance_score=7.5,
    )
    assert record.evidence_id == "E1"
    assert record.paper_id == "P1"


def test_relation_candidate_to_relation():
    """RelationCandidate.to_relation() should produce a canonical EvidenceRelation."""
    candidate = RelationCandidate(
        id="REL_1",
        source="C1",
        target="C2",
        relation="supports",
        confidence=0.85,
        rationale="Both point to the same mechanism.",
        evidence_ids=["E1", "E2"],
        source_paper_ids=["P1"],
        target_paper_ids=["P2"],
        retrieval_score=0.75,
        condition_comparability=0.9,
        candidate_origin=["shared_entity"],
    )
    rel = candidate.to_relation()
    assert rel.id == "REL_1"
    assert rel.source == "C1"
    assert rel.target == "C2"
    assert rel.relation == "supports"
    assert rel.confidence == 0.85
    assert rel.evidence_ids == ["E1", "E2"]


def test_unrelated_cannot_become_relation():
    """Candidates marked 'unrelated' should raise when converting to relation."""
    candidate = RelationCandidate(
        id="REL_BAD",
        source="C1",
        target="C2",
        relation="unrelated",
        confidence=0.1,
        rationale="No connection.",
    )
    with pytest.raises(ValueError, match="unrelated"):
        candidate.to_relation()


# ============================================================================
# Entity normalisation tests
# ============================================================================


def test_normalize_entity_maps_synonyms():
    """Known variants should map to the same canonical form."""
    assert normalize_entity("Hsp70") == normalize_entity("HSP70")
    assert normalize_entity("HSPA1A") == normalize_entity("hsp70")
    assert normalize_entity("heat shock protein 70") == "hsp70"


def test_normalize_entity_falls_back_to_cleaned():
    """Unknown names should be lowercased and stripped but not lost."""
    result = normalize_entity("  Unknown-Protein-X  ")
    assert result == "unknown-protein-x"


def test_normalize_entity_strips_punctuation():
    """Trailing commas, periods should be removed."""
    assert normalize_entity("Hsp70,") == "hsp70"
    assert normalize_entity("(p53)") == "p53"


def test_add_entity_synonym_runtime():
    """Runtime-registered synonyms should be picked up immediately."""
    add_entity_synonym("MyNewProtein", "MNP")
    assert normalize_entity("MNP") == "mynewprotein"
    assert normalize_entity("MyNewProtein") == "mynewprotein"


# ============================================================================
# Mermaid export tests
# ============================================================================


def test_evidence_graph_to_mermaid_produces_valid_structure():
    """Mermaid output should have nodes, edges, and title."""
    graph = EvidenceGraph(
        nodes=[
            EvidenceNode(id="N1", type=EvidenceNodeType.CLAIM, label="Claim A"),
            EvidenceNode(id="N2", type=EvidenceNodeType.EVIDENCE, label="Evidence B"),
            EvidenceNode(id="SRC_P1", type=EvidenceNodeType.SOURCE, label="Paper 1"),
        ],
        edges=[
            EvidenceEdge(
                source="N2", target="N1",
                relation=EvidenceEdgeRelation.SUPPORTS,
                confidence=0.88, rationale="Strong evidence.",
            ),
            EvidenceEdge(
                source="SRC_P1", target="N2",
                relation=EvidenceEdgeRelation.INVOLVES,
            ),
        ],
    )
    mermaid = evidence_graph_to_mermaid(graph)
    assert "flowchart LR" in mermaid
    assert "N1" in mermaid
    assert "N2" in mermaid
    assert "supports" in mermaid
    assert "involves" in mermaid
    assert "c=0.88" in mermaid


def test_evidence_graph_to_mermaid_escapes_quotes():
    """Double-quotes in labels should be turned into single quotes."""
    graph = EvidenceGraph(
        nodes=[
            EvidenceNode(
                id="N1", type=EvidenceNodeType.CLAIM,
                label='The "key" finding',
            ),
        ],
    )
    mermaid = evidence_graph_to_mermaid(graph)
    # The label's double-quotes become single quotes in the mermaid output
    assert "\"key\"" not in mermaid
    assert "'key'" in mermaid


# ============================================================================
# Incremental update tests
# ============================================================================


def test_find_new_entries_filters_existing():
    """_find_new_entries should return only entries not already in the graph."""
    from hypoforge.state import KnowledgeEntry, KnowledgeEntryType, ConfidenceLevel

    existing = EvidenceGraph(
        nodes=[
            EvidenceNode(
                id="N_K1", type=EvidenceNodeType.CLAIM,
                label="Entry 1", metadata={"entry_type": "established_fact"},
            ),
            EvidenceNode(
                id="N_K2", type=EvidenceNodeType.EVIDENCE,
                label="Entry 2", metadata={"entry_type": "established_fact"},
            ),
        ],
    )
    entries = [
        KnowledgeEntry(id="K1", type=KnowledgeEntryType.ESTABLISHED_FACT, content="old"),
        KnowledgeEntry(id="K2", type=KnowledgeEntryType.ESTABLISHED_FACT, content="old"),
        KnowledgeEntry(id="K3", type=KnowledgeEntryType.ESTABLISHED_FACT, content="new"),
    ]
    new_entries = M3EvidenceGraph._find_new_entries(entries, existing)
    assert len(new_entries) == 1
    assert new_entries[0].id == "K3"


def test_merge_graphs_deduplicates():
    """Merging two graphs should not create duplicate nodes or edges."""
    base = EvidenceGraph(
        nodes=[
            EvidenceNode(id="N1", type=EvidenceNodeType.CLAIM, label="A"),
        ],
        edges=[
            EvidenceEdge(source="SRC_P1", target="N1", relation=EvidenceEdgeRelation.INVOLVES),
        ],
        established_facts=["K1"],
    )
    additions = EvidenceGraph(
        nodes=[
            EvidenceNode(id="N1", type=EvidenceNodeType.CLAIM, label="A"),  # duplicate
            EvidenceNode(id="N2", type=EvidenceNodeType.CLAIM, label="B"),  # new
        ],
        edges=[
            EvidenceEdge(source="SRC_P1", target="N1", relation=EvidenceEdgeRelation.INVOLVES),  # dup
            EvidenceEdge(source="N2", target="N1", relation=EvidenceEdgeRelation.EXTENDS),  # new
        ],
        established_facts=["K1", "K2"],
    )
    merged = M3EvidenceGraph._merge_graphs(base, additions)
    # 2 unique nodes (N1, N2)
    assert len(merged.nodes) == 2
    # 2 unique edges
    assert len(merged.edges) == 2
    # 2 unique facts (K1, K2), no duplicates
    assert len(merged.established_facts) == 2
    assert set(merged.established_facts) == {"K1", "K2"}
