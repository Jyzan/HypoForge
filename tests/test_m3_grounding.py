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
from hypoforge.state import (
    M2EvidenceExport,
    M2KnowledgeExport,
    M2KnowledgeRun,
    PipelineState,
    ProblemCard,
)


def _make_m2_export(
    evidence_items: list[tuple[str, str, str, str]],
) -> M2KnowledgeExport:
    """Build a minimal M2KnowledgeExport with given evidence items.

    Each tuple: (evidence_id, paper_id, quote, normalized_claim)
    """
    return M2KnowledgeExport(
        runs=[
            M2KnowledgeRun(
                sub_question="test question",
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
    export = M2KnowledgeExport(
        runs=[
            M2KnowledgeRun(
                sub_question="q",
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
    export = M2KnowledgeExport(
        runs=[
            M2KnowledgeRun(
                sub_question="q",
                knowledge_entries=[
                    # Manually create a minimal KnowledgeEntry with type gap
                    __import__("hypoforge.state", fromlist=["KnowledgeEntry", "KnowledgeEntryType"]).KnowledgeEntry(
                        id="GAP1",
                        type=__import__("hypoforge.state", fromlist=["KnowledgeEntryType"]).KnowledgeEntryType.KNOWLEDGE_GAP,
                        content="Unknown interaction between X and Y remains to be explored.",
                        source_paper_id="P1",
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
