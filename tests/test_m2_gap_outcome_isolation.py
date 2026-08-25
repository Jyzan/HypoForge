from hypoforge.modules.m2_literature.adapter import AgenticM2Adapter
from hypoforge.state import EvidenceGapRequest, PipelineState


def test_each_m4_premise_gap_gets_an_isolated_search_task():
    adapter = object.__new__(AgenticM2Adapter)
    state = PipelineState(input_question="A affects B")
    gaps = [
        EvidenceGapRequest(
            gap_id="G1",
            premise_id="P1",
            hypothesis_id="H1",
            sub_question="What evidence tests A is present?",
            audit_claim="A is present",
        ),
        EvidenceGapRequest(
            gap_id="G2",
            premise_id="P2",
            hypothesis_id="H1",
            sub_question="What evidence tests A is present?",
            audit_claim="A is present",
        ),
    ]
    tasks = adapter._build_search_tasks(state, pending_gaps=gaps, revisit_gaps=[])
    assert len(tasks) == 2
    assert {task.m4_gap_ids for task in tasks} == {("G1",), ("G2",)}


def test_gap_search_records_completion_and_technical_error_separately():
    gap = EvidenceGapRequest(
        gap_id="G1",
        premise_id="P1",
        sub_question="Does A affect B?",
    )
    searched = AgenticM2Adapter._update_m4_gap_after_search(
        gap, completed=True, executed_queries=["A B"]
    )
    assert searched.search_completed is True
    assert searched.scientific_resolution == "unreviewed"
    assert searched.status == "searched"

    failed = AgenticM2Adapter._update_m4_gap_after_search(
        gap, completed=False, error="network timeout"
    )
    assert failed.search_completed is False
    assert failed.technical_errors == ["network timeout"]
    assert failed.scientific_resolution == "unreviewed"
