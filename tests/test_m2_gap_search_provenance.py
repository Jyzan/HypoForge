from hypoforge.modules.m2_literature.adapter import AgenticM2Adapter
from hypoforge.state import EvidenceGapRequest, PipelineState


def test_m4_gap_search_uses_audit_claim_and_preserves_origin_id():
    adapter = object.__new__(AgenticM2Adapter)
    gap = EvidenceGapRequest(
        gap_id="GAP_1",
        sub_question="What evidence tests A causes B through C?",
        audit_claim="A causes B through C.",
        status="pending",
    )
    tasks = adapter._build_search_tasks(
        PipelineState(input_question="A causes B"),
        pending_gaps=[gap],
        revisit_gaps=[],
    )
    assert tasks[0].question == "A causes B through C."
    assert tasks[0].m4_gap_ids == ("GAP_1",)


def test_search_task_with_zero_results_is_distinct_from_network_failure():
    gap = EvidenceGapRequest(
        gap_id="GAP_1",
        sub_question="A causes B",
        audit_claim="A causes B",
        status="pending",
    )
    searched = AgenticM2Adapter._update_m4_gap_after_search(
        gap, completed=True, executed_queries=["A causes B"],
    )
    failed = AgenticM2Adapter._update_m4_gap_after_search(
        gap, completed=False, error="network timeout",
    )
    assert searched.status == "searched"
    assert searched.attempts == 1
    assert failed.status == "pending"
    assert "network timeout" in failed.rationale
