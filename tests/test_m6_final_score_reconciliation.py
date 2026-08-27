from __future__ import annotations

from hypoforge.webapp import RunManager


def _legacy_state() -> dict:
    return {
        "input_question": "How can RAG reduce hallucinations?",
        "iteration_count": 2,
        "reviews": [
            {
                "dimension": "scientific_logic",
                "score": 4.0,
                "hard_gate_passed": True,
                "reasoning": "live specialist review",
                "version": 2,
            },
            {
                "dimension": "method_feasibility",
                "score": 4.0,
                "hard_gate_passed": True,
                "reasoning": "live specialist review",
                "version": 2,
            },
            {
                "dimension": "objective_evidence_consistency",
                "score": 0.0,
                "hard_gate_passed": False,
                "reasoning": (
                    "Calculated evidence_consistency score is 0.00 "
                    "(scaled to 0.0/5)."
                ),
                "version": 2,
            },
            {
                "dimension": "novelty_metric",
                "score": 0.0,
                "hard_gate_passed": False,
                "reasoning": "Calculated novelty score is 0.00 (scaled to 0.0/5).",
                "version": 2,
            },
            {
                "dimension": "testability_metric",
                "score": 5.0,
                "hard_gate_passed": True,
                "reasoning": "Calculated testability score is 1.00 (scaled to 5.0/5).",
                "version": 2,
            },
            {
                "dimension": "overall",
                "score": 2.9,
                "hard_gate_passed": False,
                "reasoning": (
                    "Computed from specialist reviews; hard-gate failures: "
                    "objective_evidence_consistency, novelty_metric"
                ),
                "version": 2,
            },
        ],
    }


def _review(payload: dict, dimension: str) -> dict:
    return next(
        item
        for item in payload["reviews"]
        if item["version"] == 2 and item["dimension"] == dimension
    )


def test_final_payload_replaces_legacy_default_zeroes_with_real_scores() -> None:
    scores = {
        "hypothesis_scores": [
            {
                "hypothesis_id": "H3",
                "independent": {
                    "evidence_consistency": 0.5,
                    "novelty": 0.7,
                    "testability": 1.0,
                },
            }
        ]
    }

    payload = RunManager._public_state_payload(
        "ui-test",
        _legacy_state(),
        scores=scores,
        is_final=True,
    )

    evidence = _review(payload, "objective_evidence_consistency")
    novelty = _review(payload, "novelty_metric")
    overall = _review(payload, "overall")
    assert evidence["score"] == 2.5
    assert evidence["superseded_live_score"] == 0.0
    assert evidence["score_source"] == "posthoc_independent"
    assert evidence["hard_gate_passed"] is False
    assert "最终独立评分已替代" in evidence["comments"]
    assert novelty["score"] == 3.5
    assert novelty["hard_gate_passed"] is True
    assert "novelty_metric" not in overall["reasoning"]
    assert payload["m6_score_reconciliation"]["status"] == "final"


def test_payload_marks_uncomputed_legacy_metrics_unavailable_not_failed() -> None:
    payload = RunManager._public_state_payload(
        "ui-test",
        _legacy_state(),
        scores=None,
        is_final=False,
    )

    evidence = _review(payload, "objective_evidence_consistency")
    novelty = _review(payload, "novelty_metric")
    overall = _review(payload, "overall")
    assert evidence["score"] is None
    assert evidence["hard_gate_passed"] is None
    assert evidence["score_status"] == "awaiting_posthoc"
    assert "不计入硬门" in evidence["comments"]
    assert novelty["score"] is None
    assert novelty["hard_gate_passed"] is None
    assert overall["hard_gate_passed"] is None
    assert payload["m6_score_reconciliation"]["status"] == "pending"
