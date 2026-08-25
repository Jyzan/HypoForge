import json
import re
import subprocess
from pathlib import Path

from hypoforge.pipeline import PipelineRunner, _route_after_m4
from hypoforge.state import EvidenceGapRequest, PipelineState, RoutingDecision


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_pending_m4_gap_is_recorded_as_auditable_m2_supplement_route():
    """Catch M4 revisits that execute in the graph but disappear from history."""

    runner = object.__new__(PipelineRunner)
    state = PipelineState(
        evidence_gap_requests=[
            EvidenceGapRequest(
                gap_id="m4-gap-1",
                hypothesis_id="H1",
                sub_question="Which evidence is missing?",
                status="pending",
            )
        ]
    )

    decision = runner._routing_decision("m4", state)

    assert decision is not None
    assert decision.from_module == "m4"
    assert decision.to_module == "supplement_m2"
    assert decision.decided_by == "m4"
    assert decision.gap_ids == ["m4-gap-1"]


def test_m4_gap_search_stops_after_two_completed_supplement_rounds():
    """Catch a fresh wording of an evidence gap bypassing the global cap."""

    state = PipelineState(
        evidence_gap_search_rounds=2,
        evidence_gap_requests=[
            EvidenceGapRequest(
                gap_id="newly-worded-gap",
                hypothesis_id="H2",
                sub_question="A newly worded unsupported claim",
                status="pending",
            )
        ],
    )

    assert _route_after_m4(state) == "continue"


def test_m4_gap_search_allows_the_second_supplement_round():
    """Catch an off-by-one cap that permits only one M4 supplement search."""

    state = PipelineState(
        evidence_gap_search_rounds=1,
        evidence_gap_requests=[
            EvidenceGapRequest(
                gap_id="second-round-gap",
                hypothesis_id="H2",
                sub_question="Evidence still missing after the first search",
                status="pending",
            )
        ],
    )

    assert _route_after_m4(state) == "search_gap"


def test_m4_gap_search_cap_does_not_record_a_false_supplement_route():
    """Catch the UI claiming M4 returned to M2 after the cap was reached."""

    runner = object.__new__(PipelineRunner)
    state = PipelineState(
        evidence_gap_search_rounds=2,
        evidence_gap_requests=[
            EvidenceGapRequest(
                gap_id="newly-worded-gap",
                hypothesis_id="H2",
                sub_question="A newly worded unsupported claim",
                status="pending",
            )
        ],
    )

    assert runner._routing_decision("m4", state) is None


def test_m4_gap_search_cap_counts_recorded_routes_when_searches_all_failed():
    """Catch two failed M2 attempts leaving the success counter at zero."""

    prior_routes = [
        RoutingDecision(
            round=round_number,
            from_module="m4",
            to_module="supplement_m2",
            decided_by="m4",
            reason="M4 evidence gap",
            gap_ids=[f"gap-{round_number}"],
        )
        for round_number in (1, 2)
    ]
    state = PipelineState(
        evidence_gap_search_rounds=0,
        routing_history=prior_routes,
        evidence_gap_requests=[
            EvidenceGapRequest(
                gap_id="third-gap",
                hypothesis_id="H3",
                sub_question="A third unsupported claim",
                status="pending",
            )
        ],
    )

    assert _route_after_m4(state) == "continue"


def test_frontend_classifies_legacy_and_standard_m4_revisits_as_supplements():
    """Catch an actual M4 revisit event being rendered as an unknown path."""

    html = (REPO_ROOT / "hypoforge" / "web" / "index.html").read_text(
        encoding="utf-8"
    )
    decision_map = re.search(
        r"const DECISION_TO_PATH\s*=\s*\{[^;]+\};", html
    )
    segment_classifier = re.search(
        r"function segmentPathType\(d\)\{.*?\n\}", html, re.DOTALL
    )
    assert decision_map is not None
    assert segment_classifier is not None

    script = f"""
{decision_map.group(0)}
{segment_classifier.group(0)}
const result = {{
  legacy: segmentPathType({{route: "m4->m2->m3->m4"}}),
  standard: segmentPathType({{from: "m4", to: "supplement_m2"}}),
  history: DECISION_TO_PATH["supplement_m2"]
}};
process.stdout.write(JSON.stringify(result));
"""
    completed = subprocess.run(
        ["node", "-e", script],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == {
        "legacy": "supplement",
        "standard": "supplement",
        "history": "supplement",
    }
