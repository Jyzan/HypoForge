import json
import subprocess
from pathlib import Path

from hypoforge.webapp import RunManager


def test_public_state_exposes_modern_m6_summary_and_legacy_runs_stay_compatible():
    summary = {
        "version": 1,
        "raw_score": 4.2,
        "final_score": 3.9,
        "dimensions": [],
        "applied_caps": [{
            "rule_id": "low_novelty",
            "maximum": 3.9,
            "reason": "novelty below threshold",
            "attribution": "hypothesis",
        }],
        "rationale": "display-only; routing uses structured verdicts",
        "complete": True,
    }
    payload = RunManager._public_state_payload(
        "run-1",
        {"input_question": "q", "m6_scoring_summary": summary},
        is_final=True,
    )

    assert payload["m6_scoring_summary"]["final_score"] == 3.9
    assert payload["m6_scoring_summary"]["applied_caps"][0]["rule_id"] == "low_novelty"

    legacy = RunManager._public_state_payload(
        "run-legacy",
        {"input_question": "q"},
        is_final=True,
    )
    assert legacy["m6_scoring_summary"] is None


def test_m6_radar_renderer_outputs_eight_axes_and_score_polygon():
    html = Path("hypoforge/web/index.html").read_text(encoding="utf-8")
    start = html.index("function m6RadarHtml")
    end = html.index("function m6ScoringSummaryHtml", start)
    renderer = html[start:end]
    dimensions = [
        {"dimension": name, "score": score}
        for name, score in zip(
            [
                "task_coverage",
                "novelty",
                "scientific_logic",
                "evidence_reliability",
                "testability",
                "experimental_rigor",
                "statistics_reproducibility",
                "technical_feasibility",
            ],
            [4.0, 2.0, 4.5, 3.5, 4.0, 3.0, 2.5, 4.0],
        )
    ]
    script = f"""
const DIM_LABELS = Object.fromEntries({json.dumps([row['dimension'] for row in dimensions])}.map(x => [x, x]));
const esc = value => String(value);
{renderer}
const output = m6RadarHtml({json.dumps(dimensions)});
process.stdout.write(output);
"""
    result = subprocess.run(
        ["node", "-e", script],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    svg = result.stdout
    assert 'aria-label="M6 八维评分雷达图"' in svg
    assert svg.count('class="m6-radar-axis"') == 8
    assert svg.count('class="m6-radar-label"') == 8
    assert 'class="m6-radar-score"' in svg
    assert "task_coverage 4.0 / 5" in svg
