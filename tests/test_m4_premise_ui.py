from pathlib import Path


UI = Path("hypoforge/web/index.html")


def test_m4_ui_exposes_premises_and_scientific_gap_resolution():
    html = UI.read_text(encoding="utf-8")
    assert "事实前提" in html
    assert "工作假设" in html
    assert "scientific_resolution" in html
    assert "contradicting_evidence_ids" in html


def test_m5_ui_exposes_bridge_validation_targets():
    html = UI.read_text(encoding="utf-8")
    assert "bridge_validations" in html
    assert "推演前提验证" in html


def test_m4_ui_prefers_final_audited_top_card_over_candidate_copy():
    html = UI.read_text(encoding="utf-8")
    merge_block = html[html.index("const allById = new Map();"):html.index("const hypotheses = [...allById.values()];")]
    assert "[...candidates, ...selected]" in merge_block
    assert "if(!allById.has(key))" not in merge_block
