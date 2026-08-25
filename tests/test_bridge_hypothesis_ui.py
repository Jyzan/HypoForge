from pathlib import Path


UI = Path("hypoforge/web/index.html")


def test_m4_event_contract_is_renderable():
    html = UI.read_text(encoding="utf-8")
    assert "audit_claims" in html
    assert "evidence_gap_requests" in html


def test_m3_hypothesis_node_has_distinct_ui_style_and_label():
    html = UI.read_text(encoding="utf-8")
    assert 'hypothesis:{zh:"待验证假设"' in html
    assert "verification_status" in html
    assert "待验证假设（非论文证据）" in html
