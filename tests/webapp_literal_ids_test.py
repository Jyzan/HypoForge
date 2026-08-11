"""Guard the web UI contract: the six legacy literals plus the iteration
visualisation element IDs introduced by the reskin.

The rewrite of ``hypoforge/web/index.html`` must keep:

1. The six literals asserted by ``test_observability_webapp.py`` —
   ephemeral credential inputs (``modelName`` / ``qwenApiKey`` /
   ``semanticApiKey``), the "no browser persistence" rule
   (``localStorage`` must not appear), and the expanded-event-detail
   mechanism (``openSequences`` / ``data-event-sequence``).
2. The element IDs consumed by the new iteration-visualisation panels.
"""

from pathlib import Path
import re


def _read_index_html() -> str:
    html_path = (
        Path(__file__).resolve().parent.parent
        / "hypoforge"
        / "web"
        / "index.html"
    )
    return html_path.read_text(encoding="utf-8")


def test_preserves_six_legacy_literals() -> None:
    html = _read_index_html()

    # Ephemeral credential inputs (never persisted).
    assert 'id="modelName"' in html
    assert 'id="qwenApiKey"' in html
    assert 'id="semanticApiKey"' in html
    # Browser storage may keep harmless UI preferences, but never credentials.
    storage_writes = re.findall(
        r"(?:localStorage|sessionStorage)\.setItem\(\s*[\"']([^\"']+)[\"']",
        html,
    )
    assert set(storage_writes) <= {"hypoforge_order", "hypoforge_sidebar"}
    credential_terms = {
        "qwenApiKey",
        "semanticApiKey",
        "qwen_api_key",
        "semantic_scholar_api_key",
    }
    assert not credential_terms.intersection(storage_writes)
    # Expanded event <details> survive incremental re-renders.
    assert "openSequences" in html
    assert "data-event-sequence" in html


def test_iteration_visualisation_element_ids_present() -> None:
    html = _read_index_html()

    # Round timeline (consumes GET /api/runs/{id}/rounds routing_history).
    assert 'id="roundsCard"' in html
    assert 'id="roundsList"' in html
    # Iteration state panel: three counters + evidence gaps.
    assert 'id="counterIteration"' in html
    assert 'id="counterRevision"' in html
    assert 'id="counterSearchRound"' in html
    assert 'id="gapList"' in html
    # Plan version switching (consumes GET /api/runs/{id}/versions).
    assert 'id="planCard"' in html
    assert 'id="planVersions"' in html
    # Followup entry (POST /api/runs with parent_run_id + followup).
    assert 'id="followupBox"' in html
    assert 'id="followupInput"' in html
    assert 'id="followupSubmit"' in html
    assert 'id="followupHint"' in html
    # Preserved capabilities: error state / artifacts / event stream.
    assert 'id="runErrorBanner"' in html
    assert 'id="artifactsList"' in html
    assert 'id="eventsList"' in html


def test_m2_fulltext_failure_attribution_contract_present() -> None:
    html = _read_index_html()

    # The M2 detail cards must read the persisted attribution fields.
    assert "fulltext_failure_category" in html
    assert "fulltext_failure_detail" in html
    # Chinese label mapping for every attribution category.
    assert "FULLTEXT_FAILURE_LABELS" in html
    assert "出版商拦截" in html
    assert "PDF 链接无效" in html
    assert "该论文客观上无全文" in html


def test_m456_structured_detail_contract_present() -> None:
    """M4-M6 pipeline nodes expose their real structured outputs, not only logs."""
    html = _read_index_html()

    assert "function m4DetailsHtml" in html
    assert "function m5DetailsHtml" in html
    assert "function m6DetailsHtml" in html

    # M4: candidates, selected hypotheses, predictions, and traceable evidence.
    assert "candidate_hypotheses" in html
    assert "top_hypotheses" in html
    assert "observable_predictions" in html
    assert "falsification_conditions" in html

    # M5: executable plan content and its evidence links.
    assert "research_plans" in html
    assert "control_groups" in html
    assert "measurement_metrics" in html
    assert "evidence_links" in html

    # M6: per-version scores, review rationale, hard gates, and suggestions.
    assert "groupReviewsByVersion" in html
    assert "hard_gate_passed" in html
    assert "review.reasoning" in html
    assert "review.suggestions" in html

    # All structured drawers must refresh from the latest persisted checkpoint.
    assert 'const STRUCTURED_DRAWER_MODULES = new Set(["m1","m2","m4","m5","m6"])' in html


def test_m3_entity_merge_audit_contract_present() -> None:
    """M3 exposes enough audit data to detect a false entity merge."""
    html = _read_index_html()

    assert "function entityMergeAuditHtml" in html
    assert "实体合并记录" in html
    assert "entity_merge_log" in html
    assert "canonical_name" in html
    assert "similarity_to_canonical" in html
    assert "redirected_edge_count" in html
    assert "deduplicated_edge_count" in html
    assert "removed_self_loop_count" in html
