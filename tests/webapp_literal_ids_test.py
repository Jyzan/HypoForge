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
    # No browser storage: credentials must stay ephemeral.
    assert "localStorage" not in html
    assert "sessionStorage" not in html
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
