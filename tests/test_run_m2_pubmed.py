from __future__ import annotations

import json

from scripts import run_m2_pubmed


class SuccessfulAdapter:
    async def __call__(self, state, config=None):
        return {"literature_results": []}


class FailingAdapter:
    async def __call__(self, state, config=None):
        raise RuntimeError("network unavailable")


def test_cli_prints_utf8_success_json(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        run_m2_pubmed,
        "build_minimal_pubmed_adapter",
        lambda **kwargs: SuccessfulAdapter(),
    )

    exit_code = run_m2_pubmed.main(["--question", "Hippo通路"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload == {"status": "ok", "literature_results": []}


def test_cli_prints_error_json_and_returns_nonzero(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        run_m2_pubmed,
        "build_minimal_pubmed_adapter",
        lambda **kwargs: FailingAdapter(),
    )

    exit_code = run_m2_pubmed.main(["--question", "question"])

    payload = json.loads(capsys.readouterr().err)
    assert exit_code == 1
    assert payload["status"] == "error"
    assert payload["error_type"] == "RuntimeError"
    assert payload["message"] == "network unavailable"
