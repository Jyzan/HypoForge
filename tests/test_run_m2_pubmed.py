from __future__ import annotations

import json

import pytest

from hypoforge.state import LiteratureResult
from scripts import run_m2_pubmed


class SuccessfulAdapter:
    async def __call__(self, state, config=None):
        return {"literature_results": [LiteratureResult(sub_question="Hippo通路")]}


class FailingAdapter:
    async def __call__(self, state, config=None):
        raise RuntimeError("network unavailable")


class LongMessageFailingAdapter:
    async def __call__(self, state, config=None):
        raise RuntimeError("network\n\t unavailable " + "x" * 600)


def test_cli_prints_utf8_success_json(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        run_m2_pubmed,
        "build_minimal_pubmed_adapter",
        lambda **kwargs: SuccessfulAdapter(),
    )

    exit_code = run_m2_pubmed.main(["--question", "Hippo通路"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload == {
        "status": "ok",
        "literature_results": [
            {
                "sub_question": "Hippo通路",
                "papers_retrieved": 0,
                "knowledge_entries": [],
            }
        ],
    }
    assert "Hippo通路" in captured.out
    assert "\\u" not in captured.out
    assert captured.err == ""


def test_cli_prints_error_json_and_returns_nonzero(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        run_m2_pubmed,
        "build_minimal_pubmed_adapter",
        lambda **kwargs: FailingAdapter(),
    )

    exit_code = run_m2_pubmed.main(["--question", "question"])

    captured = capsys.readouterr()
    payload = json.loads(captured.err)
    assert exit_code == 1
    assert payload["status"] == "error"
    assert payload["error_type"] == "RuntimeError"
    assert payload["message"] == "network unavailable"
    assert captured.out == ""


@pytest.mark.parametrize(
    ("argv", "message_fragment"),
    [
        ([], "--question"),
        (["--question", "question", "--limit", "many"], "invalid int value"),
    ],
)
def test_cli_argument_errors_are_json_and_return_one(
    argv, message_fragment, capsys
) -> None:
    exit_code = run_m2_pubmed.main(argv)

    captured = capsys.readouterr()
    payload = json.loads(captured.err)
    assert exit_code == 1
    assert payload["status"] == "error"
    assert payload["error_type"] == "CLIArgumentError"
    assert message_fragment in payload["message"]
    assert captured.out == ""


def test_cli_help_keeps_standard_success_exit(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        run_m2_pubmed.main(["--help"])

    captured = capsys.readouterr()
    assert exc_info.value.code == 0
    assert captured.out.startswith("usage:")
    assert captured.err == ""


def test_cli_sanitizes_and_truncates_error_message(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        run_m2_pubmed,
        "build_minimal_pubmed_adapter",
        lambda **kwargs: LongMessageFailingAdapter(),
    )

    exit_code = run_m2_pubmed.main(["--question", "question"])

    captured = capsys.readouterr()
    payload = json.loads(captured.err)
    assert exit_code == 1
    assert payload["message"].startswith("network unavailable ")
    assert len(payload["message"]) == 500
    assert "\n" not in payload["message"]
    assert "\t" not in payload["message"]
    assert captured.out == ""
