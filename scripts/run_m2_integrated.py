from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Sequence, TextIO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hypoforge.literature.adapter import AgenticM2Adapter
from hypoforge.literature.integrated import build_integrated_search_adapter
from hypoforge.state import PipelineState
from hypoforge.tools.qwen_client import QwenClient


class CLIArgumentError(ValueError):
    """Raised when CLI arguments cannot be parsed."""


class JSONArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise CLIArgumentError(message)


def _emit_json(payload: Any, *, stream: TextIO, indent: int | None = None) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=indent)
    encoding = getattr(stream, "encoding", None)
    if encoding:
        try:
            text.encode(encoding, errors="strict")
        except (LookupError, UnicodeEncodeError):
            text = json.dumps(payload, ensure_ascii=True, indent=indent)
    stream.write(text + "\n")


class RecordingSearchAgent:
    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate
        self.results: list[Any] = []

    async def run(self, *args: Any, **kwargs: Any) -> Any:
        result = await self.delegate.run(*args, **kwargs)
        self.results.append(result)
        return result


class RecordingReadingWorkflow:
    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate
        self.results: list[Any] = []

    async def run(self, *args: Any, **kwargs: Any) -> Any:
        result = await self.delegate.run(*args, **kwargs)
        self.results.extend(result)
        return result


async def _run(
    *, question: str, model: str, limit: int, timeout: float
) -> dict:
    client = QwenClient(model=model)
    adapter = build_integrated_search_adapter(
        client=client,
        final_k=limit,
        source_timeout_seconds=timeout,
    )
    search = RecordingSearchAgent(adapter.search_agent)
    reading = RecordingReadingWorkflow(adapter.reading_workflow)
    traced = AgenticM2Adapter(
        search_agent=search,
        reading_workflow=reading,
        budget=adapter.budget,
    )
    output = await traced(PipelineState(input_question=question))
    return {
        "status": "ok",
        "trace": {
            "search_runs": [
                item.model_dump(mode="json") for item in search.results
            ],
            "reading_results": [
                item.model_dump(mode="json") for item in reading.results
            ],
        },
        "literature_results": [
            item.model_dump(mode="json")
            for item in output["literature_results"]
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = JSONArgumentParser(description="Run integrated teammate M2 search")
    parser.add_argument("--question", required=True)
    parser.add_argument("--model", default="qwen3.6-plus")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=30.0)
    try:
        args = parser.parse_args(argv)
        payload = asyncio.run(
            _run(
                question=args.question,
                model=args.model,
                limit=args.limit,
                timeout=args.timeout,
            )
        )
    except Exception as exc:
        error = {
            "status": "error",
            "error_type": type(exc).__name__,
            "message": " ".join(str(exc).split())[:500],
        }
        _emit_json(error, stream=sys.stderr)
        return 1
    _emit_json(payload, stream=sys.stdout, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
