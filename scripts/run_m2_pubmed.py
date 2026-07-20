from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hypoforge.literature.minimal import build_minimal_pubmed_adapter
from hypoforge.modules.m2_literature_search import M2LiteratureSearch
from hypoforge.state import PipelineState


class CLIArgumentError(ValueError):
    """Raised when CLI arguments cannot be parsed."""


class JSONArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise CLIArgumentError(message)


async def _run(question: str, limit: int, timeout: float) -> dict:
    adapter = build_minimal_pubmed_adapter(
        final_k=limit,
        source_timeout_seconds=timeout,
    )
    module = M2LiteratureSearch(
        implementation="agentic",
        agentic_adapter=adapter,
    )
    output = await module(PipelineState(input_question=question))
    return {
        "status": "ok",
        "literature_results": [
            item.model_dump(mode="json")
            for item in output["literature_results"]
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = JSONArgumentParser(description="Run minimal PubMed-only M2")
    parser.add_argument("--question", required=True)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=30.0)
    try:
        args = parser.parse_args(argv)
        payload = asyncio.run(_run(args.question, args.limit, args.timeout))
    except Exception as exc:
        error = {
            "status": "error",
            "error_type": type(exc).__name__,
            "message": " ".join(str(exc).split())[:500],
        }
        print(json.dumps(error, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
