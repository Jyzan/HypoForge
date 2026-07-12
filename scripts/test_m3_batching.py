#!/usr/bin/env python
"""Small-batch smoke test for M3 relation extraction.

Default mode is offline and tests batching/partial-JSON recovery only.
Use ``--live`` to make a small number of real Qwen calls.

Examples:
    python scripts/test_m3_batching.py
    python scripts/test_m3_batching.py --live --batch-size 5 --limit 20
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hypoforge.config import PipelineConfig
from hypoforge.modules.m3_evidence_graph import M3EvidenceGraph
from hypoforge.state import KnowledgeEntry
from hypoforge.tools.qwen_client import QwenClient, _recover_partial_edges


def load_entries(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = []
    for result in data.get("literature_results", []):
        entries.extend(result.get("knowledge_entries", []))
    return entries


def offline_check() -> None:
    raw = (
        '{"edges": [{"source": "KE1", "target": "KE2", '
        '"relation": "supports"}, {"source": "KE3"'
    )
    recovered = _recover_partial_edges(raw)
    assert recovered and len(recovered["edges"]) == 1
    print("offline: partial JSON recovery passed")


async def live_check(args: argparse.Namespace) -> None:
    config = PipelineConfig.from_yaml(args.config)
    llm_config = getattr(config.qwen, args.tier)
    if not llm_config.api_key:
        raise RuntimeError("OPENAI_API_KEY is empty; use --live only with API credentials configured")

    entries = load_entries(Path(args.input))[: args.limit]
    if not entries:
        raise RuntimeError(f"No literature entries found in {args.input}")

    # Reuse M3's production schema and prompt construction, but call only the
    # requested small batches and never persist a graph.
    module = M3EvidenceGraph(
        mode="llm",
        llm_config=llm_config,
        relation_batch_size=args.batch_size,
    )
    model = module.client.model if module.client else "unknown"
    QwenClient.reset_token_totals()

    batches = [
        entries[i:i + args.batch_size]
        for i in range(0, len(entries), args.batch_size)
    ]
    print(f"live: model={model}, batches={len(batches)}, batch_size={args.batch_size}")

    for index, batch in enumerate(batches, 1):
        before_in, before_out = QwenClient.get_token_totals()
        try:
            result = await module._extract_relation_batch(
                [KnowledgeEntry.model_validate(entry) for entry in batch]
            )
            status = "partial-recovered" if result.get("_partial_json") else "valid"
            edge_count = len(result.get("edges", []))
        except Exception as exc:
            status = f"error: {exc}"
            edge_count = 0
        after_in, after_out = QwenClient.get_token_totals()
        print(
            f"batch {index}/{len(batches)}: {status}, edges={edge_count}, "
            f"input_tokens={after_in - before_in}, output_tokens={after_out - before_out}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test M3 relation extraction batches.")
    parser.add_argument("--live", action="store_true", help="Make real Qwen calls")
    parser.add_argument("--input", default="output/20260712_checkpoint.json", help="Checkpoint JSON input")
    parser.add_argument("--config", default="configs/default.yaml", help="Pipeline YAML config")
    parser.add_argument("--tier", choices=["base", "max", "plus", "turbo"], default="plus")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    offline_check()
    if args.live:
        asyncio.run(live_check(args))
    else:
        print("offline: no API calls made")


if __name__ == "__main__":
    main()
