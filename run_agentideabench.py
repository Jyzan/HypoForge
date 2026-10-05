#!/usr/bin/env python
"""Run, independently score, and compare GLM-5.1 HypoForge experiments."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path


def main() -> int:
    repo = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("check", "generate", "score", "analyze"), default="generate")
    parser.add_argument("--agentideabench-root", type=Path, default=repo.parent / "AgentIdeaBench")
    parser.add_argument("--config", type=Path, default=repo / "configs/agentideabench_glm51.yaml")
    parser.add_argument("--api-key-csv", help="Model Studio export; read at runtime, never copied")
    parser.add_argument("--sample", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--repeats", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--limit", type=int, help="Limit subfields for a smoke test")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--retry-failed", action="store_true", help="Retry failed cells; successful cells are reused")
    parser.add_argument("--workers", type=int, choices=(1, 2, 3, 4), default=1,
                        help="Concurrent pipelines in generate; concurrent critics (up to 3) in score")
    parser.add_argument("--llm-concurrency", type=int, default=8,
                        help="Shared in-flight text request cap for generation (default: 8)")
    parser.add_argument("--dry-run", action="store_true", help="Print the task plan without API calls or credentials")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.llm_concurrency < 1:
        parser.error("--llm-concurrency must be positive")
    from hypoforge.benchmarks import agentideabench as bench
    root = args.agentideabench_root.expanduser().resolve()
    output = (args.output_dir or repo / f"output/agentideabench/glm51-{args.sample}").resolve()
    try:
        if args.dry_run:
            import json
            topics = bench.select_topics(root, args.sample, args.limit)
            print(json.dumps({"output_dir": str(output), "resource_setting": "native", "topics": topics,
                              "pipeline_runs": len(topics) * args.repeats,
                              "critic_calls_minimum": len(topics) * args.repeats * 3,
                              "generation_workers": args.workers,
                              "generation_llm_concurrency": args.llm_concurrency,
                              "generation_m2_concurrency": 1,
                              "scoring_critic_concurrency": min(args.workers, 3)}, ensure_ascii=False, indent=2))
            return 0
        if args.phase == "analyze":
            complete = bench.analyze(output, root)
        else:
            config = bench.configure(args.config, args.api_key_csv)
            if not config.qwen.base.api_key:
                raise ValueError("Provide --api-key-csv or configure OPENAI_API_KEY / QWEN_API_KEY")
            if args.phase == "check":
                from hypoforge.benchmarks.agentideabench_scoring import preflight
                complete = asyncio.run(preflight(config))
            elif args.phase == "generate":
                topics = bench.select_topics(root, args.sample, args.limit)
                complete = asyncio.run(bench.generate(output, root, config, topics, args.repeats,
                                                     args.retry_failed, args.workers, args.llm_concurrency))
            else:
                from hypoforge.benchmarks.agentideabench_scoring import score
                complete = score(output, root, config.qwen.base, args.retry_failed, args.workers)
        return 0 if complete else 1
    except KeyboardInterrupt:
        print("Interrupted. Re-run the same command to continue; use --retry-failed for failed cells.")
        return 130
    except Exception as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
