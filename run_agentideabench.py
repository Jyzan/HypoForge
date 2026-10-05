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
    generation_keys = parser.add_mutually_exclusive_group()
    generation_keys.add_argument("--api-key-csv", help="Model Studio export; read at runtime, never copied")
    generation_keys.add_argument("--api-key-file", help="JSON/text API key and base URL; read at runtime")
    embedding_keys = parser.add_mutually_exclusive_group()
    embedding_keys.add_argument("--embedding-api-key-csv", help="Optional separate embedding credential export")
    embedding_keys.add_argument("--embedding-api-key-file", help="Optional separate embedding text/JSON credentials")
    parser.add_argument("--embedding-model", help="Explicit embedding model override; changes the experiment fingerprint")
    critic_keys = parser.add_mutually_exclusive_group()
    critic_keys.add_argument("--critic-api-key-csv", help="Separate credentials for the original three critics")
    critic_keys.add_argument("--critic-api-key-file", help="Separate text/JSON credentials for the original three critics")
    parser.add_argument("--check-scope", choices=("all", "generation"), default="all",
                        help="Generation checks GLM, embeddings and retrieval; all also checks the three critics")
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
            config = bench.configure(args.config, args.api_key_csv, api_key_file=args.api_key_file,
                                     embedding_csv=args.embedding_api_key_csv,
                                     embedding_file=args.embedding_api_key_file,
                                     embedding_model=args.embedding_model)
            if not config.qwen.base.api_key:
                raise ValueError("Provide --api-key-csv / --api-key-file or configure OPENAI_API_KEY / QWEN_API_KEY")
            critic_config = config.qwen.base.model_copy(deep=True)
            if args.critic_api_key_csv or args.critic_api_key_file:
                from hypoforge.tools.credentials import read_api_key_csv, read_api_key_file
                critic_config.api_key, critic_config.api_base = (
                    read_api_key_csv(args.critic_api_key_csv) if args.critic_api_key_csv
                    else read_api_key_file(args.critic_api_key_file))
                critic_config.model = "glm-5.1"
            elif (config.qwen.base.model == "Pro/zai-org/GLM-5.1"
                  and (args.phase == "score" or (args.phase == "check" and args.check_scope == "all"))):
                raise ValueError("SiliconFlow generation uses separate original critic credentials: "
                                 "provide --critic-api-key-csv / --critic-api-key-file, "
                                 "or check generation only with --check-scope generation")
            if args.phase == "check":
                from hypoforge.benchmarks.agentideabench_scoring import preflight
                complete = asyncio.run(preflight(config, critic_config=critic_config, scope=args.check_scope))
            elif args.phase == "generate":
                topics = bench.select_topics(root, args.sample, args.limit)
                complete = asyncio.run(bench.generate(output, root, config, topics, args.repeats,
                                                     args.retry_failed, args.workers, args.llm_concurrency))
            else:
                from hypoforge.benchmarks.agentideabench_scoring import score
                complete = score(output, root, critic_config, args.retry_failed, args.workers)
        return 0 if complete else 1
    except KeyboardInterrupt:
        print("Interrupted. Re-run the same command to continue; use --retry-failed for failed cells.")
        return 130
    except Exception as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
