#!/usr/bin/env python
"""
HypoForge CLI — entry point for running the AI Scientist pipeline.

Usage::

    # Fresh run
    python run_hypoforge.py \\
        --question "蛋白质如何折叠及错误折叠导致疾病的机制？" \\
        --config configs/full_pipeline.yaml

    # Resume from checkpoint (same --run-id as the interrupted run)
    python run_hypoforge.py \\
        --question "蛋白质如何折叠及错误折叠导致疾病的机制？" \\
        --config configs/full_pipeline.yaml \\
        --run-id hypoforge-abc12345 \\
        --resume

    python run_hypoforge.py \\
        --question "衰老的生物学基础是什么？" \\
        --config configs/baseline_b0.yaml \\
        --output-dir output/baseline_b0
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="HypoForge — AI Scientist: Hypothesis Generation & Research Plan Design",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--question", "-q",
        type=str,
        required=True,
        help="The frontier scientific question to analyse.",
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="./output",
        help="Directory for output files (overrides config value).",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="",
        help="Custom run identifier (auto-generated if omitted).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the last checkpoint for this --run-id.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress Rich terminal output.",
    )

    args = parser.parse_args()

    # ---- resolve config path ----
    config_path = Path(args.config)
    if not config_path.is_absolute():
        # Try relative to CWD, then relative to HypoForge root
        cwd_path = Path.cwd() / config_path
        repo_root = Path(__file__).resolve().parent.parent
        repo_path = repo_root / config_path
        if cwd_path.exists():
            config_path = cwd_path
        elif repo_path.exists():
            config_path = repo_path
        else:
            print(f"Error: config file not found at '{args.config}' (tried {cwd_path} and {repo_path})")
            sys.exit(1)

    # ---- load config ----
    from hypoforge.config import PipelineConfig

    try:
        config = PipelineConfig.from_yaml(str(config_path))
    except Exception as exc:
        print(f"Error loading config: {exc}")
        sys.exit(1)

    # ---- override output dir ----
    config.output_dir = args.output_dir

    # ---- override verbose ----
    if args.quiet:
        config.verbose = False

    # ---- run ----
    from hypoforge.pipeline import PipelineRunner

    runner = PipelineRunner(config)

    try:
        state = asyncio.run(runner.run(
            question=args.question,
            run_id=args.run_id,
            resume=args.resume,
        ))
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(130)
    except Exception as exc:
        print(f"\nPipeline failed: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ---- exit code reflects errors ----
    if state.errors:
        print(f"\nCompleted with {len(state.errors)} error(s).")
        sys.exit(1)
    else:
        print(f"\nPipeline completed successfully — {state.run_id}")
        sys.exit(0)


if __name__ == "__main__":
    main()
