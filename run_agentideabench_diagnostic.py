#!/usr/bin/env python
"""Replay standard HypoForge stages from a saved state, without benchmark export."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from hypoforge.benchmarks.agentideabench import configure, public_config
from hypoforge.state import PipelineState


def prepare_replay(config, payload: dict, *, after: str, output: Path):
    """Retain scientific state and gates, with one additional review pass."""
    state = PipelineState.model_validate(payload)
    if not state.problem_card or not state.evidence_graph or not state.evidence_graph.nodes:
        raise ValueError("Replay requires completed M1–M3 outputs")
    if state.errors or state.clarification_request:
        raise ValueError("Use the last successful stage snapshot, without pipeline errors or pending clarification")
    if after == "m4" and not state.top_hypotheses:
        raise ValueError("Replay after M4 requires accepted top hypotheses")
    replay_config = config.model_copy(deep=True)
    replay_config.output_dir = str(output)
    replay_config.memory_cache_dir = str(output / "knowledge_graph")
    replay_config.entity_cache_dir = str(output / "entity_cache")
    replay_config.max_iterations = state.iteration_count + 1
    replay_config.enable_iteration = False
    seed = state.model_dump(mode="json")
    seed.update(
        _last_module=after,
        max_iterations=replay_config.max_iterations,
        memory_cache_dir=replay_config.memory_cache_dir,
        entity_cache_dir=replay_config.entity_cache_dir,
    )
    return replay_config, seed


async def replay(config, source: Path, *, after: str, output: Path, llm_concurrency: int):
    from hypoforge.observability import RunEventRecorder
    from hypoforge.pipeline import PipelineRunner
    from hypoforge.tools.qwen_client import limit_llm_concurrency

    source_bytes = source.read_bytes()
    config, seed = prepare_replay(config, json.loads(source_bytes), after=after, output=output)
    # A fresh directory keeps previous states and reports intact.
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "kind": "diagnostic", "benchmark_sample": False,
        "source_state": str(source.resolve()),
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "resume_after": after, "additional_review_passes": 1,
        "config": public_config(config), "status": "running",
    }
    report_path = output / "diagnostic_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    runner = PipelineRunner(config)
    run_id = f"diagnostic-after-{after}"
    runner.event_recorder = RunEventRecorder(output / "telemetry", run_id)
    try:
        with limit_llm_concurrency(llm_concurrency):
            state = await runner.run(seed["input_question"], run_id=run_id, seed_state=seed)
        complete = bool(state.research_plans and state.reviews and not state.errors and not state.clarification_request)
        report.update(
            status="completed" if complete else "incomplete",
            state_path=str(output / f"{run_id}.json"),
            plans=len(state.research_plans), reviews=len(state.reviews), errors=state.errors,
            pending_clarification=bool(state.clarification_request),
        )
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(json.dumps({key: value for key, value in report.items() if key != "config"}, ensure_ascii=False, indent=2))
    return complete


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--after", choices=("m3", "m4"), required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/agentideabench_siliconflow.yaml"))
    parser.add_argument("--api-key-file", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--llm-concurrency", type=int, default=4)
    args = parser.parse_args()
    if args.llm_concurrency < 1:
        parser.error("--llm-concurrency must be positive")
    try:
        config = configure(args.config, None, api_key_file=args.api_key_file)
        return 0 if asyncio.run(replay(config, args.state, after=args.after,
                                     output=args.output_dir, llm_concurrency=args.llm_concurrency)) else 1
    except KeyboardInterrupt:
        print("Diagnostic interrupted; completed stages remain in the output directory")
        return 130
    except Exception as exc:
        print(f"Diagnostic failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
