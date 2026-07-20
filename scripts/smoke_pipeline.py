#!/usr/bin/env python
"""
Fast end-to-end smoke test — runs the REAL M1→M6 pipeline with every stage
turned down to the minimum, so you can verify basic functionality quickly
without the full ``run_hypoforge.py`` runtime.

It still calls Qwen at every LLM stage (M1–M6); it just shrinks the workload:

    M1  → keep only 1 sub-question         (max_sub_questions=1)
    M2  → search 1 paper, batch size 1      (max_papers_per_query=1, batch_size=1)
    M3  → LLM edges, 1 small batch, no bridges (enable_cross_batch=False)
    M4  → 2 candidates, keep top 1          (num_candidates=2, top_k=1)
    M5  → 1 plan (top_k=1)
    M6  → 3 specialist reviews, NO iteration (enable_iteration=False)

Total ≈ 8–9 LLM calls instead of the dozens a full run makes.

Requires ``OPENAI_API_KEY`` (see ``.env_template``).

Usage::

    python scripts/smoke_pipeline.py
    python scripts/smoke_pipeline.py -q "衰老的生物学基础是什么？"
    python scripts/smoke_pipeline.py --m4-mode multi_agent   # also exercise the ranker
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hypoforge.config import ModuleOverride, PipelineConfig
from hypoforge.pipeline import PipelineRunner


def build_fast_config(base_config: str, m4_mode: str = "direct") -> PipelineConfig:
    """Load *base_config* and apply the fast-smoke simplification knobs.

    Kept separate from the run so the wiring can be inspected/tested without
    making any API calls.
    """
    config = PipelineConfig.from_yaml(base_config)

    # No iteration loop (the single biggest time saver) and no stdin prompts.
    config.enable_iteration = False
    config.max_iterations = 1
    config.interactive = False

    def _fast(name: str, **kw) -> None:
        override = config.module_overrides.get(name) or ModuleOverride()
        override.kwargs.update(kw)
        override.kwargs.setdefault("mode", "llm")  # every stage still calls Qwen
        config.module_overrides[name] = override

    _fast("m1", max_sub_questions=1)
    _fast(
        "m2",
        max_papers_per_query=1,
        batch_size=1,
        max_search_queries=1,
        # Chinese questions must first become English literature queries.
        # Disable hidden reasoning so it cannot consume the visible response,
        # keep a larger safety budget, and retry one transient empty response.
        query_max_tokens=2048,
        query_disable_thinking=True,
        query_max_attempts=2,
    )
    _fast("m3", enable_cross_batch=False, max_relation_entries=8, relation_max_tokens=4096)
    _fast("m4", mode=m4_mode, num_candidates=2, top_k=1)
    _fast("m5")
    _fast("m6")
    return config


async def _main() -> int:
    parser = argparse.ArgumentParser(description="HypoForge fast end-to-end smoke test (real Qwen).")
    parser.add_argument("-q", "--question", default="蛋白质如何折叠及错误折叠导致疾病的机制？")
    parser.add_argument("-c", "--config", default="configs/default.yaml")
    parser.add_argument("--run-id", default="smoke")
    parser.add_argument("--m4-mode", default="direct", choices=["direct", "multi_agent"],
                        help="'direct' = generator only (fastest); 'multi_agent' also runs the ranker.")
    args = parser.parse_args()

    config = build_fast_config(args.config, m4_mode=args.m4_mode)
    runner = PipelineRunner(config)

    started = time.perf_counter()
    state = await runner.run(question=args.question, run_id=args.run_id)
    elapsed = time.perf_counter() - started

    # ---- stage-by-stage assertions (only for enabled stages) ----
    enabled = set(config.enabled_modules)
    checks: list[tuple[str, bool]] = []
    if "m1" in enabled:
        checks.append(("M1 problem_card", state.problem_card is not None))
    if "m2" in enabled:
        entry_total = sum(len(r.knowledge_entries) for r in state.literature_results)
        checks.append(("M2 literature_results", len(state.literature_results) > 0
                       and entry_total > 0))
    if "m3" in enabled:
        checks.append(("M3 evidence_graph", state.evidence_graph is not None
                       and len(state.evidence_graph.nodes) > 0))
    if "m4" in enabled:
        checks.append(("M4 top_hypotheses", len(state.top_hypotheses) > 0))
    if "m5" in enabled:
        checks.append(("M5 research_plans", len(state.research_plans) > 0))
    if "m6" in enabled:
        checks.append(("M6 reviews", len(state.reviews) > 0))

    print("\n===== SMOKE RESULTS =====")
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if state.errors:
        print(f"\n  {len(state.errors)} pipeline error(s):")
        for err in state.errors:
            print(f"    - {err.splitlines()[0]}")

    all_ok = bool(checks) and all(ok for _, ok in checks) and not state.errors
    print(
        f"\nSMOKE {'PASS' if all_ok else 'FAIL'} in {elapsed:.1f}s — "
        f"{len(state.top_hypotheses)} hypotheses, {len(state.reviews)} reviews, "
        f"{len(state.errors)} errors"
    )
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
