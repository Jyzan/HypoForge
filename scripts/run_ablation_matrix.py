import asyncio
import os
import sys
import json
import time
import shutil
import hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import csv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hypoforge.config import MasterEvaluationConfig, PipelineConfig
from hypoforge.pipeline import PipelineRunner
from hypoforge.state import PipelineState
from hypoforge.evaluation.scorer import score_pipeline_state_async

def get_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:8]

def get_m1_m3_cache_key(question: str, config: PipelineConfig) -> str:
    """Generate a cache key based on the question and M1-M3 related configs."""
    search_cfg = config.search.model_dump_json()
    grounding_cfg = config.grounding.model_dump_json()
    return f"cache_{get_hash(question + search_cfg + grounding_cfg)}"

async def ensure_m1_m3_cache(question: str, config: PipelineConfig, cache_run_id: str):
    """Run just M1-M3 to populate the cache if it doesn't exist."""
    checkpoint_path = Path(config.output_dir) / f"{cache_run_id}_checkpoint.json"
    if checkpoint_path.exists():
        return
        
    print(f"    [Cache Miss] Running M1-M3 for cache key: {cache_run_id}")
    # Create a temporary config that only runs up to M3
    temp_config = config.model_copy(deep=True)
    temp_config.enabled_modules = ["m1", "m2", "m3"]
    runner = PipelineRunner(temp_config)
    
    try:
        await runner.run(question, run_id=cache_run_id, resume=False)
    except Exception as e:
        print(f"    [Cache Error] Failed to generate M1-M3 cache: {e}")
        raise

async def main():
    print("🚀 Starting Ablation Matrix Execution...")
    
    # 1. Load Evaluation/Ablation Config
    eval_config_path = Path("configs/evaluation.yaml")
    try:
        master_cfg = MasterEvaluationConfig.from_yaml(eval_config_path)
    except Exception as e:
        print(f"❌ Failed to parse {eval_config_path}: {e}")
        sys.exit(1)
        
    ablation = master_cfg.ablation
    
    # Load pipeline configs from directory
    configs_dir = Path(ablation.configs_dir)
    if not configs_dir.exists():
        print(f"❌ Configs dir not found: {configs_dir}")
        sys.exit(1)
        
    yaml_files = list(configs_dir.glob("*.yaml"))
    if not yaml_files:
        print(f"❌ No yaml files found in {configs_dir}")
        sys.exit(1)
        
    # Cost tracking (simplified)
    # E.g. $0.004 per 1k input, $0.012 per 1k output
    QWEN_PLUS_INPUT_PRICE = 0.004 / 1000
    QWEN_PLUS_OUTPUT_PRICE = 0.012 / 1000
    total_cost_usd = 0.0
    
    # Output CSV setup
    csv_path = Path(ablation.output_csv)
    file_exists = csv_path.exists()
    
    headers = [
        "run_id", "config_name", "question", "repeat_index", 
        "top1_composite", "mean_composite", "mean_plan_completeness", 
        "latest_overall_review", "iterations", "errors", 
        "input_tokens", "output_tokens", "cost_usd", "elapsed_seconds", "failed_modules"
    ]
    
    # Read existing run_ids for resume capability
    existing_run_ids = set()
    if file_exists:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_run_ids.add(row["run_id"])
                total_cost_usd += float(row.get("cost_usd", 0.0))
                
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
            
        print(f"📊 Matrix: {len(yaml_files)} configs × {len(ablation.questions)} questions × {ablation.repeats} repeats")
        print(f"💰 Initial Budget Consumed: ${total_cost_usd:.4f} / ${ablation.max_budget_usd:.4f}")
        
        for config_path in yaml_files:
            config_name = config_path.stem
            print(f"\n📂 Loading config: {config_name}")
            try:
                pipeline_cfg = PipelineConfig.from_yaml(config_path)
            except Exception as e:
                print(f"  ⚠️ Skipping {config_name} due to config error: {e}")
                continue
                
            for q_idx, question in enumerate(ablation.questions):
                print(f"  📝 Question {q_idx+1}/{len(ablation.questions)}")
                
                # Pre-calculate M1-M3 cache to save money/time
                cache_run_id = get_m1_m3_cache_key(question, pipeline_cfg)
                cache_populated = False
                
                for repeat in range(1, ablation.repeats + 1):
                    run_id = f"{config_name}_q{q_idx}_r{repeat}_{get_hash(question)}"
                    
                    if run_id in existing_run_ids:
                        print(f"    ⏭️ Skipping {run_id} (already in CSV)")
                        continue
                        
                    if total_cost_usd >= ablation.max_budget_usd:
                        print(f"🛑 Budget exceeded (${total_cost_usd:.4f} >= ${ablation.max_budget_usd:.4f}). Stopping.")
                        return
                        
                    print(f"    ▶️ Running {run_id}...")
                    
                    # 1. Populate cache if needed
                    if not cache_populated:
                        try:
                            await ensure_m1_m3_cache(question, pipeline_cfg, cache_run_id)
                            cache_populated = True
                        except Exception:
                            print("    ❌ Cache generation failed. Skipping this question for this config.")
                            break
                            
                    # 2. Copy cache checkpoint to the actual run_id checkpoint
                    cache_ckpt = Path(pipeline_cfg.output_dir) / f"{cache_run_id}_checkpoint.json"
                    run_ckpt = Path(pipeline_cfg.output_dir) / f"{run_id}_checkpoint.json"
                    if cache_ckpt.exists():
                        shutil.copy(cache_ckpt, run_ckpt)
                    
                    # 3. Execute Pipeline
                    start_time = time.time()
                    runner = PipelineRunner(pipeline_cfg)
                    try:
                        final_state = await runner.run(question, run_id=run_id, resume=True)
                        failed_modules = []
                        if final_state.errors:
                            print(f"      ⚠️ Pipeline produced errors: {len(final_state.errors)}")
                            failed_modules = ["pipeline_error"]
                    except Exception as e:
                        print(f"      ❌ Pipeline crashed: {e}")
                        failed_modules = ["crash"]
                        final_state = PipelineState(run_id=run_id, input_question=question, errors=[str(e)])
                        
                    elapsed = time.time() - start_time
                    
                    # 4. Score State using Track C Scorer
                    try:
                        report = await score_pipeline_state_async(
                            final_state,
                            llm_config=pipeline_cfg.qwen.plus.model_dump(),
                            embed_config=pipeline_cfg.evaluation.model_dump(),
                        )
                    except Exception as e:
                        print(f"      ❌ Scorer crashed: {e}")
                        report = {"aggregate": {}}
                        
                    # Calculate cost for this run
                    in_tokens = final_state.total_input_tokens
                    out_tokens = final_state.total_output_tokens
                    run_cost = (in_tokens * QWEN_PLUS_INPUT_PRICE) + (out_tokens * QWEN_PLUS_OUTPUT_PRICE)
                    total_cost_usd += run_cost
                    
                    # Write Row
                    agg = report.get("aggregate", {})
                    row = {
                        "run_id": run_id,
                        "config_name": config_name,
                        "question": question,
                        "repeat_index": repeat,
                        "top1_composite": agg.get("top1_composite", ""),
                        "mean_composite": agg.get("mean_composite", ""),
                        "mean_plan_completeness": agg.get("mean_plan_completeness", ""),
                        "latest_overall_review": agg.get("latest_overall_review", ""),
                        "iterations": final_state.iteration_count,
                        "errors": len(final_state.errors),
                        "input_tokens": in_tokens,
                        "output_tokens": out_tokens,
                        "cost_usd": f"{run_cost:.4f}",
                        "elapsed_seconds": f"{elapsed:.1f}",
                        "failed_modules": "|".join(failed_modules)
                    }
                    writer.writerow(row)
                    f.flush()
                    print(f"      ✅ Done in {elapsed:.1f}s. Cost: ${run_cost:.4f}. Top1: {row['top1_composite']}")

if __name__ == "__main__":
    asyncio.run(main())
