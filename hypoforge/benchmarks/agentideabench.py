"""Native-resource HypoForge experiments on AgentIdeaBench's scored subset."""

from __future__ import annotations

import asyncio
import csv
import gzip
import hashlib
import json
import os
import statistics
import subprocess
import time
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

DOMAINS = ("Biology", "CS", "Chemistry", "Medicine", "Physics")
CRITICS = ("glm-5.1", "qwen3.6-plus", "kimi-k2.6")
WEIGHTS = {"originality": 2.0, "feasibility": 1.0, "clarity": 0.5,
           "impact": 1.5, "specificity": 0.5}
CUTOFF = "2026-05-31"
GENERATION_MODELS = {"glm-5.1", "Pro/zai-org/GLM-5.1"}
QUESTION = "Propose a novel, specific, and testable scientific hypothesis in the following research subfield: {subdomain}."
EXPORT_SYSTEM = """You serialize an existing scientific hypothesis for evaluation.
Return exactly one English paragraph of 80–150 whitespace-separated words,
in first-person future tense. Preserve the hypothesis's central claim,
mechanism, testable prediction and decisive test from the supplied card and
plan. Retain necessary named methods, entities and datasets already present.
Do not invent or improve a claim, method, dataset, result or factual premise.
Do not turn working assumptions into established facts. Do not include scores,
rankings, headings, bullet points, citations or commentary. Return only the
paragraph. This is compression and translation, not additional ideation."""


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


@contextmanager
def experiment_writer(output: Path):
    """One generator/scorer owns a directory; workers share that owner."""
    import fcntl
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".benchmark-writer.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Another generator/scorer is using this output directory") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def record_execution(output: Path, execution: dict) -> None:
    """Record scheduling separately, allowing worker changes when resuming."""
    with (output / "execution_history.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(execution, ensure_ascii=False) + "\n")


def released_scores(root: Path):
    path = root / "release_data/core/lit8d_scores_3seed.csv.gz"
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        yield from csv.DictReader(stream)


def scored_topics(root: Path) -> list[dict]:
    topics = {(row["domain"], row["subdomain"]) for row in released_scores(root)}
    counts = {domain: sum(d == domain for d, _ in topics) for domain in DOMAINS}
    if len(topics) != 40 or any(n != 8 for n in counts.values()):
        raise ValueError(f"Expected the Table 2 subset: 40 subfields, 8 per discipline; got {counts}")
    return [{"domain": d, "subdomain": s} for d, s in sorted(
        topics, key=lambda pair: (DOMAINS.index(pair[0]), pair[1]))]


def select_topics(root: Path, sample: str, limit: int | None = None) -> list[dict]:
    topics = scored_topics(root)
    if sample == "pilot":
        topics = [next(topic for topic in topics if topic["domain"] == d) for d in DOMAINS]
    return topics[:limit] if limit else topics


def item_id(topic: dict, index: int) -> str:
    return "hf-glm51-" + digest([topic["domain"], topic["subdomain"], index])[:20]


def cells(topics: list[dict], repeats: int):
    return [{**topic, "idea_index": index, "item_id": item_id(topic, index)}
            for topic in topics for index in range(1, repeats + 1)]


def public_config(config) -> dict:
    def clean(value):
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items() if k != "api_key"}
        if isinstance(value, list):
            return [clean(v) for v in value]
        return value
    return clean(config.model_dump(mode="json"))


def source_revision(root: Path) -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def configure(config_path: Path, csv_path: str | None, *, api_key_file: str | None = None,
              embedding_csv: str | None = None, embedding_file: str | None = None,
              embedding_model: str | None = None):
    from hypoforge.config import PipelineConfig
    from hypoforge.tools.qwen_client import QwenClient  # load .env before explicit CLI credentials
    from hypoforge.tools.credentials import read_api_key_csv, read_api_key_file

    config = PipelineConfig.from_yaml(config_path)
    if csv_path and api_key_file:
        raise ValueError("Use one generation credential file: CSV or text")
    if embedding_csv and embedding_file:
        raise ValueError("Use one embedding credential file: CSV or text")
    generation_credentials = (read_api_key_csv(csv_path) if csv_path else
                              read_api_key_file(api_key_file) if api_key_file else None)
    if generation_credentials:
        key, base = generation_credentials
        for tier in (config.qwen.base, config.qwen.max, config.qwen.plus, config.qwen.turbo):
            tier.api_key, tier.api_base = key, base
            if urlparse(base).hostname == "api.siliconflow.cn" and tier.model == "glm-5.1":
                tier.model = "Pro/zai-org/GLM-5.1"
    embedding_credentials = (read_api_key_csv(embedding_csv) if embedding_csv else
                             read_api_key_file(embedding_file) if embedding_file else
                             generation_credentials)
    if embedding_credentials:
        key, base = embedding_credentials
        # The entity-normalization embedding client resolves credentials via env.
        os.environ["ENTITY_EMBEDDING_API_KEY"] = key
        os.environ["ENTITY_EMBEDDING_BASE_URL"] = base
        config.evaluation.embedding.base_url = base
    if embedding_model is not None:
        if not embedding_model.strip():
            raise ValueError("Embedding model must be nonempty")
        config.entity_embedding_model = embedding_model.strip()
        config.evaluation.embedding.model_name = embedding_model.strip()
    config.interactive = False
    config.scoring.auto_score = False
    for tier in (config.qwen.base, config.qwen.max, config.qwen.plus, config.qwen.turbo):
        if tier.model not in GENERATION_MODELS or tier.enable_thinking is not False or tier.seed is None:
            raise ValueError("This experiment requires glm-5.1, enable_thinking=false and an explicit seed in every model tier")
    if config.run_mode != "standard" or config.enabled_modules != ["m1", "m2", "m3", "m4", "m5", "m6"]:
        raise ValueError("The benchmark adapter requires the complete standard M1–M6 pipeline")
    return config


def prepare_manifest(output: Path, root: Path, config, topics: list[dict], repeats: int) -> dict:
    from importlib.metadata import version
    repo = Path(__file__).resolve().parents[2]
    source_files = [
        Path(__file__), repo / "hypoforge/tools/qwen_client.py",
        repo / "hypoforge/tools/credentials.py",
        repo / "hypoforge/evaluation/metrics.py",
        repo / "hypoforge/modules/m1_problem_understanding.py",
        repo / "hypoforge/tools/semantic_scholar.py",
        repo / "hypoforge/tools/s2_rate_limit.py",
        repo / "hypoforge/prompts/m1_prompts.py",
        repo / "hypoforge/modules/m2_literature/search/query_planner.py",
        repo / "hypoforge/modules/m2_literature/search/round_plan.py",
        repo / "hypoforge/modules/m2_literature/search/agent.py",
        repo / "hypoforge/modules/m2_literature/reading/access.py",
        repo / "hypoforge/modules/m4_hypothesis_generation.py",
        repo / "hypoforge/strict_contracts.py",
        repo / "hypoforge/modules/m6_review_iteration.py",
        repo / "hypoforge/modules/m2_literature/reading/arxiv_resolver.py",
        repo / "hypoforge/modules/m2_literature/reading/parser.py",
        repo / "hypoforge/modules/m2_literature/reading/pdf_validation.py",
        repo / "hypoforge/benchmarks/fixtures/cff_font_check.pdf",
        repo / "requirements.txt",
    ]
    protocol = {
        "version": 1, "system": "HypoForge", "resource_setting": "native",
        "config": public_config(config), "topics": topics, "repeats": repeats,
        "generation_seeds": [config.qwen.base.seed + index for index in range(repeats)],
        "question_template": QUESTION, "selection": "final top_hypotheses[0]",
        "export_system": EXPORT_SYSTEM,
        "source_hashes": {str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in source_files},
        "pdf_dependencies": {name: version(name) for name in ("pypdf", "fonttools")},
        "semantic_scholar_transport": {
            "min_interval_seconds": max(1.1, float(os.environ.get("SEMANTIC_SCHOLAR_MIN_INTERVAL_SECONDS", "5"))),
            "search_deadline_seconds": 25,
            "metadata_min_deadline_seconds": 30,
            "coordination": "per-key cross-endpoint local-process file lock",
            "search_endpoint": "/graph/v1/paper/search",
        },
        "release_sha256": hashlib.sha256(
            (root / "release_data/core/lit8d_scores_3seed.csv.gz").read_bytes()).hexdigest(),
    }
    signature = digest(protocol)
    path = output / "manifest.json"
    if path.exists():
        manifest = read_json(path)
        if manifest["fingerprint"] != signature:
            raise ValueError("Experiment configuration/task set changed; use a new --output-dir")
        return manifest
    manifest = {"fingerprint": signature, "created_at": now(), "protocol": protocol,
                "hypoforge_revision": source_revision(repo), "agentideabench_revision": source_revision(root),
                "dependencies": {name: version(name) for name in ("langgraph", "langchain-openai", "openai", "pydantic", "httpx", "pypdf", "fonttools")},
                "expected_cells": cells(topics, repeats)}
    write_json(path, manifest)
    return manifest


def validate_paragraph(text: str) -> str:
    text = text.strip()
    if "\n" in text or "```" in text or not 80 <= len(text.split()) <= 150:
        raise ValueError("Export must be a single paragraph of 80–150 words")
    if any("\u4e00" <= char <= "\u9fff" for char in text):
        raise ValueError("Export must be in English")
    return text


def select_submission(state) -> dict:
    if state.errors:
        raise ValueError("Pipeline reported errors; see the saved pipeline state")
    if not state.top_hypotheses or not state.reviews:
        raise ValueError("No final hypothesis reviewed by M6 is available")
    card = state.top_hypotheses[0]
    from hypoforge.modules.m6_review_iteration import _is_invalid_fallback_hypothesis
    if _is_invalid_fallback_hypothesis(card):
        raise ValueError("Deterministic fallback is not a model-generated benchmark hypothesis")
    plans = [plan for plan in state.research_plans if plan.hypothesis_id == card.hypothesis_id]
    if not plans:
        raise ValueError("The selected hypothesis has no matching research plan")
    payload = card.model_dump(mode="json")
    payload.pop("scores", None)
    payload.pop("ranking_rationale", None)
    return {"hypothesis": payload, "research_plan": plans[0].model_dump(mode="json")}


async def export_submission(config, payload: dict) -> tuple[str, dict]:
    from hypoforge.tools.qwen_client import QwenClient, track_token_usage
    client = QwenClient.from_config(config.qwen.base)
    prompt = json.dumps(payload, ensure_ascii=False)
    attempts = []
    with track_token_usage() as usage:
        for _ in range(2):
            text = await client.chat(EXPORT_SYSTEM, prompt, max_tokens=1024, temperature=0.0)
            attempts.append(text)
            try:
                paragraph = validate_paragraph(text)
                return paragraph, {**usage.snapshot(), "responses": attempts}
            except ValueError:
                prompt = json.dumps(payload, ensure_ascii=False) + "\nMeet the required single-paragraph 80–150-word format."
    raise ValueError("Serializer failed the paragraph format after two attempts")


def refresh_exports(output: Path, manifest: dict) -> dict:
    rows = []
    statuses = defaultdict(int)
    for cell in manifest["expected_cells"]:
        path = output / "cells" / cell["item_id"] / "result.json"
        if path.exists():
            row = read_json(path)
            statuses[row["status"]] += 1
            if row["status"] == "success":
                rows.append(row)
        else:
            statuses["pending"] += 1
    temporary = output / "submissions.jsonl.tmp"
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    temporary.replace(output / "submissions.jsonl")
    summary = {"expected": len(manifest["expected_cells"]), "statuses": dict(statuses),
               "complete": len(rows) == len(manifest["expected_cells"])}
    write_json(output / "generation_summary.json", summary)
    return summary


async def check_pdf_environment() -> None:
    """Fail before model calls when full-text font decoding is unavailable."""
    try:
        from fontTools.cffLib import CFFFontSet  # noqa: F401
        from fontTools.ttLib import TTFont  # noqa: F401
    except ImportError as exc:
        raise ValueError(
            "PDF font decoding requires fonttools; run "
            "python -m pip install -r requirements.txt and restart Python"
        ) from exc
    from hypoforge.modules.m2_literature.models import ContentLevel, DocumentRecord
    from hypoforge.modules.m2_literature.reading.parser import PDFDocumentParser

    path = Path(__file__).parent / "fixtures/cff_font_check.pdf"
    chunks = await PDFDocumentParser().parse(DocumentRecord(
        document_id="pdf-environment-check", paper_id="pdf-environment-check",
        content_level=ContentLevel.PDF, local_path=str(path),
    ))
    if "".join(chunk.text for chunk in chunks).strip() != "B":
        raise ValueError(
            "CFF PDF font decoding check failed; update requirements.txt dependencies "
            "and restart Python"
        )
    print("PDF CFF font decoding / text extraction: OK", flush=True)


async def generate(output: Path, root: Path, config, topics: list[dict], repeats: int,
                   retry_failed: bool = False, workers: int = 1,
                   llm_concurrency: int = 8) -> bool:
    from hypoforge.tools.qwen_client import limit_llm_concurrency
    if not 1 <= workers <= 4:
        raise ValueError("Generation workers must be between 1 and 4")
    await check_pdf_environment()
    with experiment_writer(output), limit_llm_concurrency(llm_concurrency):
        started = time.monotonic()
        execution = {"phase": "generate", "started_at": now(), "workers": workers,
                     "llm_concurrency": llm_concurrency, "m2_concurrency": 1}
        try:
            return await _generate(output, root, config, topics, repeats, retry_failed, workers)
        finally:
            execution.update(finished_at=now(), elapsed_seconds=time.monotonic() - started)
            record_execution(output, execution)


async def _generate(output: Path, root: Path, config, topics: list[dict], repeats: int,
                    retry_failed: bool, workers: int) -> bool:
    from hypoforge.pipeline import PipelineRunner
    from hypoforge.tools.qwen_client import is_model_access_error
    manifest = prepare_manifest(output, root, config, topics, repeats)
    m2_gate = asyncio.Semaphore(1)

    class BatchRunner(PipelineRunner):
        def _make_node_wrapper(self, name, module):
            node = super()._make_node_wrapper(name, module)
            async def observed(state):
                label = getattr(self, "benchmark_label", "HypoForge")
                print(f"  [{label}] {name.upper()} started", flush=True)
                started_at = time.monotonic()
                result = await node(state)
                print(f"  [{label}] {name.upper()} completed ({time.monotonic() - started_at:.0f}s)", flush=True)
                return result
            if name != "m2":
                return observed

            async def serial_literature(state):
                # Queue before the native node starts its timers. This prevents
                # parallel cases from exhausting retrieval deadlines in a queue.
                async with m2_gate:
                    return await observed(state)
            return serial_literature

    async def run_cell(index, cell):
        directory = output / "cells" / cell["item_id"]
        result_path = directory / "result.json"
        old = read_json(result_path) if result_path.exists() else {}
        if old and (old["status"] == "success" or not retry_failed):
            print(f"[{index}/{len(manifest['expected_cells'])}] {old['status']}: {cell['subdomain']}", flush=True)
            return
        history = list(old.get("attempt_history", []))
        if old:
            history.append({k: old[k] for k in ("status", "attempt", "error", "elapsed_seconds",
                                               "pipeline_usage", "export_usage", "state_path") if k in old})
        record = {**cell, "idea_model": "HypoForge/glm-5.1", "track": "HypoForge-native",
                  "status": "pipeline_failed", "started_at": now(), "attempt": old.get("attempt", 0) + 1}
        record["generation_seed"] = config.qwen.base.seed + cell["idea_index"] - 1
        record["attempt_history"] = history
        print(f"[{index}/{len(manifest['expected_cells'])}] {cell['subdomain']} / replicate {cell['idea_index']}", flush=True)
        started = time.monotonic()
        try:
            # A formatting failure reuses the finished pipeline; no new idea is generated.
            if old.get("status") == "export_failed":
                record.update({k: old[k] for k in ("submission_source", "pipeline_usage", "state_path", "pipeline_telemetry")})
            else:
                run_directory = directory / f"attempt-{record['attempt']}"
                run_config = config.model_copy(deep=True)
                for tier in (run_config.qwen.base, run_config.qwen.max, run_config.qwen.plus, run_config.qwen.turbo):
                    tier.seed = record["generation_seed"]
                run_config.output_dir = str(run_directory)
                run_config.memory_cache_dir = str(run_directory / "knowledge_graph")
                run_config.entity_cache_dir = str(run_directory / "entity_cache")
                runner = BatchRunner(run_config)
                runner.benchmark_label = cell["domain"]
                run_id = f"{cell['item_id']}-a{record['attempt']}"
                if hasattr(runner, "event_recorder"):
                    from hypoforge.observability import RunEventRecorder
                    runner.event_recorder = RunEventRecorder(run_directory / "telemetry", run_id)
                record["checkpoint_path"] = str(run_directory / f"{run_id}_checkpoint.json")
                state = await runner.run(QUESTION.format(**cell), run_id=run_id)
                record["state_path"] = str(run_directory / f"{run_id}.json")
                record["pipeline_usage"] = {
                    "input": state.total_input_tokens, "output": state.total_output_tokens,
                    "calls": sum(value.get("calls", 0) for value in state.token_usage_by_module.values()),
                    "by_module": state.token_usage_by_module,
                }
                record["pipeline_telemetry"] = {
                    "iterations": state.iteration_count, "search_rounds": state.search_round,
                    "search_ledger": state.search_ledger.model_dump(mode="json"),
                    "literature_result_count": len(state.literature_results),
                    "m2": state.m2_knowledge_export.model_dump(mode="json") if state.m2_knowledge_export else None,
                }
                record["submission_source"] = select_submission(state)
            record["status"] = "export_failed"
            # Persist the selected card before serialization so it survives interruption.
            write_json(result_path, record)
            paragraph, export_usage = await export_submission(config, record["submission_source"])
            record.update(status="success", idea_text=paragraph, word_count=len(paragraph.split()),
                          export_usage=export_usage)
        except asyncio.CancelledError:
            record["error"] = "Generation interrupted; use --retry-failed to retry"
            raise
        except Exception as exc:
            record["error"] = str(exc)
            print(f"  {record['status']}: {exc}", flush=True)
            if is_model_access_error(exc):
                # Preserve completed work and stop spending on a blocked model.
                raise
        finally:
            record["elapsed_seconds"] = time.monotonic() - started
            record["finished_at"] = now()
            write_json(result_path, record)
            refresh_exports(output, manifest)

    jobs = iter(enumerate(manifest["expected_cells"], 1))

    async def worker():
        for index, cell in jobs:
            await run_cell(index, cell)

    tasks = [asyncio.create_task(worker()) for _ in range(workers)]
    try:
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    summary = refresh_exports(output, manifest)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary["complete"]


def historical_baseline(root: Path) -> dict:
    ratings = defaultdict(lambda: defaultdict(list))
    for row in released_scores(root):
        if row["idea_model"] != "z-ai/glm-5.1" or row["track"] != "C":
            continue
        key = (row["domain"], row["subdomain"], int(row["idea_index"]))
        for dimension in WEIGHTS:
            value = row.get("score_" + dimension)
            if value:
                ratings[key][dimension].append(float(value))
    return {key: {dimension: statistics.mean(sorted(values)[:-1] if len(values) > 1 else values)
                  for dimension, values in dimensions.items()}
            for key, dimensions in ratings.items()}


def weighted(dimensions: dict) -> float:
    return sum(dimensions[d] * w for d, w in WEIGHTS.items()) / sum(WEIGHTS.values())


def analyze(output: Path, root: Path) -> bool:
    from hypoforge.benchmarks.agentideabench_scoring import checked_scores
    manifest = read_json(output / "manifest.json")
    score_manifest = read_json(output / "score_manifest.json")
    if score_manifest["generation_fingerprint"] != manifest["fingerprint"]:
        raise ValueError("Score manifest does not match the generation experiment")
    release_sha = manifest["protocol"].get("release_sha256")
    if release_sha and release_sha != hashlib.sha256(
            (root / "release_data/core/lit8d_scores_3seed.csv.gz").read_bytes()).hexdigest():
        raise ValueError("Historical baseline data changed since generation")
    baseline = historical_baseline(root)
    per_cell, missing = {}, []
    for cell in manifest["expected_cells"]:
        try:
            generation = read_json(output / "cells" / cell["item_id"] / "result.json")
            if generation["status"] != "success":
                raise ValueError("generation failed")
            critic_scores = []
            for critic in CRITICS:
                row = read_json(output / "scores" / cell["item_id"] / f"{critic}.json")
                if row["idea_sha256"] != digest(generation["idea_text"]):
                    raise ValueError("score belongs to a different submission")
                critic_scores.append(checked_scores(row["scores"]))
            dimensions = {d: statistics.mean(sorted(score[d] for score in critic_scores)[:2]) for d in WEIGHTS}
            per_cell[(cell["domain"], cell["subdomain"], cell["idea_index"])] = dimensions
        except (OSError, KeyError, ValueError, TypeError) as exc:
            missing.append({**cell, "reason": str(exc)})
    report = {"complete": not missing, "scored_cells": len(per_cell), "expected_cells": len(manifest["expected_cells"]),
              "missing": missing, "resource_setting": "native", "critic_base_url": score_manifest["base_url"],
              "historical_comparison": "cross-provider, current retrieval; not resource-matched"}
    # Never turn missing or failed cells into an apparently complete headline score.
    if not missing:
        per_topic = defaultdict(list)
        base_topic = defaultdict(list)
        for (domain, subdomain, index), dimensions in per_cell.items():
            per_topic[(domain, subdomain)].append(dimensions)
        for (domain, subdomain, index), dimensions in baseline.items():
            if (domain, subdomain) in per_topic:
                base_topic[(domain, subdomain)].append(dimensions)
        means = {key: {d: statistics.mean(v[d] for v in values) for d in WEIGHTS}
                 for key, values in per_topic.items()}
        base_means = {key: {d: statistics.mean(v[d] for v in values) for d in WEIGHTS}
                      for key, values in base_topic.items()}
        dims = {d: statistics.mean(v[d] for v in means.values()) for d in WEIGHTS}
        base_dims = {d: statistics.mean(v[d] for v in base_means.values()) for d in WEIGHTS}
        report.update(dimensions=dims, weighted_total=weighted(dims), historical_baseline=weighted(base_dims),
                      delta=weighted(dims) - weighted(base_dims),
                      win_rate=statistics.mean(weighted(means[k]) > weighted(base_means[k]) for k in means),
                      table2_comparable_sample=len(means) == 40 and manifest["protocol"]["repeats"] == 3)
        # Paired bootstrap over subfields; independent repeats are averaged first.
        import random
        rng = random.Random(42)
        totals = [weighted(v) for v in means.values()]
        differences = [weighted(means[k]) - weighted(base_means[k]) for k in means]
        samples = [([rng.randrange(len(totals)) for _ in totals]) for _ in range(5000)]
        total_samples = sorted(statistics.mean(totals[i] for i in indices) for indices in samples)
        delta_samples = sorted(statistics.mean(differences[i] for i in indices) for indices in samples)
        report["weighted_total_ci95"] = [total_samples[125], total_samples[4874]]
        report["delta_ci95"] = [delta_samples[125], delta_samples[4874]]
        report["per_subdomain"] = [{"domain": k[0], "subdomain": k[1], "dimensions": value,
                                   "total": weighted(value), "baseline_total": weighted(base_means[k])}
                                  for k, value in means.items()]
    write_json(output / "comparison.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("missing", "per_subdomain")}, ensure_ascii=False, indent=2))
    return report["complete"]
