"""Independent lit8d scoring with the upstream prompts and Bailian critics.

Upstream evidence order, abstract truncation and critic prompts are preserved.
Transport failures are surfaced instead of masquerading as empty prior art.
No AgentIdeaBench databases or HypoForge pipeline state are modified.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path

from .agentideabench import (CRITICS, CUTOFF, WEIGHTS, digest, experiment_writer,
                            now, read_json, record_execution, write_json,
                            check_pdf_environment)


def load_rubric(root: Path):
    sys.path.insert(0, str(root))
    rubric = importlib.import_module("experiments.e13_litverify_rubric8")
    scorer = importlib.import_module("evaluation.absolute_scorer")
    if Path(rubric.__file__).resolve().parents[1] != root:
        raise ValueError("An AgentIdeaBench module was loaded from a different checkout")
    if rubric.V4_IDEA_CUTOFF != CUTOFF:
        raise ValueError("The upstream historical cutoff changed; review the experiment protocol")
    return rubric, scorer


def checked_scores(raw: dict) -> dict:
    values = {}
    for dimension in WEIGHTS:
        value = raw[dimension]
        value = value["score"] if isinstance(value, dict) else value
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 1 <= value <= 10:
            raise ValueError(f"Invalid {dimension} score")
        values[dimension] = float(value)
    return values


async def preflight(config, *, critic_config=None, scope="all") -> bool:
    """Local PDF decoding, then small model/embedding/retrieval requests."""
    if scope not in ("all", "generation"):
        raise ValueError("Check scope must be all or generation")
    await check_pdf_environment()
    from hypoforge.tools.qwen_client import QwenClient
    llm = QwenClient.from_config(config.qwen.base)
    response = await llm.structured_chat(
        user_prompt='Return exactly this JSON object: {"ok": true}', max_tokens=64,
        temperature=0.0, disable_thinking=True)
    if response != {"ok": True}:
        raise ValueError("GLM-5.1 structured-output check failed")
    print(f"{config.qwen.base.model} structured output / thinking disabled / seed={config.qwen.base.seed}: OK", flush=True)
    text_response = await llm.chat(
        user_prompt="Reply with OK", max_tokens=32,
        temperature=0.0, disable_thinking=True)
    if not text_response.strip():
        raise ValueError("GLM-5.1 text-output check returned an empty response")
    print(f"{config.qwen.base.model} text output / submission export: OK", flush=True)
    if config.entity_embedding_model:
        embedding_config = config.qwen.base.model_copy(deep=True)
        embedding_config.api_base = (config.evaluation.embedding.base_url
                                     or os.environ.get("ENTITY_EMBEDDING_BASE_URL")
                                     or embedding_config.api_base)
        embedding_config.api_key = (os.environ.get(config.evaluation.embedding.api_key_env_var)
                                    or os.environ.get("ENTITY_EMBEDDING_API_KEY")
                                    or embedding_config.api_key)
        api = CriticAPI(embedding_config)
        try:
            response = api.client.embeddings.create(
                model=config.entity_embedding_model,
                input=["scientific hypothesis", "graph neural networks"],
                encoding_format="float")
            if len(response.data) != 2 or any(not item.embedding for item in response.data):
                raise ValueError("Embedding check returned no vector")
            print(f"{config.entity_embedding_model}: OK", flush=True)
        finally:
            api.client.close()
    if scope == "all":
        api = CriticAPI(critic_config or config.qwen.base)
        try:
            for critic in CRITICS:
                if not api.completion(critic, "", "Reply with OK", max_tokens=32).strip():
                    raise ValueError(f"{critic} returned an empty response")
                print(f"External critic {critic}: OK", flush=True)
        finally:
            api.client.close()
    else:
        print("Generation-only check; external critics have not been checked", flush=True)
    # A search failure is distinct from an authenticated model-service failure.
    try:
        if scope == "generation":
            from hypoforge.tools.semantic_scholar import _search
            hits = await asyncio.to_thread(_search, "graph neural networks", 1)
            label = "native search"
        else:
            hits = search_prior_art("graph neural networks")
            label = "date-filtered search"
        print(f"Semantic Scholar {label}: OK ({len(hits)} hits)", flush=True)
        return True
    except Exception as exc:
        print(f"Semantic Scholar check failed: {exc}", flush=True)
        return False


class CriticAPI:
    def __init__(self, config):
        from openai import OpenAI
        self.client = OpenAI(api_key=config.api_key, base_url=config.api_base,
                             timeout=config.request_timeout_seconds, max_retries=config.max_retries)
        self.usage = {"input": 0, "output": 0, "calls": 0}

    def completion(self, model: str, system: str, user: str, max_tokens: int = 4000) -> str:
        response = self.client.chat.completions.create(
            model=model, messages=[{"role": "system", "content": system},
                                   {"role": "user", "content": user}],
            temperature=0.0, seed=42, max_tokens=max_tokens,
            extra_body={"enable_thinking": False},
        )
        self.usage["calls"] += 1
        if response.usage:
            self.usage["input"] += response.usage.prompt_tokens
            self.usage["output"] += response.usage.completion_tokens
        return response.choices[0].message.content or ""


def extract_queries(api: CriticAPI, rubric, scorer, idea: str) -> list[str]:
    prompt = rubric.EXTRACT_TEMPLATE.format(n=3, idea=idea.strip()[:4000])
    for _ in range(2):
        raw = api.completion(CRITICS[0], rubric.EXTRACT_SYSTEM, prompt)
        parsed = scorer._extract_json(raw)
        queries = parsed.get("queries", []) if parsed else []
        queries = [q.strip() for q in queries if isinstance(q, str) and q.strip()]
        if queries:
            return queries[:3]
    # Match the upstream extraction fallback, while letting transport errors propagate.
    import re
    return [" ".join(re.split(r"[.\n]", idea.strip())[0].split()[:10])]


def search_prior_art(query: str) -> list[dict]:
    import httpx
    headers = {}
    if os.environ.get("SEMANTIC_SCHOLAR_API_KEY"):
        headers["x-api-key"] = os.environ["SEMANTIC_SCHOLAR_API_KEY"]
    params = {"query": query, "fields": "paperId,title,abstract,year,citationCount,publicationDate",
              "limit": 10, "publicationDateOrYear": f":{CUTOFF}"}
    with httpx.Client(timeout=30) as client:
        for attempt in range(4):
            response = client.get("https://api.semanticscholar.org/graph/v1/paper/search",
                                  params=params, headers=headers)
            if response.status_code == 429 or response.status_code >= 500:
                if attempt < 3:
                    time.sleep((3, 6, 12)[attempt])
                    continue
            response.raise_for_status()
            data = response.json().get("data")
            if not isinstance(data, list):
                raise ValueError("Semantic Scholar returned no valid result list")
            return data
    raise ValueError("Prior-art search failed")


def build_evidence(api: CriticAPI, rubric, scorer, idea: str) -> dict:
    queries = extract_queries(api, rubric, scorer, idea)
    per_query = []
    for query in queries:
        time.sleep(1.0)  # conservative spacing; this runner scores sequentially
        per_query.append(search_prior_art(query))
    seen, evidence = set(), []
    for rank in range(10):
        for qi, hits in enumerate(per_query):
            if rank >= len(hits) or len(evidence) >= 8:
                continue
            paper = hits[rank]
            pid = paper.get("paperId")
            if not pid or pid in seen:
                continue
            seen.add(pid)
            evidence.append({"paperId": pid, "title": paper.get("title"), "year": paper.get("year"),
                             "publicationDate": paper.get("publicationDate"),
                             "citationCount": paper.get("citationCount") or 0,
                             "abstract": (paper.get("abstract") or "")[:450], "query": queries[qi]})
    return {"queries": queries, "evidence": evidence, "cutoff": CUTOFF,
            "idea_sha256": digest(idea), "created_at": now()}


def judge(api: CriticAPI, rubric, scorer, model: str, idea: str, domain: str, evidence: dict) -> dict:
    prompt = scorer.USER_TEMPLATE.format(idea=idea.strip(), domain=domain,
                                        references=rubric.format_evidence_block(evidence["evidence"], CUTOFF))
    attempts = []
    for attempt in range(3):
        raw = api.completion(model, rubric.LIT8D_SYSTEM,
                             prompt + (scorer.JSON_RETRY_SUFFIX if attempt else ""))
        attempts.append(raw)
        parsed = scorer._extract_json(raw)
        if parsed:
            try:
                checked_scores(parsed)
                return {"scores": scorer._normalise(parsed), "responses": attempts}
            except (KeyError, TypeError, ValueError):
                pass
    return {"error": "No complete five-dimensional score after three responses", "responses": attempts}


def _judge_one(config, rubric, scorer, model, idea, domain, evidence):
    # Each task owns its SDK client and usage counter. Evidence stays frozen.
    api = CriticAPI(config)
    try:
        try:
            result = judge(api, rubric, scorer, model, idea, domain, evidence)
        except Exception as exc:
            if getattr(exc, "status_code", None) in (401, 403):
                raise
            result = {"error": str(exc)}
        return {**result, "usage": dict(api.usage)}
    finally:
        api.client.close()


def score(output: Path, root: Path, config, retry_failed: bool = False,
          workers: int = 1) -> bool:
    if not 1 <= workers <= 4:
        raise ValueError("Scoring workers must be between 1 and 4")
    with experiment_writer(output), ThreadPoolExecutor(max_workers=min(workers, len(CRITICS))) as pool:
        started = time.monotonic()
        execution = {"phase": "score", "started_at": now(),
                     "critic_concurrency": min(workers, len(CRITICS)),
                     "evidence_concurrency": 1}
        try:
            api = CriticAPI(config)
            with closing(api.client):
                return _score(output, root, config, retry_failed, pool, api)
        finally:
            execution.update(finished_at=now(), elapsed_seconds=time.monotonic() - started)
            record_execution(output, execution)


def _score(output: Path, root: Path, config, retry_failed: bool, pool, api) -> bool:
    rubric, scorer = load_rubric(root)
    manifest = read_json(output / "manifest.json")
    protocol = {"generation_fingerprint": manifest["fingerprint"], "base_url": config.api_base,
                "critics": list(CRITICS), "cutoff": CUTOFF, "temperature": 0.0, "seed": 42,
                "enable_thinking": False,
                "prompt_sha256": digest([rubric.LIT8D_SYSTEM, scorer.USER_TEMPLATE,
                                         rubric.EXTRACT_SYSTEM, rubric.EXTRACT_TEMPLATE])}
    score_manifest = output / "score_manifest.json"
    if score_manifest.exists() and read_json(score_manifest) != protocol:
        raise ValueError("Scoring protocol changed; use a separate experiment directory")
    write_json(score_manifest, protocol)
    complete = True
    for index, cell in enumerate(manifest["expected_cells"], 1):
        generation_path = output / "cells" / cell["item_id"] / "result.json"
        if not generation_path.exists() or read_json(generation_path)["status"] != "success":
            complete = False
            continue
        generation = read_json(generation_path)
        idea = generation["idea_text"]
        directory = output / "scores" / cell["item_id"]
        print(f"[{index}/{len(manifest['expected_cells'])}] scoring {cell['subdomain']} / {cell['idea_index']}", flush=True)
        evidence_path = directory / "evidence.json"
        before = dict(api.usage)
        try:
            evidence = read_json(evidence_path) if evidence_path.exists() else None
            if evidence and evidence["idea_sha256"] != digest(idea):
                raise ValueError("Cached evidence belongs to a different submission")
            failure_path = directory / "evidence_error.json"
            if evidence is None:
                if failure_path.exists() and not retry_failed:
                    complete = False
                    continue
                evidence = build_evidence(api, rubric, scorer, idea)
                evidence["usage"] = {k: api.usage[k] - before[k] for k in before}
                write_json(evidence_path, evidence)
            pending = []
            for critic in CRITICS:
                path = directory / f"{critic}.json"
                old = read_json(path) if path.exists() else None
                if old and old["idea_sha256"] != digest(idea):
                    raise ValueError("Cached score belongs to a different submission")
                if old and (old.get("scores") or not retry_failed):
                    if not old.get("scores"):
                        complete = False
                    continue
                pending.append((critic, path, pool.submit(
                    _judge_one, config, rubric, scorer, critic, idea, cell["domain"], evidence)))
            for critic, path, future in pending:
                result = future.result()
                result.update(critic_model=critic, item_id=cell["item_id"], idea_sha256=digest(idea),
                              created_at=now())
                write_json(path, result)
                if "scores" not in result:
                    complete = False
        except Exception as exc:
            complete = False
            write_json(directory / "evidence_error.json", {"error": str(exc), "created_at": now(),
                                                         "idea_sha256": digest(idea)})
            print(f"  scoring failed: {exc}", flush=True)
            if getattr(exc, "status_code", None) in (401, 403):
                raise ValueError("API authentication/permission failed; correct credentials before retrying") from exc
    return complete
