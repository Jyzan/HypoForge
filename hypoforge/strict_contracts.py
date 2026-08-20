"""Strict built-in implementations for HypoForge scientific contracts.

The public interfaces stay unchanged.  These subclasses close the remaining
fail-open edge cases in the built-in M2/M3/M5 implementations and provide
pair-level entity identity decisions.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import os
import threading
from contextvars import ContextVar
from types import MethodType
from typing import Any, Sequence

from .entity_normalization import (
    EntityNormalizationService,
    EntityPairDecision,
    NormalizedEntity,
    _stable_id,
    clean_entity_surface,
)
from .literature.adapter import AgenticM2Adapter
from .modules.m2_literature_search import M2LiteratureSearch
from .literature.export import build_m2_knowledge_export_run
from .literature.models import (
    PaperRecord,
    QueryIntent,
    SearchQuery,
    SearchRunResult,
    StopReason,
)
from .memory.paper_store import PaperStore, normalize_query_text, paper_key
from .modules.m3_evidence_graph import M3EvidenceGraph
from .modules.m4_hypothesis_generation import M4HypothesisGeneration
from .modules.m5_research_plan import M5ResearchPlan
from .modules.m6_review_iteration import M6ReviewIteration
from .observability import emit_event
from .paper_sources import (
    dedupe_papers_by_key,
    dedupe_queries,
    gap_candidate_queries,
    merge_literature_increment,
    open_paper_store,
    resolve_gap_sub_question,
)
from .state import (
    EvidenceGap,
    KnowledgeEntry,
    LiteratureResult,
    M2KnowledgeExport,
    PipelineState,
    ResearchPlan,
    SearchLedger,
)

logger = logging.getLogger(__name__)


def _strict_alignment_call_chat(client: object, prompt: str) -> str:
    """Invoke semantic-alignment chat with *prompt* as a user message."""
    if not hasattr(client, "chat"):
        return ""
    chat = client.chat
    try:
        parameters = inspect.signature(chat).parameters
    except (TypeError, ValueError):
        parameters = {}

    def invoke():
        if "user_prompt" in parameters:
            return chat(user_prompt=prompt)
        return chat(prompt)

    if inspect.iscoroutinefunction(chat):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return str(asyncio.run(invoke()) or "")
        result_container: list[str] = []
        error_container: list[BaseException] = []

        async def resolve() -> None:
            try:
                result_container.append(str(await invoke() or ""))
            except BaseException as exc:
                error_container.append(exc)

        def run() -> None:
            asyncio.run(resolve())

        thread = threading.Thread(target=run)
        thread.start()
        thread.join()
        if error_container:
            raise error_container[0]
        return result_container[0] if result_container else ""
    return str(invoke() or "")


async def _strict_grounding_embed(self, texts):
    """Validate configured dense retrieval output before grounding proceeds."""
    vectors = await _ORIGINAL_GROUNDING_EMBED(self, texts)
    if not self.embedding_model:
        return vectors
    if not texts:
        return vectors
    if not vectors:
        raise RuntimeError(
            f"Grounding embedding model {self.embedding_model!r} returned no vectors "
            "for non-empty input; refusing a silent BM25-only downgrade."
        )
    if len(vectors) != len(texts):
        raise RuntimeError(
            f"Grounding embedding returned {len(vectors)} vector(s) for "
            f"{len(texts)} input text(s)."
        )
    dimensions = {len(vector) for vector in vectors if vector is not None}
    if len(dimensions) != 1 or not dimensions or 0 in dimensions:
        raise RuntimeError(
            "Grounding embedding returned empty or inconsistent vector dimensions."
        )
    return vectors


async def _strict_grounding_synthesize_one(self, item):
    """Require real semantic synthesis whenever an LLM grounding mode is enabled."""
    fallback = self._fallback_record(item)
    if self.mode not in {"llm", "api", "direct"}:
        return fallback
    if self.client is None:
        raise RuntimeError(
            f"Grounding evidence synthesis mode={self.mode!r} requires an LLM client."
        )

    from .modules.m3_grounding.models import EvidenceRecord

    evidence = item["evidence_item"]
    model_name = str(getattr(self.client, "model", "unknown"))
    cache_payload = "\0".join([
        "strict-rcs-v1",
        model_name,
        str(item.get("query") or ""),
        evidence.evidence_id,
    ]).encode("utf-8")
    cache_path = self.cache_dir / "rcs_strict" / (
        hashlib.sha256(cache_payload).hexdigest() + ".json"
    )
    if cache_path.exists():
        try:
            cached = EvidenceRecord.model_validate_json(
                cache_path.read_text(encoding="utf-8")
            )
            if (cached.context or {}).get("synthesis_contract") == "llm_strict_v1":
                return cached
        except Exception:
            logger.debug("Ignoring invalid strict RCS cache file %s", cache_path)

    schema = {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "excerpt": {"type": "string"},
            "relevance_score": {
                "type": "number", "minimum": 0, "maximum": 10,
            },
            "claims": {
                "type": "array", "items": {"type": "string"}, "maxItems": 5,
            },
            "entities": {"type": "array", "items": {"type": "string"}},
            "methods": {"type": "array", "items": {"type": "string"}},
            "limitations": {"type": "array", "items": {"type": "string"}},
            "epistemic_status": {"type": "string"},
            "context": {"type": "object"},
        },
        "required": ["summary", "excerpt", "relevance_score", "claims"],
    }
    prompt = (
        f"Research query:\n{item['query']}\n\n"
        f"Evidence from paper {evidence.paper_id} "
        f"(section={evidence.section or 'unknown'}, page={evidence.page}):\n"
        f"Quote: {evidence.quote}\n"
        f"Normalized claim: {evidence.normalized_claim}\n\n"
        "Create a retrieval-contextual summary. The excerpt must be an exact "
        "quote from the supplied evidence. Split claims into atomic statements "
        "and do not invent unsupported content."
    )
    try:
        data = await self.client.structured_chat(
            system_prompt=(
                "You extract auditable scientific evidence. Never invent text "
                "absent from the provided evidence."
            ),
            user_prompt=prompt,
            output_schema=schema,
            temperature=0.0,
            disable_thinking=True,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Grounding evidence synthesis failed for {evidence.evidence_id!r} "
            f"in configured mode={self.mode!r}."
        ) from exc

    if not isinstance(data, dict) or data.get("_parse_error"):
        raise RuntimeError(
            f"Grounding evidence synthesis returned invalid structured output "
            f"for {evidence.evidence_id!r}."
        )
    summary = str(data.get("summary") or "").strip()
    claims = [
        str(value).strip()
        for value in (data.get("claims") or [])
        if str(value).strip()
    ][:5]
    if not summary or not claims:
        raise RuntimeError(
            f"Grounding evidence synthesis returned empty required semantic "
            f"fields for {evidence.evidence_id!r}."
        )
    try:
        relevance_score = float(data["relevance_score"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Grounding evidence synthesis returned an invalid relevance score "
            f"for {evidence.evidence_id!r}."
        ) from exc
    if not 0.0 <= relevance_score <= 10.0:
        raise RuntimeError(
            f"Grounding evidence synthesis relevance score is out of range for "
            f"{evidence.evidence_id!r}."
        )

    excerpt = str(data.get("excerpt") or "").strip()
    if not excerpt or excerpt not in evidence.quote:
        # This is a deterministic provenance repair, not a semantic fallback:
        # the replacement is copied from the authoritative M2 evidence quote.
        excerpt = evidence.quote[:1200]
    record = EvidenceRecord(
        id=fallback.id,
        evidence_id=evidence.evidence_id,
        paper_id=evidence.paper_id,
        chunk_id=evidence.chunk_id,
        section=evidence.section,
        page=evidence.page,
        query=item["query"],
        quote=evidence.quote,
        normalized_claim=evidence.normalized_claim,
        summary=summary,
        excerpt=excerpt,
        relevance_score=relevance_score,
        retrieval_score=float(item["score"]),
        claims=claims,
        entities=[str(value) for value in (data.get("entities") or []) if str(value)],
        methods=[str(value) for value in (data.get("methods") or []) if str(value)],
        limitations=[
            str(value) for value in (data.get("limitations") or []) if str(value)
        ],
        epistemic_status=str(data.get("epistemic_status") or "reported"),
        context={
            **fallback.context,
            **dict(data.get("context") or {}),
            "synthesis_contract": "llm_strict_v1",
        },
    )
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(record.model_dump_json(), encoding="utf-8")
    except OSError as exc:
        logger.debug("Could not write strict RCS cache file %s: %s", cache_path, exc)
    return record


async def _strict_grounding_synthesize_evidence(self, gs):
    """Run synthesis concurrently but never convert timeout into fallback."""
    semaphore = asyncio.Semaphore(self.max_concurrency)

    async def guarded(item):
        async with semaphore:
            try:
                return await asyncio.wait_for(
                    self._synthesize_one(item),
                    timeout=self.llm_call_timeout,
                )
            except asyncio.TimeoutError as exc:
                evidence = item.get("evidence_item")
                evidence_id = getattr(evidence, "evidence_id", "")
                raise RuntimeError(
                    f"Grounding evidence synthesis timed out for {evidence_id!r}; "
                    f"mode={self.mode!r} cannot silently degrade."
                ) from exc

    records = await asyncio.gather(
        *(guarded(item) for item in gs.get("candidates", []))
    )
    unique = {}
    for record in records:
        if record.relevance_score >= self.min_relevance:
            old = unique.get(record.id)
            if old is None or record.relevance_score > old.relevance_score:
                unique[record.id] = record
    records = sorted(
        unique.values(), key=lambda record: record.relevance_score, reverse=True
    )
    report = gs["report"].model_copy(deep=True)
    report.evidence_records = len(records)
    return {"evidence_records": records, "report": report}


async def _strict_grounding_judge_relation_batch(
    self,
    pairs,
    claim_map,
    record_map,
):
    decisions = await _ORIGINAL_GROUNDING_JUDGE_RELATION_BATCH(
        self, pairs, claim_map, record_map
    )
    if self.mode in {"llm", "api", "direct"}:
        if self.client is None:
            raise RuntimeError(
                f"Grounding relation judge mode={self.mode!r} requires an LLM client."
            )
        if len(decisions) != len(pairs):
            raise RuntimeError(
                f"Grounding relation judge returned {len(decisions)}/{len(pairs)} "
                "validated pair decisions. Partial relation judgment is not allowed."
            )
        self._strict_judged_pair_ids.update(pair.id for pair in pairs)
    return decisions


async def _strict_grounding_judge_candidate_relations(self, gs):
    pairs = list(gs.get("relation_pairs", []))
    llm_mode = self.mode in {"llm", "api", "direct"}
    if llm_mode and pairs and self.client is None:
        raise RuntimeError(
            f"Grounding relation judge mode={self.mode!r} requires an LLM client."
        )
    self._strict_judged_pair_ids = set()
    result = await _ORIGINAL_GROUNDING_JUDGE_CANDIDATES(self, gs)
    if llm_mode and pairs:
        expected = {pair.id for pair in pairs}
        missing = expected - self._strict_judged_pair_ids
        if missing:
            raise RuntimeError(
                "Grounding relation judgment did not complete for recalled pair IDs: "
                f"{sorted(missing)}"
            )
    return result


async def _strict_search_agent_run(self, *args, **kwargs):
    """Propagate critical agent stage failures represented as StopReason.ERROR."""
    result = await _ORIGINAL_SEARCH_AGENT_RUN(self, *args, **kwargs)
    if result.stop_reason is StopReason.ERROR:
        details = "; ".join(result.errors[-5:]) or "unspecified agent stage failure"
        raise RuntimeError(
            "Agentic M2 search terminated with StopReason.ERROR: " + details
        )
    return result


async def _strict_scout_read(self, sub_question, papers):
    """Retry malformed semantic notes once, then omit only failed papers."""
    paper_list = list(papers)
    if not paper_list:
        return []
    if self.client is None:
        return await _ORIGINAL_SCOUT_READ(self, sub_question, paper_list)

    batches = [
        paper_list[index:index + self.batch_size]
        for index in range(0, len(paper_list), self.batch_size)
    ]
    semaphore = asyncio.Semaphore(self.max_concurrency)

    async def run_batch(batch):
        async with semaphore:
            try:
                initial_notes = await self._read_batch(sub_question, batch)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "Scout batch failed; retrying each paper once (%s: %s)",
                    type(exc).__name__,
                    exc,
                )
                initial_notes = []

            valid_by_id = {
                note.paper_id: note
                for note in initial_notes
                if note.paper_id in {paper.paper_id for paper in batch}
                and note.study_design
            }
            retry_papers = [
                paper for paper in batch if paper.paper_id not in valid_by_id
            ]
            for paper in retry_papers:
                emit_event(
                    "tool_started",
                    module="m2",
                    tool="scout_reader_retry",
                    status="running",
                    message=f"Scout reading results incomplete; retrying single paper: {paper.title}",
                    details={"paper_id": paper.paper_id},
                )
                retry_error = ""
                try:
                    retry_notes = await self._read_batch(sub_question, [paper])
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    retry_notes = []
                    retry_error = f"{type(exc).__name__}: {exc}"
                retry_note = next(
                    (
                        note for note in retry_notes
                        if note.paper_id == paper.paper_id and note.study_design
                    ),
                    None,
                )
                if retry_note is not None:
                    valid_by_id[paper.paper_id] = retry_note
                    emit_event(
                        "tool_completed",
                        module="m2",
                        tool="scout_reader_retry",
                        status="completed",
                        message=f"Single-paper scout reading retry succeeded: {paper.title}",
                        details={"paper_id": paper.paper_id},
                    )
                    continue

                logger.warning(
                    "Scout single-paper retry failed; skipping %s%s",
                    paper.paper_id,
                    f" ({retry_error})" if retry_error else "",
                )
                emit_event(
                    "tool_result",
                    module="m2",
                    tool="scout_reader_retry",
                    status="warning",
                    message=f"Single-paper scout reading retry failed again; skipping paper: {paper.title}",
                    details={
                        "paper_id": paper.paper_id,
                        "error": retry_error,
                    },
                )

            return [
                valid_by_id[paper.paper_id]
                for paper in batch
                if paper.paper_id in valid_by_id
            ]

    batch_results = await asyncio.gather(
        *(run_batch(batch) for batch in batches)
    )
    notes = [note for batch in batch_results for note in batch]
    return notes


_ENTITY_EMBEDDING_BASE_URL: ContextVar[str] = ContextVar(
    "hypoforge_entity_embedding_base_url", default=""
)
_ENTITY_EMBEDDING_KEY_ENV: ContextVar[str] = ContextVar(
    "hypoforge_entity_embedding_key_env", default=""
)


def _set_entity_embedding_context(base_url: str, key_env: str) -> None:
    _ENTITY_EMBEDDING_BASE_URL.set(str(base_url or "").strip())
    _ENTITY_EMBEDDING_KEY_ENV.set(str(key_env or "").strip())


class StrictEntityNormalizationService(EntityNormalizationService):
    """Use independent three-state decisions for each surface/candidate pair."""

    def __init__(
        self,
        *,
        embedding_base_url: str = "",
        embedding_key_env: str = "",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.embedding_base_url = (
            str(embedding_base_url or "").strip()
            or _ENTITY_EMBEDDING_BASE_URL.get()
        )
        self.embedding_key_env = (
            str(embedding_key_env or "").strip()
            or _ENTITY_EMBEDDING_KEY_ENV.get()
        )

    @classmethod
    def from_task(
        cls,
        contract,
        *,
        domains=(),
        cache_dir="",
        client=None,
        embedding_backend=None,
        embedding_model: str = "",
        embedding_base_url: str = "",
        embedding_key_env: str = "",
        run_stamp: str = "",
    ) -> "StrictEntityNormalizationService":
        """Create a service for *contract*, optionally with explicit embedding config.

        ``embedding_base_url`` and ``embedding_key_env`` can be supplied
        directly; when omitted they fall back to the ContextVars set by
        ``_set_entity_embedding_context`` (populated by StrictM2/M3 __call__).
        """
        from .entity_normalization import clean_entity_surface
        namespace = "|".join(sorted(
            clean_entity_surface(domain) for domain in domains
            if clean_entity_surface(domain)
        )) or "global"
        service = cls(
            cache_dir=cache_dir,
            namespace=namespace,
            client=client,
            embedding_backend=embedding_backend,
            embedding_model=embedding_model,
            embedding_base_url=embedding_base_url,
            embedding_key_env=embedding_key_env,
            run_stamp=run_stamp,
        )
        service.register_task_contract(contract)
        return service

    async def _embedding_candidates(
        self,
        surfaces: list[str],
        canonical_names: list[str],
    ) -> dict[str, list[str]]:
        # Ensure the configured endpoint is valid before delegating.
        if self.embedding_backend is None and self.embedding_model:
            base_url = (
                self.embedding_base_url
                or os.getenv("ENTITY_EMBEDDING_BASE_URL", "")
                or os.getenv("OPENAI_BASE_URL", "")
            )
            if not base_url:
                raise RuntimeError(
                    "Entity embedding is configured but no shared embedding endpoint is set. "
                    "Set ENTITY_EMBEDDING_BASE_URL or OPENAI_BASE_URL (or entity_embedding_base_url in the config)."
                )
            key_env = self.embedding_key_env or ""
            credential = (
                (os.getenv(key_env, "") or os.getenv("OPENAI_API_KEY", ""))
                if key_env
                else os.getenv("ENTITY_EMBEDDING_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")
            )
            if key_env and not credential:
                raise RuntimeError(
                    "Entity embedding is configured but its credential environment "
                    f"variable {key_env!r} is not set."
                )
        else:
            base_url = (
                self.embedding_base_url
                or os.getenv("ENTITY_EMBEDDING_BASE_URL", "")
                or os.getenv("OPENAI_BASE_URL", "")
            )
            key_env = self.embedding_key_env or ""
            credential = (
                (os.getenv(key_env, "") or os.getenv("OPENAI_API_KEY", ""))
                if key_env
                else os.getenv("ENTITY_EMBEDDING_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")
            )
        # Pass resolved endpoint and key directly to _embed_raw so we never
        # mutate the process environment with os.environ.setdefault.
        return await super()._embedding_candidates(
            surfaces, canonical_names,
            _base_url=base_url,
            _api_key=credential,
        )

    async def _judge(
        self,
        requests: list[dict[str, Any]],
    ) -> dict[tuple[str, str], EntityPairDecision]:
        if self.client is None or not requests:
            return {}
        schema = {
            "type": "object",
            "properties": {
                "decisions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "surface": {"type": "string"},
                            "canonical_name": {"type": "string"},
                            "same_concept": {"type": "boolean"},
                            "rationale": {"type": "string"},
                        },
                        "required": [
                            "surface",
                            "canonical_name",
                            "same_concept",
                            "rationale",
                        ],
                    },
                },
            },
            "required": ["decisions"],
        }
        try:
            payload = await self.client.structured_chat(
                system_prompt=(
                    "Decide whether each supplied entity pair denotes exactly "
                    "the same real concept. Related, broader, narrower, part, "
                    "cause, and effect concepts are different."
                ),
                user_prompt=json.dumps(requests, ensure_ascii=False, indent=2),
                output_schema=schema,
                temperature=0.0,
                disable_thinking=True,
            )
        except Exception as exc:
            raise RuntimeError(
                "Entity identity judgment failed while identity resolution was required."
            ) from exc

        allowed = {
            (clean_entity_surface(item["surface"]), clean_entity_surface(candidate))
            for item in requests
            for candidate in item["candidates"]
        }
        output: dict[tuple[str, str], EntityPairDecision] = {}
        decisions = payload.get("decisions", []) if isinstance(payload, dict) else []
        for item in decisions:
            surface = clean_entity_surface(item.get("surface", ""))
            canonical = clean_entity_surface(item.get("canonical_name", ""))
            pair = (surface, canonical)
            same = item.get("same_concept")
            if not surface or not canonical or pair not in allowed:
                continue
            if not isinstance(same, bool):
                continue
            output[pair] = EntityPairDecision(
                left=surface,
                right=canonical,
                same_concept=same,
                rationale=str(item.get("rationale", ""))[:500],
            )
        return output

    async def resolve_batch(self, names: Sequence[str] | Any):
        original_names = list(dict.fromkeys(str(name or "") for name in names))
        surfaces = {
            original: clean_entity_surface(original) for original in original_names
        }
        result = {}
        unresolved: list[str] = []
        for original, surface in surfaces.items():
            canonical_id = self.alias_to_id.get(surface)
            if canonical_id and canonical_id in self.records:
                result[original] = self.records[canonical_id]
            elif surface:
                unresolved.append(surface)

        canonical_names = sorted({record.canonical_name for record in self.records.values()})
        embedded = await self._embedding_candidates(unresolved, canonical_names)
        candidate_map: dict[str, list[str]] = {}
        for surface in unresolved:
            candidates = embedded.get(surface) or self._lexical_candidates(
                surface, canonical_names
            )
            undecided: list[str] = []
            resolved_from_cache = False
            for candidate in candidates:
                cached = self.pair_decisions.get(self._pair_key(surface, candidate))
                if cached and cached.same_concept:
                    canonical_id = self.alias_to_id.get(clean_entity_surface(candidate))
                    if canonical_id:
                        self.alias_to_id[surface] = canonical_id
                        resolved_from_cache = True
                        break
                if cached is None:
                    undecided.append(candidate)
            if not resolved_from_cache and undecided:
                candidate_map[surface] = undecided

        judged = await self._judge([
            {"surface": surface, "candidates": candidates}
            for surface, candidates in candidate_map.items()
        ])
        deferred_surfaces: set[str] = set()
        for surface, candidates in candidate_map.items():
            explicit: list[tuple[str, EntityPairDecision]] = []
            positive: list[str] = []
            unanswered = False
            for candidate in candidates:
                decision = judged.get((surface, clean_entity_surface(candidate)))
                if decision is None:
                    unanswered = True
                    continue
                explicit.append((candidate, decision))
                if decision.same_concept:
                    positive.append(candidate)
            if len(positive) > 1:
                # Multiple positives are common when distinct records denote
                # the same concept family (e.g. "crispr-cas9" vs "crispr/cas9
                # system"). One ambiguous name must not abort the whole
                # question: keep the surface as a standalone unresolved entity
                # (deliberately not cached, so later sub-questions defer the
                # same way instead of silently merging to the first positive).
                logger.warning(
                    "Entity judge returned multiple identity matches for %r: %s. "
                    "Deferring as unresolved; no merge performed.",
                    surface,
                    positive,
                )
                deferred_surfaces.add(surface)
                continue
            for candidate, decision in explicit:
                self.pair_decisions[self._pair_key(surface, candidate)] = decision
            if len(positive) == 1:
                canonical_id = self.alias_to_id.get(
                    clean_entity_surface(positive[0])
                )
                if canonical_id:
                    record = self.records[canonical_id]
                    self._upsert(
                        record.canonical_name,
                        [*record.aliases, surface],
                        decision_source="llm_pair_decision",
                    )
            elif unanswered:
                deferred_surfaces.add(surface)

        for original, surface in surfaces.items():
            if not surface:
                continue
            canonical_id = self.alias_to_id.get(surface)
            if canonical_id:
                record = self.records[canonical_id]
            elif surface in deferred_surfaces:
                record = NormalizedEntity(
                    canonical_id=_stable_id(self.namespace, surface),
                    canonical_name=surface,
                    aliases=[surface],
                    decision_source="unresolved_identity",
                )
            else:
                record = self._upsert(surface, [surface], decision_source="new_entity")
            result[original] = record
        self._save()
        return result


class StrictAgenticM2Adapter(AgenticM2Adapter):
    """Agentic supplement cache hits are read before they become evidence."""

    def __init__(
        self,
        *args,
        entity_embedding_base_url: str = "",
        entity_embedding_key_env: str = "",
        **kwargs,
    ) -> None:
        self.entity_embedding_base_url = str(entity_embedding_base_url or "").strip()
        self.entity_embedding_key_env = str(entity_embedding_key_env or "").strip()
        super().__init__(*args, **kwargs)

    async def __call__(self, state: PipelineState, config=None):
        scout = getattr(self.search_agent, "scout_reader", None)
        if isinstance(scout, _ScoutReader) and not getattr(
            scout, "_hypoforge_strict_read", False
        ):
            scout.read = MethodType(_strict_scout_read, scout)
            scout._hypoforge_strict_read = True

        if isinstance(self.search_agent, _IterativeSearchAgent) and not getattr(
            self.search_agent, "_hypoforge_strict_run", False
        ):
            self.search_agent.run = MethodType(
                _strict_search_agent_run, self.search_agent
            )
            self.search_agent._hypoforge_strict_run = True

        _set_entity_embedding_context(
            self.entity_embedding_base_url,
            self.entity_embedding_key_env,
        )
        return await super().__call__(state, config)

    @staticmethod
    def _cached_records(cached: Sequence[dict[str, Any]]) -> list[PaperRecord]:
        fields = set(PaperRecord.model_fields)
        records: list[PaperRecord] = []
        for item in cached:
            payload = {key: value for key, value in item.items() if key in fields}
            payload["sources"] = payload.get("sources") or ["paper_store"]
            try:
                records.append(PaperRecord.model_validate(payload))
            except Exception as exc:
                logger.warning("Skipping malformed cached paper candidate: %s", exc)
        return records

    async def _run_supplement(
        self,
        state: PipelineState,
        open_gaps: Sequence[EvidenceGap],
    ) -> dict[str, Any]:
        card = state.problem_card
        key_entities = card.key_entities if card else []
        domains = card.domain if card else []
        store = None
        if state.memory_cache_dir:
            try:
                store = PaperStore(state.memory_cache_dir)
            except OSError as exc:
                logger.warning("Paper cache unavailable; using live search: %s", exc)

        existing_runs = list(state.m2_knowledge_export.runs) if state.m2_knowledge_export else []
        known_keys = {paper_key(paper) for run in existing_runs for paper in run.papers}
        known_keys.update(state.search_ledger.paper_keys)
        issued_norms = {
            normalize_query_text(query) for query in state.search_ledger.queries_issued
        }
        merged_results = [item.model_copy(deep=True) for item in state.literature_results]
        existing_sub_questions = [item.sub_question for item in merged_results]
        new_runs = []
        new_query_texts: list[str] = []
        new_paper_keys: list[str] = []
        grounded_gap_ids: set[str] = set()
        remaining_budget = self.supplement_paper_budget

        for gap in open_gaps:
            sub_question = self._resolve_sub_question(gap, existing_sub_questions)
            if store is not None:
                cached_meta, cache_queries = self._cache_lookup(store, gap, known_keys)
                cached_papers = self._cached_records(cached_meta)[:remaining_budget]
                if cached_papers:
                    queries = [
                        SearchQuery(
                            query_id=f"cache-{gap.gap_id}-{index + 1}",
                            text=query,
                            intent=QueryIntent.CORE,
                            target_source="paper_store",
                            purpose="read cached candidate for evidence gap",
                            target_gap=gap.description,
                            relation_to_question="supplement",
                        )
                        for index, query in enumerate(cache_queries)
                    ]
                    cached_result = SearchRunResult(
                        sub_question=sub_question,
                        queries=queries,
                        papers_found=len(cached_papers),
                        papers_after_dedup=len(cached_papers),
                        candidates=list(cached_papers),
                        final_papers=list(cached_papers),
                        iterations=0,
                        stop_reason=StopReason.COVERAGE_SATISFIED,
                        source_result_counts={"paper_store": len(cached_papers)},
                    )
                    reading = await self.reading_workflow.run(
                        sub_question,
                        cached_papers,
                        search_context=cached_result,
                    )
                    reading = await self._normalise_reading_entities(state, list(reading))
                    self._validate_reading_contract(sub_question, cached_result, list(reading))
                    export_run = build_m2_knowledge_export_run(
                        sub_question, cached_result, reading
                    )
                    new_runs.append(export_run)
                    self._merge_increment(
                        merged_results,
                        sub_question,
                        len(cached_papers),
                        list(export_run.knowledge_entries),
                    )
                    hit_keys = [paper_key(paper) for paper in cached_papers]
                    known_keys.update(hit_keys)
                    new_paper_keys.extend(hit_keys)
                    grounded_gap_ids.add(gap.gap_id)
                    remaining_budget -= len(cached_papers)
                    for query in cache_queries:
                        store.record_query(
                            query,
                            hit_keys,
                            run_id=state.run_id,
                            round=state.search_round,
                        )
                    emit_event(
                        "memory_hit",
                        module="m2",
                        status="completed",
                        message=f"Gap {gap.gap_id} read {len(cached_papers)} cached paper candidates",
                        details={"gap_id": gap.gap_id, "hits": len(cached_papers)},
                    )
                    continue

            fresh_queries = self._gap_queries(gap, issued_norms)
            if not fresh_queries or remaining_budget <= 0:
                continue
            search_result = await self._supplement_search(
                gap,
                sub_question,
                fresh_queries,
                key_entities=key_entities,
                domains=domains,
                paper_limit=remaining_budget,
            )
            new_query_texts.extend(fresh_queries)
            new_papers = []
            for paper in search_result.final_papers:
                key = paper_key(paper)
                if key not in known_keys:
                    known_keys.add(key)
                    new_papers.append(paper)
            if not new_papers:
                continue
            filtered = search_result.model_copy(update={"final_papers": new_papers})
            reading = await self.reading_workflow.run(
                sub_question, new_papers, search_context=filtered
            )
            reading = await self._normalise_reading_entities(state, list(reading))
            contract_exempted = self._validate_reading_contract(
                sub_question, filtered, list(reading)
            )
            export_run = build_m2_knowledge_export_run(sub_question, filtered, reading)
            if (
                contract_exempted
                and not export_run.knowledge_entries
                and not export_run.evidence
            ):
                # Contextual-fallback batch only: no knowledge was produced
                # and none was promised.  Keep the gap open instead of
                # exporting an empty run and marking it grounded.
                emit_event(
                    "evidence_gap",
                    module="m2",
                    tool="evidence_gap",
                    status="warning",
                    message=(
                        f"Supplement search for evidence gap {gap.gap_id} kept "
                        "only contextual evidence; no knowledge entries were "
                        "produced; the gap stays open pending a later supplement search"
                    ),
                    details={
                        "gap_id": gap.gap_id,
                        "sub_question": sub_question,
                        "papers": len(new_papers),
                    },
                )
                continue
            new_runs.append(export_run)
            self._merge_increment(
                merged_results,
                sub_question,
                len(new_papers),
                list(export_run.knowledge_entries),
            )
            hit_keys = [paper_key(paper) for paper in new_papers]
            new_paper_keys.extend(hit_keys)
            grounded_gap_ids.add(gap.gap_id)
            remaining_budget -= len(new_papers)
            if store is not None:
                store.upsert_papers(
                    new_papers, run_id=state.run_id, round=state.search_round
                )
                for query in fresh_queries:
                    store.record_query(
                        query,
                        hit_keys,
                        run_id=state.run_id,
                        round=state.search_round,
                    )

        updated_gaps = [
            gap.model_copy(update={"status": "pending_grounding"})
            if gap.gap_id in grounded_gap_ids and gap.status == "open"
            else gap.model_copy(deep=True)
            for gap in state.evidence_gaps
        ]
        return {
            "literature_results": merged_results,
            "m2_knowledge_export": M2KnowledgeExport(runs=[*existing_runs, *new_runs]),
            "evidence_gaps": updated_gaps,
            "search_ledger": SearchLedger(
                queries_issued=[*state.search_ledger.queries_issued, *new_query_texts],
                paper_keys=[*state.search_ledger.paper_keys, *new_paper_keys],
            ),
        }


class StrictM2LiteratureSearch(M2LiteratureSearch):
    """Strict Pipeline facade using the strict Literature adapter."""

    def __init__(
        self,
        *args,
        entity_embedding_base_url: str = "",
        entity_embedding_key_env: str = "",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        base = self.adapter
        self.adapter = StrictAgenticM2Adapter(
            search_agent=base.search_agent,
            reading_workflow=base.reading_workflow,
            budget=base.budget,
            supplement_paper_budget=base.supplement_paper_budget,
            entity_judge_client=base.entity_judge_client,
            entity_embedding_model=base.entity_embedding_model,
            entity_embedding_base_url=entity_embedding_base_url,
            entity_embedding_key_env=entity_embedding_key_env,
            # Forward the agentic configuration the integrated factory set on
            # the base adapter — dropping these silently disabled sub-question
            # entity supplementation, full-text backfill and the open-access
            # expansion search in production (strict is the default).
            subquestion_entity_client=base.subquestion_entity_client,
            round_strategy=base.round_strategy,
            fulltext_backfill_enabled=base.fulltext_backfill_enabled,
            fulltext_backfill_target=base.fulltext_backfill_target,
            fulltext_backfill_max_attempts=base.fulltext_backfill_max_attempts,
            fulltext_backfill_min_relevance=base.fulltext_backfill_min_relevance,
            fulltext_backfill_min_directness=base.fulltext_backfill_min_directness,
            subquestion_concurrency=base.subquestion_concurrency,
            fresh_run_timeout_seconds=base.fresh_run_timeout_seconds,
        )


# Backward-compatible name for callers that imported the pre-facade class.
StrictAgenticM2Module = StrictM2LiteratureSearch


class StrictM3EvidenceGraph(M3EvidenceGraph):
    """M3 that cannot silently skip configured grounding or LLM relations."""

    def __init__(
        self,
        *args,
        entity_embedding_base_url: str = "",
        entity_embedding_key_env: str = "",
        **kwargs,
    ) -> None:
        self.entity_embedding_base_url = str(entity_embedding_base_url or "").strip()
        self.entity_embedding_key_env = str(entity_embedding_key_env or "").strip()
        super().__init__(*args, **kwargs)
        if isinstance(self._grounder, _GroundingWorkflow) and not getattr(
            self._grounder, "_hypoforge_strict_grounding", False
        ):
            self._grounder._embed = MethodType(
                _strict_grounding_embed, self._grounder
            )
            self._grounder._synthesize_one = MethodType(
                _strict_grounding_synthesize_one, self._grounder
            )
            self._grounder._synthesize_evidence = MethodType(
                _strict_grounding_synthesize_evidence, self._grounder
            )
            self._grounder._judge_relation_batch = MethodType(
                _strict_grounding_judge_relation_batch, self._grounder
            )
            self._grounder._judge_candidate_relations = MethodType(
                _strict_grounding_judge_candidate_relations, self._grounder
            )
            self._grounder._hypoforge_strict_grounding = True

    @staticmethod
    def _grounded_evidence_ids(graph) -> set[str]:
        if graph is None:
            return set()
        grounded: set[str] = set()
        for node in graph.nodes:
            metadata = node.metadata or {}
            if metadata.get("provenance_level") != "m2_fulltext_grounded":
                continue
            evidence_id = str(metadata.get("evidence_id") or "")
            if evidence_id:
                grounded.add(evidence_id)
            grounded.update(
                str(value) for value in metadata.get("evidence_ids", []) if value
            )
        return grounded

    @staticmethod
    def _export_evidence_ids(state: PipelineState) -> set[str]:
        export = state.m2_knowledge_export
        if export is None:
            return set()
        return {
            evidence.evidence_id
            for run in export.runs
            for evidence in run.evidence
            if evidence.evidence_id
        }

    @staticmethod
    def _grounding_fingerprint(evidence_ids: set[str]) -> str:
        payload = "\0".join(sorted(evidence_ids)).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:24]

    @staticmethod
    def _has_grounding_fingerprint(graph, fingerprint: str) -> bool:
        if graph is None or not fingerprint:
            return False
        for node in graph.nodes:
            values = (node.metadata or {}).get("grounding_evidence_fingerprints", [])
            if fingerprint in values:
                return True
        return False

    @staticmethod
    def _mark_grounding_fingerprint(graph, fingerprint: str):
        if not graph.nodes:
            raise RuntimeError(
                "Grounding completed but there is no graph node on which to record provenance."
            )
        target = next(
            (
                node for node in graph.nodes
                if (node.metadata or {}).get("provenance_level")
                == "m2_fulltext_grounded"
            ),
            graph.nodes[0],
        )
        metadata = dict(target.metadata or {})
        values = list(metadata.get("grounding_evidence_fingerprints", []))
        if fingerprint not in values:
            values.append(fingerprint)
        metadata["grounding_evidence_fingerprints"] = values
        target.metadata = metadata
        return graph

    @staticmethod
    def _dedupe_graph(graph):
        nodes_by_id = {}
        for node in graph.nodes:
            nodes_by_id[node.id] = node
        graph.nodes = list(nodes_by_id.values())

        edges_by_key = {}
        for edge in graph.edges:
            relation = getattr(edge.relation, "value", edge.relation)
            key = (
                edge.source,
                edge.target,
                str(relation),
                tuple(sorted(str(value) for value in edge.evidence_ids if value)),
            )
            edges_by_key[key] = edge
        graph.edges = list(edges_by_key.values())
        graph.established_facts = list(dict.fromkeys(graph.established_facts))
        graph.conflicts = list(dict.fromkeys(graph.conflicts))
        graph.knowledge_gaps = list(dict.fromkeys(graph.knowledge_gaps))
        return graph

    async def _extract_relation_batch(self, entries: list) -> dict:
        result = await super()._extract_relation_batch(entries)
        if isinstance(result, dict) and not result.get("_parse_error"):
            self._strict_relation_successes += 1
        return result

    async def _enhance_with_llm_batched(self, graph, all_entries):
        self._strict_relation_successes = 0
        result = await super()._enhance_with_llm_batched(graph, all_entries)
        relation_entries = list(all_entries)
        if self.max_relation_entries:
            relation_entries = relation_entries[: self.max_relation_entries]
        if relation_entries and self._strict_relation_successes == 0:
            raise RuntimeError(
                "Every configured M3 relation-extraction job failed or returned invalid output."
            )
        return result

    async def __call__(self, state: PipelineState, config=None):
        _set_entity_embedding_context(
            self.entity_embedding_base_url,
            self.entity_embedding_key_env,
        )
        if self.mode in {"llm", "direct", "api"} and self.client is None:
            raise RuntimeError(
                f"M3 mode={self.mode!r} explicitly requires an LLM client; "
                "relation extraction cannot be silently skipped."
            )
        requested_grounding = self.grounding_enabled
        has_entries = any(item.knowledge_entries for item in state.literature_results)
        current_evidence_ids: set[str] = set()
        fingerprint = ""
        if requested_grounding:
            if self._grounder is None:
                raise RuntimeError("M3 grounding is enabled but its workflow is unavailable.")
            current_evidence_ids = self._export_evidence_ids(state)
            if not current_evidence_ids:
                raise RuntimeError(
                    "M3 grounding is enabled but M2 supplied no citable evidence."
                )
            fingerprint = self._grounding_fingerprint(current_evidence_ids)

        pre_graph = state.evidence_graph
        if pre_graph is None and state.memory_cache_dir:
            query = " ".join(filter(None, [
                state.input_question,
                " ".join(state.problem_card.domain) if state.problem_card else "",
                " ".join(state.problem_card.key_entities) if state.problem_card else "",
            ]))
            pre_graph = self._load_persistent_graph(state.memory_cache_dir, query)
        already_marked = self._has_grounding_fingerprint(pre_graph, fingerprint)
        legacy_complete = bool(
            requested_grounding
            and current_evidence_ids
            and current_evidence_ids.issubset(self._grounded_evidence_ids(pre_graph))
        )
        suppress_base_grounding = bool(
            requested_grounding and (already_marked or legacy_complete)
        )
        if suppress_base_grounding:
            self.grounding_enabled = False
        try:
            result = await super().__call__(state, config)
        finally:
            self.grounding_enabled = requested_grounding

        graph = self._dedupe_graph(result["evidence_graph"])
        if requested_grounding:
            # Base M3 now routes both new-entry and checkpoint-resume calls
            # through the same grounding stage.  Strict mode only validates
            # and fingerprints that result; it must not run grounding twice.
            if suppress_base_grounding:
                graph = self._mark_grounding_fingerprint(graph, fingerprint)
            else:
                grounded_now = self._grounded_evidence_ids(graph) & current_evidence_ids
                if not grounded_now:
                    if not has_entries:
                        raise RuntimeError(
                            "M3 grounding failed while grounding a resumed historical "
                            "graph: no grounded provenance was produced for the current "
                            "M2 evidence set."
                        )
                    raise RuntimeError(
                        "M3 grounding returned successfully but produced no grounded "
                        "provenance for the current M2 evidence set."
                    )
                graph = self._mark_grounding_fingerprint(graph, fingerprint)
        result["evidence_graph"] = graph
        return result


class M4GateAllRejectedError(RuntimeError):
    """Every candidate was rejected by a strict M4 quality gate.

    A ``RuntimeError`` subclass so existing ``pytest.raises(RuntimeError,
    match=...)`` assertions keep matching.  Carries the raw review payload so
    the strict gate-recovery pass can build regeneration feedback.
    """

    def __init__(self, message, *, gate, candidate_ids, reviews):
        super().__init__(message)
        self.gate = gate  # "critic" | "falsifiability"
        self.candidate_ids = list(candidate_ids)
        self.reviews = list(reviews)  # validated verdict dicts only


class StrictM4HypothesisGeneration(M4HypothesisGeneration):
    """Make every configured M4 quality gate authoritative and complete."""

    def _attach_rankings(self, candidates, payload, context=None):
        ranked = super()._attach_rankings(candidates, payload, context)
        expected = min(self.top_k, len(candidates))
        if len(ranked) != expected:
            raise RuntimeError(
                "M4 Ranker returned an incomplete valid ranking: "
                f"expected {expected}, received {len(ranked)}. "
                "Configured multi-agent ranking cannot silently fall back."
            )
        return ranked

    async def _run_critic(self, state: PipelineState, candidates):
        from .graph_context import build_graph_context
        from .prompts.m4_prompts import (
            M4_CRITIC_SYSTEM_PROMPT,
            M4_CRITIC_USER_TEMPLATE,
        )

        schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "hypothesis_id": {"type": "string"},
                    "pass": {"type": "boolean"},
                    "critique": {"type": "string"},
                    "issues": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["hypothesis_id", "pass", "critique", "issues"],
            },
        }
        base_prompt = M4_CRITIC_USER_TEMPLATE.format(
            graph_context=build_graph_context(state).render(),
            established_facts=self._graph_bucket_text(state, "established_facts"),
            hypotheses_json=json.dumps(
                [card.model_dump(mode="json") for card in candidates],
                ensure_ascii=False,
                indent=2,
            ),
        )
        user_prompt = base_prompt
        attempt = 0
        while True:
            attempt += 1
            try:
                result = await self._tool_call(
                    "hypothesis_critic",
                    self.client.structured_chat(
                        system_prompt=M4_CRITIC_SYSTEM_PROMPT,
                        user_prompt=user_prompt,
                        output_schema=schema,
                        max_tokens=8192,
                        temperature=getattr(self.llm_config, "temperature", 0.1),
                    ),
                    details={"candidates": len(candidates), "attempt": attempt},
                )
            except Exception as exc:
                raise RuntimeError(
                    "M4 Critic stage failed; the configured quality gate cannot be skipped."
                ) from exc

            if not isinstance(result, list):
                # A format failure is not a quality decision: retry once with
                # the raw payload as feedback before treating it as fatal.
                if attempt >= 2:
                    raise RuntimeError("M4 Critic returned a non-list verdict payload.")
                logger.warning(
                    "M4 Critic returned a non-list verdict payload; retrying once "
                    "with the raw response as feedback."
                )
                user_prompt = self._critic_shape_retry_prompt(base_prompt, result)
                continue

            survivors, missing = self._parse_critic_verdicts(result, candidates)
            if missing:
                if attempt >= 2:
                    raise RuntimeError(
                        "M4 Critic did not return one valid verdict for candidate IDs: "
                        f"{sorted(missing)}"
                    )
                logger.warning(
                    "M4 Critic returned no valid verdict for candidate IDs %s; "
                    "retrying once for the missing verdicts.",
                    sorted(missing),
                )
                user_prompt = self._critic_missing_verdict_retry_prompt(
                    base_prompt, sorted(missing)
                )
                continue

            if not survivors:
                raise M4GateAllRejectedError(
                    "M4 Critic rejected every candidate; rejected hypotheses cannot "
                    "be silently re-admitted.",
                    gate="critic",
                    candidate_ids=[card.hypothesis_id for card in candidates],
                    reviews=[r for r in result if isinstance(r, dict)],
                )
            return survivors

    async def _run_falsifiability(self, state: PipelineState, candidates):
        from .graph_context import build_graph_context
        from .prompts.m4_prompts import (
            M4_FALSIFIABILITY_SYSTEM_PROMPT,
            M4_FALSIFIABILITY_USER_TEMPLATE,
        )

        schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "hypothesis_id": {"type": "string"},
                    "is_falsifiable": {"type": "boolean"},
                    "specificity_score": {
                        "type": "number", "minimum": 0, "maximum": 1
                    },
                    "assessment": {"type": "string"},
                },
                "required": ["hypothesis_id", "is_falsifiable", "assessment"],
            },
        }
        base_prompt = M4_FALSIFIABILITY_USER_TEMPLATE.format(
            graph_context=build_graph_context(state).render(),
            hypotheses_json=json.dumps(
                [card.model_dump(mode="json") for card in candidates],
                ensure_ascii=False,
                indent=2,
            ),
        )
        user_prompt = base_prompt
        attempt = 0
        while True:
            attempt += 1
            try:
                result = await self._tool_call(
                    "falsifiability_checker",
                    self.client.structured_chat(
                        system_prompt=M4_FALSIFIABILITY_SYSTEM_PROMPT,
                        user_prompt=user_prompt,
                        output_schema=schema,
                        max_tokens=8192,
                        temperature=getattr(self.llm_config, "temperature", 0.1),
                    ),
                    details={"candidates": len(candidates), "attempt": attempt},
                )
            except Exception as exc:
                raise RuntimeError(
                    "M4 Falsifiability Checker failed; the configured quality gate "
                    "cannot be skipped."
                ) from exc

            if not isinstance(result, list):
                # A format failure is not a quality decision: retry once with
                # the raw payload as feedback before treating it as fatal.
                if attempt >= 2:
                    raise RuntimeError(
                        "M4 Falsifiability Checker returned a non-list verdict payload."
                    )
                logger.warning(
                    "M4 Falsifiability Checker returned a non-list verdict payload; "
                    "retrying once with the raw response as feedback."
                )
                user_prompt = self._falsifiability_shape_retry_prompt(
                    base_prompt, result
                )
                continue

            survivors, missing = self._parse_falsifiability_verdicts(
                result, candidates
            )
            if missing:
                if attempt >= 2:
                    raise RuntimeError(
                        "M4 Falsifiability Checker did not return one valid verdict "
                        f"for candidate IDs: {sorted(missing)}"
                    )
                logger.warning(
                    "M4 Falsifiability Checker returned no valid verdict for "
                    "candidate IDs %s; retrying once for the missing verdicts.",
                    sorted(missing),
                )
                user_prompt = self._falsifiability_missing_verdict_retry_prompt(
                    base_prompt, sorted(missing)
                )
                continue

            if not survivors:
                raise M4GateAllRejectedError(
                    "M4 Falsifiability Checker rejected every candidate; "
                    "non-falsifiable hypotheses cannot be silently re-admitted.",
                    gate="falsifiability",
                    candidate_ids=[card.hypothesis_id for card in candidates],
                    reviews=[r for r in result if isinstance(r, dict)],
                )
            return survivors

    # ------------------------------------------------------------------
    # Verdict parsing + format-retry prompts (shared by the gate loops)
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_critic_verdicts(result, candidates):
        """Validate critic verdicts; returns (survivors, missing ID set)."""
        by_id = {card.hypothesis_id: card for card in candidates}
        reviewed: set[str] = set()
        survivors = []
        for review in result:
            if not isinstance(review, dict):
                continue
            hypothesis_id = str(review.get("hypothesis_id") or "")
            if hypothesis_id not in by_id or hypothesis_id in reviewed:
                continue
            if not isinstance(review.get("pass"), bool):
                continue
            reviewed.add(hypothesis_id)
            if review["pass"]:
                survivors.append(by_id[hypothesis_id])
        return survivors, set(by_id) - reviewed

    @staticmethod
    def _parse_falsifiability_verdicts(result, candidates):
        """Validate falsifiability verdicts; returns (survivors, missing ID set)."""
        by_id = {card.hypothesis_id: card for card in candidates}
        reviewed: set[str] = set()
        survivors = []
        for review in result:
            if not isinstance(review, dict):
                continue
            hypothesis_id = str(review.get("hypothesis_id") or "")
            if hypothesis_id not in by_id or hypothesis_id in reviewed:
                continue
            if not isinstance(review.get("is_falsifiable"), bool):
                continue
            reviewed.add(hypothesis_id)
            if review["is_falsifiable"]:
                survivors.append(by_id[hypothesis_id])
        return survivors, set(by_id) - reviewed

    @staticmethod
    def _critic_shape_retry_prompt(base_prompt, payload) -> str:
        """Retry prompt for a Critic response that parsed to a non-array."""
        raw = (
            str(payload.get("raw_response", payload))
            if isinstance(payload, dict)
            else str(payload)
        )
        return (
            f"{base_prompt}\n\nYour previous response could not be parsed as the "
            "required verdict array. Respond with valid JSON only: an object with "
            'a single "entries" key whose value is the array of verdict objects, '
            "one per candidate (hypothesis_id, boolean pass, critique, issues). "
            "Raw response received (truncated to 500 chars): "
            f"{raw[:500]}"
        )

    @staticmethod
    def _critic_missing_verdict_retry_prompt(base_prompt, missing) -> str:
        """Retry prompt for a Critic response with incomplete verdict coverage."""
        return (
            f"{base_prompt}\n\nYour previous response did not include one valid "
            f"verdict for every candidate. Missing hypothesis_id(s): {missing}. "
            'Return an "entries" array with exactly one verdict object per '
            'candidate: "hypothesis_id", boolean "pass", "critique", "issues".'
        )

    @staticmethod
    def _falsifiability_shape_retry_prompt(base_prompt, payload) -> str:
        """Retry prompt for a Falsifiability response that parsed to a non-array."""
        raw = (
            str(payload.get("raw_response", payload))
            if isinstance(payload, dict)
            else str(payload)
        )
        return (
            f"{base_prompt}\n\nYour previous response could not be parsed as the "
            "required verdict array. Respond with valid JSON only: an object with "
            'a single "entries" key whose value is the array of verdict objects, '
            "one per candidate (hypothesis_id, boolean is_falsifiable, "
            "assessment). Raw response received (truncated to 500 chars): "
            f"{raw[:500]}"
        )

    @staticmethod
    def _falsifiability_missing_verdict_retry_prompt(base_prompt, missing) -> str:
        """Retry prompt for a Falsifiability response with missing verdicts."""
        return (
            f"{base_prompt}\n\nYour previous response did not include one valid "
            f"verdict for every candidate. Missing hypothesis_id(s): {missing}. "
            'Return an "entries" array with exactly one verdict object per '
            'candidate: "hypothesis_id", boolean "is_falsifiable", "assessment".'
        )

    # ------------------------------------------------------------------
    # Gate-recovery: all-rejected candidates regenerate once with feedback
    # ------------------------------------------------------------------

    async def _run_quality_gates(
        self,
        state: PipelineState,
        candidates,
        *,
        generation_shortfall: bool,
        question: str = "",
        graph_context=None,
        feedback_context: str = "",
    ):
        """Run the quality gates; on all-rejected, regenerate once with the
        gate's feedback and re-run the same gate (bounded to one pass).

        The recovery keys on the structured :class:`M4GateAllRejectedError`,
        never on ``generation_shortfall`` (which is stale after the contract
        repair path).  Transport-level failures keep the base swallow-on-
        shortfall semantics.
        """
        from .graph_context import build_graph_context as _build_graph_context

        question = question or (
            state.problem_card.original_question
            if state.problem_card
            else state.input_question
        )
        graph_context = graph_context or _build_graph_context(state)

        try:
            candidates = await self._run_critic(state, candidates)
        except M4GateAllRejectedError as exc:
            candidates = await self._recover_gate_rejection(
                state,
                candidates,
                exc,
                gate_name="M4 Critic",
                feedback_block=self._critic_recovery_feedback_block(exc.reviews),
                gate_call=self._run_critic,
                question=question,
                graph_context=graph_context,
                feedback_context=feedback_context,
            )
        except Exception:
            if not generation_shortfall:
                raise
            logger.warning(
                "M4 Critic failed after the one retry still left too few "
                "hypotheses; continuing with the generated candidates."
            )

        try:
            candidates = await self._run_falsifiability(state, candidates)
        except M4GateAllRejectedError as exc:
            candidates = await self._recover_gate_rejection(
                state,
                candidates,
                exc,
                gate_name="M4 Falsifiability Checker",
                feedback_block=self._falsifiability_recovery_feedback_block(
                    exc.reviews
                ),
                gate_call=self._run_falsifiability,
                question=question,
                graph_context=graph_context,
                feedback_context=feedback_context,
            )
        except Exception:
            if not generation_shortfall:
                raise
            logger.warning(
                "M4 Falsifiability Checker failed after the one retry still "
                "left too few hypotheses; continuing with the current candidates."
            )
        return candidates

    async def _recover_gate_rejection(
        self,
        state: PipelineState,
        candidates,
        exc: M4GateAllRejectedError,
        *,
        gate_name: str,
        feedback_block: str,
        gate_call,
        question: str,
        graph_context,
        feedback_context: str,
    ):
        """One regeneration pass with the gate's feedback, then re-run the
        same gate once.  A second all-rejected result raises with review
        excerpts — the gate stays authoritative, never silently re-admits.
        """
        emit_event(
            "tool_result",
            module="m4",
            tool="hypothesis_generator_gate_recovery",
            status="warning",
            message=f"{gate_name} rejected every candidate; running one regeneration pass and re-reviewing",
            details={"gate": exc.gate, "rejected": len(candidates)},
        )
        try:
            _, regenerated, contract_failures = (
                await self._generate_hypothesis_batch(
                    state,
                    question=question,
                    graph_context=graph_context,
                    feedback_context="\n".join(
                        item for item in [feedback_context, feedback_block] if item
                    ),
                    tool_name="hypothesis_generator_gate_recovery",
                    attempt=3,
                    # Recovery only needs a few fresh candidates to merge with
                    # the rejected set — keep the pass small to bound the time.
                    requested_count=min(self.num_candidates, 3),
                )
            )
        except Exception as regen_exc:
            # A transport/format failure in the recovery pass is not a quality
            # verdict: fail cleanly with the recovery step named, instead of
            # leaking a chained exception that hides the original rejection.
            emit_event(
                "tool_failed",
                module="m4",
                tool="hypothesis_generator_gate_recovery",
                status="failed",
                message=f"{gate_name} gate-recovery regeneration failed",
                details={"gate": exc.gate, "error": type(regen_exc).__name__},
            )
            raise RuntimeError(
                f"{gate_name} gate recovery regeneration failed with "
                f"{type(regen_exc).__name__}; the configured quality gate "
                "cannot be skipped."
            ) from regen_exc
        if contract_failures:
            logger.warning(
                "M4 gate recovery regeneration dropped %d candidate(s) for "
                "contract failures",
                len(contract_failures),
            )
        candidates = self._merge_generation_candidates(candidates, regenerated)
        try:
            return await gate_call(state, candidates)
        except M4GateAllRejectedError as second_exc:
            excerpts = "; ".join(
                str(
                    r.get("critique")
                    or r.get("assessment")
                    or ""
                )[:120]
                for r in second_exc.reviews
                if isinstance(r, dict)
            )[:400]
            emit_event(
                "tool_failed",
                module="m4",
                tool="hypothesis_generator_gate_recovery",
                status="failed",
                message=f"{gate_name} still rejected every candidate after the fix; candidates cannot be admitted",
                details={"gate": exc.gate},
            )
            raise RuntimeError(
                f"{gate_name} rejected every candidate, including after one "
                "gate-recovery regeneration pass; the configured quality gate "
                f"cannot be skipped. Review excerpts: {excerpts}"
            ) from second_exc
        except Exception as re_exc:
            # Any non-rejection failure while re-running the gate (transport,
            # format) is reported cleanly; the recovery candidates never get a
            # silent pass, and the original rejection stays visible as cause.
            emit_event(
                "tool_failed",
                module="m4",
                tool="hypothesis_generator_gate_recovery",
                status="failed",
                message=f"{gate_name} could not be re-run after gate recovery",
                details={"gate": exc.gate, "error": type(re_exc).__name__},
            )
            raise RuntimeError(
                f"{gate_name} could not be re-run after gate recovery "
                f"({type(re_exc).__name__}); rejected candidates cannot be "
                "re-admitted."
            ) from re_exc

    @staticmethod
    def _critic_recovery_feedback_block(reviews) -> str:
        lines = [
            "The previous candidate batch was rejected by the Critic. "
            "Address the following critiques and generate improved, "
            "distinct scientific hypotheses:"
        ]
        for review in reviews:
            if not isinstance(review, dict):
                continue
            hypothesis_id = str(review.get("hypothesis_id") or "?")
            critique = str(review.get("critique") or "").strip()
            if critique:
                lines.append(f"- [{hypothesis_id}] {critique}")
            for issue in review.get("issues") or []:
                text = str(issue).strip()
                if text:
                    lines.append(f"  - issue: {text}")
        return "\n".join(lines)

    @staticmethod
    def _falsifiability_recovery_feedback_block(reviews) -> str:
        lines = [
            "The previous candidate batch was rejected by the Falsifiability "
            "Checker. Generate hypotheses with explicit, empirically testable "
            "falsification conditions:"
        ]
        for review in reviews:
            if not isinstance(review, dict):
                continue
            hypothesis_id = str(review.get("hypothesis_id") or "?")
            assessment = str(review.get("assessment") or "").strip()
            if assessment:
                lines.append(f"- [{hypothesis_id}] {assessment}")
        return "\n".join(lines)


class StrictM5ResearchPlan(M5ResearchPlan):
    """Require every cited evidence item to match a declared source paper."""

    async def __call__(self, state: PipelineState, config=None):
        if not state.top_hypotheses:
            raise RuntimeError(
                "M5 requires at least one top hypothesis; an empty research-plan "
                "collection is not a successful M5 result."
            )
        result = await super().__call__(state, config)
        plans = list(result.get("research_plans", []))
        expected_ids = [hypothesis.hypothesis_id for hypothesis in state.top_hypotheses]
        actual_ids = [plan.hypothesis_id for plan in plans]
        if len(actual_ids) != len(expected_ids) or sorted(actual_ids) != sorted(expected_ids):
            raise RuntimeError(
                "M5 must produce exactly one research plan for every top hypothesis; "
                f"expected IDs={expected_ids}, actual IDs={actual_ids}."
            )
        return result

    @staticmethod
    def _sanitize_evidence_links(plan: ResearchPlan, context):
        cleaned = M5ResearchPlan._sanitize_evidence_links(plan, context)
        mapping = context.evidence_to_paper
        strict_links = []
        supported_evidence: list[str] = []
        supported_papers: list[str] = []
        for link in cleaned.evidence_links:
            status = link.support_status
            evidence_ids = list(link.supporting_evidence_ids)
            declared_papers = list(link.source_paper_ids)
            mapped_papers = [mapping.get(evidence_id, "") for evidence_id in evidence_ids]
            exact_provenance = bool(evidence_ids) and all(mapped_papers) and (
                set(declared_papers) == set(mapped_papers)
            )
            if status == "supported" and not exact_provenance:
                status = "hypothesis_to_validate"
            if status == "supported":
                supported_evidence.extend(evidence_ids)
                supported_papers.extend(mapped_papers)
            strict_links.append(link.model_copy(update={"support_status": status}))
        return cleaned.model_copy(update={
            "supporting_evidence_ids": list(dict.fromkeys(supported_evidence)),
            "source_paper_ids": list(dict.fromkeys(supported_papers)),
            "evidence_links": strict_links,
        })


class StrictM6ReviewIteration(M6ReviewIteration):
    """Built-in M6 marker ensuring strict runtime hooks are loaded explicitly."""


from .literature.search.agent import IterativeSearchAgent as _IterativeSearchAgent

_ORIGINAL_SEARCH_AGENT_RUN = _IterativeSearchAgent.run


from .literature.search.scout import ScoutReader as _ScoutReader

_ORIGINAL_SCOUT_READ = _ScoutReader.read


from .modules.m3_grounding.workflow import GroundingWorkflow as _GroundingWorkflow

_ORIGINAL_GROUNDING_EMBED = _GroundingWorkflow._embed
_ORIGINAL_GROUNDING_SYNTHESIZE_ONE = _GroundingWorkflow._synthesize_one
_ORIGINAL_GROUNDING_JUDGE_RELATION_BATCH = _GroundingWorkflow._judge_relation_batch
_ORIGINAL_GROUNDING_JUDGE_CANDIDATES = _GroundingWorkflow._judge_candidate_relations


# The legacy modules imported the base service directly.  Rebind that module
# global once when strict built-ins are loaded so every built-in M2/M3 path uses
# the same pair-level identity contract.
from .literature import adapter as _adapter_module
from .modules import m3_evidence_graph as _m3_module

_adapter_module.EntityNormalizationService = StrictEntityNormalizationService
_m3_module.EntityNormalizationService = StrictEntityNormalizationService

# ``assess_task_alignment`` resolves ``_call_chat`` from its defining module at
# execution time. Rebind that helper so production Qwen clients receive the
# semantic-alignment task as ``user_prompt`` rather than accidentally as their
# first positional ``system_prompt`` argument.
from . import task_alignment as _task_alignment_module

_task_alignment_module._call_chat = _strict_alignment_call_chat
