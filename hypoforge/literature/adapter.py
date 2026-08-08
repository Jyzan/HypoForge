"""Agentic M2 module — config-driven wrapper with internal DI adapter.

Track A delivers two classes:

* ``AgenticM2Adapter`` — internal dependency-injection adapter.  Accepts
  pre-built ``search_agent``, ``reading_workflow`` and ``budget``.
  Used by ``integrated.py`` / ``minimal.py`` factories and scripts.

* ``AgenticM2Module`` — config-driven public wrapper loaded by
  ``ModuleRegistry`` when ``search.implementation == "agentic"``.
  Accepts ``llm_config`` + ``variant`` and builds the internal adapter
  via ``build_adapter_from_config()``.

Supplement rounds (cache-first incremental search)
--------------------------------------------------
When the pipeline re-enters M2 on a ``supplement_m2`` route
(``state.search_round > 0`` with at least one ``open`` evidence gap), the
adapter switches to an incremental flow instead of re-running the full
search:

1. **Cache hit first** — open gaps are probed against the persistent
   ``PaperStore`` (``memory_cache_dir``); hits are re-packaged as a
   paper-level export run (no fabricated knowledge entries — cached metadata
   carries no full-text evidence) and the gap moves to ``pending_grounding``.
2. **Gap search** — remaining gaps issue their ``suggested_queries`` through
   a one-shot ``IterativeSearchAgent`` built from the same components, with
   queries deduplicated against ``state.search_ledger`` and bounded by
   ``supplement_paper_budget``.
3. **Incremental merge** — new papers are merged into the existing
   ``literature_results`` / ``m2_knowledge_export`` (a *new* export run is
   appended so per-run provenance stays self-consistent), and the updated
   ``evidence_gaps`` / ``search_ledger`` are returned in full (LangGraph
   list fields have no reducer — the whole list must be re-emitted).

With no open gaps (or on round 0) the original full-flow path runs
unchanged.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..entity_normalization import EntityNormalizationService
from ..memory.paper_store import PaperStore, normalize_query_text, paper_key
from ..observability import emit_event
from ..protocol import ModuleProtocol
from ..state import (
    EvidenceGap,
    KnowledgeEntry,
    LiteratureResult,
    M2KnowledgeExport,
    M2KnowledgeRun,
    M2PaperExport,
    M2SearchProvenance,
    M2SearchQueryExport,
    PipelineState,
    SearchLedger,
)
from ..task_alignment import search_entities_for_sub_question
from .export import build_m2_knowledge_export_run
from .models import QueryIntent, SearchBudget, SearchQuery, StopReason
from .protocols import QueryPlannerProtocol, ReadingExtractionWorkflowProtocol
from .search import IterativeSearchAgent

logger = logging.getLogger(__name__)

_DEFAULT_SUPPLEMENT_PAPER_BUDGET = 6
_MAX_SUGGESTED_QUERIES_PER_GAP = 3
_WORD_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


# ============================================================================
# Supplement-round helpers
# ============================================================================


class _GapQueryPlanner(QueryPlannerProtocol):
    """One-shot planner that serves pre-built gap queries once, then stops."""

    def __init__(self, queries: Sequence[SearchQuery]) -> None:
        self._queries = list(queries)
        self._served = False

    async def plan(
        self,
        sub_question: str,
        key_entities: Sequence[str] = (),
        domains: Sequence[str] = (),
        question_type: str = "",
        state=None,
    ) -> List[SearchQuery]:
        if self._served:
            return []
        self._served = True
        return list(self._queries)


def _word_tokens(text: str) -> set:
    return set(_WORD_TOKEN_RE.findall(str(text or "").casefold()))


# ============================================================================
# Internal DI adapter (used by factories and scripts)
# ============================================================================


class AgenticM2Adapter(ModuleProtocol):
    """Internal dependency-injection adapter.

    Accepts pre-built *search_agent*, *reading_workflow* and *budget*.
    Not registered — loaded programmatically by factories or scripts.
    """

    module_name = "m2"
    module_version = "0.1.0-agentic-adapter"
    description = "Iterative literature search and evidence-linked reading adapter"

    def __init__(
        self,
        *,
        search_agent: IterativeSearchAgent,
        reading_workflow: ReadingExtractionWorkflowProtocol,
        budget: Optional[SearchBudget] = None,
        supplement_paper_budget: int = _DEFAULT_SUPPLEMENT_PAPER_BUDGET,
        entity_judge_client: Any = None,
        entity_embedding_model: str = "",
    ) -> None:
        self.search_agent = search_agent
        self.reading_workflow = reading_workflow
        self.budget = budget
        self.supplement_paper_budget = max(1, int(supplement_paper_budget))
        self.entity_judge_client = entity_judge_client
        self.entity_embedding_model = entity_embedding_model

    async def _normalise_reading_entities(
        self,
        state: PipelineState,
        reading_results: list,
    ) -> list:
        card = state.problem_card
        if card is None:
            return reading_results
        service = EntityNormalizationService.from_task(
            card.task_contract,
            domains=card.domain,
            cache_dir=state.entity_cache_dir or state.memory_cache_dir,
            client=self.entity_judge_client,
            embedding_model=self.entity_embedding_model,
        )
        names = [
            entity
            for reading in reading_results
            for entry in reading.knowledge_entries
            for entity in entry.entities
        ]
        resolved = await service.resolve_batch(names)
        output = []
        for reading in reading_results:
            entries = [
                entry.model_copy(update={
                    "entities": list(dict.fromkeys(
                        resolved[entity].canonical_name
                        if entity in resolved else entity
                        for entity in entry.entities
                    )),
                })
                for entry in reading.knowledge_entries
            ]
            output.append(reading.model_copy(update={"knowledge_entries": entries}))
        return output

    @staticmethod
    def _validate_reading_contract(
        sub_question: str,
        search_result: Any,
        reading_results: list,
    ) -> None:
        """Ensure the reading workflow produced at least one usable evidence item.

        Partial failures (some papers fail, others succeed) are acceptable.
        Total failure (all papers produce no citable evidence, no knowledge
        entries, or zero papers were retained) means the evidence contract
        for this sub-question is broken.
        """
        retained_count = len(search_result.final_papers)
        if retained_count == 0:
            raise RuntimeError(
                f"M2 retained zero papers for sub-question {sub_question!r}.  "
                f"The search completed but no papers met the retention criteria."
            )
        citable_count = sum(
            1 for r in reading_results
            if getattr(r, "evidence", None)
            and any(
                getattr(e, "citable", True)
                for e in r.evidence
            )
        )
        knowledge_count = sum(
            1 for r in reading_results
            if getattr(r, "knowledge_entries", None)
            and len(r.knowledge_entries) > 0
        )
        if retained_count > 0 and citable_count == 0:
            raise RuntimeError(
                f"M2 reading produced no citable evidence for sub-question "
                f"{sub_question!r} ({retained_count} retained paper(s), "
                f"0 with citable evidence)"
            )
        if retained_count > 0 and knowledge_count == 0:
            raise RuntimeError(
                f"M2 reading produced no knowledge entries for sub-question "
                f"{sub_question!r} ({retained_count} retained paper(s), "
                f"0 with knowledge entries)"
            )

    async def __call__(
        self,
        state: PipelineState,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        # ---- supplement round: cache-first incremental search -------------
        # Re-entry on a supplement_m2 route (search_round already advanced by
        # a previous M2 execution) with at least one open evidence gap.
        open_gaps = [gap for gap in state.evidence_gaps if gap.status == "open"]
        if open_gaps and state.search_round > 0:
            return await self._run_supplement(state, open_gaps)

        # ---- fresh round: full search flow (unchanged) ---------------------
        problem_card = state.problem_card
        pending_gaps = [
            gap for gap in state.evidence_gap_requests if gap.status == "pending"
        ]
        sub_questions = list(dict.fromkeys(
            gap.sub_question for gap in pending_gaps
        )) if pending_gaps else (
            problem_card.sub_questions
            if problem_card and problem_card.sub_questions
            else [state.input_question]
        )
        domains = problem_card.domain if problem_card else []
        question_type = problem_card.question_type.value if problem_card else ""

        literature_results: list[LiteratureResult] = []
        export_runs = []
        executed_queries: dict[str, list[str]] = {}
        for sub_question in sub_questions:
            key_entities = search_entities_for_sub_question(state, sub_question)
            search_result = await self.search_agent.run(
                sub_question,
                key_entities=key_entities,
                domains=domains,
                question_type=question_type,
                budget=self.budget,
            )
            if search_result.stop_reason is StopReason.ERROR:
                detail = "; ".join(search_result.errors) or "unrecoverable search error"
                raise RuntimeError(f"Agentic M2 search failed: {detail}")
            executed_queries[sub_question] = [
                query.text for query in search_result.queries
            ]

            reading_results = await self.reading_workflow.run(
                sub_question,
                search_result.final_papers,
                search_context=search_result,
            )
            reading_results = await self._normalise_reading_entities(
                state, list(reading_results)
            )
            self._validate_reading_contract(
                sub_question, search_result, list(reading_results),
            )
            export_run = build_m2_knowledge_export_run(
                sub_question,
                search_result,
                reading_results,
            )
            export_runs.append(export_run)
            literature_results.append(
                LiteratureResult(
                    sub_question=sub_question,
                    papers_retrieved=len(export_run.papers),
                    knowledge_entries=list(export_run.knowledge_entries),
                )
            )

        # Seed the paper cache (side effect only; never affects the return
        # patch) so later supplement rounds can hit it.
        if getattr(state, "memory_cache_dir", ""):
            self._persist_full_run(state, export_runs)

        # Follow-up search rounds are additive. Replacing these fields breaks
        # historical graph-ID resolution in M4 and causes repeated requests.
        merged_results: list[LiteratureResult] = []
        by_question: dict[str, LiteratureResult] = {}
        for result in [*state.literature_results, *literature_results]:
            existing = by_question.get(result.sub_question)
            if existing is None:
                copied = result.model_copy(deep=True)
                by_question[result.sub_question] = copied
                merged_results.append(copied)
                continue
            entries = {entry.id: entry for entry in existing.knowledge_entries}
            entries.update({entry.id: entry for entry in result.knowledge_entries})
            existing.knowledge_entries = list(entries.values())
            existing.papers_retrieved = max(
                existing.papers_retrieved, result.papers_retrieved
            )

        historical_runs = (
            list(state.m2_knowledge_export.runs)
            if state.m2_knowledge_export else []
        )
        result: Dict[str, Any] = {
            "literature_results": merged_results,
            "m2_knowledge_export": M2KnowledgeExport(
                runs=[*historical_runs, *export_runs]
            ),
        }
        if pending_gaps:
            pending_ids = {gap.gap_id for gap in pending_gaps}
            result["evidence_gap_requests"] = [
                gap.model_copy(update={
                    "status": "searched",
                    "attempts": gap.attempts + 1,
                    "executed_queries": list(dict.fromkeys([
                        *gap.executed_queries,
                        *executed_queries.get(gap.sub_question, []),
                    ])),
                }) if gap.gap_id in pending_ids else gap.model_copy(deep=True)
                for gap in state.evidence_gap_requests
            ]
            result["evidence_gap_search_rounds"] = (
                state.evidence_gap_search_rounds + 1
            )
        return result

    @classmethod
    def get_input_fields(cls) -> list[str]:
        return ["problem_card"]

    @classmethod
    def get_output_fields(cls) -> list[str]:
        return ["literature_results", "m2_knowledge_export"]

    # ------------------------------------------------------------------
    # Supplement round (cache-first incremental search)
    # ------------------------------------------------------------------

    async def _run_supplement(
        self,
        state: PipelineState,
        open_gaps: Sequence[EvidenceGap],
    ) -> Dict[str, Any]:
        problem_card = state.problem_card
        key_entities = problem_card.key_entities if problem_card else []
        domains = problem_card.domain if problem_card else []
        question_type = problem_card.question_type.value if problem_card else ""

        # Cache availability degrades gracefully: no memory_cache_dir → skip
        # the lookup step and go straight to searching.
        store: Optional[PaperStore] = None
        cache_dir = getattr(state, "memory_cache_dir", "")
        if cache_dir:
            try:
                store = PaperStore(cache_dir)
            except OSError as exc:
                logger.warning(
                    "M2 supplement: paper cache unavailable (%s); "
                    "falling back to live search",
                    exc,
                )
                store = None

        existing_runs = (
            list(state.m2_knowledge_export.runs) if state.m2_knowledge_export else []
        )
        known_keys = {
            paper_key(paper) for run in existing_runs for paper in run.papers
        }
        known_keys.update(state.search_ledger.paper_keys)
        issued_norms = {
            normalize_query_text(query)
            for query in state.search_ledger.queries_issued
        }

        merged_results = [
            result.model_copy(deep=True) for result in state.literature_results
        ]
        existing_sub_questions = [result.sub_question for result in merged_results]

        new_runs: List[M2KnowledgeRun] = []
        new_query_texts: List[str] = []
        new_paper_keys: List[str] = []
        attempted_gap_ids: set[str] = set()
        grounded_gap_ids: set[str] = set()  # gaps with actual evidence from live search
        remaining_budget = self.supplement_paper_budget

        for gap in open_gaps:
            sub_question = self._resolve_sub_question(gap, existing_sub_questions)

            # -- Step 1: cache hit first ----------------------------------
            if store is not None:
                cached_papers, cache_queries = self._cache_lookup(
                    store, gap, known_keys
                )
                if cached_papers:
                    hit_keys = [paper_key(paper) for paper in cached_papers]
                    new_runs.append(
                        self._cache_hit_run(
                            sub_question, gap, cache_queries, cached_papers, state
                        )
                    )
                    self._merge_increment(
                        merged_results, sub_question, len(cached_papers), []
                    )
                    known_keys.update(hit_keys)
                    new_paper_keys.extend(hit_keys)
                    attempted_gap_ids.add(gap.gap_id)
                    emit_event(
                        "memory_hit",
                        module="m2",
                        status="completed",
                        message=(
                            f"证据缺口 {gap.gap_id} 命中论文缓存 "
                            f"{len(cached_papers)} 篇"
                        ),
                        details={
                            "gap_id": gap.gap_id,
                            "hits": len(cached_papers),
                            "sub_question": sub_question,
                        },
                    )
                    for query in cache_queries:
                        store.record_query(
                            query,
                            hit_keys,
                            run_id=state.run_id,
                            round=state.search_round,
                        )
                    continue

            # -- Step 2: gap search ----------------------------------------
            attempted_gap_ids.add(gap.gap_id)
            fresh_queries = self._gap_queries(gap, issued_norms)
            if not fresh_queries or remaining_budget <= 0:
                continue

            search_result = await self._supplement_search(
                gap,
                sub_question,
                fresh_queries,
                key_entities=key_entities,
                domains=domains,
                question_type=question_type,
                paper_limit=remaining_budget,
            )
            new_query_texts.extend(fresh_queries)

            new_papers = []
            for paper in search_result.final_papers:
                key = paper_key(paper)
                if key in known_keys:
                    continue
                known_keys.add(key)
                new_papers.append(paper)

            hit_keys = [paper_key(paper) for paper in new_papers]
            if store is not None:
                for query in fresh_queries:
                    store.record_query(
                        query,
                        hit_keys,
                        run_id=state.run_id,
                        round=state.search_round,
                    )
            if not new_papers:
                continue

            filtered_result = search_result.model_copy(
                update={"final_papers": new_papers}
            )
            reading_results = await self.reading_workflow.run(
                sub_question,
                new_papers,
                search_context=filtered_result,
            )
            reading_results = await self._normalise_reading_entities(
                state, list(reading_results)
            )
            self._validate_reading_contract(
                sub_question, filtered_result, list(reading_results),
            )
            export_run = build_m2_knowledge_export_run(
                sub_question, filtered_result, reading_results
            )
            new_runs.append(export_run)
            grounded_gap_ids.add(gap.gap_id)
            self._merge_increment(
                merged_results,
                sub_question,
                len(new_papers),
                list(export_run.knowledge_entries),
            )
            new_paper_keys.extend(hit_keys)
            remaining_budget -= len(new_papers)
            if store is not None:
                store.upsert_papers(
                    new_papers, run_id=state.run_id, round=state.search_round
                )

        # -- Step 3: incremental merge & full-list return -------------------
        updated_gaps: List[EvidenceGap] = []
        for gap in state.evidence_gaps:
            if gap.gap_id in grounded_gap_ids and gap.status == "open":
                updated_gaps.append(
                    gap.model_copy(update={"status": "pending_grounding"})
                )
            else:
                updated_gaps.append(gap.model_copy(deep=True))

        ledger = SearchLedger(
            queries_issued=[*state.search_ledger.queries_issued, *new_query_texts],
            paper_keys=[*state.search_ledger.paper_keys, *new_paper_keys],
        )

        return {
            "literature_results": merged_results,
            "m2_knowledge_export": M2KnowledgeExport(
                runs=[*existing_runs, *new_runs]
            ),
            "evidence_gaps": updated_gaps,
            "search_ledger": ledger,
        }

    # -- supplement helpers -------------------------------------------------

    @staticmethod
    def _cache_lookup(
        store: PaperStore,
        gap: EvidenceGap,
        known_keys: set,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Probe the PaperStore with the gap's queries / description."""

        candidate_queries = [
            query.strip()
            for query in [*gap.suggested_queries, gap.description]
            if query and query.strip()
        ]
        hit_keys: List[str] = []
        seen: set = set()
        for query in candidate_queries:
            for key in store.lookup_query(query):
                if key not in seen and key not in known_keys:
                    seen.add(key)
                    hit_keys.append(key)
        papers = store.lookup_by_keys(hit_keys) if hit_keys else []
        return papers, candidate_queries

    @staticmethod
    def _cache_hit_run(
        sub_question: str,
        gap: EvidenceGap,
        cache_queries: Sequence[str],
        cached_papers: Sequence[Dict[str, Any]],
        state: PipelineState,
    ) -> M2KnowledgeRun:
        """Rebuild a paper-level export run from cached metadata.

        Cached metadata carries no full-text evidence, so the run contains
        papers only — knowledge entries are never fabricated; M3 grounding
        decides later whether the cached papers close the gap.
        """

        allowed_fields = set(M2PaperExport.model_fields)
        papers: List[M2PaperExport] = []
        for meta in cached_papers:
            try:
                papers.append(
                    M2PaperExport(
                        **{
                            key: value
                            for key, value in meta.items()
                            if key in allowed_fields
                        }
                    )
                )
            except Exception:
                logger.debug(
                    "M2 supplement: skipping malformed cached paper %r",
                    meta.get("paper_id", ""),
                )
        provenance = M2SearchProvenance(
            queries=[
                M2SearchQueryExport(
                    query_id=f"cache-{gap.gap_id}-{index + 1}",
                    text=query,
                    round_index=state.search_round,
                    intent="core",
                    target_source="paper_store",
                    purpose="cache lookup for evidence gap",
                    target_gap=gap.description,
                )
                for index, query in enumerate(cache_queries)
            ],
            iterations=0,
            stop_reason="cache_hit",
            papers_found=len(papers),
            papers_after_dedup=len(papers),
        )
        return M2KnowledgeRun(
            sub_question=sub_question,
            papers=papers,
            evidence=[],
            knowledge_entries=[],
            search_provenance=provenance,
        )

    @staticmethod
    def _gap_queries(gap: EvidenceGap, issued_norms: set) -> List[str]:
        """Candidate queries for a gap, deduplicated against issued ones.

        Uses ``suggested_queries`` when present; otherwise derives up to two
        rule-based queries from the description / canonical entities (no LLM
        round-trip on the supplement path).
        """

        candidates = [
            query.strip()
            for query in gap.suggested_queries
            if query and query.strip()
        ][:_MAX_SUGGESTED_QUERIES_PER_GAP]
        if not candidates:
            description = " ".join(gap.description.split())
            if description:
                candidates.append(description[:200])
            if gap.canonical_entities:
                candidates.append(" ".join(gap.canonical_entities))

        fresh: List[str] = []
        for query in candidates:
            normalized = normalize_query_text(query)
            if not normalized or normalized in issued_norms:
                continue
            issued_norms.add(normalized)
            fresh.append(query)
        return fresh

    def _source_names(self) -> List[str]:
        sources = getattr(self.search_agent, "sources", None)
        if isinstance(sources, dict):
            return list(sources)
        return []

    async def _supplement_search(
        self,
        gap: EvidenceGap,
        sub_question: str,
        query_texts: Sequence[str],
        *,
        key_entities: Sequence[str],
        domains: Sequence[str],
        question_type: str,
        paper_limit: int,
    ):
        """Run one bounded search round for a gap's queries."""

        source_names = self._source_names() or ["unknown"]
        queries: List[SearchQuery] = []
        for index, text in enumerate(query_texts):
            for source in source_names:
                queries.append(
                    SearchQuery(
                        query_id=f"sup-{gap.gap_id[:12]}-{index + 1}-{source}",
                        text=text,
                        intent=QueryIntent.CORE,
                        target_source=source,
                        purpose="supplement evidence gap",
                        target_gap=gap.description,
                        relation_to_question="supplement",
                    )
                )

        budget = SearchBudget(
            max_rounds=1,
            max_queries=max(1, len(queries)),
            max_papers=max(1, paper_limit),
        )

        base = self.search_agent
        if isinstance(base, IterativeSearchAgent):
            # Reuse the fully-configured components but swap in the one-shot
            # gap planner so the queries (and their ``target_gap``) are ours.
            agent = IterativeSearchAgent(
                query_planner=_GapQueryPlanner(queries),
                sources=list(base.sources.values()),
                deduplicator=base.deduplicator,
                ranker=base.ranker,
                scout_reader=base.scout_reader,
                coverage_evaluator=base.coverage_evaluator,
                final_k=min(base.final_k, max(1, paper_limit)),
                candidate_limit=base.candidate_limit,
                per_query_limit=base.per_query_limit,
                source_timeout_seconds=base.source_timeout_seconds,
                retention_judge_client=base.retention_judge_client,
            )
        else:
            agent = base  # DI fakes: degrade to a plain run() call

        result = await agent.run(
            sub_question,
            key_entities=key_entities,
            domains=domains,
            question_type=question_type,
            budget=budget,
        )
        if result.stop_reason is StopReason.ERROR:
            detail = (
                "; ".join(result.errors)
                or "unrecoverable supplement search error"
            )
            raise RuntimeError(f"Agentic M2 supplement search failed: {detail}")
        return result

    @staticmethod
    def _resolve_sub_question(gap: EvidenceGap, existing: Sequence[str]) -> str:
        """Target sub-question for a gap; falls back to best token overlap."""

        target = (gap.target_sub_question or "").strip()
        if target:
            return target
        if not existing:
            return " ".join(gap.description.split()) or "supplement search"
        tokens = _word_tokens(gap.description)
        best, best_score = existing[0], -1
        for candidate in existing:
            score = len(tokens & _word_tokens(candidate))
            if score > best_score:
                best, best_score = candidate, score
        return best

    @staticmethod
    def _merge_increment(
        merged_results: List[LiteratureResult],
        sub_question: str,
        paper_count: int,
        entries: Sequence[KnowledgeEntry],
    ) -> None:
        """Merge a supplement increment into the running literature results."""

        for result in merged_results:
            if result.sub_question == sub_question:
                result.papers_retrieved += paper_count
                seen_ids = {entry.id for entry in result.knowledge_entries}
                result.knowledge_entries.extend(
                    entry for entry in entries if entry.id not in seen_ids
                )
                return
        merged_results.append(
            LiteratureResult(
                sub_question=sub_question,
                papers_retrieved=paper_count,
                knowledge_entries=list(entries),
            )
        )

    def _persist_full_run(
        self,
        state: PipelineState,
        export_runs: Sequence[M2KnowledgeRun],
    ) -> None:
        """Seed the PaperStore after a fresh full search (best effort)."""

        try:
            store = PaperStore(state.memory_cache_dir)
            for run in export_runs:
                keys = store.upsert_papers(
                    run.papers, run_id=state.run_id, round=state.search_round
                )
                for query in run.search_provenance.queries:
                    store.record_query(
                        query.text,
                        keys,
                        run_id=state.run_id,
                        round=state.search_round,
                    )
        except Exception as exc:  # cache must never break the main flow
            logger.warning("M2: failed to seed paper cache: %s", exc)


# ============================================================================
# Public config-driven wrapper (loaded by ModuleRegistry)
# ============================================================================


class AgenticM2Module(ModuleProtocol):
    """Config-driven agentic M2 module.

    Loaded by ``ModuleRegistry`` when ``search.implementation == "agentic"``.
    Internally builds an ``AgenticM2Adapter`` via ``build_adapter_from_config()``
    and delegates all calls to it.
    """

    module_name = "m2"
    module_version = "0.2.0-agentic"
    description = "Agentic literature search and evidence-linked reading module"

    def __init__(
        self,
        llm_config: Optional[Any] = None,
        variant: str = "integrated",
        **kwargs,
    ) -> None:
        # ModuleRegistry injects these orchestration-only values, while the
        # Track A factories accept only concrete adapter construction args.
        kwargs.pop("implementation", None)
        kwargs.pop("query_llm_config", None)
        supplement_paper_budget = kwargs.pop("supplement_paper_budget", None)
        self.adapter = build_adapter_from_config(
            llm_config=llm_config,
            variant=variant,
            **kwargs,
        )
        if supplement_paper_budget is not None:
            self.adapter.supplement_paper_budget = max(
                1, int(supplement_paper_budget)
            )

    async def __call__(
        self,
        state: PipelineState,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return await self.adapter(state, config)

    @classmethod
    def get_input_fields(cls) -> list[str]:
        return ["problem_card"]

    @classmethod
    def get_output_fields(cls) -> list[str]:
        return ["literature_results", "m2_knowledge_export"]


# ============================================================================
# Factory
# ============================================================================


def build_adapter_from_config(
    *,
    llm_config: Optional[Any] = None,
    variant: str = "integrated",
    **kwargs,
) -> AgenticM2Adapter:
    """Build an ``AgenticM2Adapter`` from config parameters.

    Parameters
    ----------
    llm_config :
        LLM configuration for the Qwen client (required for ``variant="integrated"``).
    variant :
        ``"integrated"`` — full multi-source search + reading pipeline.
        ``"minimal"`` — PubMed-only rule-based pipeline (no LLM required).
    **kwargs :
        Forwarded to ``build_integrated_search_adapter()`` or
        ``build_minimal_pubmed_adapter()`` (e.g. *final_k*, *budget*,
        *source_timeout_seconds*).
    """
    if variant == "integrated":
        if llm_config is None:
            raise ValueError("variant='integrated' requires llm_config")
        # Lazy import to avoid circular dependency (integrated.py imports
        # AgenticM2Adapter from this module).
        from ..tools.qwen_client import QwenClient
        from .integrated import build_integrated_search_adapter

        client = QwenClient.from_config(llm_config)
        return _build_or_explain(
            build_integrated_search_adapter, variant, client=client, **kwargs
        )

    if variant == "minimal":
        from .minimal import build_minimal_pubmed_adapter
        return _build_or_explain(build_minimal_pubmed_adapter, variant, **kwargs)

    raise ValueError(
        f"Unknown variant {variant!r}; expected 'integrated' or 'minimal'"
    )


def _build_or_explain(factory, variant: str, **kwargs) -> AgenticM2Adapter:
    """Call *factory* and turn a cryptic ``unexpected keyword argument``
    ``TypeError`` into an actionable message.

    The agentic M2 factories declare explicit parameters (no ``**kwargs``),
    so a stray config key — typically a leftover *legacy* M2 kwarg such as
    ``mode`` / ``max_papers_per_query`` / ``batch_size`` — fails fast rather
    than being silently ignored.  We surface the offending key and the
    variant instead of the raw factory signature error.
    """
    try:
        return factory(**kwargs)
    except TypeError as exc:
        if "unexpected keyword argument" in str(exc):
            raise TypeError(
                f"Agentic M2 (variant={variant!r}) received an unsupported "
                f"config kwarg: {exc}. Check module_overrides.m2.kwargs — "
                f"legacy M2 keys (mode/max_papers_per_query/batch_size) and "
                f"search-budget keys (max_rounds/max_queries/max_papers) are "
                f"not accepted here; budget is set via the top-level `search:` "
                f"block."
            ) from exc
        raise
