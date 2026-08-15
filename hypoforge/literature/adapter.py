"""Agentic M2 module — config-driven wrapper with internal DI adapter.

Track A delivers two classes:

* ``AgenticM2Adapter`` — internal dependency-injection adapter.  Accepts
  pre-built ``search_agent``, ``reading_workflow`` and ``budget``.
  Used by ``integrated.py`` / ``minimal.py`` factories and scripts.

* ``AgenticM2Module`` — Literature-layer configuration wrapper used by
  the Pipeline-facing facade in ``hypoforge.modules.m2_literature_search``.
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
from ..task_alignment import (
    search_entities_for_sub_question,
    sub_question_entity_terms,
)
from .export import build_m2_knowledge_export_run
from .models import (
    PaperRetentionDecision,
    QueryIntent,
    SearchBudget,
    SearchQuery,
    StopReason,
)
from .protocols import QueryPlannerProtocol, ReadingExtractionWorkflowProtocol
from .search import IterativeSearchAgent
from .search.ranking import rerank_with_scout
from .subquestion_entities import generate_subquestion_entities

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
        supplement_entities: Sequence[str] = (),
        focus_entities: Sequence[str] = (),
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
        subquestion_entity_client: Any = None,
        round_strategy: str = "entity_group",
        fulltext_backfill_enabled: bool = False,
        fulltext_backfill_target: int = 0,
        fulltext_backfill_max_attempts: int = 8,
        fulltext_backfill_min_relevance: float = 0.70,
        fulltext_backfill_min_directness: float = 0.45,
    ) -> None:
        self.search_agent = search_agent
        self.reading_workflow = reading_workflow
        self.budget = budget
        self.supplement_paper_budget = max(1, int(supplement_paper_budget))
        self.entity_judge_client = entity_judge_client
        self.entity_embedding_model = entity_embedding_model
        # M2-local supplementary entity generation (search-only scope; the
        # results never leave the M2 search data flow).
        self.subquestion_entity_client = subquestion_entity_client
        # Round organisation for the primary search flow.  "entity_group"
        # uses the must/unmust grouping strategy + zero-result rescue;
        # "coverage" keeps the legacy coverage-driven iteration.  The
        # supplement (gap) path always stays on the legacy one-shot flow.
        self.round_strategy = round_strategy
        if fulltext_backfill_target < 0 or fulltext_backfill_max_attempts <= 0:
            raise ValueError("full-text backfill limits are invalid")
        if not 0 <= fulltext_backfill_min_relevance <= 1:
            raise ValueError("fulltext_backfill_min_relevance must be in [0, 1]")
        if not 0 <= fulltext_backfill_min_directness <= 1:
            raise ValueError("fulltext_backfill_min_directness must be in [0, 1]")
        self.fulltext_backfill_enabled = bool(fulltext_backfill_enabled)
        self.fulltext_backfill_target = int(fulltext_backfill_target)
        self.fulltext_backfill_max_attempts = int(fulltext_backfill_max_attempts)
        self.fulltext_backfill_min_relevance = float(fulltext_backfill_min_relevance)
        self.fulltext_backfill_min_directness = float(fulltext_backfill_min_directness)

    @staticmethod
    def _is_parsed_fulltext(reading: Any) -> bool:
        level = getattr(getattr(reading, "content_level", None), "value", "")
        return level in {"structured_fulltext", "pdf", "html", "ocr"} and int(
            getattr(reading, "chunks_parsed", 0) or 0
        ) > 0

    async def _backfill_fulltext(
        self,
        sub_question: str,
        search_result: Any,
        reading_results: list,
        *,
        citation_floor: int = 0,
        attempted_ids: set[str] | None = None,
    ) -> tuple[Any, list]:
        """Try lower-ranked strict candidates until the parsed-text target is met."""

        if not self.fulltext_backfill_enabled:
            return search_result, reading_results
        target = self.fulltext_backfill_target or self.search_agent.final_k
        fulltext_count = sum(self._is_parsed_fulltext(item) for item in reading_results)
        if fulltext_count >= target:
            return search_result, reading_results

        selected_ids = {paper.paper_id for paper in search_result.final_papers}
        attempted = attempted_ids if attempted_ids is not None else set()
        note_by_id = {note.paper_id: note for note in search_result.scout_notes}
        candidates = []
        for paper in search_result.candidates:
            if paper.paper_id in selected_ids:
                continue
            if paper.paper_id in attempted:
                continue
            if int(paper.citation_count or 0) < citation_floor:
                continue
            if float(paper.rank_scores.get("semantic_not_applicable", 0.0)) >= 0.5:
                continue
            note = note_by_id.get(paper.paper_id)
            relevance = float(
                note.relevance_to_question if note is not None
                else paper.rank_scores.get("query_relevance", 0.0)
            )
            directness = float(
                (note.directness_to_question if note is not None else relevance) or 0.0
            )
            if (
                relevance < self.fulltext_backfill_min_relevance
                or directness < self.fulltext_backfill_min_directness
            ):
                continue
            candidates.append(paper)
        candidates.sort(key=lambda paper: (
            -int(paper.citation_count or 0),
            -float(paper.rank_scores.get("post_scout_total", 0.0)),
            paper.paper_id,
        ))

        retained = list(search_result.final_papers)
        readings = list(reading_results)
        attempts = 0
        decisions = list(search_result.retention_decisions)
        for paper in candidates:
            if fulltext_count >= target or attempts >= self.fulltext_backfill_max_attempts:
                break
            attempts += 1
            attempted.add(paper.paper_id)
            probe = getattr(self.reading_workflow, "probe_parsed_fulltext", None)
            if callable(probe) and not await probe(paper):
                continue
            one_paper_context = search_result.model_copy(update={"final_papers": [paper]})
            result = list(await self.reading_workflow.run(
                sub_question,
                [paper],
                search_context=one_paper_context,
            ))
            if not result or not self._is_parsed_fulltext(result[0]):
                continue
            retained.append(paper)
            readings.extend(result)
            selected_ids.add(paper.paper_id)
            fulltext_count += 1
            decisions.append(PaperRetentionDecision(
                paper_id=paper.paper_id,
                decision="retain",
                roles=["fulltext_backfill"],
                reason=(
                    "strictly relevant candidate retained after successful "
                    "open-fulltext download and parsing"
                ),
                rank_position=(
                    next(
                        index for index, candidate in enumerate(
                            search_result.candidates, 1
                        ) if candidate.paper_id == paper.paper_id
                    )
                ),
            ))

        if attempts:
            emit_event(
                "tool_result",
                module="m2",
                tool="fulltext_backfill",
                status="completed" if fulltext_count >= target else "warning",
                message=(
                    f"全文回填完成：{fulltext_count}/{target}，"
                    f"尝试候选 {attempts} 篇"
                ),
                details={
                    "sub_question": sub_question,
                    "target": target,
                    "parsed_fulltexts": fulltext_count,
                    "attempts": attempts,
                },
            )
        return search_result.model_copy(update={
            "final_papers": retained,
            "retention_decisions": decisions,
        }), readings

    def _one_round_budget(self) -> SearchBudget:
        base = self.budget or SearchBudget()
        return base.model_copy(update={"max_rounds": 1})

    async def _merge_search_phases(self, compact: Any, expanded: Any) -> Any:
        """Merge compact and conditional expansion results without double counts."""

        papers = await self.search_agent.deduplicator.deduplicate([
            *compact.candidates,
            *expanded.candidates,
        ])
        ranked = await self.search_agent.ranker.rank(
            compact.sub_question,
            papers,
            limit=min(self.search_agent.candidate_limit, len(papers)),
        )
        note_by_id = {
            note.paper_id: note
            for note in [*compact.scout_notes, *expanded.scout_notes]
        }
        notes = [
            note_by_id[paper.paper_id]
            for paper in ranked if paper.paper_id in note_by_id
        ]
        ranked = rerank_with_scout(
            ranked,
            notes,
            selection_limit=self.search_agent.final_k,
        )
        counts = dict(compact.source_result_counts)
        for source, count in expanded.source_result_counts.items():
            counts[source] = counts.get(source, 0) + count
        return compact.model_copy(update={
            "queries": [*compact.queries, *expanded.queries],
            "papers_found": compact.papers_found + expanded.papers_found,
            "papers_after_dedup": len(papers),
            "candidates": ranked,
            "failed_sources": list(dict.fromkeys([
                *compact.failed_sources, *expanded.failed_sources,
            ])),
            "iterations": compact.iterations + expanded.iterations,
            "errors": [*compact.errors, *expanded.errors],
            "source_result_counts": counts,
            "scout_notes": list(note_by_id.values()),
        })

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
            # Per-run cache isolation: one stable stamp (the pipeline run_id,
            # fixed at run start) shared by every construction in this run.
            run_stamp=state.run_id,
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
        Total reading failure after papers were retained remains an error.
        Zero retained papers is represented as an explicit evidence gap rather
        than crashing all other sub-questions in M2.
        """
        retained_count = len(search_result.final_papers)
        if retained_count == 0:
            return
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
        literature_results: list[LiteratureResult] = []
        export_runs = []
        executed_queries: dict[str, list[str]] = {}
        for sub_question in sub_questions:
            key_entities = search_entities_for_sub_question(state, sub_question)
            # M2-local entity generation: the LLM sees this sub-question's
            # M1 entities and may add uncovered search concepts.  Results
            # are search-scoped only — never written back to the problem
            # card / task contract.
            supplement_entities = await generate_subquestion_entities(
                self.subquestion_entity_client,
                sub_question,
                existing_terms=[
                    *sub_question_entity_terms(state, sub_question),
                    *key_entities,
                ],
                domains=domains,
            )
            search_result = await self.search_agent.run(
                sub_question,
                key_entities=key_entities,
                domains=domains,
                # The validated benchmark used one compact discovery round,
                # then expanded only when parsed full text was insufficient.
                budget=(
                    self._one_round_budget()
                    if self.fulltext_backfill_enabled else self.budget
                ),
                supplement_entities=supplement_entities,
                round_strategy=self.round_strategy,
            )
            if search_result.stop_reason is StopReason.ERROR:
                detail = "; ".join(search_result.errors) or "unrecoverable search error"
                raise RuntimeError(f"Agentic M2 search failed: {detail}")
            executed_queries[sub_question] = [
                query.text for query in search_result.queries
            ]

            if not search_result.final_papers:
                emit_event(
                    "tool_result",
                    module="m2",
                    tool="evidence_gap",
                    status="warning",
                    message="本子问题未保留论文，记录证据缺口并继续其他子问题",
                    details={
                        "sub_question": sub_question,
                        "papers_found": search_result.papers_found,
                        "stop_reason": (
                            search_result.stop_reason.value
                            if search_result.stop_reason is not None else None
                        ),
                        "errors": list(search_result.errors),
                    },
                )

            reading_results = await self.reading_workflow.run(
                sub_question,
                search_result.final_papers,
                search_context=search_result,
            )
            attempted_backfill_ids: set[str] = set()
            search_result, reading_results = await self._backfill_fulltext(
                sub_question,
                search_result,
                list(reading_results),
                citation_floor=100,
                attempted_ids=attempted_backfill_ids,
            )
            target = self.fulltext_backfill_target or int(
                getattr(self.search_agent, "final_k", 1)
            )
            parsed_count = sum(
                self._is_parsed_fulltext(item) for item in reading_results
            )
            if self.fulltext_backfill_enabled and parsed_count < target:
                expanded = await self.search_agent.run(
                    sub_question,
                    key_entities=key_entities,
                    domains=domains,
                    question_type="open_access_expansion",
                    budget=self._one_round_budget(),
                    existing_papers=search_result.candidates,
                    supplement_entities=supplement_entities,
                    round_strategy="coverage",
                )
                if expanded.candidates:
                    search_result = await self._merge_search_phases(
                        search_result, expanded
                    )
                    for citation_floor in (100, 20, 0):
                        search_result, reading_results = (
                            await self._backfill_fulltext(
                                sub_question,
                                search_result,
                                list(reading_results),
                                citation_floor=citation_floor,
                                attempted_ids=attempted_backfill_ids,
                            )
                        )
                        if sum(
                            self._is_parsed_fulltext(item)
                            for item in reading_results
                        ) >= target:
                            break
                else:
                    emit_event(
                        "tool_result",
                        module="m2",
                        tool="expanded_open_fulltext_discovery",
                        status="warning",
                        message="扩展检索未返回候选，继续使用紧凑轮结果",
                        details={
                            "sub_question": sub_question,
                            "errors": list(expanded.errors),
                        },
                    )
            executed_queries[sub_question] = [
                query.text for query in search_result.queries
            ]
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
            emit_event(
                "tool_result",
                module="m2",
                tool="m2_export",
                status="completed",
                message="M2 to M3 evidence export completed",
                details={
                    "sub_question": sub_question,
                    "papers": len(export_run.papers),
                    "evidence": len(export_run.evidence),
                    "knowledge_entries": len(export_run.knowledge_entries),
                },
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
                scout_candidate_limit=base.scout_candidate_limit,
                scout_timeout_seconds=base.scout_timeout_seconds,
                source_timeout_seconds=base.source_timeout_seconds,
                retention_judge_client=base.retention_judge_client,
            )
        else:
            agent = base  # DI fakes: degrade to a plain run() call

        result = await agent.run(
            sub_question,
            key_entities=key_entities,
            domains=domains,
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
# Literature-layer config-driven wrapper
# ============================================================================


class AgenticM2Module(ModuleProtocol):
    """Literature-layer configuration wrapper.

    The Pipeline-facing facade lives in
    ``hypoforge.modules.m2_literature_search``. This class builds an
    ``AgenticM2Adapter`` from configuration and delegates calls to it.
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
