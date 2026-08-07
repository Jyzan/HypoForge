"""
Pipeline orchestrator — builds the LangGraph StateGraph for M1→M6.

This is the heart of HypoForge: it wires modules together, adds
conditional iteration edges, and compiles the graph.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Literal

from langgraph.graph import END, StateGraph

from .config import PipelineConfig
from .display.panels import render_module_result, render_phase_done, render_phase_header
from .observability import RunEventRecorder, bind_recorder
from .protocol import ModuleProtocol
from .registry import ModuleRegistry, SkillRegistry
from .state import FollowupRequest, PipelineState, RoutingDecision, SearchLedger

logger = logging.getLogger(__name__)


class PipelineCancelled(Exception):
    """Raised by a node wrapper when a user-requested stop is pending.

    Cooperative cancellation: the currently executing module is always
    allowed to finish first; the *next* module refuses to start and raises
    this exception, which :meth:`PipelineRunner.run` catches to persist the
    accumulated state and emit the ``run_cancelled`` event.
    """

    def __init__(self, module: str) -> None:
        super().__init__(f"pipeline cancelled before module {module}")
        self.module = module


def _route_after_m6(
    state: PipelineState, config: Any
) -> Literal["end", "revise_m4", "supplement_m2"]:
    """Routing decision after M6 (v2 contract — priority order is strict).

    1. ``iteration_count >= max_iterations`` → ``"end"`` (the ONLY hard stop;
       supplement rounds also consume this M6 review budget);
    2. ``m6_evidence_revisit`` on **and** verdict present **and**
       ``sufficient == False`` **and** an ``open`` gap exists **and**
       ``search_round < max_search_rounds`` → ``"supplement_m2"``;
    3. evidence sufficient (verdict ``None`` — switch off / fail-open — or
       ``sufficient == True``) **and** this round's ``overall`` ≥
       ``review_score_threshold`` → ``"end"``;
    4. everything else → ``"revise_m4"`` (the legacy single-hop feedback loop).

    With the switch off the verdict is ``None``, rule 2 never fires and rule 3
    degenerates to the legacy threshold check, so default behaviour is exactly
    equivalent to the pre-refactor router.

    ``config`` may be ``None`` — feature switches are then treated as
    disabled, keeping the function usable as a pure routing oracle.
    """
    # 1. global hard stop — wins over everything
    if state.iteration_count >= state.max_iterations:
        return "end"

    # 2. evidence-insufficient supplement (wins over the score threshold)
    revisit = config is not None and getattr(config, "m6_evidence_revisit", False)
    if revisit:
        verdict = state.evidence_verdict
        has_open_gap = any(g.status == "open" for g in state.evidence_gaps)
        if (
            verdict is not None
            and not verdict.sufficient
            and has_open_gap
            and state.search_round < getattr(config, "max_search_rounds", 2)
        ):
            return "supplement_m2"

    # 3. evidence sufficient + quality threshold met → end.
    # The verdict is only meaningful while the switch is on; with it off any
    # (stale) verdict is ignored, keeping behaviour equivalent to legacy.
    verdict = state.evidence_verdict if revisit else None
    evidence_sufficient = verdict is None or verdict.sufficient
    recent = [r for r in state.reviews if r.version == state.iteration_count]
    overall = [r for r in recent if r.dimension.value == "overall"]
    if evidence_sufficient and overall and overall[0].score >= state.review_score_threshold:
        return "end"

    # 4. legacy feedback loop
    return "revise_m4"


def _route_after_m1(
    state: PipelineState, config: Any
) -> Literal["search_m2", "direct_m4"]:
    """Routing decision after M1 (v2 contract, followup search-free triage).

    ``followup`` present **and** ``followup.skip_search is True`` →
    ``"direct_m4"``; every other situation → ``"search_m2"`` (the legacy
    ``m1 → m2`` behaviour).

    Guards on top of the contract rule: the skip only fires while the
    ``followup_routing`` switch is on **and** a non-empty evidence graph
    exists to reuse — routing straight to M4 without any evidence artifact
    would produce an empty pipeline, so that degenerate case fails open to
    ``"search_m2"``.
    """
    if (
        config is not None
        and getattr(config, "followup_routing", False)
        and state.followup is not None
        and state.followup.skip_search is True
        and state.evidence_graph is not None
        and (state.evidence_graph.nodes or state.evidence_graph.edges)
    ):
        return "direct_m4"
    return "search_m2"


def _should_continue_iterating(state: PipelineState) -> str:
    """Back-compat thin wrapper over :func:`_route_after_m6`.

    Returns the legacy ``"end"`` / ``"iterate"`` vocabulary with all feature
    switches disabled; kept so existing tests and callers keep working.
    """
    return "end" if _route_after_m6(state, None) == "end" else "iterate"


# ---------------------------------------------------------------------------
# Follow-up (seed) run construction — v2 contract
# ---------------------------------------------------------------------------

# Whitelist of fields inherited from the parent run's final state.  Everything
# else is reset to its PipelineState default (explicitly — a parent state JSON
# may contain junk keys like ``_last_module``, so never splat it whole).
_SEED_INHERIT_FIELDS = (
    "problem_card",
    "literature_results",
    "m2_knowledge_export",
    "evidence_graph",
    "grounding_report",
    "best_hypotheses",
    "search_ledger",
    "memory_cache_dir",
)


def build_followup_seed(
    seed_state: dict,
    followup_text: str,
    run_id: str,
    config: PipelineConfig,
) -> PipelineState:
    """Build the initial state of a follow-up run from a parent final state.

    Pure function (no runner needed) so it is directly unit-testable.

    * **Inherited** (whitelist only): ``problem_card``, ``literature_results``,
      ``m2_knowledge_export``, ``evidence_graph``, ``grounding_report``,
      ``best_hypotheses``, ``search_ledger``, ``memory_cache_dir``;
    * **Reset** to defaults: counters (``iteration_count`` / ``revision_count``
      / ``search_round`` = 0), ``reviews`` / ``candidate_hypotheses`` /
      ``top_hypotheses`` / ``research_plans`` / ``errors`` /
      ``routing_history`` / ``evidence_gaps`` empty, ``evidence_verdict``
      ``None``, ``metrics`` {}, token counters 0;
    * **Set fresh**: ``input_question`` = *followup_text*, ``followup``
      (``FollowupRequest`` bound to the parent run id), ``parent_run_id``,
      ``run_id`` = *run_id*, ``max_iterations`` = ``config.max_iterations``
      (a fresh review budget).
    """
    inherited: Dict[str, Any] = {}
    for field in _SEED_INHERIT_FIELDS:
        value = seed_state.get(field)
        if value is not None:
            inherited[field] = value

    parent_run_id = str(seed_state.get("run_id", "") or "")
    memory_cache_dir = inherited.pop("memory_cache_dir", "") or config.memory_cache_dir

    return PipelineState(
        input_question=followup_text,
        run_id=run_id,
        parent_run_id=parent_run_id,
        followup=FollowupRequest(text=followup_text, parent_run_id=parent_run_id),
        max_iterations=config.max_iterations,
        memory_cache_dir=memory_cache_dir,
        **inherited,
    )


class PipelineRunner:
    """
    Builds and runs the HypoForge pipeline.

    Usage::

        config = PipelineConfig.from_yaml("configs/full_pipeline.yaml")
        runner = PipelineRunner(config)
        final_state = await runner.run("蛋白质如何折叠？")
    """

    _MODULE_TOOLS = {
        "m1": ["qwen_problem_understanding"],
        "m2": ["agentic_literature_pipeline"],
        "m3": ["rule_graph_builder", "qwen_relation_extractor"],
        "m4": ["hypothesis_generator", "hypothesis_ranker"],
        "m5": ["research_plan_designer"],
        "m6": ["specialist_reviewers", "overall_score_aggregator", "evidence_sufficiency_judge"],
    }

    def __init__(
        self,
        config: PipelineConfig,
        event_recorder: RunEventRecorder | None = None,
        cancel_event: Any = None,
    ):
        self.config = config
        self.event_recorder = event_recorder
        # Cooperative cancellation: a threading.Event-like object set by the
        # web layer while a run is in progress.  ``None`` in CLI / test runs.
        self.cancel_event = cancel_event
        # Set by ``run()`` when the run actually stopped due to cancellation.
        self.cancelled = False
        self._graph = None
        self._skills = None

    def _cancel_requested(self) -> bool:
        return self.cancel_event is not None and bool(
            self.cancel_event.is_set()
        )

    def _record_event(self, event_type: str, **kwargs: Any) -> None:
        if self.event_recorder is not None:
            self.event_recorder.emit(event_type, **kwargs)

    @staticmethod
    def _summarize_result(name: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Return a compact UI-friendly summary without duplicating full state."""

        summary: Dict[str, Any] = {"output_fields": sorted(result)}
        if name == "m1" and result.get("problem_card") is not None:
            card = result["problem_card"]
            summary.update(
                {
                    "sub_questions": len(card.sub_questions),
                    "domains": list(card.domain),
                    "key_entities": len(card.key_entities),
                }
            )
        elif name == "m2":
            literature = result.get("literature_results") or []
            export = result.get("m2_knowledge_export")
            summary.update(
                {
                    "sub_questions": len(literature),
                    "papers_retrieved": sum(item.papers_retrieved for item in literature),
                    "knowledge_entries": sum(
                        len(item.knowledge_entries) for item in literature
                    ),
                    "export_runs": len(export.runs) if export is not None else 0,
                }
            )
        elif name == "m3" and result.get("evidence_graph") is not None:
            graph = result["evidence_graph"]
            summary.update({"nodes": len(graph.nodes), "edges": len(graph.edges)})
        elif name == "m4":
            summary.update(
                {
                    "candidates": len(result.get("candidate_hypotheses") or []),
                    "top_hypotheses": len(result.get("top_hypotheses") or []),
                }
            )
        elif name == "m5":
            summary["research_plans"] = len(result.get("research_plans") or [])
        elif name == "m6":
            reviews = result.get("reviews") or []
            summary.update(
                {
                    "reviews": len(reviews),
                    "iteration_count": result.get("iteration_count"),
                    "overall_score": next(
                        (
                            review.score
                            for review in reversed(reviews)
                            if review.dimension.value == "overall"
                        ),
                        None,
                    ),
                }
            )
        return summary

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self) -> StateGraph:
        """Build (or return cached) the compiled StateGraph."""
        if self._graph is not None:
            return self._graph

        # --- instantiate modules ---
        # Trigger imports so @ModuleRegistry.register fires
        from . import modules as _  # noqa: F401

        # Trigger skill imports and instantiate fresh per-runner skills.
        from . import skills as _  # noqa: F401
        if self._skills is None:
            self._skills = SkillRegistry.build_enabled(self.config.enabled_skills)

        all_modules = ModuleRegistry.build_all(self.config)

        # Filter to enabled modules, preserving order
        enabled = self.config.enabled_modules
        missing = [name for name in enabled if name not in all_modules]
        if missing:
            raise ValueError(f"Unknown or unavailable enabled modules: {missing}")
        if not enabled:
            raise ValueError("enabled_modules must contain at least one module")
        active = {name: all_modules[name] for name in enabled if name in all_modules}

        # --- build graph ---
        workflow = StateGraph(PipelineState)

        for name, mod in active.items():
            workflow.add_node(name, self._make_node_wrapper(name, mod))

        # --- wire edges (linear, plus routing-aware conditional edges) ---
        # HARD ACCEPTANCE: with *both* feature switches off the topology must
        # be byte-for-byte the legacy one (linear spine + two-way m6 edge), so
        # conditional edges are only installed when routing is actually
        # enabled AND both routing targets exist.  The m1 followup edge needs
        # m2+m4 as targets; the m6 three-way edge needs m2+m4.  If either
        # target is missing from enabled_modules we keep the legacy wiring so
        # behaviour is unchanged.
        routing_enabled = bool(
            self.config.followup_routing or self.config.m6_evidence_revisit
        )
        routing_targets_available = routing_enabled and {"m2", "m4"} <= set(active)
        for i in range(len(enabled) - 1):
            src, dst = enabled[i], enabled[i + 1]
            if src == "m1" and dst == "m2" and routing_targets_available:
                workflow.add_conditional_edges(
                    src,
                    self._decide_after_m1,
                    {"search_m2": "m2", "direct_m4": "m4"},
                )
            else:
                workflow.add_edge(src, dst)

        # --- entry point ---
        workflow.set_entry_point(enabled[0])

        # --- conditional iteration edge ---
        last_module = enabled[-1]
        if self.config.enable_iteration and last_module == "m6":
            if routing_targets_available:
                workflow.add_conditional_edges(
                    last_module,
                    self._decide_after_m6,
                    {"end": END, "revise_m4": "m4", "supplement_m2": "m2"},
                )
            else:
                iteration_target = self.config.iteration_module_target
                workflow.add_conditional_edges(
                    last_module,
                    _should_continue_iterating,
                    {
                        "iterate": iteration_target,
                        "end": END,
                    },
                )
        else:
            workflow.add_edge(last_module, END)

        self._graph = workflow.compile()
        return self._graph

    def _make_node_wrapper(self, name: str, mod: ModuleProtocol):
        """Wrap a module callable so it prints Rich headers, handles errors,
        and runs Skill hooks before/after execution.

        Two additions on top of the vanilla wrapper:

        1. **Skip** — if the module's output fields are already populated in
           *state* (i.e. we are resuming from a checkpoint), the module is
           skipped and the existing state is returned unchanged.

        2. **Checkpoint** — after the module succeeds, the full state is
           persisted to disk so a later run can pick up from here.

        3. **Skills** — before the module runs, all enabled skills get their
           ``before()`` hook.  After the module runs, all enabled skills get
           their ``after()`` hook (with access to both pre- and post-state).
           Skill-returned patches are merged into the node's return value.
        """
        output_fields = set(mod.get_output_fields())

        # Resolve enabled skill instances once per node wrapper
        enabled_skills = self._skills or []

        class SkillHookError(RuntimeError):
            pass

        class SkillPatchError(ValueError):
            pass

        def validate_patch(patch: Any, hook: str) -> Dict[str, Any]:
            if patch is None:
                return {}
            if not isinstance(patch, dict):
                raise TypeError(f"Skill {hook} hook must return a dict patch")
            unknown = set(patch) - set(PipelineState.model_fields)
            if unknown:
                raise ValueError(
                    f"Skill {hook} hook returned unknown state fields: {sorted(unknown)}"
                )
            return patch

        async def run_hook(skill, hook: str, *args) -> Dict[str, Any]:
            try:
                return validate_patch(await getattr(skill, hook)(*args), hook)
            except Exception as exc:
                logger.exception("Skill %s %s hook failed", skill.skill_name, hook)
                if self.config.skill_fail_fast:
                    raise SkillHookError(str(exc)) from exc
                return {}

        async def node_fn(state: PipelineState) -> Dict:
            # --- Cancellation checkpoint (before module execution) ---
            # A pending stop request means this module must not start: the
            # accumulated state up to the previous module is final.
            if self._cancel_requested():
                self._record_event(
                    "module_cancelled",
                    module=name,
                    status="cancelled",
                    message=(
                        f"{name.upper()} 未执行：已收到停止请求，"
                        f"流程在当前成果处终止"
                    ),
                    details={"tools": self._MODULE_TOOLS.get(name, [])},
                )
                raise PipelineCancelled(name)

            # --- Resume: skip already-completed modules ---
            if output_fields and self._should_skip_module(name, output_fields, state):
                self._record_event(
                    "module_skipped",
                    module=name,
                    status="skipped",
                    message=f"{name.upper()} 已由 checkpoint 完成，跳过执行",
                    details={"tools": self._MODULE_TOOLS.get(name, [])},
                )
                if self.config.verbose:
                    from .display import console, COLORS
                    console.print(
                        f"  [{COLORS['muted']}][SKIP] [{name.upper()}] "
                        f"already complete (checkpoint resume)[/{COLORS['muted']}]"
                    )
                # Routing bookkeeping must happen even when the module is
                # skipped (resume), so the conditional edges and the recorded
                # history always agree.
                patch: Dict[str, Any] = {}
                self._append_routing(name, state, patch)
                return patch

            try:
                module_started_at = time.perf_counter()
                self._record_event(
                    "module_started",
                    module=name,
                    status="running",
                    message=f"{name.upper()} 开始执行",
                    details={
                        "description": mod.description,
                        "tools": self._MODULE_TOOLS.get(name, []),
                        "iteration_count": state.iteration_count,
                    },
                )
                # M4 collects interactive revision guidance before its phase
                # header, so the UI reads: guidance -> M4 -> revised output.
                pre_header_patch: Dict[str, Any] = {}
                collect_guidance = getattr(mod, "collect_user_guidance", None)
                if callable(collect_guidance):
                    collected = await collect_guidance(state)
                    if collected is not None:
                        if not isinstance(collected, dict):
                            raise TypeError(
                                f"Module {name} collect_user_guidance must return a dict patch"
                            )
                        unknown = set(collected) - set(PipelineState.model_fields)
                        if unknown:
                            raise ValueError(
                                f"Module {name} collect_user_guidance returned unknown "
                                f"state fields: {sorted(unknown)}"
                            )
                        pre_header_patch.update(collected)

                state_before_hooks_dict = state.model_dump(mode="python")
                state_before_hooks_dict.update(pre_header_patch)
                state_before_hooks = PipelineState(**state_before_hooks_dict)

                if self.config.verbose:
                    render_phase_header(name, mod.description)

                # --- before hooks ---
                before_patches: Dict = {}
                for skill in enabled_skills:
                    before_patches.update(
                        await run_hook(skill, "before", name, state_before_hooks)
                    )

                state_for_module_dict = state_before_hooks.model_dump(mode="python")
                state_for_module_dict.update(before_patches)
                state_for_module = PipelineState(**state_for_module_dict)

                # Snapshot state before module execution (for after hooks)
                state_before = state_for_module

                # --- execute module ---
                result = await mod(state_for_module)
                if result is None:
                    result = {}
                if not isinstance(result, dict):
                    raise TypeError(f"Module {name} must return a dict")
                unknown_result = set(result) - set(PipelineState.model_fields)
                if unknown_result:
                    raise ValueError(
                        f"Module {name} returned unknown state fields: {sorted(unknown_result)}"
                    )

                # --- build post-execution state for after hooks ---
                state_after_dict = state_for_module.model_dump(mode="python")
                state_after_dict.update(result)
                state_after = PipelineState(**state_after_dict)

                # --- after hooks ---
                after_patches: Dict = {}
                for skill in enabled_skills:
                    patch = await run_hook(
                        skill, "after", name, state_before, result, state_after
                    )
                    collisions = set(patch) & output_fields
                    if collisions:
                        raise SkillPatchError(
                            f"Skill {skill.skill_name} after hook cannot overwrite "
                            f"module output fields: {sorted(collisions)}"
                        )
                    after_patches.update(patch)

                # Merge all patches into the result
                final = {
                    **pre_header_patch,
                    **before_patches,
                    **result,
                    **after_patches,
                }

                # --- search-round bookkeeping: every *actual* M2 execution is
                # one round (independent of iteration_count).  M2 itself may
                # override by returning its own ``search_round``. ---
                if name == "m2" and "search_round" not in final:
                    final["search_round"] = state.search_round + 1
                if name == "m2" and "search_ledger" not in final:
                    # Keep the standalone M2 adapter's legacy two-field
                    # contract intact while recording first-round de-dup data
                    # when M2 runs inside the pipeline.
                    from .memory.paper_store import paper_key

                    export = final.get("m2_knowledge_export")
                    runs = getattr(export, "runs", []) if export is not None else []
                    issued_queries = [
                        query.text
                        for run in runs
                        for query in run.search_provenance.queries
                        if query.text
                    ]
                    issued_papers = [
                        paper_key(paper)
                        for run in runs
                        for paper in run.papers
                    ]
                    final["search_ledger"] = SearchLedger(
                        queries_issued=list(dict.fromkeys([
                            *state.search_ledger.queries_issued,
                            *issued_queries,
                        ])),
                        paper_keys=list(dict.fromkeys([
                            *state.search_ledger.paper_keys,
                            *issued_papers,
                        ])),
                    )

                # --- routing bookkeeping (m1/m6) ---
                # Recompute the post-module state exactly as the conditional
                # edge will see it, then append the RoutingDecision.
                post_state_dict = state_after.model_dump(mode="python")
                post_state_dict.update(after_patches)
                self._append_routing(name, PipelineState(**post_state_dict), final)

                if self.config.verbose:
                    render_module_result(name, state, final)
                    render_phase_done(name)
                # --- checkpoint: save state after each successful module ---
                self._save_checkpoint(name, state, final)
                elapsed = time.perf_counter() - module_started_at
                snapshot_path = None
                if self.event_recorder is not None:
                    merged_snapshot = state.model_dump(mode="json")
                    for key, value in final.items():
                        if hasattr(value, "model_dump"):
                            merged_snapshot[key] = value.model_dump(mode="json")
                        elif isinstance(value, list):
                            merged_snapshot[key] = [
                                item.model_dump(mode="json")
                                if hasattr(item, "model_dump")
                                else item
                                for item in value
                            ]
                        else:
                            merged_snapshot[key] = value
                    snapshot_path = self.event_recorder.save_snapshot(
                        f"{name}-r{state.search_round}-iter{state.iteration_count}",
                        merged_snapshot,
                    )
                details = self._summarize_result(name, final)
                if snapshot_path is not None:
                    details["snapshot"] = str(snapshot_path)
                self._record_event(
                    "module_completed",
                    module=name,
                    status="completed",
                    message=f"{name.upper()} 执行完成",
                    elapsed_seconds=elapsed,
                    details=details,
                )
                return final
            except Exception as exc:
                import traceback
                elapsed = (
                    time.perf_counter() - module_started_at
                    if "module_started_at" in locals()
                    else None
                )
                self._record_event(
                    "module_failed",
                    module=name,
                    status="failed",
                    message=f"{name.upper()} 执行失败：{type(exc).__name__}: {exc}",
                    elapsed_seconds=elapsed,
                )
                if self.config.verbose:
                    from .display import console, COLORS
                    console.print(f"  [{COLORS['error']}][ERR] [{name.upper()}] ERROR: {exc}[/{COLORS['error']}]")
                if isinstance(exc, (SkillHookError, SkillPatchError)):
                    raise
                return {"errors": state.errors + [f"[{name}] {exc}\n{traceback.format_exc()}"]}

        return node_fn

    @staticmethod
    def _is_module_done(state: PipelineState, output_fields: set) -> bool:
        """Return True if every output field already has a non-trivial value."""
        for field in output_fields:
            value = getattr(state, field, None)
            if value is None:
                return False
            if isinstance(value, (list, dict)) and len(value) == 0:
                return False
        return True

    def _should_skip_module(self, name: str, output_fields: set, state: PipelineState) -> bool:
        """Return True if *name* can be skipped (checkpoint resume).

        Routing-aware rules:

        * In followup runs with ``followup_routing`` enabled, M1 re-runs its
          triage unless this run's ``routing_history`` already records an m1
          decision (checkpoint resume after triage).
        * In a ``supplement_m2`` round (the latest routing decision routes
          m6 → m2), M2/M3 must re-run even if their outputs already exist.
        * For followup runs, M2/M3 re-run unless M1 explicitly decided
          ``skip_search=True`` (in which case they are skipped).
        * Modules in the iteration loop (M4/M5/M6) are NOT skipped while
          ``iteration_count < max_iterations``, because they may need to run
          additional rounds.
        * Everything else keeps the legacy "done ⇒ skip" semantics.
        """
        if not self._is_module_done(state, output_fields):
            return False

        if name == "m1" and state.followup is not None and self.config.followup_routing:
            # Followup exemption: the seed state inherits the parent run's
            # ``problem_card``, so the legacy "done ⇒ skip" rule would bypass
            # M1 entirely — but M1's followup triage
            # (``_understand_followup``) is the ONLY producer of
            # ``followup.skip_search``, and without it the ``direct_m4``
            # branch of ``_route_after_m1`` is unreachable.  Force a re-run
            # unless this run's routing_history already records an m1
            # decision (checkpoint resume after triage already executed).
            m1_decided = any(
                decision.from_module == "m1" for decision in state.routing_history
            )
            if m1_decided:
                return True  # triage already ran in this run (resume)
            logger.info("M1 forced to re-run for followup triage (skip suppressed)")
            self._record_event(
                "module_skip_suppressed",
                module="m1",
                status="running",
                message="M1 处于追问运行（followup），强制不跳过，重跑 triage",
                details={
                    "reason": "followup triage required",
                    "tools": self._MODULE_TOOLS.get("m1", []),
                },
            )
            return False

        if name in ("m2", "m3"):
            last_route = state.routing_history[-1] if state.routing_history else None
            if (
                last_route is not None
                and last_route.from_module == "m6"
                and last_route.to_module == "supplement_m2"
            ):
                # Supplement round: force fresh search / graph build.  A
                # supplement round is identified by the latest routing
                # decision (which implies search_round >= 1 by bookkeeping);
                # matching on the decision — rather than search_round alone —
                # keeps checkpoint-resume skip semantics intact.  Emit the
                # explanatory event/log required by the P2 skip rule: the
                # forced re-run applies even where a skip flag would
                # otherwise allow skipping.
                logger.info(
                    "%s forced to re-run in supplement round (skip suppressed)",
                    name.upper(),
                )
                self._record_event(
                    "module_skip_suppressed",
                    module=name,
                    status="running",
                    message=(
                        f"{name.upper()} 处于补搜轮（supplement_m2），"
                        f"强制不跳过，重新执行"
                    ),
                    details={
                        "reason": "supplement_m2 round",
                        "search_round": state.search_round,
                        "tools": self._MODULE_TOOLS.get(name, []),
                    },
                )
                return False  # supplement round: force fresh search / graph build
            if state.followup is not None:
                if self.config.followup_routing and state.followup.skip_search is True:
                    return True  # explicit skip_search → skip (module_skipped event)
                return False  # followup asks for a fresh search round
            return True  # done and nothing asks for a re-run

        # M2/M3 (and M1 outside followup runs) are safe to skip once done
        # (they only run once)
        iteration_modules = {"m4", "m5", "m6"}
        if name not in iteration_modules:
            return True

        # In the iteration loop, only skip if we've exhausted all iterations
        return state.iteration_count >= state.max_iterations

    # ------------------------------------------------------------------
    # Routing bookkeeping
    # ------------------------------------------------------------------

    def _decide_after_m6(self, state: PipelineState) -> str:
        """Conditional-edge callback: route after M6."""
        return _route_after_m6(state, self.config)

    def _decide_after_m1(self, state: PipelineState) -> str:
        """Conditional-edge callback: route after M1."""
        return _route_after_m1(state, self.config)

    def _routing_decision(self, name: str, state: PipelineState) -> RoutingDecision | None:
        """Compute the RoutingDecision produced after *name* (pure).

        ``round`` is a monotonically increasing audit counter:
        ``len(routing_history) + 1``.
        """
        round_no = len(state.routing_history) + 1
        if name == "m6":
            to_module = _route_after_m6(state, self.config)
            if to_module == "supplement_m2":
                decided_by: Literal["m6", "m1", "policy"] = "m6"
                reason = "evidence verdict insufficient — open gaps remain"
                gap_ids = [g.gap_id for g in state.evidence_gaps if g.status == "open"]
            elif to_module == "end":
                decided_by = "policy"
                reason = "iteration budget exhausted or review threshold met"
                gap_ids = []
            else:
                decided_by = "policy"
                reason = "revise hypotheses via the M4 feedback loop"
                gap_ids = []
            return RoutingDecision(
                round=round_no,
                from_module="m6",
                to_module=to_module,
                decided_by=decided_by,
                reason=reason,
                gap_ids=gap_ids,
            )
        if name == "m1":
            to_module = _route_after_m1(state, self.config)
            if to_module == "direct_m4":
                decided_by = "m1"
                reason = "followup adds no new entities/mechanisms/domains"
            else:
                decided_by = "policy"
                reason = "standard literature search path"
            return RoutingDecision(
                round=round_no,
                from_module="m1",
                to_module=to_module,
                decided_by=decided_by,
                reason=reason,
            )
        return None

    def _record_routing_event(self, decision: RoutingDecision) -> None:
        self._record_event(
            "routing_decision",
            module=decision.from_module,
            status="completed",
            message=(
                f"路由决策：{decision.from_module} → {decision.to_module}"
                f"（{decision.decided_by}）"
            ),
            details={
                "from": decision.from_module,
                "to": decision.to_module,
                "decided_by": decision.decided_by,
                "reason": decision.reason,
                "gap_ids": list(decision.gap_ids),
                "round": decision.round,
            },
        )

    def _append_routing(self, name: str, state: PipelineState, patch: Dict[str, Any]) -> None:
        """If *name* is a routing module, emit the routing_decision event and
        append the decision to ``routing_history`` via *patch*.

        ``state`` must be the post-module state (what the conditional edge
        sees); ``patch`` is the node return dict being assembled.  When M6
        routes to ``revise_m4`` this is also where ``revision_count`` gets
        incremented (audit-only counter).
        """
        if name not in ("m1", "m6"):
            return
        decision = self._routing_decision(name, state)
        if decision is None:
            return
        self._record_routing_event(decision)
        patch["routing_history"] = list(state.routing_history) + [decision]
        if name == "m6" and decision.to_module == "revise_m4":
            patch["revision_count"] = state.revision_count + 1
        if name == "m1" and decision.to_module == "direct_m4":
            # The conditional edge bypasses M2/M3 entirely, so their wrappers
            # never run — emit the explicit module_skipped events here to keep
            # the event stream consistent with the skip_search contract.
            for skipped in ("m2", "m3"):
                if skipped in self.config.enabled_modules:
                    self._record_event(
                        "module_skipped",
                        module=skipped,
                        status="skipped",
                        message=(
                            f"{skipped.upper()} 因追问免检索（skip_search）跳过执行"
                        ),
                        details={
                            "tools": self._MODULE_TOOLS.get(skipped, []),
                            "reason": "followup skip_search",
                        },
                    )

    def _checkpoint_path(self) -> str:
        """File path for the checkpoint JSON."""
        from pathlib import Path
        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        return str(out_dir / f"{self._current_run_id}_checkpoint.json")

    def _save_checkpoint(self, module_name: str, state: PipelineState, result: Dict) -> None:
        """Persist state after *module_name* completes."""
        import json
        from pydantic import BaseModel

        merged = state.model_dump(mode="json")
        # result values may be Pydantic models — serialise them too
        for key, value in result.items():
            if isinstance(value, BaseModel):
                merged[key] = value.model_dump(mode="json")
            elif isinstance(value, list) and value and isinstance(value[0], BaseModel):
                merged[key] = [v.model_dump(mode="json") for v in value]
            else:
                merged[key] = value
        merged["_last_module"] = module_name
        try:
            with open(self._checkpoint_path(), "w", encoding="utf-8") as fh:
                json.dump(merged, fh, ensure_ascii=False, indent=2)
        except OSError:
            pass  # non-critical

    def _load_checkpoint(self) -> dict | None:
        """Load a checkpoint dict if one exists, or None."""
        import json
        ckpt_path = self._checkpoint_path()
        try:
            with open(ckpt_path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    async def run(
        self,
        question: str,
        run_id: str = "",
        resume: bool = False,
        followup_text: str = "",
        seed_state: dict | None = None,
    ) -> PipelineState:
        """
        Execute the full pipeline for a given scientific question.

        Parameters
        ----------
        question : str
            The frontier scientific question (e.g. from Science 125).
        run_id : str
            Optional identifier for this run (auto-generated if empty).
        resume : bool
            If True, attempt to load a checkpoint from a previous run with
            the same *run_id*.  Already-completed modules are skipped.
        followup_text : str
            Follow-up question text.  Only meaningful together with
            *seed_state*; it overrides *question* as the input question
            (falls back to *question* when empty).
        seed_state : dict | None
            Final state (dict / loaded output JSON) of a previous run.  When
            provided, this run is a **follow-up run**: whitelisted M2/M3
            knowledge artifacts are inherited, everything else resets, and a
            fresh iteration budget is granted (see :func:`build_followup_seed`).

        Returns
        -------
        PipelineState
            The final state after all modules (and iterations) have completed.
        """
        import uuid

        if not run_id:
            run_id = f"hypoforge-{uuid.uuid4().hex[:8]}"
        self._current_run_id = run_id
        self.cancelled = False
        run_started_at = time.perf_counter()
        self._record_event(
            "run_started",
            status="running",
            message="HypoForge M1-M6 流程开始",
            details={
                "question": question,
                "enabled_modules": list(self.config.enabled_modules),
                "model": self.config.qwen.plus.model,
            },
        )

        graph = self._build_graph()

        # ---- initial state: followup run > checkpoint resume > fresh ----
        checkpoint = self._load_checkpoint() if resume else None
        if seed_state is not None:
            initial_state = build_followup_seed(
                seed_state=seed_state,
                followup_text=followup_text or question,
                run_id=run_id,
                config=self.config,
            )
            question = initial_state.input_question
            if self.config.verbose:
                from .display import console, COLORS
                console.print()
                console.print(
                    f"  [{COLORS['success']}][FOLLOWUP] Built on parent run "
                    f"[bold]{initial_state.parent_run_id}[/bold] — M2/M3 knowledge "
                    f"artifacts inherited, M4-M6 will re-run"
                    f"[/{COLORS['success']}]"
                )
        elif checkpoint:
            last_module = checkpoint.pop("_last_module", "?")
            initial_state = PipelineState(**checkpoint)
            if self.config.verbose:
                from .display import console, COLORS
                console.print()
                console.print(
                    f"  [{COLORS['success']}][RESUME] Loaded checkpoint from "
                    f"[bold]{last_module}[/bold] — {len(checkpoint)} state fields restored"
                    f"[/{COLORS['success']}]"
                )
        else:
            initial_state = PipelineState(
                input_question=question,
                run_id=run_id,
                max_iterations=self.config.max_iterations,
                memory_cache_dir=self.config.memory_cache_dir,
            )

        # Iteration stop-threshold comes from config (single source of truth),
        # applied on both fresh and resumed runs.
        initial_state.review_score_threshold = self.config.scoring.review_threshold

        if self.config.verbose:
            from .display import console, COLORS
            from rich.panel import Panel
            console.print()
            console.print(Panel(
                f"[bold]{question}[/bold]",
                title=f"[bold {COLORS['primary']}]HypoForge Pipeline — {run_id}",
                border_style=COLORS["primary"],
            ))

        # Stream through the graph.  A pending cancellation stops the run
        # cooperatively: the module currently executing finishes, the next
        # node wrapper raises PipelineCancelled, and the state accumulated
        # up to here becomes the run's final (persisted) state.
        final_state_dict = None
        with bind_recorder(self.event_recorder):
            try:
                async for chunk in graph.astream(
                    initial_state,
                    stream_mode="values",
                    config={"recursion_limit": 100},
                ):
                    final_state_dict = chunk
            except PipelineCancelled as exc:
                self.cancelled = True
                self._cancelled_before_module = getattr(exc, "module", "")

        if final_state_dict is None:
            if self.cancelled:
                # Cancelled before any module completed — keep whatever the
                # seed/initial state already holds so followups still work.
                final_state_dict = initial_state.model_dump(mode="python")
            else:
                raise RuntimeError("Pipeline produced no output.")

        # Reconstruct PipelineState from the final dict
        final_state = PipelineState(**final_state_dict)

        # ---- populate token stats from QwenClient ----
        from .tools.qwen_client import QwenClient
        final_state.total_input_tokens, final_state.total_output_tokens = QwenClient.get_token_totals()
        QwenClient.reset_token_totals()

        if self.cancelled:
            # Graceful stop: persist the accumulated state (problem card,
            # literature knowledge, evidence graph, …) so this run can serve
            # as the parent of a follow-up run, then end without scoring.
            self._record_event(
                "run_cancelled",
                status="cancelled",
                message=(
                    "流程已被手动停止：后续模块不再执行，"
                    "已完成模块的成果已保留，可作为后续追问的父运行"
                ),
                elapsed_seconds=time.perf_counter() - run_started_at,
                details={
                    "question": question,
                    "stopped_before": getattr(
                        self, "_cancelled_before_module", ""
                    ),
                },
            )
            self._save_output(final_state)
            return final_state

        # ---- final summary ----
        if self.config.verbose:
            from .display.panels import render_final_summary
            render_final_summary(final_state)

        # ---- save output ----
        self._save_output(final_state)

        # ---- automated scoring report (single source: config.scoring) ----
        if self.config.scoring.auto_score:
            from .evaluation.scorer import save_scoring_report_async
            # The independent metrics use a lightweight (turbo) model for
            # LLM-as-judge evaluations so they don't add meaningful latency.
            metric_llm_config = self.config.get_llm_for_tier("turbo")
            try:
                score_started_at = time.perf_counter()
                self._record_event(
                    "scoring_started",
                    module="m6",
                    tool="posthoc_scorer",
                    status="running",
                    message="独立评分开始",
                )
                scores_path = await save_scoring_report_async(
                    final_state,
                    self.config.output_dir,
                    self.config.scoring.hypothesis_weights,
                    llm_config=metric_llm_config,
                    embed_config=self.config.evaluation.model_dump(),
                )
                self._record_event(
                    "scoring_completed",
                    module="m6",
                    tool="posthoc_scorer",
                    status="completed",
                    message="独立评分完成",
                    elapsed_seconds=time.perf_counter() - score_started_at,
                    details={"scores_path": str(scores_path)},
                )
                if self.config.verbose:
                    from .display import console, COLORS
                    console.print(
                        f"  [{COLORS['muted']}]Scores saved to {scores_path}[/{COLORS['muted']}]"
                    )
            except Exception as exc:
                import logging
                logging.getLogger(__name__).warning("Scoring report failed: %s", exc)
                self._record_event(
                    "scoring_failed",
                    module="m6",
                    tool="posthoc_scorer",
                    status="failed",
                    message=f"独立评分失败：{type(exc).__name__}: {exc}",
                )

        self._record_event(
            "run_completed",
            status="completed" if not final_state.errors else "completed_with_errors",
            message="HypoForge M1-M6 流程结束",
            elapsed_seconds=time.perf_counter() - run_started_at,
            details={
                "errors": len(final_state.errors),
                "input_tokens": final_state.total_input_tokens,
                "output_tokens": final_state.total_output_tokens,
            },
        )

        return final_state

    # ------------------------------------------------------------------
    # Output serialisation
    # ------------------------------------------------------------------

    def _save_output(self, state: PipelineState) -> None:
        """Persist the final state as JSON in the output directory."""
        import json
        from pathlib import Path

        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / f"{state.run_id}.json"
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(state.model_dump(mode="json"), fh, ensure_ascii=False, indent=2)

        if self.config.verbose:
            from .display import console, COLORS
            console.print(f"  [{COLORS['muted']}]Output saved to {out_path}[/{COLORS['muted']}]")
