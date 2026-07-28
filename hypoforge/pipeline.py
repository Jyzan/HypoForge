"""
Pipeline orchestrator — builds the LangGraph StateGraph for M1→M6.

This is the heart of HypoForge: it wires modules together, adds
conditional iteration edges, and compiles the graph.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict

from langgraph.graph import END, StateGraph

from .config import PipelineConfig
from .display.panels import render_module_result, render_phase_done, render_phase_header
from .observability import RunEventRecorder, bind_recorder
from .protocol import ModuleProtocol
from .registry import ModuleRegistry, SkillRegistry
from .state import PipelineState

logger = logging.getLogger(__name__)


def _should_continue_iterating(state: PipelineState) -> str:
    """Determine whether to iterate (M6 → M4) or end the pipeline."""
    if state.iteration_count >= state.max_iterations:
        return "end"

    # Check latest overall review score
    recent = [r for r in state.reviews if r.version == state.iteration_count]
    overall = [r for r in recent if r.dimension.value == "overall"]
    if overall and overall[0].score >= state.review_score_threshold:
        return "end"  # quality threshold met

    return "iterate"


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
        "m6": ["specialist_reviewers", "overall_score_aggregator"],
    }

    def __init__(
        self,
        config: PipelineConfig,
        event_recorder: RunEventRecorder | None = None,
    ):
        self.config = config
        self.event_recorder = event_recorder
        self._graph = None
        self._skills = None

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

        # --- wire linear edges ---
        for i in range(len(enabled) - 1):
            workflow.add_edge(enabled[i], enabled[i + 1])

        # --- entry point ---
        workflow.set_entry_point(enabled[0])

        # --- conditional iteration edge ---
        last_module = enabled[-1]
        if self.config.enable_iteration and last_module == "m6":
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
                return {}

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
                        f"{name}-iteration-{state.iteration_count}",
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

        Modules in the iteration loop (M4/M5/M6) are NOT skipped when
        ``iteration_count < max_iterations``, because they may need to
        run additional rounds.
        """
        if not self._is_module_done(state, output_fields):
            return False

        # M1–M3 always safe to skip once done (they only run once)
        iteration_modules = {"m4", "m5", "m6"}
        if name not in iteration_modules:
            return True

        # In the iteration loop, only skip if we've exhausted all iterations
        return state.iteration_count >= state.max_iterations

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

    async def run(self, question: str, run_id: str = "", resume: bool = False) -> PipelineState:
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

        Returns
        -------
        PipelineState
            The final state after all modules (and iterations) have completed.
        """
        import uuid

        if not run_id:
            run_id = f"hypoforge-{uuid.uuid4().hex[:8]}"
        self._current_run_id = run_id
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

        # ---- checkpoint resume ----
        checkpoint = self._load_checkpoint() if resume else None
        if checkpoint:
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

        # Stream through the graph
        final_state_dict = None
        with bind_recorder(self.event_recorder):
            async for chunk in graph.astream(
                initial_state,
                stream_mode="values",
                config={"recursion_limit": 100},
            ):
                final_state_dict = chunk

        if final_state_dict is None:
            raise RuntimeError("Pipeline produced no output.")

        # Reconstruct PipelineState from the final dict
        final_state = PipelineState(**final_state_dict)

        # ---- populate token stats from QwenClient ----
        from .tools.qwen_client import QwenClient
        final_state.total_input_tokens, final_state.total_output_tokens = QwenClient.get_token_totals()
        QwenClient.reset_token_totals()

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
