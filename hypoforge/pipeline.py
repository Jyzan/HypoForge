"""
Pipeline orchestrator — builds the LangGraph StateGraph for M1→M6.

This is the heart of HypoForge: it wires modules together, adds
conditional iteration edges, and compiles the graph.
"""

from __future__ import annotations

from typing import Dict

from langgraph.graph import END, StateGraph

from .config import PipelineConfig
from .display.panels import render_module_result, render_phase_done, render_phase_header
from .protocol import ModuleProtocol
from .registry import ModuleRegistry
from .state import PipelineState


def _should_continue_iterating(state: PipelineState) -> str:
    """Determine whether to iterate (M6 → M4) or end the pipeline."""
    if state.iteration_count >= state.max_iterations:
        return "end"

    # Check latest overall review score
    recent = [r for r in state.reviews if r.version == state.iteration_count]
    overall = [r for r in recent if r.dimension.value == "overall"]
    if overall and overall[0].score >= 4.0:
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

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._graph = None

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

        all_modules = ModuleRegistry.build_all(self.config)

        # Filter to enabled modules, preserving order
        enabled = self.config.enabled_modules
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
        """Wrap a module callable so it prints Rich headers and handles errors.

        Two additions on top of the vanilla wrapper:

        1. **Skip** — if the module's output fields are already populated in
           *state* (i.e. we are resuming from a checkpoint), the module is
           skipped and the existing state is returned unchanged.

        2. **Checkpoint** — after the module succeeds, the full state is
           persisted to disk so a later run can pick up from here.
        """
        output_fields = set(mod.get_output_fields())

        async def node_fn(state: PipelineState) -> Dict:
            # --- Resume: skip already-completed modules ---
            if output_fields and self._should_skip_module(name, output_fields, state):
                if self.config.verbose:
                    from .display import console, COLORS
                    console.print(
                        f"  [{COLORS['muted']}][SKIP] [{name.upper()}] "
                        f"already complete (checkpoint resume)[/{COLORS['muted']}]"
                    )
                return {}

            if self.config.verbose:
                render_phase_header(name, mod.description)

            try:
                result = await mod(state)
                if self.config.verbose:
                    render_module_result(name, state, result)
                    render_phase_done(name)
                # --- checkpoint: save state after each successful module ---
                self._save_checkpoint(name, state, result)
                return result
            except Exception as exc:
                import traceback
                if self.config.verbose:
                    from .display import console, COLORS
                    console.print(f"  [{COLORS['error']}][ERR] [{name.upper()}] ERROR: {exc}[/{COLORS['error']}]")
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
