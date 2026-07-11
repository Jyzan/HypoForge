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
        """Wrap a module callable so it prints Rich headers and handles errors."""

        async def node_fn(state: PipelineState) -> Dict:
            if self.config.verbose:
                render_phase_header(name, mod.description)

            try:
                result = await mod(state)
                if self.config.verbose:
                    render_module_result(name, state, result)
                    render_phase_done(name)
                return result
            except Exception as exc:
                import traceback
                if self.config.verbose:
                    from .display import console, COLORS
                    console.print(f"  [{COLORS['error']}][ERR] [{name.upper()}] ERROR: {exc}[/{COLORS['error']}]")
                return {"errors": state.errors + [f"[{name}] {exc}\n{traceback.format_exc()}"]}

        return node_fn

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    async def run(self, question: str, run_id: str = "") -> PipelineState:
        """
        Execute the full pipeline for a given scientific question.

        Parameters
        ----------
        question : str
            The frontier scientific question (e.g. from Science 125).
        run_id : str
            Optional identifier for this run (auto-generated if empty).

        Returns
        -------
        PipelineState
            The final state after all modules (and iterations) have completed.
        """
        import uuid

        if not run_id:
            run_id = f"hypoforge-{uuid.uuid4().hex[:8]}"

        graph = self._build_graph()

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
