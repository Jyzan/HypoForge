"""
M4: Hypothesis Generation & Screening (core innovation module).

Multi-agent pipeline: Generator → Ranker.  On revision rounds it folds in the
latest reviewer feedback and optional human guidance, and keeps the best
hypotheses seen across iterations.

Output: ``candidate_hypotheses`` + ``top_hypotheses`` in state.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from typing import Any, Dict, List, Optional

from ..protocol import ModuleProtocol
from ..prompts.m4_prompts import (
    M4_CRITIC_SYSTEM_PROMPT,
    M4_CRITIC_USER_TEMPLATE,
    M4_FALSIFIABILITY_SYSTEM_PROMPT,
    M4_FALSIFIABILITY_USER_TEMPLATE,
    M4_GENERATOR_SYSTEM_PROMPT,
    M4_GENERATOR_USER_TEMPLATE,
    M4_RANKER_SYSTEM_PROMPT,
    M4_RANKER_USER_TEMPLATE,
)
from ..evaluation.rubric import (
    HYPOTHESIS_DIMENSIONS,
    composite_score,
    hypothesis_rubric_block,
    normalise_weights,
    weights_summary,
)
from ..registry import ModuleRegistry
from ..state import HypothesisCard, PipelineState
from ..tools.qwen_client import QwenClient

logger = logging.getLogger(__name__)


@ModuleRegistry.register
class M4HypothesisGeneration(ModuleProtocol):
    module_name = "m4"
    module_version = "0.1.0"
    description = "Multi-agent hypothesis generation: Generator → Critic → Falsifiability → Ranker"

    # ------------------------------------------------------------------
    # Configurable parameters
    # ------------------------------------------------------------------

    def __init__(
        self,
        num_candidates: int = 7,
        top_k: int = 3,
        mode: str = "multi_agent",  # "direct" | "multi_agent"
        llm_config: Optional[Any] = None,
        ranker_llm_config: Optional[Any] = None,
        weights: Optional[Dict[str, float]] = None,
        interactive: bool = False,
        **kwargs,
    ):
        self.num_candidates = num_candidates
        self.top_k = top_k
        self.mode = mode
        self.llm_config = llm_config
        # Composite weights come from a single source (PipelineConfig.scoring →
        # rubric defaults); the ranker prompt and the recompute below share them.
        self.weights = normalise_weights(weights)
        # When True (and on a real TTY), pause between iterations to let the user
        # type guidance for the next revision — Claude-Code style.
        self.interactive = interactive
        self.client = QwenClient.from_config(llm_config) if llm_config else None
        # Ranker uses a separate model tier to reduce self-scoring bias.
        # Falls back to the main client when no separate config is provided.
        _ranker_cfg = ranker_llm_config if ranker_llm_config is not None else llm_config
        self.ranker_client = QwenClient.from_config(_ranker_cfg) if _ranker_cfg else None

    # ------------------------------------------------------------------
    # LLM helpers
    # ------------------------------------------------------------------

    def _entry_lookup(self, state: PipelineState) -> Dict[str, str]:
        entries: Dict[str, str] = {}
        for lr in state.literature_results:
            for entry in lr.knowledge_entries:
                entries[entry.id] = entry.content
        return entries

    def _graph_bucket_text(self, state: PipelineState, bucket: str) -> str:
        graph = state.evidence_graph
        entry_text = self._entry_lookup(state)
        ids = getattr(graph, bucket, []) if graph else []
        lines = [f"- {entry_id}: {entry_text.get(entry_id, entry_id)}" for entry_id in ids]
        return "\n".join(lines) if lines else "- None available"

    def _normalise_hypotheses(self, payload: Any) -> List[HypothesisCard]:
        if isinstance(payload, dict):
            payload = payload.get("hypotheses") or payload.get("items") or payload.get("data") or []

        cards: List[HypothesisCard] = []
        for idx, item in enumerate(payload or [], start=1):
            if not isinstance(item, dict):
                continue
            item = dict(item)
            item.setdefault("hypothesis_id", f"H{idx}")
            item.setdefault("scores", {})
            card = HypothesisCard.model_validate(item)
            # The ranker may omit composite or return an inconsistent value.
            # Recompute it from the four dimension scores via the shared rubric
            # (single source of truth for the weights) whenever they are all
            # present, so the displayed score and ranking match the documented
            # formula.
            if all(key in card.scores for key in HYPOTHESIS_DIMENSIONS):
                card.scores["composite"] = composite_score(card.scores, self.weights)
            cards.append(card)
        return cards

    def _rank_top(self, cards: List[HypothesisCard]) -> List[HypothesisCard]:
        return sorted(
            cards,
            key=lambda h: h.scores.get("composite", 0),
            reverse=True,
        )[: self.top_k]

    # ------------------------------------------------------------------
    # Iteration feedback loop (C): reviews + human guidance → next round
    # ------------------------------------------------------------------

    async def _prompt_user_guidance(self, state: PipelineState) -> str:
        """Claude-Code-style pause: let the user type guidance for this revision
        round, or press Enter to skip.

        No-op (returns "") unless ``interactive`` is set *and* we are attached to
        a real TTY — so tests, pipes, and ``--quiet`` runs never block on stdin.
        """
        if not self.interactive or not sys.stdin.isatty():
            return ""
        try:
            from ..display.panels import render_guidance_prompt
            render_guidance_prompt(state.iteration_count)
            text = await asyncio.to_thread(input, "  > ")
        except (EOFError, KeyboardInterrupt):
            return ""
        return text.strip()

    def _build_feedback_context(self, state: PipelineState, guidance: List[str]) -> str:
        """Assemble reviewer feedback + prior hypotheses + human guidance into a
        revision block for the generator (empty string on the first round)."""
        if state.iteration_count <= 0:
            return ""

        parts: List[str] = [f"\n--- Revision guidance (iteration {state.iteration_count}) ---"]

        if state.top_hypotheses:
            parts.append("Prior top hypotheses (revise these, don't restart):")
            parts.extend(f"- [{h.hypothesis_id}] {h.statement}" for h in state.top_hypotheses)

        latest = max((r.version for r in state.reviews), default=0)
        recent = [
            r for r in state.reviews
            if r.version == latest and r.dimension.value != "overall"
        ]
        if recent:
            parts.append("\nReviewer feedback (address these):")
            for r in recent:
                fix = r.suggestions or r.comments or ""
                parts.append(f"- [{r.dimension.value}] {r.score:.1f}/5 — {fix}")

        if guidance:
            parts.append("\nUser guidance (highest priority — follow closely):")
            parts.extend(f"- {g}" for g in guidance)

        parts.append("")
        return "\n".join(parts)

    def _update_best(self, state: PipelineState, top: List[HypothesisCard]) -> List[HypothesisCard]:
        """Keep the best hypotheses seen across all iterations, so a weaker
        revision cannot discard a stronger earlier result."""
        pool = list(top) + list(state.best_hypotheses or [])
        dedup: Dict[str, HypothesisCard] = {}
        for card in pool:
            key = card.statement.strip()
            existing = dedup.get(key)
            if existing is None or card.scores.get("composite", 0) > existing.scores.get("composite", 0):
                dedup[key] = card
        return self._rank_top(list(dedup.values()))

    async def _run_llm(self, state: PipelineState, feedback_context: str = "") -> Dict[str, Any]:
        assert self.client is not None
        question = state.problem_card.original_question if state.problem_card else state.input_question
        rubric_block = hypothesis_rubric_block()

        # ── Step 1: Generator ──────────────────────────────────────────
        generator_schema = {
            "type": "array",
            "items": HypothesisCard.model_json_schema(),
        }
        generated = await self.client.structured_chat(
            system_prompt=M4_GENERATOR_SYSTEM_PROMPT.format(
                num_candidates=self.num_candidates,
                rubric_block=rubric_block,
            ),
            user_prompt=M4_GENERATOR_USER_TEMPLATE.format(
                knowledge_gaps=self._graph_bucket_text(state, "knowledge_gaps"),
                established_facts=self._graph_bucket_text(state, "established_facts"),
                conflicts=self._graph_bucket_text(state, "conflicts"),
                original_question=question,
                num_candidates=self.num_candidates,
                feedback_context=feedback_context,
            ),
            output_schema=generator_schema,
            max_tokens=16384,
            temperature=max(getattr(self.llm_config, "temperature", 0.1), 0.4),
        )
        candidates = self._normalise_hypotheses(generated)
        if not candidates:
            raise ValueError("M4 generator returned no valid hypotheses")

        if self.mode != "multi_agent":
            top = self._rank_top(candidates)
            return {
                "candidate_hypotheses": candidates,
                "top_hypotheses": top[: self.top_k],
            }

        # ── Step 2: Critic ─────────────────────────────────────────────
        candidates = await self._run_critic(state, candidates)

        # ── Step 3: Falsifiability Checker ─────────────────────────────
        candidates = await self._run_falsifiability(state, candidates)

        # ── Step 4: Ranker (separate model tier) ───────────────────────
        ranker_client = self.ranker_client or self.client
        ranker_item_schema = HypothesisCard.model_json_schema()
        ranker_item_schema["properties"]["scores"] = {
            "type": "object",
            "properties": {
                "novelty": {"type": "number", "minimum": 0, "maximum": 1},
                "scientific_soundness": {"type": "number", "minimum": 0, "maximum": 1},
                "testability": {"type": "number", "minimum": 0, "maximum": 1},
                "evidence_consistency": {"type": "number", "minimum": 0, "maximum": 1},
                "composite": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": [
                "novelty",
                "scientific_soundness",
                "testability",
                "evidence_consistency",
            ],
        }
        ranker_schema = {"type": "array", "items": ranker_item_schema}
        ranked = await ranker_client.structured_chat(
            system_prompt=M4_RANKER_SYSTEM_PROMPT.format(
                top_k=self.top_k,
                weights_formula=weights_summary(self.weights),
                rubric_block=hypothesis_rubric_block(),
            ),
            user_prompt=M4_RANKER_USER_TEMPLATE.format(
                hypotheses_json=json.dumps(
                    [h.model_dump(mode="json") for h in candidates],
                    ensure_ascii=False,
                    indent=2,
                ),
                top_k=self.top_k,
            ),
            output_schema=ranker_schema,
            max_tokens=8192,
            temperature=getattr(self.llm_config, "temperature", 0.1),
        )
        top = self._normalise_hypotheses(ranked)
        required_scores = {
            "novelty",
            "scientific_soundness",
            "testability",
            "evidence_consistency",
        }
        if not top or not all(required_scores.issubset(h.scores) for h in top):
            logger.warning("M4 ranker returned incomplete scores; ranking generated candidates instead")
            top = self._rank_top(candidates)

        return {
            "candidate_hypotheses": candidates,
            "top_hypotheses": top[: self.top_k],
        }

    # ------------------------------------------------------------------
    # Multi-agent sub-steps (Critic, Falsifiability)
    # ------------------------------------------------------------------

    async def _run_critic(
        self, state: PipelineState, candidates: List[HypothesisCard]
    ) -> List[HypothesisCard]:
        """Filter candidates through the Critic agent.

        Drops hypotheses with logical gaps or external inconsistencies,
        and applies any ``suggested_revision`` to the surviving statements.
        Falls back to the original list if every candidate is rejected.
        """
        critic_schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "hypothesis_id": {"type": "string"},
                    "pass": {"type": "boolean"},
                    "critique": {"type": "string"},
                    "issues": {"type": "array", "items": {"type": "string"}},
                    "suggested_revision": {"type": "string"},
                },
                "required": ["hypothesis_id", "pass", "critique", "issues"],
            },
        }
        try:
            result = await self.client.structured_chat(
                system_prompt=M4_CRITIC_SYSTEM_PROMPT,
                user_prompt=M4_CRITIC_USER_TEMPLATE.format(
                    established_facts=self._graph_bucket_text(state, "established_facts"),
                    hypotheses_json=json.dumps(
                        [h.model_dump(mode="json") for h in candidates],
                        ensure_ascii=False,
                        indent=2,
                    ),
                ),
                output_schema=critic_schema,
                max_tokens=8192,
                temperature=getattr(self.llm_config, "temperature", 0.1),
            )
        except Exception as exc:
            logger.warning("M4 critic failed (%s); keeping all candidates", exc)
            return candidates

        reviews = result if isinstance(result, list) else []
        id_to_candidate = {h.hypothesis_id: h for h in candidates}
        survivors: List[HypothesisCard] = []

        for review in reviews:
            if not isinstance(review, dict):
                continue
            if not review.get("pass", False):
                logger.info("M4 critic rejected %s: %s", review.get("hypothesis_id"), review.get("critique", "")[:120])
                continue
            hid = review.get("hypothesis_id", "")
            card = id_to_candidate.get(hid)
            if card is None:
                continue
            # Apply suggested revision when provided (non-empty, differs from original)
            revision = (review.get("suggested_revision") or "").strip()
            if revision and revision != card.statement.strip():
                card = card.model_copy(update={"statement": revision})
            survivors.append(card)

        if len(survivors) < 2:
            logger.warning("M4 critic left %d survivors (need ≥2); keeping all %d candidates",
                           len(survivors), len(candidates))
            return candidates
        logger.info("M4 critic: %d/%d passed", len(survivors), len(candidates))
        return survivors

    async def _run_falsifiability(
        self, state: PipelineState, candidates: List[HypothesisCard]
    ) -> List[HypothesisCard]:
        """Filter candidates through the Falsifiability Checker.

        Drops hypotheses that cannot be empirically falsified.
        Falls back to the original list if every candidate is rejected.
        """
        falsifiability_schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "hypothesis_id": {"type": "string"},
                    "is_falsifiable": {"type": "boolean"},
                    "specificity_score": {"type": "number", "minimum": 0, "maximum": 1},
                    "assessment": {"type": "string"},
                },
                "required": ["hypothesis_id", "is_falsifiable", "assessment"],
            },
        }
        try:
            result = await self.client.structured_chat(
                system_prompt=M4_FALSIFIABILITY_SYSTEM_PROMPT,
                user_prompt=M4_FALSIFIABILITY_USER_TEMPLATE.format(
                    hypotheses_json=json.dumps(
                        [h.model_dump(mode="json") for h in candidates],
                        ensure_ascii=False,
                        indent=2,
                    ),
                ),
                output_schema=falsifiability_schema,
                max_tokens=8192,
                temperature=getattr(self.llm_config, "temperature", 0.1),
            )
        except Exception as exc:
            logger.warning("M4 falsifiability checker failed (%s); keeping all candidates", exc)
            return candidates

        reviews = result if isinstance(result, list) else []
        id_to_candidate = {h.hypothesis_id: h for h in candidates}
        survivors: List[HypothesisCard] = []

        for review in reviews:
            if not isinstance(review, dict):
                continue
            if not review.get("is_falsifiable", False):
                logger.info("M4 falsifiability rejected %s: %s", review.get("hypothesis_id"), review.get("assessment", "")[:120])
                continue
            hid = review.get("hypothesis_id", "")
            card = id_to_candidate.get(hid)
            if card is None:
                continue
            survivors.append(card)

        if len(survivors) < 2:
            logger.warning("M4 falsifiability left %d survivors (need ≥2); keeping incoming candidates",
                           len(survivors))
            return candidates
        logger.info("M4 falsifiability: %d/%d passed", len(survivors), len(candidates))
        return survivors

    # ------------------------------------------------------------------
    # ModuleProtocol implementation
    # ------------------------------------------------------------------

    async def __call__(
        self,
        state: PipelineState,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if self.client is None:
            raise RuntimeError(
                "M4 requires an LLM client — pass llm_config / set OPENAI_API_KEY."
            )

        # On revision rounds, optionally gather human guidance (Claude-Code-style)
        # and fold it — plus the latest reviewer feedback — into the generator.
        guidance = list(state.user_guidance)
        if state.iteration_count > 0:
            extra = await self._prompt_user_guidance(state)
            if extra:
                guidance.append(f"[iter {state.iteration_count}] {extra}")
        feedback_context = self._build_feedback_context(state, guidance)

        result = await self._run_llm(state, feedback_context)
        result["user_guidance"] = guidance
        result["best_hypotheses"] = self._update_best(state, result.get("top_hypotheses", []))
        return result

    @classmethod
    def get_input_fields(cls) -> List[str]:
        return ["evidence_graph", "problem_card", "literature_results"]

    @classmethod
    def get_output_fields(cls) -> List[str]:
        return ["candidate_hypotheses", "top_hypotheses"]
