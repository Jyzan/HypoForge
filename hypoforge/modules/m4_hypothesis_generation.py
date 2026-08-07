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
import re
import sys
import time
from typing import Any, Dict, List, Optional

from ..observability import emit_event
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


_EDITORIAL_STATEMENT_RE = re.compile(
    r"^\s*(?:"
    r"(?:please\s+)?consider\b|"
    r"to\s+(?:better\s+)?(?:strengthen|improve|clarify|address|validate)\b|"
    r"(?:we|the authors?)\s+(?:hypothesize|suggest|recommend|propose|should|need\s+to)\b|"
    r"(?:this|the)\s+hypothesis\s+(?:should|must|needs?\s+to|would\s+benefit)\b|"
    r"(?:a|the)\s+(?:stronger|revised)\s+hypothesis\s+(?:would|should)\b|"
    r"it\s+(?:is|would\s+be)\s+(?:important|helpful|useful|necessary|recommended)\b|"
    r"it\s+would\s+(?:strengthen|improve|clarify|address|validate)\b|"
    r"(?:future|further)\s+(?:work|research)\b|"
    r"(?:add|include|provide|clarify|acknowledge|discuss|investigate|explore)\b|"
    r"(?:建议|请考虑|考虑(?:增加|加入|补充)|应当|应该|需要(?:增加|加入|补充)|"
    r"为(?:了)?(?:加强|增强|改进|验证)|未来(?:工作|研究)|(?:本|该)假设(?:应|应该|需要))"
    r")",
    re.IGNORECASE,
)


def _followup_requirement_block(state: PipelineState) -> str:
    """Format ``state.followup.text`` as a priority requirement block.

    Returns ``""`` on non-followup runs, so prompt construction stays
    byte-for-byte identical to the legacy behaviour when no followup exists.
    """
    followup = getattr(state, "followup", None)
    text = (followup.text or "").strip() if followup is not None else ""
    if not text:
        return ""
    return (
        "\n--- 用户追问要求 (user follow-up instruction, highest priority, must be honoured) ---\n"
        f"{text}\n"
    )


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

    @staticmethod
    async def _observe_tool(
        tool: str,
        operation,
        *,
        details: Optional[Dict[str, Any]] = None,
    ) -> Any:
        started_at = time.monotonic()
        emit_event(
            "tool_started",
            module="m4",
            tool=tool,
            status="running",
            message=f"M4 Agent 开始：{tool}",
            details=details,
        )
        try:
            result = await operation
        except BaseException as exc:
            emit_event(
                "tool_failed",
                module="m4",
                tool=tool,
                status="failed",
                message=f"M4 Agent 失败：{tool}：{type(exc).__name__}: {exc}",
                elapsed_seconds=time.monotonic() - started_at,
                details=details,
            )
            raise
        emit_event(
            "tool_completed",
            module="m4",
            tool=tool,
            status="completed",
            message=f"M4 Agent 完成：{tool}",
            elapsed_seconds=time.monotonic() - started_at,
            details=details,
        )
        return result

    def _entry_lookup(self, state: PipelineState) -> Dict[str, str]:
        entries: Dict[str, str] = {}
        for lr in state.literature_results:
            for entry in lr.knowledge_entries:
                entries[entry.id] = entry.content
        return entries

    @staticmethod
    def _is_scientific_statement(statement: str) -> bool:
        """Return whether *statement* has the form of a scientific claim.

        This is deliberately a narrow guard against editorial instructions,
        not a substitute for scientific evaluation.  It protects state
        boundaries even when an LLM ignores prompt-level requirements.
        """
        text = str(statement or "").strip().lstrip("-*•").strip()
        if not text or text.endswith(("?", "？")):
            return False
        return _EDITORIAL_STATEMENT_RE.match(text) is None

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
            if not self._is_scientific_statement(card.statement):
                logger.warning(
                    "M4 discarded editorial/non-claim statement for %s: %s",
                    card.hypothesis_id,
                    card.statement[:160],
                )
                continue
            # A generator payload may contain preliminary scores. Recompute any
            # composite from the four dimensions via the shared rubric
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

    def _attach_rankings(
        self,
        candidates: List[HypothesisCard],
        payload: Any,
    ) -> List[HypothesisCard]:
        """Attach ranker-only output to the original candidate objects.

        The ranker is intentionally not trusted to return hypothesis content.
        This makes statements, mechanisms, predictions, and evidence immutable
        across the ranking boundary.
        """
        if isinstance(payload, dict):
            payload = payload.get("rankings") or payload.get("items") or payload.get("data") or []

        by_id = {card.hypothesis_id: card for card in candidates}
        ranked_cards: List[HypothesisCard] = []
        seen: set[str] = set()
        for item in payload or []:
            if not isinstance(item, dict):
                continue
            hypothesis_id = str(item.get("hypothesis_id") or "")
            if hypothesis_id in seen or hypothesis_id not in by_id:
                continue
            raw_scores = item.get("scores")
            if not isinstance(raw_scores, dict):
                continue
            try:
                scores = {
                    dimension: float(raw_scores[dimension])
                    for dimension in HYPOTHESIS_DIMENSIONS
                }
            except (KeyError, TypeError, ValueError):
                continue
            if not all(0.0 <= value <= 1.0 for value in scores.values()):
                continue
            scores["composite"] = composite_score(scores, self.weights)
            ranked_cards.append(by_id[hypothesis_id].model_copy(update={
                "ranking_rationale": str(item.get("ranking_rationale") or "").strip(),
                "scores": scores,
            }))
            seen.add(hypothesis_id)

        return self._rank_top(ranked_cards)

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
            from ..display.panels import render_guidance_prompt, render_guidance_result
            render_guidance_prompt(state.iteration_count)
            text = await asyncio.to_thread(input, "  > ")
        except (EOFError, KeyboardInterrupt):
            return ""
        guidance = text.strip()
        render_guidance_result(guidance)
        return guidance

    async def collect_user_guidance(self, state: PipelineState) -> Dict[str, Any]:
        """Collect the next revision instruction before the M4 header renders.

        ``PipelineRunner`` calls this pre-phase hook before printing the M4
        phase header.  Returning a state patch keeps the captured instruction
        in the normal LangGraph state flow and checkpoint output.
        """
        if state.iteration_count <= 0:
            return {}
        extra = await self._prompt_user_guidance(state)
        if not extra:
            return {}
        guidance = list(state.user_guidance)
        guidance.append(f"[iter {state.iteration_count}] {extra}")
        return {"user_guidance": guidance}

    def _build_feedback_context(self, state: PipelineState, guidance: List[str]) -> str:
        """Assemble reviewer feedback + prior hypotheses + human guidance into a
        revision block for the generator (empty string on the first round,
        except in followup runs where the follow-up instruction is always
        injected)."""
        followup_block = _followup_requirement_block(state)
        if state.iteration_count <= 0:
            return followup_block

        parts: List[str] = [f"\n--- Revision guidance (iteration {state.iteration_count}) ---"]

        prior_top = [
            hypothesis
            for hypothesis in state.top_hypotheses
            if self._is_scientific_statement(hypothesis.statement)
        ]
        if prior_top:
            parts.append("Prior top hypotheses (revise these, don't restart):")
            parts.extend(f"- [{h.hypothesis_id}] {h.statement}" for h in prior_top)

        latest = max((r.version for r in state.reviews), default=0)
        recent = [
            r for r in state.reviews
            if r.version == latest
            and r.dimension.value in {"scientific_logic", "evidence_consistency"}
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
        return followup_block + "\n".join(parts)

    def _update_best(self, state: PipelineState, top: List[HypothesisCard]) -> List[HypothesisCard]:
        """Keep the best hypotheses seen across all iterations, so a weaker
        revision cannot discard a stronger earlier result."""
        pool = [
            card
            for card in list(top) + list(state.best_hypotheses or [])
            if self._is_scientific_statement(card.statement)
        ]
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
        generated = await self._observe_tool(
            "hypothesis_generator",
            self.client.structured_chat(
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
            ),
            details={
                "requested_candidates": self.num_candidates,
                "iteration": state.iteration_count + 1,
            },
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
        ranker_item_schema = {
            "type": "object",
            "properties": {
                "hypothesis_id": {"type": "string"},
                "ranking_rationale": {"type": "string"},
                "scores": {
                    "type": "object",
                    "properties": {
                        dimension: {"type": "number", "minimum": 0, "maximum": 1}
                        for dimension in HYPOTHESIS_DIMENSIONS
                    },
                    "required": list(HYPOTHESIS_DIMENSIONS),
                },
            },
            "required": ["hypothesis_id", "ranking_rationale", "scores"],
        }
        ranker_schema = {"type": "array", "items": ranker_item_schema}
        ranked = await self._observe_tool(
            "hypothesis_ranker",
            ranker_client.structured_chat(
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
            ),
            details={"candidates": len(candidates), "top_k": self.top_k},
        )
        top = self._attach_rankings(candidates, ranked)
        if not top:
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

        Drops hypotheses with logical gaps or external inconsistencies.  The
        critic cannot rewrite candidate content; revisions belong to the next
        Generator pass.
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
                },
                "required": ["hypothesis_id", "pass", "critique", "issues"],
            },
        }
        try:
            result = await self._observe_tool(
                "hypothesis_critic",
                self.client.structured_chat(
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
                ),
                details={"candidates": len(candidates)},
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
            result = await self._observe_tool(
                "falsifiability_checker",
                self.client.structured_chat(
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
                ),
                details={"candidates": len(candidates)},
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

        # PipelineRunner collects optional human guidance before rendering the
        # M4 phase header.  Fold that state — plus reviewer feedback — into the
        # generator here.
        guidance = list(state.user_guidance)
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
