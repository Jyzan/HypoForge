"""
Evaluation metrics for automated hypothesis scoring.

Each metric implements ``MetricProtocol`` and is registered via
``@MetricRegistry.register``.

**Independence contract.**  A metric must either compute its score
*independently* of the generator's self-assessment, or declare
``independent = False``.  Metrics that would otherwise just echo M4's own
``hypothesis.scores`` are marked ``implemented = False`` until a real,
grounded implementation (retrieval / evidence-graph / embeddings) exists —
the scorer skips them rather than laundering a self-reported number into what
looks like an objective evaluation.

Plan completeness is a *plan-level* metric (it needs a ``ResearchPlan``, not a
``HypothesisCard``) and therefore lives in :mod:`hypoforge.evaluation.scorer`
as ``score_plan_completeness`` rather than in this hypothesis-metric registry.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from ..protocol import MetricProtocol
from ..state import HypothesisCard, KnowledgeEntry

logger = logging.getLogger(__name__)


# ============================================================================
# Metric Registry (lightweight — mirrors ToolRegistry pattern)
# ============================================================================

class MetricRegistry:
    _metrics: dict[str, type[MetricProtocol]] = {}

    @classmethod
    def register(cls, metric_cls: type[MetricProtocol]) -> type[MetricProtocol]:
        cls._metrics[metric_cls.metric_name] = metric_cls
        return metric_cls

    @classmethod
    def get(cls, name: str) -> type[MetricProtocol] | None:
        return cls._metrics.get(name)

    @classmethod
    def list_all(cls) -> list[str]:
        return sorted(cls._metrics.keys())

    @classmethod
    def list_implemented(cls) -> list[str]:
        """Names of metrics that have a real, non-stub implementation."""
        return sorted(n for n, m in cls._metrics.items() if getattr(m, "implemented", True))


# ============================================================================
# Independent, implemented metrics
# ============================================================================

@MetricRegistry.register
class TestabilityMetric(MetricProtocol):
    """LLM-as-judge evaluation of testability.

    Unlike the earlier checklist (which awarded points for *any* non-empty
    content), this metric asks a lightweight LLM to assess whether each
    observable prediction is concrete / directional / measurable, and whether
    each falsification condition would genuinely disprove the hypothesis.

    Falls back to the legacy checklist when no LLM client is available.
    """

    metric_name = "testability"
    metric_description = (
        "LLM-as-judge — evaluates whether predictions are concrete, directional, "
        "and measurable, and whether falsification conditions genuinely disprove "
        "the hypothesis."
    )
    independent = True
    implemented = True

    # ── rubric to keep the LLM anchored ──────────────────────────────
    _system_prompt = """\
You evaluate whether a scientific hypothesis is genuinely testable.  Do NOT \
credit vague or unfalsifiable placeholder text.

Score two aspects independently:

1. **Observable predictions** (0.0 – 0.5):
   - 0.0 = no predictions, or predictions so vague they cannot be tested \
("something will change", "the mechanism will be revealed").
   - 0.2 = at least one prediction names a measurable quantity but omits \
direction, magnitude, or experimental context.
   - 0.35 = predictions are mostly concrete but one is under-specified (e.g. \
missing the expected direction of change).
   - 0.5 = predictions are concrete, directional, specify what should be \
measured under what conditions, and each one can be mapped to a real experiment.

2. **Falsification conditions** (0.0 – 0.5):
   - 0.0 = no conditions, or the stated "condition" would not actually \
falsify the hypothesis (e.g. "if the hypothesis is wrong, we will see \
nothing" — circular).
   - 0.2 = at least one condition is testable in principle but vague about \
what outcome counts as falsifying.
   - 0.35 = conditions are relevant but one is not experimentally decisive.
   - 0.5 = conditions are specific, experimentally accessible, and a null \
result under those conditions would definitively disprove the core claim.

Return a JSON object with exactly these keys:
{"predictions_score": <float 0.0–0.5>, "falsification_score": <float 0.0–0.5>, "rationale": "<one sentence>"}
"""

    _user_template = """\
Hypothesis statement: {statement}

Observable predictions:
{predictions}

Falsification conditions:
{falsification}
"""

    # ── construction ─────────────────────────────────────────────────

    def __init__(self, llm_config: Any = None):
        # Lazy import to avoid circularity at module level
        from ..tools.qwen_client import QwenClient  # noqa: F811
        self._client = QwenClient.from_config(llm_config) if llm_config else None

    # ── compute ──────────────────────────────────────────────────────

    async def compute(
        self,
        hypothesis: HypothesisCard,
        knowledge_entries: List[KnowledgeEntry],
        **kwargs,
    ) -> float:
        if self._client is None:
            return self._legacy_checklist(hypothesis)

        try:
            return await self._llm_evaluate(hypothesis)
        except Exception as exc:
            logger.warning("TestabilityMetric LLM evaluation failed (%s); falling back to checklist", exc)
            return self._legacy_checklist(hypothesis)

    async def batch_compute(
        self,
        hypotheses: List[HypothesisCard],
        knowledge_entries: List[KnowledgeEntry],
        **kwargs,
    ) -> List[float]:
        return [await self.compute(h, knowledge_entries, **kwargs) for h in hypotheses]

    # ── internals ────────────────────────────────────────────────────

    async def _llm_evaluate(self, hypothesis: HypothesisCard) -> float:
        predictions_text = self._format_list(hypothesis.observable_predictions)
        falsification_text = self._format_list(hypothesis.falsification_conditions)

        schema = {
            "type": "object",
            "properties": {
                "predictions_score": {"type": "number", "minimum": 0.0, "maximum": 0.5},
                "falsification_score": {"type": "number", "minimum": 0.0, "maximum": 0.5},
                "rationale": {"type": "string"},
            },
            "required": ["predictions_score", "falsification_score"],
        }
        result = await self._client.structured_chat(
            system_prompt=self._system_prompt,
            user_prompt=self._user_template.format(
                statement=hypothesis.statement,
                predictions=predictions_text,
                falsification=falsification_text,
            ),
            output_schema=schema,
            max_tokens=512,
            temperature=0.0,
        )
        result = result if isinstance(result, dict) else {}
        p_score = max(0.0, min(0.5, float(result.get("predictions_score", 0.0))))
        f_score = max(0.0, min(0.5, float(result.get("falsification_score", 0.0))))
        return round(p_score + f_score, 4)

    @staticmethod
    def _format_list(items: Optional[List[str]]) -> str:
        if not items:
            return "(none provided)"
        return "\n".join(f"- {item}" for item in items)

    @staticmethod
    def _legacy_checklist(hypothesis: HypothesisCard) -> float:
        """Simple presence check — used only as a fallback when no LLM is available."""
        score = 0.0
        if hypothesis.observable_predictions:
            score += 0.5
        if hypothesis.falsification_conditions:
            score += 0.5
        return score


# ============================================================================
# Declared-but-not-yet-implemented metrics
#
# These require grounding the score in something *other than* the generator's
# own output (a retrieval index, the evidence graph, an embedding model).
# Until that exists they are flagged ``implemented = False`` and the scorer
# skips them — they must never fall back to ``hypothesis.scores[...]``.
# ============================================================================

@MetricRegistry.register
class NoveltyMetric(MetricProtocol):
    metric_name = "novelty"
    metric_description = (
        "Retrieval-grounded novelty vs. the retrieved corpus / knowledge graph "
        "(pending: embeddings or memory.bm25_index search)."
    )
    independent = True
    implemented = False  # TODO: wire to memory/bm25_index or an embedding model

    async def compute(self, hypothesis: HypothesisCard, knowledge_entries: List[KnowledgeEntry], **kwargs) -> float:
        raise NotImplementedError(
            "Independent novelty scoring is not implemented yet; it must not "
            "fall back to the generator's self-reported novelty score."
        )

    async def batch_compute(self, hypotheses: List[HypothesisCard], knowledge_entries: List[KnowledgeEntry], **kwargs) -> List[float]:
        raise NotImplementedError


@MetricRegistry.register
class EvidenceConsistencyMetric(MetricProtocol):
    metric_name = "evidence_consistency"
    metric_description = (
        "Consistency with the evidence graph — does the hypothesis contradict "
        "established facts / CONTRADICTS edges? (pending: graph-grounded check)."
    )
    independent = True
    implemented = False  # TODO: ground in EvidenceGraph edges / LLM-as-judge with citations

    async def compute(self, hypothesis: HypothesisCard, knowledge_entries: List[KnowledgeEntry], **kwargs) -> float:
        raise NotImplementedError(
            "Independent evidence-consistency scoring is not implemented yet; it "
            "must not fall back to the generator's self-reported score."
        )

    async def batch_compute(self, hypotheses: List[HypothesisCard], knowledge_entries: List[KnowledgeEntry], **kwargs) -> List[float]:
        raise NotImplementedError
