"""
Evaluation metrics for automated hypothesis and plan scoring.

Each metric implements ``MetricProtocol`` and is registered via
``@MetricRegistry.register``.
"""

from __future__ import annotations

from typing import List

from ..protocol import MetricProtocol
from ..state import HypothesisCard, KnowledgeEntry


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


# ============================================================================
# Stub Metrics
# ============================================================================

@MetricRegistry.register
class NoveltyMetric(MetricProtocol):
    metric_name = "novelty"
    metric_description = "Cosine distance between hypothesis embedding and literature embeddings"

    async def compute(self, hypothesis: HypothesisCard, knowledge_entries: List[KnowledgeEntry], **kwargs) -> float:
        # Stub — returns the score from the hypothesis card or a fixed value
        return hypothesis.scores.get("novelty", 0.7)

    async def batch_compute(self, hypotheses: List[HypothesisCard], knowledge_entries: List[KnowledgeEntry], **kwargs) -> List[float]:
        return [await self.compute(h, knowledge_entries, **kwargs) for h in hypotheses]


@MetricRegistry.register
class TestabilityMetric(MetricProtocol):
    metric_name = "testability"
    metric_description = "Checklist-based: are observable predictions and falsification conditions present?"

    async def compute(self, hypothesis: HypothesisCard, knowledge_entries: List[KnowledgeEntry], **kwargs) -> float:
        score = 0.0
        if hypothesis.observable_predictions:
            score += 0.5
        if hypothesis.falsification_conditions:
            score += 0.5
        return score

    async def batch_compute(self, hypotheses: List[HypothesisCard], knowledge_entries: List[KnowledgeEntry], **kwargs) -> List[float]:
        return [await self.compute(h, knowledge_entries, **kwargs) for h in hypotheses]


@MetricRegistry.register
class EvidenceConsistencyMetric(MetricProtocol):
    metric_name = "evidence_consistency"
    metric_description = "LLM-as-judge: 1-5 score for consistency with known evidence"

    async def compute(self, hypothesis: HypothesisCard, knowledge_entries: List[KnowledgeEntry], **kwargs) -> float:
        # Stub — returns the score from the hypothesis card
        return hypothesis.scores.get("evidence_consistency", 0.8)

    async def batch_compute(self, hypotheses: List[HypothesisCard], knowledge_entries: List[KnowledgeEntry], **kwargs) -> List[float]:
        return [await self.compute(h, knowledge_entries, **kwargs) for h in hypotheses]


@MetricRegistry.register
class PlanCompletenessMetric(MetricProtocol):
    metric_name = "plan_completeness"
    metric_description = "Checklist: what fraction of the 10 research plan elements are filled?"

    async def compute(self, hypothesis: HypothesisCard, knowledge_entries: List[KnowledgeEntry], **kwargs) -> float:
        # Requires a ResearchPlan — called from scorer
        return 0.0  # Overridden by scorer when plan is available

    async def batch_compute(self, hypotheses: List[HypothesisCard], knowledge_entries: List[KnowledgeEntry], **kwargs) -> List[float]:
        return [0.0 for _ in hypotheses]
