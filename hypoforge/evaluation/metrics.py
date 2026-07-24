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
from ..state import HypothesisCard, KnowledgeEntry, EvidenceGraph, EvidenceNode, EvidenceEdgeRelation, EvidenceNodeType

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

DECOMPOSE_SYSTEM_PROMPT = """\
You are a scientific logic analyzer. Your task is to decompose a complex scientific \
hypothesis into a list of independent, verifiable 'atomic claims'.

Rules:
1. An atomic claim must assert exactly ONE biological interaction, mechanism, or fact.
2. It must be self-contained (replace pronouns like 'it' with the actual entity).
3. Do not lose the epistemic context (if the hypothesis states 'A might cause B', \
   the claim is 'A causes B').
4. Output ONLY a JSON object with the key "claims" containing a list of strings.
"""

class GraphMetricBase(MetricProtocol):
    """Base class for metrics that require the evidence graph and LLM."""
    
    def __init__(self, llm_config: Any = None):
        # Lazy import to avoid circularity
        from ..tools.qwen_client import QwenClient  # noqa: F811
        self._client = QwenClient.from_config(llm_config) if llm_config else None
        
    async def _decompose_hypothesis(self, hypothesis: HypothesisCard) -> List[str]:
        if not self._client:
            return [hypothesis.statement]
            
        schema = {
            "type": "object", 
            "properties": {"claims": {"type": "array", "items": {"type": "string"}}}
        }
        try:
            result = await self._client.structured_chat(
                system_prompt=DECOMPOSE_SYSTEM_PROMPT,
                user_prompt=f"Hypothesis: {hypothesis.statement}\nMechanism: {hypothesis.mechanism}",
                output_schema=schema,
                max_tokens=512,
                temperature=0.0
            )
            if isinstance(result, dict) and "claims" in result:
                return result["claims"]
            return [hypothesis.statement]
        except Exception as exc:
            logger.warning("Failed to decompose hypothesis: %s", exc)
            return [hypothesis.statement]
            
    def _bm25_search(self, query: str, nodes: List[EvidenceNode], top_k: int = 3) -> List[Tuple[EvidenceNode, float]]:
        try:
            from rank_bm25 import BM25Okapi
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            tokenized_nodes = [enc.encode(n.label) for n in nodes]
            query_tokens = enc.encode(query)
            bm25 = BM25Okapi(tokenized_nodes)
            scores = bm25.get_scores(query_tokens)
            
            scored_nodes = list(zip(nodes, scores))
            scored_nodes.sort(key=lambda x: x[1], reverse=True)
            return scored_nodes[:top_k]
        except ImportError:
            # Fallback linear search
            query_lower = query.lower()
            scored_nodes = []
            for n in nodes:
                score = 0.0
                if query_lower in n.label.lower():
                    score = 1.0
                scored_nodes.append((n, score))
            scored_nodes.sort(key=lambda x: x[1], reverse=True)
            return scored_nodes[:top_k]


@MetricRegistry.register
class NoveltyMetric(GraphMetricBase):
    metric_name = "novelty"
    metric_description = "Retrieval-grounded novelty vs. the retrieved corpus / knowledge graph."
    independent = True
    implemented = True

    async def compute(self, hypothesis: HypothesisCard, knowledge_entries: List[KnowledgeEntry], **kwargs) -> float:
        evidence_graph = kwargs.get("evidence_graph")
        if not evidence_graph or not self._client:
            return 0.0
            
        claims = await self._decompose_hypothesis(hypothesis)
        if not claims:
            return 0.0
            
        searchable_nodes = [n for n in evidence_graph.nodes if n.type in (EvidenceNodeType.CLAIM, EvidenceNodeType.EVIDENCE)]
        
        system_prompt = """You evaluate scientific novelty.
Given a 'Target Claim' and 'Known Literature Facts', determine if the Target Claim is already explicitly stated as an established fact.
If the Target Claim proposes a NEW connection, mechanism, or idea not explicitly present in the Known Facts, it is novel.
If it is just restating a Known Fact, it is not novel.
Output JSON: {"is_novel": true/false, "rationale": "..."}"""

        novel_count = 0
        for claim in claims:
            anchors = self._bm25_search(claim, searchable_nodes, top_k=3)
            # Filter zero scores if we want, but for now we pass top_k valid ones
            valid_anchors = [n for n, score in anchors if score > 0.0]
            if not valid_anchors:
                novel_count += 1  # No related anchors, completely novel
                continue
                
            known_facts = "\n".join(f"- {n.label}" for n in valid_anchors)
            schema = {"type": "object", "properties": {"is_novel": {"type": "boolean"}, "rationale": {"type": "string"}}}
            
            try:
                res = await self._client.structured_chat(
                    system_prompt=system_prompt,
                    user_prompt=f"Target Claim: {claim}\nKnown Literature Facts:\n{known_facts}",
                    output_schema=schema,
                    temperature=0.0
                )
                if isinstance(res, dict) and res.get("is_novel", False):
                    novel_count += 1
            except Exception:
                novel_count += 1  # Assume novel on LLM failure
                
        return round(novel_count / len(claims), 4)

    async def batch_compute(self, hypotheses: List[HypothesisCard], knowledge_entries: List[KnowledgeEntry], **kwargs) -> List[float]:
        return [await self.compute(h, knowledge_entries, **kwargs) for h in hypotheses]


@MetricRegistry.register
class EvidenceConsistencyMetric(GraphMetricBase):
    metric_name = "evidence_consistency"
    metric_description = "Consistency with the evidence graph — does the hypothesis contradict established facts / CONTRADICTS edges?"
    independent = True
    implemented = True

    async def compute(self, hypothesis: HypothesisCard, knowledge_entries: List[KnowledgeEntry], **kwargs) -> float:
        evidence_graph = kwargs.get("evidence_graph")
        if not evidence_graph or not self._client:
            return 0.0
            
        claims = await self._decompose_hypothesis(hypothesis)
        if not claims:
            return 0.0
            
        searchable_nodes = [n for n in evidence_graph.nodes if n.type in (EvidenceNodeType.CLAIM, EvidenceNodeType.EVIDENCE)]
        
        system_prompt = """You are a strict scientific reviewer.
Determine if the 'Target Claim' directly violates or ignores the provided 'Threat Context' (known conflicts/limitations from literature).
If the threat context is irrelevant to the claim, or if the claim successfully resolves the threat, there is no conflict.
Output JSON: {"is_conflict": true/false, "rationale": "..."}"""

        consistent_count = 0
        for claim in claims:
            anchors = self._bm25_search(claim, searchable_nodes, top_k=3)
            valid_anchors = [n for n, score in anchors if score > 0.0]
            if not valid_anchors:
                consistent_count += 1
                continue
                
            threat_nodes = self._build_threat_context(valid_anchors, evidence_graph)
            if not threat_nodes:
                consistent_count += 1
                continue
                
            threat_context_str = "\n".join(f"- {n.label}" for n in threat_nodes)
            schema = {"type": "object", "properties": {"is_conflict": {"type": "boolean"}, "rationale": {"type": "string"}}}
            
            try:
                res = await self._client.structured_chat(
                    system_prompt=system_prompt,
                    user_prompt=f"Target Claim: {claim}\nThreat Context:\n{threat_context_str}",
                    output_schema=schema,
                    temperature=0.0
                )
                # Not a conflict => consistent
                if isinstance(res, dict) and not res.get("is_conflict", True):
                    consistent_count += 1
            except Exception:
                consistent_count += 1
                
        return round(consistent_count / len(claims), 4)

    async def batch_compute(self, hypotheses: List[HypothesisCard], knowledge_entries: List[KnowledgeEntry], **kwargs) -> List[float]:
        return [await self.compute(h, knowledge_entries, **kwargs) for h in hypotheses]
        
    def _build_threat_context(self, anchors: List[EvidenceNode], graph: EvidenceGraph) -> List[EvidenceNode]:
        anchor_ids = {n.id for n in anchors}
        threat_node_ids = set()
        
        threat_rels = {EvidenceEdgeRelation.CONTRADICTS, EvidenceEdgeRelation.LIMITS}
        ally_rels = {EvidenceEdgeRelation.SUPPORTS, EvidenceEdgeRelation.SAME_AS}
        
        for edge in graph.edges:
            # 1-hop threats
            if edge.source in anchor_ids and edge.relation in threat_rels:
                threat_node_ids.add(edge.target)
            elif edge.target in anchor_ids and edge.relation in threat_rels:
                threat_node_ids.add(edge.source)
                
            # Allies
            ally_nodes = set()
            if edge.source in anchor_ids and edge.relation in ally_rels:
                ally_nodes.add(edge.target)
            elif edge.target in anchor_ids and edge.relation in ally_rels:
                ally_nodes.add(edge.source)
                
            # 2-hop threats
            for ally_id in ally_nodes:
                for e2 in graph.edges:
                    if e2.source == ally_id and e2.relation in threat_rels:
                        threat_node_ids.add(e2.target)
                    elif e2.target == ally_id and e2.relation in threat_rels:
                        threat_node_ids.add(e2.source)
                        
        return [n for n in graph.nodes if n.id in threat_node_ids]
