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
import asyncio
import math
import numpy as np
import yaml
import os
from collections import deque
import networkx as nx
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Type

from pydantic import BaseModel, Field

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
            max_tokens=2048,
            temperature=0.1,
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
2. Extract the 'subject' (e.g. 'Gene A') and the 'object' (e.g. 'Protein B') of the claim.
3. IMPORTANT: For both subject and object, provide an array of synonyms ('subject_synonyms' and 'object_synonyms'). This array MUST include common English and Chinese translations, academic aliases, and abbreviations.
4. Extract the 'relation' (e.g. 'inhibits').
5. Include the full sentence as 'claim' (it must be self-contained).
6. Output ONLY a JSON object with the key "claims" containing a list of these objects.
"""

class GraphMetricBase(MetricProtocol):
    """Base class for metrics that require the evidence graph and LLM."""
    
    def __init__(self, llm_config: Any = None):
        # Lazy import to avoid circularity
        from ..tools.qwen_client import QwenClient  # noqa: F811
        self._client = QwenClient.from_config(llm_config) if llm_config else None
        
    async def _decompose_hypothesis(self, hypothesis: HypothesisCard) -> List[Dict[str, str]]:
        if not self._client:
            return [{"subject": "", "relation": "", "object": "", "claim": hypothesis.statement}]
            
        schema = {
            "type": "object", 
            "properties": {
                "claims": {
                    "type": "array", 
                    "items": {
                        "type": "object",
                        "properties": {
                            "subject": {"type": "string"},
                            "subject_synonyms": {"type": "array", "items": {"type": "string"}},
                            "relation": {"type": "string"},
                            "object": {"type": "string"},
                            "object_synonyms": {"type": "array", "items": {"type": "string"}},
                            "claim": {"type": "string"}
                        },
                        "required": ["subject", "subject_synonyms", "relation", "object", "object_synonyms", "claim"]
                    }
                }
            }
        }
        try:
            result = await self._client.structured_chat(
                system_prompt=DECOMPOSE_SYSTEM_PROMPT,
                user_prompt=f"Hypothesis: {hypothesis.statement}\nMechanism: {hypothesis.mechanism}",
                output_schema=schema,
                max_tokens=2048,
                temperature=0.1
            )
            if isinstance(result, dict) and "claims" in result and result["claims"]:
                return result["claims"]
        except Exception as exc:
            logger.warning("Structured chat failed: %s. Attempting manual fallback.", exc)
            
        # Fallback to manual parsing
        try:
            import json
            import re
            raw_res = await self._client.chat(
                system_prompt=DECOMPOSE_SYSTEM_PROMPT,
                user_prompt=f"Hypothesis: {hypothesis.statement}\nMechanism: {hypothesis.mechanism}\n\nOUTPUT EXACTLY ONE JSON OBJECT WITH A 'claims' ARRAY.",
                max_tokens=2048,
                temperature=0.1
            )
            match = re.search(r'\{.*\}', raw_res, re.DOTALL)
            if match:
                parsed = json.loads(match.group(0))
                if "claims" in parsed and parsed["claims"]:
                    return parsed["claims"]
            logger.warning("Manual fallback failed to find valid claims array.")
            return [{"subject": "", "relation": "", "object": "", "claim": hypothesis.statement}]
        except Exception as exc:
            logger.warning("Failed to decompose hypothesis during fallback: %s", exc)
            return [{"subject": "", "relation": "", "object": "", "claim": hypothesis.statement}]
            
    def _keyword_search(self, queries: List[str], nodes: List[EvidenceNode]) -> List[EvidenceNode]:
        """Find all nodes that contain any of the keywords in label or metadata fields.

        Search both ``n.label`` and key text metadata fields (searchable_text,
        quote, summary, normalized_claim) — not just label — so nodes whose
        label is a single word (ENTITY) or a paper title (SOURCE) still match
        when their metadata carries the evidence content.
        """
        if not queries:
            return []
        queries_lower = [q.lower() for q in queries if q]
        if not queries_lower:
            return []

        matched = []
        for n in nodes:
            searchable_parts = [n.label.lower()]
            for key in ("searchable_text", "quote", "summary", "normalized_claim"):
                value = n.metadata.get(key, "")
                if isinstance(value, str) and value:
                    searchable_parts.append(value.lower())
            combined = " ".join(searchable_parts)
            if any(q in combined for q in queries_lower):
                matched.append(n)
                continue
                
            meta_str = str(n.metadata).lower()
            if any(q in meta_str for q in queries_lower):
                matched.append(n)
                
        return matched


@MetricRegistry.register
class NoveltyMetric(GraphMetricBase):
    metric_name = "novelty"
    metric_description = "Retrieval-grounded novelty vs. the retrieved corpus / knowledge graph."
    independent = True
    implemented = True

    async def compute(
        self,
        hypothesis: HypothesisCard,
        knowledge_entries: List[KnowledgeEntry],
        **kwargs,
    ) -> float | Tuple[float, Dict[str, Any]]:
        evidence_graph = kwargs.get("evidence_graph")
        if not evidence_graph or not self._client:
            return 0.0
            
        claims = await self._decompose_hypothesis(hypothesis)
        if not claims:
            return 0.0
            
        # [FIX]: Only use ENTITY nodes for keyword matching to avoid false positives and hub short-circuits.
        searchable_nodes = [n for n in evidence_graph.nodes if n.type == EvidenceNodeType.ENTITY]
        
        # Build networkx graph for fast BFS, omitting conflict edges
        G = nx.Graph()
        for n in evidence_graph.nodes:
            G.add_node(n.id)
        
        conflict_rels = {EvidenceEdgeRelation.CONTRADICTS, EvidenceEdgeRelation.LIMITS}
        for edge in evidence_graph.edges:
            if edge.relation not in conflict_rels:
                G.add_edge(edge.source, edge.target)
                
        total_score = 0.0
        trace_data = {"claims_novelty": []}
        for claim_obj in claims:
            subject_str = claim_obj.get("subject", "")
            subject_synonyms = claim_obj.get("subject_synonyms", [])
            object_str = claim_obj.get("object", "")
            object_synonyms = claim_obj.get("object_synonyms", [])
            
            s_a_queries = [subject_str] + subject_synonyms
            s_b_queries = [object_str] + object_synonyms
            
            s_a = self._keyword_search(s_a_queries, searchable_nodes)
            s_b = self._keyword_search(s_b_queries, searchable_nodes)
            
            if not s_a or not s_b:
                total_score += 1.0  # Concept truly missing from the literature
                trace_data["claims_novelty"].append({
                    "claim": claim_obj.get("claim", ""),
                    "start_set": [{"id": n.id, "label": n.label} for n in s_a],
                    "end_set": [{"id": n.id, "label": n.label} for n in s_b],
                    "distance": None,
                    "score": 1.0,
                })
                continue

            # Both subject and object nodes exist in the graph.
            # If they are in disconnected subgraphs (rule-graph vs grounding-graph),
            # BFS will find no path, but the concepts clearly exist — score
            # moderately novel, not fully novel.
            s_a_ids = {n.id for n in s_a}
            s_b_ids = {n.id for n in s_b}

            # Shortest path between any node in s_a and any node in s_b
            min_dist = float('inf')
            queue = deque([(node_id, 0) for node_id in s_a_ids])
            visited = set(s_a_ids)

            while queue:
                current_id, dist = queue.popleft()
                if current_id in s_b_ids:
                    min_dist = dist
                    break

                for neighbor in G.neighbors(current_id):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))

            # Map distance to novelty score.
            # disconnected subgraphs → 0.5 (concepts exist, not yet linked)
            # short paths → low novelty (already well-explored)
            # long paths → higher novelty (novel connection between distant concepts)
            if math.isinf(min_dist):
                score = 0.5
            elif min_dist <= 1:
                score = 0.0
            elif min_dist == 2:
                score = 0.4
            elif min_dist == 3:
                score = 0.6
            else:
                score = 0.8
                
            total_score += score
            trace_data["claims_novelty"].append({
                "claim": claim_obj.get("claim", ""),
                "start_set": [{"id": n.id, "label": n.label} for n in s_a],
                "end_set": [{"id": n.id, "label": n.label} for n in s_b],
                # ``None`` serializes as JSON null; Infinity is not valid JSON.
                "distance": None if math.isinf(min_dist) else min_dist,
                "score": score
            })
            
        return round(total_score / len(claims), 4), trace_data

    async def batch_compute(self, hypotheses: List[HypothesisCard], knowledge_entries: List[KnowledgeEntry], **kwargs) -> List[float]:
        results = [await self.compute(h, knowledge_entries, **kwargs) for h in hypotheses]
        return [float(result[0] if isinstance(result, tuple) else result) for result in results]


@MetricRegistry.register
class EvidenceConsistencyMetric(GraphMetricBase):
    metric_name = "evidence_consistency"
    metric_description = "Consistency with the evidence graph — does the hypothesis contradict established facts / CONTRADICTS edges?"
    independent = True
    implemented = True

    def __init__(self, llm_config: Any = None):
        super().__init__(llm_config)
        self.embeddings = None
        self.similarity_threshold = 0.8
        self._load_embedding_config()
        
    def _load_embedding_config(self):
        try:
            from pathlib import Path
            config_path = Path("configs/evaluation.yaml")
            if config_path.exists():
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)
                    emb_cfg = cfg.get("evaluation", {}).get("embedding", {})
                    self.similarity_threshold = cfg.get("evaluation", {}).get("consistency", {}).get("similarity_threshold", 0.8)
                    
                    if emb_cfg:
                        from langchain_openai import OpenAIEmbeddings
                        api_key = os.environ.get(emb_cfg.get("api_key_env_var", "DASHSCOPE_API_KEY"), "")
                        if api_key:
                            self.embeddings = OpenAIEmbeddings(
                                model=emb_cfg.get("model_name", "text-embedding-v3"),
                                api_key=api_key,
                                base_url=emb_cfg.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1")
                            )
        except Exception as e:
            logger.warning(f"Failed to load embedding config: {e}")

    def _cosine_similarity(self, vec1, vec2):
        v1, v2 = np.array(vec1), np.array(vec2)
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        return float(np.dot(v1, v2) / norm) if norm > 0 else 0.0

    async def compute(
        self,
        hypothesis: HypothesisCard,
        knowledge_entries: List[KnowledgeEntry],
        **kwargs,
    ) -> float | Tuple[float, Dict[str, Any]]:
        evidence_graph = kwargs.get("evidence_graph")
        if not evidence_graph or not self._client:
            return 0.0
            
        claims = await self._decompose_hypothesis(hypothesis)
        if not claims:
            return 0.0
            
        # [FIX]: Only use ENTITY nodes as anchors.
        searchable_nodes = [n for n in evidence_graph.nodes if n.type == EvidenceNodeType.ENTITY]
        node_embeddings = []
        if self.embeddings and searchable_nodes:
            texts = [n.label for n in searchable_nodes]
            try:
                node_embeddings = await self.embeddings.aembed_documents(texts)
            except Exception as e:
                logger.warning(f"Embedding failed, falling back: {e}")
                
        system_prompt = """You are a strict scientific reviewer.
Determine if the 'Target Claim' directly violates or ignores the provided 'Threat Context' (known conflicts/limitations from literature).
If the threat context is irrelevant to the claim, or if the claim successfully resolves the threat, there is no conflict.
Output JSON: {"is_conflict": true/false, "rationale": "..."}"""

        consistent_count = 0
        trace_data = {"atomic_claims": []}
        for claim_obj in claims:
            claim_text = claim_obj.get("claim", "")
            valid_anchors = []
            
            if self.embeddings and node_embeddings:
                try:
                    claim_emb = await self.embeddings.aembed_query(claim_text)
                    for n, n_emb in zip(searchable_nodes, node_embeddings):
                        sim = self._cosine_similarity(claim_emb, n_emb)
                        if sim >= self.similarity_threshold:
                            valid_anchors.append(n)
                except Exception as e:
                    logger.warning(f"Query embedding failed: {e}")
                    subject_queries = [claim_obj.get("subject", "")] + claim_obj.get("subject_synonyms", [])
                    valid_anchors = self._keyword_search(subject_queries, searchable_nodes)
            else:
                subject_queries = [claim_obj.get("subject", "")] + claim_obj.get("subject_synonyms", [])
                valid_anchors = self._keyword_search(subject_queries, searchable_nodes)
                
            if not valid_anchors:
                consistent_count += 1
                trace_data["atomic_claims"].append({"claim": claim_text, "matched_anchors": [], "threat_context": [], "conflict_evaluation": None})
                continue
                
            threat_nodes = self._build_threat_context(valid_anchors, evidence_graph)
            if not threat_nodes:
                consistent_count += 1
                trace_data["atomic_claims"].append({"claim": claim_text, "matched_anchors": [n.id for n in valid_anchors], "threat_context": [], "conflict_evaluation": None})
                continue
                
            threat_context_str = "\n".join(f"- {n.label}" for n in threat_nodes)
            schema = {"type": "object", "properties": {"is_conflict": {"type": "boolean"}, "rationale": {"type": "string"}}}
            
            try:
                res = await self._client.structured_chat(
                    system_prompt=system_prompt,
                    user_prompt=f"Target Claim: {claim_text}\nThreat Context:\n{threat_context_str}",
                    output_schema=schema,
                    max_tokens=2048,
                    temperature=0.1,
                )
                if isinstance(res, dict) and not res.get("is_conflict", True):
                    consistent_count += 1
                trace_data["atomic_claims"].append({"claim": claim_text, "matched_anchors": [n.id for n in valid_anchors], "threat_context": [n.id for n in threat_nodes], "conflict_evaluation": res})
            except Exception as e:
                consistent_count += 1
                trace_data["atomic_claims"].append({"claim": claim_text, "error": str(e)})
                
        return round(consistent_count / len(claims), 4), trace_data

    async def batch_compute(self, hypotheses: List[HypothesisCard], knowledge_entries: List[KnowledgeEntry], **kwargs) -> List[float]:
        results = [await self.compute(h, knowledge_entries, **kwargs) for h in hypotheses]
        return [float(result[0] if isinstance(result, tuple) else result) for result in results]
        
    def _build_threat_context(self, anchors: List[EvidenceNode], graph: EvidenceGraph) -> List[EvidenceNode]:
        anchor_ids = {n.id for n in anchors}
        threat_node_ids = set()
        
        threat_rels = {EvidenceEdgeRelation.CONTRADICTS, EvidenceEdgeRelation.LIMITS}
        ally_rels = {
            EvidenceEdgeRelation.SUPPORTS, 
            EvidenceEdgeRelation.SAME_AS,
            EvidenceEdgeRelation.EXTENDS,
            EvidenceEdgeRelation.REFINES
        }
        
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
