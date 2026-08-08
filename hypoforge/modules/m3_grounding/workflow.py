"""
M3 Evidence Grounding — consumes M2KnowledgeExport, produces claims & relations.

Architecture (Track B refactored)::

    collect_m2_evidence → index_evidence → plan_queries → retrieve_evidence
    → synthesize_evidence → atomize_claims → recall_relation_candidates
    → judge_candidate_relations → select_relations → finalize

This module does **NOT** download or parse full-text papers.  All evidence
originates from ``M2EvidenceExport`` items inside ``M2KnowledgeExport``,
which Track A's ``FullTextReadingWorkflow`` has already produced.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, TypedDict

from langgraph.graph import END, StateGraph

from hypoforge.state import (
    M2EvidenceExport,
    M2KnowledgeExport,
    PipelineState,
)
from hypoforge.tools.qwen_client import QwenClient
from .models import (
    AtomicClaim,
    EvidenceRecord,
    EvidenceRelation,
    GroundingReport,
    RelationCandidate,
    RelationPair,
)
from .evidence_gams import EvidenceGraphGAMS
from .relation_retrieval import RelationCandidateRetriever

logger = logging.getLogger(__name__)


# ============================================================================
# Utility helpers
# ============================================================================


def _hash(*parts: str, length: int = 16) -> str:
    raw = "\x1f".join(str(p) for p in parts)
    return hashlib.sha256(raw.encode("utf-8", errors="ignore")).hexdigest()[:length]


def _tokens(text: str) -> List[str]:
    """Language-agnostic sparse tokens (words plus individual CJK chars)."""
    return re.findall(r"[a-z0-9][a-z0-9_.-]*|[一-鿿]", text.lower())


def _normalise(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi <= lo:
        return [1.0 if hi > 0 else 0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


# ============================================================================
# Subgraph state
# ============================================================================


class GroundingState(TypedDict, total=False):
    pipeline_state: PipelineState
    m2_evidence_items: List[M2EvidenceExport]
    evidence_texts: List[str]
    queries: List[str]
    candidates: List[Dict[str, Any]]
    evidence_records: List[EvidenceRecord]
    claims: List[AtomicClaim]
    relation_pairs: List[RelationPair]
    relation_candidates: List[RelationCandidate]
    relations: List[EvidenceRelation]
    report: GroundingReport


# ============================================================================
# Grounding Workflow
# ============================================================================


class GroundingWorkflow:
    """Compile and execute the M3-internal evidence-grounding subgraph.

    Unlike earlier development versions, this workflow consumes
    ``M2KnowledgeExport`` (produced by Track A) and does **not** download
    or parse full-text papers.
    """

    def __init__(
        self,
        client: Optional[QwenClient] = None,
        mode: str = "rule",
        cache_dir: str = ".hypoforge_cache/m3_grounding",
        embedding_model: str = "",
        retrieve_k: int = 30,
        evidence_k: int = 12,
        min_relevance: float = 5.0,
        max_queries: int = 16,
        max_evidence_items: int = 200,
        max_concurrency: int = 4,
        relation_selection_mode: str = "direct",
        relation_candidate_k: int = 12,
        relation_max_pairs: int = 60,
        relation_batch_size: int = 10,
        relation_min_confidence: float = 0.65,
        evidence_gams_iterations: int = 128,
        evidence_gams_exploration_weight: float = 0.35,
        evidence_gams_seed: int = 42,
        llm_call_timeout: float = 120.0,
        relation_judge_retries: int = 2,
    ):
        self.client = client
        self.mode = mode
        self.cache_dir = Path(cache_dir)
        self.embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL", "")
        self.retrieve_k = max(1, retrieve_k)
        self.evidence_k = max(1, evidence_k)
        self.min_relevance = min(10.0, max(0.0, min_relevance))
        self.max_queries = max(1, max_queries)
        self.max_evidence_items = max(1, max_evidence_items)
        self.max_concurrency = max(1, max_concurrency)
        self.relation_selection_mode = (
            relation_selection_mode
            if relation_selection_mode in {"direct", "evidence_gams"}
            else "direct"
        )
        self.relation_batch_size = max(1, relation_batch_size)
        self.relation_min_confidence = min(1.0, max(0.0, relation_min_confidence))
        self.evidence_gams_iterations = max(1, evidence_gams_iterations)
        self.llm_call_timeout = max(10.0, float(llm_call_timeout))
        self.relation_judge_retries = max(0, int(relation_judge_retries))
        self.relation_retriever = RelationCandidateRetriever(
            max_candidates_per_claim=relation_candidate_k,
            max_pairs_total=relation_max_pairs,
        )
        self.evidence_gams = EvidenceGraphGAMS(
            minimum_confidence=self.relation_min_confidence,
            exploration_weight=evidence_gams_exploration_weight,
            random_seed=evidence_gams_seed,
        )
        self._compiled = self._build_graph().compile()

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(GroundingState)
        graph.add_node("collect_m2_evidence", self._collect_m2_evidence)
        graph.add_node("index_evidence", self._index_evidence)
        graph.add_node("plan_queries", self._plan_queries)
        graph.add_node("retrieve_evidence", self._retrieve_evidence)
        graph.add_node("synthesize_evidence", self._synthesize_evidence)
        graph.add_node("atomize_claims", self._atomize_claims)
        graph.add_node("recall_relation_candidates", self._recall_relation_candidates)
        graph.add_node("judge_candidate_relations", self._judge_candidate_relations)
        graph.add_node("select_relations", self._select_relations)
        graph.add_node("finalize", self._finalize)
        graph.set_entry_point("collect_m2_evidence")
        graph.add_edge("collect_m2_evidence", "index_evidence")
        graph.add_edge("index_evidence", "plan_queries")
        graph.add_edge("plan_queries", "retrieve_evidence")
        graph.add_edge("retrieve_evidence", "synthesize_evidence")
        graph.add_edge("synthesize_evidence", "atomize_claims")
        graph.add_edge("atomize_claims", "recall_relation_candidates")
        graph.add_edge("recall_relation_candidates", "judge_candidate_relations")
        graph.add_edge("judge_candidate_relations", "select_relations")
        graph.add_edge("select_relations", "finalize")
        graph.add_edge("finalize", END)
        return graph

    async def run(self, state: PipelineState) -> GroundingState:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        initial: GroundingState = {
            "pipeline_state": state,
            "report": GroundingReport(
                retrieval_backend="hybrid" if self.embedding_model else "bm25+mmr"
            ),
        }
        return await self._compiled.ainvoke(initial)

    # ==================================================================
    # Node 1: collect M2 evidence
    # ==================================================================

    async def _collect_m2_evidence(self, gs: GroundingState) -> Dict[str, Any]:
        """Collect all M2EvidenceExport items from M2KnowledgeExport.

        This replaces the old *resolve_sources* + *acquire_fulltext* +
        *parse_and_chunk* chain.  M2 has already downloaded, parsed, chunked,
        and extracted evidence — we just collect it.
        """
        state = gs["pipeline_state"]
        m2_export: Optional[M2KnowledgeExport] = state.m2_knowledge_export

        if m2_export is None or not m2_export.runs:
            report = gs["report"].model_copy(deep=True)
            report.warnings.append(
                "m2_knowledge_export is empty or None — "
                "M3 grounding requires agentic M2 (search.implementation='agentic'). "
                "Set grounding.enabled=false or run the agentic M2 pipeline first."
            )
            return {"m2_evidence_items": [], "report": report}

        evidence_items: List[M2EvidenceExport] = []
        for run in m2_export.runs:
            for item in run.evidence:
                if item.citable:
                    evidence_items.append(item)

        # Deduplicate by evidence_id (M2KnowledgeRun already validates uniqueness
        # within a run, but cross-run duplicates are possible).
        seen: set = set()
        deduped: List[M2EvidenceExport] = []
        for item in evidence_items:
            if item.evidence_id not in seen:
                seen.add(item.evidence_id)
                deduped.append(item)

        deduped = deduped[: self.max_evidence_items]

        report = gs["report"].model_copy(deep=True)
        report.papers_total = len({item.paper_id for item in deduped})
        report.full_text_papers = report.papers_total  # M2 already resolved full-text
        report.evidence_records = len(deduped)

        return {"m2_evidence_items": deduped, "report": report}

    # ==================================================================
    # Node 2: index evidence
    # ==================================================================

    async def _index_evidence(self, gs: GroundingState) -> Dict[str, Any]:
        """Build BM25-searchable text representations of each evidence item."""
        items = gs.get("m2_evidence_items", [])
        texts: List[str] = []
        for item in items:
            # Combine quote + normalized_claim + section context for retrieval
            parts = [item.quote, item.normalized_claim]
            if item.section:
                parts.append(item.section)
            texts.append(" ".join(p for p in parts if p))
        return {"evidence_texts": texts}

    # ==================================================================
    # Node 3: plan queries
    # ==================================================================

    async def _plan_queries(self, gs: GroundingState) -> Dict[str, Any]:
        """Generate retrieval queries from problem card and M2 knowledge gaps.

        Enhanced: also reads from M2KnowledgeRun.knowledge_entries to harvest
        gap/conflict statements as targeted retrieval queries.
        """
        state = gs["pipeline_state"]
        queries: List[str] = [state.input_question]

        if state.problem_card:
            queries = [
                state.problem_card.original_question,
                *state.problem_card.sub_questions,
            ]
            entities = " ".join(state.problem_card.key_entities[:12])
            if entities:
                relations = " ".join(
                    requirement.relation
                    for requirement in state.problem_card.task_contract.requirements
                    if requirement.relation
                )
                queries.extend([
                    f"{entities} {relations}".strip(),
                    f"{entities} methods evaluation metrics",
                    f"{entities} contradictory evidence limitations boundary conditions",
                ])

        # Harvest gaps and conflicts from M2KnowledgeExport knowledge_entries.
        m2_export = state.m2_knowledge_export
        if m2_export:
            for run in m2_export.runs:
                for entry in run.knowledge_entries:
                    if entry.type.value in {"knowledge_gap", "conflicting_evidence"}:
                        queries.append(entry.content)

        # Also harvest from legacy literature_results for backward compatibility.
        for lr in state.literature_results:
            for entry in lr.knowledge_entries:
                if entry.type.value in {"knowledge_gap", "conflicting_evidence"}:
                    queries.append(entry.content)

        deduped: List[str] = []
        seen: set = set()
        for query in queries:
            query = re.sub(r"\s+", " ", query or "").strip()
            key = query.lower()
            if query and key not in seen:
                seen.add(key)
                deduped.append(query)
        deduped = deduped[: self.max_queries]

        report = gs["report"].model_copy(deep=True)
        report.queries_total = len(deduped)
        return {"queries": deduped, "report": report}

    # ==================================================================
    # Node 4: retrieve evidence
    # ==================================================================

    @staticmethod
    def _bm25_scores_texts(query: str, documents: List[str]) -> List[float]:
        docs = [_tokens(doc) for doc in documents]
        q = _tokens(query)
        if not q or not docs:
            return [0.0] * len(docs)
        n = len(docs)
        avgdl = sum(len(d) for d in docs) / max(1, n)
        df = Counter(token for token in set(q) for doc in docs if token in set(doc))
        scores: List[float] = []
        for doc in docs:
            tf = Counter(doc)
            score = 0.0
            for token in q:
                freq = tf[token]
                if not freq:
                    continue
                idf = math.log(1 + (n - df[token] + 0.5) / (df[token] + 0.5))
                denom = freq + 1.2 * (1 - 0.75 + 0.75 * len(doc) / max(avgdl, 1))
                score += idf * (freq * 2.2 / denom)
            scores.append(score)
        return scores

    async def _embed(self, texts: List[str]) -> Optional[List[List[float]]]:
        if not self.embedding_model:
            return None
        try:
            import httpx
            import os as _os

            api_key = ""
            base_url = ""
            if self.client:
                api_key = getattr(self.client, "api_key", "")
                base_url = getattr(self.client, "api_base", "")
            api_key = api_key or _os.getenv("ENTITY_EMBEDDING_API_KEY", "") or _os.getenv("OPENAI_API_KEY", "")
            base_url = base_url or _os.getenv("ENTITY_EMBEDDING_BASE_URL", "") or _os.getenv("OPENAI_BASE_URL", "")
            endpoint = base_url.rstrip("/") + "/embeddings"
            # MaaS limit: 10 inputs per request.
            all_embeddings: list[list[float]] = []
            batch_size = 10
            async with httpx.AsyncClient(timeout=30.0) as cli:
                for i in range(0, len(texts), batch_size):
                    batch = texts[i : i + batch_size]
                    resp = await cli.post(
                        endpoint,
                        headers={"Authorization": f"Bearer {api_key}"},
                        json={"model": self.embedding_model, "input": batch},
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    all_embeddings.extend(
                        item["embedding"] for item in data["data"]
                    )
                return all_embeddings
        except Exception as exc:
            raise RuntimeError(
                f"Grounding embedding failed for model "
                f"{self.embedding_model!r}.  Dense retrieval is explicitly "
                f"enabled (grounding_embedding_model is set); a silent "
                f"BM25-only fallback would violate the evidence contract. "
                f"Original error: {exc}"
            ) from exc

    async def _retrieve_evidence(self, gs: GroundingState) -> Dict[str, Any]:
        """BM25 (optionally + embedding) retrieval over M2EvidenceExport items."""
        items = gs.get("m2_evidence_items", [])
        texts = gs.get("evidence_texts", [])
        queries = gs.get("queries", [])

        if not items or not queries:
            return {"candidates": []}

        # Embed once for the whole run.
        vectors = await self._embed(texts + queries)
        doc_vectors = vectors[:len(items)] if vectors else None
        query_vectors = vectors[len(items):] if vectors else None

        candidates: List[Dict[str, Any]] = []
        for qi, query in enumerate(queries):
            sparse = _normalise(self._bm25_scores_texts(query, texts))
            dense = (
                _normalise([_cosine(query_vectors[qi], v) for v in doc_vectors])
                if doc_vectors and query_vectors
                else [0.0] * len(items)
            )
            relevance = [
                0.55 * s + 0.45 * d if vectors else s
                for s, d in zip(sparse, dense)
            ]

            selected: List[int] = []
            pool = sorted(
                range(len(items)), key=lambda i: relevance[i], reverse=True
            )[: self.retrieve_k]

            while pool and len(selected) < self.evidence_k:
                def _mmr(i: int) -> float:
                    diversity = 0.0
                    for j in selected:
                        if vectors and doc_vectors:
                            sim = _cosine(doc_vectors[i], doc_vectors[j])
                        else:
                            a_tok = set(_tokens(texts[i]))
                            b_tok = set(_tokens(texts[j]))
                            sim = len(a_tok & b_tok) / max(1, len(a_tok | b_tok))
                        diversity = max(diversity, sim)
                    paper_penalty = 0.12 if any(
                        items[j].paper_id == items[i].paper_id for j in selected
                    ) else 0.0
                    return 0.75 * relevance[i] - 0.25 * diversity - paper_penalty

                best = max(pool, key=_mmr)
                pool.remove(best)
                selected.append(best)

            candidates.extend(
                {"query": query, "evidence_item": items[i], "score": relevance[i]}
                for i in selected
            )

        report = gs["report"].model_copy(deep=True)
        if self.embedding_model and not vectors:
            report.retrieval_backend = "bm25+mmr (embedding fallback)"
            report.warnings.append(
                "Configured embedding endpoint failed; sparse retrieval was used."
            )
        return {"candidates": candidates, "report": report}

    # ==================================================================
    # Node 5: synthesize evidence (RCS adapted for M2 evidence)
    # ==================================================================

    def _fallback_record(self, item: Dict[str, Any]) -> EvidenceRecord:
        """Build an EvidenceRecord from an M2EvidenceExport without LLM."""
        evidence: M2EvidenceExport = item["evidence_item"]
        score = min(10.0, max(0.0, float(item["score"]) * 10.0))
        return EvidenceRecord(
            id="ER_" + _hash(item["query"], evidence.evidence_id),
            evidence_id=evidence.evidence_id,
            paper_id=evidence.paper_id,
            chunk_id=evidence.chunk_id,
            section=evidence.section,
            page=evidence.page,
            query=item["query"],
            quote=evidence.quote,
            normalized_claim=evidence.normalized_claim,
            summary=evidence.normalized_claim,
            excerpt=evidence.quote[:1200],
            relevance_score=score,
            retrieval_score=float(item["score"]),
            claims=[evidence.normalized_claim] if evidence.normalized_claim else [],
            epistemic_status="m2_extracted",
            context={
                "section": evidence.section,
                "page": evidence.page,
                "citable": evidence.citable,
                "relevance_score_m2": evidence.relevance_score,
            },
        )

    async def _synthesize_one(self, item: Dict[str, Any]) -> EvidenceRecord:
        """LLM-enriched evidence synthesis for one retrieval candidate."""
        fallback = self._fallback_record(item)
        if not self.client or self.mode not in {"llm", "api", "direct"}:
            return fallback

        evidence: M2EvidenceExport = item["evidence_item"]
        model_name = getattr(self.client, "model", "unknown")
        cache_path = self.cache_dir / "rcs" / (
            _hash("rcs-v2", model_name, item["query"], evidence.evidence_id)
            + ".json"
        )
        if cache_path.exists():
            try:
                return EvidenceRecord.model_validate_json(
                    cache_path.read_text(encoding="utf-8")
                )
            except Exception:
                logger.debug("Ignoring invalid RCS cache file %s", cache_path)

        schema = {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "excerpt": {"type": "string"},
                "relevance_score": {
                    "type": "number", "minimum": 0, "maximum": 10,
                },
                "claims": {
                    "type": "array", "items": {"type": "string"}, "maxItems": 5,
                },
                "entities": {"type": "array", "items": {"type": "string"}},
                "methods": {"type": "array", "items": {"type": "string"}},
                "limitations": {"type": "array", "items": {"type": "string"}},
                "epistemic_status": {"type": "string"},
                "context": {"type": "object"},
            },
            "required": ["summary", "excerpt", "relevance_score", "claims"],
        }

        prompt = (
            f"Research query:\n{item['query']}\n\n"
            f"Evidence from paper {evidence.paper_id}"
            f" (section={evidence.section or 'unknown'}, page={evidence.page}):\n"
            f"Quote: {evidence.quote}\n"
            f"Normalized claim: {evidence.normalized_claim}\n\n"
            "Create a retrieval-contextual summary (RCS). The excerpt MUST be an "
            "exact quote from the evidence text. Score relevance to the query, "
            "not truth. Split claims into atomic statements. Preserve population, "
            "intervention, outcome, method, and limitations in context."
        )
        try:
            data = await self.client.structured_chat(
                system_prompt=(
                    "You extract auditable scientific evidence. "
                    "Never invent text absent from the provided evidence."
                ),
                user_prompt=prompt,
                output_schema=schema,
                max_tokens=4096,
                temperature=0.0,
                disable_thinking=True,
            )
            excerpt = str(data.get("excerpt") or "").strip()
            if excerpt and excerpt not in evidence.quote:
                excerpt = evidence.quote[:1200]
            record = EvidenceRecord(
                id=fallback.id,
                evidence_id=evidence.evidence_id,
                paper_id=evidence.paper_id,
                chunk_id=evidence.chunk_id,
                section=evidence.section,
                page=evidence.page,
                query=item["query"],
                quote=evidence.quote,
                normalized_claim=evidence.normalized_claim,
                summary=str(data.get("summary") or evidence.normalized_claim),
                excerpt=excerpt or evidence.quote[:1200],
                relevance_score=float(
                    data.get("relevance_score", fallback.relevance_score)
                ),
                retrieval_score=float(item["score"]),
                claims=list(data.get("claims") or [evidence.normalized_claim])[:5],
                entities=list(data.get("entities") or []),
                methods=list(data.get("methods") or []),
                limitations=list(data.get("limitations") or []),
                epistemic_status=str(
                    data.get("epistemic_status") or "reported"
                ),
                context={
                    **fallback.context,
                    **dict(data.get("context") or {}),
                },
            )
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(record.model_dump_json(), encoding="utf-8")
            return record
        except Exception as exc:
            logger.warning(
                "Evidence synthesis failed for %s; using fallback: %s",
                evidence.evidence_id, exc,
            )
            return fallback

    async def _synthesize_evidence(self, gs: GroundingState) -> Dict[str, Any]:
        """Run LLM evidence synthesis (RCS) on retrieval candidates."""
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def _guarded(item: Dict[str, Any]) -> EvidenceRecord:
            async with semaphore:
                try:
                    return await asyncio.wait_for(
                        self._synthesize_one(item),
                        timeout=self.llm_call_timeout,
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "Evidence synthesis timed out for %s; using fallback",
                        item.get("evidence_item", {}).evidence_id,
                    )
                    return self._fallback_record(item)

        records = await asyncio.gather(
            *(_guarded(x) for x in gs.get("candidates", []))
        )
        # Deduplicate and filter by relevance.
        unique: Dict[str, EvidenceRecord] = {}
        for record in records:
            if record.relevance_score >= self.min_relevance:
                old = unique.get(record.id)
                if old is None or record.relevance_score > old.relevance_score:
                    unique[record.id] = record
        records = sorted(
            unique.values(), key=lambda r: r.relevance_score, reverse=True
        )
        report = gs["report"].model_copy(deep=True)
        report.evidence_records = len(records)
        return {"evidence_records": records, "report": report}

    # ==================================================================
    # Node 6: atomize claims
    # ==================================================================

    async def _atomize_claims(self, gs: GroundingState) -> Dict[str, Any]:
        """Deduplicate and merge claims from evidence records into AtomicClaims.

        Each AtomicClaim.evidence_ids points to M2EvidenceExport.evidence_id
        (via the EvidenceRecord.evidence_id bridge).
        """
        claims: Dict[str, AtomicClaim] = {}
        for record in gs.get("evidence_records", []):
            for statement in record.claims:
                statement = re.sub(r"\s+", " ", statement).strip()
                if len(statement) < 20:
                    continue
                key = re.sub(r"\W+", "", statement).lower()
                claim = claims.get(key)
                if claim:
                    if record.evidence_id not in claim.evidence_ids:
                        claim.evidence_ids.append(record.evidence_id)
                    if record.paper_id not in claim.paper_ids:
                        claim.paper_ids.append(record.paper_id)
                    claim.entities = list(
                        dict.fromkeys(claim.entities + record.entities)
                    )
                    claim.confidence = min(1.0, claim.confidence + 0.1)
                else:
                    claims[key] = AtomicClaim(
                        id="CLM_" + _hash(statement),
                        statement=statement,
                        evidence_ids=[record.evidence_id],
                        paper_ids=[record.paper_id] if record.paper_id else [],
                        entities=record.entities,
                        confidence=min(
                            0.95, 0.35 + record.relevance_score / 15
                        ),
                    )
        claim_list = list(claims.values())
        report = gs["report"].model_copy(deep=True)
        report.claims_total = len(claim_list)
        return {"claims": claim_list, "report": report}

    # ==================================================================
    # Node 7: recall relation candidates
    # ==================================================================

    async def _recall_relation_candidates(
        self, gs: GroundingState
    ) -> Dict[str, Any]:
        pairs = self.relation_retriever.recall(
            gs.get("claims", []), gs.get("evidence_records", [])
        )
        report = gs["report"].model_copy(deep=True)
        report.relation_pairs_recalled = len(pairs)
        return {"relation_pairs": pairs, "report": report}

    # ==================================================================
    # Node 8: judge candidate relations
    # ==================================================================

    @staticmethod
    def _claim_payload(
        claim: AtomicClaim,
        record_map: Dict[str, EvidenceRecord],
    ) -> Dict[str, Any]:
        evidence = []
        paper_ids: List[str] = []
        for evidence_id in claim.evidence_ids[:3]:
            # Look up the EvidenceRecord that wraps this M2 evidence_id.
            record = record_map.get(evidence_id)
            if record is None:
                # The evidence_id IS the record id (EvidenceRecord.id is
                # "ER_" + hash, but evidence_id is the M2EvidenceExport id).
                # Try finding by matching evidence_id field.
                for rec in record_map.values():
                    if rec.evidence_id == evidence_id:
                        record = rec
                        break
            if record is None:
                continue
            if record.paper_id:
                paper_ids.append(record.paper_id)
            evidence.append({
                "evidence_id": record.evidence_id,
                "paper_id": record.paper_id,
                "query": record.query,
                "summary": record.summary[:1000],
                "excerpt": record.excerpt[:1200],
                "entities": record.entities[:20],
                "methods": record.methods[:8],
                "limitations": record.limitations[:8],
                "context": record.context,
                "relevance_score": record.relevance_score,
            })
        return {
            "node_type": "claim",
            "id": claim.id,
            "statement": claim.statement,
            "entities": claim.entities,
            "paper_ids": list(dict.fromkeys(paper_ids)),
            "evidence": evidence,
        }

    @staticmethod
    def _evidence_payload(record: EvidenceRecord) -> Dict[str, Any]:
        return {
            "node_type": "evidence_record",
            "id": record.id,
            "evidence_id": record.evidence_id,
            "paper_id": record.paper_id,
            "query": record.query,
            "summary": record.summary[:1000],
            "excerpt": record.excerpt[:1200],
            "claims": record.claims[:5],
            "entities": record.entities[:20],
            "methods": record.methods[:8],
            "limitations": record.limitations[:8],
            "context": record.context,
            "relevance_score": record.relevance_score,
        }

    def _relation_node_payload(
        self,
        node_id: str,
        claim_map: Dict[str, AtomicClaim],
        record_map: Dict[str, EvidenceRecord],
    ) -> Dict[str, Any]:
        if node_id in claim_map:
            return self._claim_payload(claim_map[node_id], record_map)
        if node_id in record_map:
            return self._evidence_payload(record_map[node_id])
        raise KeyError(node_id)

    async def _judge_relation_batch(
        self,
        pairs: List[RelationPair],
        claim_map: Dict[str, AtomicClaim],
        record_map: Dict[str, EvidenceRecord],
    ) -> List[RelationCandidate]:
        if not self.client:
            return []
        cached: List[RelationCandidate] = []
        pending: List[RelationPair] = []
        cache_paths: Dict[str, Path] = {}
        model_name = getattr(self.client, "model", "unknown")
        for pair in pairs:
            source_payload = self._relation_node_payload(
                pair.source, claim_map, record_map
            )
            target_payload = self._relation_node_payload(
                pair.target, claim_map, record_map
            )
            cache_path = self.cache_dir / "relations" / (
                _hash(
                    "relation-v2", model_name, pair.id,
                    json.dumps(source_payload, ensure_ascii=False, sort_keys=True),
                    json.dumps(target_payload, ensure_ascii=False, sort_keys=True),
                ) + ".json"
            )
            cache_paths[pair.id] = cache_path
            if cache_path.exists():
                try:
                    cached.append(RelationCandidate.model_validate_json(
                        cache_path.read_text(encoding="utf-8")
                    ))
                    continue
                except Exception:
                    logger.debug(
                        "Ignoring invalid relation cache file %s", cache_path
                    )
            pending.append(pair)

        pairs = pending
        if not pairs:
            return cached

        schema = {
            "type": "object",
            "properties": {
                "relations": {
                    "type": "array",
                    "maxItems": len(pairs),
                    "items": {
                        "type": "object",
                        "properties": {
                            "pair_id": {"type": "string"},
                            "source": {"type": "string"},
                            "target": {"type": "string"},
                            "relation": {
                                "type": "string",
                                "enum": [
                                    "supports", "contradicts", "extends",
                                    "limits", "same_as", "refines",
                                    "unrelated",
                                ],
                            },
                            "confidence": {
                                "type": "number", "minimum": 0, "maximum": 1,
                            },
                            "condition_comparability": {
                                "type": "number", "minimum": 0, "maximum": 1,
                            },
                            "rationale": {"type": "string"},
                            "evidence_ids": {
                                "type": "array", "items": {"type": "string"},
                            },
                        },
                        "required": [
                            "pair_id", "source", "target", "relation",
                            "confidence", "condition_comparability",
                            "rationale", "evidence_ids",
                        ],
                    },
                },
            },
            "required": ["relations"],
        }

        payload = []
        for pair in pairs:
            payload.append({
                "pair_id": pair.id,
                "pair_type": f"{pair.source_type}_to_{pair.target_type}",
                "retrieval_score": pair.retrieval_score,
                "entity_overlap": pair.entity_overlap,
                "candidate_origin": pair.candidate_origin,
                "source": self._relation_node_payload(
                    pair.source, claim_map, record_map
                ),
                "target": self._relation_node_payload(
                    pair.target, claim_map, record_map
                ),
            })

        data = await self.client.structured_chat(
            system_prompt=(
                "Judge each recalled scientific pair conservatively. "
                "Return one decision per pair. "
                "Use UNRELATED when no defensible semantic relation exists. "
                "For EVIDENCE_RECORD_TO_CLAIM, only SUPPORTS, CONTRADICTS, "
                "LIMITS or UNRELATED are allowed and the evidence record must "
                "remain the source. For CLAIM_TO_CLAIM, never use SUPPORTS; "
                "only CONTRADICTS, EXTENDS, LIMITS, SAME_AS, REFINES or "
                "UNRELATED are allowed. CONTRADICTS requires incompatible "
                "conclusions under comparable population, intervention, "
                "outcome, dose, time and method. If conditions differ and "
                "one result only narrows another, use LIMITS. Direction "
                "matters: source SUPPORTS/EXTENDS/REFINES target. "
                "Cite only supplied evidence IDs."
            ),
            user_prompt="Recalled pairs:\n" + json.dumps(payload, ensure_ascii=False),
            output_schema=schema,
            max_tokens=max(6000, 1800 * len(pairs)),
            temperature=0.0,
            disable_thinking=True,
        )

        pair_map = {pair.id: pair for pair in pairs}
        decisions: List[RelationCandidate] = []
        for raw_item in data.get("relations", []):
            pair = pair_map.get(str(raw_item.get("pair_id") or ""))
            if not pair:
                continue
            source = str(raw_item.get("source") or "")
            target = str(raw_item.get("target") or "")
            source_payload = self._relation_node_payload(
                pair.source, claim_map, record_map
            )
            target_payload = self._relation_node_payload(
                pair.target, claim_map, record_map
            )
            endpoint_statements: Dict[str, str] = {}
            for endpoint_id, endpoint_payload in (
                (pair.source, source_payload),
                (pair.target, target_payload),
            ):
                for field in ("statement", "summary", "excerpt"):
                    value = str(endpoint_payload.get(field) or "").strip()
                    if value:
                        endpoint_statements[value] = endpoint_id
            source = endpoint_statements.get(source.strip(), source)
            target = endpoint_statements.get(target.strip(), target)
            if {source, target} != {pair.source, pair.target} or source == target:
                continue
            if pair.source_type == "evidence_record":
                source, target = pair.source, pair.target
            evidence_ids = [
                eid for eid in raw_item.get("evidence_ids", [])
                if eid in set(pair.source_evidence_ids + pair.target_evidence_ids)
            ]
            if not evidence_ids:
                evidence_ids = list(dict.fromkeys(
                    pair.source_evidence_ids + pair.target_evidence_ids
                ))
            source_papers = (
                pair.source_paper_ids
                if source == pair.source
                else pair.target_paper_ids
            )
            target_papers = (
                pair.target_paper_ids
                if target == pair.target
                else pair.source_paper_ids
            )
            relation = str(raw_item.get("relation") or "unrelated")
            allowed_relations = (
                {"supports", "contradicts", "limits", "unrelated"}
                if pair.source_type == "evidence_record"
                else {
                    "contradicts", "extends", "limits",
                    "same_as", "refines", "unrelated",
                }
            )
            rationale = str(raw_item.get("rationale") or "")
            if relation not in allowed_relations:
                rationale = (
                    f"Type constraint rejected {relation} for "
                    f"{pair.source_type}_to_{pair.target_type}. " + rationale
                )
                relation = "unrelated"
            candidate = RelationCandidate(
                id="REL_" + _hash(pair.id, source, target, relation),
                source=source,
                target=target,
                relation=relation,  # type: ignore[arg-type]
                confidence=float(raw_item.get("confidence", 0.0)),
                condition_comparability=float(
                    raw_item.get("condition_comparability", 0.5)
                ),
                rationale=rationale,
                evidence_ids=evidence_ids,
                source_paper_ids=source_papers,
                target_paper_ids=target_papers,
                retrieval_score=pair.retrieval_score,
                entity_overlap=pair.entity_overlap,
                candidate_origin=pair.candidate_origin,
            )
            decisions.append(candidate)
            cache_path = cache_paths.get(pair.id)
            if cache_path:
                try:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    cache_path.write_text(
                        candidate.model_dump_json(), encoding="utf-8"
                    )
                except OSError:
                    logger.debug(
                        "Could not write relation cache file %s", cache_path
                    )
        if len(decisions) != len(pairs):
            logger.warning(
                "Relation validation retained %d/%d decisions for pairs %s; raw=%s",
                len(decisions), len(pairs),
                [pair.id for pair in pairs],
                json.dumps(data, ensure_ascii=False)[:4000],
            )
        return cached + decisions

    async def _judge_candidate_relations(
        self, gs: GroundingState
    ) -> Dict[str, Any]:
        pairs = gs.get("relation_pairs", [])
        candidates: List[RelationCandidate] = []
        if self.client and self.mode in {"llm", "api", "direct"} and pairs:
            claim_map = {claim.id: claim for claim in gs.get("claims", [])}
            record_map = {
                record.id: record for record in gs.get("evidence_records", [])
            }
            batches = [
                pairs[index:index + self.relation_batch_size]
                for index in range(0, len(pairs), self.relation_batch_size)
            ]
            semaphore = asyncio.Semaphore(self.max_concurrency)

            async def _guarded(
                batch: List[RelationPair],
            ) -> List[RelationCandidate]:
                async with semaphore:
                    for attempt in range(self.relation_judge_retries + 1):
                        try:
                            decisions = await asyncio.wait_for(
                                self._judge_relation_batch(
                                    batch, claim_map, record_map
                                ),
                                timeout=self.llm_call_timeout,
                            )
                            if len(decisions) == len(batch):
                                return decisions
                            logger.warning(
                                "Relation batch returned %d/%d decisions "
                                "(attempt %d)",
                                len(decisions), len(batch), attempt + 1,
                            )
                        except asyncio.TimeoutError:
                            logger.warning(
                                "Relation candidate batch timed out for "
                                "%d pair(s) (attempt %d)",
                                len(batch), attempt + 1,
                            )
                        except Exception as exc:
                            logger.warning(
                                "Relation candidate batch failed (attempt %d): %s",
                                attempt + 1, exc,
                            )
                    return []

            results = await asyncio.gather(
                *(_guarded(batch) for batch in batches)
            )
            deduplicated: Dict[tuple, RelationCandidate] = {}
            for candidate in [
                item for batch in results for item in batch
            ]:
                key = (candidate.source, candidate.target, candidate.relation)
                old = deduplicated.get(key)
                if old is None or candidate.confidence > old.confidence:
                    deduplicated[key] = candidate
            candidates = list(deduplicated.values())

        report = gs["report"].model_copy(deep=True)
        report.relation_candidates_judged = len(candidates)
        return {"relation_candidates": candidates, "report": report}

    # ==================================================================
    # Node 9: select relations
    # ==================================================================

    @staticmethod
    def _provenance_relations(
        claims: List[AtomicClaim],
        records: List[EvidenceRecord],
    ) -> List[EvidenceRelation]:
        """Create provenance edges linking each claim to its source evidence."""
        # Build lookup: evidence_id → record
        record_by_evidence: Dict[str, EvidenceRecord] = {}
        for record in records:
            record_by_evidence[record.evidence_id] = record

        relations: List[EvidenceRelation] = []
        for claim in claims:
            for evidence_id in claim.evidence_ids:
                record = record_by_evidence.get(evidence_id)
                rel_id = "REL_" + _hash(evidence_id, claim.id, "supports")
                relations.append(EvidenceRelation(
                    id=rel_id,
                    source=evidence_id,
                    target=claim.id,
                    relation="supports",
                    confidence=claim.confidence,
                    rationale="Atomic claim grounded in this M2 evidence item.",
                    evidence_ids=[evidence_id],
                    source_paper_ids=(
                        [record.paper_id] if record and record.paper_id else []
                    ),
                    target_paper_ids=(
                        [record.paper_id] if record and record.paper_id else []
                    ),
                ))
        return relations

    async def _select_relations(self, gs: GroundingState) -> Dict[str, Any]:
        candidates = gs.get("relation_candidates", [])
        provenance = self._provenance_relations(
            gs.get("claims", []), gs.get("evidence_records", [])
        )
        if self.relation_selection_mode == "evidence_gams":
            selected_candidates, trace = self.evidence_gams.search(
                candidates, iterations=self.evidence_gams_iterations
            )
        else:
            selected_candidates = [
                c for c in candidates
                if c.relation != "unrelated"
                and c.confidence >= self.relation_min_confidence
            ]
            trace = {
                "mode": "direct",
                "candidate_count": len([
                    c for c in candidates if c.relation != "unrelated"
                ]),
                "selected_count": len(selected_candidates),
                "minimum_confidence": self.relation_min_confidence,
                "selected_edge_ids": sorted(
                    c.id for c in selected_candidates
                ),
                "rejected_edge_ids": sorted(
                    c.id for c in candidates
                    if c not in selected_candidates
                ),
            }
        semantic_relations = [
            c.to_relation() for c in selected_candidates
        ]
        relations = provenance + semantic_relations
        report = gs["report"].model_copy(deep=True)
        report.relation_selection_mode = self.relation_selection_mode
        report.relation_candidates_selected = len(selected_candidates)
        report.relations_total = len(relations)
        report.relation_search = trace
        return {"relations": relations, "report": report}

    # ==================================================================
    # Node 10: finalize
    # ==================================================================

    async def _finalize(self, gs: GroundingState) -> Dict[str, Any]:
        report = gs["report"].model_copy(deep=True)
        if not gs.get("evidence_records"):
            report.warnings.append(
                "No evidence passed the relevance threshold; "
                "M3 will retain its seed graph."
            )
        manifest = self.cache_dir / "last_grounding_report.json"
        try:
            manifest.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        except Exception as exc:
            logger.debug("Could not write grounding report: %s", exc)
        return {"report": report}
