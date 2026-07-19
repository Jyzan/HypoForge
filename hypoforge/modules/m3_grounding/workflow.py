"""PaperQA-inspired full-text grounding as a nested LangGraph workflow.

This module deliberately lives inside M3: the top-level M1 -> M2 -> M3 -> M4
contract is unchanged.  It uses lawful/open sources only and records every
fallback so an abstract-derived record is never mistaken for full-text proof.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, TypedDict

import httpx
from langgraph.graph import END, StateGraph

from ...state import LiteratureResult, PipelineState
from ...tools.qwen_client import QwenClient
from .models import (
    AtomicClaim,
    EvidenceRecord,
    EvidenceRelation,
    FullTextChunk,
    GroundingReport,
    PaperSource,
    RelationCandidate,
    RelationPair,
)
from .evidence_gams import EvidenceGraphGAMS
from .relation_retrieval import RelationCandidateRetriever

logger = logging.getLogger(__name__)


def _hash(*parts: str, length: int = 16) -> str:
    raw = "\x1f".join(str(p) for p in parts)
    return hashlib.sha256(raw.encode("utf-8", errors="ignore")).hexdigest()[:length]


def _tokens(text: str) -> List[str]:
    """Language-agnostic sparse tokens (words plus individual CJK chars)."""
    return re.findall(r"[a-z0-9][a-z0-9_.-]*|[\u4e00-\u9fff]", text.lower())


def _compact_identifier(text: str) -> str:
    """Normalize IDs/filenames; unlike ``\\W``, this also removes underscores."""
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", text.lower())


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


def _is_retrievable_chunk(chunk: FullTextChunk) -> bool:
    """Exclude bibliography-like sections from evidence retrieval.

    BioC XML exposes each reference as a text passage.  Those passages often
    score well lexically because titles repeat the query terms, but they are
    not evidence from the paper's own results or discussion.
    """
    section = (chunk.section or "").strip().upper()
    blocked = {"REF", "REFERENCE", "REFERENCES", "BIBLIOGRAPHY", "ACK", "ACKNOWLEDGMENTS"}
    return section not in blocked


def _is_error_payload(payload: bytes) -> bool:
    """Recognise the small HTML/text error pages returned by OA providers."""
    prefix = payload.lstrip()[:2000].lower()
    return (
        prefix.startswith(b"[error]")
        or prefix.startswith(b"<html")
        or b"no result can be found" in prefix
    )


class GroundingState(TypedDict, total=False):
    pipeline_state: PipelineState
    sources: List[PaperSource]
    chunks: List[FullTextChunk]
    queries: List[str]
    candidates: List[Dict[str, Any]]
    evidence_records: List[EvidenceRecord]
    claims: List[AtomicClaim]
    relation_pairs: List[RelationPair]
    relation_candidates: List[RelationCandidate]
    relations: List[EvidenceRelation]
    report: GroundingReport


class FullTextEvidenceGrounding:
    """Compile and execute the M3-internal evidence-grounding subgraph."""

    def __init__(
        self,
        client: Optional[QwenClient] = None,
        mode: str = "rule",
        cache_dir: str = ".hypoforge_cache/m3_grounding",
        local_paper_dir: str = "papers",
        enable_network_fulltext: bool = False,
        embedding_model: str = "",
        chunk_chars: int = 9000,
        chunk_overlap_chars: int = 1000,
        retrieve_k: int = 30,
        evidence_k: int = 12,
        min_relevance: float = 5.0,
        max_queries: int = 16,
        max_sources: int = 60,
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
        self.local_paper_dir = Path(local_paper_dir)
        self.enable_network_fulltext = enable_network_fulltext
        self.embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL", "")
        self.chunk_chars = max(1500, chunk_chars)
        self.chunk_overlap_chars = min(max(0, chunk_overlap_chars), self.chunk_chars // 2)
        self.retrieve_k = max(1, retrieve_k)
        self.evidence_k = max(1, evidence_k)
        self.min_relevance = min(10.0, max(0.0, min_relevance))
        self.max_queries = max(1, max_queries)
        self.max_sources = max(1, max_sources)
        self.max_concurrency = max(1, max_concurrency)
        self.relation_selection_mode = (
            relation_selection_mode if relation_selection_mode in {"direct", "evidence_gams"}
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

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(GroundingState)
        graph.add_node("resolve_sources", self._resolve_sources)
        graph.add_node("acquire_fulltext", self._acquire_fulltext)
        graph.add_node("parse_and_chunk", self._parse_and_chunk)
        graph.add_node("plan_queries", self._plan_queries)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("rcs", self._rcs)
        graph.add_node("atomize_claims", self._atomize_claims)
        graph.add_node("recall_relation_candidates", self._recall_relation_candidates)
        graph.add_node("judge_candidate_relations", self._judge_candidate_relations)
        graph.add_node("select_relations", self._select_relations)
        graph.add_node("finalize", self._finalize)
        graph.set_entry_point("resolve_sources")
        graph.add_edge("resolve_sources", "acquire_fulltext")
        graph.add_edge("acquire_fulltext", "parse_and_chunk")
        graph.add_edge("parse_and_chunk", "plan_queries")
        graph.add_edge("plan_queries", "retrieve")
        graph.add_edge("retrieve", "rcs")
        graph.add_edge("rcs", "atomize_claims")
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

    async def _resolve_sources(self, gs: GroundingState) -> Dict[str, Any]:
        state = gs["pipeline_state"]
        grouped: Dict[str, PaperSource] = {}
        for lr in state.literature_results:
            for entry in lr.knowledge_entries:
                sid = (entry.source_paper_id or entry.source_paper_title or entry.id).strip()
                if not sid:
                    continue
                source = grouped.setdefault(
                    sid,
                    PaperSource(id=sid, title=entry.source_paper_title or sid),
                )
                source.seed_entry_ids.append(entry.id)
                source.seed_text += ("\n" if source.seed_text else "") + entry.content
                upper = sid.upper()
                if upper.startswith("PMID:"):
                    source.pmid = sid.split(":", 1)[1]
                elif upper.startswith("DOI:"):
                    source.doi = sid.split(":", 1)[1]
                elif sid.startswith("10."):
                    source.doi = sid
                elif sid.startswith("http"):
                    source.url = sid
        sources = list(grouped.values())[: self.max_sources]
        report = gs["report"].model_copy(deep=True)
        report.papers_total = len(sources)
        return {"sources": sources, "report": report}

    def _local_match(self, source: PaperSource) -> Optional[Path]:
        if not self.local_paper_dir.exists():
            return None
        keys = [_hash(source.id, length=8), _compact_identifier(source.id)]
        title_key = _compact_identifier(source.title)[:40]
        for path in self.local_paper_dir.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {".pdf", ".txt", ".xml"}:
                continue
            stem = _compact_identifier(path.stem)
            if any(k and k in stem for k in keys) or (title_key and title_key in stem):
                return path
        return None

    async def _download(self, url: str, target: Path) -> bool:
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=45) as http:
                response = await http.get(url, headers={"User-Agent": "HypoForge/0.3 (open-access research client)"})
                response.raise_for_status()
                content_type = response.headers.get("content-type", "")
                is_error_page = _is_error_payload(response.content)
                if (len(response.content) < 500 or is_error_page
                        or ("html" in content_type and target.suffix in {".pdf", ".xml"})):
                    return False
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(response.content)
                return True
        except Exception as exc:
            logger.debug("Full-text download failed for %s: %s", url, exc)
            return False

    async def _openalex_oa_url(self, source: PaperSource) -> str:
        identifier = ""
        if source.doi:
            identifier = "https://doi.org/" + source.doi
        elif "openalex.org/" in source.url:
            identifier = source.url.rsplit("/", 1)[-1]
        elif "openalex.org/" in source.id:
            identifier = source.id.rsplit("/", 1)[-1]
        if not identifier:
            return ""
        try:
            url = "https://api.openalex.org/works/" + identifier
            async with httpx.AsyncClient(follow_redirects=True, timeout=25) as http:
                data = (await http.get(url)).json()
            source.doi = source.doi or str(data.get("doi") or "").replace("https://doi.org/", "")
            best = data.get("best_oa_location") or {}
            return str(best.get("pdf_url") or best.get("landing_page_url") or "")
        except Exception:
            return ""

    async def _pmc_xml_url(self, source: PaperSource) -> str:
        if not source.pmid:
            return ""
        try:
            params = {"dbfrom": "pubmed", "db": "pmc", "id": source.pmid, "retmode": "json"}
            async with httpx.AsyncClient(timeout=25) as http:
                data = (await http.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi", params=params)).json()
            links = data.get("linksets", [{}])[0].get("linksetdbs", [])
            pmcid = next((str(x) for db in links for x in db.get("links", [])), "")
            return f"https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_xml/PMC{pmcid}/unicode" if pmcid else ""
        except Exception:
            return ""

    async def _acquire_one(self, source: PaperSource) -> PaperSource:
        local = self._local_match(source)
        if local:
            source.full_text_path = str(local.resolve())
            source.acquisition_status = "local_fulltext"
            return source

        paper_cache = self.cache_dir / "papers"
        for suffix in (".pdf", ".xml", ".txt"):
            cached = paper_cache / f"{_hash(source.id)}{suffix}"
            if cached.exists() and cached.stat().st_size > 500:
                try:
                    with cached.open("rb") as handle:
                        is_valid = not _is_error_payload(handle.read(2000))
                except OSError:
                    is_valid = False
                if is_valid:
                    source.full_text_path = str(cached)
                    source.acquisition_status = "cached_fulltext"
                    return source

        if self.enable_network_fulltext:
            # PMC can return an error page for non-OA records.  Do not let that
            # prevent the OpenAlex OA location from being tried afterwards.
            urls = [await self._pmc_xml_url(source), await self._openalex_oa_url(source)]
            for url in dict.fromkeys(url for url in urls if url):
                suffix = ".pdf" if ".pdf" in url.lower() else ".xml"
                target = paper_cache / f"{_hash(source.id)}{suffix}"
                if await self._download(url, target):
                    source.full_text_path = str(target)
                    source.url = url
                    source.acquisition_status = "open_access_fulltext"
                    return source

        # M2 no longer carries raw abstracts, so this is explicitly a seed-entry fallback.
        source.acquisition_status = "m2_seed_fallback"
        source.acquisition_note = "No lawful full text found; grounded only against M2-extracted seed text."
        return source

    async def _acquire_fulltext(self, gs: GroundingState) -> Dict[str, Any]:
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def guarded(source: PaperSource) -> PaperSource:
            async with semaphore:
                return await self._acquire_one(source.model_copy(deep=True))

        sources = await asyncio.gather(*(guarded(s) for s in gs.get("sources", [])))
        report = gs["report"].model_copy(deep=True)
        report.paper_status = {s.id: s.acquisition_status for s in sources}
        report.full_text_papers = sum("fulltext" in s.acquisition_status for s in sources)
        report.fallback_papers = sum(s.acquisition_status.endswith("fallback") for s in sources)
        if report.fallback_papers:
            report.warnings.append(
                f"{report.fallback_papers} paper(s) lacked accessible full text and used M2 seed text."
            )
        return {"sources": sources, "report": report}

    def _extract_document(self, source: PaperSource) -> List[tuple[str, int, str]]:
        """Return (section, page, text), preserving the strongest available locator."""
        if not source.full_text_path:
            return [("M2 seed fallback", 0, source.seed_text)] if source.seed_text else []
        path = Path(source.full_text_path)
        if path.suffix.lower() == ".pdf":
            try:
                from pypdf import PdfReader
                return [("", i + 1, page.extract_text() or "") for i, page in enumerate(PdfReader(str(path)).pages)]
            except Exception as exc:
                logger.warning("Could not parse PDF %s: %s", path, exc)
                return [("M2 seed fallback", 0, source.seed_text)] if source.seed_text else []
        if path.suffix.lower() == ".xml":
            try:
                root = ET.parse(path).getroot()
                records: List[tuple[str, int, str]] = []
                for passage in root.findall(".//passage"):
                    section = passage.findtext("infon[@key='section_type']") or passage.findtext("infon[@key='type']") or ""
                    text = passage.findtext("text") or ""
                    if text.strip():
                        records.append((section, 0, text))
                if records:
                    return records
                return [("", 0, " ".join(root.itertext()))]
            except Exception as exc:
                logger.warning("Could not parse XML %s: %s", path, exc)
        try:
            return [("", 0, path.read_text(encoding="utf-8", errors="ignore"))]
        except Exception:
            return [("M2 seed fallback", 0, source.seed_text)] if source.seed_text else []

    def _chunk_source(self, source: PaperSource) -> List[FullTextChunk]:
        chunks: List[FullTextChunk] = []
        for section, page, text in self._extract_document(source):
            text = re.sub(r"[ \t]+", " ", text).strip()
            if not text:
                continue
            step = self.chunk_chars - self.chunk_overlap_chars
            for start in range(0, len(text), step):
                piece = text[start:start + self.chunk_chars].strip()
                if len(piece) < 120 and chunks:
                    chunks[-1].text += " " + piece
                    chunks[-1].end_char += len(piece)
                    continue
                cid = "CHK_" + _hash(source.id, str(page), str(start), piece[:100])
                chunks.append(FullTextChunk(
                    id=cid, paper_id=source.id, text=piece, section=section,
                    page=page or None, start_char=start, end_char=start + len(piece),
                    source_path=source.full_text_path,
                ))
                if start + self.chunk_chars >= len(text):
                    break
        return chunks

    async def _parse_and_chunk(self, gs: GroundingState) -> Dict[str, Any]:
        chunks = [c for source in gs.get("sources", []) for c in self._chunk_source(source)]
        report = gs["report"].model_copy(deep=True)
        report.chunks_total = len(chunks)
        return {"chunks": chunks, "report": report}

    async def _plan_queries(self, gs: GroundingState) -> Dict[str, Any]:
        state = gs["pipeline_state"]
        queries: List[str] = [state.input_question]
        if state.problem_card:
            queries = [state.problem_card.original_question, *state.problem_card.sub_questions]
            entities = " ".join(state.problem_card.key_entities[:12])
            if entities:
                queries.extend([
                    f"{entities} molecular mechanism causal pathway",
                    f"{entities} experimental method population outcome",
                    f"{entities} contradictory evidence limitations boundary conditions",
                ])
        # M2 gaps and conflicts are valuable retrieval queries, but avoid exploding query count.
        for lr in state.literature_results:
            for entry in lr.knowledge_entries:
                if entry.type.value in {"knowledge_gap", "conflicting_evidence"}:
                    queries.append(entry.content)
        deduped: List[str] = []
        seen = set()
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

    @staticmethod
    def _bm25_scores(query: str, chunks: List[FullTextChunk]) -> List[float]:
        docs = [_tokens(c.text) for c in chunks]
        q = _tokens(query)
        if not q or not docs:
            return [0.0] * len(docs)
        n = len(docs)
        avgdl = sum(len(d) for d in docs) / max(1, n)
        df = Counter(token for token in set(q) for doc in docs if token in set(doc))
        scores = []
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
            from langchain_openai import OpenAIEmbeddings
            kwargs: Dict[str, Any] = {"model": self.embedding_model}
            if self.client:
                kwargs.update({"api_key": self.client.api_key, "base_url": self.client.api_base})
            return await OpenAIEmbeddings(**kwargs).aembed_documents(texts)
        except Exception as exc:
            logger.warning("Embedding unavailable; continuing with sparse retrieval: %s", exc)
            return None

    async def _retrieve(self, gs: GroundingState) -> Dict[str, Any]:
        chunks, queries = gs.get("chunks", []), gs.get("queries", [])
        if not chunks or not queries:
            return {"candidates": []}
        # Embed once for the whole run. Query vectors are requested in the same call
        # for maximum compatibility with OpenAI-compatible endpoints.
        vectors = await self._embed([c.text for c in chunks] + queries)
        doc_vectors = vectors[:len(chunks)] if vectors else None
        query_vectors = vectors[len(chunks):] if vectors else None
        candidates: List[Dict[str, Any]] = []
        for qi, query in enumerate(queries):
            sparse = _normalise(self._bm25_scores(query, chunks))
            dense = (
                _normalise([_cosine(query_vectors[qi], v) for v in doc_vectors])
                if doc_vectors and query_vectors else [0.0] * len(chunks)
            )
            relevance = [0.55 * s + 0.45 * d if vectors else s for s, d in zip(sparse, dense)]
            selected: List[int] = []
            eligible = [i for i, chunk in enumerate(chunks) if _is_retrievable_chunk(chunk)]
            # Malformed documents can expose only an unlabelled/reference-like
            # body. In that case retain the old fallback instead of returning nothing.
            if not eligible:
                eligible = list(range(len(chunks)))
            pool = sorted(eligible, key=lambda i: relevance[i], reverse=True)[: self.retrieve_k]
            while pool and len(selected) < self.evidence_k:
                def mmr(i: int) -> float:
                    diversity = 0.0
                    for j in selected:
                        if vectors and doc_vectors:
                            similarity = _cosine(doc_vectors[i], doc_vectors[j])
                        else:
                            a, b = set(_tokens(chunks[i].text)), set(_tokens(chunks[j].text))
                            similarity = len(a & b) / max(1, len(a | b))
                        diversity = max(diversity, similarity)
                    # Mild cross-paper reward prevents one long review dominating context.
                    paper_penalty = 0.12 if any(chunks[j].paper_id == chunks[i].paper_id for j in selected) else 0.0
                    return 0.75 * relevance[i] - 0.25 * diversity - paper_penalty
                best = max(pool, key=mmr)
                pool.remove(best)
                selected.append(best)
            candidates.extend({"query": query, "chunk": chunks[i], "score": relevance[i]} for i in selected)
        report = gs["report"].model_copy(deep=True)
        if self.embedding_model and not vectors:
            report.retrieval_backend = "bm25+mmr (embedding fallback)"
            report.warnings.append("Configured embedding endpoint failed; sparse retrieval was used.")
        return {"candidates": candidates, "report": report}

    def _fallback_record(self, item: Dict[str, Any]) -> EvidenceRecord:
        chunk: FullTextChunk = item["chunk"]
        sentences = re.split(r"(?<=[.!?。！？])\s+", chunk.text)
        excerpt = max(sentences, key=lambda s: len(set(_tokens(s)) & set(_tokens(item["query"])))) if sentences else chunk.text[:800]
        score = min(10.0, max(0.0, float(item["score"]) * 10.0))
        return EvidenceRecord(
            id="ER_" + _hash(item["query"], chunk.id), chunk_id=chunk.id,
            paper_id=chunk.paper_id, query=item["query"], summary=excerpt[:900],
            excerpt=excerpt[:1200], relevance_score=score,
            retrieval_score=float(item["score"]), claims=[excerpt[:700]],
            epistemic_status="retrieval_only",
            context={"section": chunk.section, "page": chunk.page,
                     "start_char": chunk.start_char, "end_char": chunk.end_char},
        )

    async def _rcs_one(self, item: Dict[str, Any]) -> EvidenceRecord:
        fallback = self._fallback_record(item)
        if not self.client or self.mode not in {"llm", "api", "direct"}:
            return fallback
        chunk: FullTextChunk = item["chunk"]
        model_name = getattr(self.client, "model", "unknown")
        cache_path = self.cache_dir / "rcs" / (
            _hash("rcs-v1", model_name, item["query"], chunk.id, chunk.text) + ".json"
        )
        if cache_path.exists():
            try:
                return EvidenceRecord.model_validate_json(cache_path.read_text(encoding="utf-8"))
            except Exception:
                logger.debug("Ignoring invalid RCS cache file %s", cache_path)
        schema = {
            "type": "object",
            "properties": {
                "summary": {"type": "string"}, "excerpt": {"type": "string"},
                "relevance_score": {"type": "number", "minimum": 0, "maximum": 10},
                "claims": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
                "entities": {"type": "array", "items": {"type": "string"}},
                "methods": {"type": "array", "items": {"type": "string"}},
                "limitations": {"type": "array", "items": {"type": "string"}},
                "epistemic_status": {"type": "string"},
                "context": {"type": "object"},
            },
            "required": ["summary", "excerpt", "relevance_score", "claims"],
        }
        prompt = (
            f"Research query:\n{item['query']}\n\nPaper chunk ({chunk.paper_id}, "
            f"section={chunk.section or 'unknown'}, page={chunk.page}):\n{chunk.text}\n\n"
            "Create a retrieval-contextual summary (RCS). The excerpt must be an exact, "
            "short quote copied from the chunk. Score relevance, not truth. Split claims "
            "into atomic statements and preserve population/intervention/outcome/method/limitations in context."
        )
        try:
            data = await self.client.structured_chat(
                system_prompt="You extract auditable scientific evidence. Never invent text absent from the chunk.",
                user_prompt=prompt, output_schema=schema, max_tokens=4096,
                temperature=0.0, disable_thinking=False,
            )
            excerpt = str(data.get("excerpt") or "").strip()
            if excerpt and excerpt not in chunk.text:
                excerpt = fallback.excerpt
            record = EvidenceRecord(
                id=fallback.id, chunk_id=chunk.id, paper_id=chunk.paper_id,
                query=item["query"], summary=str(data.get("summary") or fallback.summary),
                excerpt=excerpt or fallback.excerpt,
                relevance_score=float(data.get("relevance_score", fallback.relevance_score)),
                retrieval_score=float(item["score"]), claims=list(data.get("claims") or fallback.claims)[:5],
                entities=list(data.get("entities") or []), methods=list(data.get("methods") or []),
                limitations=list(data.get("limitations") or []),
                epistemic_status=str(data.get("epistemic_status") or "reported"),
                context={**fallback.context, **dict(data.get("context") or {})},
            )
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(record.model_dump_json(), encoding="utf-8")
            return record
        except Exception as exc:
            logger.warning("RCS failed for %s; using extractive fallback: %s", chunk.id, exc)
            return fallback

    async def _rcs(self, gs: GroundingState) -> Dict[str, Any]:
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def guarded(item: Dict[str, Any]) -> EvidenceRecord:
            async with semaphore:
                try:
                    return await asyncio.wait_for(
                        self._rcs_one(item), timeout=self.llm_call_timeout
                    )
                except asyncio.TimeoutError:
                    logger.warning("RCS timed out for %s; using extractive fallback", item["chunk"].id)
                    return self._fallback_record(item)

        records = await asyncio.gather(*(guarded(x) for x in gs.get("candidates", [])))
        # Deduplicate repeated query/chunk evidence and keep only relevant records.
        unique: Dict[str, EvidenceRecord] = {}
        for record in records:
            if record.relevance_score >= self.min_relevance:
                old = unique.get(record.id)
                if old is None or record.relevance_score > old.relevance_score:
                    unique[record.id] = record
        records = sorted(unique.values(), key=lambda r: r.relevance_score, reverse=True)
        report = gs["report"].model_copy(deep=True)
        report.evidence_records = len(records)
        return {"evidence_records": records, "report": report}

    async def _atomize_claims(self, gs: GroundingState) -> Dict[str, Any]:
        claims: Dict[str, AtomicClaim] = {}
        for record in gs.get("evidence_records", []):
            for statement in record.claims:
                statement = re.sub(r"\s+", " ", statement).strip()
                if len(statement) < 20:
                    continue
                key = re.sub(r"\W+", "", statement).lower()
                claim = claims.get(key)
                if claim:
                    if record.id not in claim.evidence_record_ids:
                        claim.evidence_record_ids.append(record.id)
                    claim.entities = list(dict.fromkeys(claim.entities + record.entities))
                    claim.confidence = min(1.0, claim.confidence + 0.1)
                else:
                    claims[key] = AtomicClaim(
                        id="CLM_" + _hash(statement), statement=statement,
                        evidence_record_ids=[record.id], entities=record.entities,
                        confidence=min(0.95, 0.35 + record.relevance_score / 15),
                    )
        claim_list = list(claims.values())
        report = gs["report"].model_copy(deep=True)
        report.claims_total = len(claim_list)
        return {"claims": claim_list, "report": report}

    async def _recall_relation_candidates(self, gs: GroundingState) -> Dict[str, Any]:
        pairs = self.relation_retriever.recall(
            gs.get("claims", []), gs.get("evidence_records", [])
        )
        report = gs["report"].model_copy(deep=True)
        report.relation_pairs_recalled = len(pairs)
        return {"relation_pairs": pairs, "report": report}

    @staticmethod
    def _claim_payload(
        claim: AtomicClaim,
        record_map: Dict[str, EvidenceRecord],
    ) -> Dict[str, Any]:
        evidence = []
        paper_ids: List[str] = []
        for record_id in claim.evidence_record_ids[:3]:
            record = record_map.get(record_id)
            if not record:
                continue
            if record.paper_id:
                paper_ids.append(record.paper_id)
            evidence.append({
                "evidence_id": record.id,
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
            source_payload = self._relation_node_payload(pair.source, claim_map, record_map)
            target_payload = self._relation_node_payload(pair.target, claim_map, record_map)
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
                    logger.debug("Ignoring invalid relation cache file %s", cache_path)
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
                                    "supports", "contradicts", "extends", "limits",
                                    "same_as", "refines", "unrelated",
                                ],
                            },
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                            "condition_comparability": {
                                "type": "number", "minimum": 0, "maximum": 1,
                            },
                            "rationale": {"type": "string"},
                            "evidence_ids": {
                                "type": "array", "items": {"type": "string"},
                            },
                        },
                        "required": [
                            "pair_id", "source", "target", "relation", "confidence",
                            "condition_comparability", "rationale", "evidence_ids",
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
                "source": self._relation_node_payload(pair.source, claim_map, record_map),
                "target": self._relation_node_payload(pair.target, claim_map, record_map),
            })
        data = await self.client.structured_chat(
            system_prompt=(
                "Judge each recalled scientific pair conservatively. Return one decision per pair. "
                "Use UNRELATED when no defensible semantic relation exists. For EVIDENCE_RECORD_TO_CLAIM, "
                "only SUPPORTS, CONTRADICTS, LIMITS or UNRELATED are allowed and the evidence record must "
                "remain the source. For CLAIM_TO_CLAIM, never use SUPPORTS; only CONTRADICTS, EXTENDS, "
                "LIMITS, SAME_AS, REFINES or UNRELATED are allowed. CONTRADICTS requires "
                "incompatible conclusions under comparable population, intervention, outcome, dose, "
                "time and method. If conditions differ and one result only narrows another, use LIMITS. "
                "Direction matters: source SUPPORTS/EXTENDS/REFINES target. Cite only supplied evidence IDs."
            ),
            user_prompt="Recalled pairs:\n" + json.dumps(payload, ensure_ascii=False),
            output_schema=schema,
            max_tokens=max(6000, 1800 * len(pairs)),
            temperature=0.0,
            disable_thinking=False,
        )
        pair_map = {pair.id: pair for pair in pairs}
        decisions: List[RelationCandidate] = []
        for raw in data.get("relations", []):
            pair = pair_map.get(str(raw.get("pair_id") or ""))
            if not pair:
                continue
            source, target = str(raw.get("source") or ""), str(raw.get("target") or "")
            source_payload = self._relation_node_payload(pair.source, claim_map, record_map)
            target_payload = self._relation_node_payload(pair.target, claim_map, record_map)
            endpoint_statements: Dict[str, str] = {}
            for endpoint_id, endpoint_payload in (
                (pair.source, source_payload), (pair.target, target_payload)
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
                evidence_id for evidence_id in raw.get("evidence_ids", [])
                if evidence_id in set(pair.source_evidence_ids + pair.target_evidence_ids)
            ]
            if not evidence_ids:
                evidence_ids = list(dict.fromkeys(
                    pair.source_evidence_ids + pair.target_evidence_ids
                ))
            source_papers = pair.source_paper_ids if source == pair.source else pair.target_paper_ids
            target_papers = pair.target_paper_ids if target == pair.target else pair.source_paper_ids
            relation = str(raw.get("relation") or "unrelated")
            allowed_relations = (
                {"supports", "contradicts", "limits", "unrelated"}
                if pair.source_type == "evidence_record"
                else {"contradicts", "extends", "limits", "same_as", "refines", "unrelated"}
            )
            rationale = str(raw.get("rationale") or "")
            if relation not in allowed_relations:
                rationale = (
                    f"Type constraint rejected {relation} for {pair.source_type}_to_{pair.target_type}. "
                    + rationale
                )
                relation = "unrelated"
            candidate = RelationCandidate(
                id="REL_" + _hash(pair.id, source, target, relation),
                source=source,
                target=target,
                relation=relation,
                confidence=float(raw.get("confidence", 0.0)),
                condition_comparability=float(raw.get("condition_comparability", 0.5)),
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
                    cache_path.write_text(candidate.model_dump_json(), encoding="utf-8")
                except OSError:
                    logger.debug("Could not write relation cache file %s", cache_path)
        if len(decisions) != len(pairs):
            logger.warning(
                "Relation validation retained %d/%d decisions for pairs %s; raw=%s",
                len(decisions), len(pairs), [pair.id for pair in pairs],
                json.dumps(data, ensure_ascii=False)[:4000],
            )
        return cached + decisions

    async def _judge_candidate_relations(self, gs: GroundingState) -> Dict[str, Any]:
        pairs = gs.get("relation_pairs", [])
        candidates: List[RelationCandidate] = []
        if self.client and self.mode in {"llm", "api", "direct"} and pairs:
            claim_map = {claim.id: claim for claim in gs.get("claims", [])}
            record_map = {record.id: record for record in gs.get("evidence_records", [])}
            batches = [
                pairs[index:index + self.relation_batch_size]
                for index in range(0, len(pairs), self.relation_batch_size)
            ]
            semaphore = asyncio.Semaphore(self.max_concurrency)

            async def guarded(batch: List[RelationPair]) -> List[RelationCandidate]:
                async with semaphore:
                    for attempt in range(self.relation_judge_retries + 1):
                        try:
                            decisions = await asyncio.wait_for(
                                self._judge_relation_batch(batch, claim_map, record_map),
                                timeout=self.llm_call_timeout,
                            )
                            if len(decisions) == len(batch):
                                return decisions
                            logger.warning(
                                "Relation batch returned %d/%d decisions (attempt %d)",
                                len(decisions), len(batch), attempt + 1,
                            )
                        except asyncio.TimeoutError:
                            logger.warning(
                                "Relation candidate batch timed out for %d pair(s) (attempt %d)",
                                len(batch), attempt + 1,
                            )
                        except Exception as exc:
                            logger.warning(
                                "Relation candidate batch failed (attempt %d): %s",
                                attempt + 1, exc,
                            )
                    return []

            results = await asyncio.gather(*(guarded(batch) for batch in batches))
            deduplicated: Dict[tuple[str, str, str], RelationCandidate] = {}
            for candidate in [item for batch in results for item in batch]:
                key = (candidate.source, candidate.target, candidate.relation)
                old = deduplicated.get(key)
                if old is None or candidate.confidence > old.confidence:
                    deduplicated[key] = candidate
            candidates = list(deduplicated.values())
        report = gs["report"].model_copy(deep=True)
        report.relation_candidates_judged = len(candidates)
        return {"relation_candidates": candidates, "report": report}

    @staticmethod
    def _provenance_relations(
        claims: List[AtomicClaim],
        records: List[EvidenceRecord],
    ) -> List[EvidenceRelation]:
        record_map = {record.id: record for record in records}
        relations: List[EvidenceRelation] = []
        for claim in claims:
            for record_id in claim.evidence_record_ids:
                record = record_map.get(record_id)
                relations.append(EvidenceRelation(
                    id="REL_" + _hash(record_id, claim.id, "supports"),
                    source=record_id,
                    target=claim.id,
                    relation="supports",
                    confidence=claim.confidence,
                    rationale="Atomic claim extracted from this evidence record.",
                    evidence_ids=[record_id],
                    source_paper_ids=[record.paper_id] if record and record.paper_id else [],
                    target_paper_ids=[record.paper_id] if record and record.paper_id else [],
                    retrieval_score=(record.retrieval_score if record else 0.0),
                    condition_comparability=1.0,
                    candidate_origin=["provenance"],
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
                candidate for candidate in candidates
                if candidate.relation != "unrelated"
                and candidate.confidence >= self.relation_min_confidence
            ]
            trace = {
                "mode": "direct",
                "candidate_count": len([
                    candidate for candidate in candidates if candidate.relation != "unrelated"
                ]),
                "selected_count": len(selected_candidates),
                "minimum_confidence": self.relation_min_confidence,
                "selected_edge_ids": sorted(candidate.id for candidate in selected_candidates),
                "rejected_edge_ids": sorted(
                    candidate.id for candidate in candidates if candidate not in selected_candidates
                ),
            }
        semantic_relations = [candidate.to_relation() for candidate in selected_candidates]
        relations = provenance + semantic_relations
        report = gs["report"].model_copy(deep=True)
        report.relation_selection_mode = self.relation_selection_mode
        report.relation_candidates_selected = len(selected_candidates)
        report.relations_total = len(relations)
        report.relation_search = trace
        return {"relations": relations, "report": report}

    async def _finalize(self, gs: GroundingState) -> Dict[str, Any]:
        report = gs["report"].model_copy(deep=True)
        if not gs.get("evidence_records"):
            report.warnings.append("No evidence passed the RCS relevance threshold; M3 will retain its seed graph.")
        manifest = self.cache_dir / "last_grounding_report.json"
        try:
            manifest.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        except Exception as exc:
            logger.debug("Could not write grounding report: %s", exc)
        return {"report": report}
