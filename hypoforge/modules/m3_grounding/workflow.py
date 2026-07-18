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
)

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
        graph.add_node("judge_relations", self._judge_relations)
        graph.add_node("finalize", self._finalize)
        graph.set_entry_point("resolve_sources")
        graph.add_edge("resolve_sources", "acquire_fulltext")
        graph.add_edge("acquire_fulltext", "parse_and_chunk")
        graph.add_edge("parse_and_chunk", "plan_queries")
        graph.add_edge("plan_queries", "retrieve")
        graph.add_edge("retrieve", "rcs")
        graph.add_edge("rcs", "atomize_claims")
        graph.add_edge("atomize_claims", "judge_relations")
        graph.add_edge("judge_relations", "finalize")
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
                user_prompt=prompt, output_schema=schema, max_tokens=2500,
                temperature=0.0, disable_thinking=True,
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
                return await self._rcs_one(item)

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

    async def _judge_relations(self, gs: GroundingState) -> Dict[str, Any]:
        claims = gs.get("claims", [])
        relations: List[EvidenceRelation] = []
        # Every claim keeps explicit provenance to its evidence record.
        for claim in claims:
            for erid in claim.evidence_record_ids:
                relations.append(EvidenceRelation(
                    source=erid, target=claim.id, relation="supports",
                    confidence=claim.confidence, rationale="Atomic claim extracted from this evidence record.",
                ))
        # Cross-claim judgements are LLM-only and intentionally bounded/conservative.
        if self.client and self.mode in {"llm", "api", "direct"} and 1 < len(claims) <= 80:
            schema = {
                "type": "object", "properties": {"relations": {"type": "array", "maxItems": 80,
                    "items": {"type": "object", "properties": {
                        "source": {"type": "string"}, "target": {"type": "string"},
                        "relation": {"type": "string", "enum": ["contradicts", "extends", "limits", "same_as", "refines"]},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "rationale": {"type": "string"}},
                        "required": ["source", "target", "relation", "confidence", "rationale"]}}},
                "required": ["relations"],
            }
            payload = [{"id": c.id, "statement": c.statement, "entities": c.entities} for c in claims]
            try:
                data = await self.client.structured_chat(
                    system_prompt=("Judge scientific claim relations conservatively. Contradiction requires "
                                   "incompatible conclusions under comparable population, intervention, outcome, "
                                   "dose, time and method; methodological differences usually LIMIT, not contradict."),
                    user_prompt="Claims:\n" + json.dumps(payload, ensure_ascii=False),
                    output_schema=schema, max_tokens=5000, temperature=0.0, disable_thinking=True,
                )
                valid = {c.id for c in claims}
                for raw in data.get("relations", []):
                    if raw.get("source") in valid and raw.get("target") in valid and raw.get("source") != raw.get("target"):
                        rel = EvidenceRelation.model_validate(raw)
                        if rel.confidence >= 0.65:
                            relations.append(rel)
            except Exception as exc:
                logger.warning("Cross-claim relation judgement failed: %s", exc)
        report = gs["report"].model_copy(deep=True)
        report.relations_total = len(relations)
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
