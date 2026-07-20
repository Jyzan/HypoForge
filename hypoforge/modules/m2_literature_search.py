"""
M2: Literature Search & Knowledge Extraction.

Searches PubMed + OpenAlex (or Semantic Scholar) for each sub-question,
then uses Qwen to extract six categories of structured knowledge from
the retrieved paper abstracts.

Performs real search + **batch** LLM extraction end-to-end: papers are grouped
by ``batch_size`` (default 5) so 15 papers need only 3 LLM calls instead of 15.

Output: ``literature_results`` (list of ``LiteratureResult``) in state.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional

from ..protocol import ModuleProtocol
from ..prompts.m2_prompts import (
    M2_BATCH_EXTRACTION_SYSTEM_PROMPT,
    M2_BATCH_EXTRACTION_USER_TEMPLATE,
    M2_EXTRACTION_SYSTEM_PROMPT,
    M2_EXTRACTION_USER_TEMPLATE,
    M2_PAPER_BLOCK_TEMPLATE,
    M2_SEARCH_QUERY_SYSTEM_PROMPT,
    M2_SEARCH_QUERY_TEMPLATE,
)
from ..registry import ModuleRegistry
from ..state import (
    ConfidenceLevel,
    KnowledgeEntry,
    KnowledgeEntryType,
    LiteratureResult,
    PipelineState,
)
from ..tools.qwen_client import QwenClient, _salvage_string_arrays

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Query-extraction helper (called by _generate_search_queries)
# ---------------------------------------------------------------------------

def _extract_queries_from_text(raw_text: str) -> List[str]:
    """Pull search-query strings out of a free-text LLM response.

    The prompt instructs the model to return one query per line (no JSON, no
    markdown).  This function parses those lines, stripping common noise
    (numbering, bullets, leading/trailing punctuation).  It also falls back to
    JSON parsing for backward compatibility with older prompts.
    """
    text = raw_text.strip()
    if not text:
        return []

    # 1. Try JSON (backward compat — older prompts asked for JSON)
    try:
        data = json.loads(text)
        if isinstance(data, dict) and isinstance(data.get("queries"), list):
            queries = [str(q).strip() for q in data["queries"] if str(q).strip()]
            if queries:
                return list(dict.fromkeys(queries))
    except (json.JSONDecodeError, TypeError):
        pass

    # 2. Salvage garbled / truncated JSON
    salvaged = _salvage_string_arrays(text)
    if salvaged:
        for key in ("queries", "items", "entries"):
            items = salvaged.get(key)
            if items:
                cleaned = [str(q).strip() for q in items if str(q).strip()]
                if cleaned:
                    return list(dict.fromkeys(cleaned))

    # 3. Primary path: one query per line
    lines = text.splitlines()
    candidates = []
    for line in lines:
        line = line.strip().strip('"\'')
        # Remove common prefixes: "1.", "1)", "-", "*", "•"
        line = re.sub(r'^(?:\d+[.)]\s*|[-*•]\s*)+', '', line).strip()
        # Filter out obvious non-query lines (explanations, JSON brackets, etc.)
        if not line or line in ('{', '}', '[', ']'):
            continue
        if len(line) > 5:  # real queries are never this short
            candidates.append(line)
    return list(dict.fromkeys(candidates))

# ---------------------------------------------------------------------------
# Helpers — paper merge / dedup
# ---------------------------------------------------------------------------

def _doi_key(paper: dict) -> str:
    """Normalise a DOI for dedup."""
    doi = (paper.get("doi") or "").strip().lower()
    # Strip common prefixes
    for prefix in ("https://doi.org/", "doi:", "doi "):
        if doi.startswith(prefix):
            doi = doi[len(prefix):]
    return doi


def _merge_deduplicate(pubmed: List[dict], academic: List[dict]) -> List[dict]:
    """Merge PubMed + OpenAlex/S2 results, deduplicate by DOI, then by title prefix."""
    seen_doi: set[str] = set()
    seen_title_prefix: set[str] = set()
    merged: List[dict] = []

    for paper in pubmed + academic:
        # DOI dedup
        dk = _doi_key(paper)
        if dk and dk in seen_doi:
            continue
        # Title-prefix dedup (first 80 chars, lowercased)
        tp = paper.get("title", "").strip().lower()[:80]
        if tp and tp in seen_title_prefix:
            continue

        if dk:
            seen_doi.add(dk)
        if tp:
            seen_title_prefix.add(tp)
        merged.append(paper)

    # Sort by citation count descending, then year descending
    merged.sort(
        key=lambda p: (p.get("citation_count", 0), p.get("year", 0)),
        reverse=True,
    )
    return merged


# ---------------------------------------------------------------------------
# M2 module
# ---------------------------------------------------------------------------

@ModuleRegistry.register
class M2LiteratureSearch(ModuleProtocol):
    """Literature retrieval + six-category structured knowledge extraction.

    In ``mode="llm"``:
        1. Use each sub_question as a search query (or generate queries via Qwen).
        2. Search PubMed + OpenAlex/S2 in parallel.
        3. Merge, deduplicate, and keep the top-N papers.
        4. For each paper, call Qwen to extract KnowledgeEntry items.
        5. Return ``LiteratureResult`` per sub_question.

    TODO (组员可扩展):
        - **Query expansion**: 用 Qwen 对每个 sub_question 生成 2-3 个变体查询
          （MeSH 术语、同义词），提高召回率。当前直接用 sub_question 原文搜索。
        - **Batch extraction** ✅: 已实现——通过 ``batch_size`` 参数将 N 篇论文
          合并为 ceil(N/batch_size) 次 LLM 调用，大幅减少 API 开销。
        - **Cross-paper relation detection**: 在提取完所有论文的知识条目后，
          调 Qwen 检测跨论文的 supports / contradicts 关系（当前留给 M3 做）。
        - **Citation-aware relevance**: 利用引用次数加权排序；当前仅按 citation_count
          降序排列，可加入与查询的语义相似度。
        - **Result caching**: 对相同查询缓存搜索结果（基于 query hash），
          避免重复 API 调用（PubMed / OpenAlex 均有速率限制）。
        - **Query type routing**: 临床问题优先 PubMed，工程/CS 问题优先 OpenAlex；
          当前无差别调用两个后端。
    """

    module_name = "m2"
    module_version = "0.2.0"
    description = "Literature retrieval + six-category structured knowledge extraction"

    # ------------------------------------------------------------------
    # Configurable
    # ------------------------------------------------------------------

    def __init__(
        self,
        search_tools: Optional[List[str]] = None,
        mode: str = "llm",
        llm_config: Optional[Any] = None,
        query_llm_config: Optional[Any] = None,
        max_papers_per_query: int = 10,
        batch_size: int = 5,
        max_search_queries: int = 3,
        query_max_tokens: int = 2048,
        query_disable_thinking: bool = True,
        query_max_attempts: int = 2,
        **kwargs,
    ):
        self.search_tools = search_tools or ["semantic_scholar", "pubmed"]
        self.mode = mode
        self.llm_config = llm_config
        self.max_papers_per_query = max_papers_per_query
        self.batch_size = max(batch_size, 1)
        # How many English search queries to synthesise per sub-question.  0 keeps
        # the raw sub-question (no query generation).
        self.max_search_queries = max(0, max_search_queries)
        self.query_max_tokens = max(128, query_max_tokens)
        self.query_disable_thinking = query_disable_thinking
        self.query_max_attempts = max(1, query_max_attempts)
        self.client = QwenClient.from_config(llm_config) if llm_config else None
        # Query generation uses a separate (more reliable) model tier because
        # the turbo model often returns empty output for translation tasks on
        # the Alibaba Cloud MaaS endpoint.  Falls back to the main client.
        _qcfg = query_llm_config if query_llm_config is not None else llm_config
        self.query_llm_config = _qcfg
        self.query_client = QwenClient.from_config(_qcfg) if _qcfg else None

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
                "M2 requires an LLM client — pass llm_config / set OPENAI_API_KEY."
            )
        sub_questions = (
            state.problem_card.sub_questions
            if state.problem_card
            else [state.input_question]
        )
        key_entities = (
            state.problem_card.key_entities if state.problem_card else []
        )
        results = await self._run_real(sub_questions, key_entities)
        return {"literature_results": results}

    # ------------------------------------------------------------------
    # Real pipeline
    # ------------------------------------------------------------------

    async def _run_real(
        self,
        sub_questions: List[str],
        key_entities: Optional[List[str]] = None,
    ) -> List[LiteratureResult]:
        """Execute real search + LLM extraction for each sub_question."""
        from ..tools.pubmed_search import PubMedTool
        from ..tools.semantic_scholar import SemanticScholarTool

        pubmed = PubMedTool()
        academic = SemanticScholarTool()
        limit = self.max_papers_per_query
        key_entities = key_entities or []

        results: List[LiteratureResult] = []
        for sq in sub_questions:
            logger.info("M2: searching for sub_question=%r", sq[:80])

            # --- Step 0: synthesise English search queries ---
            # PubMed / OpenAlex return ~0 results for Chinese-language queries,
            # so we translate the (possibly Chinese) sub-question into focused
            # English queries before searching.
            queries = await self._generate_search_queries(sq, key_entities)
            logger.info("M2: %d search queries: %s", len(queries), [q[:60] for q in queries])

            # --- Step 1: search every query across both backends ---
            search_calls = []
            for q in queries:
                search_calls.append(("pubmed", pubmed.search(q, limit=limit)))
                search_calls.append(("academic", academic.search(q, limit=limit)))
            raw_results = await asyncio.gather(
                *(call for _, call in search_calls),
                return_exceptions=True,
            )

            pubmed_papers: List[dict] = []
            acad_papers: List[dict] = []
            for (backend, _), res in zip(search_calls, raw_results):
                if isinstance(res, Exception):
                    logger.warning("%s search failed: %s", backend, res)
                    continue
                (pubmed_papers if backend == "pubmed" else acad_papers).extend(res)

            # --- Step 2: merge & deduplicate ---
            all_papers = _merge_deduplicate(pubmed_papers, acad_papers)
            top_papers = all_papers[:limit]
            logger.info("M2: %d papers after merge (PubMed=%d, academic=%d)",
                        len(all_papers), len(pubmed_papers), len(acad_papers))

            # --- Step 3: batch-extract knowledge entries ---
            # Papers are grouped into batches to reduce LLM round-trips.
            # e.g. 15 papers × batch_size=5 → 3 LLM calls instead of 15.
            all_entries: List[KnowledgeEntry] = []
            if top_papers:
                batches = [
                    top_papers[i:i + self.batch_size]
                    for i in range(0, len(top_papers), self.batch_size)
                ]
                logger.info(
                    "M2: %d papers → %d batches (batch_size=%d)",
                    len(top_papers), len(batches), self.batch_size,
                )

                # Run batches concurrently (capped by Semaphore to avoid rate limits)
                sem = asyncio.Semaphore(3)

                async def _bounded_batch(batch):
                    async with sem:
                        return await self._extract_batch(batch)

                entry_lists = await asyncio.gather(
                    *[_bounded_batch(b) for b in batches],
                    return_exceptions=True,
                )
                for entries in entry_lists:
                    if isinstance(entries, Exception):
                        logger.warning("M2 batch extraction failed: %s", entries)
                        continue
                    all_entries.extend(entries)

            results.append(LiteratureResult(
                sub_question=sq,
                papers_retrieved=len(top_papers),
                knowledge_entries=all_entries,
            ))

        return results

    # ------------------------------------------------------------------
    # Search-query generation (English, from any-language sub-question)
    # ------------------------------------------------------------------

    async def _generate_search_queries(
        self,
        sub_question: str,
        key_entities: List[str],
    ) -> List[str]:
        """Translate a (possibly non-English) sub-question into English search queries.

        PubMed and OpenAlex return almost nothing for Chinese-language queries, so
        we ask Qwen for a few focused English queries.  Falls back to the raw
        sub-question when generation is disabled, unavailable, or fails.

        Uses a plain-text chat (one query per line) because the turbo-tier model
        often returns empty output when asked for JSON — but handles simple
        line-by-line instructions reliably.
        """
        if self.query_client is None or self.max_search_queries <= 0:
            return [sub_question]

        user_prompt = M2_SEARCH_QUERY_TEMPLATE.format(
            sub_question=sub_question,
            entities=", ".join(key_entities) if key_entities else "(none provided)",
        )
        last_error: Optional[Exception] = None

        # Qwen reasoning models may otherwise spend the entire small completion
        # budget on hidden reasoning and return an empty ``content`` field.  The
        # task is a direct translation/query-rewrite, so thinking is unnecessary.
        for attempt in range(1, self.query_max_attempts + 1):
            try:
                raw_text = await self.query_client.chat(
                    system_prompt=M2_SEARCH_QUERY_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    max_tokens=self.query_max_tokens,
                    temperature=getattr(self.query_llm_config, "temperature", 0.1),
                    disable_thinking=self.query_disable_thinking,
                )
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "M2 query generation attempt %d/%d failed: %s",
                    attempt,
                    self.query_max_attempts,
                    exc,
                )
                continue

            cleaned = _extract_queries_from_text(raw_text)
            if cleaned:
                return cleaned[: self.max_search_queries]
            logger.warning(
                "M2 query generation attempt %d/%d returned empty output",
                attempt,
                self.query_max_attempts,
            )

        detail = f": {last_error}" if last_error is not None else ""
        raise RuntimeError(
            "M2 could not generate an English literature-search query after "
            f"{self.query_max_attempts} attempts"
            f"{detail}. Refusing to silently search the original non-English question."
        )

    # ------------------------------------------------------------------
    # Batch knowledge extraction
    # ------------------------------------------------------------------

    async def _extract_batch(self, papers: List[dict]) -> List[KnowledgeEntry]:
        """Extract KnowledgeEntry items from a batch of papers in ONE LLM call.

        This replaces N individual ``_extract_from_paper`` calls with a single
        call, reducing M2's LLM round-trips from ``papers × sub_questions``
        down to ``ceil(papers / batch_size) × sub_questions``.
        """
        assert self.client is not None

        if not papers:
            return []

        # --- Build paper ID map ---
        paper_map: Dict[str, dict] = {}
        paper_blocks: List[str] = []
        for idx, paper in enumerate(papers):
            pmid = paper.get("pmid", "")
            paper_id = paper.get("paper_id", "")
            source_id = f"PMID:{pmid}" if pmid else (paper_id or f"IDX{idx}")

            paper_map[source_id] = paper

            paper_blocks.append(M2_PAPER_BLOCK_TEMPLATE.format(
                index=idx + 1,
                paper_id=source_id,
                title=paper.get("title", "Unknown"),
                authors=", ".join(paper.get("authors", [])[:5]),
                year=paper.get("year", ""),
                journal=paper.get("journal", ""),
                abstract=paper.get("abstract", "") or "(No abstract available)",
            ))

        user_prompt = M2_BATCH_EXTRACTION_USER_TEMPLATE.format(
            paper_count=len(papers),
            papers_text="\n".join(paper_blocks),
        )

        schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": [e.value for e in KnowledgeEntryType],
                    },
                    "content": {"type": "string"},
                    "confidence": {
                        "type": ["string", "null"],
                        "enum": [c.value for c in ConfidenceLevel] + [None],
                    },
                    "source_paper_id": {"type": "string"},
                    "source_paper_title": {"type": "string"},
                    "entities": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["type", "content", "source_paper_id", "entities"],
            },
        }

        try:
            raw_entries = await self.client.structured_chat(
                system_prompt=M2_BATCH_EXTRACTION_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                output_schema=schema,
                max_tokens=16384,
                temperature=getattr(self.llm_config, "temperature", 0.1),
            )
        except Exception:
            logger.exception("Batch extraction failed (%d papers)", len(papers))
            return []

        # Normalise: structured_chat may return dict or list
        if isinstance(raw_entries, dict):
            raw_entries = (
                raw_entries.get("entries")
                or raw_entries.get("data")
                or raw_entries.get("items")
                or []
            )

        entries: List[KnowledgeEntry] = []
        for i, raw in enumerate(raw_entries or []):
            if not isinstance(raw, dict):
                continue
            try:
                source_id = raw.get("source_paper_id", "")
                source_title = raw.get("source_paper_title", "")

                # If the LLM didn't set source_paper_title, look it up
                if not source_title and source_id in paper_map:
                    source_title = paper_map[source_id].get("title", "")

                entries.append(KnowledgeEntry(
                    id=f"KE_{source_id.replace(':', '_')}_{i}",
                    type=KnowledgeEntryType(raw.get("type", "established_fact")),
                    content=raw.get("content", ""),
                    confidence=(
                        ConfidenceLevel(raw["confidence"])
                        if raw.get("confidence")
                        else None
                    ),
                    source_paper_id=source_id,
                    source_paper_title=source_title,
                    entities=raw.get("entities", []),
                ))
            except Exception:
                logger.debug("Skipping invalid batch entry: %s", raw)

        logger.debug(
            "Batch extraction: %d papers → %d entries",
            len(papers), len(entries),
        )
        return entries

    # ------------------------------------------------------------------
    # Per-paper knowledge extraction (legacy — used when batch_size=1)
    # ------------------------------------------------------------------

    async def _extract_from_paper(self, paper: dict, index: int) -> List[KnowledgeEntry]:
        """Call Qwen to extract KnowledgeEntry items from a single paper."""
        assert self.client is not None

        # Build paper identifier
        pmid = paper.get("pmid", "")
        paper_id = paper.get("paper_id", "")
        source_id = f"PMID:{pmid}" if pmid else (paper_id or f"IDX{index}")

        logger.debug("M2: extracting knowledge from %s", source_id)

        user_prompt = M2_EXTRACTION_USER_TEMPLATE.format(
            title=paper.get("title", "Unknown"),
            authors=", ".join(paper.get("authors", [])[:5]),
            year=paper.get("year", ""),
            journal=paper.get("journal", ""),
            doi=paper.get("doi", ""),
            abstract=paper.get("abstract", "") or "(No abstract available)",
        )

        schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": [e.value for e in KnowledgeEntryType],
                    },
                    "content": {"type": "string"},
                    "confidence": {
                        "type": ["string", "null"],
                        "enum": [c.value for c in ConfidenceLevel] + [None],
                    },
                    "entities": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["type", "content", "entities"],
            },
        }

        try:
            raw_entries = await self.client.structured_chat(
                system_prompt=M2_EXTRACTION_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                output_schema=schema,
                max_tokens=16384,  # Knowledge extraction can produce large responses
                temperature=getattr(self.llm_config, "temperature", 0.1),
            )
        except Exception:
            logger.exception("Qwen extraction failed for %s", source_id)
            return []

        if isinstance(raw_entries, dict):
            raw_entries = (
                raw_entries.get("entries")
                or raw_entries.get("data")
                or raw_entries.get("items")
                or []
            )

        entries: List[KnowledgeEntry] = []
        for i, raw in enumerate(raw_entries or []):
            if not isinstance(raw, dict):
                continue
            try:
                entries.append(KnowledgeEntry(
                    id=f"KE_{source_id.replace(':', '_')}_{i}",
                    type=KnowledgeEntryType(raw.get("type", "established_fact")),
                    content=raw.get("content", ""),
                    confidence=(
                        ConfidenceLevel(raw["confidence"])
                        if raw.get("confidence")
                        else None
                    ),
                    source_paper_id=source_id,
                    source_paper_title=paper.get("title", ""),
                    entities=raw.get("entities", []),
                ))
            except Exception:
                logger.debug("Skipping invalid entry from %s: %s", source_id, raw)

        return entries

    @classmethod
    def get_input_fields(cls) -> List[str]:
        return ["problem_card"]

    @classmethod
    def get_output_fields(cls) -> List[str]:
        return ["literature_results"]
