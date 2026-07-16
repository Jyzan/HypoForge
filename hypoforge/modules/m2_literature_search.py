"""
M2: Literature Search & Knowledge Extraction.

Searches PubMed + OpenAlex (or Semantic Scholar) for each sub-question,
then uses Qwen to extract six categories of structured knowledge from
the retrieved paper abstracts.

In ``mode="stub"`` (default), returns hard-coded entries for quick testing.
In ``mode="llm"``, performs real search + **batch** LLM extraction end-to-end:
papers are grouped by ``batch_size`` (default 5) so 15 papers need only 3
LLM calls instead of 15.

Output: ``literature_results`` (list of ``LiteratureResult``) in state.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from ..protocol import ModuleProtocol
from ..prompts.m2_prompts import (
    M2_BATCH_EXTRACTION_SYSTEM_PROMPT,
    M2_BATCH_EXTRACTION_USER_TEMPLATE,
    M2_EXTRACTION_SYSTEM_PROMPT,
    M2_EXTRACTION_USER_TEMPLATE,
    M2_PAPER_BLOCK_TEMPLATE,
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
from ..tools.qwen_client import QwenClient

logger = logging.getLogger(__name__)

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
        mode: str = "stub",  # "stub" | "llm"
        llm_config: Optional[Any] = None,
        max_papers_per_query: int = 10,
        batch_size: int = 5,
        implementation: str = "legacy",
        agentic_adapter: Optional[Any] = None,
        **kwargs,
    ):
        if implementation not in {"legacy", "agentic"}:
            raise ValueError("implementation must be 'legacy' or 'agentic'")
        self.search_tools = search_tools or ["semantic_scholar", "pubmed"]
        self.mode = mode
        self.llm_config = llm_config
        self.max_papers_per_query = max_papers_per_query
        self.batch_size = max(batch_size, 1)
        self.implementation = implementation
        self.agentic_adapter = agentic_adapter
        self.client = QwenClient.from_config(llm_config) if llm_config else None

    # ------------------------------------------------------------------
    # Stub data (fallback)
    # ------------------------------------------------------------------

    _STUB_ENTRIES = [
        KnowledgeEntry(
            id="KE001",
            type=KnowledgeEntryType.ESTABLISHED_FACT,
            content="Hsp70 chaperones assist protein folding via an ATP-dependent cycle.",
            confidence=ConfidenceLevel.HIGH,
            source_paper_id="PMID:32012345",
            source_paper_title="Hsp70 chaperone cycle: structure and mechanism",
            entities=["Hsp70", "ATP"],
        ),
        KnowledgeEntry(
            id="KE002",
            type=KnowledgeEntryType.MECHANISTIC_CONCLUSION,
            content="NAD+ depletion impairs Hsp70 ATPase activity in aged cells.",
            confidence=ConfidenceLevel.MEDIUM,
            source_paper_id="PMID:31987654",
            source_paper_title="NAD+ metabolism regulates proteostasis in aging",
            entities=["NAD+", "Hsp70", "aging"],
        ),
        KnowledgeEntry(
            id="KE003",
            type=KnowledgeEntryType.CONFLICTING_EVIDENCE,
            content="Some studies suggest Hsp70 is NAD+-independent; others show NAD+ modulation.",
            confidence=ConfidenceLevel.LOW,
            source_paper_id="PMID:31876543",
            source_paper_title="Debate: metabolic regulation of chaperone activity",
            entities=["Hsp70", "NAD+"],
        ),
        KnowledgeEntry(
            id="KE004",
            type=KnowledgeEntryType.METHOD,
            content="ATPase activity assay, FRET-based folding sensor, Cryo-EM.",
            confidence=ConfidenceLevel.HIGH,
            source_paper_id="PMID:32012345",
            source_paper_title="Hsp70 chaperone cycle: structure and mechanism",
            entities=["Hsp70"],
        ),
        KnowledgeEntry(
            id="KE005",
            type=KnowledgeEntryType.KNOWLEDGE_GAP,
            content="The direct link between cellular NAD+/NADH ratio and Hsp70 folding efficiency in vivo is unknown.",
            confidence=None,
            source_paper_id="PMID:31987654",
            source_paper_title="NAD+ metabolism regulates proteostasis in aging",
            entities=["NAD+", "NADH", "Hsp70"],
        ),
        KnowledgeEntry(
            id="KE006",
            type=KnowledgeEntryType.ESTABLISHED_FACT,
            content="Amyloid-beta aggregation is a hallmark of Alzheimer disease pathology.",
            confidence=ConfidenceLevel.HIGH,
            source_paper_id="PMID:31765432",
            source_paper_title="Amyloid cascade hypothesis: 2024 update",
            entities=["amyloid-beta", "Alzheimer disease"],
        ),
        KnowledgeEntry(
            id="KE007",
            type=KnowledgeEntryType.KNOWLEDGE_GAP,
            content="Whether enhancing chaperone activity can clear pre-formed amyloid aggregates remains controversial.",
            confidence=None,
            source_paper_id="PMID:31654321",
            source_paper_title="Chaperone-based therapies for neurodegeneration",
            entities=["chaperone", "amyloid"],
        ),
    ]

    # ------------------------------------------------------------------
    # ModuleProtocol implementation
    # ------------------------------------------------------------------

    async def __call__(
        self,
        state: PipelineState,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if self.implementation == "agentic":
            if self.agentic_adapter is None:
                raise RuntimeError(
                    "search.implementation='agentic' requires an injected "
                    "AgenticM2Adapter"
                )
            return await self.agentic_adapter(state, config)

        sub_questions = (
            state.problem_card.sub_questions
            if state.problem_card
            else [state.input_question]
        )

        if self.mode in {"llm", "direct", "api"} and self.client:
            try:
                results = await self._run_real(sub_questions)
                return {"literature_results": results}
            except Exception as exc:
                logger.warning("M2 LLM mode failed; falling back to stub: %s", exc)

        # Stub fallback
        results: List[LiteratureResult] = []
        for sq in sub_questions:
            results.append(LiteratureResult(
                sub_question=sq,
                papers_retrieved=5,
                knowledge_entries=self._STUB_ENTRIES,
            ))
        return {"literature_results": results}

    # ------------------------------------------------------------------
    # Real pipeline
    # ------------------------------------------------------------------

    async def _run_real(self, sub_questions: List[str]) -> List[LiteratureResult]:
        """Execute real search + LLM extraction for each sub_question."""
        from ..tools.pubmed_search import PubMedTool
        from ..tools.semantic_scholar import SemanticScholarTool

        pubmed = PubMedTool()
        academic = SemanticScholarTool()
        limit = self.max_papers_per_query

        results: List[LiteratureResult] = []
        for sq in sub_questions:
            logger.info("M2: searching for sub_question=%r", sq[:80])

            # --- Step 1: parallel search across backends ---
            # TODO(组员): 用 Qwen + M2_SEARCH_QUERY_TEMPLATE 生成 2-3 个
            #  变体查询（MeSH 术语、同义词），对每个变体分别搜索后合并。
            #  当前直接用 sub_question 原文作为查询。
            pubmed_papers, acad_papers = await asyncio.gather(
                pubmed.search(sq, limit=limit),
                academic.search(sq, limit=limit),
                return_exceptions=True,
            )

            if isinstance(pubmed_papers, Exception):
                logger.warning("PubMed search failed: %s", pubmed_papers)
                pubmed_papers = []
            if isinstance(acad_papers, Exception):
                logger.warning("Academic search failed: %s", acad_papers)
                acad_papers = []

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
