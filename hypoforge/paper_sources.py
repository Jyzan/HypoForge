"""Shared paper-source junction helpers for supplement ("gap-filling") search.

The agentic M2 adapter keeps its supplement flow self-contained; the legacy
M2 module (``modules/m2_literature_search.py``) reuses the helpers here so
both implementations share ONE set of semantics for:

* opening the persistent :class:`~hypoforge.memory.paper_store.PaperStore`
  (gracefully degrading to ``None`` when the cache dir is unavailable);
* persisting fresh search results (papers + the queries that produced them)
  as best-effort side effects that never break the main flow;
* deduplicating papers against already-known keys and queries against the
  ``search_ledger`` normalised-query set;
* merging a supplement increment into ``literature_results`` per
  sub-question **without ever clearing existing entries**;
* resolving the literature sub-question a gap targets.

All functions are pure / best-effort: none of them raises into the pipeline.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .memory.paper_store import PaperStore, normalize_query_text, paper_key
from .state import EvidenceGap, KnowledgeEntry, LiteratureResult

logger = logging.getLogger(__name__)

_WORD_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
MAX_SUGGESTED_QUERIES_PER_GAP = 3


# ---------------------------------------------------------------------------
# Store access / persistence junction
# ---------------------------------------------------------------------------


def open_paper_store(cache_dir: str) -> Optional[PaperStore]:
    """Open a :class:`PaperStore` for *cache_dir*, or ``None`` when unusable.

    Empty dir or OS-level failure both degrade to ``None`` — the paper cache
    is an optimisation, never a hard dependency.
    """

    if not cache_dir:
        return None
    try:
        return PaperStore(cache_dir)
    except OSError as exc:
        logger.warning("paper cache unavailable (%s); continuing without it", exc)
        return None


def persist_search_results(
    store: Optional[PaperStore],
    papers: Sequence[Any],
    queries: Sequence[str],
    *,
    run_id: str = "",
    round: int = 0,
    source: str = "",
) -> List[str]:
    """Best-effort junction: upsert *papers* and record query→keys mappings.

    Returns the paper keys (empty list when the store is unavailable).
    Never raises — cache writes must not break the search flow.
    """

    if store is None or not papers:
        return []
    try:
        keys = store.upsert_papers(
            papers, run_id=run_id, round=round, source=source
        )
        for query in queries or []:
            if str(query or "").strip():
                store.record_query(query, keys, run_id=run_id, round=round)
        return keys
    except Exception as exc:  # cache must never break the main flow
        logger.warning("failed to persist papers into the cache: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Dedup helpers
# ---------------------------------------------------------------------------


def dedupe_papers_by_key(
    papers: Sequence[Any], known_keys: set
) -> Tuple[List[Any], List[str]]:
    """Split *papers* into (fresh papers, fresh keys) against *known_keys*.

    Order is preserved; duplicates inside *papers* are dropped too.
    """

    fresh: List[Any] = []
    fresh_keys: List[str] = []
    for paper in papers or []:
        key = paper_key(paper)
        if key in known_keys:
            continue
        known_keys.add(key)
        fresh.append(paper)
        fresh_keys.append(key)
    return fresh, fresh_keys


def gap_candidate_queries(gap: EvidenceGap) -> List[str]:
    """Candidate query texts for a gap (no LLM round-trip).

    Uses ``suggested_queries`` when present; otherwise derives rule-based
    queries from the description / canonical entities.
    """

    candidates = [
        query.strip()
        for query in gap.suggested_queries
        if query and query.strip()
    ][:MAX_SUGGESTED_QUERIES_PER_GAP]
    if not candidates:
        description = " ".join(str(gap.description or "").split())
        if description:
            candidates.append(description[:200])
        if gap.canonical_entities:
            candidates.append(" ".join(gap.canonical_entities))
    return candidates


def dedupe_queries(
    candidate_texts: Sequence[str], issued_norms: set
) -> List[str]:
    """Drop candidates already present in the ledger (normalised compare).

    *issued_norms* is mutated: accepted queries are added so consecutive gaps
    in the same round never re-issue each other's queries.
    """

    fresh: List[str] = []
    for query in candidate_texts or []:
        normalized = normalize_query_text(query)
        if not normalized or normalized in issued_norms:
            continue
        issued_norms.add(normalized)
        fresh.append(query)
    return fresh


# ---------------------------------------------------------------------------
# literature_results incremental merge (never clears existing entries)
# ---------------------------------------------------------------------------


def merge_literature_increment(
    merged_results: List[LiteratureResult],
    sub_question: str,
    paper_count: int,
    entries: Sequence[KnowledgeEntry],
) -> None:
    """Merge a supplement increment into the running literature results.

    The sub-question's existing result is extended in place (knowledge
    entries deduplicated by stable id); an unseen sub-question gets a fresh
    ``LiteratureResult`` appended.  Nothing is ever removed.
    """

    for result in merged_results:
        if result.sub_question == sub_question:
            result.papers_retrieved += paper_count
            seen_ids = {entry.id for entry in result.knowledge_entries}
            result.knowledge_entries.extend(
                entry for entry in entries if entry.id not in seen_ids
            )
            return
    merged_results.append(
        LiteratureResult(
            sub_question=sub_question,
            papers_retrieved=paper_count,
            knowledge_entries=list(entries),
        )
    )


def resolve_gap_sub_question(
    gap: EvidenceGap, existing: Sequence[str]
) -> str:
    """Target sub-question for a gap; falls back to best token overlap."""

    target = (gap.target_sub_question or "").strip()
    if target:
        return target
    if not existing:
        return " ".join(str(gap.description or "").split()) or "supplement search"

    def tokens(text: str) -> set:
        return set(_WORD_TOKEN_RE.findall(str(text or "").casefold()))

    description_tokens = tokens(gap.description)
    best, best_score = existing[0], -1
    for candidate in existing:
        score = len(description_tokens & tokens(candidate))
        if score > best_score:
            best, best_score = candidate, score
    return best
