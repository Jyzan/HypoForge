"""
BM25-based search index for the persistent knowledge graph.

Adapted from BioDSA's ``biodsa/memory/memory_graph/bm25_index.py``.
Provides fast semantic search over entities with disk persistence
and incremental update support.

Dependencies (optional, fall back to linear search if missing):
  - ``rank_bm25`` — BM25Okapi scoring
  - ``tiktoken`` — token normalisation
"""

from __future__ import annotations

import logging
import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional

from .schema import Entity, calculate_entities_hash

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency detection
# ---------------------------------------------------------------------------

try:
    import tiktoken

    _TIKTOKEN_ENCODER = tiktoken.encoding_for_model("gpt-4o")
    HAS_TIKTOKEN = True
except (ImportError, Exception):
    _TIKTOKEN_ENCODER = None
    HAS_TIKTOKEN = False

try:
    from rank_bm25 import BM25Okapi  # noqa: F401

    HAS_BM25 = True
except (ImportError, Exception):
    BM25Okapi = None  # type: ignore[assignment,misc]
    HAS_BM25 = False

if not HAS_BM25:
    logger.info(
        "rank_bm25 not available — falling back to linear string search. "
        "Install with: pip install rank_bm25 tiktoken"
    )

# ---------------------------------------------------------------------------
# Stop-word set (English)
# ---------------------------------------------------------------------------

STOPWORD_SET: set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "aren't", "as", "at", "be", "because", "been",
    "before", "being", "below", "between", "both", "but", "by", "can't",
    "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
    "doing", "don't", "down", "during", "each", "few", "for", "from",
    "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having",
    "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers",
    "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll",
    "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its",
    "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other",
    "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
    "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't",
    "so", "some", "such", "than", "that", "that's", "the", "their",
    "theirs", "them", "themselves", "then", "there", "there's", "these",
    "they", "they'd", "they'll", "they're", "they've", "this", "those",
    "through", "to", "too", "under", "until", "up", "very", "was",
    "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't",
    "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't",
    "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've",
    "your", "yours", "yourself", "yourselves",
}


def _remove_stop_words(text: str) -> str:
    """Strip English stop-words from *text*."""
    return " ".join(w for w in text.split() if w not in STOPWORD_SET)


def _tokenize(text: str) -> List[str]:
    """Tokenize *text* for BM25 indexing.

    1. Lowercase + remove stop-words.
    2. If tiktoken is available, round-trip through the encoder to
       drop tokens the model considers invalid.
    3. Extract ``\\b\\w+\\b`` tokens with a regex.
    """
    text = _remove_stop_words(text.lower())

    if HAS_TIKTOKEN and _TIKTOKEN_ENCODER is not None:
        try:
            tokens = _TIKTOKEN_ENCODER.encode(text)
            text = _TIKTOKEN_ENCODER.decode(tokens)
        except Exception:
            pass  # fall through to regex

    return re.findall(r"\b\w+\b", text)


# ============================================================================
# BM25SearchIndex
# ============================================================================


class BM25SearchIndex:
    """BM25-powered search index over ``Entity`` objects.

    Falls back to linear substring matching when ``rank_bm25`` is
    not installed.

    Parameters
    ----------
    k1 : float
        BM25 term-frequency saturation (default 0.5 — tuned for small collections).
    b : float
        BM25 length-normalisation penalty (default 0.3).
    epsilon : float
        BM25 epsilon boost for small collections (default 0.5).
    """

    def __init__(
        self,
        k1: float = 0.5,
        b: float = 0.3,
        epsilon: float = 0.5,
    ):
        self._k1 = k1
        self._b = b
        self._epsilon = epsilon

        self.bm25: Optional[object] = None
        self.entity_docs: List[str] = []
        self.entity_names: List[str] = []
        self.tokenized_docs: List[List[str]] = []
        self._is_built: bool = False
        self._entities_hash: Optional[str] = None
        self._name_to_index: Dict[str, int] = {}

    # ==================================================================
    # Build / search
    # ==================================================================

    def build_index(self, entities: List[Entity]) -> None:
        """Build (or rebuild) the index from a list of entities."""
        self.entity_names = []
        self.entity_docs = []
        self.tokenized_docs = []
        self._name_to_index = {}

        for i, entity in enumerate(entities):
            doc_text = " ".join(
                [entity.name, entity.entity_type, *entity.observations]
            )
            self.entity_names.append(entity.name)
            self.entity_docs.append(doc_text)
            self._name_to_index[entity.name] = i

            if HAS_BM25:
                self.tokenized_docs.append(_tokenize(doc_text))

        if HAS_BM25 and self.tokenized_docs:
            self.bm25 = BM25Okapi(  # type: ignore[call-arg]
                self.tokenized_docs,
                k1=self._k1,
                b=self._b,
                epsilon=self._epsilon,
            )

        self._entities_hash = calculate_entities_hash(entities)
        self._is_built = True

    def search(self, query: str, top_k: Optional[int] = None) -> List[str]:
        """Return entity names ranked by BM25 score for *query*."""
        if not self._is_built:
            return []

        # --- Fallback: linear token-overlap match ---
        if not HAS_BM25:
            q_tokens = set(_tokenize(query))
            if not q_tokens:
                return []
            scored = []
            for i, doc in enumerate(self.entity_docs):
                doc_tokens = set(_tokenize(doc))
                overlap = len(q_tokens & doc_tokens)
                if overlap > 0:
                    scored.append((overlap, self.entity_names[i]))
            scored.sort(key=lambda x: x[0], reverse=True)
            if top_k:
                scored = scored[:top_k]
            return [name for _, name in scored]

        if self.bm25 is None or not self.tokenized_docs:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)  # type: ignore[attr-defined]

        # Small-collection hybrid scoring (BM25 can return all zeros)
        if len(self.entity_names) < 10:
            qset = set(query_tokens)
            qlen = len(query_tokens)
            overlap = [
                len(qset & set(dt)) / qlen if qlen > 0 else 0
                for dt in self.tokenized_docs
            ]
            scores = [
                s if s > 0 else (ov * 0.1 if ov > 0 else 0.0)
                for s, ov in zip(scores, overlap)
            ]

        scored = [
            (s, n) for s, n in zip(scores, self.entity_names) if s > 0
        ]
        scored.sort(key=lambda x: x[0], reverse=True)

        if top_k:
            scored = scored[:top_k]

        return [name for _, name in scored]

    def is_built(self) -> bool:
        return self._is_built

    def clear(self) -> None:
        self.bm25 = None
        self.entity_docs.clear()
        self.entity_names.clear()
        self.tokenized_docs.clear()
        self._is_built = False
        self._entities_hash = None
        self._name_to_index.clear()

    # ==================================================================
    # Disk persistence
    # ==================================================================

    def save_to_disk(self, file_path: Path) -> None:
        """Pickle the index to *file_path* (only for >50 entities)."""
        if not self._is_built:
            return
        if len(self.entity_names) < 50:
            return  # small index — rebuild is faster than I/O

        data: dict = {
            "entity_names": self.entity_names,
            "entity_docs": self.entity_docs,
            "tokenized_docs": self.tokenized_docs,
            "entities_hash": self._entities_hash,
            "has_bm25": HAS_BM25,
            "bm25_data": self.bm25 if HAS_BM25 else None,
        }
        file_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(file_path, "wb") as fh:
                pickle.dump(data, fh)
        except Exception:
            logger.warning("Failed to save BM25 index to %s", file_path, exc_info=True)

    def load_from_disk(self, file_path: Path, entities_hash: str) -> bool:
        """Restore index from *file_path* if the entity hash matches."""
        if not file_path.exists():
            return False
        try:
            with open(file_path, "rb") as fh:
                data = pickle.load(fh)
        except Exception:
            logger.warning("Failed to load BM25 index from %s", file_path, exc_info=True)
            return False

        if data.get("entities_hash") != entities_hash:
            return False
        if data.get("has_bm25") != HAS_BM25:
            return False

        self.entity_names = data.get("entity_names", [])
        self.entity_docs = data.get("entity_docs", [])
        self.tokenized_docs = data.get("tokenized_docs", [])
        self._entities_hash = data.get("entities_hash")

        if HAS_BM25 and data.get("bm25_data") is not None:
            self.bm25 = data["bm25_data"]
        else:
            self.bm25 = None

        self._is_built = True
        return True

    def is_valid_for_entities(self, entities: List[Entity]) -> bool:
        if not self._is_built:
            return False
        return self._entities_hash == calculate_entities_hash(entities)

    # ==================================================================
    # Incremental updates
    # ==================================================================

    def add_entities_incremental(
        self, new_entities: List[Entity], all_entities: List[Entity]
    ) -> None:
        """Add entities without full rebuild (rebuilds lazily for ≥100 docs)."""
        if not self._is_built:
            self.build_index(all_entities)
            return

        for entity in new_entities:
            doc_text = " ".join(
                [entity.name, entity.entity_type, *entity.observations]
            )
            new_idx = len(self.entity_names)
            self.entity_names.append(entity.name)
            self.entity_docs.append(doc_text)
            self._name_to_index[entity.name] = new_idx
            if HAS_BM25:
                self.tokenized_docs.append(_tokenize(doc_text))

        if HAS_BM25 and self.tokenized_docs:
            if len(self.tokenized_docs) < 100:
                self.bm25 = BM25Okapi(  # type: ignore[call-arg]
                    self.tokenized_docs,
                    k1=self._k1,
                    b=self._b,
                    epsilon=self._epsilon,
                )
            else:
                self.bm25 = None
                self._is_built = False  # lazy rebuild on next search

        self._entities_hash = calculate_entities_hash(all_entities)

    def remove_entities_incremental(
        self, entity_names_to_remove: List[str], all_entities: List[Entity]
    ) -> None:
        """Remove entities; rebuilds BM25 lazily for ≥100 docs."""
        if not self._is_built:
            self.build_index(all_entities)
            return

        remove_set = set(entity_names_to_remove)
        indices = sorted(
            [self._name_to_index.get(n) for n in remove_set if n in self._name_to_index],
            reverse=True,
        )
        for idx in indices:
            self.entity_names.pop(idx)
            self.entity_docs.pop(idx)
            if HAS_BM25 and idx < len(self.tokenized_docs):
                self.tokenized_docs.pop(idx)

        self._name_to_index = {n: i for i, n in enumerate(self.entity_names)}

        if HAS_BM25 and self.tokenized_docs:
            if len(self.tokenized_docs) < 100:
                self.bm25 = BM25Okapi(  # type: ignore[call-arg]
                    self.tokenized_docs,
                    k1=self._k1,
                    b=self._b,
                    epsilon=self._epsilon,
                )
            else:
                self.bm25 = None
                self._is_built = False
        elif HAS_BM25:
            self.bm25 = None

        self._entities_hash = calculate_entities_hash(all_entities)

    def update_entity_incremental(
        self, entity_name: str, updated_entity: Entity, all_entities: List[Entity]
    ) -> bool:
        """Update one entity in-place; rebuilds BM25 lazily for ≥100 docs."""
        if not self._is_built:
            self.build_index(all_entities)
            return True

        idx = self._name_to_index.get(entity_name)
        if idx is None:
            return False

        doc_text = " ".join(
            [updated_entity.name, updated_entity.entity_type, *updated_entity.observations]
        )
        self.entity_names[idx] = updated_entity.name
        self.entity_docs[idx] = doc_text
        if HAS_BM25:
            self.tokenized_docs[idx] = _tokenize(doc_text)
            if self.tokenized_docs:
                if len(self.tokenized_docs) < 100:
                    self.bm25 = BM25Okapi(  # type: ignore[call-arg]
                        self.tokenized_docs,
                        k1=self._k1,
                        b=self._b,
                        epsilon=self._epsilon,
                    )
                else:
                    self.bm25 = None
                    self._is_built = False

        self._entities_hash = calculate_entities_hash(all_entities)
        return True
