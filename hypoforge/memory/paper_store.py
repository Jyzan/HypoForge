"""Paper-level persistent cache for M2 supplement ("gap-filling") searches.

Two append-only JSONL files live inside the shared ``memory_cache_dir``:

* ``papers.jsonl``     — one line per paper upsert, keyed by a normalised
  identifier (DOI → PMID → arXiv ID → title hash).  Later lines for the same
  key win on read; ``first_seen_run`` / ``first_seen_round`` are preserved.
* ``query_cache.jsonl`` — one line per recorded query mapping a normalised
  query hash to the paper keys it produced.  The last line per hash wins.

Design notes
------------
* Single-writer scenario: a plain ``threading.Lock`` plus append mode is
  enough; no compaction or rewriting happens.
* Missing or partially corrupt files degrade gracefully to empty results —
  the cache is an optimisation, never a hard dependency.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

logger = logging.getLogger(__name__)

_DOI_PREFIXES = ("https://doi.org/", "http://doi.org/", "doi:", "doi ")
_PUNCTUATION_RE = re.compile(r"[^\w\s]", re.UNICODE)
_ARXIV_PREFIX_RE = re.compile(r"(?i)^arxiv:")
_QUERY_HASH_RE = re.compile(r"^[0-9a-f]{12}$")

PaperLike = Union[Dict[str, Any], Any]  # dict or pydantic/object with attrs


# ---------------------------------------------------------------------------
# Key normalisation helpers (shared with the M2 adapter)
# ---------------------------------------------------------------------------


def _get_field(paper: PaperLike, name: str, default: Any = "") -> Any:
    if isinstance(paper, dict):
        return paper.get(name, default)
    return getattr(paper, name, default)


def _normalize_doi(raw: Any) -> str:
    doi = str(raw or "").strip().lower()
    for prefix in _DOI_PREFIXES:
        if doi.startswith(prefix):
            doi = doi[len(prefix):]
            break
    return doi.strip()


def normalize_query_text(query: str) -> str:
    """Lowercase, strip punctuation and collapse whitespace."""

    text = str(query or "").casefold()
    text = _PUNCTUATION_RE.sub(" ", text)
    return " ".join(text.split())


def query_hash(query: str) -> str:
    """Stable 12-hex-char hash of the normalised query text."""

    return hashlib.sha1(normalize_query_text(query).encode("utf-8")).hexdigest()[:12]


def paper_key(paper: PaperLike) -> str:
    """Normalised dedup key with priority DOI → PMID → arXiv ID → title hash."""

    doi = _normalize_doi(_get_field(paper, "doi"))
    if doi:
        return f"doi:{doi}"

    pmid = str(_get_field(paper, "pmid") or "").strip()
    if pmid:
        return f"pmid:{pmid}"

    external_ids = _get_field(paper, "external_ids", {}) or {}
    if isinstance(external_ids, dict):
        for name, value in external_ids.items():
            if str(name).casefold() == "arxiv" and str(value or "").strip():
                arxiv_id = _ARXIV_PREFIX_RE.sub("", str(value).strip()).lower()
                return f"arxiv:{arxiv_id}"

    title = " ".join(str(_get_field(paper, "title") or "").split()).casefold()
    if title:
        digest = hashlib.sha1(title.encode("utf-8")).hexdigest()[:12]
        return f"title:{digest}"

    fallback = str(_get_field(paper, "paper_id") or "")
    digest = hashlib.sha1(fallback.encode("utf-8")).hexdigest()[:12]
    return f"pid:{digest}"


# ---------------------------------------------------------------------------
# PaperStore
# ---------------------------------------------------------------------------


class PaperStore:
    """Append-only paper + query cache backed by JSONL files."""

    def __init__(self, cache_dir: Union[str, Path]) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.papers_path = self.cache_dir / "papers.jsonl"
        self.query_cache_path = self.cache_dir / "query_cache.jsonl"
        self._lock = threading.Lock()
        self._papers: Dict[str, Dict[str, Any]] = {}
        self._queries: Dict[str, Dict[str, Any]] = {}
        self._loaded = False

    # -- loading ------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._papers = self._read_records(self.papers_path, "key")
        self._queries = self._read_records(self.query_cache_path, "query_hash")
        self._loaded = True

    @staticmethod
    def _read_records(path: Path, key_field: str) -> Dict[str, Dict[str, Any]]:
        records: Dict[str, Dict[str, Any]] = {}
        if not path.exists():
            return records
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            logger.warning("PaperStore: cannot read %s: %s", path, exc)
            return records
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = record.get(key_field)
            if key:
                records[str(key)] = record
        return records

    @staticmethod
    def _to_dict(paper: PaperLike) -> Dict[str, Any]:
        if hasattr(paper, "model_dump"):
            return paper.model_dump(mode="json")
        if isinstance(paper, dict):
            return dict(paper)
        return {k: v for k, v in vars(paper).items() if not k.startswith("_")}

    def _append(self, path: Path, record: Dict[str, Any]) -> None:
        try:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.warning("PaperStore: cannot append to %s: %s", path, exc)

    # -- papers -------------------------------------------------------------

    def upsert_papers(
        self,
        papers: Sequence[PaperLike],
        run_id: str = "",
        round: int = 0,
        source: str = "",
    ) -> List[str]:
        """Append/upsert papers; returns their keys in input order.

        ``source`` is optional provenance: the name of the search backend /
        junction that produced the records (e.g. ``"legacy_m2"``).  Every
        distinct source that ever touches a paper is kept in the record's
        ``sources_seen`` list.
        """

        keys: List[str] = []
        with self._lock:
            self._ensure_loaded()
            for paper in papers or []:
                key = paper_key(paper)
                keys.append(key)
                existing = self._papers.get(key)
                sources_seen: List[str] = list(
                    (existing or {}).get("sources_seen") or []
                )
                if source and source not in sources_seen:
                    sources_seen.append(source)
                record = {
                    "key": key,
                    "first_seen_run": (
                        existing.get("first_seen_run", run_id)
                        if existing is not None
                        else run_id
                    ),
                    "first_seen_round": (
                        existing.get("first_seen_round", round)
                        if existing is not None
                        else round
                    ),
                    "last_seen_run": run_id,
                    "last_seen_round": round,
                    "sources_seen": sources_seen,
                    "paper": self._to_dict(paper),
                }
                self._papers[key] = record
                self._append(self.papers_path, record)
        return keys

    def lookup_by_keys(self, keys: Sequence[str]) -> List[Dict[str, Any]]:
        """Return cached paper metadata for known keys (input order, deduped)."""

        with self._lock:
            self._ensure_loaded()
            papers: List[Dict[str, Any]] = []
            seen: set = set()
            for key in keys or []:
                key = str(key)
                if key in seen:
                    continue
                record = self._papers.get(key)
                if record is None:
                    continue
                seen.add(key)
                papers.append(dict(record.get("paper") or {}))
            return papers

    # Contract aliases (P2): same behaviour under the names used by the
    # supplement-search design document.
    papers_by_ids = lookup_by_keys

    # -- query cache ----------------------------------------------------------

    def lookup_by_query(self, query: str) -> List[str]:
        """Alias of :meth:`lookup_query` accepting text OR a precomputed hash.

        A bare 12-hex-char string that is already a known query hash is used
        verbatim; anything else is treated as query text and normalised.
        """

        text = str(query or "")
        if _QUERY_HASH_RE.match(text):
            with self._lock:
                self._ensure_loaded()
                record = self._queries.get(text)
                if record is not None:
                    return [str(key) for key in (record.get("paper_keys") or [])]
        return self.lookup_query(text)

    def lookup_query(self, query: str) -> List[str]:
        """Paper keys previously recorded for this (normalised) query."""

        with self._lock:
            self._ensure_loaded()
            record = self._queries.get(query_hash(query))
            if record is None:
                return []
            return [str(key) for key in (record.get("paper_keys") or [])]

    def record_query(
        self,
        query: str,
        paper_keys: Sequence[str],
        run_id: str = "",
        round: int = 0,
    ) -> str:
        """Record query → paper keys mapping; returns the query hash."""

        digest = query_hash(query)
        record = {
            "query_hash": digest,
            "query": str(query or ""),
            "paper_keys": [str(key) for key in (paper_keys or [])],
            "run_id": run_id,
            "round": round,
        }
        with self._lock:
            self._ensure_loaded()
            self._queries[digest] = record
            self._append(self.query_cache_path, record)
        return digest
