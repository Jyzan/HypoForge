from __future__ import annotations

import hashlib
import logging
import re
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

from ...tools.pubmed_search import pubmed_count, search_pubmed_strict
from ..models import FulltextStatus, PaperRecord, SearchQuery
from ..protocols import LiteratureSourceProtocol

logger = logging.getLogger(__name__)

PubMedBackend = Callable[
    [str, int],
    Awaitable[Sequence[Mapping[str, Any]]],
]

# Async count probe: (query_text, timeout_seconds | None) -> hit count.
PubMedCountProbe = Callable[[str, float | None], Awaitable[int]]

# Mirrors ``query_planner._FIELD_TAG_PATTERN`` (duplicated here to avoid a
# sources -> search import edge).
_FIELD_TAG_PATTERN = re.compile(r"\[[^\]]+\]")
_TOKEN_PATTERN = re.compile(r'"[^"]*"|\(|\)|[^\s"()]+')

#: Maximum esearch-count probes per query in the relaxation ladder.
MAX_RELAXATION_PROBES = 3


def _normalized_title(value: object) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).split())


def _paper_id(pmid: str, doi: str, title: str) -> str:
    if pmid:
        return f"PMID:{pmid}"
    if doi:
        return f"DOI:{doi}"
    digest = hashlib.sha256(_normalized_title(title).encode("utf-8")).hexdigest()[:16]
    return f"TITLE:{digest}"


def _to_paper_record(raw: Mapping[str, Any]) -> PaperRecord | None:
    pmid = str(raw.get("pmid") or "").strip()
    doi = PaperRecord.normalize_doi(raw.get("doi"))
    title = str(raw.get("title") or "").strip()
    if not pmid and not doi and not title:
        return None
    abstract = str(raw.get("abstract") or "").strip()
    year_value = raw.get("year")
    try:
        parsed_year = int(year_value) if year_value else 0
    except (TypeError, ValueError):
        parsed_year = 0
    year = parsed_year if 1000 <= parsed_year <= 3000 else None
    return PaperRecord(
        paper_id=_paper_id(pmid, doi, title),
        title=title or doi or f"PubMed {pmid}",
        abstract=abstract,
        authors=[str(item) for item in raw.get("authors") or [] if str(item).strip()],
        year=year,
        journal=str(raw.get("journal") or "").strip(),
        doi=doi,
        pmid=pmid,
        sources=["pubmed"],
        fulltext_status=(
            FulltextStatus.ABSTRACT_ONLY if abstract else FulltextStatus.UNKNOWN
        ),
    )


# ---------------------------------------------------------------------------
# Zero-result relaxation ladder
# ---------------------------------------------------------------------------

def strip_field_tags(text: str) -> str:
    """Remove every PubMed field tag, keeping quoted phrases and AND/OR."""
    return " ".join(_FIELD_TAG_PATTERN.sub(" ", text).split())


def split_top_level_and_clauses(text: str) -> list[str]:
    """Split *text* on top-level ``AND`` (parentheses/quotes respected)."""
    clauses: list[list[str]] = []
    current: list[str] = []
    depth = 0
    for token in _TOKEN_PATTERN.findall(text):
        if token == "(":
            depth += 1
            current.append(token)
        elif token == ")":
            depth = max(0, depth - 1)
            current.append(token)
        elif depth == 0 and token.upper() == "AND":
            if current:
                clauses.append(current)
                current = []
        else:
            current.append(token)
    if current:
        clauses.append(current)
    return [" ".join(clause) for clause in clauses if clause]


def _canon(text: str) -> str:
    return " ".join(text.split()).casefold()


def relaxation_candidates(original_text: str) -> list[str]:
    """Ordered relaxation ladder for a zero-result PubMed query.

    - L1: strip every field tag (lets PubMed ATM auto-map terms);
    - L2: drop AND clauses right-to-left, always keeping the first two
      core concepts;
    - L3: keep the first 2-3 concepts with phrase quotes removed.

    Duplicates (case/whitespace-insensitive) of earlier candidates and of
    the original query are omitted.
    """
    seen = {_canon(original_text)}
    candidates: list[str] = []

    def add(candidate: str) -> None:
        candidate = " ".join(candidate.split())
        if candidate and _canon(candidate) not in seen:
            seen.add(_canon(candidate))
            candidates.append(candidate)

    # L1 — strip field tags.
    stripped = strip_field_tags(original_text)
    add(stripped)

    clauses = split_top_level_and_clauses(stripped)

    # L2 — drop AND clauses from the right, keep at least 2 concepts.
    if len(clauses) >= 3:
        for keep in range(len(clauses) - 1, 1, -1):
            add(" AND ".join(clauses[:keep]))

    # L3 — first 2-3 concepts, unquoted.
    core = clauses[:3]
    if len(core) >= 2:
        add(" AND ".join(clause.replace('"', "") for clause in core))
        add(" AND ".join(clause.replace('"', "") for clause in core[:2]))

    return candidates


async def _default_count_probe(text: str, timeout_seconds: float | None) -> int:
    return await pubmed_count(text, timeout_seconds=timeout_seconds)


async def run_zero_result_relaxation(
    original_text: str,
    fetch: Callable[[str], Awaitable[Sequence[Mapping[str, Any]]]],
    *,
    count_probe: PubMedCountProbe | None = None,
    max_probes: int = MAX_RELAXATION_PROBES,
    deadline: float | None = None,
) -> tuple[list[Mapping[str, Any]], str]:
    """Try progressively relaxed queries after the original returned nothing.

    ``fetch`` runs the real backend for a candidate query; ``count_probe``
    (when provided) performs a cheap esearch count first so ``efetch`` is
    only spent on candidates with hits. At most *max_probes* attempts are
    made. Returns ``(rows, relaxed_from)`` where ``relaxed_from`` is the
    original query text when a relaxed candidate hit, else ``("", "")``-
    style ``([], "")``.
    """
    attempts = 0
    for candidate in relaxation_candidates(original_text):
        if attempts >= max_probes:
            break
        if deadline is not None and deadline - time.monotonic() <= 0:
            break
        attempts += 1
        if count_probe is not None:
            remaining = None if deadline is None else deadline - time.monotonic()
            try:
                count = await count_probe(candidate, remaining)
            except Exception:
                logger.debug(
                    "PubMed relaxation count probe failed for %r",
                    candidate,
                    exc_info=True,
                )
                continue
            if count <= 0:
                logger.debug(
                    "PubMed relaxation probe found no hits for %r", candidate
                )
                continue
        if deadline is not None and deadline - time.monotonic() <= 0:
            break
        try:
            rows = list(await fetch(candidate))
        except Exception:
            logger.warning(
                "PubMed relaxation fetch failed for %r", candidate, exc_info=True
            )
            continue
        if rows:
            logger.info(
                "PubMed zero-result relaxation hit: original=%r relaxed=%r hits=%d",
                original_text,
                candidate,
                len(rows),
            )
            return rows, original_text
    return [], ""


def _tag_relaxed(
    rows: Sequence[Mapping[str, Any]],
    relaxed_from: str,
) -> list[PaperRecord]:
    records = [
        record
        for record in (_to_paper_record(row) for row in rows)
        if record is not None
    ]
    if relaxed_from:
        for record in records:
            record.relaxed_from = relaxed_from
    return records


# ---------------------------------------------------------------------------
# Source
# ---------------------------------------------------------------------------

class PubMedLiteratureSource(LiteratureSourceProtocol):
    source_name = "pubmed"

    def __init__(
        self,
        backend: PubMedBackend | None = None,
        *,
        timeout_seconds: float = 30.0,
        enable_relaxation: bool = True,
        count_probe: PubMedCountProbe | None = None,
        max_relaxation_probes: int = MAX_RELAXATION_PROBES,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_relaxation_probes <= 0:
            raise ValueError("max_relaxation_probes must be positive")
        self._uses_default_backend = backend is None
        self.backend = backend or search_pubmed_strict
        self.timeout_seconds = timeout_seconds
        self.enable_relaxation = enable_relaxation
        self.count_probe = count_probe
        self.max_relaxation_probes = max_relaxation_probes

    def _resolve_count_probe(self) -> PubMedCountProbe | None:
        if self.count_probe is not None:
            return self.count_probe
        # Only probe NCBI for real when the real backend is in use; injected
        # test backends fall back to probe-free (blind) relaxation.
        return _default_count_probe if self._uses_default_backend else None

    async def search(self, query: SearchQuery, limit: int = 20) -> list[PaperRecord]:
        deadline = time.monotonic() + self.timeout_seconds

        async def fetch(text: str, timeout_seconds: float | None = None):
            if self._uses_default_backend:
                return await self.backend(
                    text,
                    limit,
                    timeout_seconds=timeout_seconds or self.timeout_seconds,
                )
            return await self.backend(text, limit)

        rows = list(await fetch(query.text))
        relaxed_from = ""
        if not rows and self.enable_relaxation:
            rows, relaxed_from = await run_zero_result_relaxation(
                query.text,
                lambda text: fetch(
                    text,
                    timeout_seconds=(
                        None
                        if not self._uses_default_backend
                        else max(deadline - time.monotonic(), 0.01)
                    ),
                ),
                count_probe=self._resolve_count_probe(),
                max_probes=self.max_relaxation_probes,
                deadline=deadline,
            )
        return _tag_relaxed(rows, relaxed_from)
