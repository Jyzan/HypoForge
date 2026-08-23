"""Canonical, metadata-preserving paper deduplication."""

from __future__ import annotations

from collections.abc import Sequence
from difflib import SequenceMatcher
from typing import Iterable

from ..models import FulltextStatus, PaperRecord
from ..protocols import PaperDeduplicatorProtocol
from ._text import normalize_text, tokenize


_FULLTEXT_RANK = {
    FulltextStatus.UNKNOWN: 0,
    FulltextStatus.UNAVAILABLE: 0,
    FulltextStatus.FAILED: 0,
    FulltextStatus.ABSTRACT_ONLY: 1,
    FulltextStatus.XML_AVAILABLE: 2,
    FulltextStatus.HTML_AVAILABLE: 2,
    FulltextStatus.PDF_AVAILABLE: 2,
    FulltextStatus.DOWNLOADED: 3,
}
_UNTITLED = {"untitled", "(untitled)", "unknown", "(unknown)"}


def _normalize_doi(value: str) -> str:
    doi = (value or "").strip().casefold()
    for prefix in (
        "https://doi.org/",
        "http://doi.org/",
        "https://dx.doi.org/",
        "http://dx.doi.org/",
        "doi:",
        "doi ",
    ):
        if doi.startswith(prefix):
            doi = doi[len(prefix) :]
            break
    return doi.strip().rstrip(".,;)")


def _normalize_identifier(value: str, *prefixes: str) -> str:
    identifier = (value or "").strip().casefold()
    for prefix in prefixes:
        normalized_prefix = prefix.casefold()
        if identifier.startswith(normalized_prefix):
            identifier = identifier[len(normalized_prefix) :]
            break
    return identifier.strip()


def _identifier_keys(paper: PaperRecord) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    external_lookup = {
        normalize_text(namespace).replace(" ", "_"): str(value)
        for namespace, value in paper.external_ids.items()
    }
    doi = _normalize_doi(paper.doi or external_lookup.get("doi", ""))
    pmid = _normalize_identifier(
        paper.pmid or external_lookup.get("pmid", ""), "pmid:"
    )
    pmcid = _normalize_identifier(
        paper.pmcid or external_lookup.get("pmcid", ""), "pmcid:"
    )
    if doi:
        keys.add(("doi", doi))
    if pmid:
        keys.add(("pmid", pmid))
    if pmcid:
        keys.add(("pmcid", pmcid))
    for namespace, value in paper.external_ids.items():
        normalized_namespace = normalize_text(namespace).replace(" ", "_")
        normalized_value = _normalize_identifier(str(value))
        if normalized_namespace and normalized_value:
            keys.add((f"external:{normalized_namespace}", normalized_value))
    paper_id = _normalize_identifier(paper.paper_id)
    if paper_id and paper_id != "unknown" and not paper_id.endswith(":unknown"):
        keys.add(("paper_id", paper_id))
    return keys


def _normalized_title(paper: PaperRecord) -> str:
    title = normalize_text(paper.title)
    return "" if title in _UNTITLED else title


def _metadata_conflicts(left: PaperRecord, right: PaperRecord) -> bool:
    """Reject records whose stable identifiers explicitly disagree."""

    pairs = (
        (_normalize_doi(left.doi), _normalize_doi(right.doi)),
        (
            _normalize_identifier(left.pmid, "pmid:"),
            _normalize_identifier(right.pmid, "pmid:"),
        ),
        (
            _normalize_identifier(left.pmcid, "pmcid:"),
            _normalize_identifier(right.pmcid, "pmcid:"),
        ),
    )
    return any(
        bool(left_value and right_value and left_value != right_value)
        for left_value, right_value in pairs
    )


def _titles_match(left: PaperRecord, right: PaperRecord) -> bool:
    left_title = _normalized_title(left)
    right_title = _normalized_title(right)
    if not left_title or not right_title:
        return False
    if left_title == right_title:
        return True
    if min(len(left_title), len(right_title)) < 20:
        return False
    if (
        left.year is not None
        and right.year is not None
        and abs(left.year - right.year) > 1
    ):
        return False
    ratio = SequenceMatcher(None, left_title, right_title).ratio()
    left_tokens = set(tokenize(left_title))
    right_tokens = set(tokenize(right_title))
    union = left_tokens | right_tokens
    jaccard = len(left_tokens & right_tokens) / len(union) if union else 0.0
    return ratio >= 0.96 and jaccard >= 0.85


def _matches(left: PaperRecord, right: PaperRecord) -> bool:
    if _metadata_conflicts(left, right):
        return False
    if _identifier_keys(left) & _identifier_keys(right):
        return True
    return _titles_match(left, right)


def _stable_union(left: Iterable[str], right: Iterable[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in [*left, *right]:
        normalized = normalize_text(str(value))
        if normalized and normalized not in seen:
            output.append(str(value).strip())
            seen.add(normalized)
    return output


def _merge_records(canonical: PaperRecord, incoming: PaperRecord) -> PaperRecord:
    external_ids = dict(incoming.external_ids)
    external_ids.update(canonical.external_ids)
    rank_scores = dict(incoming.rank_scores)
    for key, value in canonical.rank_scores.items():
        rank_scores[key] = max(value, rank_scores.get(key, value))

    if canonical.is_open_access is True or incoming.is_open_access is True:
        is_open_access = True
    elif canonical.is_open_access is False or incoming.is_open_access is False:
        is_open_access = False
    else:
        is_open_access = None

    fulltext_status = max(
        (canonical.fulltext_status, incoming.fulltext_status),
        key=lambda status: _FULLTEXT_RANK[status],
    )
    citations = [
        value
        for value in (canonical.citation_count, incoming.citation_count)
        if value is not None
    ]
    canonical_title = _normalized_title(canonical)

    return canonical.model_copy(
        update={
            "title": canonical.title if canonical_title else incoming.title,
            "abstract": max(
                (canonical.abstract, incoming.abstract), key=lambda value: len(value or "")
            ),
            "authors": max(
                (canonical.authors, incoming.authors), key=lambda value: len(value)
            ),
            "year": canonical.year if canonical.year is not None else incoming.year,
            "journal": canonical.journal or incoming.journal,
            "doi": canonical.doi or incoming.doi,
            "pmid": canonical.pmid or incoming.pmid,
            "pmcid": canonical.pmcid or incoming.pmcid,
            "external_ids": external_ids,
            "citation_count": max(citations) if citations else None,
            "publication_type": canonical.publication_type
            or incoming.publication_type,
            "sources": _stable_union(canonical.sources, incoming.sources),
            "is_open_access": is_open_access,
            "fulltext_status": fulltext_status,
            "rank_scores": rank_scores,
        }
    )


class PaperDeduplicator(PaperDeduplicatorProtocol):
    """Deduplicate papers while preserving a stable canonical identity."""

    tool_name = "paper_deduplicator"

    async def deduplicate(
        self,
        papers: Sequence[PaperRecord],
        existing_papers: Sequence[PaperRecord] = (),
    ) -> list[PaperRecord]:
        canonicals = [paper.model_copy(deep=True) for paper in existing_papers]
        output_indices: list[int] = []
        output_seen: set[int] = set()

        for paper in papers:
            incoming = paper.model_copy(deep=True)
            match_index = next(
                (
                    index
                    for index, canonical in enumerate(canonicals)
                    if _matches(canonical, incoming)
                ),
                None,
            )
            if match_index is None:
                canonicals.append(incoming)
                match_index = len(canonicals) - 1
            else:
                canonicals[match_index] = _merge_records(
                    canonicals[match_index], incoming
                )
            if match_index not in output_seen:
                output_indices.append(match_index)
                output_seen.add(match_index)

        return [canonicals[index] for index in output_indices]
