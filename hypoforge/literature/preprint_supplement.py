"""Conditional Science-125 disciplinary-preprint full-text supplement.

This module is intentionally separate from the primary M2 discovery stack.
It reuses the semantic query portfolio already executed by M2 and is called
only after the existing candidate backfill and open-access expansion paths
have failed to reach the parsed-full-text target.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
import hashlib
import html
import json
import re
import time
from typing import Any
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from .models import (
    FulltextStatus,
    PaperRecord,
    QueryIntent,
    SearchQuery,
    SearchRunResult,
    StopReason,
)
from .protocols import (
    LiteratureSourceProtocol,
    PaperDeduplicatorProtocol,
    PaperRankerProtocol,
    ScoutReaderProtocol,
)
from .search.ranking import rerank_with_scout
from .sources.specialist_sources import CrossrefSource, EuropePMCSource


JsonBackend = Callable[[str, float], Awaitable[dict[str, Any]]]
TextBackend = Callable[[str, float], Awaitable[str]]

_FIELD_TAG_PATTERN = re.compile(r"\[[^\]]+\]")
_SITE_PATTERN = re.compile(r"\bsite:\S+", re.IGNORECASE)
_TOKEN_PATTERN = re.compile(r"[\w\u3400-\u9fff-]+", re.UNICODE)


def _clean(value: object) -> str:
    return " ".join(str(value or "").split())


def _normalized_doi_url(value: object) -> str:
    text = _clean(value)
    return text.rsplit("doi.org/", 1)[-1] if "doi.org/" in text else text


def _stable_id(prefix: str, title: str, native: str = "", doi: str = "") -> str:
    normalized_doi = PaperRecord.normalize_doi(doi)
    if normalized_doi:
        return f"DOI:{normalized_doi}"
    if native:
        return f"{prefix.upper()}:{native}"
    digest = hashlib.sha256(title.casefold().encode("utf-8")).hexdigest()[:20]
    return f"{prefix.upper()}-TITLE:{digest}"


def _year(value: object) -> int | None:
    match = re.search(r"\b(?:19|20)\d{2}\b", str(value or ""))
    return int(match.group()) if match else None


async def _default_json_backend(url: str, timeout: float) -> dict[str, Any]:
    def fetch() -> dict[str, Any]:
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "HypoForge/0.1 (academic preprint supplement)",
            },
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    return await asyncio.to_thread(fetch)


async def _default_text_backend(url: str, timeout: float) -> str:
    def fetch() -> str:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "HypoForge/0.1 (academic preprint supplement)"},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read().decode("utf-8", errors="replace")

    return await asyncio.to_thread(fetch)


def sanitize_preprint_query(text: str) -> str:
    """Convert an already-issued backend query into portable preprint syntax."""

    text = _FIELD_TAG_PATTERN.sub(" ", str(text or ""))
    text = _SITE_PATTERN.sub(" ", text)
    text = re.sub(r"\b(?:title|abstract|keyword):", " ", text, flags=re.I)
    return _clean(text)


def preprint_query_variants(
    sub_question: str,
    key_entities: Sequence[str],
    existing_queries: Sequence[SearchQuery],
    *,
    limit: int = 2,
) -> list[str]:
    """Reuse the primary M2 portfolio, with deterministic portable fallbacks."""

    variants: list[str] = []
    seen: set[str] = set()
    for query in existing_queries:
        text = sanitize_preprint_query(query.text)
        key = text.casefold()
        if text and key not in seen:
            seen.add(key)
            variants.append(text)
    fallbacks = [
        sanitize_preprint_query(sub_question),
        _clean(" ".join(str(entity) for entity in key_entities[:4])),
    ]
    for text in fallbacks:
        key = text.casefold()
        if text and key not in seen:
            seen.add(key)
            variants.append(text)
    return variants[: max(1, int(limit))]


class OpenRxivSource(LiteratureSourceProtocol):
    """Keyword discovery for bioRxiv/medRxiv through Europe PMC indexing."""

    def __init__(
        self,
        provider: str,
        *,
        timeout_seconds: float = 15.0,
        europe_pmc: EuropePMCSource | None = None,
    ) -> None:
        provider = provider.casefold().strip()
        if provider not in {"biorxiv", "medrxiv"}:
            raise ValueError("OpenRxivSource provider must be biorxiv or medrxiv")
        self.provider = provider
        self.source_name = provider
        self.timeout_seconds = float(timeout_seconds)
        self._europe_pmc = europe_pmc or EuropePMCSource(
            timeout_seconds=timeout_seconds
        )

    def _pdf_url(self, doi: str) -> str:
        if not doi or not doi.startswith(("10.1101/", "10.64898/")):
            return ""
        return f"https://www.{self.provider}.org/content/{doi}v1.full.pdf"

    async def search(self, query: SearchQuery, limit: int = 20) -> list[PaperRecord]:
        routed = query.model_copy(update={
            "text": f"({sanitize_preprint_query(query.text)}) AND PUBLISHER:{self.provider}",
            "target_source": "europe_pmc",
        })
        rows = await self._europe_pmc.search(routed, limit=limit)
        output: list[PaperRecord] = []
        for row in rows:
            external_ids = dict(row.external_ids)
            pdf_url = self._pdf_url(row.doi)
            if pdf_url:
                external_ids["oa_pdf_url"] = pdf_url
            output.append(row.model_copy(update={
                "sources": list(dict.fromkeys([*row.sources, self.provider])),
                "publication_type": "preprint",
                "external_ids": external_ids,
                "is_open_access": True,
                "fulltext_status": (
                    FulltextStatus.PDF_AVAILABLE if pdf_url else row.fulltext_status
                ),
            }))
        return output


class CrossrefPreprintSource(LiteratureSourceProtocol):
    """Source-specific Crossref discovery for preprint servers with DOI prefixes."""

    def __init__(
        self,
        source_name: str,
        prefix: str,
        *,
        mailto: str = "",
        timeout_seconds: float = 15.0,
        backend: JsonBackend | None = None,
    ) -> None:
        self.source_name = source_name.casefold().strip()
        self.prefix = prefix.strip()
        self.mailto = mailto.strip()
        self.timeout_seconds = float(timeout_seconds)
        self._backend = backend or _default_json_backend

    async def search(self, query: SearchQuery, limit: int = 20) -> list[PaperRecord]:
        params = {
            "query.bibliographic": sanitize_preprint_query(query.text),
            "filter": f"prefix:{self.prefix}",
            "rows": str(min(max(1, limit), 20)),
            "select": (
                "DOI,title,author,published,container-title,"
                "is-referenced-by-count,abstract,type,link,URL"
            ),
        }
        if self.mailto:
            params["mailto"] = self.mailto
        await asyncio.to_thread(CrossrefSource._wait_for_slot)
        data = await self._backend(
            "https://api.crossref.org/works?" + urllib.parse.urlencode(params),
            self.timeout_seconds,
        )
        output: list[PaperRecord] = []
        for row in (data.get("message") or {}).get("items") or []:
            title_values = row.get("title") or []
            title = _clean(title_values[0] if title_values else "")
            if not title:
                continue
            doi = PaperRecord.normalize_doi(row.get("DOI"))
            authors = [
                _clean(" ".join(filter(None, [author.get("given"), author.get("family")])))
                for author in row.get("author") or []
                if isinstance(author, dict)
            ]
            dates = ((row.get("published") or {}).get("date-parts") or [[]])[0]
            abstract = _clean(re.sub(r"<[^>]+>", " ", str(row.get("abstract") or "")))
            pdf_url = next((
                _clean(link.get("URL"))
                for link in row.get("link") or []
                if isinstance(link, dict)
                and (
                    "pdf" in _clean(link.get("content-type")).casefold()
                    or "/pdf/" in _clean(link.get("URL")).casefold()
                )
            ), "")
            external_ids = {"landing_url": _clean(row.get("URL"))}
            if pdf_url:
                external_ids["oa_pdf_url"] = pdf_url
            output.append(PaperRecord(
                paper_id=_stable_id(self.source_name, title, doi=doi),
                title=title,
                abstract=abstract,
                authors=[author for author in authors if author],
                year=int(dates[0]) if dates else None,
                journal="; ".join(row.get("container-title") or []),
                doi=doi,
                external_ids=external_ids,
                citation_count=int(row.get("is-referenced-by-count") or 0),
                publication_type="preprint",
                sources=[self.source_name],
                is_open_access=True,
                fulltext_status=(
                    FulltextStatus.PDF_AVAILABLE if pdf_url
                    else FulltextStatus.ABSTRACT_ONLY if abstract
                    else FulltextStatus.UNKNOWN
                ),
            ))
        return output


class OSFPreprintSource(LiteratureSourceProtocol):
    """Provider-scoped OSF discovery with DOI/OA-PDF enrichment hooks."""

    def __init__(
        self,
        provider: str,
        *,
        timeout_seconds: float = 15.0,
        backend: JsonBackend | None = None,
    ) -> None:
        self.provider = provider.casefold().strip()
        self.source_name = self.provider
        self.timeout_seconds = float(timeout_seconds)
        self._backend = backend or _default_json_backend

    @staticmethod
    def _filter_terms(text: str) -> list[str]:
        tokens = [
            token for token in _TOKEN_PATTERN.findall(sanitize_preprint_query(text))
            if len(token) >= 4
            and token.casefold() not in {
                "and", "or", "not", "review", "systematic", "evidence",
                "mechanism", "foundational", "open", "access", "preprint",
            }
        ]
        unique = list(dict.fromkeys(tokens))[:4]
        if not unique:
            return []
        # OSF's string filter behaves like a phrase match.  Keep the compact
        # semantic query first, then relax to its concepts only when needed.
        return [" ".join(unique), *unique]

    async def _pdf_from_primary_file(self, row: Mapping[str, Any]) -> str:
        primary = ((row.get("relationships") or {}).get("primary_file") or {})
        file_id = ((primary.get("data") or {}).get("id") or "")
        related = (((primary.get("links") or {}).get("related") or {}).get("href") or "")
        if not file_id and not related:
            return ""
        url = related or f"https://api.osf.io/v2/files/{file_id}/"
        data = await self._backend(url, self.timeout_seconds)
        links = ((data.get("data") or {}).get("links") or {})
        return _clean(links.get("download"))

    async def search(self, query: SearchQuery, limit: int = 20) -> list[PaperRecord]:
        terms = self._filter_terms(query.text)
        if not terms:
            return []
        base = f"https://api.osf.io/v2/preprint_providers/{self.provider}/preprints/"
        rows: list[Mapping[str, Any]] = []
        seen_ids: set[str] = set()
        for field in ("title", "description"):
            for term in terms:
                params = urllib.parse.urlencode({
                    f"filter[{field}]": term,
                    "page[size]": min(max(limit, 1), 10),
                })
                data = await self._backend(
                    base + "?" + params, self.timeout_seconds
                )
                for row in data.get("data") or []:
                    native = _clean(row.get("id"))
                    if native and native not in seen_ids:
                        seen_ids.add(native)
                        rows.append(row)
                if len(rows) >= limit:
                    break
            if len(rows) >= limit:
                break

        output: list[PaperRecord] = []
        for row in rows[:limit]:
            attributes = row.get("attributes") or {}
            links = row.get("links") or {}
            title = _clean(attributes.get("title"))
            if not title:
                continue
            preprint_doi = PaperRecord.normalize_doi(
                _normalized_doi_url(links.get("preprint_doi"))
            )
            journal_doi = PaperRecord.normalize_doi(
                _normalized_doi_url(links.get("doi") or attributes.get("doi"))
            )
            doi = preprint_doi or journal_doi
            try:
                pdf_url = await self._pdf_from_primary_file(row)
            except Exception:
                pdf_url = ""
            external_ids = {
                "landing_url": _clean(links.get("html")),
                "preprint_doi": preprint_doi,
            }
            if pdf_url:
                external_ids["oa_pdf_url"] = pdf_url
            output.append(PaperRecord(
                paper_id=_stable_id(
                    self.provider, title, native=_clean(row.get("id")), doi=doi
                ),
                title=title,
                abstract=_clean(attributes.get("description")),
                year=_year(attributes.get("date_published")),
                doi=doi,
                external_ids=external_ids,
                citation_count=0,
                publication_type="preprint",
                sources=[self.provider],
                is_open_access=True,
                fulltext_status=(
                    FulltextStatus.PDF_AVAILABLE if pdf_url
                    else FulltextStatus.ABSTRACT_ONLY
                ),
            ))
        return output


class MechanicsArxivSource(LiteratureSourceProtocol):
    """MechanicsArXiv OAI-PMH harvest searched locally by title/abstract."""

    source_name = "mechanicsarxiv"
    endpoint = (
        "https://www.mechanicsarxiv.org/index.php/engineering/oai"
        "?verb=ListRecords&metadataPrefix=oai_dc"
    )

    def __init__(
        self,
        *,
        timeout_seconds: float = 15.0,
        text_backend: TextBackend | None = None,
    ) -> None:
        self.timeout_seconds = float(timeout_seconds)
        self._text_backend = text_backend or _default_text_backend
        self._records: list[PaperRecord] | None = None
        self._load_lock = asyncio.Lock()

    async def _load(self) -> list[PaperRecord]:
        if self._records is not None:
            return self._records
        async with self._load_lock:
            if self._records is not None:
                return self._records
            payload = await self._text_backend(self.endpoint, self.timeout_seconds)
            root = ET.fromstring(payload)
            ns = {
                "o": "http://www.openarchives.org/OAI/2.0/",
                "dc": "http://purl.org/dc/elements/1.1/",
            }
            records: list[PaperRecord] = []
            for record in root.findall(".//o:record", ns):
                def values(name: str) -> list[str]:
                    return [
                        _clean(node.text)
                        for node in record.findall(f".//dc:{name}", ns)
                        if _clean(node.text)
                    ]

                titles = values("title")
                if not titles:
                    continue
                identifiers = values("identifier")
                relations = values("relation")
                landing = identifiers[0] if identifiers else ""
                records.append(PaperRecord(
                    paper_id=_stable_id(
                        self.source_name, titles[0], native=landing
                    ),
                    title=titles[0],
                    abstract=" ".join(values("description")),
                    authors=values("creator"),
                    year=_year((values("date") or [""])[0]),
                    external_ids={
                        "landing_url": landing,
                        "relation_url": relations[0] if relations else "",
                    },
                    publication_type="preprint",
                    sources=[self.source_name],
                    is_open_access=True,
                    # A relation is a landing/version page, not yet a verified
                    # PDF.  ``_add_pdf`` upgrades this only after extraction.
                    fulltext_status=FulltextStatus.ABSTRACT_ONLY,
                ))
            self._records = records
            return records

    @staticmethod
    def _overlap(text: str, paper: PaperRecord) -> int:
        tokens = {
            token.casefold() for token in _TOKEN_PATTERN.findall(text)
            if len(token) > 2 and token.casefold() not in {"and", "or", "not"}
        }
        haystack = f"{paper.title} {paper.abstract}".casefold()
        return sum(token in haystack for token in tokens)

    async def _add_pdf(self, paper: PaperRecord) -> PaperRecord:
        relation = paper.external_ids.get("relation_url", "")
        if not relation:
            return paper
        page = await self._text_backend(relation, self.timeout_seconds)
        matches = re.findall(
            r'https?://[^"\s<>]+/preprint/download/[^"\s<>]+', page
        ) or re.findall(r'href="([^"]+/preprint/download/[^"]+)"', page)
        if not matches:
            return paper
        external_ids = dict(paper.external_ids)
        external_ids["oa_pdf_url"] = urllib.parse.urljoin(
            relation, html.unescape(matches[0])
        )
        return paper.model_copy(update={
            "external_ids": external_ids,
            "fulltext_status": FulltextStatus.PDF_AVAILABLE,
        })

    async def search(self, query: SearchQuery, limit: int = 20) -> list[PaperRecord]:
        records = await self._load()
        text = sanitize_preprint_query(query.text)
        ranked = sorted(
            records,
            key=lambda paper: (-self._overlap(text, paper), paper.title),
        )
        selected = [paper for paper in ranked if self._overlap(text, paper) > 0][
            :limit
        ]
        output: list[PaperRecord] = []
        for paper in selected:
            try:
                output.append(await self._add_pdf(paper))
            except Exception:
                output.append(paper)
        return output


def route_science125_preprint_sources(
    domains: Sequence[str],
    sub_question: str,
    *,
    max_sources: int = 2,
) -> list[str]:
    """Route only the twelve Science-125 domains to additive preprint sources."""

    output: list[str] = []

    def add(*names: str) -> None:
        for name in names:
            if name not in output:
                output.append(name)

    def contains(text: str, terms: Sequence[str]) -> bool:
        return any(term in text for term in terms)

    def apply_rules(text: str) -> None:
        if contains(
            text, ("medicine", "health", "clinical", "医学", "健康", "临床")
        ):
            add("medrxiv")
        if contains(text, ("neuro", "brain", "神经", "脑")):
            add("biorxiv", "medrxiv")
        elif contains(
            text,
            ("biology", "biolog", "cell", "genom", "生物", "细胞", "基因"),
        ):
            add("biorxiv")
        if contains(
            text, ("chemistry", "chemical", "molecule", "化学", "分子")
        ):
            add("chemrxiv")
            if contains(
                text,
                ("electrochem", "battery", "storage", "电化学", "电池", "储能"),
            ):
                add("ecsarxiv")
        # An explicit Energy domain must win over generic engineering words in
        # the sub-question, otherwise the two-source cap would hide ECSArXiv.
        if contains(
            text,
            ("energy", "hydrogen", "battery", "renewable", "能源", "氢能", "电池", "可再生"),
        ):
            add("ecsarxiv", "engrxiv")
        if contains(
            text,
            (
                "engineering", "materials", "mechanic", "fracture", "fluid",
                "工程", "材料", "力学", "断裂", "流体",
            ),
        ):
            add("mechanicsarxiv", "engrxiv")
        if contains(
            text,
            (
                "information science", "computer", "algorithm",
                "artificial intelligence", "machine learning",
                "信息", "计算机", "算法", "人工智能",
            ),
        ):
            add("techrxiv")
        if contains(
            text,
            (
                "ecology", "species", "evolution", "conservation",
                "生态", "物种", "进化",
            ),
        ):
            add("ecoevorxiv", "biorxiv")
        if contains(
            text,
            (
                "climate", "ocean", "earth", "geolog", "environment",
                "气候", "海洋", "地球", "地质", "环境",
            ),
        ):
            add("eartharxiv")

    question_text = sub_question.casefold()
    # Strong question-level Earth-system cues override a coarse Science-125
    # label such as "Ecology".  Without this, Q110 exhausts both slots on
    # EcoEvoRxiv/bioRxiv before EarthArXiv can be considered.
    if contains(
        question_text,
        (
            "earthquake", "seismic", "tsunami", "hurricane", "cyclone",
            "weather event", "geophys", "地震", "海啸", "飓风", "气旋",
        ),
    ):
        add("eartharxiv")

    # M1's explicit domain otherwise has priority; question text adds
    # interdisciplinary sources only when a slot remains.
    apply_rules(" ".join(map(str, domains)).casefold())
    apply_rules(question_text)

    # Mathematics, physics and astronomy already use arXiv in the unchanged
    # primary route, so the additive layer deliberately avoids a duplicate call.
    return output[: max(0, int(max_sources))]


def build_science125_preprint_sources(
    *,
    crossref_mailto: str = "",
    timeout_seconds: float = 15.0,
) -> dict[str, LiteratureSourceProtocol]:
    """Build the additive source registry; public reads require no new keys."""

    return {
        "biorxiv": OpenRxivSource("biorxiv", timeout_seconds=timeout_seconds),
        "medrxiv": OpenRxivSource("medrxiv", timeout_seconds=timeout_seconds),
        "chemrxiv": CrossrefPreprintSource(
            "chemrxiv", "10.26434", mailto=crossref_mailto,
            timeout_seconds=timeout_seconds,
        ),
        "techrxiv": CrossrefPreprintSource(
            "techrxiv", "10.36227", mailto=crossref_mailto,
            timeout_seconds=timeout_seconds,
        ),
        "mechanicsarxiv": MechanicsArxivSource(
            timeout_seconds=timeout_seconds
        ),
        "engrxiv": OSFPreprintSource("engrxiv", timeout_seconds=timeout_seconds),
        "eartharxiv": OSFPreprintSource(
            "eartharxiv", timeout_seconds=timeout_seconds
        ),
        "ecoevorxiv": OSFPreprintSource(
            "ecoevorxiv", timeout_seconds=timeout_seconds
        ),
        "ecsarxiv": OSFPreprintSource("ecsarxiv", timeout_seconds=timeout_seconds),
    }


class Science125PreprintSupplementer:
    """Search and Scout-filter an additive domain preprint portfolio."""

    tool_name = "science125_preprint_supplement"

    def __init__(
        self,
        *,
        sources: Mapping[str, LiteratureSourceProtocol],
        deduplicator: PaperDeduplicatorProtocol,
        ranker: PaperRankerProtocol,
        scout_reader: ScoutReaderProtocol,
        max_sources: int = 2,
        per_source_limit: int = 2,
        max_candidates: int = 4,
        source_timeout_seconds: float = 15.0,
        scout_timeout_seconds: float = 60.0,
        min_relevance: float = 0.70,
        min_directness: float = 0.45,
    ) -> None:
        if min(max_sources, per_source_limit, max_candidates) <= 0:
            raise ValueError("preprint supplement limits must be positive")
        self.sources = {name.casefold(): source for name, source in sources.items()}
        self.deduplicator = deduplicator
        self.ranker = ranker
        self.scout_reader = scout_reader
        self.max_sources = int(max_sources)
        self.per_source_limit = int(per_source_limit)
        self.max_candidates = int(max_candidates)
        self.source_timeout_seconds = float(source_timeout_seconds)
        self.scout_timeout_seconds = float(scout_timeout_seconds)
        self.min_relevance = float(min_relevance)
        self.min_directness = float(min_directness)

    async def search(
        self,
        sub_question: str,
        *,
        domains: Sequence[str],
        key_entities: Sequence[str],
        existing_queries: Sequence[SearchQuery],
        existing_papers: Sequence[PaperRecord],
    ) -> SearchRunResult:
        started = time.monotonic()
        source_names = [
            name for name in route_science125_preprint_sources(
                domains, sub_question, max_sources=self.max_sources
            )
            if name in self.sources
        ]
        variants = preprint_query_variants(
            sub_question, key_entities, existing_queries, limit=2
        )
        queries = [
            SearchQuery(
                query_id=f"preprint-{source_name}-{index}",
                text=text,
                intent=QueryIntent.CORE,
                target_source=source_name,
                purpose="conditional_preprint_fulltext_supplement",
                relation_to_question=(
                    "Reuses the primary M2 semantic query after existing "
                    "full-text routes remain below target."
                ),
            )
            for source_name in source_names
            for index, text in enumerate(variants, 1)
        ]
        errors: list[str] = []
        counts: dict[str, int] = {}

        async def run(query: SearchQuery) -> list[PaperRecord]:
            try:
                rows = await asyncio.wait_for(
                    self.sources[query.target_source].search(
                        query, limit=self.per_source_limit
                    ),
                    timeout=self.source_timeout_seconds,
                )
                counts[query.target_source] = (
                    counts.get(query.target_source, 0) + len(rows)
                )
                return rows
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                errors.append(
                    f"{query.target_source}: {type(exc).__name__}: {exc}"
                )
                return []

        batches = await asyncio.gather(*(run(query) for query in queries))
        raw = [paper for batch in batches for paper in batch]
        candidates = await self.deduplicator.deduplicate(
            raw, existing_papers=existing_papers
        )
        ranked = await self.ranker.rank(
            sub_question, candidates, limit=min(self.max_candidates, len(candidates))
        )
        if not ranked:
            return SearchRunResult(
                sub_question=sub_question,
                queries=queries,
                papers_found=len(raw),
                papers_after_dedup=len(candidates),
                candidates=[],
                failed_sources=list(dict.fromkeys(
                    error.split(":", 1)[0] for error in errors
                )),
                errors=errors,
                source_result_counts=counts,
                stage_elapsed_seconds={
                    "conditional_preprint_search": time.monotonic() - started,
                },
                stop_reason=StopReason.NO_RESULTS,
            )
        try:
            notes = await asyncio.wait_for(
                self.scout_reader.read(sub_question, ranked),
                timeout=self.scout_timeout_seconds,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            errors.append(f"preprint_scout: {type(exc).__name__}: {exc}")
            notes = []
        note_by_id = {note.paper_id: note for note in notes}
        admitted = [
            paper for paper in ranked
            if (
                (note := note_by_id.get(paper.paper_id)) is not None
                and note.relevance_to_question >= self.min_relevance
                and (
                    note.directness_to_question
                    if note.directness_to_question is not None
                    else note.relevance_to_question
                ) >= self.min_directness
            )
        ]
        admitted = rerank_with_scout(
            admitted,
            [note_by_id[paper.paper_id] for paper in admitted],
            selection_limit=min(2, len(admitted)),
        )
        return SearchRunResult(
            sub_question=sub_question,
            queries=queries,
            papers_found=len(raw),
            papers_after_dedup=len(candidates),
            candidates=admitted[: self.max_candidates],
            scout_notes=[note_by_id[paper.paper_id] for paper in admitted],
            failed_sources=list(dict.fromkeys(
                error.split(":", 1)[0] for error in errors
                if not error.startswith("preprint_scout:")
            )),
            errors=errors,
            source_result_counts=counts,
            stage_elapsed_seconds={
                "conditional_preprint_search": time.monotonic() - started,
            },
            stop_reason=(
                StopReason.PLAN_COMPLETE if admitted else StopReason.NO_RESULTS
            ),
        )
