"""Legal open-access enrichment and multi-location full-text resolution."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from ..models import ContentLevel, DocumentRecord, FulltextStatus, PaperRecord
from ..protocols import FulltextResolverProtocol


_FULLTEXT_LEVELS = {
    ContentLevel.STRUCTURED_FULLTEXT,
    ContentLevel.PDF,
    ContentLevel.HTML,
    ContentLevel.OCR,
}


def _clean(value: object) -> str:
    return " ".join(str(value or "").split())


def _json_get(
    url: str,
    *,
    timeout: float,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "HypoForge/0.1 (open-access resolution)",
            **(headers or {}),
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


@dataclass(frozen=True)
class AccessEnrichment:
    paper: PaperRecord
    urls: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()


class OpenAccessEnricher:
    """Resolve DOI records through OA metadata services without bypassing paywalls."""

    def __init__(
        self,
        *,
        email: str = "",
        openalex_api_key: str = "",
        semantic_scholar_api_key: str = "",
        timeout_seconds: float = 12.0,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.email = _clean(
            email or os.environ.get("UNPAYWALL_EMAIL")
            or os.environ.get("OPENALEX_MAILTO")
        )
        self.openalex_api_key = _clean(
            openalex_api_key or os.environ.get("OPENALEX_API_KEY")
        )
        self.semantic_scholar_api_key = _clean(
            semantic_scholar_api_key or os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
        )
        self.timeout_seconds = float(timeout_seconds)

    async def _request(
        self, provider: str, url: str, headers: dict[str, str] | None = None
    ) -> tuple[str, dict[str, Any] | None, str]:
        try:
            data = await asyncio.to_thread(
                _json_get,
                url,
                timeout=self.timeout_seconds,
                headers=headers,
            )
            return provider, data, ""
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return provider, None, f"{provider}: {type(exc).__name__}: {exc}"

    async def enrich(self, paper: PaperRecord) -> AccessEnrichment:
        urls: list[str] = []
        existing = _clean(paper.external_ids.get("oa_pdf_url"))
        if existing:
            urls.append(existing)
        if not paper.doi:
            return AccessEnrichment(paper=paper, urls=tuple(urls))

        doi = PaperRecord.normalize_doi(paper.doi)
        requests = []
        if self.email:
            requests.append(self._request(
                "unpaywall",
                "https://api.unpaywall.org/v2/"
                + urllib.parse.quote(doi, safe="")
                + "?" + urllib.parse.urlencode({"email": self.email}),
            ))
        openalex_params: dict[str, str] = {}
        if self.openalex_api_key:
            openalex_params["api_key"] = self.openalex_api_key
        if self.email:
            openalex_params["mailto"] = self.email
        openalex_url = (
            "https://api.openalex.org/works/"
            + urllib.parse.quote("https://doi.org/" + doi, safe="")
        )
        if openalex_params:
            openalex_url += "?" + urllib.parse.urlencode(openalex_params)
        requests.append(self._request("openalex", openalex_url))
        requests.append(self._request(
            "europe_pmc",
            "https://www.ebi.ac.uk/europepmc/webservices/rest/search?"
            + urllib.parse.urlencode({
                "query": f'DOI:"{doi}"', "format": "json",
                "pageSize": 3, "resultType": "core",
            }),
        ))
        if self.semantic_scholar_api_key:
            requests.append(self._request(
                "semantic_scholar",
                "https://api.semanticscholar.org/graph/v1/paper/"
                + urllib.parse.quote("DOI:" + doi, safe=":")
                + "?fields=openAccessPdf,externalIds",
                {"x-api-key": self.semantic_scholar_api_key},
            ))

        pmcid = _clean(paper.pmcid)
        errors: list[str] = []
        for provider, data, error in await asyncio.gather(*requests):
            if error:
                errors.append(error)
                continue
            if not isinstance(data, dict):
                continue
            if provider == "unpaywall":
                locations = [data.get("best_oa_location"), *(data.get("oa_locations") or [])]
                candidates = [
                    _clean(location.get("url_for_pdf"))
                    for location in locations if isinstance(location, dict)
                ]
            elif provider == "openalex":
                locations = [data.get("best_oa_location"), *(data.get("locations") or [])]
                candidates = [
                    _clean(location.get("pdf_url"))
                    for location in locations if isinstance(location, dict)
                ]
                raw_pmcid = _clean((data.get("ids") or {}).get("pmcid"))
                if not pmcid and raw_pmcid:
                    pmcid = raw_pmcid.rstrip("/").rsplit("/", 1)[-1]
            elif provider == "europe_pmc":
                rows = ((data.get("resultList") or {}).get("result") or [])
                candidates = []
                if not pmcid:
                    pmcid = next(
                        (_clean(row.get("pmcid")) for row in rows if _clean(row.get("pmcid"))),
                        "",
                    )
            else:
                pdf = data.get("openAccessPdf") or {}
                candidates = [_clean(pdf.get("url"))] if isinstance(pdf, dict) else []
                if not pmcid:
                    pmcid = _clean((data.get("externalIds") or {}).get("PubMedCentral"))
            for candidate in candidates:
                if candidate and candidate not in urls:
                    urls.append(candidate)

        external_ids = dict(paper.external_ids)
        if urls:
            external_ids["oa_pdf_url"] = urls[0]
        updated = paper.model_copy(update={
            "pmcid": pmcid,
            "external_ids": external_ids,
            "is_open_access": (
                True if pmcid or urls or external_ids.get("arxiv")
                else paper.is_open_access
            ),
            "fulltext_status": (
                FulltextStatus.PDF_AVAILABLE if urls else paper.fulltext_status
            ),
        })
        return AccessEnrichment(updated, tuple(urls), tuple(errors))


class EnrichingFulltextResolver(FulltextResolverProtocol):
    """Try PMCID/arXiv and every verified OA URL before abstract fallback."""

    def __init__(
        self,
        resolver: FulltextResolverProtocol,
        enricher: OpenAccessEnricher,
    ) -> None:
        self.resolver = resolver
        self.enricher = enricher

    async def resolve(self, paper: PaperRecord) -> DocumentRecord:
        enrichment = await self.enricher.enrich(paper)
        variants: list[PaperRecord] = []
        if enrichment.paper.pmcid or enrichment.paper.external_ids.get("arxiv"):
            variants.append(enrichment.paper)
        for url in enrichment.urls:
            external_ids = dict(enrichment.paper.external_ids)
            external_ids["oa_pdf_url"] = url
            variants.append(enrichment.paper.model_copy(update={"external_ids": external_ids}))
        if not variants:
            variants.append(enrichment.paper)

        seen: set[tuple[str, str, str]] = set()
        documents: list[DocumentRecord] = []
        for variant in variants:
            signature = (
                variant.pmcid,
                variant.external_ids.get("arxiv", ""),
                variant.external_ids.get("oa_pdf_url", ""),
            )
            if signature in seen:
                continue
            seen.add(signature)
            document = await self.resolver.resolve(variant)
            documents.append(document)
            if document.content_level in _FULLTEXT_LEVELS and document.local_path:
                return document

        preferred = next(
            (document for document in documents if document.local_path),
            documents[-1],
        )
        combined = "; ".join(dict.fromkeys([
            *enrichment.errors,
            *(document.retrieval_error for document in documents if document.retrieval_error),
        ]))
        return preferred.model_copy(update={
            "retrieval_error": combined or preferred.retrieval_error,
            "retrieval_failure_detail": combined or preferred.retrieval_failure_detail,
        })

    async def resolve_abstract(self, paper: PaperRecord) -> DocumentRecord:
        fallback = getattr(self.resolver, "resolve_abstract", None)
        if not callable(fallback):
            raise RuntimeError("abstract fallback is unavailable")
        return await fallback(paper)
