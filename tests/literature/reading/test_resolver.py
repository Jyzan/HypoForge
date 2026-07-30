from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from hypoforge.literature.models import ContentLevel, PaperRecord
from hypoforge.literature.reading.resolver import PMCFulltextResolver


BIOC_PAYLOAD = [
    {
        "source": "PMC",
        "documents": [
            {
                "id": "PMC123",
                "passages": [
                    {
                        "infons": {"section_type": "ABSTRACT"},
                        "text": "YAP evidence.",
                    },
                    {
                        "infons": {"section_type": "RESULTS"},
                        "text": "Full result.",
                    },
                ],
            }
        ],
    }
]


def paper(**updates) -> PaperRecord:
    values = {
        "paper_id": "PMID:123",
        "title": "Hippo paper",
        "abstract": "Abstract fallback.",
        "pmid": "123",
        "pmcid": "PMC123",
        "sources": ["pubmed"],
    }
    values.update(updates)
    return PaperRecord(**values)


@pytest.mark.asyncio
async def test_resolver_fetches_pmc_fulltext_and_reuses_cache(tmp_path: Path) -> None:
    calls: list[tuple[str, float]] = []

    async def backend(url: str, timeout: float) -> bytes:
        calls.append((url, timeout))
        return json.dumps(BIOC_PAYLOAD).encode()

    resolver = PMCFulltextResolver(
        cache_dir=tmp_path, timeout_seconds=4.5, backend=backend
    )

    first = await resolver.resolve(paper())
    second = await resolver.resolve(paper())

    assert first.content_level is ContentLevel.STRUCTURED_FULLTEXT
    assert first.source_uri.endswith("/PMC123/unicode")
    assert Path(first.local_path).read_bytes()
    assert first.retrieval_error == ""
    assert second == first
    assert len(calls) == 1
    assert calls[0][1] == 4.5


@pytest.mark.asyncio
async def test_resolver_uses_pmid_when_pmcid_is_missing(tmp_path: Path) -> None:
    urls: list[str] = []

    async def backend(url: str, timeout: float) -> bytes:
        urls.append(url)
        return json.dumps(BIOC_PAYLOAD).encode()

    result = await PMCFulltextResolver(tmp_path, backend=backend).resolve(
        paper(pmcid="")
    )

    assert result.content_level is ContentLevel.STRUCTURED_FULLTEXT
    assert urls[0].endswith("/123/unicode")


@pytest.mark.asyncio
async def test_resolver_degrades_to_abstract_on_network_failure(tmp_path: Path) -> None:
    async def backend(url: str, timeout: float) -> bytes:
        raise OSError("network down")

    result = await PMCFulltextResolver(tmp_path, backend=backend).resolve(paper())

    assert result.content_level is ContentLevel.ABSTRACT
    assert json.loads(Path(result.local_path).read_text(encoding="utf-8"))["text"] == (
        "Abstract fallback."
    )
    assert "network down" in result.retrieval_error


@pytest.mark.asyncio
async def test_resolver_reports_no_content_without_fulltext_or_abstract(
    tmp_path: Path,
) -> None:
    async def backend(url: str, timeout: float) -> bytes:
        raise OSError("network down")

    result = await PMCFulltextResolver(tmp_path, backend=backend).resolve(
        paper(abstract="")
    )

    assert result.content_level is ContentLevel.METADATA
    assert result.local_path == ""
    assert "network down" in result.retrieval_error
    assert "no readable content" in result.retrieval_error


@pytest.mark.asyncio
async def test_resolver_rejects_payload_without_nonempty_passages(tmp_path: Path) -> None:
    async def backend(url: str, timeout: float) -> bytes:
        return b'{"documents": [{"passages": [{"text": ""}]}]}'

    result = await PMCFulltextResolver(tmp_path, backend=backend).resolve(paper())

    assert result.content_level is ContentLevel.ABSTRACT
    assert "no non-empty passages" in result.retrieval_error


@pytest.mark.asyncio
async def test_resolver_labels_bioc_not_found_response_as_fulltext_unavailable(
    tmp_path: Path,
) -> None:
    async def backend(url: str, timeout: float) -> bytes:
        return b"[Error] : No result can be found."

    result = await PMCFulltextResolver(tmp_path, backend=backend).resolve(paper())

    assert result.content_level is ContentLevel.ABSTRACT
    assert "PMC Open Access full text unavailable" in result.retrieval_error
    assert "JSONDecodeError" not in result.retrieval_error


@pytest.mark.asyncio
async def test_resolver_does_not_swallow_cancellation(tmp_path: Path) -> None:
    async def backend(url: str, timeout: float) -> bytes:
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await PMCFulltextResolver(tmp_path, backend=backend).resolve(paper())
