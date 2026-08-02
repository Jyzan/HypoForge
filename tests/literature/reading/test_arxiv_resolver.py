from __future__ import annotations

import asyncio
import json
import socket
import time
from pathlib import Path

import pytest

from hypoforge.literature.reading import arxiv_resolver as resolver_module
from hypoforge.literature.models import (
    ContentLevel,
    DocumentRecord,
    FulltextStatus,
    PaperRecord,
)
from hypoforge.literature.reading.arxiv_resolver import (
    ArxivDownloadTimeoutError,
    ArxivPDFResolver,
    _read_bounded_response,
)
from hypoforge.literature.reading.routing import RoutingFulltextResolver


def arxiv_paper(
    *,
    paper_id: str = "ARXIV:2401.12345",
    arxiv_id: str = "2401.12345v2",
    abstract: str = "Useful abstract evidence.",
) -> PaperRecord:
    external_ids = {"arxiv": arxiv_id} if arxiv_id else {}
    return PaperRecord(
        paper_id=paper_id,
        title="A useful preprint",
        abstract=abstract,
        external_ids=external_ids,
        sources=["arxiv"],
        is_open_access=True,
        fulltext_status=FulltextStatus.PDF_AVAILABLE,
    )


class FakeResponse:
    def __init__(
        self,
        chunks: list[bytes],
        headers: dict[str, str] | None = None,
    ) -> None:
        self.chunks = list(chunks)
        self.headers = headers or {}
        self.read_calls = 0

    def read(self, size: int) -> bytes:
        self.read_calls += 1
        return self.chunks.pop(0) if self.chunks else b""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None


class FakeClock:
    def __init__(self, values: list[float]) -> None:
        self.values = iter(values)

    def __call__(self) -> float:
        return next(self.values)


def test_stream_reader_enforces_total_deadline_between_chunks() -> None:
    response = FakeResponse([b"%PDF-", b"a" * 10, b"b" * 10])

    with pytest.raises(ArxivDownloadTimeoutError, match="180"):
        _read_bounded_response(
            response,
            max_bytes=1_000,
            deadline=180.0,
            clock=FakeClock([0.0, 10.0, 40.0, 50.0, 100.0, 181.0]),
        )

    assert response.read_calls == 3


def test_stream_reader_rejects_content_length_before_reading() -> None:
    response = FakeResponse([], headers={"Content-Length": "1001"})

    with pytest.raises(ValueError, match="size limit"):
        _read_bounded_response(
            response,
            max_bytes=1_000,
            deadline=180.0,
            clock=lambda: 0.0,
        )

    assert response.read_calls == 0


def test_stream_reader_rejects_accumulated_size() -> None:
    response = FakeResponse([b"%PDF-", b"x" * 20])

    with pytest.raises(ValueError, match="size limit"):
        _read_bounded_response(
            response,
            max_bytes=10,
            deadline=180.0,
            clock=lambda: 0.0,
        )


def test_stream_reader_shortens_blocking_read_to_remaining_deadline() -> None:
    reader, writer = socket.socketpair()
    reader.settimeout(0.1)

    class BlockingResponse:
        headers: dict[str, str] = {}

        def settimeout(self, seconds: float) -> None:
            reader.settimeout(seconds)

        def read(self, size: int) -> bytes:
            return reader.recv(size)

    started = time.monotonic()
    try:
        with pytest.raises(ArxivDownloadTimeoutError, match="download deadline"):
            _read_bounded_response(
                BlockingResponse(),
                max_bytes=1_000,
                deadline=started + 0.02,
                clock=time.monotonic,
                timeout_seconds=0.02,
            )
    finally:
        reader.close()
        writer.close()

    assert time.monotonic() - started < 0.06


@pytest.mark.asyncio
async def test_default_download_timeout_degrades_without_partial_pdf(
    monkeypatch,
    tmp_path: Path,
) -> None:
    response = FakeResponse([b"%PDF-partial"])
    clock = FakeClock([0.0, 0.0, 2.0])
    monkeypatch.setattr(resolver_module, "_monotonic", clock)
    monkeypatch.setattr(
        resolver_module.urllib.request,
        "urlopen",
        lambda request, timeout: response,
    )
    resolver = ArxivPDFResolver(
        tmp_path,
        timeout_seconds=0.5,
        download_timeout_seconds=1.0,
    )

    document = await resolver.resolve(arxiv_paper())

    assert document.content_level is ContentLevel.ABSTRACT
    assert "download deadline" in document.retrieval_error
    assert not list(tmp_path.rglob("*.tmp"))
    assert not list(tmp_path.rglob("paper.pdf"))


@pytest.mark.asyncio
async def test_default_fetch_has_outer_deadline_even_if_reader_blocks(
    monkeypatch,
    tmp_path: Path,
) -> None:
    async def slow_default_fetch(*args) -> bytes:
        await asyncio.sleep(0.1)
        return b"%PDF-1.4\nlate payload"

    monkeypatch.setattr(resolver_module, "_default_fetch", slow_default_fetch)
    resolver = ArxivPDFResolver(
        tmp_path,
        download_timeout_seconds=0.01,
    )

    started = time.monotonic()
    document = await resolver.resolve(arxiv_paper())
    elapsed = time.monotonic() - started

    assert elapsed < 0.05
    assert document.content_level is ContentLevel.ABSTRACT
    assert "download deadline" in document.retrieval_error
    assert not list(tmp_path.rglob("paper.pdf"))


@pytest.mark.asyncio
async def test_arxiv_resolver_downloads_valid_pdf_once_and_reuses_cache(
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, float]] = []

    async def fetch(url: str, timeout: float) -> bytes:
        calls.append((url, timeout))
        return b"%PDF-1.4\nvalid-test-payload"

    resolver = ArxivPDFResolver(tmp_path, backend=fetch)

    first = await resolver.resolve(arxiv_paper())
    second = await resolver.resolve(arxiv_paper())

    assert first.content_level is ContentLevel.PDF
    assert first.local_path == second.local_path
    assert Path(first.local_path).read_bytes().startswith(b"%PDF-")
    assert first.source_uri == "https://arxiv.org/pdf/2401.12345v2"
    assert first.license == ""
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_arxiv_resolver_preserves_legacy_identifier_slash(
    tmp_path: Path,
) -> None:
    urls: list[str] = []

    async def fetch(url: str, timeout: float) -> bytes:
        urls.append(url)
        return b"%PDF-1.4\nlegacy"

    resolver = ArxivPDFResolver(tmp_path, backend=fetch)
    await resolver.resolve(
        arxiv_paper(
            paper_id="ARXIV:hep-th/9901001",
            arxiv_id="hep-th/9901001v3",
        )
    )

    assert urls == ["https://arxiv.org/pdf/hep-th/9901001v3"]


@pytest.mark.asyncio
async def test_arxiv_resolver_invalid_pdf_degrades_to_abstract(
    tmp_path: Path,
) -> None:
    async def fetch(url: str, timeout: float) -> bytes:
        return b"<html>rate limited</html>"

    document = await ArxivPDFResolver(tmp_path, backend=fetch).resolve(
        arxiv_paper()
    )

    assert document.content_level is ContentLevel.ABSTRACT
    assert "invalid PDF" in document.retrieval_error
    payload = json.loads(Path(document.local_path).read_text(encoding="utf-8"))
    assert payload == {
        "format": "hypoforge_abstract_v1",
        "text": "Useful abstract evidence.",
    }


@pytest.mark.asyncio
async def test_arxiv_resolver_enforces_size_limit_and_reports_no_abstract(
    tmp_path: Path,
) -> None:
    async def fetch(url: str, timeout: float) -> bytes:
        return b"%PDF-" + (b"x" * 100)

    document = await ArxivPDFResolver(
        tmp_path,
        max_pdf_bytes=20,
        backend=fetch,
    ).resolve(arxiv_paper(abstract=""))

    assert document.content_level is ContentLevel.METADATA
    assert document.local_path == ""
    assert "size limit" in document.retrieval_error
    assert "no readable content" in document.retrieval_error


@pytest.mark.asyncio
async def test_arxiv_resolver_missing_identifier_degrades_without_network(
    tmp_path: Path,
) -> None:
    calls = 0

    async def fetch(url: str, timeout: float) -> bytes:
        nonlocal calls
        calls += 1
        return b"%PDF-1.4"

    document = await ArxivPDFResolver(tmp_path, backend=fetch).resolve(
        arxiv_paper(
            paper_id="DOI:10.1000/missing-arxiv-id",
            arxiv_id="",
        )
    )

    assert document.content_level is ContentLevel.ABSTRACT
    assert "identifier unavailable" in document.retrieval_error
    assert calls == 0


@pytest.mark.asyncio
async def test_arxiv_resolver_propagates_cancellation(tmp_path: Path) -> None:
    async def fetch(url: str, timeout: float) -> bytes:
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await ArxivPDFResolver(tmp_path, backend=fetch).resolve(arxiv_paper())


@pytest.mark.asyncio
async def test_arxiv_resolver_can_explicitly_materialize_abstract_fallback(
    tmp_path: Path,
) -> None:
    document = await ArxivPDFResolver(tmp_path).resolve_abstract(arxiv_paper())

    assert document.content_level is ContentLevel.ABSTRACT
    assert "full text unavailable" in document.retrieval_error
    assert Path(document.local_path).is_file()


class FakeResolver:
    def __init__(self, label: str) -> None:
        self.label = label
        self.calls: list[str] = []

    async def resolve(self, paper: PaperRecord) -> DocumentRecord:
        self.calls.append(paper.paper_id)
        return DocumentRecord(
            document_id=f"doc:{self.label}",
            paper_id=paper.paper_id,
            content_level=ContentLevel.METADATA,
        )

    async def resolve_abstract(self, paper: PaperRecord) -> DocumentRecord:
        self.calls.append(f"abstract:{paper.paper_id}")
        return DocumentRecord(
            document_id=f"doc:{self.label}:abstract",
            paper_id=paper.paper_id,
            content_level=ContentLevel.ABSTRACT,
            local_path="abstract.json",
        )


@pytest.mark.asyncio
async def test_routing_resolver_chooses_arxiv_by_source_or_external_id() -> None:
    pmc = FakeResolver("pmc")
    arxiv = FakeResolver("arxiv")
    resolver = RoutingFulltextResolver(pmc, arxiv)
    pubmed = PaperRecord(
        paper_id="PMID:1",
        title="Published paper",
        sources=["pubmed"],
    )
    cross_source = PaperRecord(
        paper_id="DOI:10.1/example",
        title="Preprint found elsewhere",
        sources=["semantic_scholar"],
        external_ids={"arxiv": "2401.99999"},
    )

    assert (await resolver.resolve(arxiv_paper())).document_id == "doc:arxiv"
    assert (await resolver.resolve(cross_source)).document_id == "doc:arxiv"
    assert (await resolver.resolve(pubmed)).document_id == "doc:pmc"
    assert arxiv.calls == ["ARXIV:2401.12345", "DOI:10.1/example"]
    assert pmc.calls == ["PMID:1"]

    fallback = await resolver.resolve_abstract(arxiv_paper())
    assert fallback.document_id == "doc:arxiv:abstract"
    assert arxiv.calls[-1] == "abstract:ARXIV:2401.12345"

    pubmed_fallback = await resolver.resolve_abstract(pubmed)
    assert pubmed_fallback.document_id == "doc:pmc:abstract"
    assert pmc.calls[-1] == "abstract:PMID:1"
