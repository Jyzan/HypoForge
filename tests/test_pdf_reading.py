"""PDF font decoding, bounded downloads, cache recovery, and honest fallback."""

import asyncio
import http.client
import json
import sys
import time
import urllib.error
from io import BytesIO
from pathlib import Path

import pytest

from hypoforge.benchmarks import agentideabench as bench
from hypoforge.benchmarks import agentideabench_scoring as scoring
from hypoforge.config import PipelineConfig
from hypoforge.modules.m2_literature.models import (
    ContentLevel, DocumentRecord, PaperReadingResult, PaperRecord,
)
from hypoforge.modules.m2_literature.reading.arxiv_resolver import (
    ArxivPDFResolver, _read_bounded_response,
)
from hypoforge.modules.m2_literature.reading.parser import (
    BioCDocumentParser, DocumentParseError, PDFDocumentParser,
)
from hypoforge.modules.m2_literature.reading.pdf_validation import (
    PDFIntegrityError, validate_pdf_stream,
)
from hypoforge.modules.m2_literature.reading.retriever import HybridEvidenceRetriever
from hypoforge.modules.m2_literature.reading.routing import RoutingDocumentParser
from hypoforge.modules.m2_literature.reading.store import InMemoryChunkStore
from hypoforge.modules.m2_literature.reading.workflow import FullTextReadingWorkflow
from hypoforge.tools.qwen_client import QwenClient


FIXTURE = Path(bench.__file__).parent / "fixtures/cff_font_check.pdf"


@pytest.fixture
def pdf_bytes():
    return FIXTURE.read_bytes()


@pytest.fixture
def paper():
    return PaperRecord(
        paper_id="ARXIV:2601.00001", title="PDF test", sources=["arxiv"],
        abstract="A real abstract describing graph neural network evidence.",
    )


@pytest.mark.asyncio
async def test_real_cff_font_decoding_passes_preflight(caplog, capsys):
    # This PDF maps the byte for A to glyph B in its embedded CFF font.
    # Merely importing fonttools is insufficient to produce the right text.
    await bench.check_pdf_environment()
    assert "PDF CFF font decoding / text extraction: OK" in capsys.readouterr().out
    assert "fontTools is required" not in caplog.text


@pytest.mark.asyncio
async def test_missing_fonttools_fails_before_any_model_call(monkeypatch):
    monkeypatch.setitem(sys.modules, "fontTools.cffLib", None)

    def no_model_calls(*args, **kwargs):
        pytest.fail("Model calls must not precede PDF environment validation")

    monkeypatch.setattr(QwenClient, "from_config", no_model_calls)
    with pytest.raises(ValueError, match="requires fonttools"):
        await scoring.preflight(PipelineConfig())


@pytest.mark.asyncio
async def test_old_font_decoder_is_detected_before_generation(tmp_path, monkeypatch):
    from pypdf import _font
    monkeypatch.setattr(_font, "HAS_FONTTOOLS", False)
    with pytest.raises(ValueError, match="CFF PDF font decoding check failed"):
        await bench.generate(tmp_path, tmp_path, PipelineConfig(), [], 1)
    assert not (tmp_path / "manifest.json").exists()


@pytest.mark.parametrize("suffix", [b"", b"\n", b"\r\n", b"\ntrailing publisher comment\n"])
def test_valid_pdf_footer_allows_trailing_material(pdf_bytes, suffix):
    validate_pdf_stream(BytesIO(pdf_bytes + suffix))


@pytest.mark.parametrize("payload", [
    b"<html>publisher login</html>",
    b"%PDF-1.7\ntruncated body",
    b"%PDF-1.7\n%%EOFX\n",
    b"%PDF-1.7\n%%EOF\nstartxref\n456\n",
])
def test_invalid_pdf_envelopes_are_rejected(payload):
    with pytest.raises(PDFIntegrityError):
        validate_pdf_stream(BytesIO(payload))


@pytest.mark.asyncio
async def test_parser_rejects_truncation_before_pypdf_warnings(tmp_path, pdf_bytes, caplog):
    path = tmp_path / "truncated.pdf"
    path.write_bytes(pdf_bytes.rsplit(b"%%EOF", 1)[0])
    with pytest.raises(DocumentParseError, match="%%EOF marker missing"):
        await PDFDocumentParser().parse(DocumentRecord(
            document_id="bad", paper_id="bad", content_level=ContentLevel.PDF,
            local_path=str(path),
        ))
    assert "EOF marker not found" not in caplog.text


class HTTPResponse(BytesIO):
    def __init__(self, payload, length):
        super().__init__(payload)
        self.headers = {"Content-Length": str(length)}


def test_http_content_length_mismatch_detects_truncated_transfer(pdf_bytes):
    with pytest.raises(PDFIntegrityError, match="Incomplete PDF download"):
        _read_bounded_response(
            HTTPResponse(pdf_bytes, len(pdf_bytes) + 100),
            max_bytes=10_000, deadline=60.0, clock=lambda: 0.0,
        )
    assert _read_bounded_response(
        HTTPResponse(pdf_bytes, len(pdf_bytes)),
        max_bytes=10_000, deadline=60.0, clock=lambda: 0.0,
    ) == pdf_bytes


@pytest.mark.asyncio
@pytest.mark.parametrize("first_failure", ["missing_eof", "incomplete_http"])
async def test_incomplete_download_is_retried_and_only_complete_pdf_is_cached(
    tmp_path, pdf_bytes, paper, first_failure,
):
    calls = []

    async def fetch(url, timeout):
        calls.append(url)
        assert not (resolver._paper_dir(paper) / "paper.pdf").exists()
        if len(calls) == 1:
            if first_failure == "incomplete_http":
                raise http.client.IncompleteRead(b"%PDF-1.7\n", 200)
            return pdf_bytes.rsplit(b"%%EOF", 1)[0]
        return pdf_bytes

    resolver = ArxivPDFResolver(tmp_path, backend=fetch)
    document = await resolver.resolve(paper)
    assert document.content_level is ContentLevel.PDF
    assert Path(document.local_path).read_bytes() == pdf_bytes
    assert (await PDFDocumentParser().parse(document))[0].text == "B"
    assert (await resolver.resolve(paper)).content_level is ContentLevel.PDF
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_truncated_cached_pdf_is_preserved_and_replaced(tmp_path, pdf_bytes, paper):
    calls = []

    async def fetch(url, timeout):
        calls.append(url)
        return pdf_bytes

    resolver = ArxivPDFResolver(tmp_path, backend=fetch)
    path = resolver._paper_dir(paper) / "paper.pdf"
    path.parent.mkdir(parents=True)
    truncated = pdf_bytes.rsplit(b"%%EOF", 1)[0]
    path.write_bytes(truncated)
    document = await resolver.resolve(paper)
    assert document.content_level is ContentLevel.PDF
    assert path.read_bytes() == pdf_bytes
    invalid = list(path.parent.glob("paper.invalid-*.pdf"))
    assert len(invalid) == 1 and invalid[0].read_bytes() == truncated
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_persistently_incomplete_pdf_records_abstract_degradation(tmp_path, paper):
    calls = []

    async def fetch(url, timeout):
        calls.append(url)
        return b"%PDF-1.7\ntruncated"

    class Reader:
        async def read(self, question, value, evidence):
            assert evidence and evidence[0].section == "abstract"
            return PaperReadingResult(paper_id=value.paper_id, evidence=evidence)

    store = InMemoryChunkStore()
    resolver = ArxivPDFResolver(tmp_path, backend=fetch)
    workflow = FullTextReadingWorkflow(
        resolver=resolver,
        parser=RoutingDocumentParser(BioCDocumentParser(), PDFDocumentParser()),
        retriever=HybridEvidenceRetriever(store), reader=Reader(), store=store,
    )
    result, = await workflow.run("graph neural network evidence", [paper])
    assert len(calls) == 2
    assert result.degraded_to_abstract and result.content_level is ContentLevel.ABSTRACT
    assert result.fulltext_failure_category == "other"
    assert "integrity failure" in result.fulltext_failure_detail
    assert "%%EOF marker missing" in " ".join(result.errors)
    assert result.chunks_parsed > 0 and result.evidence
    assert not (resolver._paper_dir(paper) / "paper.pdf").exists()
    abstract_path = resolver._paper_dir(paper) / "abstract.json"
    assert json.loads(abstract_path.read_text())["text"] == paper.abstract


@pytest.mark.asyncio
async def test_default_http_transport_retries_incomplete_transfer(
    tmp_path, pdf_bytes, paper, monkeypatch,
):
    requests = []

    def open_response(request, timeout):
        requests.append(request.full_url)
        return HTTPResponse(
            pdf_bytes, len(pdf_bytes) + (10 if len(requests) == 1 else 0),
        )

    monkeypatch.setattr(urllib.request, "urlopen", open_response)
    resolver = ArxivPDFResolver(tmp_path, download_timeout_seconds=1.0)
    document = await resolver.resolve(paper)
    assert document.content_level is ContentLevel.PDF
    assert len(requests) == 2
    assert Path(document.local_path).read_bytes() == pdf_bytes


@pytest.mark.asyncio
async def test_incomplete_pdf_without_abstract_retains_failure(tmp_path, paper):
    async def fetch(url, timeout):
        return b"%PDF-1.7\ntruncated"

    resolver = ArxivPDFResolver(tmp_path, backend=fetch)
    document = await resolver.resolve(paper.model_copy(update={"abstract": ""}))
    assert document.content_level is ContentLevel.METADATA
    assert "%%EOF marker missing" in document.retrieval_error
    assert document.retrieval_failure_category == "other"
    assert not document.local_path


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["html", "http429"])
async def test_invalid_links_and_rate_limits_do_not_trigger_pdf_retry(tmp_path, paper, kind):
    calls = []

    async def fetch(url, timeout):
        calls.append(url)
        if kind == "http429":
            raise urllib.error.HTTPError(url, 429, "Too Many Requests", {}, None)
        return b"<html>publisher login</html>"

    document = await ArxivPDFResolver(tmp_path, backend=fetch).resolve(paper)
    assert document.content_level is ContentLevel.ABSTRACT
    assert document.retrieval_failure_category == (
        "publisher_blocked" if kind == "http429" else "invalid_pdf_link"
    )
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_pdf_retry_shares_original_download_deadline(tmp_path, paper):
    calls = []

    async def fetch(url, timeout):
        calls.append(timeout)
        await asyncio.sleep(0.02 if len(calls) == 1 else 1.0)
        return b"%PDF-1.7\ntruncated"

    started = time.monotonic()
    document = await ArxivPDFResolver(
        tmp_path, backend=fetch, download_timeout_seconds=0.05,
    ).resolve(paper)
    assert len(calls) == 2 and calls[1] < calls[0]
    assert time.monotonic() - started < 0.3
    assert document.content_level is ContentLevel.ABSTRACT
    assert "Download timed out" in document.retrieval_failure_detail


@pytest.mark.asyncio
async def test_cancellation_does_not_turn_into_abstract_fallback(tmp_path, paper):
    started = asyncio.Event()

    async def fetch(url, timeout):
        started.set()
        await asyncio.Event().wait()

    resolver = ArxivPDFResolver(tmp_path, backend=fetch)
    task = asyncio.create_task(resolver.resolve(paper))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not list(tmp_path.rglob("*.pdf"))
    assert not list(tmp_path.rglob("abstract.json"))
