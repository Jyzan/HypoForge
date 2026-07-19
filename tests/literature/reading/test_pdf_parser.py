from __future__ import annotations

from pathlib import Path

import pytest

from hypoforge.literature.models import ContentLevel, DocumentChunk, DocumentRecord
from hypoforge.literature.reading import parser as parser_module
from hypoforge.literature.reading.parser import (
    DocumentParseError,
    PDFDocumentParser,
)
from hypoforge.literature.reading.routing import RoutingDocumentParser


def pdf_document(path: Path) -> DocumentRecord:
    return DocumentRecord(
        document_id="document:arxiv:pdf",
        paper_id="ARXIV:2401.12345",
        content_level=ContentLevel.PDF,
        source_uri="https://arxiv.org/pdf/2401.12345",
        local_path=str(path),
    )


class FakePage:
    def __init__(self, text: str | None) -> None:
        self.text = text

    def extract_text(self) -> str | None:
        return self.text


class FakeReader:
    def __init__(self, texts: list[str | None]) -> None:
        self.pages = [FakePage(text) for text in texts]


@pytest.mark.asyncio
async def test_pdf_parser_emits_page_aware_chunks(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        parser_module,
        "PdfReader",
        lambda path: FakeReader(
            ["First page evidence.", "Second page evidence."]
        ),
    )
    path = tmp_path / "paper.pdf"
    path.write_bytes(b"%PDF-1.4\nfixture")

    chunks = await PDFDocumentParser(
        target_chars=100,
        overlap_chars=20,
    ).parse(pdf_document(path))

    assert [chunk.page for chunk in chunks] == [1, 2]
    assert [chunk.section for chunk in chunks] == ["page_1", "page_2"]
    assert [chunk.text for chunk in chunks] == [
        "First page evidence.",
        "Second page evidence.",
    ]
    assert [(chunk.start_offset, chunk.end_offset) for chunk in chunks] == [
        (0, 20),
        (0, 21),
    ]
    assert len({chunk.chunk_id for chunk in chunks}) == 2


@pytest.mark.asyncio
async def test_pdf_parser_windows_long_pages_without_losing_page_number(
    monkeypatch,
    tmp_path: Path,
) -> None:
    text = "Sentence of page evidence. " * 12
    monkeypatch.setattr(
        parser_module,
        "PdfReader",
        lambda path: FakeReader([text]),
    )
    path = tmp_path / "long.pdf"
    path.write_bytes(b"%PDF-1.4\nfixture")

    chunks = await PDFDocumentParser(
        target_chars=100,
        overlap_chars=20,
    ).parse(pdf_document(path))

    assert len(chunks) > 1
    assert {chunk.page for chunk in chunks} == {1}
    assert all(chunk.start_offset < chunk.end_offset for chunk in chunks)


@pytest.mark.asyncio
async def test_pdf_parser_rejects_empty_extraction(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        parser_module,
        "PdfReader",
        lambda path: FakeReader(["", "  ", None]),
    )
    path = tmp_path / "empty.pdf"
    path.write_bytes(b"%PDF-1.4\nfixture")

    with pytest.raises(DocumentParseError, match="no readable PDF pages"):
        await PDFDocumentParser().parse(pdf_document(path))


@pytest.mark.asyncio
async def test_pdf_parser_reports_reader_failure(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def fail(path):
        raise ValueError("broken cross-reference table")

    monkeypatch.setattr(parser_module, "PdfReader", fail)
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"%PDF-broken")

    with pytest.raises(DocumentParseError, match="broken cross-reference table"):
        await PDFDocumentParser().parse(pdf_document(path))


class FakeParser:
    def __init__(self, label: str) -> None:
        self.label = label
        self.calls: list[ContentLevel] = []

    async def parse(self, document: DocumentRecord) -> list[DocumentChunk]:
        self.calls.append(document.content_level)
        return [
            DocumentChunk(
                chunk_id=f"chunk:{self.label}",
                document_id=document.document_id,
                paper_id=document.paper_id,
                text=self.label,
            )
        ]


@pytest.mark.asyncio
async def test_routing_parser_uses_pdf_only_for_pdf_content(tmp_path: Path) -> None:
    structured = FakeParser("structured")
    pdf = FakeParser("pdf")
    router = RoutingDocumentParser(structured, pdf)
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    abstract = DocumentRecord(
        document_id="doc:abstract",
        paper_id="PMID:1",
        content_level=ContentLevel.ABSTRACT,
        local_path=str(tmp_path / "abstract.json"),
    )

    assert (await router.parse(pdf_document(pdf_path)))[0].text == "pdf"
    assert (await router.parse(abstract))[0].text == "structured"
    assert pdf.calls == [ContentLevel.PDF]
    assert structured.calls == [ContentLevel.ABSTRACT]
