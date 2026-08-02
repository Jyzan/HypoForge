from __future__ import annotations

import json
from pathlib import Path

import pytest

from hypoforge.literature.models import ContentLevel, DocumentRecord
from hypoforge.literature.reading.parser import BioCDocumentParser, DocumentParseError


def document(path: Path, level: ContentLevel) -> DocumentRecord:
    return DocumentRecord(
        document_id="doc:1",
        paper_id="PMID:1",
        content_level=level,
        local_path=str(path),
    )


def write_bioc(path: Path, passages: list[dict]) -> None:
    path.write_text(
        json.dumps(
            [{"source": "PMC", "documents": [{"id": "1", "passages": passages}]}]
        ),
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_parser_reads_real_bioc_collection_shape_and_sections(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bioc.json"
    write_bioc(
        path,
        [
            {
                "offset": 0,
                "infons": {"section_type": "INTRO", "type": "title_1"},
                "text": "Hippo Signaling in Organ Size Control",
            },
            {"offset": 0, "infons": {"section_type": "ABSTRACT"}, "text": "A"},
            {"offset": 10, "infons": {"section_type": "METHODS"}, "text": "M"},
            {"offset": 20, "infons": {"section_type": "RESULTS"}, "text": "R"},
            {"offset": 30, "infons": {"section_type": "DISCUSS"}, "text": "D"},
            {
                "offset": 40,
                "infons": {"section_type": "LIMITATIONS"},
                "text": "L",
            },
            {
                "offset": 50,
                "infons": {"section_type": "REF", "type": "ref"},
                "text": "Highly relevant cited paper title about YAP and TAZ",
            },
        ],
    )
    parser = BioCDocumentParser(target_chars=100, overlap_chars=20)

    first = await parser.parse(document(path, ContentLevel.STRUCTURED_FULLTEXT))
    second = await parser.parse(document(path, ContentLevel.STRUCTURED_FULLTEXT))

    assert {chunk.section for chunk in first} == {
        "abstract",
        "methods",
        "results",
        "discussion",
        "limitations",
    }
    assert first == second
    assert all(chunk.end_offset >= chunk.start_offset for chunk in first)
    assert all("Organ Size Control" not in chunk.text for chunk in first)
    assert all("cited paper title" not in chunk.text for chunk in first)


@pytest.mark.asyncio
async def test_parser_splits_long_passage_with_bounded_overlap(tmp_path: Path) -> None:
    path = tmp_path / "bioc.json"
    text = " ".join(f"Sentence {index} contains YAP evidence." for index in range(20))
    write_bioc(
        path,
        [{"offset": 100, "infons": {"section_type": "RESULTS"}, "text": text}],
    )

    chunks = await BioCDocumentParser(
        target_chars=120, overlap_chars=25
    ).parse(document(path, ContentLevel.STRUCTURED_FULLTEXT))

    assert len(chunks) > 2
    assert all(chunk.section == "results" for chunk in chunks)
    assert all(len(chunk.text) <= 150 for chunk in chunks)
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)


@pytest.mark.asyncio
async def test_parser_aligns_ascii_overlap_to_word_boundary(tmp_path: Path) -> None:
    path = tmp_path / "word-boundary.json"
    text = (
        "A " * 25
        + "mitogen-activated protein kinase controls Hippo signaling "
        + "tail " * 20
    )
    write_bioc(
        path,
        [{"offset": 0, "infons": {"section_type": "RESULTS"}, "text": text}],
    )

    chunks = await BioCDocumentParser(
        target_chars=80, overlap_chars=20
    ).parse(document(path, ContentLevel.STRUCTURED_FULLTEXT))

    def ascii_word(character: str) -> bool:
        return character.isascii() and (character.isalnum() or character in "_-")

    for chunk in chunks[1:]:
        start = chunk.start_offset or 0
        assert not (ascii_word(text[start - 1]) and ascii_word(text[start]))


@pytest.mark.asyncio
async def test_parser_reads_abstract_fallback_document(tmp_path: Path) -> None:
    path = tmp_path / "abstract.json"
    path.write_text(
        json.dumps({"format": "hypoforge_abstract_v1", "text": "Abstract evidence."}),
        encoding="utf-8",
    )

    chunks = await BioCDocumentParser().parse(document(path, ContentLevel.ABSTRACT))

    assert len(chunks) == 1
    assert chunks[0].section == "abstract"
    assert chunks[0].text == "Abstract evidence."


@pytest.mark.asyncio
async def test_parser_rejects_malformed_or_empty_documents(tmp_path: Path) -> None:
    malformed = tmp_path / "bad.json"
    malformed.write_text("{bad", encoding="utf-8")
    empty = tmp_path / "empty.json"
    write_bioc(empty, [{"infons": {"section_type": "RESULTS"}, "text": ""}])

    parser = BioCDocumentParser()
    with pytest.raises(DocumentParseError, match="doc:1"):
        await parser.parse(document(malformed, ContentLevel.STRUCTURED_FULLTEXT))
    with pytest.raises(DocumentParseError, match="no readable passages"):
        await parser.parse(document(empty, ContentLevel.STRUCTURED_FULLTEXT))
