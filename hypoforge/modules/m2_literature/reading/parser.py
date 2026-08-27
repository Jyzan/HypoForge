"""Parse cached BioC or abstract documents into stable, section-aware chunks."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from pathlib import Path

from pypdf import PdfReader

from ..models import ContentLevel, DocumentChunk, DocumentRecord
from ..protocols import DocumentParserProtocol


class DocumentParseError(ValueError):
    """Raised when a cached document cannot yield attributable text chunks."""


_SECTION_ALIASES = {
    "abstract": "abstract",
    "intro": "introduction",
    "background": "introduction",
    "method": "methods",
    "material": "methods",
    "result": "results",
    "discuss": "discussion",
    "limitation": "limitations",
    "concl": "conclusion",
}


def _section_name(infons: object) -> str:
    values: list[str] = []
    if isinstance(infons, dict):
        for key in ("section_type", "section", "type"):
            value = infons.get(key)
            if value:
                values.append(str(value).casefold())
    raw = " ".join(values)
    for fragment, normalized in _SECTION_ALIASES.items():
        if fragment in raw:
            return normalized
    return "other"


def _is_ignored_passage(infons: object) -> bool:
    if not isinstance(infons, dict):
        return False
    passage_type = str(infons.get("type") or "").casefold()
    section_type = str(infons.get("section_type") or "").casefold()
    return (
        passage_type == "title"
        or passage_type.startswith("title_")
        or passage_type in {"ref", "reference"}
        or section_type in {"ref", "reference", "references"}
    )


def _is_ascii_word_character(character: str) -> bool:
    return character.isascii() and (
        character.isalnum() or character in "_-"
    )


def _bioc_documents(payload: object) -> list[dict]:
    if isinstance(payload, list):
        output: list[dict] = []
        for item in payload:
            output.extend(_bioc_documents(item))
        return output
    if isinstance(payload, dict):
        if isinstance(payload.get("passages"), list):
            return [payload]
        nested = payload.get("documents")
        if isinstance(nested, list):
            output: list[dict] = []
            for item in nested:
                output.extend(_bioc_documents(item))
            return output
    return []


def _window_text(text: str, target: int, overlap: int) -> list[tuple[int, int, str]]:
    stripped_start = len(text) - len(text.lstrip())
    stripped_end = len(text.rstrip())
    if stripped_end <= stripped_start:
        return []
    if stripped_end - stripped_start <= target:
        return [(stripped_start, stripped_end, text[stripped_start:stripped_end])]

    windows: list[tuple[int, int, str]] = []
    start = stripped_start
    while start < stripped_end:
        hard_end = min(start + target, stripped_end)
        end = hard_end
        if hard_end < stripped_end:
            lower = start + max(target // 2, 1)
            segment = text[lower:hard_end]
            sentence_ends = [match.end() for match in re.finditer(r"[.!?。！？]\s+", segment)]
            if sentence_ends:
                end = lower + sentence_ends[-1]
            else:
                space = text.rfind(" ", lower, hard_end)
                if space > start:
                    end = space
        chunk_start = start
        chunk_end = max(end, start + 1)
        chunk = text[chunk_start:chunk_end].strip()
        leading = len(text[chunk_start:chunk_end]) - len(text[chunk_start:chunk_end].lstrip())
        trailing = len(text[chunk_start:chunk_end].rstrip())
        actual_start = chunk_start + leading
        actual_end = chunk_start + trailing
        if chunk:
            windows.append((actual_start, actual_end, chunk))
        if chunk_end >= stripped_end:
            break
        next_start = max(chunk_end - overlap, start + 1)
        while next_start < chunk_end and text[next_start].isspace():
            next_start += 1
        if (
            0 < next_start < chunk_end
            and _is_ascii_word_character(text[next_start - 1])
            and _is_ascii_word_character(text[next_start])
        ):
            while (
                next_start < chunk_end
                and _is_ascii_word_character(text[next_start])
            ):
                next_start += 1
            while next_start < chunk_end and text[next_start].isspace():
                next_start += 1
        start = next_start
    return windows


def _chunk_id(
    document: DocumentRecord,
    section: str,
    ordinal: int,
    text: str,
) -> str:
    raw = f"{document.document_id}\0{section}\0{ordinal}\0{text}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"{document.document_id}:chunk:{digest}"


class BioCDocumentParser(DocumentParserProtocol):
    def __init__(self, target_chars: int = 900, overlap_chars: int = 120) -> None:
        if target_chars < 50:
            raise ValueError("target_chars must be at least 50")
        if overlap_chars < 0 or overlap_chars >= target_chars:
            raise ValueError("overlap_chars must be non-negative and below target_chars")
        self.target_chars = target_chars
        self.overlap_chars = overlap_chars

    @staticmethod
    def _chunk_id(document: DocumentRecord, section: str, ordinal: int, text: str) -> str:
        return _chunk_id(document, section, ordinal, text)

    async def parse(self, document: DocumentRecord) -> list[DocumentChunk]:
        if not document.local_path:
            raise DocumentParseError(f"{document.document_id}: local document unavailable")
        path = Path(document.local_path)
        try:
            raw = await asyncio.to_thread(path.read_text, encoding="utf-8")
            payload = json.loads(raw)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise DocumentParseError(f"{document.document_id}: {type(exc).__name__}: {exc}") from exc

        passages: list[tuple[str, int, str]] = []
        if document.content_level is ContentLevel.ABSTRACT:
            if isinstance(payload, dict) and payload.get("format") == "hypoforge_abstract_v1":
                text = str(payload.get("text") or "")
                if text.strip():
                    passages.append(("abstract", 0, text))
        else:
            for item in _bioc_documents(payload):
                raw_passages = item.get("passages", [])
                if not isinstance(raw_passages, list):
                    continue
                for passage in raw_passages:
                    if not isinstance(passage, dict):
                        continue
                    if _is_ignored_passage(passage.get("infons")):
                        continue
                    text = str(passage.get("text") or "")
                    if not text.strip():
                        continue
                    try:
                        offset = max(int(passage.get("offset") or 0), 0)
                    except (TypeError, ValueError):
                        offset = 0
                    passages.append((_section_name(passage.get("infons")), offset, text))

        chunks: list[DocumentChunk] = []
        ordinal = 0
        for section, passage_offset, text in passages:
            for start, end, chunk_text in _window_text(
                text, self.target_chars, self.overlap_chars
            ):
                ordinal += 1
                chunks.append(
                    DocumentChunk(
                        chunk_id=self._chunk_id(document, section, ordinal, chunk_text),
                        document_id=document.document_id,
                        paper_id=document.paper_id,
                        section=section,
                        start_offset=passage_offset + start,
                        end_offset=passage_offset + end,
                        text=chunk_text,
                    )
                )
        if not chunks:
            raise DocumentParseError(f"{document.document_id}: no readable passages")
        return chunks


class PDFDocumentParser(DocumentParserProtocol):
    """Extract arXiv PDF text into stable, page-attributed chunks."""

    def __init__(self, target_chars: int = 900, overlap_chars: int = 120) -> None:
        if target_chars < 50:
            raise ValueError("target_chars must be at least 50")
        if overlap_chars < 0 or overlap_chars >= target_chars:
            raise ValueError("overlap_chars must be non-negative and below target_chars")
        self.target_chars = target_chars
        self.overlap_chars = overlap_chars

    @staticmethod
    def _extract_pages(path: Path) -> list[tuple[int, str]]:
        reader = PdfReader(str(path))
        pages: list[tuple[int, str]] = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = str(page.extract_text() or "")
            if text.strip():
                pages.append((page_number, text))
        return pages

    async def parse(self, document: DocumentRecord) -> list[DocumentChunk]:
        if document.content_level is not ContentLevel.PDF:
            raise DocumentParseError(
                f"{document.document_id}: expected PDF content"
            )
        if not document.local_path:
            raise DocumentParseError(
                f"{document.document_id}: local PDF unavailable"
            )
        path = Path(document.local_path)
        try:
            pages = await asyncio.to_thread(self._extract_pages, path)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise DocumentParseError(
                f"{document.document_id}: {type(exc).__name__}: {exc}"
            ) from exc

        chunks: list[DocumentChunk] = []
        ordinal = 0
        for page_number, text in pages:
            section = f"page_{page_number}"
            for start, end, chunk_text in _window_text(
                text,
                self.target_chars,
                self.overlap_chars,
            ):
                ordinal += 1
                chunks.append(
                    DocumentChunk(
                        chunk_id=_chunk_id(
                            document,
                            section,
                            ordinal,
                            chunk_text,
                        ),
                        document_id=document.document_id,
                        paper_id=document.paper_id,
                        section=section,
                        page=page_number,
                        start_offset=start,
                        end_offset=end,
                        text=chunk_text,
                    )
                )
        if not chunks:
            raise DocumentParseError(
                f"{document.document_id}: no readable PDF pages"
            )
        return chunks
