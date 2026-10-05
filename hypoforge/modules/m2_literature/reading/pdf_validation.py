"""Cheap PDF envelope checks before caching or text extraction.

These checks detect incomplete downloads, not every possible PDF corruption.
The actual parser still validates the document structure and readable pages.
"""

from __future__ import annotations

import re
from typing import BinaryIO


class PDFIntegrityError(ValueError):
    """A purported PDF is missing its header or final revision marker."""


_EOF = re.compile(rb"(?:^|[\r\n])%%EOF(?:[\x00\t\n\f\r ]|$)")
_TAIL_BYTES = 65_536


def validate_pdf_stream(stream: BinaryIO) -> None:
    """Check a seekable PDF without loading the whole document into memory."""
    stream.seek(0, 2)
    size = stream.tell()
    stream.seek(0)
    if not stream.read(1024).lstrip().startswith(b"%PDF-"):
        raise PDFIntegrityError("invalid PDF response: missing %PDF- header")
    stream.seek(max(0, size - _TAIL_BYTES))
    tail = stream.read(_TAIL_BYTES)
    markers = list(_EOF.finditer(tail))
    if not markers:
        raise PDFIntegrityError(
            "PDF integrity check failed: %%EOF marker missing; "
            "possible incomplete download"
        )
    # A previous revision's EOF must not hide a truncated incremental update.
    if re.search(rb"(?:^|[\r\n])startxref(?:[\t\r\n ]|$)", tail[markers[-1].end():]):
        raise PDFIntegrityError(
            "PDF integrity check failed: incomplete final revision after %%EOF"
        )
