"""Structured attribution for M2 full-text retrieval failures.

Every full-text retrieval failure that is not a code bug must land in one of
these categories so downstream exports and the web UI can explain why a
paper degraded to its abstract.  Categories are plain string constants so
they serialize directly into run JSON and stay readable by the frontend.
"""

from __future__ import annotations

import re

# Publisher anti-crawling rejection (HTTP 401/403/407/418/429/451 etc.).
PUBLISHER_BLOCKED = "publisher_blocked"
# The link is not a real PDF (landing page / HTML body / non-PDF redirect).
INVALID_PDF_LINK = "invalid_pdf_link"
# Objectively no full text: no OA PDF, no PMID/PMCID, no PMC OA copy.
NO_FULLTEXT_AVAILABLE = "no_fulltext_available"
# Catch-all carrying the raw error text; guarantees every failure has one.
OTHER = "other"

#: Rejection-style HTTP status codes treated as publisher anti-crawling.
BLOCKED_HTTP_CODES = frozenset({401, 403, 407, 418, 429, 451})

_HTTP_ERROR = re.compile(r"HTTP Error (?P<code>\d{3})")


def category_for_status_code(code: int) -> str:
    """Map an HTTP status code to a failure category."""

    return PUBLISHER_BLOCKED if code in BLOCKED_HTTP_CODES else OTHER


def attribute_error_text(error: str) -> str:
    """Classify a flat error string when no structured context survives.

    Used by workflow early-exit paths where only the error message is
    available.  Always returns a category — never an empty string.
    """

    text = str(error or "")
    match = _HTTP_ERROR.search(text)
    if match and int(match.group("code")) in BLOCKED_HTTP_CODES:
        return PUBLISHER_BLOCKED
    lowered = text.casefold()
    if "invalid pdf" in lowered:
        return INVALID_PDF_LINK
    if any(
        marker in lowered
        for marker in (
            "identifier unavailable",
            "full text unavailable",
            "no non-empty passages",
            "no readable content",
        )
    ):
        return NO_FULLTEXT_AVAILABLE
    return OTHER
