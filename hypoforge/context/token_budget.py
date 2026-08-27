"""Small dependency-free token estimates used before provider calls."""

from __future__ import annotations

import math


def estimate_input_tokens(text: str, *, message_overhead: int = 8) -> int:
    """Return a conservative mixed ASCII/CJK token estimate.

    Provider tokenizers remain authoritative when usage metadata arrives.
    This estimator is deterministic and intentionally treats every non-ASCII
    code point as one token while grouping ASCII at four characters/token.
    """

    value = str(text or "")
    if not value:
        return 0
    ascii_chars = sum(ord(char) < 128 for char in value)
    non_ascii_chars = len(value) - ascii_chars
    return max(
        1,
        math.ceil(ascii_chars / 4) + non_ascii_chars + message_overhead,
    )
