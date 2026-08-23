"""Small deterministic text helpers shared by literature search tools."""

from __future__ import annotations

import re
import unicodedata


_WORD_PATTERN = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_CJK_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


def normalize_text(text: str) -> str:
    """Return a comparison-friendly Unicode string."""
    normalized = unicodedata.normalize("NFKC", text or "").casefold()
    normalized = re.sub(r"[^\w\u3400-\u4dbf\u4e00-\u9fff]+", " ", normalized)
    return " ".join(normalized.split())


def tokenize(text: str) -> list[str]:
    """Tokenize Latin text and add unigrams/bigrams for CJK text."""
    normalized = unicodedata.normalize("NFKC", text or "").casefold()
    tokens = _WORD_PATTERN.findall(normalized)
    cjk = _CJK_PATTERN.findall(normalized)
    tokens.extend(cjk)
    tokens.extend("".join(cjk[index : index + 2]) for index in range(len(cjk) - 1))
    return tokens


def lexical_relevance(query: str, title: str, abstract: str) -> float:
    """Estimate query coverage, weighting title matches above abstract matches."""
    query_tokens = set(tokenize(query))
    if not query_tokens:
        return 0.0
    title_tokens = set(tokenize(title))
    abstract_tokens = set(tokenize(abstract))
    title_coverage = len(query_tokens & title_tokens) / len(query_tokens)
    abstract_coverage = len(query_tokens & abstract_tokens) / len(query_tokens)
    return min(1.0, 0.7 * title_coverage + 0.3 * abstract_coverage)
