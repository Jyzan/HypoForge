"""Pure-Python multi-intent BM25 retrieval with diversity and context expansion."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from collections.abc import Sequence

from ..models import DocumentChunk, EvidenceChunk
from ..protocols import EvidenceRetrieverProtocol
from .store import InMemoryChunkStore

_TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_-]*|\d+(?:\.\d+)?|[\u4e00-\u9fff]")

_INTENTS: dict[str, tuple[str, dict[str, float]]] = {
    "core": ("evidence association effect", {}),
    "mechanism": (
        "mechanism pathway causal activation inhibition",
        {"results": 0.18, "discussion": 0.12},
    ),
    "conflict": (
        "contradict negative no effect limitation uncertainty",
        {"limitations": 0.24, "discussion": 0.12, "results": 0.08},
    ),
    "method": (
        "method experiment assay model measurement CRISPR sequencing",
        {"methods": 0.25},
    ),
}


def _tokens(text: str) -> list[str]:
    return [item.casefold() for item in _TOKEN_PATTERN.findall(text)]


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _first_sentence(text: str) -> str:
    match = re.search(r"[.!?。！？](?:\s|$)", text)
    return text[: match.end()].strip()[:300] if match else text.strip()[:300]


class HybridEvidenceRetriever(EvidenceRetrieverProtocol):
    def __init__(
        self,
        store: InMemoryChunkStore,
        *,
        max_context_chars: int = 12_000,
        mmr_lambda: float = 0.75,
    ) -> None:
        if max_context_chars <= 0:
            raise ValueError("max_context_chars must be positive")
        if not 0.0 <= mmr_lambda <= 1.0:
            raise ValueError("mmr_lambda must be between zero and one")
        self.store = store
        self.max_context_chars = max_context_chars
        self.mmr_lambda = mmr_lambda

    @staticmethod
    def _bm25(documents: list[list[str]], query: list[str]) -> list[float]:
        if not documents or not query:
            return [0.0 for _ in documents]
        count = len(documents)
        average_length = sum(len(item) for item in documents) / max(count, 1)
        document_frequency = Counter(
            term for document in documents for term in set(document)
        )
        scores: list[float] = []
        for document in documents:
            frequencies = Counter(document)
            score = 0.0
            for term in set(query):
                frequency = frequencies.get(term, 0)
                if frequency == 0:
                    continue
                df = document_frequency.get(term, 0)
                inverse = math.log(1.0 + (count - df + 0.5) / (df + 0.5))
                denominator = frequency + 1.5 * (
                    1.0 - 0.75 + 0.75 * len(document) / max(average_length, 1.0)
                )
                score += inverse * frequency * 2.5 / denominator
            scores.append(score)
        return scores

    @staticmethod
    def _evidence(
        chunk: DocumentChunk,
        relevance: float,
        *,
        citable: bool = True,
    ) -> EvidenceChunk:
        digest = hashlib.sha256(chunk.chunk_id.encode("utf-8")).hexdigest()[:16]
        return EvidenceChunk(
            evidence_id=f"{chunk.paper_id}:evidence:{digest}",
            paper_id=chunk.paper_id,
            chunk_id=chunk.chunk_id,
            section=chunk.section,
            page=chunk.page,
            quote=chunk.text,
            normalized_claim=_first_sentence(chunk.text),
            relevance_score=max(0.0, min(relevance, 1.0)),
            citable=citable,
        )

    async def retrieve(
        self,
        query: str,
        paper_ids: Sequence[str],
        top_k: int = 10,
    ) -> list[EvidenceChunk]:
        clean_query = " ".join(query.split())
        chunks = self.store.get(paper_ids)
        if not clean_query or not chunks or top_k <= 0:
            return []

        document_tokens = [_tokens(item.text) for item in chunks]
        base_query_tokens = _tokens(clean_query)
        query_phrases = {
            term for term in base_query_tokens if len(term) >= 3 and not term.isdigit()
        }
        merged_scores = [0.0 for _ in chunks]
        per_intent = max(2, math.ceil(top_k / len(_INTENTS)))

        for suffix, section_weights in _INTENTS.values():
            intent_tokens = [*base_query_tokens, *_tokens(suffix)]
            scores = self._bm25(document_tokens, intent_tokens)
            adjusted: list[tuple[float, int]] = []
            for index, (chunk, score) in enumerate(zip(chunks, scores, strict=True)):
                lower = chunk.text.casefold()
                entity_bonus = 0.04 * sum(term in lower for term in query_phrases)
                section_bonus = section_weights.get(chunk.section.casefold(), 0.0)
                total = score + entity_bonus + section_bonus
                adjusted.append((total, index))
            adjusted.sort(key=lambda item: (-item[0], item[1]))
            for score, index in adjusted[:per_intent]:
                merged_scores[index] = max(merged_scores[index], score)

        maximum = max(merged_scores, default=0.0)
        if maximum <= 0.0:
            fallback_sections = {"abstract", "results", "discussion", "methods"}
            fallback = [
                index for index, item in enumerate(chunks)
                if item.section.casefold() in fallback_sections
            ][:top_k]
            return [self._evidence(chunks[index], 0.01) for index in fallback]

        relevance = [score / maximum for score in merged_scores]
        candidates = [index for index, score in enumerate(merged_scores) if score > 0.0]
        selected: list[int] = []
        context_only: set[int] = set()
        selected_chars = 0
        token_sets = [set(item) for item in document_tokens]

        while candidates and len(selected) < top_k:
            best_index = max(
                candidates,
                key=lambda index: (
                    self.mmr_lambda * relevance[index]
                    - (1.0 - self.mmr_lambda)
                    * max(
                        (_jaccard(token_sets[index], token_sets[item]) for item in selected),
                        default=0.0,
                    ),
                    -index,
                ),
            )
            candidates.remove(best_index)
            size = len(chunks[best_index].text)
            if selected_chars + size <= self.max_context_chars:
                selected.append(best_index)
                selected_chars += size

            if len(selected) == 1 and len(selected) < top_k:
                for neighbor in (best_index - 1, best_index + 1):
                    if not 0 <= neighbor < len(chunks):
                        continue
                    if chunks[neighbor].paper_id != chunks[best_index].paper_id:
                        continue
                    if chunks[neighbor].section != chunks[best_index].section:
                        continue
                    size = len(chunks[neighbor].text)
                    if selected_chars + size > self.max_context_chars:
                        continue
                    selected.append(neighbor)
                    context_only.add(neighbor)
                    selected_chars += size
                    if neighbor in candidates:
                        candidates.remove(neighbor)
                    break

        return [
            self._evidence(
                chunks[index],
                relevance[index],
                citable=index not in context_only,
            )
            for index in selected
        ]
