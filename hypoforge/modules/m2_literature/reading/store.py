"""Small in-memory chunk catalog used during one reading workflow run."""

from __future__ import annotations

from collections.abc import Sequence

from ..models import DocumentChunk


class InMemoryChunkStore:
    def __init__(self) -> None:
        self._by_paper: dict[str, list[DocumentChunk]] = {}

    def replace(self, paper_id: str, chunks: Sequence[DocumentChunk]) -> None:
        self._by_paper[paper_id] = [item.model_copy(deep=True) for item in chunks]

    def get(self, paper_ids: Sequence[str]) -> list[DocumentChunk]:
        output: list[DocumentChunk] = []
        for paper_id in paper_ids:
            output.extend(
                item.model_copy(deep=True) for item in self._by_paper.get(paper_id, [])
            )
        return output
