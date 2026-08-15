"""OpenAlex adapter for the iterative literature-search source protocol."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Mapping, Optional, Sequence

from hypoforge.literature.models import PaperRecord, SearchQuery
from hypoforge.literature.protocols import LiteratureSourceProtocol

from .academic_source import _dict_to_record


class OpenAlexBackendError(OSError):
    """Raised when the real OpenAlex backend cannot complete a search."""


OpenAlexBackend = Callable[
    [str, int],
    Awaitable[Sequence[Mapping[str, Any]]],
]


def _identifiable(row: Mapping[str, Any]) -> bool:
    return any(
        str(row.get(key) or "").strip()
        for key in ("paper_id", "doi", "title")
    )


class OpenAlexSource(LiteratureSourceProtocol):
    """Search OpenAlex independently of the Semantic Scholar source."""

    source_name = "openalex"

    def __init__(
        self,
        backend: Optional[OpenAlexBackend] = None,
        api_key: str = "",
        mailto: str = "",
        open_access_only: bool = False,
    ) -> None:
        self._api_key = str(api_key or "").strip()
        self._mailto = str(mailto or "").strip()
        self._open_access_only = bool(open_access_only)
        self._default_backend = backend is None
        if backend is None:
            from hypoforge.tools.semantic_scholar import search_openalex_strict

            self._backend = search_openalex_strict
        else:
            self._backend = backend

    async def search(
        self,
        query: SearchQuery,
        limit: int = 20,
    ) -> list[PaperRecord]:
        try:
            if self._default_backend:
                raw = await self._backend(
                    query.text,
                    limit,
                    api_key=self._api_key,
                    mailto=self._mailto,
                    open_access_only=self._open_access_only,
                )
            else:
                raw = await self._backend(query.text, limit)
        except Exception as exc:
            raise OpenAlexBackendError(
                f"openalex backend failed: {type(exc).__name__}: {exc}"
            ) from exc
        return [
            _dict_to_record(dict(row), self.source_name)
            for row in raw
            if _identifiable(row)
        ]


class OpenAlexOpenAccessSource(OpenAlexSource):
    """Citation-sorted OpenAlex search restricted to open-access works."""

    source_name = "openalex_oa"

    def __init__(self, *, api_key: str = "", mailto: str = "") -> None:
        super().__init__(
            api_key=api_key,
            mailto=mailto,
            open_access_only=True,
        )
