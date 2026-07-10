"""
Semantic Scholar API tool stub.

Replace with real API calls in Phase 1.
"""

from __future__ import annotations

from typing import List

from ..protocol import ToolProtocol
from ..registry import ToolRegistry


@ToolRegistry.register
class SemanticScholarTool(ToolProtocol):
    tool_name = "semantic_scholar"
    tool_description = "Search academic papers via Semantic Scholar API"

    def __init__(self, api_key: str = ""):
        self.api_key = api_key

    async def search(self, query: str, limit: int = 20, **kwargs) -> List[dict]:
        """Stub: returns placeholder results."""
        return [
            {
                "title": f"Paper about {query[:60]}",
                "abstract": "Abstract placeholder — implement real API call.",
                "year": 2024,
                "doi": "10.xxxx/example",
                "authors": ["Author A", "Author B"],
                "journal": "Nature Biomedical Engineering",
            }
        ]

    async def fetch(self, identifier: str, **kwargs) -> dict:
        return {"title": "Placeholder", "abstract": "Implement real fetch."}
