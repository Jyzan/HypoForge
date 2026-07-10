"""
PubMed E-utilities tool stub.

Replace with real API calls in Phase 1.
"""

from __future__ import annotations

from typing import List

from ..protocol import ToolProtocol
from ..registry import ToolRegistry


@ToolRegistry.register
class PubMedTool(ToolProtocol):
    tool_name = "pubmed"
    tool_description = "Search biomedical literature via PubMed E-utilities API"

    def __init__(self, api_key: str = ""):
        self.api_key = api_key  # NCBI API key (optional but recommended)

    async def search(self, query: str, limit: int = 20, **kwargs) -> List[dict]:
        """Stub: returns placeholder results."""
        return [
            {
                "title": f"PubMed result for: {query[:60]}",
                "abstract": "Abstract placeholder — implement real E-utilities call.",
                "year": 2024,
                "pmid": "12345678",
                "authors": ["Smith J", "Jones K"],
                "journal": "Cell",
            }
        ]

    async def fetch(self, identifier: str, **kwargs) -> dict:
        return {"title": "Placeholder", "abstract": "Implement real fetch.", "pmid": identifier}
