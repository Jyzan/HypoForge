"""Pipeline-facing M2 literature module.

This module intentionally contains no search or reading implementation. It is
the stable M1-M6 module boundary and delegates the actual literature workflow
to :mod:`hypoforge.literature`.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..literature.adapter import AgenticM2Module
from ..registry import ModuleRegistry
from ..state import PipelineState


@ModuleRegistry.register
class M2LiteratureSearch(AgenticM2Module):
    """Thin Pipeline facade over the Literature-layer M2 implementation."""

    module_name = "m2"
    module_version = "0.3.0-literature-facade"
    description = "Literature search and evidence-linked knowledge extraction"

    async def __call__(
        self,
        state: PipelineState,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return await self.adapter(state, config)

    @classmethod
    def get_input_fields(cls) -> list[str]:
        return ["problem_card"]

    @classmethod
    def get_output_fields(cls) -> list[str]:
        return ["literature_results", "m2_knowledge_export"]
