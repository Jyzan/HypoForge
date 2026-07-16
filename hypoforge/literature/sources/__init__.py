"""Single-source literature search implementations.

Each module wraps a legacy ``ToolProtocol`` backend so it conforms to
``LiteratureSourceProtocol`` for use with ``IterativeSearchAgent``.
"""

from .academic_source import AcademicSource
from .pubmed_source import PubMedSource

__all__ = ["AcademicSource", "PubMedSource"]
