"""Single-source literature search implementations.

Each module wraps a legacy ``ToolProtocol`` backend so it conforms to
``LiteratureSourceProtocol`` for use with ``IterativeSearchAgent``.
"""

from .academic_source import AcademicSource
from .arxiv_source import ArxivBackendError, ArxivSource
from .openalex_source import OpenAlexBackendError, OpenAlexSource
from .pubmed import PubMedBackend, PubMedLiteratureSource
from .pubmed_source import PubMedSource

__all__ = [
    "AcademicSource",
    "ArxivBackendError",
    "ArxivSource",
    "OpenAlexBackendError",
    "OpenAlexSource",
    "PubMedBackend",
    "PubMedLiteratureSource",
    "PubMedSource",
]
