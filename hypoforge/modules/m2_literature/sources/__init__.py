"""Single-source literature search implementations.

Each module wraps a legacy ``ToolProtocol`` backend so it conforms to
``LiteratureSourceProtocol`` for use with ``IterativeSearchAgent``.
"""

from .academic_source import AcademicSource
from .arxiv_source import ArxivBackendError, ArxivSource
from .openalex_source import (
    OpenAlexBackendError,
    OpenAlexOpenAccessSource,
    OpenAlexSource,
)
from .openalex_enriched_scholar import OpenAlexEnrichedScholarSource
from .pubmed import PubMedBackend, PubMedLiteratureSource
from .pubmed_source import PubMedSource
from .scholar_proxy_source import ScholarProxyError, ScholarProxySource
from .specialist_sources import (
    AdsSource,
    CrossrefSource,
    DblpSource,
    EuropePMCSource,
    InspireSource,
    NtrsSource,
    OstiSource,
    SpecialistSourceError,
    ZbMathSource,
)

__all__ = [
    "AcademicSource",
    "ArxivBackendError",
    "ArxivSource",
    "OpenAlexBackendError",
    "OpenAlexEnrichedScholarSource",
    "OpenAlexOpenAccessSource",
    "OpenAlexSource",
    "PubMedBackend",
    "PubMedLiteratureSource",
    "PubMedSource",
    "ScholarProxyError",
    "ScholarProxySource",
    "AdsSource",
    "CrossrefSource",
    "DblpSource",
    "EuropePMCSource",
    "InspireSource",
    "NtrsSource",
    "OstiSource",
    "SpecialistSourceError",
    "ZbMathSource",
]
