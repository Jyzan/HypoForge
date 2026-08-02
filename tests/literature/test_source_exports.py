from hypoforge.literature.sources import (
    AcademicSource,
    PubMedBackend,
    PubMedLiteratureSource,
    PubMedSource,
)


def test_primary_and_backup_sources_are_exported_together() -> None:
    assert AcademicSource.source_name == "semantic_scholar"
    assert PubMedSource.source_name == "pubmed"
    assert PubMedLiteratureSource.source_name == "pubmed"
    assert PubMedBackend is not None
