"""Route literature documents by source and content representation."""

from __future__ import annotations

from ..models import ContentLevel, DocumentChunk, DocumentRecord, PaperRecord
from ..protocols import DocumentParserProtocol, FulltextResolverProtocol


class RoutingFulltextResolver(FulltextResolverProtocol):
    """Use arXiv PDF resolution only for records linked to arXiv."""

    def __init__(
        self,
        pmc_resolver: FulltextResolverProtocol,
        arxiv_resolver: FulltextResolverProtocol,
    ) -> None:
        self.pmc_resolver = pmc_resolver
        self.arxiv_resolver = arxiv_resolver

    def _resolver_for(self, paper: PaperRecord) -> FulltextResolverProtocol:
        if "arxiv" in paper.sources or paper.external_ids.get("arxiv"):
            return self.arxiv_resolver
        return self.pmc_resolver

    async def resolve(self, paper: PaperRecord) -> DocumentRecord:
        return await self._resolver_for(paper).resolve(paper)

    async def resolve_abstract(self, paper: PaperRecord) -> DocumentRecord:
        fallback = getattr(self._resolver_for(paper), "resolve_abstract", None)
        if not callable(fallback):
            raise RuntimeError("source-aware abstract fallback is unavailable")
        return await fallback(paper)


class RoutingDocumentParser(DocumentParserProtocol):
    """Use the PDF parser only for PDF documents."""

    def __init__(
        self,
        document_parser: DocumentParserProtocol,
        pdf_parser: DocumentParserProtocol,
    ) -> None:
        self.document_parser = document_parser
        self.pdf_parser = pdf_parser

    async def parse(self, document: DocumentRecord) -> list[DocumentChunk]:
        if document.content_level is ContentLevel.PDF:
            return await self.pdf_parser.parse(document)
        return await self.document_parser.parse(document)
