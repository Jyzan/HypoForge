from __future__ import annotations

import pytest
from pydantic import ValidationError

from hypoforge.state import (
    ConfidenceLevel,
    KnowledgeEntry,
    KnowledgeEntryType,
    M2CoverageExport,
    M2EvidenceExport,
    M2KnowledgeExport,
    M2KnowledgeRun,
    M2PaperExport,
    M2SearchProvenance,
    PipelineState,
)


def valid_run() -> M2KnowledgeRun:
    return M2KnowledgeRun(
        sub_question="How does force regulate YAP?",
        papers=[
            M2PaperExport(
                paper_id="PMID:1",
                title="Force and YAP",
                sources=["pubmed"],
                reading_summary="Force promotes YAP import.",
                content_level="structured_fulltext",
                document_id="document:1",
                document_source_uri="https://example.test/pmc/1",
                document_license="PMC Open Access subset",
            )
        ],
        evidence=[
            M2EvidenceExport(
                evidence_id="ev-1",
                paper_id="PMID:1",
                chunk_id="chunk-1",
                section="results",
                page=3,
                quote="Force promoted YAP nuclear import.",
                normalized_claim="Force promotes YAP import.",
                relevance_score=0.95,
                citable=True,
            )
        ],
        knowledge_entries=[
            KnowledgeEntry(
                id="knowledge-1",
                type=KnowledgeEntryType.MECHANISTIC_CONCLUSION,
                content="Force promotes YAP nuclear import.",
                confidence=ConfidenceLevel.HIGH,
                source_paper_id="PMID:1",
                source_paper_title="Force and YAP",
                entities=["YAP"],
                evidence_ids=["ev-1"],
            )
        ],
        search_provenance=M2SearchProvenance(
            coverage=M2CoverageExport(sufficient=True),
            stop_reason="coverage_satisfied",
        ),
    )


def test_package_survives_pipeline_state_json_round_trip() -> None:
    package = M2KnowledgeExport(runs=[valid_run()])
    state = PipelineState(m2_knowledge_export=package)

    restored = PipelineState.model_validate_json(state.model_dump_json())

    assert restored.m2_knowledge_export == package
    assert package.schema_version == "m2-knowledge-export/v1"
    assert package.runs[0].evidence[0].quote.startswith("Force")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda data: data["papers"].append(dict(data["papers"][0])), "duplicate paper_id"),
        (
            lambda data: data["evidence"].append(dict(data["evidence"][0])),
            "duplicate evidence_id",
        ),
        (
            lambda data: data["knowledge_entries"].append(
                dict(data["knowledge_entries"][0])
            ),
            "duplicate knowledge id",
        ),
        (
            lambda data: data["evidence"][0].__setitem__("paper_id", "missing"),
            "unknown paper",
        ),
        (
            lambda data: data["knowledge_entries"][0].__setitem__(
                "source_paper_id", "missing"
            ),
            "unknown paper",
        ),
        (
            lambda data: data["knowledge_entries"][0].__setitem__(
                "evidence_ids", ["missing"]
            ),
            "unknown evidence",
        ),
        (
            lambda data: data["knowledge_entries"][0].__setitem__("evidence_ids", []),
            "no evidence_ids",
        ),
    ],
)
def test_run_rejects_broken_provenance(mutation, message) -> None:
    payload = valid_run().model_dump(mode="python")
    mutation(payload)

    with pytest.raises(ValidationError, match=message):
        M2KnowledgeRun.model_validate(payload)


def test_run_rejects_evidence_assigned_to_a_different_knowledge_source_paper() -> None:
    payload = valid_run().model_dump(mode="python")
    second_paper = dict(payload["papers"][0])
    second_paper.update({"paper_id": "PMID:2", "title": "Second paper"})
    payload["papers"].append(second_paper)
    payload["evidence"][0]["paper_id"] = "PMID:2"

    with pytest.raises(
        ValidationError,
        match=r"knowledge 'knowledge-1' references evidence 'ev-1' from a different paper",
    ):
        M2KnowledgeRun.model_validate(payload)
