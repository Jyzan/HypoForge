from pathlib import Path

import pytest

from hypoforge.modules.m3_evidence_graph import M3EvidenceGraph
from hypoforge.state import (
    ConfidenceLevel,
    EvidenceEdgeRelation,
    EvidenceNodeType,
    KnowledgeEntry,
    KnowledgeEntryType,
    LiteratureResult,
    PipelineState,
    ProblemCard,
)


@pytest.mark.asyncio
async def test_m3_grounding_uses_local_fulltext_and_preserves_m4_contract(tmp_path: Path):
    papers = tmp_path / "papers"
    papers.mkdir()
    (papers / "PMID_12345.txt").write_text(
        "Results. Protein X directly activates pathway Y in human hepatocytes. "
        "The effect was abolished by inhibitor Z. The experiment used three biological replicates.",
        encoding="utf-8",
    )
    seed = KnowledgeEntry(
        id="K1", type=KnowledgeEntryType.MECHANISTIC_CONCLUSION,
        content="Protein X may affect pathway Y.", confidence=ConfidenceLevel.MEDIUM,
        source_paper_id="PMID:12345", source_paper_title="Protein X study",
        entities=["Protein X", "pathway Y"],
    )
    state = PipelineState(
        input_question="How does Protein X activate pathway Y?",
        problem_card=ProblemCard(
            original_question="How does Protein X activate pathway Y?",
            sub_questions=["Does inhibitor Z block Protein X signalling?"],
            key_entities=["Protein X", "pathway Y", "inhibitor Z"],
        ),
        literature_results=[LiteratureResult(
            sub_question="How does Protein X activate pathway Y?",
            papers_retrieved=1, knowledge_entries=[seed],
        )],
    )
    module = M3EvidenceGraph(
        mode="rule", local_paper_dir=str(papers), cache_dir=str(tmp_path / "cache"),
        enable_network_fulltext=False, min_relevance=0,
    )
    output = await module(state)
    graph = output["evidence_graph"]

    assert graph.grounding_report["full_text_papers"] == 1
    assert any(node.type == EvidenceNodeType.CHUNK for node in graph.nodes)
    assert any(node.type == EvidenceNodeType.EVIDENCE_RECORD for node in graph.nodes)
    assert any(edge.relation == EvidenceEdgeRelation.GROUNDS for edge in graph.edges)
    enriched_ids = {
        entry.id for result in output["literature_results"] for entry in result.knowledge_entries
    }
    assert any(entry_id.startswith("ER_") for entry_id in enriched_ids)
    assert set(graph.established_facts).issubset(enriched_ids)


@pytest.mark.asyncio
async def test_m3_marks_seed_fallback_instead_of_claiming_fulltext(tmp_path: Path):
    seed = KnowledgeEntry(
        id="K1", type=KnowledgeEntryType.ESTABLISHED_FACT,
        content="A sufficiently detailed seed sentence about kinase A activating substrate B.",
        source_paper_id="PMID:99999999", source_paper_title="Unavailable paper",
    )
    state = PipelineState(
        input_question="Does kinase A activate substrate B?",
        literature_results=[LiteratureResult(sub_question="kinase A", knowledge_entries=[seed])],
    )
    module = M3EvidenceGraph(
        mode="rule", local_paper_dir=str(tmp_path / "missing"),
        cache_dir=str(tmp_path / "cache"), enable_network_fulltext=False,
        min_relevance=0,
    )
    graph = (await module(state))["evidence_graph"]
    assert graph.grounding_report["fallback_papers"] == 1
    source = next(node for node in graph.nodes if node.type == EvidenceNodeType.SOURCE)
    assert source.metadata["acquisition_status"] == "m2_seed_fallback"
    assert graph.grounding_report["warnings"]
