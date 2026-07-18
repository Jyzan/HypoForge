"""Network smoke test for M3 without a local ``papers/`` directory or an LLM.

It uses PubMed for one real article, asks M3 to obtain lawful open text from
PMC/OpenAlex, and verifies that M4's existing evidence-bucket interface can
consume the enriched ``literature_results`` produced by M3.
"""

from __future__ import annotations

import asyncio
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hypoforge.modules.m3_evidence_graph import M3EvidenceGraph
from hypoforge.modules.m4_hypothesis_generation import M4HypothesisGeneration
from hypoforge.config import PipelineConfig
from hypoforge.state import (
    ConfidenceLevel,
    KnowledgeEntry,
    KnowledgeEntryType,
    LiteratureResult,
    PipelineState,
    ProblemCard,
)
from hypoforge.tools.pubmed_search import PubMedTool


async def main(use_llm: bool = False, test_relations: bool = False) -> None:
    query = "alpha-synuclein Parkinson disease open access"
    papers = await PubMedTool().search(query, limit=3)
    if not papers:
        raise RuntimeError("PubMed returned no papers; check network/NCBI availability.")

    # Try a few live PubMed hits because not every PubMed record has OA text.
    for paper in papers:
        pmid = str(paper.get("pmid") or "")
        if not pmid:
            continue
        state = PipelineState(
            input_question="How does alpha-synuclein misfolding contribute to Parkinson disease?",
            problem_card=ProblemCard(
                original_question="How does alpha-synuclein misfolding contribute to Parkinson disease?",
                sub_questions=["What evidence supports alpha-synuclein propagation?"],
                key_entities=["alpha-synuclein", "Parkinson disease"],
            ),
            literature_results=[LiteratureResult(
                sub_question="What evidence supports alpha-synuclein propagation?",
                papers_retrieved=1,
                knowledge_entries=[KnowledgeEntry(
                    id=f"SEED_{pmid}",
                    type=KnowledgeEntryType.MECHANISTIC_CONCLUSION,
                    content=(paper.get("abstract") or paper.get("title") or "")[:3000],
                    confidence=ConfidenceLevel.MEDIUM,
                    source_paper_id=f"PMID:{pmid}",
                    source_paper_title=str(paper.get("title") or pmid),
                    entities=["alpha-synuclein", "Parkinson disease"],
                )],
            )],
        )
        llm_config = PipelineConfig.from_yaml("configs/full_pipeline.yaml").get_llm_for_tier("plus") if use_llm else None
        module = M3EvidenceGraph(
            mode="llm" if use_llm else "rule", llm_config=llm_config,
            local_paper_dir="__no_local_papers__",
            cache_dir=".hypoforge_cache/m3_network_smoke",
            enable_network_fulltext=True, max_sources=1, max_queries=1,
            retrieve_k=3, evidence_k=2 if test_relations else 1, min_relevance=0,
        )
        result = await module(state)
        graph = result["evidence_graph"]
        report = graph.grounding_report
        if report.get("full_text_papers", 0) <= 0:
            continue

        # This is the M3 -> M4 contract: M4 reads bucket IDs from the graph
        # and resolves them against the enriched literature_results.
        m4_state = state.model_copy(update={
            "evidence_graph": graph,
            "literature_results": result["literature_results"],
        })
        bucket_text = M4HypothesisGeneration(mode="stub")._graph_bucket_text(
            m4_state, "established_facts"
        )
        assert "Exact excerpt:" in bucket_text, "M4 could not see M3 full-text evidence"
        assert report["chunks_total"] > 0
        assert report["evidence_records"] > 0
        assert report["claims_total"] > 0
        if use_llm:
            evidence_nodes = [node for node in graph.nodes if node.type.value == "evidence_record"]
            assert evidence_nodes and evidence_nodes[0].metadata["epistemic_status"] != "retrieval_only"
        print(json.dumps({
            "pmid": pmid,
            "title": paper.get("title"),
            "report": report,
            "graph_nodes": len(graph.nodes),
            "graph_edges": len(graph.edges),
            "m4_bucket_preview": bucket_text[:500],
        }, ensure_ascii=False, indent=2))
        return

    raise RuntimeError("None of the three PubMed hits yielded lawful OA full text via PMC/OpenAlex.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", action="store_true", help="Run Qwen RCS instead of extractive fallback.")
    parser.add_argument("--relations", action="store_true", help="Use two chunks and trigger cross-claim relation judgement when possible.")
    args = parser.parse_args()
    asyncio.run(main(use_llm=args.llm, test_relations=args.relations))
