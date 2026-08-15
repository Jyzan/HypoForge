#!/usr/bin/env python3
"""Focused live check of the production M1→M2→parsed-fulltext path.

This script deliberately stops after M2.  It instantiates the same modules and
configuration used by the application; it does not duplicate search or
full-text logic in a benchmark-only implementation.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hypoforge.config import PipelineConfig
from hypoforge.literature.models import EvidenceLinkedKnowledge, PaperReadingResult
from hypoforge.modules.m1_problem_understanding import M1ProblemUnderstanding
from hypoforge.modules.m2_literature_search import M2LiteratureSearch
from hypoforge.state import ConfidenceLevel, KnowledgeEntryType, PipelineState


FULLTEXT_LEVELS = {"structured_fulltext", "pdf", "html", "ocr"}


class DownloadOnlyPaperReader:
    """End the smoke test after resolver/parser/retriever, without Qwen reading."""

    async def read(self, sub_question, paper, evidence):
        citable = next((item for item in evidence if item.citable), None)
        return PaperReadingResult(
            paper_id=paper.paper_id,
            summary="Download-only smoke test: Qwen paper reading skipped.",
            knowledge_entries=(
                [EvidenceLinkedKnowledge(
                    entry_id=f"download-smoke:{paper.paper_id}",
                    entry_type=KnowledgeEntryType.ESTABLISHED_FACT,
                    content="Download and parsing smoke-test evidence only.",
                    confidence=ConfidenceLevel.LOW,
                    evidence_ids=[citable.evidence_id],
                )]
                if citable is not None else []
            ),
        )


def load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def summarize(state: PipelineState, elapsed_seconds: float) -> dict:
    export = state.m2_knowledge_export
    runs = list(export.runs) if export is not None else []
    papers = [paper for run in runs for paper in run.papers]
    source_calls = Counter(
        query.target_source for run in runs for query in run.search_provenance.queries
    )
    fulltexts = [
        paper for paper in papers
        if paper.content_level in FULLTEXT_LEVELS and paper.chunks_parsed > 0
    ]
    return {
        "question": state.input_question,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "problem_card": (
            state.problem_card.model_dump(mode="json")
            if state.problem_card is not None else None
        ),
        "sub_questions": len(runs),
        "selected_papers": len(papers),
        "parsed_fulltexts": len(fulltexts),
        "source_query_counts": dict(source_calls),
        "runs": [
            {
                "sub_question": run.sub_question,
                "source_result_counts": dict(run.search_provenance.source_result_counts),
                "failed_sources": list(run.search_provenance.failed_sources),
                "papers": [
                    {
                        "title": paper.title,
                        "doi": paper.doi,
                        "citations": paper.citation_count,
                        "sources": list(paper.sources),
                        "content_level": paper.content_level,
                        "chunks_parsed": paper.chunks_parsed,
                        "fulltext_source": paper.document_source_uri,
                        "fulltext_failure_category": paper.fulltext_failure_category,
                    }
                    for paper in run.papers
                ],
            }
            for run in runs
        ],
    }


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", default="蛋白质是如何折叠的？")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/web_ui.yaml")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "output/m1_m2_search_optimization_smoke.json",
    )
    args = parser.parse_args()
    load_dotenv(args.env_file)
    config = PipelineConfig.from_yaml(args.config)

    m1_kwargs = config.get_module_kwargs("m1")
    m1_tier = str(m1_kwargs.pop("llm_tier", "base"))
    m1 = M1ProblemUnderstanding(
        llm_config=config.get_llm_for_tier(m1_tier), **m1_kwargs
    )
    m2_kwargs = config.get_module_kwargs("m2")
    m2_tier = str(m2_kwargs.pop("llm_tier", "plus"))
    m2 = M2LiteratureSearch(
        llm_config=config.get_llm_for_tier(m2_tier), **m2_kwargs
    )
    m2.adapter.reading_workflow.reader = DownloadOnlyPaperReader()

    started = time.perf_counter()
    state = PipelineState(input_question=args.question)
    state = state.model_copy(update=await m1(state))
    state = state.model_copy(update=await m2(state))
    payload = summarize(state, time.perf_counter() - started)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
