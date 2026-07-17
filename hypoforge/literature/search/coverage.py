"""Evidence coverage aggregation and gap synthesis."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any

from ..models import (
    CoverageReport,
    EvidenceBucket,
    PaperRecord,
    ScoutNote,
    SearchState,
)
from ..protocols import CoverageEvaluatorProtocol


logger = logging.getLogger(__name__)


_REVIEW_PATTERN = re.compile(
    r"\b(systematic review|meta-analysis|meta analysis|review)\b", re.IGNORECASE
)
_METHOD_PATTERN = re.compile(
    r"\b(randomi[sz]ed|cohort|case-control|cross-sectional|in vitro|in vivo|"
    r"animal study|assay|trial|experiment|sequencing|proteomics|cryo-?em)\b",
    re.IGNORECASE,
)
_SYSTEM_PROMPT = """\
You assess topic coverage from paper-level Scout notes. The caller computes
evidence buckets and sufficiency deterministically. Return only concise covered
topics, scientifically indispensable missing topics, and a rationale. Do not
invent findings, do not request full-text details, and do not decide whether
coverage is sufficient."""


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _stable_strings(values: Any, *, limit: int = 20) -> list[str]:
    if not isinstance(values, list):
        return []
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _clean_text(value)
        key = text.casefold()
        if text and key not in seen:
            output.append(text)
            seen.add(key)
        if len(output) >= limit:
            break
    return output


class CoverageEvaluator(CoverageEvaluatorProtocol):
    """Combine Scout judgments under a deterministic balanced-coverage gate."""

    tool_name = "coverage_evaluator"

    def __init__(
        self,
        client: Any | None = None,
        *,
        relevance_threshold: float = 0.55,
        min_relevant_papers: int = 5,
        min_bucket_count: int = 4,
        max_tokens: int = 2048,
        current_year: int | None = None,
    ) -> None:
        if not 0.0 <= relevance_threshold <= 1.0:
            raise ValueError("relevance_threshold must be between 0 and 1")
        if min_relevant_papers <= 0:
            raise ValueError("min_relevant_papers must be positive")
        if not 1 <= min_bucket_count <= len(EvidenceBucket):
            raise ValueError("min_bucket_count must fit the evidence bucket set")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        self.client = client
        self.relevance_threshold = relevance_threshold
        self.min_relevant_papers = min_relevant_papers
        self.min_bucket_count = min_bucket_count
        self.max_tokens = max_tokens
        self.current_year = current_year or datetime.now(timezone.utc).year

    async def evaluate(
        self,
        sub_question: str,
        papers: Sequence[PaperRecord],
        scout_notes: Sequence[ScoutNote],
        state: SearchState,
    ) -> CoverageReport:
        papers_by_id = {paper.paper_id: paper for paper in papers}
        relevant: list[tuple[ScoutNote, PaperRecord]] = []
        seen_ids: set[str] = set()
        for note in scout_notes:
            paper = papers_by_id.get(note.paper_id)
            if (
                paper is None
                or note.paper_id in seen_ids
                or note.relevance_to_question < self.relevance_threshold
            ):
                continue
            relevant.append((note, paper))
            seen_ids.add(note.paper_id)

        covered: set[EvidenceBucket] = set()
        fallback_topics: list[str] = []
        for note, paper in relevant:
            buckets = set(note.evidence_buckets)
            haystack = f"{paper.publication_type} {paper.title} {note.study_design}"
            if paper.year is not None and paper.year >= self.current_year - 3:
                buckets.add(EvidenceBucket.RECENT)
            if (
                paper.year is not None
                and paper.year <= self.current_year - 10
                and (paper.citation_count or 0) >= 100
            ):
                buckets.add(EvidenceBucket.CLASSIC)
            if _REVIEW_PATTERN.search(haystack):
                buckets.add(EvidenceBucket.REVIEW)
            if _METHOD_PATTERN.search(haystack):
                buckets.add(EvidenceBucket.METHODOLOGICAL)
            covered.update(buckets)
            fallback_topics.extend(
                value for value in [note.main_topic, *note.mechanisms] if value
            )

        all_buckets = set(EvidenceBucket)
        missing_buckets = all_buckets - covered
        hard_gaps = self._hard_gaps(
            sub_question=sub_question,
            relevant_count=len(relevant),
            covered=covered,
        )
        covered_topics = _stable_strings(fallback_topics)
        model_missing: list[str] = []
        model_rationale = ""
        if relevant and self.client is not None:
            try:
                covered_topics, model_missing, model_rationale = await self._synthesize(
                    sub_question=sub_question,
                    relevant=relevant,
                    state=state,
                    fallback_topics=covered_topics,
                )
            except Exception as exc:
                logger.warning(
                    "Coverage synthesis failed; using deterministic fallback (%s)",
                    type(exc).__name__,
                )

        missing_topics = _stable_strings([*model_missing, *hard_gaps])
        hard_gate = (
            len(relevant) >= self.min_relevant_papers
            and len(covered) >= self.min_bucket_count
            and EvidenceBucket.SUPPORTING in covered
            and EvidenceBucket.CONTRADICTING in covered
        )
        sufficient = hard_gate and not model_missing
        if sufficient:
            missing_topics = []

        summary = (
            f"{len(relevant)} relevant papers; "
            f"{len(covered)}/{len(EvidenceBucket)} evidence buckets covered."
        )
        rationale = f"{summary} {model_rationale}".strip()
        if not model_rationale:
            rationale = (
                f"{summary} Balanced coverage requirements are "
                f"{'satisfied' if sufficient else 'not yet satisfied'}."
            )
        return CoverageReport(
            covered_buckets=covered,
            missing_buckets=missing_buckets,
            covered_topics=covered_topics,
            missing_topics=missing_topics,
            sufficient=sufficient,
            rationale=rationale,
        )

    def _hard_gaps(
        self,
        *,
        sub_question: str,
        relevant_count: int,
        covered: set[EvidenceBucket],
    ) -> list[str]:
        gaps: list[str] = []
        if relevant_count < self.min_relevant_papers:
            gaps.append(
                f"additional relevant studies for {sub_question} "
                f"({self.min_relevant_papers - relevant_count} more needed)"
            )
        mandatory = (
            EvidenceBucket.SUPPORTING,
            EvidenceBucket.CONTRADICTING,
        )
        missing_mandatory = 0
        for bucket in mandatory:
            if bucket not in covered:
                gaps.append(f"{bucket.value} evidence for {sub_question}")
                missing_mandatory += 1
        if len(covered) < self.min_bucket_count:
            needed = max(
                0,
                self.min_bucket_count - len(covered) - missing_mandatory,
            )
            optional_missing = [
                bucket
                for bucket in EvidenceBucket
                if bucket not in covered and bucket not in mandatory
            ]
            gaps.extend(
                f"{bucket.value} evidence for {sub_question}"
                for bucket in optional_missing[:needed]
            )
        return gaps

    async def _synthesize(
        self,
        *,
        sub_question: str,
        relevant: list[tuple[ScoutNote, PaperRecord]],
        state: SearchState,
        fallback_topics: list[str],
    ) -> tuple[list[str], list[str], str]:
        payload = [
            {
                "paper_id": note.paper_id,
                "main_topic": note.main_topic,
                "mechanisms": note.mechanisms,
                "controversies": note.controversies,
                "study_design": note.study_design,
                "evidence_summary": note.evidence_summary,
                "evidence_buckets": sorted(
                    bucket.value for bucket in note.evidence_buckets
                ),
            }
            for note, _ in relevant
        ]
        schema = {
            "type": "object",
            "properties": {
                "covered_topics": {"type": "array", "items": {"type": "string"}},
                "missing_topics": {"type": "array", "items": {"type": "string"}},
                "rationale": {"type": "string"},
            },
            "required": ["covered_topics", "missing_topics", "rationale"],
        }
        result = await self.client.structured_chat(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=(
                f"Research sub-question:\n{sub_question}\n\n"
                f"Previous missing topics: {sorted(state.missing_topics)}\n\n"
                "Return missing_topics only for indispensable scientific topics, "
                "not merely for optional evidence categories.\n\n"
                f"Scout notes:\n{json.dumps(payload, ensure_ascii=False)}"
            ),
            output_schema=schema,
            max_tokens=self.max_tokens,
            temperature=0.0,
            disable_thinking=True,
        )
        if not isinstance(result, dict):
            return fallback_topics, [], ""
        covered_topics = _stable_strings(result.get("covered_topics")) or fallback_topics
        missing_topics = _stable_strings(result.get("missing_topics"))
        rationale = _clean_text(result.get("rationale"))[:1000]
        return covered_topics, missing_topics, rationale
