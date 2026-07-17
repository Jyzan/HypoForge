"""Batched title-and-abstract Scout Reading with conservative fallback."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
from collections import Counter
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any

from ..models import EvidenceBucket, PaperRecord, ScoutNote
from ..protocols import ScoutReaderProtocol
from ._text import lexical_relevance, tokenize


logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = """\
You screen scientific papers using title, abstract, and metadata only.
Return one grounded Scout note per paper. Never imply that you read full text.
Use evidence buckets only when the supplied material supports the label:
supporting, contradicting, review, recent, classic, methodological.
Keep evidence_summary concise and state uncertainty when the abstract is limited.
Do not invent citations, entities, mechanisms, study designs, or conclusions."""

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "in", "is", "it", "of", "on", "or", "that", "the", "this", "to",
    "using", "via", "was", "were", "with", "we", "our", "study",
}
_REVIEW_PATTERN = re.compile(
    r"\b(systematic review|meta-analysis|meta analysis|review article|review)\b",
    re.IGNORECASE,
)
_METHOD_PATTERN = re.compile(
    r"\b(randomi[sz]ed|cohort|case-control|cross-sectional|in vitro|in vivo|"
    r"animal model|mouse model|assay|cryo-?em|sequencing|proteomics|"
    r"transcriptomics|simulation|experiment|trial)\b",
    re.IGNORECASE,
)
_MECHANISM_PATTERN = re.compile(
    r"\b(regulat|inhibit|activat|mediat|interact|bind|pathway|mechanism|"
    r"phosphorylat|expression|signal)\w*\b",
    re.IGNORECASE,
)
_CONTRADICT_PATTERN = re.compile(
    r"\b(contrary|contradict|inconsistent|failed to|fails to|did not|"
    r"no association|not associated|not support|however|disputed)\b",
    re.IGNORECASE,
)
_SUPPORT_PATTERN = re.compile(
    r"\b(demonstrat(?:e|es|ed)|show(?:s|ed)?|support(?:s|ed)?|confirm(?:s|ed)?|"
    r"associated with|significantly|evidence for)\b",
    re.IGNORECASE,
)
_DESIGN_PATTERNS = (
    ("systematic review", re.compile(r"\bsystematic review\b", re.IGNORECASE)),
    ("meta-analysis", re.compile(r"\bmeta[- ]analysis\b", re.IGNORECASE)),
    ("randomized controlled trial", re.compile(r"\brandomi[sz]ed.*trial\b", re.IGNORECASE)),
    ("cohort study", re.compile(r"\bcohort\b", re.IGNORECASE)),
    ("case-control study", re.compile(r"\bcase[- ]control\b", re.IGNORECASE)),
    ("in vitro study", re.compile(r"\bin vitro\b", re.IGNORECASE)),
    ("animal study", re.compile(r"\b(animal|mouse|mice|rat) model\b", re.IGNORECASE)),
)


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _string_list(value: Any, *, limit: int = 20) -> list[str]:
    values = value if isinstance(value, list) else [value] if isinstance(value, str) else []
    output: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = _clean_text(item)
        key = text.casefold()
        if text and key not in seen:
            output.append(text)
            seen.add(key)
        if len(output) >= limit:
            break
    return output


def _sentences(text: str) -> list[str]:
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", _clean_text(text))
        if sentence.strip()
    ]


def _study_design(paper: PaperRecord) -> str:
    haystack = f"{paper.publication_type} {paper.title} {paper.abstract}"
    for label, pattern in _DESIGN_PATTERNS:
        if pattern.search(haystack):
            return label
    return _clean_text(paper.publication_type)


def _fallback_buckets(paper: PaperRecord, *, current_year: int) -> set[EvidenceBucket]:
    haystack = f"{paper.publication_type} {paper.title} {paper.abstract}"
    buckets: set[EvidenceBucket] = set()
    if _REVIEW_PATTERN.search(haystack):
        buckets.add(EvidenceBucket.REVIEW)
    if _METHOD_PATTERN.search(haystack):
        buckets.add(EvidenceBucket.METHODOLOGICAL)
    if paper.year is not None and paper.year >= current_year - 3:
        buckets.add(EvidenceBucket.RECENT)
    if (
        paper.year is not None
        and paper.year <= current_year - 10
        and (paper.citation_count or 0) >= 100
    ):
        buckets.add(EvidenceBucket.CLASSIC)
    if _CONTRADICT_PATTERN.search(paper.abstract):
        buckets.add(EvidenceBucket.CONTRADICTING)
    if _SUPPORT_PATTERN.search(paper.abstract):
        buckets.add(EvidenceBucket.SUPPORTING)
    return buckets


def _fallback_note(
    sub_question: str,
    paper: PaperRecord,
    *,
    current_year: int,
) -> ScoutNote:
    tokens = [token for token in tokenize(f"{paper.title} {paper.abstract}") if token not in _STOPWORDS]
    counts = Counter(tokens)
    key_terms = [token for token, _ in counts.most_common(10)]
    entities = _string_list(
        re.findall(r"\b[A-Z][A-Z0-9+\-]{1,12}\b", f"{paper.title} {paper.abstract}"),
        limit=10,
    )
    sentences = _sentences(paper.abstract)
    mechanisms = [sentence for sentence in sentences if _MECHANISM_PATTERN.search(sentence)][:3]
    controversies = [sentence for sentence in sentences if _CONTRADICT_PATTERN.search(sentence)][:3]
    summary = " ".join(sentences[:2]) if sentences else paper.title
    rank_relevance = paper.rank_scores.get("query_relevance", 0.0)
    relevance = max(
        lexical_relevance(sub_question, paper.title, paper.abstract),
        rank_relevance if math.isfinite(rank_relevance) else 0.0,
    )
    return ScoutNote(
        paper_id=paper.paper_id,
        main_topic=paper.title,
        key_terms=key_terms,
        entities=entities,
        mechanisms=mechanisms,
        important_authors=list(paper.authors[:3]),
        controversies=controversies,
        relevance_to_question=max(0.0, min(1.0, relevance)),
        evidence_buckets=_fallback_buckets(paper, current_year=current_year),
        study_design=_study_design(paper),
        evidence_summary=summary[:600],
    )


class ScoutReader(ScoutReaderProtocol):
    """Use structured LLM extraction for lightweight, attributable screening."""

    tool_name = "scout_reader"

    def __init__(
        self,
        client: Any | None = None,
        *,
        batch_size: int = 8,
        max_concurrency: int = 2,
        max_tokens: int = 4096,
        abstract_char_limit: int = 6000,
        current_year: int | None = None,
    ) -> None:
        for name, value in (
            ("batch_size", batch_size),
            ("max_concurrency", max_concurrency),
            ("max_tokens", max_tokens),
            ("abstract_char_limit", abstract_char_limit),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        self.client = client
        self.batch_size = batch_size
        self.max_concurrency = max_concurrency
        self.max_tokens = max_tokens
        self.abstract_char_limit = abstract_char_limit
        self.current_year = current_year or datetime.now(timezone.utc).year

    async def read(
        self,
        sub_question: str,
        papers: Sequence[PaperRecord],
    ) -> list[ScoutNote]:
        paper_list = list(papers)
        if not paper_list:
            return []
        if self.client is None:
            return [
                _fallback_note(sub_question, paper, current_year=self.current_year)
                for paper in paper_list
            ]

        batches = [
            paper_list[index : index + self.batch_size]
            for index in range(0, len(paper_list), self.batch_size)
        ]
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def run_batch(batch: list[PaperRecord]) -> list[ScoutNote]:
            async with semaphore:
                try:
                    return await self._read_batch(sub_question, batch)
                except Exception as exc:
                    logger.warning(
                        "Scout batch failed; using deterministic fallback (%s)",
                        type(exc).__name__,
                    )
                    return [
                        _fallback_note(
                            sub_question, paper, current_year=self.current_year
                        )
                        for paper in batch
                    ]

        results = await asyncio.gather(*(run_batch(batch) for batch in batches))
        return [note for batch_notes in results for note in batch_notes]

    async def _read_batch(
        self,
        sub_question: str,
        papers: list[PaperRecord],
    ) -> list[ScoutNote]:
        payload = [
            {
                "paper_id": paper.paper_id,
                "title": paper.title,
                "abstract": paper.abstract[: self.abstract_char_limit],
                "authors": paper.authors,
                "year": paper.year,
                "publication_type": paper.publication_type,
                "citation_count": paper.citation_count,
            }
            for paper in papers
        ]
        schema = {
            "type": "object",
            "properties": {
                "notes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "paper_id": {"type": "string"},
                            "main_topic": {"type": "string"},
                            "key_terms": {"type": "array", "items": {"type": "string"}},
                            "entities": {"type": "array", "items": {"type": "string"}},
                            "mechanisms": {"type": "array", "items": {"type": "string"}},
                            "important_authors": {"type": "array", "items": {"type": "string"}},
                            "controversies": {"type": "array", "items": {"type": "string"}},
                            "candidate_citations": {"type": "array", "items": {"type": "string"}},
                            "relevance_to_question": {"type": "number", "minimum": 0, "maximum": 1},
                            "evidence_buckets": {
                                "type": "array",
                                "items": {"type": "string", "enum": [bucket.value for bucket in EvidenceBucket]},
                            },
                            "study_design": {"type": "string"},
                            "evidence_summary": {"type": "string"},
                        },
                        "required": ["paper_id", "relevance_to_question", "evidence_buckets"],
                    },
                }
            },
            "required": ["notes"],
        }
        result = await self.client.structured_chat(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=(
                f"Research sub-question:\n{sub_question}\n\n"
                f"Papers to screen:\n{json.dumps(payload, ensure_ascii=False)}"
            ),
            output_schema=schema,
            max_tokens=self.max_tokens,
            temperature=0.0,
            disable_thinking=True,
        )
        raw_notes = result.get("notes", []) if isinstance(result, dict) else []
        papers_by_id = {paper.paper_id: paper for paper in papers}
        parsed: dict[str, ScoutNote] = {}
        for raw in raw_notes if isinstance(raw_notes, list) else []:
            if not isinstance(raw, dict):
                continue
            paper_id = _clean_text(raw.get("paper_id"))
            paper = papers_by_id.get(paper_id)
            if paper is None or paper_id in parsed:
                continue
            buckets: set[EvidenceBucket] = set()
            for value in raw.get("evidence_buckets", []):
                try:
                    buckets.add(EvidenceBucket(str(value)))
                except ValueError:
                    continue
            buckets.update(_fallback_buckets(paper, current_year=self.current_year))
            try:
                relevance = float(raw.get("relevance_to_question", 0.0))
                if not math.isfinite(relevance):
                    relevance = 0.0
            except (TypeError, ValueError):
                relevance = 0.0
            fallback = _fallback_note(
                sub_question, paper, current_year=self.current_year
            )
            parsed[paper_id] = ScoutNote(
                paper_id=paper_id,
                main_topic=_clean_text(raw.get("main_topic")) or fallback.main_topic,
                key_terms=_string_list(raw.get("key_terms")) or fallback.key_terms,
                entities=_string_list(raw.get("entities")) or fallback.entities,
                mechanisms=_string_list(raw.get("mechanisms")) or fallback.mechanisms,
                important_authors=_string_list(raw.get("important_authors"))
                or fallback.important_authors,
                controversies=_string_list(raw.get("controversies")),
                candidate_citations=_string_list(raw.get("candidate_citations")),
                relevance_to_question=max(0.0, min(1.0, relevance)),
                evidence_buckets=buckets,
                study_design=_clean_text(raw.get("study_design"))
                or fallback.study_design,
                evidence_summary=(
                    _clean_text(raw.get("evidence_summary"))
                    or fallback.evidence_summary
                )[:600],
            )

        return [
            parsed.get(paper.paper_id)
            or _fallback_note(sub_question, paper, current_year=self.current_year)
            for paper in papers
        ]
