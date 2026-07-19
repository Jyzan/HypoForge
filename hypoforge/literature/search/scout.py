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
from ._text import lexical_relevance, normalize_text, tokenize


logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = """\
You screen scientific papers using title, abstract, and metadata only.
Return one grounded Scout note per paper. Never imply that you read full text.
Use evidence buckets only when the supplied material supports the label:
supporting, contradicting, review, recent, classic, methodological.
Keep evidence_summary concise and state uncertainty when the abstract is limited.
Estimate directness_to_question from 0 to 1: use 1 only when the paper directly
tests the central relation in the question; contextual reviews and indirect
associations should receive a lower value.
For supporting_evidence and contradicting_evidence, copy only exact sentences
from the supplied abstract that support the respective direction. Return an
empty list when no such sentence is available; never paraphrase or infer one.
Do not invent citations, entities, mechanisms, study designs, or conclusions."""

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "using",
    "via",
    "was",
    "were",
    "with",
    "we",
    "our",
    "study",
}
_QUESTION_GENERIC_TOKENS = {
    "does",
    "do",
    "how",
    "what",
    "whether",
    "which",
    "who",
    "why",
    "direct",
    "directly",
    "effect",
    "effects",
    "role",
    "roles",
    "regulate",
    "regulates",
    "regulation",
    "activity",
    "evidence",
    "mechanism",
    "mechanisms",
    "paper",
    "papers",
    "research",
    "study",
    "studies",
    "cell",
    "cells",
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
    r"no association|not associated|not support|disputed)\b",
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
    (
        "randomized controlled trial",
        re.compile(r"\brandomi[sz]ed.*trial\b", re.IGNORECASE),
    ),
    ("cohort study", re.compile(r"\bcohort\b", re.IGNORECASE)),
    ("case-control study", re.compile(r"\bcase[- ]control\b", re.IGNORECASE)),
    ("in vitro study", re.compile(r"\bin vitro\b", re.IGNORECASE)),
    ("animal study", re.compile(r"\b(animal|mouse|mice|rat) model\b", re.IGNORECASE)),
)


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _string_list(value: Any, *, limit: int = 20) -> list[str]:
    values = (
        value if isinstance(value, list) else [value] if isinstance(value, str) else []
    )
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


def _stable_sentences(sentences: Sequence[str], *, limit: int = 3) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for sentence in sentences:
        text = _clean_text(sentence)
        key = normalize_text(text)
        if text and key and key not in seen:
            output.append(text)
            seen.add(key)
        if len(output) >= limit:
            break
    return output


def _question_focus_tokens(sub_question: str) -> set[str]:
    """Return question terms that should appear in direct supporting evidence.

    We deliberately exclude generic interrogative and scientific-process terms.
    This keeps a paper with an unrelated "however" from being marked as a
    contradiction merely because it happens to mention one broad topic word.
    """

    tokens = {
        token
        for token in tokenize(sub_question)
        if token not in _STOPWORDS and token not in _QUESTION_GENERIC_TOKENS
    }
    return tokens or set(tokenize(sub_question))


def _question_overlap(
    sub_question: str,
    *,
    title: str,
    abstract: str,
) -> float:
    focus = _question_focus_tokens(sub_question)
    if not focus:
        return 0.0
    title_tokens = set(tokenize(title))
    abstract_tokens = set(tokenize(abstract))
    title_coverage = len(focus & title_tokens) / len(focus)
    abstract_coverage = len(focus & abstract_tokens) / len(focus)
    return max(0.0, min(1.0, 0.65 * title_coverage + 0.35 * abstract_coverage))


def _sentence_is_question_anchored(
    sub_question: str,
    paper: PaperRecord,
    sentence: str,
) -> bool:
    focus = _question_focus_tokens(sub_question)
    if not focus:
        return False
    sentence_tokens = set(tokenize(sentence))
    overlap = len(focus & sentence_tokens)
    # Directional evidence must stand on its own as an abstract sentence. A
    # title can provide context for a reader, but must not turn a generic
    # "this result was confirmed" sentence into evidence for every entity in
    # that title. Short focused questions may have one essential anchor;
    # broader questions require two anchors in the sentence itself.
    required = 1 if len(focus) <= 2 else 2
    return overlap >= required


def _has_directional_evidence(
    sub_question: str,
    paper: PaperRecord,
    pattern: re.Pattern[str],
) -> bool:
    return bool(_directional_sentences(sub_question, paper, pattern))


def _directional_sentences(
    sub_question: str,
    paper: PaperRecord,
    pattern: re.Pattern[str],
) -> list[str]:
    return _stable_sentences(
        [
            sentence
            for sentence in _sentences(paper.abstract)
            if pattern.search(sentence)
            and _sentence_is_question_anchored(sub_question, paper, sentence)
        ]
    )


def _model_evidence_sentences(
    sub_question: str,
    paper: PaperRecord,
    value: Any,
) -> list[str]:
    """Resolve model-supplied evidence quotes back to exact abstract sentences.

    A structured response may shorten a sentence, but it must still be a
    literal substring of an anchored abstract sentence. This admits concise
    quotation while rejecting model-only paraphrases or fabricated claims.
    """

    abstract_sentences = _sentences(paper.abstract)
    matched: list[str] = []
    for proposed in _string_list(value, limit=6):
        normalized_proposed = normalize_text(proposed)
        if len(normalized_proposed) < 12:
            continue
        for sentence in abstract_sentences:
            normalized_sentence = normalize_text(sentence)
            if (
                normalized_proposed in normalized_sentence
                and _sentence_is_question_anchored(sub_question, paper, sentence)
            ):
                matched.append(sentence)
                break
    return _stable_sentences(matched)


def _grounded_directional_evidence(
    sub_question: str,
    paper: PaperRecord,
    value: Any,
    pattern: re.Pattern[str],
) -> list[str]:
    """Combine validated model quotations with deterministic fallback matches."""

    return _stable_sentences(
        [
            *_model_evidence_sentences(sub_question, paper, value),
            *_directional_sentences(sub_question, paper, pattern),
        ]
    )


def _coerce_score(value: object) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score if math.isfinite(score) else None


def _grounded_relevance(
    model_relevance: float | None,
    directness: float,
) -> float:
    """Keep LLM relevance useful without allowing ungrounded high scores.

    A model can recognise synonyms, so it is not discarded outright. Its score
    is instead capped by a deterministic lexical/directness floor. Completely
    off-topic papers therefore remain below the default coverage threshold.
    """

    grounding = max(0.0, min(1.0, directness))
    if model_relevance is None:
        return grounding
    return max(0.0, min(1.0, min(model_relevance, 0.45 + 0.55 * grounding)))


def _study_design(paper: PaperRecord) -> str:
    haystack = f"{paper.publication_type} {paper.title} {paper.abstract}"
    for label, pattern in _DESIGN_PATTERNS:
        if pattern.search(haystack):
            return label
    return _clean_text(paper.publication_type)


def _fallback_buckets(
    sub_question: str,
    paper: PaperRecord,
    *,
    current_year: int,
) -> set[EvidenceBucket]:
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
    if _has_directional_evidence(sub_question, paper, _CONTRADICT_PATTERN):
        buckets.add(EvidenceBucket.CONTRADICTING)
    if _has_directional_evidence(sub_question, paper, _SUPPORT_PATTERN):
        buckets.add(EvidenceBucket.SUPPORTING)
    return buckets


def _fallback_note(
    sub_question: str,
    paper: PaperRecord,
    *,
    current_year: int,
) -> ScoutNote:
    tokens = [
        token
        for token in tokenize(f"{paper.title} {paper.abstract}")
        if token not in _STOPWORDS
    ]
    counts = Counter(tokens)
    key_terms = [token for token, _ in counts.most_common(10)]
    entities = _string_list(
        re.findall(r"\b[A-Z][A-Z0-9+\-]{1,12}\b", f"{paper.title} {paper.abstract}"),
        limit=10,
    )
    sentences = _sentences(paper.abstract)
    mechanisms = [
        sentence for sentence in sentences if _MECHANISM_PATTERN.search(sentence)
    ][:3]
    supporting_evidence = _directional_sentences(
        sub_question,
        paper,
        _SUPPORT_PATTERN,
    )
    contradicting_evidence = _directional_sentences(
        sub_question,
        paper,
        _CONTRADICT_PATTERN,
    )
    summary = " ".join(sentences[:2]) if sentences else paper.title
    rank_relevance = paper.rank_scores.get("query_relevance", 0.0)
    legacy_relevance = max(
        lexical_relevance(sub_question, paper.title, paper.abstract),
        rank_relevance if math.isfinite(rank_relevance) else 0.0,
    )
    directness = _question_overlap(
        sub_question,
        title=paper.title,
        abstract=paper.abstract,
    )
    relevance = max(
        directness,
        min(legacy_relevance, 0.45 + 0.55 * directness),
    )
    return ScoutNote(
        paper_id=paper.paper_id,
        main_topic=paper.title,
        key_terms=key_terms,
        entities=entities,
        mechanisms=mechanisms,
        important_authors=list(paper.authors[:3]),
        controversies=contradicting_evidence,
        relevance_to_question=max(0.0, min(1.0, relevance)),
        directness_to_question=directness,
        supporting_evidence=supporting_evidence,
        contradicting_evidence=contradicting_evidence,
        evidence_buckets=_fallback_buckets(
            sub_question,
            paper,
            current_year=current_year,
        ),
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
                            "mechanisms": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "important_authors": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "controversies": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "candidate_citations": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "relevance_to_question": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                            },
                            "directness_to_question": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                            },
                            "supporting_evidence": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "contradicting_evidence": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "evidence_buckets": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": [bucket.value for bucket in EvidenceBucket],
                                },
                            },
                            "study_design": {"type": "string"},
                            "evidence_summary": {"type": "string"},
                        },
                        "required": [
                            "paper_id",
                            "relevance_to_question",
                            "evidence_buckets",
                        ],
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
            model_buckets: set[EvidenceBucket] = set()
            for value in raw.get("evidence_buckets", []):
                try:
                    model_buckets.add(EvidenceBucket(str(value)))
                except ValueError:
                    continue
            fallback = _fallback_note(
                sub_question, paper, current_year=self.current_year
            )
            fallback_buckets = _fallback_buckets(
                sub_question,
                paper,
                current_year=self.current_year,
            )
            supporting_evidence = _grounded_directional_evidence(
                sub_question,
                paper,
                raw.get("supporting_evidence"),
                _SUPPORT_PATTERN,
            )
            contradicting_evidence = _grounded_directional_evidence(
                sub_question,
                paper,
                raw.get("contradicting_evidence"),
                _CONTRADICT_PATTERN,
            )
            # Directional labels are materially stronger than metadata labels.
            # Keep them only when we can retain an exact abstract sentence
            # anchored to the asserted direction and the actual sub-question.
            buckets = {
                bucket
                for bucket in model_buckets | fallback_buckets
                if bucket
                not in {EvidenceBucket.SUPPORTING, EvidenceBucket.CONTRADICTING}
            }
            if supporting_evidence:
                buckets.add(EvidenceBucket.SUPPORTING)
            if contradicting_evidence:
                buckets.add(EvidenceBucket.CONTRADICTING)
            model_relevance = _coerce_score(raw.get("relevance_to_question"))
            deterministic_directness = _question_overlap(
                sub_question,
                title=paper.title,
                abstract=paper.abstract,
            )
            model_directness = _coerce_score(raw.get("directness_to_question"))
            directness = deterministic_directness
            if model_directness is not None:
                # Model-only directness must not convert an unanchored paper
                # into direct evidence. It can only provide a small allowance
                # for lexical variants absent from the title and abstract.
                directness = max(
                    deterministic_directness,
                    min(max(0.0, min(1.0, model_directness)), 0.15),
                )
            relevance = _grounded_relevance(
                model_relevance,
                directness,
            )
            parsed[paper_id] = ScoutNote(
                paper_id=paper_id,
                main_topic=_clean_text(raw.get("main_topic")) or fallback.main_topic,
                key_terms=_string_list(raw.get("key_terms")) or fallback.key_terms,
                entities=_string_list(raw.get("entities")) or fallback.entities,
                mechanisms=_string_list(raw.get("mechanisms")) or fallback.mechanisms,
                important_authors=_string_list(raw.get("important_authors"))
                or fallback.important_authors,
                controversies=_stable_sentences(
                    [
                        *contradicting_evidence,
                        *_model_evidence_sentences(
                            sub_question,
                            paper,
                            raw.get("controversies"),
                        ),
                    ]
                ),
                candidate_citations=_string_list(raw.get("candidate_citations")),
                relevance_to_question=relevance,
                directness_to_question=directness,
                supporting_evidence=supporting_evidence,
                contradicting_evidence=contradicting_evidence,
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
