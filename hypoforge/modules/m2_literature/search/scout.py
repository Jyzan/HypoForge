"""Batched, evidence-grounded semantic screening of paper abstracts."""

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

_RELATIONS = {
    "supports",
    "contradicts",
    "mixed",
    "insufficient",
    "not_applicable",
}
_STUDY_TYPES = {
    "experimental",
    "observational",
    "review",
    "method",
    "protocol",
    "computational",
    "other",
}
_METHODOLOGICAL_TYPES = {"method", "protocol", "computational"}
SCOUT_PROMPT_VERSION = "semantic-grounding-v1"
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
    "is", "it", "of", "on", "or", "that", "the", "this", "to", "using",
    "via", "was", "were", "with", "we", "our", "study",
}
_SYSTEM_PROMPT = """\
You screen scientific papers using only the supplied title, metadata, and
numbered abstract sentences. Judge the semantic relationship between each
paper and the research sub-question; do not classify by cue words such as
"however", "significantly", "showed", or "did not".

Use exactly one relation:
- supports: the abstract directly provides evidence for the questioned claim;
- contradicts: the abstract directly provides evidence against it;
- mixed: separate abstract sentences provide evidence in both directions;
- insufficient: the paper is relevant but the abstract cannot establish a
  direction, including when it addresses only one component of a broader
  question, provides a useful method, review, background, or adjacent result,
  or says that the exact relation was not tested;
- not_applicable: the paper concerns a clearly different research object or
  scientific domain and cannot contribute even background or methodological
  evidence to the sub-question.

Scientific questions are often novel, so a useful paper is not required to
answer the whole sub-question. Low directness alone is never sufficient for
`not_applicable`; use `insufficient` for relevant partial or contextual work.

Every supports/contradicts judgment must cite the corresponding supplied
sentence IDs. Never invent or rewrite sentence IDs. Return one result per
paper, preserve paper_id exactly, and do not claim to have read full text.

Score anchors:
- relevance 0.0-0.2: different object/domain or only shares generic vocabulary;
  0.5: same broad topic but does not answer the requested relationship;
  0.8-1.0: directly studies the required object and relationship.
- directness 0.0-0.2: background mention only; 0.5: relevant proxy/partial
  result; 0.8-1.0: a method, experiment, or conclusion sentence directly
  answers the sub-question. Directness above 0.7 requires at least one valid
  supporting or contradicting sentence ID. A paper that evaluates only one
  component (for example environmental noise) cannot be scored as a complete
  method for a broader task (for example sim-to-real transfer).

Required task entities/domains anchor relevance, but a paper may still be
useful when it studies a named component, method family, measurement, or
closely adjacent object needed to solve the question. A paper about a clearly
different object and domain is not_applicable even if it shares only generic
terms such as adaptation, transfer, stress, or simulation."""


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




def _valid_entity(value: object) -> bool:
    """Defensively accept only short noun-phrase-like entity surfaces."""

    text = _clean_text(value)
    if not text or len(text) > 100 or len(text.split()) > 8:
        return False
    if (
        text.count("(") != text.count(")")
        or text.count("（") != text.count("）")
    ):
        return False
    if text.endswith((".", "。", "?", "？", "!", "！", ":", "：", ";", "；")):
        return False
    if re.search(r"[.!?。！？](?:\s|$)", text):
        return False
    if re.search(r"[,，;；:：]", text):
        return False
    if re.match(
        r"(?i)^(this|that|these|those|the|a|an|we|our|it|paper|study|this paper|this study|results)",
        text,
    ) and len(text.split()) >= 4:
        return False
    return True


def _validated_entities(value: Any, fallback: Sequence[str] = ()) -> list[str]:
    """Filter LLM entities and fall back to the first valid candidate."""

    candidates = _string_list(value, limit=20)
    valid = [item for item in candidates if _valid_entity(item)]
    if valid:
        return valid
    return [item for item in _string_list(list(fallback), limit=20) if _valid_entity(item)]

def _sentences(text: str) -> list[str]:
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?。！？])\s+", _clean_text(text))
        if sentence.strip()
    ]


def _stable_sentences(values: Sequence[str], *, limit: int = 3) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _clean_text(value)
        key = normalize_text(text)
        if text and key and key not in seen:
            output.append(text)
            seen.add(key)
        if len(output) >= limit:
            break
    return output


def _sentence_catalog(
    paper: PaperRecord,
    *,
    char_limit: int | None = None,
) -> list[dict[str, str]]:
    catalog: list[dict[str, str]] = []
    used = 0
    for index, sentence in enumerate(_sentences(paper.abstract), start=1):
        extra = len(sentence) + (1 if catalog else 0)
        if char_limit is not None and used + extra > char_limit:
            break
        catalog.append(
            {
                "sentence_id": f"{paper.paper_id}:S{index}",
                "text": sentence,
            }
        )
        used += extra
    return catalog


def _sentences_from_ids(paper: PaperRecord, value: Any) -> list[str]:
    lookup = {
        item["sentence_id"]: item["text"]
        for item in _sentence_catalog(paper)
    }
    return _stable_sentences(
        [
            lookup[sentence_id]
            for sentence_id in _string_list(value, limit=8)
            if sentence_id in lookup
        ]
    )


def _legacy_evidence_sentences(paper: PaperRecord, value: Any) -> list[str]:
    """Validate legacy quote responses without making a semantic judgment."""

    abstract_sentences = _sentences(paper.abstract)
    matched: list[str] = []
    for proposed in _string_list(value, limit=8):
        normalized = normalize_text(proposed)
        if len(normalized) < 12:
            continue
        for sentence in abstract_sentences:
            if normalized == normalize_text(sentence):
                matched.append(sentence)
                break
    return _stable_sentences(matched)


def _coerce_score(value: object, default: float) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(score):
        return default
    return max(0.0, min(1.0, score))


def _local_terms(paper: PaperRecord) -> list[str]:
    tokens = [
        token
        for token in tokenize(f"{paper.title} {paper.abstract}")
        if token not in _STOPWORDS
    ]
    return [token for token, _ in Counter(tokens).most_common(10)]


def _local_entities(paper: PaperRecord) -> list[str]:
    return _string_list(
        re.findall(
            r"\b[A-Z][A-Z0-9+\-]{1,12}\b",
            f"{paper.title} {paper.abstract}",
        ),
        limit=10,
    )


def _metadata_buckets(
    paper: PaperRecord,
    *,
    current_year: int,
) -> set[EvidenceBucket]:
    buckets: set[EvidenceBucket] = set()
    if paper.year is not None and paper.year >= current_year - 3:
        buckets.add(EvidenceBucket.RECENT)
    if (
        paper.year is not None
        and paper.year <= current_year - 10
        and (paper.citation_count or 0) >= 100
    ):
        buckets.add(EvidenceBucket.CLASSIC)
    return buckets


def _fallback_note(
    sub_question: str,
    paper: PaperRecord,
    *,
    current_year: int,
) -> ScoutNote:
    """Return a conservative note without guessing scientific semantics."""

    sentences = _sentences(paper.abstract)
    lexical = lexical_relevance(sub_question, paper.title, paper.abstract)
    # Reserve <= 0.20 for an explicit LLM ``not_applicable`` judgment so
    # downstream ranking can distinguish failure/insufficiency without adding
    # a required field to the public ScoutNote contract.
    conservative_relevance = min(0.49, max(0.21, lexical))
    return ScoutNote(
        paper_id=paper.paper_id,
        main_topic=paper.title,
        key_terms=_local_terms(paper),
        entities=_local_entities(paper),
        important_authors=list(paper.authors[:3]),
        relevance_to_question=conservative_relevance,
        directness_to_question=min(0.35, conservative_relevance),
        evidence_buckets=_metadata_buckets(
            paper,
            current_year=current_year,
        ),
        study_design="",
        evidence_summary=(" ".join(sentences[:2]) if sentences else paper.title)[:600],
    )


class ScoutReader(ScoutReaderProtocol):
    """Use one batched LLM judgment as the semantic source for downstream tools."""

    tool_name = "scout_reader"
    prompt_version = SCOUT_PROMPT_VERSION

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
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.warning(
                        "Scout batch failed; using conservative fallback (%s: %s)",
                        type(exc).__name__,
                        exc,
                    )
                    logger.debug(
                        "Scout batch failure traceback",
                        exc_info=True,
                    )
                    return [
                        _fallback_note(
                            sub_question,
                            paper,
                            current_year=self.current_year,
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
                "sentences": _sentence_catalog(
                    paper,
                    char_limit=self.abstract_char_limit,
                ),
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
                            "relevance": {"type": "number", "minimum": 0, "maximum": 1},
                            "directness": {"type": "number", "minimum": 0, "maximum": 1},
                            "relation": {
                                "type": "string",
                                "enum": sorted(_RELATIONS),
                            },
                            "supporting_sentence_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "contradicting_sentence_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "study_type": {
                                "type": "string",
                                "enum": sorted(_STUDY_TYPES),
                            },
                            "mechanisms": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "entities": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "limitations": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "reason": {"type": "string"},
                        },
                        "required": [
                            "paper_id",
                            "relevance",
                            "directness",
                            "relation",
                            "supporting_sentence_ids",
                            "contradicting_sentence_ids",
                            "study_type",
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

            fallback = _fallback_note(
                sub_question,
                paper,
                current_year=self.current_year,
            )
            required_fields = {
                "relevance",
                "directness",
                "relation",
                "supporting_sentence_ids",
                "contradicting_sentence_ids",
                "study_type",
            }
            relation = _clean_text(raw.get("relation")).casefold()
            study_type = _clean_text(raw.get("study_type")).casefold()
            if (
                not required_fields.issubset(raw)
                or relation not in _RELATIONS
                or study_type not in _STUDY_TYPES
            ):
                parsed[paper_id] = fallback
                continue

            supporting = _sentences_from_ids(
                paper,
                raw.get("supporting_sentence_ids"),
            )
            contradicting = _sentences_from_ids(
                paper,
                raw.get("contradicting_sentence_ids"),
            )
            # Backward-compatible validation for callers returning exact quotes.
            if not supporting:
                supporting = _legacy_evidence_sentences(
                    paper,
                    raw.get("supporting_evidence"),
                )
            if not contradicting:
                contradicting = _legacy_evidence_sentences(
                    paper,
                    raw.get("contradicting_evidence"),
                )
            if relation == "supports" and not supporting:
                relation = "insufficient"
            elif relation == "contradicts" and not contradicting:
                relation = "insufficient"
            elif relation == "mixed" and not (supporting and contradicting):
                relation = "insufficient"

            buckets = _metadata_buckets(
                paper,
                current_year=self.current_year,
            )
            if study_type == "review":
                buckets.add(EvidenceBucket.REVIEW)
            if study_type in _METHODOLOGICAL_TYPES:
                buckets.add(EvidenceBucket.METHODOLOGICAL)
            if relation in {"supports", "mixed"}:
                buckets.add(EvidenceBucket.SUPPORTING)
            if relation in {"contradicts", "mixed"}:
                buckets.add(EvidenceBucket.CONTRADICTING)

            relevance = _coerce_score(
                raw.get("relevance", raw.get("relevance_to_question")),
                fallback.relevance_to_question,
            )
            directness = _coerce_score(
                raw.get("directness", raw.get("directness_to_question")),
                fallback.directness_to_question or 0.0,
            )
            if directness > 0.7 and not (supporting or contradicting):
                directness = 0.7
            if relation == "not_applicable":
                relevance = min(relevance, 0.20)
                directness = min(directness, 0.20)
            elif relation == "insufficient":
                relevance = max(relevance, 0.21)

            parsed[paper_id] = ScoutNote(
                paper_id=paper_id,
                main_topic=paper.title,
                key_terms=fallback.key_terms,
                entities=_validated_entities(raw.get("entities"), fallback.entities),
                mechanisms=_string_list(raw.get("mechanisms")),
                important_authors=list(paper.authors[:3]),
                controversies=list(contradicting),
                candidate_citations=[],
                relevance_to_question=relevance,
                directness_to_question=directness,
                supporting_evidence=supporting,
                contradicting_evidence=contradicting,
                evidence_buckets=buckets,
                study_design=study_type,
                evidence_summary=fallback.evidence_summary,
            )

        return [
            parsed.get(paper.paper_id)
            or _fallback_note(
                sub_question,
                paper,
                current_year=self.current_year,
            )
            for paper in papers
        ]
