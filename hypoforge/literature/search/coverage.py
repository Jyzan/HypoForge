"""Evidence-matrix coverage assessment over grounded Scout judgments."""

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
from ._text import normalize_text, tokenize


logger = logging.getLogger(__name__)

COVERAGE_PROMPT_VERSION = "evidence-matrix-v5"
_SYSTEM_PROMPT = """\
You assess whether a set of title-and-abstract Scout judgments covers the
scientific sub-question. Build a compact evidence matrix. A covered facet must
cite supplied paper IDs, and any directional claim must cite supplied sentence
IDs. Do not invent papers, sentences, findings, or full-text details.

Derive necessary facets from the exact current sub-question only. Key entities
and domains are background context, not a checklist: an entity that is not
required by the sub-question must not become a missing facet. Do not import
sibling questions such as post-translational modifications or co-chaperones
unless the current sub-question explicitly asks for them.

Coverage is existential and may be distributed across papers: no single paper
must cover every facet. Adding contextual or irrelevant papers cannot make a
previously supported necessary facet missing unless valid contradictory
evidence is supplied. Do not demand quantitative kinetics, a particular assay,
or a specific regulator unless the exact sub-question requires it.

Treat terms introduced by "e.g.", "for example", "such as", or "例如" as
illustrative alternatives, not a checklist. For a "which factors" question,
grounded identification of one or more valid factors and their relevant
functional effect can be sufficient; exhaustive coverage of every example or
factor class is not required. Evidence about an ATPase domain, nucleotide
binding/exchange, or ATP-dependent function can establish modulation of an
ATPase cycle without a numeric hydrolysis-rate measurement.

Return at most three missing topics. A missing topic must be indispensable to
the question, not merely a desirable evidence category. Preserve stable topics
from the previous round when they remain unresolved."""
_SCIENTIFIC_GAP_TERMS = {
    "assay",
    "cohort",
    "effect",
    "mechanism",
    "entity",
    "evidence",
    "mechanism",
    "method",
    "measurement",
    "outcome",
    "pathway",
    "phenomenon",
    "phenotype",
    "population",
    "relation",
    "relationship",
    "result",
    "target",
    "机制",
    "方法",
    "结果",
    "实体",
    "效应",
    "表型",
    "测量",
}
_DIRECTIONAL_FACET_TERMS = {
    "against",
    "contradict",
    "effect",
    "negative",
    "opposing",
    "positive",
    "refute",
    "relation",
    "relationship",
    "support",
    "反对",
    "反驳",
    "支持",
    "效应",
    "关系",
    "机制",
}


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


def _valid_missing_topic(
    topic: str,
    *,
    context_tokens: set[str],
    previous_topics: set[str],
) -> bool:
    normalized = normalize_text(topic)
    if not normalized:
        return False
    if normalized in previous_topics:
        return True
    return bool(
        set(tokenize(topic)) & (context_tokens | _SCIENTIFIC_GAP_TERMS)
    )


def _directional_facet(facet: str) -> bool:
    tokens = set(tokenize(facet))
    return bool(
        tokens & _DIRECTIONAL_FACET_TERMS
        or any(term in normalize_text(facet) for term in _DIRECTIONAL_FACET_TERMS)
    )


def _question_context_entities(
    sub_question: str,
    entities: Sequence[str],
) -> list[str]:
    question = normalize_text(sub_question)
    question_tokens = set(tokenize(sub_question))
    selected: list[str] = []
    for entity in entities:
        normalized = normalize_text(entity)
        entity_tokens = set(tokenize(entity))
        acronym = "".join(
            token[0]
            for token in entity_tokens
            if token not in {"and", "of", "the"}
        )
        if (
            normalized in question
            or bool(entity_tokens & question_tokens)
            or acronym
            and (
                acronym in question_tokens
                or f"{acronym}s" in question_tokens
            )
        ):
            selected.append(entity)
    return selected


def _abstract_sentences(paper: PaperRecord) -> list[str]:
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?。！？])\s+", _clean_text(paper.abstract))
        if sentence.strip()
    ]


def _sentence_lookup(papers: Sequence[PaperRecord]) -> dict[str, str]:
    return {
        f"{paper.paper_id}:S{index}": sentence
        for paper in papers
        for index, sentence in enumerate(_abstract_sentences(paper), start=1)
    }


def _evidence_sentence_ids(
    note: ScoutNote,
    paper: PaperRecord,
    evidence: Sequence[str],
) -> list[str]:
    normalized_evidence = {normalize_text(item) for item in evidence if item}
    return [
        f"{paper.paper_id}:S{index}"
        for index, sentence in enumerate(_abstract_sentences(paper), start=1)
        if normalize_text(sentence) in normalized_evidence
    ]


def _validated_buckets(
    note: ScoutNote,
    paper: PaperRecord,
    *,
    current_year: int,
) -> tuple[set[EvidenceBucket], list[str], list[str]]:
    buckets = {
        bucket
        for bucket in note.evidence_buckets
        if bucket not in {EvidenceBucket.SUPPORTING, EvidenceBucket.CONTRADICTING}
    }
    supporting_ids = _evidence_sentence_ids(
        note,
        paper,
        note.supporting_evidence,
    )
    contradicting_ids = _evidence_sentence_ids(
        note,
        paper,
        note.contradicting_evidence,
    )
    if supporting_ids:
        buckets.add(EvidenceBucket.SUPPORTING)
    if contradicting_ids:
        buckets.add(EvidenceBucket.CONTRADICTING)
    if paper.year is not None and paper.year >= current_year - 3:
        buckets.add(EvidenceBucket.RECENT)
    if (
        paper.year is not None
        and paper.year <= current_year - 10
        and (paper.citation_count or 0) >= 100
    ):
        buckets.add(EvidenceBucket.CLASSIC)
    return buckets, supporting_ids, contradicting_ids


class CoverageEvaluator(CoverageEvaluatorProtocol):
    """Combine a semantic evidence matrix with deterministic safety gates."""

    tool_name = "coverage_evaluator"
    prompt_version = COVERAGE_PROMPT_VERSION

    def __init__(
        self,
        client: Any | None = None,
        *,
        relevance_threshold: float = 0.55,
        min_relevant_papers: int = 3,
        min_bucket_count: int = 2,
        selection_limit: int | None = None,
        max_tokens: int = 2048,
        current_year: int | None = None,
    ) -> None:
        if not 0.0 <= relevance_threshold <= 1.0:
            raise ValueError("relevance_threshold must be between 0 and 1")
        if min_relevant_papers <= 0:
            raise ValueError("min_relevant_papers must be positive")
        if not 1 <= min_bucket_count <= len(EvidenceBucket):
            raise ValueError("min_bucket_count must fit the evidence bucket set")
        if selection_limit is not None and selection_limit <= 0:
            raise ValueError("selection_limit must be positive when provided")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        self.client = client
        self.relevance_threshold = relevance_threshold
        self.min_relevant_papers = min_relevant_papers
        self.min_bucket_count = min_bucket_count
        self.selection_limit = selection_limit
        self.max_tokens = max_tokens
        self.current_year = current_year or datetime.now(timezone.utc).year

    async def evaluate(
        self,
        sub_question: str,
        papers: Sequence[PaperRecord],
        scout_notes: Sequence[ScoutNote],
        state: SearchState,
    ) -> CoverageReport:
        selected = list(papers)
        if self.selection_limit is not None:
            selected = selected[: self.selection_limit]
        notes_by_id = {note.paper_id: note for note in scout_notes}
        relevant = [
            (notes_by_id[paper.paper_id], paper)
            for paper in selected
            if paper.paper_id in notes_by_id
            and notes_by_id[paper.paper_id].relevance_to_question
            >= self.relevance_threshold
        ]

        covered: set[EvidenceBucket] = set()
        supporting_papers: set[str] = set()
        directional_ids: set[str] = set()
        for note, paper in relevant:
            buckets, supporting_ids, contradicting_ids = _validated_buckets(
                note,
                paper,
                current_year=self.current_year,
            )
            covered.update(buckets)
            if supporting_ids:
                supporting_papers.add(paper.paper_id)
            directional_ids.update(supporting_ids)
            directional_ids.update(contradicting_ids)

        hard_gaps = self._profile_gaps(
            state=state,
            relevant=relevant,
            covered=covered,
            supporting_papers=supporting_papers,
            directional_ids=directional_ids,
        )
        covered_topics: list[str] = []
        model_missing: list[str] = []
        model_rationale = ""
        semantic_sufficient = False
        if relevant and self.client is not None:
            try:
                (
                    covered_topics,
                    model_missing,
                    model_rationale,
                    semantic_sufficient,
                ) = await self._assess_matrix(
                    sub_question=sub_question,
                    relevant=relevant,
                    state=state,
                )
            except Exception as exc:
                logger.warning(
                    "Coverage matrix failed; retaining conservative gaps (%s)",
                    type(exc).__name__,
                )

        missing_topics = _stable_strings(
            [*model_missing, *hard_gaps],
            limit=3,
        )
        hard_gate = not hard_gaps
        sufficient = semantic_sufficient and hard_gate
        if sufficient:
            missing_topics = []

        all_buckets = set(EvidenceBucket)
        summary = (
            f"{len(relevant)} relevant papers in the coverage selection; "
            f"{len(covered)}/{len(all_buckets)} evidence buckets covered."
        )
        rationale = f"{summary} {model_rationale}".strip()
        if not model_rationale:
            rationale = (
                f"{summary} Semantic coverage was not confirmed; "
                "search remains conservative."
            )
        return CoverageReport(
            covered_buckets=covered,
            missing_buckets=all_buckets - covered,
            covered_topics=covered_topics,
            missing_topics=missing_topics,
            sufficient=sufficient,
            rationale=rationale,
        )

    def _profile_gaps(
        self,
        *,
        state: SearchState,
        relevant: list[tuple[ScoutNote, PaperRecord]],
        covered: set[EvidenceBucket],
        supporting_papers: set[str],
        directional_ids: set[str],
    ) -> list[str]:
        gaps: list[str] = []
        question_type = state.question_type
        required_count = (
            2 if question_type == "phenomenon_discovery"
            else self.min_relevant_papers
        )
        if len(relevant) < required_count:
            gaps.append(
                f"additional relevant studies ({required_count - len(relevant)} more needed)"
            )

        direct = any(
            (note.directness_to_question or 0.0) >= self.relevance_threshold
            and (
                note.study_design in {"experimental", "observational"}
                or question_type == "method_development"
                and note.study_design in {"method", "protocol", "computational"}
            )
            for note, _ in relevant
        )
        if question_type == "method_development":
            if EvidenceBucket.METHODOLOGICAL not in covered:
                gaps.append("direct methodological evidence")
            if not direct:
                gaps.append("a method directly addressing the sub-question")
        elif question_type == "phenomenon_discovery":
            if not supporting_papers:
                gaps.append("evidence supporting the reported phenomenon")
        else:
            if not supporting_papers:
                gaps.append("evidence supporting the proposed mechanism")
            if not direct or not directional_ids:
                gaps.append("direct mechanism evidence")

        return _stable_strings(gaps)

    async def _assess_matrix(
        self,
        *,
        sub_question: str,
        relevant: list[tuple[ScoutNote, PaperRecord]],
        state: SearchState,
    ) -> tuple[list[str], list[str], str, bool]:
        context_entities = _question_context_entities(
            sub_question,
            sorted(state.key_entities),
        )
        sentence_lookup = _sentence_lookup([paper for _, paper in relevant])
        valid_paper_ids = {paper.paper_id for _, paper in relevant}
        payload: list[dict[str, Any]] = []
        valid_sentence_ids: set[str] = set()
        for note, paper in relevant:
            supporting_ids = _evidence_sentence_ids(
                note,
                paper,
                note.supporting_evidence,
            )
            contradicting_ids = _evidence_sentence_ids(
                note,
                paper,
                note.contradicting_evidence,
            )
            valid_sentence_ids.update(supporting_ids)
            valid_sentence_ids.update(contradicting_ids)
            evidence_sentences = [
                {
                    "sentence_id": sentence_id,
                    "direction": (
                        "supporting"
                        if sentence_id in supporting_ids
                        else "contradicting"
                    ),
                    "text": sentence_lookup[sentence_id],
                }
                for sentence_id in [*supporting_ids, *contradicting_ids]
                if sentence_id in sentence_lookup
            ]
            payload.append({
                "paper_id": paper.paper_id,
                "title": paper.title,
                "relevance": note.relevance_to_question,
                "directness": note.directness_to_question,
                "study_type": note.study_design,
                "mechanisms": note.mechanisms,
                "supporting_sentence_ids": supporting_ids,
                "contradicting_sentence_ids": contradicting_ids,
                "evidence_sentences": evidence_sentences,
                "evidence_summary": note.evidence_summary,
            })
        schema = {
            "type": "object",
            "properties": {
                "facets": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "facet": {"type": "string"},
                            "status": {
                                "type": "string",
                                "enum": ["covered", "partial", "missing"],
                            },
                            "paper_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "sentence_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": [
                            "facet",
                            "status",
                            "paper_ids",
                            "sentence_ids",
                        ],
                    },
                },
                "missing_topics": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 3,
                },
                "sufficient": {"type": "boolean"},
                "rationale": {"type": "string"},
            },
            "required": ["facets", "missing_topics", "sufficient", "rationale"],
        }
        result = await self.client.structured_chat(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=(
                f"Research sub-question: {sub_question}\n"
                f"Question type: {state.question_type or 'unknown'}\n"
                f"Key entities present in this sub-question: {context_entities}\n"
                f"Domains: {sorted(state.domains)}\n"
                f"Previous missing topics: {sorted(state.missing_topics)}\n\n"
                f"Grounded Scout judgments:\n{json.dumps(payload, ensure_ascii=False)}"
            ),
            output_schema=schema,
            max_tokens=self.max_tokens,
            temperature=0.0,
            disable_thinking=True,
        )
        if not isinstance(result, dict):
            return [], [], "", False

        validated_facets: list[tuple[str, str]] = []
        for raw in result.get("facets", []):
            if not isinstance(raw, dict):
                continue
            facet = _clean_text(raw.get("facet"))
            status = _clean_text(raw.get("status")).casefold()
            if not facet or status not in {"covered", "partial", "missing"}:
                continue
            raw_paper_ids = _stable_strings(raw.get("paper_ids"))
            paper_ids = [
                paper_id
                for paper_id in raw_paper_ids
                if paper_id in valid_paper_ids
            ]
            raw_sentence_ids = _stable_strings(raw.get("sentence_ids"))
            sentence_ids = [
                sentence_id
                for sentence_id in raw_sentence_ids
                if sentence_id in sentence_lookup
                and sentence_id in valid_sentence_ids
            ]
            if status == "covered" and not paper_ids:
                status = "partial"
            if status == "covered" and raw_sentence_ids and not sentence_ids:
                status = "partial"
            if status == "covered" and _directional_facet(facet):
                cited_papers = {
                    sentence_id.rsplit(":S", 1)[0]
                    for sentence_id in sentence_ids
                }
                if not sentence_ids or not cited_papers.issubset(set(paper_ids)):
                    status = "partial"
            validated_facets.append((facet, status))

        covered_topics = [
            facet for facet, status in validated_facets if status == "covered"
        ]
        facet_gaps = [
            facet for facet, status in validated_facets if status != "covered"
        ]
        context_tokens = set(
            tokenize(
                " ".join(
                    [
                        sub_question,
                        *context_entities,
                        *state.domains,
                        *(
                            value
                            for note, _ in relevant
                            for value in [*note.entities, *note.mechanisms]
                        ),
                    ]
                )
            )
        )
        previous_topics = {
            normalize_text(topic) for topic in state.missing_topics if topic
        }
        proposed_gaps = [
            *_stable_strings(result.get("missing_topics"), limit=3),
            *facet_gaps,
        ]
        missing_topics = _stable_strings(
            [
                topic
                for topic in proposed_gaps
                if _valid_missing_topic(
                    topic,
                    context_tokens=context_tokens,
                    previous_topics=previous_topics,
                )
            ],
            limit=3,
        )
        semantic_sufficient = bool(
            result.get("sufficient")
            and validated_facets
            and all(status == "covered" for _, status in validated_facets)
            and not missing_topics
        )
        rationale = _clean_text(result.get("rationale"))[:1000]
        return covered_topics, missing_topics, rationale, semantic_sufficient
