"""Recall a compact, diverse pool of claim pairs for semantic relation judging.

Adapted for Track B: uses canonical ``AtomicClaim.evidence_ids`` (pointing to
``M2EvidenceExport.evidence_id``) instead of the old ``evidence_record_ids``.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Dict, Iterable, List, Sequence, Set

from ...vocabulary import normalize_entity
from .models import AtomicClaim, EvidenceRecord, RelationPair


def _hash(*parts: str, length: int = 16) -> str:
    raw = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(raw.encode("utf-8", errors="ignore")).hexdigest()[:length]


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9][a-z0-9_.+-]*|[一-鿿]", (text or "").lower())


def _normalise_entity(value: str) -> str:
    # Reuse M3's central vocabulary so rule-graph deduplication and Track B
    # relation recall agree on entity identity.
    return normalize_entity(value)


def _normalise(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi <= lo:
        return [1.0 if hi > 0 else 0.0 for _ in values]
    return [(value - lo) / (hi - lo) for value in values]


def _bm25_scores(query: str, documents: Sequence[str]) -> List[float]:
    docs = [_tokens(document) for document in documents]
    query_tokens = _tokens(query)
    if not query_tokens or not docs:
        return [0.0] * len(docs)
    n_docs = len(docs)
    avg_len = sum(len(document) for document in docs) / max(n_docs, 1)
    document_frequency = Counter(
        token
        for token in set(query_tokens)
        for document in docs
        if token in set(document)
    )
    scores: List[float] = []
    for document in docs:
        frequencies = Counter(document)
        score = 0.0
        for token in query_tokens:
            frequency = frequencies[token]
            if not frequency:
                continue
            inverse_frequency = math.log(
                1
                + (n_docs - document_frequency[token] + 0.5)
                / (document_frequency[token] + 0.5)
            )
            denominator = frequency + 1.2 * (
                1 - 0.75 + 0.75 * len(document) / max(avg_len, 1)
            )
            score += inverse_frequency * (frequency * 2.2 / denominator)
        scores.append(score)
    return scores


class RelationCandidateRetriever:
    """Rule + dual-query BM25 recall with lightweight MMR diversification.

    Adapted for Track B: ``AtomicClaim.evidence_ids`` now references
    ``M2EvidenceExport.evidence_id`` directly.  The record map is keyed
    by ``EvidenceRecord.evidence_id`` for consistent lookups.
    """

    POSITIVE_TERMS = (
        "support confirm reproduce replicate increase improve extend consistent"
    )
    NEGATIVE_TERMS = (
        "contradict no effect ineffective reduce decrease fail limitation boundary"
    )

    def __init__(
        self,
        max_candidates_per_claim: int = 12,
        max_pairs_total: int = 60,
        cross_source_ratio: float = 0.5,
        mmr_lambda: float = 0.78,
    ):
        self.max_candidates_per_claim = max(1, max_candidates_per_claim)
        self.max_pairs_total = max(1, max_pairs_total)
        self.cross_source_ratio = min(1.0, max(0.0, cross_source_ratio))
        self.mmr_lambda = min(1.0, max(0.0, mmr_lambda))

    # ---- Lookup helpers (keyed by evidence_id for claim compatibility) ----

    @staticmethod
    def _record_map(
        records: Iterable[EvidenceRecord],
    ) -> Dict[str, EvidenceRecord]:
        """Build a lookup keyed by ``evidence_id`` (M2EvidenceExport ref)."""
        return {record.evidence_id: record for record in records}

    @staticmethod
    def _paper_ids(
        claim: AtomicClaim,
        record_map: Dict[str, EvidenceRecord],
    ) -> List[str]:
        return list(dict.fromkeys(
            record_map[eid].paper_id
            for eid in claim.evidence_ids
            if eid in record_map and record_map[eid].paper_id
        ))

    @staticmethod
    def _queries(
        claim: AtomicClaim,
        record_map: Dict[str, EvidenceRecord],
    ) -> List[str]:
        return list(dict.fromkeys(
            record_map[eid].query
            for eid in claim.evidence_ids
            if eid in record_map and record_map[eid].query
        ))

    @staticmethod
    def _entities(
        claim: AtomicClaim,
        record_map: Dict[str, EvidenceRecord],
    ) -> Dict[str, str]:
        values = list(claim.entities)
        for eid in claim.evidence_ids:
            if eid in record_map:
                values.extend(record_map[eid].entities)
        normalised: Dict[str, str] = {}
        for value in values:
            key = _normalise_entity(value)
            if key:
                normalised.setdefault(key, value)
        return normalised

    @staticmethod
    def _text(
        claim: AtomicClaim,
        record_map: Dict[str, EvidenceRecord],
    ) -> str:
        context_parts: List[str] = [
            claim.statement,
            " ".join(claim.entities),
        ]
        for eid in claim.evidence_ids:
            record = record_map.get(eid)
            if not record:
                continue
            context_parts.extend([
                record.query,
                record.summary,
                record.quote,
                record.normalized_claim,
                " ".join(record.methods),
                " ".join(record.limitations),
                " ".join(
                    str(value) for value in record.context.values() if value
                ),
            ])
        return " ".join(part for part in context_parts if part)

    @staticmethod
    def _jaccard(left: str, right: str) -> float:
        a, b = set(_tokens(left)), set(_tokens(right))
        return len(a & b) / max(len(a | b), 1)

    def recall(
        self,
        claims: List[AtomicClaim],
        records: List[EvidenceRecord],
    ) -> List[RelationPair]:
        if len(claims) < 2:
            return []

        record_map = self._record_map(records)
        texts = [self._text(claim, record_map) for claim in claims]
        entities = [
            self._entities(claim, record_map) for claim in claims
        ]
        papers = [
            self._paper_ids(claim, record_map) for claim in claims
        ]
        queries = [
            set(
                query.lower()
                for query in self._queries(claim, record_map)
            )
            for claim in claims
        ]

        pair_map: Dict[tuple[str, str], RelationPair] = {}
        for anchor_index, anchor in enumerate(claims):
            positive_query = (
                f"{anchor.statement} {' '.join(anchor.entities)} "
                f"{self.POSITIVE_TERMS}"
            )
            negative_query = (
                f"{anchor.statement} {' '.join(anchor.entities)} "
                f"{self.NEGATIVE_TERMS}"
            )
            positive_scores = _normalise(
                _bm25_scores(positive_query, texts)
            )
            negative_scores = _normalise(
                _bm25_scores(negative_query, texts)
            )

            ranked: List[tuple[float, int, List[str], List[str]]] = []
            for target_index, target in enumerate(claims):
                if target_index == anchor_index:
                    continue
                overlap_keys = sorted(
                    set(entities[anchor_index])
                    & set(entities[target_index])
                )
                overlap = [
                    entities[anchor_index][key] for key in overlap_keys
                ]
                cross_source = (
                    bool(papers[anchor_index] and papers[target_index])
                    and bool(
                        set(papers[anchor_index]).isdisjoint(
                            papers[target_index]
                        )
                    )
                )
                same_query = bool(
                    queries[anchor_index] & queries[target_index]
                )
                semantic_score = max(
                    positive_scores[target_index],
                    negative_scores[target_index],
                )
                lexical_similarity = self._jaccard(
                    texts[anchor_index], texts[target_index]
                )
                origins: List[str] = []
                if overlap:
                    origins.append("shared_entity")
                if cross_source:
                    origins.append("cross_source")
                if same_query:
                    origins.append("same_query")
                if positive_scores[target_index] >= 0.25:
                    origins.append("bm25_positive")
                if negative_scores[target_index] >= 0.25:
                    origins.append("bm25_negative")

                rule_score = min(1.0, 0.18 * len(overlap))
                score = min(
                    1.0,
                    0.50 * semantic_score
                    + 0.20 * lexical_similarity
                    + rule_score
                    + (0.08 if cross_source else 0.0)
                    + (0.07 if same_query else 0.0),
                )
                if (
                    not origins
                    or (not overlap and semantic_score < 0.18 and not same_query)
                ):
                    continue
                ranked.append(
                    (score, target_index, overlap, origins)
                )

            selected: List[
                tuple[float, int, List[str], List[str]]
            ] = []
            remaining = sorted(ranked, reverse=True)
            while (
                remaining
                and len(selected) < self.max_candidates_per_claim
            ):
                def mmr(
                    item: tuple[float, int, List[str], List[str]],
                ) -> float:
                    score, target_index, _, _ = item
                    redundancy = max(
                        (
                            self._jaccard(
                                texts[target_index],
                                texts[chosen[1]],
                            )
                            for chosen in selected
                        ),
                        default=0.0,
                    )
                    same_paper_penalty = (
                        0.10
                        if any(
                            set(papers[target_index])
                            & set(papers[chosen[1]])
                            for chosen in selected
                        )
                        else 0.0
                    )
                    return (
                        self.mmr_lambda * score
                        - (1 - self.mmr_lambda) * redundancy
                        - same_paper_penalty
                    )

                best = max(remaining, key=mmr)
                remaining.remove(best)
                selected.append(best)

            for score, target_index, overlap, origins in selected:
                target = claims[target_index]
                source_id, target_id = sorted((anchor.id, target.id))
                key = (source_id, target_id)
                pair = RelationPair(
                    id="PAIR_" + _hash(source_id, target_id),
                    source=source_id,
                    target=target_id,
                    retrieval_score=score,
                    entity_overlap=overlap,
                    candidate_origin=list(dict.fromkeys(origins)),
                    source_evidence_ids=(
                        anchor.evidence_ids
                        if source_id == anchor.id
                        else target.evidence_ids
                    ),
                    target_evidence_ids=(
                        target.evidence_ids
                        if target_id == target.id
                        else anchor.evidence_ids
                    ),
                    source_paper_ids=(
                        papers[anchor_index]
                        if source_id == anchor.id
                        else papers[target_index]
                    ),
                    target_paper_ids=(
                        papers[target_index]
                        if target_id == target.id
                        else papers[anchor_index]
                    ),
                )
                old = pair_map.get(key)
                if (
                    old is None
                    or pair.retrieval_score > old.retrieval_score
                ):
                    pair_map[key] = pair
                elif old:
                    old.candidate_origin = list(dict.fromkeys(
                        old.candidate_origin + pair.candidate_origin
                    ))
                    old.entity_overlap = list(dict.fromkeys(
                        old.entity_overlap + pair.entity_overlap
                    ))

        # EvidenceRecord → Claim recall (source_type="evidence_record").
        # This is the only candidate type allowed to become
        # SUPPORTS/CONTRADICTS/LIMITS edges.
        for record in records:
            record_entities = {
                key: value
                for value in record.entities
                if (key := _normalise_entity(value))
            }
            record_text = " ".join([
                record.query,
                record.summary,
                record.excerpt,
                record.quote,
                record.normalized_claim,
                " ".join(record.claims),
                " ".join(record.methods),
                " ".join(record.limitations),
                " ".join(
                    str(value)
                    for value in record.context.values()
                    if value
                ),
            ])
            for claim_index, claim in enumerate(claims):
                # Don't pair a claim with evidence it's already built from.
                if record.evidence_id in claim.evidence_ids:
                    continue
                overlap_keys = sorted(
                    set(record_entities) & set(entities[claim_index])
                )
                overlap = [
                    record_entities[key] for key in overlap_keys
                ]
                same_query = (
                    record.query.lower() in queries[claim_index]
                    if record.query
                    else False
                )
                cross_source = (
                    bool(record.paper_id and papers[claim_index])
                    and (record.paper_id not in papers[claim_index])
                )
                lexical_similarity = self._jaccard(
                    record_text, texts[claim_index]
                )
                origins = ["evidence_claim"]
                if overlap:
                    origins.append("shared_entity")
                if same_query:
                    origins.append("same_query")
                if cross_source:
                    origins.append("cross_source")
                if lexical_similarity >= 0.12:
                    origins.append("semantic_overlap")
                if (
                    not overlap
                    and not same_query
                    and lexical_similarity < 0.12
                ):
                    continue
                score = min(
                    1.0,
                    0.42 * lexical_similarity
                    + min(0.24, 0.10 * len(overlap))
                    + (0.10 if same_query else 0.0)
                    + (0.10 if cross_source else 0.0)
                    + 0.14 * (record.relevance_score / 10.0),
                )
                pair = RelationPair(
                    id="PAIR_"
                    + _hash("evidence_claim", record.evidence_id, claim.id),
                    source=record.evidence_id,
                    target=claim.id,
                    source_type="evidence_record",
                    target_type="claim",
                    retrieval_score=score,
                    entity_overlap=overlap,
                    candidate_origin=origins,
                    source_evidence_ids=[record.evidence_id],
                    target_evidence_ids=claim.evidence_ids,
                    source_paper_ids=(
                        [record.paper_id] if record.paper_id else []
                    ),
                    target_paper_ids=papers[claim_index],
                )
                pair_map[(pair.source, pair.target)] = pair

        ranked_pairs = sorted(
            pair_map.values(),
            key=lambda pair: (-pair.retrieval_score, pair.id),
        )
        evidence_claim_pairs = [
            pair
            for pair in ranked_pairs
            if pair.source_type == "evidence_record"
        ]
        cross_source_pairs = [
            pair
            for pair in ranked_pairs
            if pair.source_paper_ids
            and pair.target_paper_ids
            and set(pair.source_paper_ids).isdisjoint(
                pair.target_paper_ids
            )
        ]
        evidence_quota = min(
            len(evidence_claim_pairs),
            math.ceil(self.max_pairs_total * 0.5),
        )
        selected = evidence_claim_pairs[:evidence_quota]
        selected_ids = {pair.id for pair in selected}
        cross_source_quota = min(
            len(cross_source_pairs),
            math.ceil(self.max_pairs_total * self.cross_source_ratio),
        )
        for pair in cross_source_pairs:
            if (
                sum(
                    bool(
                        item.source_paper_ids
                        and item.target_paper_ids
                    )
                    and set(item.source_paper_ids).isdisjoint(
                        item.target_paper_ids
                    )
                    for item in selected
                )
                >= cross_source_quota
            ):
                break
            if pair.id not in selected_ids:
                selected.append(pair)
                selected_ids.add(pair.id)
        selected.extend(
            pair
            for pair in ranked_pairs
            if pair.id not in selected_ids
        )
        return selected[: self.max_pairs_total]
