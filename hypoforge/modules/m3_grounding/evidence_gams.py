"""Graph-augmented Monte Carlo search over judged M3 relation candidates."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

from .models import RelationCandidate


@dataclass
class _StateStats:
    visits: int = 0
    value_sum: float = 0.0
    reward: float = 0.0
    components: Dict[str, float] = field(default_factory=dict)

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visits if self.visits else self.reward


class EvidenceGraphGAMS:
    """Search candidate-edge subsets using four graph-aware mutation operators."""

    OPERATORS = (
        "primary_expansion",
        "intra_branch_evolution",
        "cross_branch_reference",
        "multi_branch_aggregation",
    )

    def __init__(
        self,
        minimum_confidence: float = 0.35,
        exploration_weight: float = 0.35,
        rollout_depth: int = 3,
        random_seed: int = 42,
    ):
        self.minimum_confidence = min(1.0, max(0.0, minimum_confidence))
        self.exploration_weight = max(0.0, exploration_weight)
        self.rollout_depth = max(0, rollout_depth)
        self.random = random.Random(random_seed)
        self.random_seed = random_seed

    @staticmethod
    def _same_pair(left: RelationCandidate, right: RelationCandidate) -> bool:
        return {left.source, left.target} == {right.source, right.target}

    def _logical_penalty(self, selected: Sequence[RelationCandidate]) -> float:
        penalty = 0.0
        for index, left in enumerate(selected):
            for right in selected[index + 1:]:
                if not self._same_pair(left, right):
                    continue
                relations = {left.relation, right.relation}
                if relations == {"supports", "contradicts"}:
                    comparability = max(left.condition_comparability, right.condition_comparability)
                    penalty += 0.16 + 0.18 * comparability
                elif left.relation != right.relation and relations <= {"same_as", "contradicts", "refines"}:
                    penalty += 0.10
        return min(penalty, 0.70)

    def score(
        self,
        candidates: Sequence[RelationCandidate],
        indices: Iterable[int],
    ) -> Tuple[float, Dict[str, float]]:
        selected_indices = frozenset(indices)
        selected = [candidates[index] for index in selected_indices]
        if not selected:
            return 0.0, {
                "confidence": 0.0,
                "coverage": 0.0,
                "provenance": 0.0,
                "cross_source": 0.0,
                "comparability": 0.0,
                "retrieval_quality": 0.0,
                "credible_edge_recall": 0.0,
                "low_confidence_penalty": 0.0,
                "logical_penalty": 0.0,
                "semantic_mismatch_penalty": 0.0,
                "source_dependence_penalty": 0.0,
                "over_density_penalty": 0.0,
            }

        confidence = sum(candidate.confidence for candidate in selected) / len(selected)
        possible_nodes = {endpoint for candidate in candidates for endpoint in (candidate.source, candidate.target)}
        covered_nodes = {endpoint for candidate in selected for endpoint in (candidate.source, candidate.target)}
        coverage = len(covered_nodes) / max(len(possible_nodes), 1)
        provenance = sum(bool(candidate.evidence_ids) for candidate in selected) / len(selected)
        cross_source = sum(
            bool(candidate.source_paper_ids)
            and bool(candidate.target_paper_ids)
            and set(candidate.source_paper_ids).isdisjoint(candidate.target_paper_ids)
            for candidate in selected
        ) / len(selected)
        conditional = [
            candidate.condition_comparability
            for candidate in selected
            if candidate.relation in {"contradicts", "limits", "supports"}
        ]
        comparability = sum(conditional) / len(conditional) if conditional else 1.0
        retrieval_quality = sum(candidate.retrieval_score for candidate in selected) / len(selected)
        credible_indices = {
            index for index, candidate in enumerate(candidates)
            if candidate.confidence >= self.minimum_confidence
            and bool(candidate.evidence_ids)
            and candidate.retrieval_score >= 0.45
            and (
                candidate.relation not in {"supports", "contradicts"}
                or candidate.condition_comparability >= 0.80
            )
        }
        credible_edge_recall = len(selected_indices & credible_indices) / max(len(credible_indices), 1)

        low_confidence_penalty = sum(
            max(0.0, self.minimum_confidence - candidate.confidence)
            for candidate in selected
        ) / len(selected)
        logical_penalty = self._logical_penalty(selected)
        semantic_mismatch_penalty = sum(
            max(0.0, 0.80 - candidate.condition_comparability)
            for candidate in selected
            if candidate.relation in {"supports", "contradicts"}
        ) * 1.20
        source_dependence_penalty = sum(
            bool(candidate.source_paper_ids)
            and bool(candidate.target_paper_ids)
            and not set(candidate.source_paper_ids).isdisjoint(candidate.target_paper_ids)
            for candidate in selected
        ) / len(selected) * 0.08
        density = len(selected) / max(len(candidates), 1)
        over_density_penalty = max(0.0, density - 0.62) * 0.20

        reward = (
            0.32 * confidence
            + 0.06 * coverage
            + 0.16 * provenance
            + 0.08 * cross_source
            + 0.18 * comparability
            + 0.14 * retrieval_quality
            + 0.06 * credible_edge_recall
            - 0.45 * low_confidence_penalty
            - logical_penalty
            - semantic_mismatch_penalty
            - source_dependence_penalty
            - over_density_penalty
        )
        components = {
            "confidence": confidence,
            "coverage": coverage,
            "provenance": provenance,
            "cross_source": cross_source,
            "comparability": comparability,
            "retrieval_quality": retrieval_quality,
            "credible_edge_recall": credible_edge_recall,
            "low_confidence_penalty": low_confidence_penalty,
            "logical_penalty": logical_penalty,
            "semantic_mismatch_penalty": semantic_mismatch_penalty,
            "source_dependence_penalty": source_dependence_penalty,
            "over_density_penalty": over_density_penalty,
        }
        return min(max(reward, 0.0), 1.0), components

    def _register(
        self,
        states: Dict[FrozenSet[int], _StateStats],
        candidates: Sequence[RelationCandidate],
        indices: FrozenSet[int],
    ) -> _StateStats:
        if indices not in states:
            reward, components = self.score(candidates, indices)
            states[indices] = _StateStats(reward=reward, components=components)
        return states[indices]

    def _select_state(self, states: Dict[FrozenSet[int], _StateStats]) -> FrozenSet[int]:
        total_visits = max(sum(stat.visits for stat in states.values()), 1)

        def uct(item: tuple[FrozenSet[int], _StateStats]) -> float:
            _, stat = item
            if stat.visits == 0:
                return stat.reward + self.exploration_weight
            exploration = self.exploration_weight * math.sqrt(math.log(total_visits + 1) / stat.visits)
            return stat.mean_value + exploration

        return max(states.items(), key=uct)[0]

    def _best_addition(
        self,
        states: Dict[FrozenSet[int], _StateStats],
        candidates: Sequence[RelationCandidate],
        current: FrozenSet[int],
        allowed: Optional[Iterable[int]] = None,
    ) -> FrozenSet[int]:
        pool = list(allowed) if allowed is not None else list(range(len(candidates)))
        options = [index for index in pool if index not in current]
        if not options:
            return current
        return max(
            (current | {index} for index in options),
            key=lambda state: self._register(states, candidates, frozenset(state)).reward,
        )

    def _primary_expansion(
        self,
        states: Dict[FrozenSet[int], _StateStats],
        candidates: Sequence[RelationCandidate],
        current: FrozenSet[int],
    ) -> FrozenSet[int]:
        return frozenset(self._best_addition(states, candidates, current))

    def _intra_branch_evolution(
        self,
        states: Dict[FrozenSet[int], _StateStats],
        candidates: Sequence[RelationCandidate],
        current: FrozenSet[int],
    ) -> FrozenSet[int]:
        options: List[FrozenSet[int]] = [current]
        for index in current:
            reduced = frozenset(set(current) - {index})
            options.append(reduced)
            options.append(frozenset(self._best_addition(states, candidates, reduced)))
        return max(options, key=lambda state: self._register(states, candidates, state).reward)

    def _elite_states(self, states: Dict[FrozenSet[int], _StateStats], limit: int = 8) -> List[FrozenSet[int]]:
        return [state for state, _ in sorted(
            states.items(), key=lambda item: item[1].reward, reverse=True
        )[:limit]]

    def _cross_branch_reference(
        self,
        states: Dict[FrozenSet[int], _StateStats],
        candidates: Sequence[RelationCandidate],
        current: FrozenSet[int],
    ) -> FrozenSet[int]:
        donors = [state for state in self._elite_states(states) if state != current and state - current]
        if not donors:
            return self._primary_expansion(states, candidates, current)
        donor = self.random.choice(donors[:4])
        return frozenset(self._best_addition(states, candidates, current, donor - current))

    def _prune_union(
        self,
        states: Dict[FrozenSet[int], _StateStats],
        candidates: Sequence[RelationCandidate],
        merged: FrozenSet[int],
    ) -> FrozenSet[int]:
        current = merged
        improved = True
        while improved and current:
            improved = False
            current_reward = self._register(states, candidates, current).reward
            options = [frozenset(set(current) - {index}) for index in current]
            best = max(options, key=lambda state: self._register(states, candidates, state).reward)
            if self._register(states, candidates, best).reward > current_reward + 1e-9:
                current = best
                improved = True
        return current

    def _multi_branch_aggregation(
        self,
        states: Dict[FrozenSet[int], _StateStats],
        candidates: Sequence[RelationCandidate],
        current: FrozenSet[int],
    ) -> FrozenSet[int]:
        elites = [state for state in self._elite_states(states) if state != current]
        if not elites:
            return self._primary_expansion(states, candidates, current)
        partner = self.random.choice(elites[:4])
        return self._prune_union(states, candidates, frozenset(current | partner))

    def _rollout(
        self,
        states: Dict[FrozenSet[int], _StateStats],
        candidates: Sequence[RelationCandidate],
        current: FrozenSet[int],
    ) -> FrozenSet[int]:
        state = current
        for _ in range(self.rollout_depth):
            remaining = [index for index in range(len(candidates)) if index not in state]
            if not remaining:
                break
            sample = self.random.sample(remaining, min(4, len(remaining)))
            proposal = frozenset(self._best_addition(states, candidates, state, sample))
            if self._register(states, candidates, proposal).reward + 1e-9 < self._register(states, candidates, state).reward:
                break
            state = proposal
        return state

    def search(
        self,
        candidates: Sequence[RelationCandidate],
        iterations: int = 128,
    ) -> Tuple[List[RelationCandidate], Dict[str, object]]:
        searchable = [candidate for candidate in candidates if candidate.relation != "unrelated"]
        if not searchable:
            return [], {
                "mode": "evidence_gams",
                "candidate_count": 0,
                "selected_count": 0,
                "states_explored": 1,
                "best_reward": 0.0,
                "direct_accept_all_reward": 0.0,
                "operator_counts": {operator: 0 for operator in self.OPERATORS},
                "random_seed": self.random_seed,
            }

        states: Dict[FrozenSet[int], _StateStats] = {}
        empty = frozenset()
        self._register(states, searchable, empty)
        for index in range(len(searchable)):
            self._register(states, searchable, frozenset({index}))

        operator_counts = {operator: 0 for operator in self.OPERATORS}
        operations = {
            "primary_expansion": self._primary_expansion,
            "intra_branch_evolution": self._intra_branch_evolution,
            "cross_branch_reference": self._cross_branch_reference,
            "multi_branch_aggregation": self._multi_branch_aggregation,
        }
        for iteration in range(max(1, iterations)):
            operator = self.OPERATORS[iteration % len(self.OPERATORS)]
            operator_counts[operator] += 1
            parent = self._select_state(states)
            child = operations[operator](states, searchable, parent)
            child = self._rollout(states, searchable, frozenset(child))
            parent_stat = self._register(states, searchable, parent)
            child_stat = self._register(states, searchable, child)
            child_stat.visits += 1
            child_stat.value_sum += child_stat.reward
            parent_stat.visits += 1
            parent_stat.value_sum += child_stat.reward

        best_state, best_stat = max(states.items(), key=lambda item: item[1].reward)
        direct_state = frozenset(range(len(searchable)))
        direct_stat = self._register(states, searchable, direct_state)
        selected = [searchable[index] for index in sorted(best_state)]
        selected_ids = {candidate.id for candidate in selected}
        trace: Dict[str, object] = {
            "mode": "evidence_gams",
            "candidate_count": len(searchable),
            "selected_count": len(selected),
            "states_explored": len(states),
            "iterations": max(1, iterations),
            "best_reward": round(best_stat.reward, 6),
            "best_components": {key: round(value, 6) for key, value in best_stat.components.items()},
            "direct_accept_all_reward": round(direct_stat.reward, 6),
            "direct_accept_all_components": {
                key: round(value, 6) for key, value in direct_stat.components.items()
            },
            "operator_counts": operator_counts,
            "selected_edge_ids": sorted(selected_ids),
            "rejected_edge_ids": sorted(candidate.id for candidate in searchable if candidate.id not in selected_ids),
            "random_seed": self.random_seed,
        }
        return selected, trace
