"""
Single source of truth for HypoForge scoring.

Historically three subsystems scored things independently and drifted apart:

* **M4 Ranker** — hypothesis dimensions in ``[0, 1]`` with a hard-coded
  ``0.30/0.25/0.25/0.20`` composite (duplicated in the prompt *and* the code).
* **M6 Reviewer** — review dimensions on a ``1–5`` Likert scale.
* **evaluation/** — a fourth set of metrics that echoed M4's self-scores.

This module centralises the *definitions* (dimension names, weights, the
composite formula, rubric anchors, and cross-scale normalisation) so every
subsystem references the same numbers.  ``PipelineConfig.scoring`` can override
the weights; everything else derives from here.
"""

from __future__ import annotations

from typing import Dict, Mapping, Sequence

# ---------------------------------------------------------------------------
# Hypothesis dimensions (M4, scored in [0, 1])
# ---------------------------------------------------------------------------

HYPOTHESIS_DIMENSIONS: tuple[str, ...] = (
    "novelty",
    "scientific_soundness",
    "testability",
    "evidence_consistency",
)

DEFAULT_HYPOTHESIS_WEIGHTS: Dict[str, float] = {
    "novelty": 0.30,
    "scientific_soundness": 0.25,
    "testability": 0.25,
    "evidence_consistency": 0.20,
}

# Rubric anchors — what each dimension means and what high/low looks like.
# Surfaced in the scoring report (and available for prompt injection) so scores
# are given against explicit criteria rather than an unanchored gut feeling.
HYPOTHESIS_RUBRIC: Dict[str, str] = {
    "novelty": (
        "How different is this from existing published work? "
        "1.0 = a genuinely new mechanism/target absent from the literature; "
        "0.5 = a recombination of known ideas; 0.0 = restates an established fact."
    ),
    "scientific_soundness": (
        "Is the causal chain internally consistent and free of logical gaps? "
        "1.0 = every step mechanistically justified; 0.0 = hand-waving or self-contradiction."
    ),
    "testability": (
        "Can it be tested with current methods, with specific predictions and "
        "falsification conditions? 1.0 = directional, quantitative, falsifiable; "
        "0.0 = vague or unfalsifiable."
    ),
    "evidence_consistency": (
        "Is it compatible with the known evidence base? "
        "1.0 = consistent with established facts and engages the conflicts; "
        "0.0 = contradicts well-established evidence."
    ),
}


# ---------------------------------------------------------------------------
# Review dimensions (M6, scored on a 1–5 Likert scale)
# ---------------------------------------------------------------------------

REVIEW_DIMENSIONS: tuple[str, ...] = (
    "scientific_logic",
    "objective_evidence_consistency",
    "method_feasibility",
)

REVIEW_SCALE_MIN = 1.0
REVIEW_SCALE_MAX = 5.0


# ---------------------------------------------------------------------------
# Weight helpers
# ---------------------------------------------------------------------------

def normalise_weights(
    weights: Mapping[str, float] | None,
    dims: Sequence[str] = HYPOTHESIS_DIMENSIONS,
) -> Dict[str, float]:
    """Return weights restricted to *dims*, renormalised to sum to 1.0.

    Falls back to :data:`DEFAULT_HYPOTHESIS_WEIGHTS`, then to a uniform
    distribution if the provided weights are all zero / missing.
    """
    src = dict(weights) if weights else dict(DEFAULT_HYPOTHESIS_WEIGHTS)
    picked = {d: max(0.0, float(src.get(d, 0.0))) for d in dims}
    total = sum(picked.values())
    if total <= 0:
        return {d: 1.0 / len(dims) for d in dims}
    return {d: w / total for d, w in picked.items()}


def composite_score(
    scores: Mapping[str, float],
    weights: Mapping[str, float] | None = None,
) -> float:
    """Weighted composite over whichever hypothesis dimensions are present.

    Only dimensions that actually appear in *scores* contribute; the weights
    are renormalised over the present dimensions so a missing dimension does
    not silently drag the composite toward zero.
    """
    present = [d for d in HYPOTHESIS_DIMENSIONS if d in scores]
    if not present:
        return 0.0
    w = normalise_weights(weights, present)
    return round(sum(w[d] * float(scores[d]) for d in present), 4)


def normalise_review_score(score: float) -> float:
    """Map a 1–5 Likert review score to ``[0, 1]`` (for cross-scale reconciliation)."""
    clamped = max(REVIEW_SCALE_MIN, min(REVIEW_SCALE_MAX, float(score)))
    return round((clamped - REVIEW_SCALE_MIN) / (REVIEW_SCALE_MAX - REVIEW_SCALE_MIN), 4)


def weights_summary(weights: Mapping[str, float] | None = None) -> str:
    """Prompt-/human-readable ``composite = 0.30×novelty + ...`` string."""
    w = normalise_weights(weights)
    terms = " + ".join(f"{w[d]:.2f}×{d}" for d in HYPOTHESIS_DIMENSIONS)
    return f"composite = {terms}"


# ---------------------------------------------------------------------------
# Prompt-injection helpers (rubric anchors — single source for M4/M6 prompts)
# ---------------------------------------------------------------------------

REVIEW_RUBRIC: Dict[str, str] = {
    "task_alignment": (
        "1 = different research object/domain; 3 = partial goal coverage or ambiguous object; "
        "5 = exact object, domain, and requested-goal alignment."
    ),
    "scientific_logic": (
        "1 = incoherent or circular; 3 = plausible but with logical gaps; "
        "5 = rigorous — every step justified and the predictions follow from the mechanism."
    ),
    "objective_evidence_consistency": (
        "1 = central claims are unsupported or contradicted; 3 = the mechanism is "
        "plausible but at least one central step is only partially supported; "
        "5 = every central causal step is directly entailed by canonical evidence."
    ),
    "method_feasibility": (
        "1 = not executable as written; 3 = feasible but controls / sample size / statistics "
        "are underspecified; 5 = ready to execute — controls, power, and analysis all specified."
    ),
}


def hypothesis_rubric_block() -> str:
    """Bulleted rubric anchors for the four hypothesis dimensions (for prompts)."""
    return "\n".join(f"- **{dim}** — {HYPOTHESIS_RUBRIC[dim]}" for dim in HYPOTHESIS_DIMENSIONS)


def review_rubric_line(dimension: str) -> str:
    """One-line 1–5 scoring anchor for an M6 review dimension (empty if unknown)."""
    anchor = REVIEW_RUBRIC.get(dimension)
    return f"Scoring anchors — {anchor}" if anchor else ""
