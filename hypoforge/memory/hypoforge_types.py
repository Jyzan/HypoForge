"""Shared memory-layer type helpers (P2 knowledge-retention contract).

This module is the canonical home for cross-module type-level helpers of
the persistent memory layer:

* :func:`stable_entry_id` — the stable ``KnowledgeEntry.id`` derivation
  (``KE_{sha1(normalised content + source_paper_id)[:12]}``).  M2 uses the
  identical algorithm when extracting entries, so the same evidence always
  maps to the same id across runs.  That stability is what makes M3
  incremental dedup (``_find_new_entries``) and BM25 index updates work
  across supplement rounds.
* :data:`GapGain` — the type of the per-gap new-evidence counter that M3
  publishes after each round (see the ``metrics["m3_gap_gain"]`` contract
  in :mod:`hypoforge.modules.m3_evidence_graph`).

No imports from ``hypoforge.state`` on purpose — this module must stay
import-cycle-free so both the memory layer and the pipeline modules can
use it.
"""

from __future__ import annotations

import hashlib
from typing import Dict

# dict[gap_id, int] — number of *new effective* evidence entries gained by
# each evidence gap in the latest M3 round.  Consumed by P2 (gap status
# transitions / no-improvement accounting).
GapGain = Dict[str, int]

_STABLE_ID_PREFIX = "KE_"
_STABLE_ID_DIGEST_LEN = 12


def normalize_entry_content(content: str) -> str:
    """Whitespace-collapse + casefold normalisation used for id hashing."""
    return " ".join(str(content or "").split()).casefold()


def stable_entry_id(content: str, source_paper_id: str) -> str:
    """Stable ``KnowledgeEntry`` id for a piece of extracted knowledge.

    ``KE_{sha1(normalised content \\x00 source_paper_id)[:12]}``

    Content-based (never list-index-based), so re-extracting the same
    evidence from the same paper in a later round/run yields the *same*
    id — required for M3 incremental dedup and BM25 index reuse.  The
    paper id is part of the key so identical sentences from different
    papers remain distinct evidence items.
    """
    normalised = normalize_entry_content(content)
    digest = hashlib.sha1(
        (normalised + "\u0000" + str(source_paper_id or "")).encode("utf-8")
    ).hexdigest()[:_STABLE_ID_DIGEST_LEN]
    return f"{_STABLE_ID_PREFIX}{digest}"
