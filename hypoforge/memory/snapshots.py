"""Lossless per-round EvidenceGraph snapshots.

Every M3 completion writes the full ``EvidenceGraph`` (via
``model_dump(mode="json")``) to ``{output_dir}/graph-round-{N}.json``.
These snapshots are the authoritative lossless record — unlike the
``KnowledgeGraphManager`` JSONL store, nothing goes through the lossy
EvidenceGraph → KnowledgeGraph conversion.

Followup runs can seed from the parent run's latest round via
:func:`load_latest_graph_round`.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)

_SNAPSHOT_PATTERN = re.compile(r"^graph-round-(\d+)\.json$")


def graph_round_snapshot_path(output_dir: Union[str, Path], round_no: int) -> Path:
    """Path of the round snapshot file for *round_no*."""
    return Path(output_dir) / f"graph-round-{int(round_no)}.json"


def save_graph_round_snapshot(
    evidence_graph: Any,
    output_dir: Union[str, Path],
    round_no: int,
) -> Optional[Path]:
    """Persist *evidence_graph* losslessly as ``graph-round-{N}.json``.

    Same round overwrites its own file; different rounds use different
    files.  Returns the written path, or ``None`` on failure.
    """
    try:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = graph_round_snapshot_path(out_dir, round_no)
        payload = evidence_graph.model_dump(mode="json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        logger.info(
            "M3: wrote lossless round snapshot %s (%d nodes, %d edges)",
            path, len(getattr(evidence_graph, "nodes", [])),
            len(getattr(evidence_graph, "edges", [])),
        )
        return path
    except Exception as exc:  # never break the pipeline for a snapshot
        logger.warning("M3: failed to write graph-round snapshot: %s", exc)
        return None


def load_latest_graph_round(output_dir: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """Load the highest-numbered ``graph-round-{N}.json`` in *output_dir*.

    Returns the parsed dict (a full ``EvidenceGraph`` JSON dump) or
    ``None`` if the directory is missing or contains no snapshot.
    """
    try:
        out_dir = Path(output_dir)
        if not out_dir.is_dir():
            return None
        best_round = -1
        best_file: Optional[Path] = None
        for file in out_dir.glob("graph-round-*.json"):
            match = _SNAPSHOT_PATTERN.match(file.name)
            if not match:
                continue
            round_no = int(match.group(1))
            if round_no > best_round:
                best_round = round_no
                best_file = file
        if best_file is None:
            return None
        with open(best_file, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            return None
        return data
    except Exception as exc:
        logger.warning("Failed to load latest graph round from %s: %s", output_dir, exc)
        return None
