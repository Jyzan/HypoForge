"""Read a Model Studio credential export without copying it into the repo."""

from __future__ import annotations

import csv
from pathlib import Path
from urllib.parse import urlparse


def read_api_key_csv(path: str | Path) -> tuple[str, str]:
    """Accept vertical key/value exports and ordinary one-record CSV exports."""
    with Path(path).expanduser().open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream))
    key_names = {"apiKey", "api_key"}
    base_names = {"openAiCompatible", "base_url"}
    if rows and key_names.intersection(rows[0]) and base_names.intersection(rows[0]):
        if len(rows) != 2:
            raise ValueError("Expected a single API credential in the CSV export")
        values = dict(zip(rows[0], rows[1]))
    else:
        values = {row[0].strip(): row[1].strip() for row in rows if len(row) >= 2}
    key = values.get("apiKey") or values.get("api_key") or ""
    base = values.get("openAiCompatible") or values.get("base_url") or ""
    if not key or not base:
        raise ValueError("CSV must contain apiKey and openAiCompatible (or api_key/base_url)")
    parsed = urlparse(base)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("The CSV endpoint must be an HTTPS URL without embedded credentials")
    return key, base.rstrip("/")
