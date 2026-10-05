"""Read a Model Studio credential export without copying it into the repo."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from urllib.parse import urlparse


def _validate_credentials(key: str, base: str) -> tuple[str, str]:
    if not key or not base:
        raise ValueError("Credential file must contain an API key and an API base URL")
    parsed = urlparse(base)
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username
            or parsed.password or parsed.query or parsed.fragment):
        raise ValueError("The API endpoint must be an HTTPS URL without embedded credentials or query parameters")
    return key, base.rstrip("/")


def read_api_key_file(path: str | Path) -> tuple[str, str]:
    """Read a JSON object or a text file containing one sk- key and one URL.

    Straight and curly quotation marks around the URL are accepted. Values
    are used at runtime and are never copied into experiment metadata.
    """
    raw = Path(path).expanduser().read_text(encoding="utf-8-sig")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        key = payload.get("api_key") or payload.get("apiKey") or ""
        base = payload.get("base_url") or payload.get("openAiCompatible") or ""
        if not isinstance(key, str) or not isinstance(base, str):
            raise ValueError("API key and base URL must be strings")
        return _validate_credentials(key.strip(), base.strip())
    urls = re.findall(r"https?://[^\s\"'“”‘’<>]+", raw)
    keys = re.findall(r"(?<![A-Za-z0-9_-])sk-[A-Za-z0-9_-]+", raw)
    if len(urls) != 1 or len(keys) != 1:
        raise ValueError("Text credential file must contain exactly one sk- API key and one HTTPS base URL")
    return _validate_credentials(keys[0], urls[0])


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
    return _validate_credentials(key, base)
