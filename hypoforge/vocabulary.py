"""Domain-aware terminology normalisation for HypoForge.

The packaged vocabulary deliberately distinguishes *equivalent aliases* from
merely related search terms.  Only equivalent aliases may collapse M3 entity
nodes.  Related terms are available for retrieval/query expansion, but never
cause graph-node merging.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set


VOCABULARY_PATH = Path(__file__).with_name("data") / "science125_vocabulary.json"


def _lookup_key(value: str) -> str:
    """Return a Unicode-stable key used only for alias lookup."""
    value = unicodedata.normalize("NFKC", value or "").casefold().strip()
    value = value.replace("α", "alpha").replace("β", "beta").replace("κ", "kappa")
    value = re.sub(r"[^0-9a-z+\u4e00-\u9fff]+", "", value)
    return value


def _fallback(value: str) -> str:
    """Keep the historic M3 fallback contract for unknown entity names."""
    cleaned = unicodedata.normalize("NFKC", value or "").strip().casefold()
    return cleaned.rstrip(".,;:)-]").lstrip("([")


@lru_cache(maxsize=1)
def load_vocabulary() -> dict:
    with VOCABULARY_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=1)
def _indexes() -> tuple[Dict[str, Set[str]], Dict[str, Dict[str, Set[str]]]]:
    global_index: Dict[str, Set[str]] = defaultdict(set)
    domain_index: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
    for entry in load_vocabulary()["entries"]:
        canonical = entry["preferred"].casefold()
        domain = entry["domain"]
        surfaces = [entry["preferred"], entry.get("preferred_zh", ""), *entry.get("aliases", [])]
        for surface in surfaces:
            key = _lookup_key(surface)
            if not key:
                continue
            global_index[key].add(canonical)
            domain_index[domain][key].add(canonical)
    return dict(global_index), {k: dict(v) for k, v in domain_index.items()}


def normalize_entity(name: str, domains: Optional[Iterable[str]] = None) -> str:
    """Map an entity surface form to one unambiguous preferred label.

    Domain matches take priority.  Without domain context, an alias is merged
    only when it identifies exactly one concept globally.  Ambiguous strings
    such as ``AI`` or ``handedness`` therefore remain untouched instead of
    silently joining unrelated graph nodes.
    """
    key = _lookup_key(name)
    global_index, domain_index = _indexes()
    if domains:
        matches: Set[str] = set()
        for domain in domains:
            matches.update(domain_index.get(domain, {}).get(key, set()))
        if len(matches) == 1:
            return next(iter(matches))
    matches = global_index.get(key, set())
    if len(matches) == 1:
        return next(iter(matches))
    return _fallback(name)


def related_terms(name: str, domains: Optional[Iterable[str]] = None) -> List[str]:
    """Return controlled retrieval terms without changing graph identity."""
    canonical = normalize_entity(name, domains)
    wanted_domains = set(domains or [])
    terms: List[str] = []
    for entry in load_vocabulary()["entries"]:
        if entry["preferred"].casefold() != canonical:
            continue
        if wanted_domains and entry["domain"] not in wanted_domains:
            continue
        terms.extend(entry.get("related_terms", []))
    return list(dict.fromkeys(terms))


def equivalent_aliases(name: str, domains: Optional[Iterable[str]] = None) -> List[str]:
    """Return the safe equivalent surface forms for a concept."""
    canonical = normalize_entity(name, domains)
    wanted_domains = set(domains or [])
    terms: List[str] = []
    for entry in load_vocabulary()["entries"]:
        if entry["preferred"].casefold() != canonical:
            continue
        if wanted_domains and entry["domain"] not in wanted_domains:
            continue
        terms.extend([entry["preferred"], entry.get("preferred_zh", ""), *entry.get("aliases", [])])
    return [term for term in dict.fromkeys(terms) if term]


def legacy_synonym_map() -> Dict[str, Set[str]]:
    """Compatibility view consumed by older M3 imports and tests."""
    result: Dict[str, Set[str]] = defaultdict(set)
    for entry in load_vocabulary()["entries"]:
        canonical = entry["preferred"].casefold()
        result[canonical].update(
            term.casefold()
            for term in [entry["preferred"], entry.get("preferred_zh", ""), *entry.get("aliases", [])]
            if term
        )
    return dict(result)


def add_runtime_alias(canonical: str, variant: str, domain: str = "runtime") -> None:
    """Add a process-local alias while retaining the legacy public API."""
    global_index, domain_index = _indexes()
    preferred = _fallback(canonical)
    key = _lookup_key(variant)
    global_index.setdefault(key, set()).add(preferred)
    domain_index.setdefault(domain, {}).setdefault(key, set()).add(preferred)
