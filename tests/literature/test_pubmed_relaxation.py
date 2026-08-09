"""Tests for the PubMed zero-result relaxation ladder, planner prompt
hardening, biomedical PubMed-first ordering, and the config switch."""

from __future__ import annotations

import pytest

from hypoforge.config import PipelineConfig, SearchConfig
from hypoforge.literature.models import SearchQuery, SearchState
from hypoforge.literature.search.query_planner import (
    _QUERY_PLANNING_SYSTEM,
    QueryPlanner,
    _is_biomedical,
)
from hypoforge.literature.search.search_tool import LiteratureSearchTool
from hypoforge.literature.sources import pubmed as pubmed_module
from hypoforge.literature.sources.pubmed import (
    PubMedLiteratureSource,
    relaxation_candidates,
    split_top_level_and_clauses,
    strip_field_tags,
)
from hypoforge.literature.sources.pubmed_source import PubMedSource
from hypoforge.tools import pubmed_search as pubmed_tools


def pubmed_query(text: str) -> SearchQuery:
    return SearchQuery(
        query_id="q-1",
        text=text,
        target_source="pubmed",
        purpose="core",
        relation_to_question="test",
    )


def row(pmid: str, title: str = "Relaxed hit") -> dict:
    return {"pmid": pmid, "title": title, "abstract": "Abstract text."}


# ---------------------------------------------------------------------------
# Ladder candidate generation (pure helpers)
# ---------------------------------------------------------------------------

def test_strip_field_tags_keeps_phrases_and_operators() -> None:
    stripped = strip_field_tags(
        '"protein misfolding"[MeSH Terms] AND chaperone[tiab]'
    )
    assert stripped == '"protein misfolding" AND chaperone'


def test_split_top_level_and_clauses_respects_parentheses_and_quotes() -> None:
    clauses = split_top_level_and_clauses('(a OR b) AND "c AND d" AND e')
    assert clauses == ["( a OR b )", '"c AND d"', "e"]


def test_relaxation_candidates_ladder_order() -> None:
    original = '"amyloid fibril"[tiab] AND "heat shock"[tiab] AND aging[tiab]'

    candidates = relaxation_candidates(original)

    assert candidates == [
        # L1 — strip field tags, keep quoted phrases
        '"amyloid fibril" AND "heat shock" AND aging',
        # L2 — drop rightmost AND clause, keep first two concepts
        '"amyloid fibril" AND "heat shock"',
        # L3 — first 3 / first 2 concepts unquoted
        "amyloid fibril AND heat shock AND aging",
        "amyloid fibril AND heat shock",
    ]
    assert original not in candidates


def test_relaxation_candidates_deduplicate_and_skip_single_concept() -> None:
    assert relaxation_candidates("Hippo") == []


# ---------------------------------------------------------------------------
# PubMedLiteratureSource relaxation behaviour (mocked esearch count/efetch)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_l1_strip_field_tags_hit_is_tagged() -> None:
    original = '"protein misfolding"[MeSH Terms] AND chaperone[MeSH Terms]'
    relaxed = '"protein misfolding" AND chaperone'
    calls: list[str] = []

    async def backend(text: str, limit: int):
        calls.append(text)
        return [row("1")] if text == relaxed else []

    probes: list[str] = []

    async def probe(text: str, timeout_seconds: float | None = None) -> int:
        probes.append(text)
        return 23598 if text == relaxed else 0

    source = PubMedLiteratureSource(backend=backend, count_probe=probe)
    records = await source.search(pubmed_query(original), limit=4)

    assert calls == [original, relaxed]
    assert probes == [relaxed]
    assert len(records) == 1
    assert records[0].relaxed_from == original


@pytest.mark.asyncio
async def test_l2_dropping_and_clauses_hit() -> None:
    original = (
        "Hsp70[tiab] AND ATPase[tiab] AND proteostasis[tiab] AND aging[tiab]"
    )
    l1 = "Hsp70 AND ATPase AND proteostasis AND aging"
    l2 = "Hsp70 AND ATPase AND proteostasis"
    calls: list[str] = []

    async def backend(text: str, limit: int):
        calls.append(text)
        return [row("2")] if text == l2 else []

    probes: list[str] = []

    async def probe(text: str, timeout_seconds: float | None = None) -> int:
        probes.append(text)
        return 36 if text == l2 else 0

    source = PubMedLiteratureSource(backend=backend, count_probe=probe)
    records = await source.search(pubmed_query(original), limit=4)

    assert calls == [original, l2]
    assert probes == [l1, l2]
    assert [record.relaxed_from for record in records] == [original]


@pytest.mark.asyncio
async def test_l3_unquoted_core_concepts_hit() -> None:
    original = '"amyloid fibril"[tiab] AND "heat shock"[tiab] AND aging[tiab]'
    candidates = relaxation_candidates(original)
    l3 = candidates[2]
    calls: list[str] = []

    async def backend(text: str, limit: int):
        calls.append(text)
        return [row("3")] if text == l3 else []

    probes: list[str] = []

    async def probe(text: str, timeout_seconds: float | None = None) -> int:
        probes.append(text)
        return 12 if text == l3 else 0

    source = PubMedLiteratureSource(backend=backend, count_probe=probe)
    records = await source.search(pubmed_query(original), limit=4)

    assert probes == candidates[:3]
    assert calls == [original, l3]
    assert records[0].relaxed_from == original


@pytest.mark.asyncio
async def test_all_levels_miss_returns_empty_without_tags() -> None:
    original = '"x y"[MeSH Terms] AND z[tiab] AND w[tiab] AND v[tiab]'
    calls: list[str] = []

    async def backend(text: str, limit: int):
        calls.append(text)
        return []

    probes: list[str] = []

    async def probe(text: str, timeout_seconds: float | None = None) -> int:
        probes.append(text)
        return 0

    source = PubMedLiteratureSource(backend=backend, count_probe=probe)
    records = await source.search(pubmed_query(original), limit=4)

    assert records == []
    assert calls == [original]
    assert probes == relaxation_candidates(original)[:3]


@pytest.mark.asyncio
async def test_probe_hit_but_fetch_empty_keeps_relaxing() -> None:
    original = "a[tiab] AND b[tiab] AND c[tiab] AND d[tiab]"

    async def backend(text: str, limit: int):
        return []

    probes: list[str] = []

    async def probe(text: str, timeout_seconds: float | None = None) -> int:
        probes.append(text)
        return 1

    source = PubMedLiteratureSource(backend=backend, count_probe=probe)
    records = await source.search(pubmed_query(original), limit=4)

    assert records == []
    assert len(probes) == 3  # hard cap per query


@pytest.mark.asyncio
async def test_probe_attempts_are_capped_at_three() -> None:
    original = '"amyloid fibril"[tiab] AND "heat shock"[tiab] AND aging[tiab]'
    assert len(relaxation_candidates(original)) > 3

    async def backend(text: str, limit: int):
        return []

    probes: list[str] = []

    async def probe(text: str, timeout_seconds: float | None = None) -> int:
        probes.append(text)
        return 0

    source = PubMedLiteratureSource(backend=backend, count_probe=probe)
    await source.search(pubmed_query(original), limit=4)

    assert len(probes) == 3


@pytest.mark.asyncio
async def test_relaxation_disabled_keeps_single_call() -> None:
    original = '"protein misfolding"[MeSH Terms]'
    calls: list[str] = []

    async def backend(text: str, limit: int):
        calls.append(text)
        return []

    async def probe(text: str, timeout_seconds: float | None = None) -> int:
        raise AssertionError("probe must not run when relaxation is disabled")

    source = PubMedLiteratureSource(
        backend=backend, count_probe=probe, enable_relaxation=False
    )
    records = await source.search(pubmed_query(original), limit=4)

    assert records == []
    assert calls == [original]


@pytest.mark.asyncio
async def test_injected_backend_relaxes_blindly_without_ncbi_probe() -> None:
    original = "Hsp70[tiab] AND ATPase[tiab] AND proteostasis[tiab]"
    relaxed = "Hsp70 AND ATPase AND proteostasis"
    calls: list[str] = []

    async def backend(text: str, limit: int):
        calls.append(text)
        return [row("9")] if text == relaxed else []

    source = PubMedLiteratureSource(backend=backend)
    records = await source.search(pubmed_query(original), limit=4)

    assert calls == [original, relaxed]
    assert records[0].relaxed_from == original


@pytest.mark.asyncio
async def test_probe_failure_is_skipped_not_fatal() -> None:
    original = "a[tiab] AND b[tiab] AND c[tiab]"
    candidates = relaxation_candidates(original)

    async def backend(text: str, limit: int):
        return [row("5")] if text == candidates[1] else []

    async def probe(text: str, timeout_seconds: float | None = None) -> int:
        if text == candidates[0]:
            raise OSError("ncbi unreachable")
        return 4

    source = PubMedLiteratureSource(backend=backend, count_probe=probe)
    records = await source.search(pubmed_query(original), limit=4)

    assert records and records[0].relaxed_from == original


# ---------------------------------------------------------------------------
# PubMedSource (integrated variant) relaxation behaviour
# ---------------------------------------------------------------------------

class FakeStrictTool:
    def __init__(self, rows_for: dict[str, list[dict]]) -> None:
        self.rows_for = rows_for
        self.calls: list[tuple[str, int]] = []

    async def search_strict(self, query: str, limit: int = 20, **kwargs):
        self.calls.append((query, limit))
        return self.rows_for.get(query, [])


@pytest.mark.asyncio
async def test_pubmed_source_relaxes_and_tags_records() -> None:
    original = '"protein misfolding"[MeSH Terms] AND chaperone[tiab]'
    relaxed = '"protein misfolding" AND chaperone'
    tool = FakeStrictTool({relaxed: [row("11")]})

    async def probe(text: str, timeout_seconds: float | None = None) -> int:
        return 7 if text == relaxed else 0

    source = PubMedSource(tool=tool, count_probe=probe)
    records = await source.search(pubmed_query(original), limit=5)

    assert tool.calls == [(original, 5), (relaxed, 5)]
    assert len(records) == 1
    assert records[0].relaxed_from == original


@pytest.mark.asyncio
async def test_pubmed_source_relaxation_disabled() -> None:
    original = '"protein misfolding"[MeSH Terms]'
    tool = FakeStrictTool({})

    source = PubMedSource(tool=tool, enable_relaxation=False)
    records = await source.search(pubmed_query(original), limit=5)

    assert records == []
    assert tool.calls == [(original, 5)]


@pytest.mark.asyncio
async def test_pubmed_source_default_wiring_probes_then_fetches(
    monkeypatch,
) -> None:
    original = "amyloid[tiab] AND zzzfake[tiab]"
    relaxed = "amyloid AND zzzfake"
    probe_calls: list[str] = []

    async def stub_count(text: str, timeout_seconds: float | None = None) -> int:
        probe_calls.append(text)
        return 3 if text == relaxed else 0

    def strict_sync(query: str, limit: int, deadline=None):
        if query == relaxed:
            return [row("77", "Default wiring hit")]
        return []

    monkeypatch.setattr(pubmed_module, "pubmed_count", stub_count)
    monkeypatch.setattr(
        pubmed_tools, "_search_pubmed_strict_sync", strict_sync
    )

    source = PubMedSource()
    records = await source.search(pubmed_query(original), limit=5)

    assert probe_calls == [relaxed]
    assert len(records) == 1
    assert records[0].pmid == "77"
    assert records[0].relaxed_from == original


# ---------------------------------------------------------------------------
# Query planner prompt hardening
# ---------------------------------------------------------------------------

def test_planner_prompt_forbids_invented_mesh_headings() -> None:
    assert "NEVER invent MeSH headings from memory" in _QUERY_PLANNING_SYSTEM
    assert "automatic term mapping" in _QUERY_PLANNING_SYSTEM
    assert '"protein misfolding"[MeSH Terms]' in _QUERY_PLANNING_SYSTEM


def test_planner_prompt_requires_two_concept_safety_net() -> None:
    assert "broad-recall safety-net query" in _QUERY_PLANNING_SYSTEM
    assert "2 most central" in _QUERY_PLANNING_SYSTEM


# ---------------------------------------------------------------------------
# Biomedical domains → PubMed-first ordering
# ---------------------------------------------------------------------------

def test_is_biomedical_keyword_matching() -> None:
    assert _is_biomedical(["Neuroscience"])
    assert _is_biomedical(["medicine"])
    assert _is_biomedical(["molecular biology"])
    assert not _is_biomedical(["computer science"])
    assert not _is_biomedical([])


class FakeClient:
    def __init__(self, response: dict) -> None:
        self.response = response

    async def structured_chat(self, **kwargs):
        return self.response


def planner_with(response: dict) -> QueryPlanner:
    return QueryPlanner(FakeClient(response), LiteratureSearchTool.TOOL_DEFINITIONS)


@pytest.mark.asyncio
async def test_biomedical_domains_move_pubmed_queries_first() -> None:
    response = {
        "queries": [
            {"text": "cs query", "tool": "semantic_scholar",
             "purpose": "core_mechanism", "reasoning": "r"},
            {"text": "arxiv query", "tool": "arxiv",
             "purpose": "methods", "reasoning": "r"},
        ]
    }

    queries = await planner_with(response).plan(
        "question", domains=["neuroscience", "molecular biology"],
        state=SearchState(),
    )

    assert queries[0].target_source == "pubmed"
    assert {item.target_source for item in queries} == {
        "pubmed", "semantic_scholar", "arxiv",
        "openalex",
    }


@pytest.mark.asyncio
async def test_biomedical_repeated_pubmed_queries_also_move_first() -> None:
    response = {
        "queries": [
            {"text": "s2 query", "tool": "semantic_scholar",
             "purpose": "review", "reasoning": "r"},
            {"text": "pubmed query 1", "tool": "pubmed",
             "purpose": "core_mechanism", "reasoning": "r"},
            {"text": "pubmed query 2", "tool": "pubmed",
             "purpose": "recent_research", "reasoning": "r"},
        ]
    }

    queries = await planner_with(response).plan(
        "question", domains=["medicine"], state=SearchState(),
    )

    assert [item.target_source for item in queries[:2]] == ["pubmed", "pubmed"]


@pytest.mark.asyncio
async def test_non_biomedical_domains_keep_original_order() -> None:
    response = {
        "queries": [
            {"text": "cs query", "tool": "semantic_scholar",
             "purpose": "core_mechanism", "reasoning": "r"},
            {"text": "arxiv query", "tool": "arxiv",
             "purpose": "methods", "reasoning": "r"},
        ]
    }

    queries = await planner_with(response).plan(
        "question", domains=["computer science"], state=SearchState(),
    )

    assert [item.target_source for item in queries[:2]] == [
        "semantic_scholar", "arxiv",
    ]
    assert {item.target_source for item in queries} == {
        "pubmed", "semantic_scholar", "openalex", "arxiv",
    }


# ---------------------------------------------------------------------------
# Config switch wiring
# ---------------------------------------------------------------------------

def test_zero_result_relaxation_defaults_true_and_injects_kwargs() -> None:
    config = PipelineConfig(search=SearchConfig(implementation="agentic"))

    kwargs = config.get_module_kwargs("m2")

    assert config.search.zero_result_relaxation is True
    assert kwargs["zero_result_relaxation"] is True


def test_zero_result_relaxation_false_is_forwarded() -> None:
    config = PipelineConfig(search=SearchConfig(
        implementation="agentic", zero_result_relaxation=False,
    ))

    assert config.get_module_kwargs("m2")["zero_result_relaxation"] is False


def test_literature_search_tool_forwards_relaxation_switch() -> None:
    tool = LiteratureSearchTool(zero_result_relaxation=False)

    pubmed = next(
        source for source in tool.as_source_list()
        if source.source_name == "pubmed"
    )

    assert pubmed.enable_relaxation is False


def test_minimal_factory_forwards_relaxation_switch() -> None:
    from hypoforge.literature.minimal import build_minimal_pubmed_adapter

    adapter = build_minimal_pubmed_adapter(
        final_k=2, zero_result_relaxation=False,
    )

    source = adapter.search_agent.sources["pubmed"]
    assert source.enable_relaxation is False
