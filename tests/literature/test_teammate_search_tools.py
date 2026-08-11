from __future__ import annotations

import asyncio
import json
import threading
import time

import pytest

from hypoforge.literature.models import FulltextStatus, SearchQuery, SearchState
from hypoforge.literature.search.query_planner import QueryPlanner
from hypoforge.literature.search.search_tool import LiteratureSearchTool
from hypoforge.literature.sources.academic_source import AcademicSource
from hypoforge.literature.sources.pubmed_source import PubMedSource
from hypoforge.tools import pubmed_search, semantic_scholar
from hypoforge.tools.pubmed_search import PubMedTool
from hypoforge.tools.semantic_scholar import SemanticScholarTool


def test_semantic_scholar_keyed_requests_stay_below_one_request_per_second(
    monkeypatch,
) -> None:
    """Catch S2 keyed requests accidentally reverting to the old 100 req/s assumption."""
    sleeps: list[float] = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self) -> bytes:
            return json.dumps({"data": []}).encode("utf-8")

    monkeypatch.setattr(semantic_scholar, "_last_request_time", 100.0)
    monkeypatch.setattr(semantic_scholar.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(semantic_scholar.time, "sleep", sleeps.append)

    class FakeOpener:
        @staticmethod
        def open(request, timeout):
            return FakeResponse()

    monkeypatch.setattr(
        semantic_scholar, "_NO_PROXY_OPENER", FakeOpener()
    )

    semantic_scholar._http_get_json(
        "https://api.semanticscholar.org/graph/v1/paper/search",
        s2_api_key="test-key",
    )

    assert len(sleeps) == 1
    assert sleeps[0] >= 1.0


def test_semantic_scholar_rate_limit_serializes_concurrent_requests(
    monkeypatch,
) -> None:
    """Catch concurrent M2 workers bypassing the process-wide S2 request budget."""
    original_sleep = time.sleep
    start = threading.Barrier(3)
    state_lock = threading.Lock()
    active_sleepers = 0
    max_active_sleepers = 0

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self) -> bytes:
            return b'{"data": []}'

    def tracked_sleep(seconds: float) -> None:
        nonlocal active_sleepers, max_active_sleepers
        with state_lock:
            active_sleepers += 1
            max_active_sleepers = max(max_active_sleepers, active_sleepers)
        original_sleep(0.02)
        with state_lock:
            active_sleepers -= 1

    def request() -> None:
        start.wait()
        semantic_scholar._http_get_json(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            s2_api_key="test-key",
        )

    monkeypatch.setattr(
        semantic_scholar,
        "_last_request_time",
        time.monotonic(),
    )
    monkeypatch.setattr(semantic_scholar.time, "sleep", tracked_sleep)

    class FakeOpener:
        @staticmethod
        def open(request, timeout):
            return FakeResponse()

    monkeypatch.setattr(semantic_scholar, "_NO_PROXY_OPENER", FakeOpener())
    workers = [threading.Thread(target=request) for _ in range(2)]
    for worker in workers:
        worker.start()
    start.wait()
    for worker in workers:
        worker.join(timeout=1)

    assert not any(worker.is_alive() for worker in workers)
    assert max_active_sleepers == 1


class FakeStructuredClient:
    def __init__(self, response=None, error: Exception | None = None):
        self.response = response
        self.error = error
        self.calls = []

    async def structured_chat(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


class FakeLegacyTool:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    async def search(self, query: str, limit: int = 20):
        self.calls.append((query, limit))
        return list(self.rows)


class FakeStrictTool:
    def __init__(self, error: Exception):
        self.error = error
        self.legacy_called = False

    async def search_strict(self, query: str, limit: int = 20):
        raise self.error

    async def search(self, query: str, limit: int = 20):
        self.legacy_called = True
        return []


@pytest.mark.asyncio
async def test_semantic_scholar_constructor_key_selects_instance_backend(
    monkeypatch,
) -> None:
    calls = []

    def search(query: str, limit: int, api_key: str, deadline=None):
        calls.append((query, limit, api_key))
        return []

    def openalex_search(query: str, limit: int, api_key: str, deadline=None):
        raise AssertionError("Semantic Scholar must not fall back to OpenAlex")

    monkeypatch.setattr(semantic_scholar, "_s2_search", search)
    monkeypatch.setattr(semantic_scholar, "_oa_search", openalex_search)
    # Keep the constructor contract test independent from a developer's real
    # local OpenAlex credential loaded through .env.
    monkeypatch.setattr(semantic_scholar, "_OPENALEX_API_KEY", "")
    tool = SemanticScholarTool(api_key="instance-key")

    await tool.search_strict("Hippo", limit=3)

    assert tool.backend_name == "semantic_scholar"
    assert calls == [("Hippo", 3, "instance-key")]
    assert tool.api_key == "instance-key"


@pytest.mark.asyncio
async def test_openalex_instance_key_does_not_switch_semantic_source(monkeypatch) -> None:
    calls = []

    def search(query: str, limit: int, api_key: str = "", deadline=None):
        calls.append((query, limit, api_key))
        return []

    monkeypatch.setattr(semantic_scholar, "_s2_search", search)
    tool = SemanticScholarTool(
        api_key="instance-key",
        openalex_api_key="openalex-key",
    )

    await tool.search_strict("Hippo", limit=4)

    assert tool.backend_name == "semantic_scholar"
    assert calls == [("Hippo", 4, "instance-key")]


@pytest.mark.asyncio
async def test_academic_source_error_names_actual_backend() -> None:
    class FailingOpenAlexTool:
        backend_name = "openalex"

        async def search_strict(self, query: str, limit: int = 20):
            raise OSError("offline")

    source = AcademicSource(tool=FailingOpenAlexTool())
    query = SearchQuery(
        query_id="q1",
        text="Hippo",
        target_source="semantic_scholar",
        purpose="test",
        relation_to_question="test",
    )

    with pytest.raises(OSError, match="openalex backend failed"):
        await source.search(query)


@pytest.mark.asyncio
async def test_query_planner_emits_both_teammate_backends() -> None:
    client = FakeStructuredClient({"queries": [
        {"text": "Hippo[tiab] AND YAP[tiab]", "tool": "pubmed",
         "purpose": "core_mechanism", "reasoning": "Direct mechanism"},
        {"text": '"YAP TAZ" mechanotransduction',
         "tool": "semantic_scholar", "purpose": "supporting_evidence",
         "reasoning": "Cross-disciplinary evidence"},
    ]})
    tool = LiteratureSearchTool(
        pubmed=PubMedSource(tool=FakeLegacyTool([])),
        academic=AcademicSource(tool=FakeLegacyTool([])),
    )
    planner = QueryPlanner(client, tool.tool_definitions)

    queries = await planner.plan("Hippo YAP TAZ", state=SearchState())

    assert {q.target_source for q in queries} == {
        "pubmed", "semantic_scholar", "openalex", "arxiv"
    }
    assert [q.relation_to_question for q in queries[:2]] == [
        "Direct mechanism", "Cross-disciplinary evidence"
    ]
    assert all(query.purpose == "cross_source_coverage" for query in queries[2:])
    assert {query.target_source for query in queries[2:]} == {"openalex", "arxiv"}
    assert client.calls[0]["disable_thinking"] is True


@pytest.mark.asyncio
async def test_query_planner_supplies_relation_when_model_omits_reasoning() -> None:
    client = FakeStructuredClient({"queries": [{
        "text": "Hippo[tiab]", "tool": "pubmed",
        "purpose": "core_mechanism",
    }]})
    planner = QueryPlanner(client, LiteratureSearchTool.TOOL_DEFINITIONS)

    queries = await planner.plan("Hippo", state=SearchState())

    assert len(queries) == 4
    assert queries[0].text == "Hippo[tiab]"
    assert queries[0].target_source == "pubmed"
    assert (
        queries[0].relation_to_question
        == "Targets the core_mechanism dimension."
    )


@pytest.mark.asyncio
async def test_query_planner_sanitizes_academic_syntax_and_is_deterministic() -> None:
    client = FakeStructuredClient({"queries": [{
        "text": '"Hippo pathway"[tiab] AND YAP[MeSH Terms]',
        "tool": "semantic_scholar",
        "purpose": "review",
        "reasoning": "cross-source",
    }]})
    planner = QueryPlanner(client, LiteratureSearchTool.TOOL_DEFINITIONS)

    queries = await planner.plan("Hippo", state=SearchState())

    assert queries[0].text == '"Hippo pathway" AND YAP'
    assert client.calls[0]["temperature"] == 0.0


@pytest.mark.asyncio
async def test_query_planner_keeps_legacy_fallback_outside_strict_mode() -> None:
    planner = QueryPlanner(
        FakeStructuredClient(error=OSError("Qwen unavailable")),
        LiteratureSearchTool.TOOL_DEFINITIONS,
    )

    queries = await planner.plan("Hippo", state=SearchState())

    assert [query.query_id for query in queries] == [
        "q-fb-1",
        "q-fb-2",
        "q-fb-3",
        "q-fb-4",
    ]
    assert [query.target_source for query in queries] == [
        "arxiv",
        "openalex",
        "pubmed",
        "semantic_scholar",
    ]


@pytest.mark.asyncio
async def test_pubmed_source_uses_normalized_doi_and_abstract_status() -> None:
    source = PubMedSource(tool=FakeLegacyTool([{
        "doi": "https://doi.org/10.1000/ABC",
        "title": "Hippo study",
        "abstract": "Real abstract.",
    }]))
    query = (await QueryPlanner(
        FakeStructuredClient({"queries": [{
            "text": "Hippo", "tool": "pubmed",
            "purpose": "core", "reasoning": "test",
        }]}), LiteratureSearchTool.TOOL_DEFINITIONS,
    ).plan("Hippo", state=SearchState()))[0]

    paper = (await source.search(query))[0]

    assert paper.doi == "10.1000/abc"
    assert paper.paper_id == "DOI:10.1000/abc"
    assert paper.fulltext_status is FulltextStatus.ABSTRACT_ONLY


@pytest.mark.asyncio
async def test_pubmed_source_preserves_unknown_citation_count() -> None:
    source = PubMedSource(tool=FakeLegacyTool([{
        "pmid": "123",
        "title": "Hippo study",
        "abstract": "Evidence.",
    }]))
    query = SearchQuery(
        query_id="q1",
        text="Hippo",
        target_source="pubmed",
        purpose="test",
        relation_to_question="test",
    )

    papers = await source.search(query)

    assert papers[0].citation_count is None


@pytest.mark.asyncio
async def test_academic_source_maps_stable_identity_and_abstract_status() -> None:
    source = AcademicSource(tool=FakeLegacyTool([
        {
            "source": "semantic_scholar",
            "paper_id": "legacy-123",
            "doi": "https://doi.org/10.2000/ABC",
            "title": "Academic Hippo study",
            "abstract": "Real academic abstract.",
        },
        {
            "doi": "https://doi.org/10.3000/XYZ",
            "title": "Academic study without abstract",
            "abstract": "   ",
        },
    ]))
    query = (await QueryPlanner(
        FakeStructuredClient({"queries": [{
            "text": "Hippo", "tool": "semantic_scholar",
            "purpose": "supporting_evidence", "reasoning": "test",
        }]}), LiteratureSearchTool.TOOL_DEFINITIONS,
    ).plan("Hippo", state=SearchState()))[0]

    papers = await source.search(query)

    assert [paper.doi for paper in papers] == [
        "10.2000/abc", "10.3000/xyz"
    ]
    assert [paper.paper_id for paper in papers] == [
        "S2:legacy-123", "DOI:10.3000/xyz"
    ]
    assert [paper.fulltext_status for paper in papers] == [
        FulltextStatus.ABSTRACT_ONLY, FulltextStatus.UNKNOWN
    ]


@pytest.mark.asyncio
async def test_academic_source_preserves_s2_pubmed_and_oa_pdf_metadata() -> None:
    source = AcademicSource(tool=FakeLegacyTool([{
        "source": "semantic_scholar",
        "paper_id": "legacy-oa",
        "title": "Open manipulator study",
        "abstract": "Direct manipulation evidence.",
        "pmid": "12345",
        "pmcid": "PMC12345",
        "external_ids": {
            "PubMed": "12345",
            "PubMedCentral": "PMC12345",
        },
        "oa_pdf_url": "https://example.test/open.pdf",
        "is_open_access": True,
    }]))
    query = SearchQuery(
        query_id="q-oa",
        text="robot manipulation sim-to-real",
        target_source="semantic_scholar",
        purpose="method",
        relation_to_question="direct",
    )

    paper = (await source.search(query))[0]

    assert paper.pmid == "12345"
    assert paper.pmcid == "PMC12345"
    assert paper.external_ids["oa_pdf_url"] == "https://example.test/open.pdf"
    assert paper.is_open_access is True
    assert paper.fulltext_status is FulltextStatus.PDF_AVAILABLE


@pytest.mark.asyncio
async def test_sources_skip_completely_unidentifiable_rows() -> None:
    pubmed = PubMedSource(tool=FakeLegacyTool([{}]))
    academic = AcademicSource(tool=FakeLegacyTool([{}]))
    planner = QueryPlanner(FakeStructuredClient({"queries": [
        {"text": "Hippo", "tool": "pubmed", "purpose": "core",
         "reasoning": "test"},
        {"text": "Hippo", "tool": "semantic_scholar", "purpose": "core",
         "reasoning": "test"},
    ]}), LiteratureSearchTool.TOOL_DEFINITIONS)
    planned = await planner.plan("Hippo", state=SearchState())
    by_source = {item.target_source: item for item in planned}
    pubmed_query = by_source["pubmed"]
    academic_query = by_source["semantic_scholar"]

    assert await pubmed.search(pubmed_query) == []
    assert await academic.search(academic_query) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("source_type", [PubMedSource, AcademicSource])
async def test_sources_prefer_injected_strict_search_and_propagate_failures(
    source_type,
) -> None:
    tool = FakeStrictTool(OSError("provider unavailable"))
    source = source_type(tool=tool)
    target = source.source_name
    query = (await QueryPlanner(
        FakeStructuredClient({"queries": [{
            "text": "Hippo", "tool": target,
            "purpose": "core", "reasoning": "test",
        }]}), LiteratureSearchTool.TOOL_DEFINITIONS,
    ).plan("Hippo", state=SearchState()))[0]

    with pytest.raises(OSError, match="provider unavailable"):
        await source.search(query)

    assert tool.legacy_called is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("source_type", "tool_type"),
    [
        (PubMedSource, PubMedTool),
        (AcademicSource, SemanticScholarTool),
    ],
)
async def test_default_sources_use_strict_tool_path(
    monkeypatch, source_type, tool_type
) -> None:
    def strict_failure(*args, **kwargs):
        raise OSError("default provider unavailable")

    async def legacy_search_must_not_run(self, query: str, limit: int = 20):
        raise AssertionError("legacy swallowing path was used")

    if tool_type is PubMedTool:
        monkeypatch.setattr(
            pubmed_search, "_search_pubmed_strict_sync", strict_failure
        )
    else:
        monkeypatch.setattr(semantic_scholar, "_search", strict_failure)
    monkeypatch.setattr(tool_type, "search", legacy_search_must_not_run)
    source = source_type()
    query = (await QueryPlanner(
        FakeStructuredClient({"queries": [{
            "text": "Hippo", "tool": source.source_name,
            "purpose": "core", "reasoning": "test",
        }]}), LiteratureSearchTool.TOOL_DEFINITIONS,
    ).plan("Hippo", state=SearchState()))[0]

    with pytest.raises(OSError, match="default provider unavailable"):
        await source.search(query)


@pytest.mark.asyncio
async def test_source_keeps_legacy_fake_search_compatibility() -> None:
    tool = FakeLegacyTool([{"pmid": "123", "title": "Legacy fake"}])
    source = PubMedSource(tool=tool)
    query = (await QueryPlanner(
        FakeStructuredClient({"queries": [{
            "text": "Hippo", "tool": "pubmed",
            "purpose": "core", "reasoning": "test",
        }]}), LiteratureSearchTool.TOOL_DEFINITIONS,
    ).plan("Hippo", state=SearchState()))[0]

    papers = await source.search(query, limit=3)

    assert [paper.paper_id for paper in papers] == ["PMID:123"]
    assert tool.calls == [("Hippo", 3)]


@pytest.mark.asyncio
async def test_academic_source_offloads_blocking_strict_backend(
    monkeypatch,
) -> None:
    entered = threading.Event()
    release = threading.Event()
    worker_threads = []

    def blocking_search(query: str, limit: int):
        worker_threads.append(threading.get_ident())
        entered.set()
        release.wait(timeout=1)
        return []

    monkeypatch.setattr(semantic_scholar, "_search", blocking_search)
    source = AcademicSource(tool=SemanticScholarTool())
    query = (await QueryPlanner(
        FakeStructuredClient({"queries": [{
            "text": "Hippo", "tool": "semantic_scholar",
            "purpose": "core", "reasoning": "test",
        }]}), LiteratureSearchTool.TOOL_DEFINITIONS,
    ).plan("Hippo", state=SearchState()))[0]
    main_thread = threading.get_ident()
    started = time.monotonic()

    try:
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(source.search(query), timeout=0.02)
    finally:
        release.set()

    assert time.monotonic() - started < 0.2
    assert entered.is_set()
    assert worker_threads != [main_thread]
