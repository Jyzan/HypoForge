# M2 Minimal PubMed Tools Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an explicit M2-only path that searches real PubMed and completes the existing agentic M2 workflow with deterministic minimal-rule Tools.

**Architecture:** A strict asynchronous PubMed helper feeds a `LiteratureSourceProtocol` adapter. Small deterministic implementations provide query planning, deduplication, ranking, Scout Reading, coverage, and abstract-only evidence extraction; a factory injects them into the existing `IterativeSearchAgent` and `AgenticM2Adapter`. A separate CLI invokes M2 directly without M1 and never fabricates or silently falls back.

**Tech Stack:** Python 3.11+, asyncio, urllib/NCBI E-utilities, Pydantic v2, pytest, pytest-asyncio.

## Global Constraints

- Search only PubMed; do not add Semantic Scholar, OpenAlex, Europe PMC, or another source.
- Return only real PubMed metadata, explicit empty results, or explicit errors; never fabricate papers.
- Do not call Qwen or any other LLM.
- Preserve `M2LiteratureSearch` legacy defaults and the existing requirement that unconfigured `agentic` mode receives an injected adapter.
- Do not implement PDF download, full-text parsing, vector storage, RAG, citation chaining, MeSH expansion, or multi-round gap refinement.
- Default automated tests must not access the network; inject PubMed-shaped responses and exceptions.
- Never commit `.env`, API keys, PubMed response caches, output JSON, PDFs, or full text.
- Use `C:\Users\LIU\Desktop\挑战杯\核心代码\HypoForge\.venv\Scripts\python.exe` for all Python and pytest commands in this worktree.

---

## File Structure

- Modify `hypoforge/tools/pubmed_search.py`: expose a strict, threaded PubMed search helper while preserving legacy swallowing behavior.
- Create `hypoforge/literature/sources/pubmed.py`: map strict PubMed dictionaries to `PaperRecord` and propagate source failures.
- Modify `hypoforge/literature/sources/__init__.py`: export `PubMedLiteratureSource`.
- Create `hypoforge/literature/minimal.py`: deterministic planner, deduplicator, ranker, Scout, coverage evaluator, abstract reader, and assembly factory.
- Create `scripts/run_m2_pubmed.py`: M2-only JSON CLI.
- Create `tests/test_pubmed_search_strict.py`: strict helper and legacy compatibility tests.
- Create `tests/literature/test_pubmed_source.py`: source mapping, malformed data, and failure tests.
- Create `tests/literature/test_minimal_tools.py`: deterministic Tool and end-to-end M2 tests.
- Create `tests/test_run_m2_pubmed.py`: CLI success/error tests without network.
- Modify `hypoforge/literature/README.md`: document explicit minimal usage and limitations.

---

### Task 1: Strict PubMed Search Helper

**Files:**
- Modify: `hypoforge/tools/pubmed_search.py`
- Create: `tests/test_pubmed_search_strict.py`

**Interfaces:**
- Consumes: existing `_esearch(query: str, limit: int) -> List[str]` and `_efetch_batch(pmids: List[str]) -> List[Dict[str, Any]]`.
- Produces: `async search_pubmed_strict(query: str, limit: int = 20) -> List[dict]` that propagates every source exception.
- Preserves: `PubMedTool.search(...)` continues returning `[]` on source failures.

- [ ] **Step 1: Write failing strict-helper tests**

Create `tests/test_pubmed_search_strict.py`:

```python
from __future__ import annotations

import pytest

from hypoforge.tools import pubmed_search


@pytest.mark.asyncio
async def test_strict_pubmed_search_fetches_metadata_in_a_thread(monkeypatch) -> None:
    monkeypatch.setattr(pubmed_search, "_esearch", lambda query, limit: ["123"])
    monkeypatch.setattr(
        pubmed_search,
        "_efetch_batch",
        lambda pmids: [{"pmid": "123", "title": "A paper"}],
    )

    result = await pubmed_search.search_pubmed_strict("Hippo YAP", limit=3)

    assert result == [{"pmid": "123", "title": "A paper"}]


@pytest.mark.asyncio
async def test_strict_pubmed_search_propagates_network_failure(monkeypatch) -> None:
    def fail(query: str, limit: int):
        raise OSError("network unavailable")

    monkeypatch.setattr(pubmed_search, "_esearch", fail)

    with pytest.raises(OSError, match="network unavailable"):
        await pubmed_search.search_pubmed_strict("Hippo YAP")


@pytest.mark.asyncio
async def test_legacy_pubmed_tool_still_returns_empty_on_failure(monkeypatch) -> None:
    def fail(query: str, limit: int):
        raise OSError("network unavailable")

    monkeypatch.setattr(pubmed_search, "_esearch", fail)

    assert await pubmed_search.PubMedTool().search("Hippo YAP") == []
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
& 'C:\Users\LIU\Desktop\挑战杯\核心代码\HypoForge\.venv\Scripts\python.exe' -m pytest tests/test_pubmed_search_strict.py -q
```

Expected: collection or attribute failure because `search_pubmed_strict` does not exist.

- [ ] **Step 3: Add the strict threaded helper**

In `hypoforge/tools/pubmed_search.py`, add `import asyncio`, then add:

```python
def _search_pubmed_strict_sync(query: str, limit: int) -> List[dict]:
    bounded_limit = min(max(limit, 1), 100)
    pmids = _esearch(query, limit=bounded_limit)
    return _efetch_batch(pmids)


async def search_pubmed_strict(query: str, limit: int = 20) -> List[dict]:
    """Search PubMed without swallowing failures or blocking the event loop."""
    return await asyncio.to_thread(_search_pubmed_strict_sync, query, limit)
```

Do not change `PubMedTool.search()` or the existing `search_pubmed()` convenience function.

- [ ] **Step 4: Run focused and legacy tests**

Run:

```powershell
& 'C:\Users\LIU\Desktop\挑战杯\核心代码\HypoForge\.venv\Scripts\python.exe' -m pytest tests/test_pubmed_search_strict.py tests/test_pipeline.py -q
```

Expected: all selected tests pass; the legacy failure test returns `[]`.

- [ ] **Step 5: Commit Task 1**

```powershell
git add hypoforge/tools/pubmed_search.py tests/test_pubmed_search_strict.py
git commit -m "feat(m2): expose strict PubMed search"
```

---

### Task 2: PubMed Literature Source Adapter

**Files:**
- Create: `hypoforge/literature/sources/pubmed.py`
- Modify: `hypoforge/literature/sources/__init__.py`
- Create: `tests/literature/test_pubmed_source.py`

**Interfaces:**
- Consumes: `SearchQuery`, `PaperRecord`, `LiteratureSourceProtocol`, and `search_pubmed_strict(query, limit)`.
- Produces: `PubMedLiteratureSource(backend=None)` with `source_name = "pubmed"` and `async search(query, limit) -> list[PaperRecord]`.
- Injectable backend signature: `Callable[[str, int], Awaitable[Sequence[Mapping[str, Any]]]]`.

- [ ] **Step 1: Write failing adapter tests**

Create `tests/literature/test_pubmed_source.py`:

```python
from __future__ import annotations

import pytest

from hypoforge.literature.models import FulltextStatus, QueryIntent, SearchQuery
from hypoforge.literature.sources.pubmed import PubMedLiteratureSource


def query() -> SearchQuery:
    return SearchQuery(
        query_id="q-1",
        text="Hippo AND YAP AND TAZ",
        intent=QueryIntent.CORE,
        target_source="pubmed",
        purpose="Find direct PubMed evidence",
        relation_to_question="Uses named pathway entities",
    )


@pytest.mark.asyncio
async def test_pubmed_source_maps_real_shaped_metadata() -> None:
    calls = []

    async def backend(text: str, limit: int):
        calls.append((text, limit))
        return [{
            "pmid": "123",
            "title": "Hippo signaling controls organ size",
            "abstract": "Hippo signaling regulates YAP activity.",
            "authors": ["Example A"],
            "year": 2024,
            "journal": "Example Journal",
            "doi": "https://doi.org/10.1/example",
        }]

    result = await PubMedLiteratureSource(backend=backend).search(query(), limit=5)

    assert calls == [("Hippo AND YAP AND TAZ", 5)]
    assert len(result) == 1
    assert result[0].paper_id == "PMID:123"
    assert result[0].pmid == "123"
    assert result[0].doi == "10.1/example"
    assert result[0].sources == ["pubmed"]
    assert result[0].fulltext_status is FulltextStatus.ABSTRACT_ONLY


@pytest.mark.asyncio
async def test_pubmed_source_propagates_backend_failure() -> None:
    async def backend(text: str, limit: int):
        raise OSError("network unavailable")

    with pytest.raises(OSError, match="network unavailable"):
        await PubMedLiteratureSource(backend=backend).search(query())


@pytest.mark.asyncio
async def test_pubmed_source_skips_only_unidentifiable_records() -> None:
    async def backend(text: str, limit: int):
        return [
            {"pmid": "", "doi": "", "title": ""},
            {"pmid": "", "doi": "10.2/doi-only", "title": "DOI paper"},
            {"pmid": "", "doi": "", "title": "Title only paper"},
        ]

    result = await PubMedLiteratureSource(backend=backend).search(query())

    assert len(result) == 2
    assert result[0].paper_id == "DOI:10.2/doi-only"
    assert result[1].paper_id.startswith("TITLE:")
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
& 'C:\Users\LIU\Desktop\挑战杯\核心代码\HypoForge\.venv\Scripts\python.exe' -m pytest tests/literature/test_pubmed_source.py -q
```

Expected: import failure because `hypoforge.literature.sources.pubmed` does not exist.

- [ ] **Step 3: Implement metadata conversion and strict source**

Create `hypoforge/literature/sources/pubmed.py` with these public definitions:

```python
from __future__ import annotations

import hashlib
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

from ...tools.pubmed_search import search_pubmed_strict
from ..models import FulltextStatus, PaperRecord, SearchQuery
from ..protocols import LiteratureSourceProtocol

PubMedBackend = Callable[
    [str, int],
    Awaitable[Sequence[Mapping[str, Any]]],
]


def _normalized_title(value: object) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).split())


def _paper_id(pmid: str, doi: str, title: str) -> str:
    if pmid:
        return f"PMID:{pmid}"
    if doi:
        return f"DOI:{doi}"
    digest = hashlib.sha256(_normalized_title(title).encode("utf-8")).hexdigest()[:16]
    return f"TITLE:{digest}"


def _to_paper_record(raw: Mapping[str, Any]) -> PaperRecord | None:
    pmid = str(raw.get("pmid") or "").strip()
    doi = str(raw.get("doi") or "").strip()
    title = str(raw.get("title") or "").strip()
    if not pmid and not doi and not title:
        return None
    abstract = str(raw.get("abstract") or "").strip()
    year_value = raw.get("year")
    try:
        parsed_year = int(year_value) if year_value else 0
    except (TypeError, ValueError):
        parsed_year = 0
    year = parsed_year if 1000 <= parsed_year <= 3000 else None
    return PaperRecord(
        paper_id=_paper_id(pmid, doi, title),
        title=title or doi or f"PubMed {pmid}",
        abstract=abstract,
        authors=[str(item) for item in raw.get("authors") or [] if str(item).strip()],
        year=year,
        journal=str(raw.get("journal") or "").strip(),
        doi=doi,
        pmid=pmid,
        sources=["pubmed"],
        fulltext_status=(
            FulltextStatus.ABSTRACT_ONLY if abstract else FulltextStatus.UNKNOWN
        ),
    )


class PubMedLiteratureSource(LiteratureSourceProtocol):
    source_name = "pubmed"

    def __init__(self, backend: PubMedBackend | None = None) -> None:
        self.backend = backend or search_pubmed_strict

    async def search(self, query: SearchQuery, limit: int = 20) -> list[PaperRecord]:
        rows = await self.backend(query.text, limit)
        records = (_to_paper_record(row) for row in rows)
        return [record for record in records if record is not None]
```

Modify `hypoforge/literature/sources/__init__.py` to export both
`PubMedBackend` and `PubMedLiteratureSource`.

- [ ] **Step 4: Run adapter and model tests**

Run:

```powershell
& 'C:\Users\LIU\Desktop\挑战杯\核心代码\HypoForge\.venv\Scripts\python.exe' -m pytest tests/literature/test_pubmed_source.py tests/literature/test_models.py tests/literature/test_protocols.py -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit Task 2**

```powershell
git add hypoforge/literature/sources tests/literature/test_pubmed_source.py
git commit -m "feat(m2): adapt PubMed to literature source protocol"
```

---

### Task 3: Deterministic Minimal Search Tools

**Files:**
- Create: `hypoforge/literature/minimal.py`
- Create: `tests/literature/test_minimal_tools.py`

**Interfaces:**
- Produces: `RuleBasedQueryPlanner`, `ExactPaperDeduplicator`, `MetadataPaperRanker`, `AbstractScoutReader`, and `SingleSourceCoverageEvaluator`.
- Consumes: existing Tool Protocols and Pydantic literature models.
- Shared helper: `_latin_terms(text: str, limit: int = 8) -> list[str]`.

- [ ] **Step 1: Write failing planner, dedup, rank, Scout, and coverage tests**

Create `tests/literature/test_minimal_tools.py` with imports and these tests:

```python
from __future__ import annotations

import pytest

from hypoforge.literature.minimal import (
    AbstractScoutReader,
    ExactPaperDeduplicator,
    MetadataPaperRanker,
    RuleBasedQueryPlanner,
    SingleSourceCoverageEvaluator,
)
from hypoforge.literature.models import (
    CoverageReport,
    EvidenceBucket,
    PaperRecord,
    SearchState,
)


def paper(paper_id: str, *, title: str, abstract: str = "", year: int | None = None,
          pmid: str = "", doi: str = "") -> PaperRecord:
    return PaperRecord(
        paper_id=paper_id,
        title=title,
        abstract=abstract,
        year=year,
        pmid=pmid,
        doi=doi,
        sources=["pubmed"],
    )


@pytest.mark.asyncio
async def test_rule_planner_builds_one_entity_query_then_stops() -> None:
    planner = RuleBasedQueryPlanner()
    first = await planner.plan(
        "Hippo–YAP/TAZ如何限制器官大小？",
        key_entities=["Hippo", "YAP", "TAZ"],
        state=SearchState(),
    )
    second = await planner.plan(
        "Hippo–YAP/TAZ如何限制器官大小？",
        state=SearchState(queries_used=first),
    )

    assert [item.text for item in first] == ["Hippo AND YAP AND TAZ"]
    assert first[0].target_source == "pubmed"
    assert second == []


@pytest.mark.asyncio
async def test_exact_deduplicator_reuses_catalog_by_pmid_doi_and_title() -> None:
    canonical = paper("catalog", title="Hippo Signaling", pmid="123")
    incoming = [
        paper("new-pmid", title="Other", pmid="123"),
        paper("new-doi", title="DOI copy", doi="10.1/same"),
        paper("new-doi-2", title="Different", doi="10.1/same"),
        paper("new-title", title="  Hippo signaling!  "),
    ]

    result = await ExactPaperDeduplicator().deduplicate(
        incoming,
        existing_papers=[canonical],
    )

    assert result[0] is canonical
    assert [item.paper_id for item in result] == ["catalog", "new-doi"]


@pytest.mark.asyncio
async def test_metadata_ranker_prefers_abstract_then_recent_year() -> None:
    ranked = await MetadataPaperRanker().rank(
        "question",
        [
            paper("old", title="Old", abstract="Evidence.", year=2010),
            paper("none", title="No abstract", year=2025),
            paper("new", title="New", abstract="Evidence.", year=2024),
        ],
        limit=2,
    )

    assert [item.paper_id for item in ranked] == ["new", "old"]
    assert ranked[0].rank_scores["has_abstract"] == 1.0


@pytest.mark.asyncio
async def test_scout_and_coverage_return_bounded_deterministic_output() -> None:
    papers = [paper(
        "p1",
        title="Hippo YAP signaling",
        abstract="TAZ controls mechanotransduction and organ size.",
    )]
    notes = await AbstractScoutReader().read("question", papers)
    report = await SingleSourceCoverageEvaluator().evaluate(
        "question", papers, notes, SearchState()
    )

    assert notes[0].paper_id == "p1"
    assert {"Hippo", "YAP", "TAZ"}.issubset(set(notes[0].key_terms))
    assert report.sufficient is True
    assert EvidenceBucket.SUPPORTING in report.covered_buckets


@pytest.mark.asyncio
async def test_coverage_is_insufficient_without_papers() -> None:
    report = await SingleSourceCoverageEvaluator().evaluate(
        "question", [], [], SearchState()
    )
    assert report == CoverageReport(
        missing_buckets={EvidenceBucket.SUPPORTING},
        missing_topics=["PubMed evidence"],
        sufficient=False,
        rationale="PubMed returned no valid candidate papers.",
    )
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
& 'C:\Users\LIU\Desktop\挑战杯\核心代码\HypoForge\.venv\Scripts\python.exe' -m pytest tests/literature/test_minimal_tools.py -q
```

Expected: import failure because `hypoforge.literature.minimal` does not exist.

- [ ] **Step 3: Implement the deterministic search Tools**

Create `hypoforge/literature/minimal.py`. Use this bounded token helper and
class behavior:

```python
from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

from .models import (
    CoverageReport,
    EvidenceBucket,
    PaperRecord,
    QueryIntent,
    ScoutNote,
    SearchQuery,
    SearchState,
)
from .protocols import (
    CoverageEvaluatorProtocol,
    PaperDeduplicatorProtocol,
    PaperRankerProtocol,
    QueryPlannerProtocol,
    ScoutReaderProtocol,
)

_STOP_WORDS = {
    "and", "are", "does", "for", "from", "how", "into", "the", "what",
    "when", "where", "which", "with",
}


def _latin_terms(text: str, limit: int = 8) -> list[str]:
    found = re.findall(r"[A-Za-z][A-Za-z0-9-]{1,}", text)
    output: list[str] = []
    seen: set[str] = set()
    for term in found:
        key = term.casefold()
        if key in _STOP_WORDS or key in seen:
            continue
        seen.add(key)
        output.append(term)
        if len(output) >= limit:
            break
    return output
```

Implement the classes with these exact rules:

```python
class RuleBasedQueryPlanner(QueryPlannerProtocol):
    async def plan(self, sub_question: str, key_entities: Sequence[str] = (),
                   domains: Sequence[str] = (), question_type: str = "",
                   state: SearchState | None = None) -> list[SearchQuery]:
        if state and state.queries_used:
            return []
        terms = _latin_terms(" ".join([*key_entities, sub_question]), limit=6)
        text = " AND ".join(terms) if terms else sub_question.strip()
        digest = hashlib.sha256(text.casefold().encode("utf-8")).hexdigest()[:12]
        return [SearchQuery(
            query_id=f"pubmed-{digest}", text=text, intent=QueryIntent.CORE,
            target_source="pubmed", purpose="Find direct PubMed evidence",
            relation_to_question="Uses explicit entities from the sub-question",
        )]
```

Add these exact normalized-key and Tool implementations:

```python
def _normalized_title(title: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", title.casefold()).split())


def _paper_keys(paper: PaperRecord) -> list[str]:
    keys: list[str] = []
    if paper.pmid:
        keys.append(f"pmid:{paper.pmid.casefold()}")
    if paper.doi:
        keys.append(f"doi:{paper.doi.casefold()}")
    title = _normalized_title(paper.title)
    if title:
        keys.append(f"title:{title}")
    return keys


class ExactPaperDeduplicator(PaperDeduplicatorProtocol):
    async def deduplicate(
        self,
        papers: Sequence[PaperRecord],
        existing_papers: Sequence[PaperRecord] = (),
    ) -> list[PaperRecord]:
        canonical_by_key: dict[str, PaperRecord] = {}
        for paper in existing_papers:
            for key in _paper_keys(paper):
                canonical_by_key.setdefault(key, paper)

        output: list[PaperRecord] = []
        returned_ids: set[str] = set()
        for paper in papers:
            canonical = next(
                (canonical_by_key[key] for key in _paper_keys(paper)
                 if key in canonical_by_key),
                paper,
            )
            for key in [*_paper_keys(paper), *_paper_keys(canonical)]:
                canonical_by_key.setdefault(key, canonical)
            if canonical.paper_id not in returned_ids:
                returned_ids.add(canonical.paper_id)
                output.append(canonical)
        return output


class MetadataPaperRanker(PaperRankerProtocol):
    async def rank(
        self,
        sub_question: str,
        papers: Sequence[PaperRecord],
        limit: int,
    ) -> list[PaperRecord]:
        indexed = list(enumerate(papers))
        indexed.sort(key=lambda item: (
            0 if item[1].abstract.strip() else 1,
            -(item[1].year or 0),
            item[0],
        ))
        output: list[PaperRecord] = []
        for _, paper in indexed[:limit]:
            scores = dict(paper.rank_scores)
            scores.update({
                "has_abstract": 1.0 if paper.abstract.strip() else 0.0,
                "publication_year": float(paper.year or 0),
            })
            output.append(paper.model_copy(update={"rank_scores": scores}))
        return output


class AbstractScoutReader(ScoutReaderProtocol):
    async def read(
        self,
        sub_question: str,
        papers: Sequence[PaperRecord],
    ) -> list[ScoutNote]:
        notes: list[ScoutNote] = []
        for paper in papers:
            terms = _latin_terms(f"{paper.title} {paper.abstract}", limit=8)
            notes.append(ScoutNote(
                paper_id=paper.paper_id,
                main_topic=paper.title,
                key_terms=terms,
                entities=terms,
                relevance_to_question=0.8 if terms else 0.2,
            ))
        return notes


class SingleSourceCoverageEvaluator(CoverageEvaluatorProtocol):
    async def evaluate(
        self,
        sub_question: str,
        papers: Sequence[PaperRecord],
        scout_notes: Sequence[ScoutNote],
        state: SearchState,
    ) -> CoverageReport:
        return CoverageReport(
            covered_buckets={EvidenceBucket.SUPPORTING} if papers else set(),
            missing_buckets=(
                set() if papers else {EvidenceBucket.SUPPORTING}
            ),
            covered_topics=["PubMed evidence"] if papers else [],
            missing_topics=[] if papers else ["PubMed evidence"],
            sufficient=bool(papers),
            rationale=(
                "At least one valid PubMed candidate is available for workflow testing."
                if papers else "PubMed returned no valid candidate papers."
            ),
        )
```

- [ ] **Step 4: Run focused tests**

Run:

```powershell
& 'C:\Users\LIU\Desktop\挑战杯\核心代码\HypoForge\.venv\Scripts\python.exe' -m pytest tests/literature/test_minimal_tools.py tests/literature/test_search_agent.py -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit Task 3**

```powershell
git add hypoforge/literature/minimal.py tests/literature/test_minimal_tools.py
git commit -m "feat(m2): add deterministic minimal search tools"
```

---

### Task 4: Abstract Reading Workflow and Assembly Factory

**Files:**
- Modify: `hypoforge/literature/minimal.py`
- Modify: `tests/literature/test_minimal_tools.py`

**Interfaces:**
- Produces: `AbstractReadingWorkflow` implementing `ReadingExtractionWorkflowProtocol`.
- Produces: `build_minimal_pubmed_adapter(*, backend=None, final_k: int = 5, source_timeout_seconds: float = 30.0) -> AgenticM2Adapter`.
- Consumes: `PubMedBackend`, `PubMedLiteratureSource`, existing Search Agent, Adapter, and deterministic search Tools.

- [ ] **Step 1: Add failing reading and end-to-end M2 tests**

Append to `tests/literature/test_minimal_tools.py`:

```python
from hypoforge.literature.minimal import (
    AbstractReadingWorkflow,
    build_minimal_pubmed_adapter,
)
from hypoforge.modules.m2_literature_search import M2LiteratureSearch
from hypoforge.state import PipelineState


@pytest.mark.asyncio
async def test_abstract_reader_links_one_real_abstract_sentence() -> None:
    source = paper(
        "PMID:123",
        title="Hippo signaling",
        abstract="Hippo signaling restrains YAP activity. A second sentence.",
        pmid="123",
    )

    result = (await AbstractReadingWorkflow().run("question", [source]))[0]

    assert result.degraded_to_abstract is True
    assert result.evidence[0].quote == "Hippo signaling restrains YAP activity."
    assert result.knowledge_entries[0].evidence_ids == [result.evidence[0].evidence_id]
    assert "PubMed abstract reports" in result.knowledge_entries[0].content


@pytest.mark.asyncio
async def test_abstract_reader_does_not_invent_content_when_abstract_missing() -> None:
    source = paper("PMID:123", title="No abstract", pmid="123")

    result = (await AbstractReadingWorkflow().run("question", [source]))[0]

    assert result.degraded_to_abstract is True
    assert result.evidence == []
    assert result.knowledge_entries == []
    assert result.errors == ["PubMed abstract unavailable"]


@pytest.mark.asyncio
async def test_factory_runs_m2_directly_with_real_shaped_backend() -> None:
    async def backend(text: str, limit: int):
        return [{
            "pmid": "123",
            "title": "Hippo signaling",
            "abstract": "Hippo signaling restrains YAP activity.",
            "year": 2024,
        }]

    module = M2LiteratureSearch(
        implementation="agentic",
        agentic_adapter=build_minimal_pubmed_adapter(backend=backend, final_k=3),
    )

    output = await module(PipelineState(input_question="Hippo YAP TAZ organ size"))

    result = output["literature_results"][0]
    assert result.sub_question == "Hippo YAP TAZ organ size"
    assert result.papers_retrieved == 1
    assert result.knowledge_entries[0].source_paper_id == "PMID:123"


@pytest.mark.asyncio
async def test_factory_returns_empty_result_when_pubmed_returns_none() -> None:
    async def backend(text: str, limit: int):
        return []

    module = M2LiteratureSearch(
        implementation="agentic",
        agentic_adapter=build_minimal_pubmed_adapter(backend=backend),
    )

    output = await module(PipelineState(input_question="no matching topic"))

    result = output["literature_results"][0]
    assert result.papers_retrieved == 0
    assert result.knowledge_entries == []


@pytest.mark.asyncio
async def test_factory_surfaces_pubmed_network_failure() -> None:
    async def backend(text: str, limit: int):
        raise OSError("network unavailable")

    module = M2LiteratureSearch(
        implementation="agentic",
        agentic_adapter=build_minimal_pubmed_adapter(backend=backend),
    )

    with pytest.raises(RuntimeError, match="network unavailable"):
        await module(PipelineState(input_question="question"))
```

- [ ] **Step 2: Run new tests and verify RED**

Run:

```powershell
& 'C:\Users\LIU\Desktop\挑战杯\核心代码\HypoForge\.venv\Scripts\python.exe' -m pytest tests/literature/test_minimal_tools.py -q
```

Expected: import failures for `AbstractReadingWorkflow` and
`build_minimal_pubmed_adapter`.

- [ ] **Step 3: Implement abstract extraction**

Add imports for `AgenticM2Adapter`, `IterativeSearchAgent`,
`PubMedLiteratureSource`, `PubMedBackend`, `ConfidenceLevel`, and
`KnowledgeEntryType`. Implement:

```python
class AbstractReadingWorkflow(ReadingExtractionWorkflowProtocol):
    async def run(self, sub_question: str,
                  papers: Sequence[PaperRecord]) -> list[PaperReadingResult]:
        results: list[PaperReadingResult] = []
        for paper in papers:
            abstract = " ".join(paper.abstract.split())
            if not abstract:
                results.append(PaperReadingResult(
                    paper_id=paper.paper_id,
                    summary=paper.title,
                    degraded_to_abstract=True,
                    errors=["PubMed abstract unavailable"],
                ))
                continue
            sentence = re.split(r"(?<=[.!?。！？])\s+", abstract, maxsplit=1)[0][:600]
            evidence_id = f"{paper.paper_id}:abstract:1"
            evidence = EvidenceChunk(
                evidence_id=evidence_id,
                paper_id=paper.paper_id,
                chunk_id=f"{paper.paper_id}:abstract",
                section="abstract",
                quote=sentence,
                normalized_claim=sentence,
                relevance_score=0.7,
            )
            digest = hashlib.sha256(evidence_id.encode("utf-8")).hexdigest()[:12]
            entry = EvidenceLinkedKnowledge(
                entry_id=f"pubmed-{digest}",
                entry_type=KnowledgeEntryType.ESTABLISHED_FACT,
                content=f"PubMed abstract reports: {sentence}",
                confidence=ConfidenceLevel.MEDIUM,
                entities=_latin_terms(f"{sub_question} {paper.title}", limit=6),
                evidence_ids=[evidence_id],
            )
            results.append(PaperReadingResult(
                paper_id=paper.paper_id,
                summary=sentence,
                evidence=[evidence],
                knowledge_entries=[entry],
                degraded_to_abstract=True,
            ))
        return results
```

- [ ] **Step 4: Implement the explicit factory**

Add:

```python
def build_minimal_pubmed_adapter(
    *,
    backend: PubMedBackend | None = None,
    final_k: int = 5,
    source_timeout_seconds: float = 30.0,
) -> AgenticM2Adapter:
    source = PubMedLiteratureSource(backend=backend)
    agent = IterativeSearchAgent(
        query_planner=RuleBasedQueryPlanner(),
        sources=[source],
        deduplicator=ExactPaperDeduplicator(),
        ranker=MetadataPaperRanker(),
        scout_reader=AbstractScoutReader(),
        coverage_evaluator=SingleSourceCoverageEvaluator(),
        final_k=final_k,
        candidate_limit=final_k,
        per_query_limit=final_k,
        source_timeout_seconds=source_timeout_seconds,
    )
    return AgenticM2Adapter(
        search_agent=agent,
        reading_workflow=AbstractReadingWorkflow(),
        budget=SearchBudget(max_rounds=2, max_queries=2, max_papers=final_k),
    )
```

Validate `final_k > 0` before constructing `SearchBudget`; raise
`ValueError("final_k must be positive")` otherwise.

- [ ] **Step 5: Run end-to-end M2 tests**

Run:

```powershell
& 'C:\Users\LIU\Desktop\挑战杯\核心代码\HypoForge\.venv\Scripts\python.exe' -m pytest tests/literature/test_minimal_tools.py tests/literature/test_adapter.py tests/literature/test_search_agent.py -q
```

Expected: all selected tests pass; success, zero-result, missing-abstract, and
network-error paths are distinguished.

- [ ] **Step 6: Commit Task 4**

```powershell
git add hypoforge/literature/minimal.py tests/literature/test_minimal_tools.py
git commit -m "feat(m2): assemble minimal PubMed workflow"
```

---

### Task 5: M2-only CLI, Documentation, and Final Verification

**Files:**
- Create: `scripts/run_m2_pubmed.py`
- Create: `tests/test_run_m2_pubmed.py`
- Modify: `hypoforge/literature/README.md`

**Interfaces:**
- Produces CLI: `python scripts/run_m2_pubmed.py --question <text> --limit 5 --timeout 30`.
- Success JSON: `{"status": "ok", "literature_results": [...]}` and exit code `0`.
- Error JSON on stderr: `{"status": "error", "error_type": <class>, "message": <sanitized message>}` and exit code `1`.

- [ ] **Step 1: Write failing CLI tests with injected adapter factory**

Create `tests/test_run_m2_pubmed.py`:

```python
from __future__ import annotations

import json

from scripts import run_m2_pubmed


class SuccessfulAdapter:
    async def __call__(self, state, config=None):
        return {"literature_results": []}


class FailingAdapter:
    async def __call__(self, state, config=None):
        raise RuntimeError("network unavailable")


def test_cli_prints_utf8_success_json(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        run_m2_pubmed,
        "build_minimal_pubmed_adapter",
        lambda **kwargs: SuccessfulAdapter(),
    )

    exit_code = run_m2_pubmed.main(["--question", "Hippo通路"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload == {"status": "ok", "literature_results": []}


def test_cli_prints_error_json_and_returns_nonzero(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        run_m2_pubmed,
        "build_minimal_pubmed_adapter",
        lambda **kwargs: FailingAdapter(),
    )

    exit_code = run_m2_pubmed.main(["--question", "question"])

    payload = json.loads(capsys.readouterr().err)
    assert exit_code == 1
    assert payload["status"] == "error"
    assert payload["error_type"] == "RuntimeError"
    assert payload["message"] == "network unavailable"
```

- [ ] **Step 2: Run CLI tests and verify RED**

Run:

```powershell
& 'C:\Users\LIU\Desktop\挑战杯\核心代码\HypoForge\.venv\Scripts\python.exe' -m pytest tests/test_run_m2_pubmed.py -q
```

Expected: import failure because `scripts/run_m2_pubmed.py` does not exist.

- [ ] **Step 3: Implement the M2-only CLI**

Create `scripts/run_m2_pubmed.py`:

```python
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hypoforge.literature.minimal import build_minimal_pubmed_adapter
from hypoforge.modules.m2_literature_search import M2LiteratureSearch
from hypoforge.state import PipelineState


async def _run(question: str, limit: int, timeout: float) -> dict:
    adapter = build_minimal_pubmed_adapter(
        final_k=limit,
        source_timeout_seconds=timeout,
    )
    module = M2LiteratureSearch(
        implementation="agentic",
        agentic_adapter=adapter,
    )
    output = await module(PipelineState(input_question=question))
    return {
        "status": "ok",
        "literature_results": [
            item.model_dump(mode="json")
            for item in output["literature_results"]
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run minimal PubMed-only M2")
    parser.add_argument("--question", required=True)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args(argv)
    try:
        payload = asyncio.run(_run(args.question, args.limit, args.timeout))
    except Exception as exc:
        error = {
            "status": "error",
            "error_type": type(exc).__name__,
            "message": " ".join(str(exc).split())[:500],
        }
        print(json.dumps(error, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run CLI tests**

Run:

```powershell
& 'C:\Users\LIU\Desktop\挑战杯\核心代码\HypoForge\.venv\Scripts\python.exe' -m pytest tests/test_run_m2_pubmed.py -q
```

Expected: both tests pass without network access.

- [ ] **Step 5: Document explicit usage and limitations**

Append a `Minimal PubMed smoke-test path` section to
`hypoforge/literature/README.md` containing:

```markdown
## Minimal PubMed smoke-test path

This explicit development path searches only real PubMed metadata and uses
deterministic rules for the remaining M2 Tools. It does not call an LLM, read
PDFs, perform RAG, fabricate papers, or change the default legacy M2 path.

```powershell
python scripts/run_m2_pubmed.py `
  --question "Hippo YAP TAZ organ size mechanotransduction" `
  --limit 5
```

PubMed network/HTTP/parse failures produce an error and a non-zero exit code.
Zero matches produce an empty literature result. There is no synthetic
fallback.
```

- [ ] **Step 6: Run the complete offline verification suite**

Run:

```powershell
& 'C:\Users\LIU\Desktop\挑战杯\核心代码\HypoForge\.venv\Scripts\python.exe' -m pytest -q
& 'C:\Users\LIU\Desktop\挑战杯\核心代码\HypoForge\.venv\Scripts\python.exe' -m compileall -q hypoforge scripts tests
git diff --check
```

Expected: all tests pass, compileall exits `0`, and `git diff --check` prints
nothing.

- [ ] **Step 7: Run one manual real-PubMed integration test**

Run:

```powershell
& 'C:\Users\LIU\Desktop\挑战杯\核心代码\HypoForge\.venv\Scripts\python.exe' scripts/run_m2_pubmed.py --question 'Hippo YAP TAZ organ size mechanotransduction' --limit 5 --timeout 30
```

Expected when PubMed is reachable: exit `0`, `status` is `ok`,
`papers_retrieved` is at least `1`, every source paper ID begins with `PMID:`,
and every knowledge entry points to a returned paper. If PubMed is unreachable,
the command must exit `1` with `status: error`; do not replace it with fake data.

- [ ] **Step 8: Inspect the final staged scope and commit Task 5**

Run:

```powershell
git status --short
git diff --check
git diff --stat develop/m2-agentic...HEAD
git add scripts/run_m2_pubmed.py tests/test_run_m2_pubmed.py hypoforge/literature/README.md
git commit -m "feat(m2): add PubMed-only smoke test runner"
```

Expected: only the planned source, Tool, tests, CLI, and documentation changes
exist on `feat/m2-minimal-pubmed-tools`; no output or credential files appear.

---

## Final Review Gate

- [ ] Confirm the complete test suite passes after the final commit.
- [ ] Confirm `git status --short` is empty.
- [ ] Confirm `git log --oneline develop/m2-agentic..HEAD` contains only the
  design/plan and five planned implementation commits.
- [ ] Confirm no `.env`, API key, response cache, output JSON, PDF, or full text
  is tracked.
- [ ] Review the diff against
  `docs/superpowers/specs/2026-07-16-m2-minimal-pubmed-tools-design.md`.
- [ ] Do not merge or push until the user explicitly approves the reviewed
  implementation.
