"""LLM-driven query planner — generates source-aware search queries.

Implements ``QueryPlannerProtocol`` from the agentic literature skeleton.
Uses QwenClient with structured output to generate ``SearchQuery`` items
that include backend selection and coverage intent.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Sequence

from hypoforge.literature.models import QueryIntent, SearchQuery, SearchState
from hypoforge.literature.protocols import QueryPlannerProtocol
from hypoforge.tools.qwen_client import QwenClient

logger = logging.getLogger(__name__)

_FIELD_TAG_PATTERN = re.compile(r"\[[^\]]+\]")


def _sanitize_query(text: str, backend: str) -> str:
    if backend != "pubmed":
        text = _FIELD_TAG_PATTERN.sub("", text)
    return " ".join(text.split())

# ---------------------------------------------------------------------------
# Prompts (mirrored from m2_branch for standalone use)
# ---------------------------------------------------------------------------

_QUERY_PLANNING_SYSTEM = """\
You are a scientific literature search strategist. Your job is to generate \
high-quality, targeted search queries for a given research sub-question and \
select the most appropriate search tool for each query.

## Available Search Tools

{tool_descriptions}

## Query Construction Rules

### For PubMed queries (biomedical / life sciences ONLY):
- Prefer field tags for precision: `[tiab]` for title+abstract, `[MeSH Terms]` for MeSH.
- Break the sub-question into **short key concepts** (1-3 words each), then combine \
  them with AND/OR as needed. Don't put long phrases inside operators.
- **Good examples (short concepts, clear structure):**
  - `Hsp70[tiab] AND ATPase[tiab] AND protein folding[MeSH Terms]`
  - `"heat shock protein"[tiab] AND (aging OR senescence) AND proteostasis`
  - `NAD[tiab] AND chaperone[tiab] AND (aged OR aging)`
- **Bad examples:**
  - `the role of heat shock protein 70 in ATP-dependent protein folding` ← too long, no tags
  - `Hsp70 mechanism` ← single vague term

### For Semantic Scholar/OpenAlex queries (cross-disciplinary discovery):
- Use phrase quotes for multi-word terms: `"protein folding"`.
- Keep each concept short; combine with AND/OR when helpful.
- S2 syntax is simpler than PubMed — keyword search works fine.

### For arXiv queries (recent technical preprints):
- Prefer arXiv for computer science, mathematics, physics, statistics,
  electrical engineering, and other technical preprints.
- Use short keyword phrases and AND/OR; never use PubMed field tags.
- Remember that arXiv results may not have completed peer review.

## Coverage Dimensions

In the FIRST round, you MUST cover at least 4 of these 6 dimensions:
1. **core_mechanism** — central mechanism or pathway
2. **supporting_evidence** — evidence supporting established models
3. **opposing_evidence** — evidence that contradicts or challenges
4. **review** — recent review articles summing the field
5. **recent_research** — original research from the last 2-3 years
6. **methods** — key experimental methods or techniques

In later rounds, target only the gaps identified by the coverage evaluator.

## Instructions

1. Generate **4-6** queries in the first round (fewer in later rounds, targeting gaps).
2. Keep each concept/term short (1-3 words).
3. Assign each query to the most appropriate tool based on domain.
4. Never repeat a query that has already been used.
5. Never select a search tool listed as unavailable.

Return valid JSON only — no markdown fences, no extra text."""

_QUERY_PLANNING_USER_FIRST = """\
## Sub-Question
{sub_question}

## Domain & Context
- Domains: {domains}
- Key Entities: {entities}
- Question Type: {question_type}

## Current Search State
- Round: 1/{max_rounds}
- Papers found so far: 0
- Queries already used: (none)
- Current gaps identified: (none — first round)
- Unavailable search tools: {unavailable_sources}

## Task
Generate 3–6 search queries covering at least 3 different coverage dimensions.
For each query, output:
- `text`: the search query string. Use short concepts (recommend 1-3 words each, no more than 5 words), \
  field tags like `[tiab]`/`[MeSH Terms]` for PubMed, and AND/OR to combine.
- `tool`: one of the available search tools listed above
- `purpose`: which coverage dimension this query targets
- `reasoning`: why this query is needed

Output a JSON object with a `queries` array and a `reasoning` string."""

_QUERY_PLANNING_USER_RETRY = """\
## Sub-Question
{sub_question}

## Domain & Context
- Domains: {domains}
- Key Entities: {entities}
- Question Type: {question_type}

## Current Search State
- Round: {round}/{max_rounds}
- Papers found so far: {paper_count}
- Queries already used: {queries_used}
- Current gaps identified: {gaps}
- Unavailable search tools: {unavailable_sources}

## Task
Generate 2–4 targeted queries to fill the gaps above. Use the same query \
style as round 1: short concepts, field tags for PubMed, AND/OR as needed.
For each query, output `text`, `tool`, `purpose`, and `reasoning`.

Output a JSON object with a `queries` array and a `reasoning` string."""


# ---------------------------------------------------------------------------
# QueryPlanner
# ---------------------------------------------------------------------------

class QueryPlanner(QueryPlannerProtocol):
    """LLM-driven query planning with tool-aware backend selection.

    Parameters
    ----------
    client : QwenClient
        LLM client for structured generation.
    tool_definitions : list[dict]
        Tool schemas from ``LiteratureSearchTool.tool_definitions``.
    max_rounds : int
        Default max search rounds (used in prompt context).
    """

    tool_name = "query_planner"

    def __init__(
        self,
        client: QwenClient,
        tool_definitions: Optional[List[Dict[str, Any]]] = None,
        max_rounds: int = 3,
        strict: bool = False,
    ):
        self._client = client
        self._tool_definitions = tool_definitions or []
        self._max_rounds = max_rounds
        self.strict = strict
        self._valid_backends: set[str] = {
            d["name"] for d in self._tool_definitions
        }

    # ------------------------------------------------------------------
    # QueryPlannerProtocol implementation
    # ------------------------------------------------------------------

    async def plan(
        self,
        sub_question: str,
        key_entities: Sequence[str] = (),
        domains: Sequence[str] = (),
        question_type: str = "",
        state: SearchState | None = None,
    ) -> List[SearchQuery]:
        """Generate source-aware search queries.

        When ``state`` is None (first round), generates initial queries
        covering multiple evidence dimensions.  When ``state`` carries
        round/gap information from a prior iteration, the planner
        produces targeted follow-up queries.
        """
        if state is not None:
            # Iterative round — use state to inform gap-filling queries
            round_num = state.round_index + 1
            paper_count = state.unique_papers_seen
            queries_used = [
                f"[{q.target_source}] {q.text}"
                for q in state.queries_used[-20:]
            ]
            gaps = sorted(state.missing_topics)
            unavailable_sources = sorted(state.unavailable_sources)
        else:
            round_num = 1
            paper_count = 0
            queries_used = []
            gaps = []
            unavailable_sources = []

        return await self._plan_impl(
            sub_question=sub_question,
            domains=list(domains),
            entities=list(key_entities),
            question_type=question_type,
            round_num=round_num,
            paper_count=paper_count,
            queries_used=queries_used,
            gaps=gaps,
            unavailable_sources=unavailable_sources,
        )

    # ------------------------------------------------------------------
    # Extended interface — for iterative search
    # ------------------------------------------------------------------

    async def plan_next(
        self,
        sub_question: str,
        round_num: int,
        paper_count: int,
        queries_used: List[str],
        gaps: List[str],
        domains: Optional[List[str]] = None,
        entities: Optional[List[str]] = None,
        question_type: str = "",
    ) -> List[SearchQuery]:
        """Generate follow-up queries targeting specific gaps.

        Called in iterative search rounds 2+.
        """
        return await self._plan_impl(
            sub_question=sub_question,
            domains=domains or [],
            entities=entities or [],
            question_type=question_type,
            round_num=round_num,
            paper_count=paper_count,
            queries_used=queries_used,
            gaps=gaps,
            unavailable_sources=[],
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _plan_impl(
        self,
        sub_question: str,
        domains: List[str],
        entities: List[str],
        question_type: str,
        round_num: int,
        paper_count: int,
        queries_used: List[str],
        gaps: List[str],
        unavailable_sources: List[str],
    ) -> List[SearchQuery]:
        """Core planning logic — shared by plan() and plan_next()."""
        tool_descriptions = "\n\n".join(
            f"### {d['display_name']} (`{d['name']}`)\n{d['description']}"
            for d in self._tool_definitions
        )

        system = _QUERY_PLANNING_SYSTEM.format(tool_descriptions=tool_descriptions)

        if round_num <= 1 and not queries_used:
            user = _QUERY_PLANNING_USER_FIRST.format(
                sub_question=sub_question,
                domains=", ".join(domains) if domains else "unknown",
                entities=", ".join(entities) if entities else "unknown",
                question_type=question_type or "unknown",
                max_rounds=self._max_rounds,
                unavailable_sources=(
                    ", ".join(unavailable_sources) if unavailable_sources else "(none)"
                ),
            )
        else:
            user = _QUERY_PLANNING_USER_RETRY.format(
                sub_question=sub_question,
                domains=", ".join(domains) if domains else "unknown",
                entities=", ".join(entities) if entities else "unknown",
                question_type=question_type or "unknown",
                round=round_num,
                max_rounds=self._max_rounds,
                paper_count=paper_count,
                queries_used="\n".join(queries_used[-20:]) if queries_used else "(none)",
                gaps="\n".join(gaps) if gaps else "(none — first round)",
                unavailable_sources=(
                    ", ".join(unavailable_sources) if unavailable_sources else "(none)"
                ),
            )

        schema = {
            "type": "object",
            "properties": {
                "queries": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "tool": {
                                "type": "string",
                                "enum": sorted(self._valid_backends),
                            },
                            "purpose": {"type": "string"},
                            "reasoning": {"type": "string"},
                        },
                        "required": ["text", "tool", "purpose"],
                    },
                },
                "reasoning": {"type": "string"},
            },
            "required": ["queries"],
        }

        try:
            result = await self._client.structured_chat(
                system_prompt=system,
                user_prompt=user,
                output_schema=schema,
                max_tokens=4096,
                temperature=0.0,
                disable_thinking=True,
            )
        except Exception:
            logger.exception("Query planning failed for %r", sub_question[:60])
            if self.strict:
                raise
            return self._fallback_queries(sub_question, unavailable_sources)

        # Normalise
        if isinstance(result, list):
            raw_queries = result
        elif isinstance(result, dict):
            raw_queries = result.get("queries", [])
        else:
            raw_queries = []

        if not raw_queries:
            if self.strict:
                raise RuntimeError("query planner returned no queries")
            return self._fallback_queries(sub_question, unavailable_sources)

        # Convert to SearchQuery model, filtering invalid backends
        queries: List[SearchQuery] = []
        for i, raw in enumerate(raw_queries):
            if not isinstance(raw, dict):
                continue
            backend = raw.get("tool", "").strip()
            if (
                backend not in self._valid_backends
                or backend in unavailable_sources
            ):
                continue

            text = _sanitize_query(raw.get("text", "").strip(), backend)
            if not text:
                continue

            purpose = raw.get("purpose", "").strip()
            intent = _purpose_to_intent(purpose)
            reasoning = str(raw.get("reasoning") or "").strip()
            relation = reasoning or f"Targets the {purpose or 'search'} dimension."

            queries.append(SearchQuery(
                query_id=f"q-{round_num}-{i + 1}",
                text=text,
                intent=intent,
                target_source=backend,
                purpose=purpose or "search",
                target_gap="",
                relation_to_question=relation,
            ))

        if not queries:
            if self.strict:
                raise RuntimeError("query planner returned no valid queries")
            return self._fallback_queries(sub_question, unavailable_sources)

        if round_num == 1:
            queries = self._ensure_first_round_coverage(
                sub_question,
                queries,
                unavailable_sources,
            )

        logger.debug("Planned %d queries (round %d)", len(queries), round_num)
        return queries

    def _ensure_first_round_coverage(
        self,
        sub_question: str,
        queries: Sequence[SearchQuery],
        unavailable_sources: Sequence[str],
    ) -> List[SearchQuery]:
        """Add one deterministic query for every omitted available backend."""
        primary: list[SearchQuery] = []
        repeated: list[SearchQuery] = []
        selected: set[str] = set()
        for item in queries:
            if item.target_source in selected:
                repeated.append(item)
                continue
            selected.add(item.target_source)
            primary.append(item)
        unavailable = set(unavailable_sources)
        missing = self._valid_backends - selected - unavailable
        coverage: list[SearchQuery] = []
        for backend in sorted(missing):
            text = _sanitize_query(sub_question, backend)
            if not text:
                continue
            coverage.append(
                SearchQuery(
                    query_id=f"q-1-coverage-{backend}",
                    text=text,
                    round_index=0,
                    intent=QueryIntent.CORE,
                    target_source=backend,
                    purpose="cross_source_coverage",
                    target_gap="",
                    relation_to_question=(
                        "Guarantees first-round coverage of the "
                        f"{backend} source."
                    ),
                )
            )
        # Put one query per source first so the agent's query-budget slicing
        # cannot discard coverage while retaining repeated same-source work.
        return [*primary, *coverage, *repeated]

    def _fallback_queries(
        self,
        sub_question: str,
        unavailable_sources: Sequence[str] = (),
    ) -> List[SearchQuery]:
        """Fallback: use sub_question directly on all known backends."""
        queries: List[SearchQuery] = []
        available = self._valid_backends - set(unavailable_sources)
        for i, backend in enumerate(sorted(available)):
            queries.append(SearchQuery(
                query_id=f"q-fb-{i + 1}",
                text=sub_question,
                intent=QueryIntent.CORE,
                target_source=backend,
                purpose="fallback",
                target_gap="",
                relation_to_question="Direct fallback — planner failed.",
            ))
        logger.info("Using %d fallback queries", len(queries))
        return queries


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _purpose_to_intent(purpose: str) -> QueryIntent:
    """Map a free-text purpose string to a QueryIntent enum."""
    p = purpose.lower()
    if "core" in p or "mechanism" in p:
        return QueryIntent.CORE
    if "synonym" in p or "mesh" in p:
        return QueryIntent.MESH
    if "contradict" in p or "oppos" in p or "conflict" in p:
        return QueryIntent.CONTRADICTORY
    if "recent" in p:
        return QueryIntent.RECENT
    if "review" in p:
        return QueryIntent.REVIEW
    if "citation" in p:
        return QueryIntent.CITATION
    return QueryIntent.CORE
