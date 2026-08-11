"""LLM-driven query planner — generates source-aware search queries.

Implements ``QueryPlannerProtocol`` from the agentic literature skeleton.
Uses QwenClient with structured output to generate ``SearchQuery`` items
that include backend selection and coverage intent.
"""

from __future__ import annotations

import json
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


def _query_tokens(text: str) -> list[str]:
    """Return backend-agnostic lexical tokens for anchor checks."""
    return re.findall(r"[\w\u3400-\u9fff]+", text.casefold())


def _contains_core_entity(query: str, entity: str) -> bool:
    """Whether *query* already contains the entity as a phrase or token set.

    LLMs sometimes insert operators or field tags between words of a
    multi-word entity. Treating the entity as present when all of its tokens
    occur avoids adding a duplicated anchor merely because the phrase is not
    contiguous.
    """
    query_tokens = _query_tokens(query)
    entity_tokens = _query_tokens(entity)
    if not entity_tokens:
        return True
    if " ".join(entity_tokens) in " ".join(query_tokens):
        return True
    return all(token in query_tokens for token in entity_tokens)


def _anchor_query(text: str, core_entity: str, backend: str) -> str:
    """Add a missing core entity without destroying the LLM query structure.

    The anchor is deliberately placed as a grouped conjunct rather than
    appended as loose trailing words (``query entity``), which can alter the
    parser's precedence and produce malformed backend queries.
    """
    entity = " ".join(str(core_entity or "").split()).strip()
    query = _sanitize_query(text, backend).strip()
    if not entity or not query or _contains_core_entity(query, entity):
        return query
    quoted_entity = json.dumps(entity, ensure_ascii=False)
    return _sanitize_query(f"{quoted_entity} AND ({query})", backend)

# ---------------------------------------------------------------------------
# Prompts (mirrored from m2_branch for standalone use)
# ---------------------------------------------------------------------------

_QUERY_PLANNING_SYSTEM = """\
You are a scientific literature search strategist. Your job is to generate \
high-quality, targeted search queries for a given research sub-question and \
select the most appropriate search tool for each query. Remember that the \
current year is 2026.

## Available Search Tools

{tool_descriptions}

## Query Construction Rules

### For PubMed queries (biomedical / life sciences ONLY):
- Prefer `[tiab]` (title/abstract) or bare keywords, which let PubMed's \
  automatic term mapping resolve terminology safely.
- **NEVER invent MeSH headings from memory.** Use `[MeSH Terms]` ONLY when \
  you are certain the exact MeSH descriptor exists — MeSH is exact-form \
  sensitive (e.g. `"protein misfolding"[MeSH Terms]` and \
  `"amyloid fibril"[MeSH Terms]` match NOTHING; the real heading is \
  `Amyloid Fibrils`). When unsure, drop the tag and use `[tiab]` or a bare \
  phrase instead.
- Break the sub-question into **short key concepts** (1-3 words each), then combine \
  them with AND/OR as needed. Don't put long phrases inside operators.
- Combining 3+ concepts with pure AND frequently yields an empty \
  intersection. For every PubMed sub-question you MUST include at least ONE \
  broad-recall safety-net query built from only the **2 most central \
  concepts** (e.g. `Hsp70[tiab] AND aging`).
- **Good examples (short concepts, clear structure):**
  - `Hsp70[tiab] AND ATPase[tiab] AND protein folding[MeSH Terms]`
  - `"heat shock protein"[tiab] AND (aging OR senescence) AND proteostasis`
  - `NAD[tiab] AND chaperone[tiab] AND (aged OR aging)`
  - `Hsp70[tiab] AND aging` ← required 2-concept broad-recall safety net
- **Bad examples:**
  - `the role of heat shock protein 70 in ATP-dependent protein folding` ← too long, no tags
  - `Hsp70 mechanism` ← single vague term
  - `"protein misfolding"[MeSH Terms]` ← fabricated MeSH heading, 0 hits

### For Semantic Scholar/OpenAlex queries (cross-disciplinary discovery):
- Use phrase quotes for multi-word terms: `"protein folding"`.
- Keep each concept short (no more than 3 words); combine with AND/OR when helpful. No more than 3 concepts.
- S2 syntax is simpler than PubMed — keyword search works fine.

### For arXiv queries (recent technical preprints):
- Prefer arXiv for computer science, mathematics, physics, statistics,
  electrical engineering, and other technical preprints.
- Use short keyword phrases and AND/OR; never use PubMed field tags.
- Each keyword should be no more than 3 words. No more than 3 key words combined by AND/OR.
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
4. For every query, provide ``term_importance`` scores for substantive terms.
   Score the central entities and the requested relation/method highest; score
   generic modifiers such as "role", "study", "using", "novel", or "effect"
   lower. These scores drive deterministic zero-result relaxation, so they
   must reflect semantic importance rather than the term's position.
5. Never repeat a query that has already been used.
6. Never select a search tool listed as unavailable.
7. When the request lists ``Round Focus Entities``, every query of this \
round MUST be built around those concepts: each query must contain all of \
them (combine with AND / field tags appropriate to the tool). They are the \
round's mandatory concepts, not optional decoration.
8. When the request lists ``Supplementary Search Concepts``, treat them as \
optional extra concepts added by M2 for retrieval: you MAY weave them \
into queries when they improve recall, but they never replace the key \
entities or the round focus entities.

Return valid JSON only — no markdown fences, no extra text."""

_QUERY_PLANNING_USER_FIRST = """\
## Sub-Question
{sub_question}

## Domain & Context
- Domains: {domains}
- Key Entities: {entities}
{supplements}
## Current Search State
- Round: 1/{max_rounds}
- Papers found so far: 0
- Queries already used: (none)
- Current gaps identified: (none — first round)
- Unavailable search tools: {unavailable_sources}

## Task
Generate 3–6 search queries covering at least 3 different coverage dimensions. Since it is the first round, the queries should cover all tools available.
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
{supplements}
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
        supplement_entities: Sequence[str] = (),
        focus_entities: Sequence[str] = (),
    ) -> List[SearchQuery]:
        """Generate source-aware search queries.

        When ``state`` is None (first round), generates initial queries
        covering multiple evidence dimensions.  When ``state`` carries
        round/gap information from a prior iteration, the planner
        produces targeted follow-up queries.

        ``focus_entities`` are the round's mandatory concepts assigned by
        the entity-group round strategy: every generated query is built
        around them (and deterministically anchored to them).
        """
        # ``question_type`` is accepted only for historical caller
        # compatibility; planning is derived from the atomic sub-question.
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
            round_num=round_num,
            paper_count=paper_count,
            queries_used=queries_used,
            gaps=gaps,
            unavailable_sources=unavailable_sources,
            supplements=list(supplement_entities),
            focus=list(focus_entities),
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
        round_num: int,
        paper_count: int,
        queries_used: List[str],
        gaps: List[str],
        unavailable_sources: List[str],
        supplements: Optional[List[str]] = None,
        focus: Optional[List[str]] = None,
    ) -> List[SearchQuery]:
        """Core planning logic — shared by plan() and plan_next()."""
        tool_descriptions = "\n\n".join(
            f"### {d['display_name']} (`{d['name']}`)\n{d['description']}"
            for d in self._tool_definitions
        )

        system = _QUERY_PLANNING_SYSTEM.format(tool_descriptions=tool_descriptions)

        supplement_line = (
            "- Supplementary Search Concepts (added by M2 for this "
            "sub-question's retrieval only; optional to use): "
            + ", ".join(supplements) + "\n\n"
            if supplements else "\n"
        )
        focus_list = [str(term).strip() for term in (focus or []) if str(term).strip()]
        focus_line = (
            "- Round Focus Entities (MANDATORY concepts for this round; every "
            "query must be built around ALL of them): "
            + ", ".join(focus_list) + "\n\n"
            if focus_list else ""
        )

        if round_num <= 1 and not queries_used:
            user = _QUERY_PLANNING_USER_FIRST.format(
                sub_question=sub_question,
                domains=", ".join(domains) if domains else "unknown",
                entities=", ".join(entities) if entities else "unknown",
                supplements=focus_line + supplement_line,
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
                supplements=focus_line + supplement_line,
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
                            "term_importance": {
                                "type": "object",
                                "additionalProperties": {
                                    "type": "number", "minimum": 0, "maximum": 1
                                },
                                "description": (
                                    "Semantic importance of each substantive query "
                                    "term; 1 is indispensable and 0 is a removable "
                                    "modifier. Preserve key entities and relation terms."
                                ),
                            },
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
            # The round's focus entities are mandatory concepts: anchor every
            # one the LLM omitted.  Then keep M1's first key entity as a
            # structured conjunct as well; never append loose words to the
            # tail of the query.
            for focus_entity in focus_list:
                text = _anchor_query(text, focus_entity, backend)
            if entities:
                text = _anchor_query(text, entities[0], backend)

            purpose = raw.get("purpose", "").strip()
            intent = _purpose_to_intent(purpose)
            reasoning = str(raw.get("reasoning") or "").strip()
            relation = reasoning or f"Targets the {purpose or 'search'} dimension."
            raw_importance = raw.get("term_importance")
            term_importance = (
                raw_importance if isinstance(raw_importance, dict) else {}
            )

            queries.append(SearchQuery(
                query_id=f"q-{round_num}-{i + 1}",
                text=text,
                intent=intent,
                target_source=backend,
                purpose=purpose or "search",
                target_gap="",
                relation_to_question=relation,
                term_importance=term_importance,
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
                domains,
            )

        logger.debug("Planned %d queries (round %d)", len(queries), round_num)
        return queries

    def _ensure_first_round_coverage(
        self,
        sub_question: str,
        queries: Sequence[SearchQuery],
        unavailable_sources: Sequence[str],
        domains: Sequence[str] = (),
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
        ordered = [*primary, *coverage, *repeated]
        if _is_biomedical(domains):
            # Biomedical questions live or die on PubMed recall; move every
            # PubMed query to the front so max_queries budget slicing cannot
            # squeeze PubMed out.
            ordered = [
                *[q for q in ordered if q.target_source == "pubmed"],
                *[q for q in ordered if q.target_source != "pubmed"],
            ]
        return ordered

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

_BIOMEDICAL_DOMAIN_KEYWORDS = (
    "bio",          # biology, biomedical, biochemistry, bioinformatics
    "medic",        # medicine, medical
    "clinical",
    "neuro",        # neuroscience, neurology, neurobiology
    "health",
    "physiol",
    "pharma",
    "genet",
    "immun",
    "oncolog",
    "cancer",
    "molecular",
    "patholog",
    "epidemiol",
    "psychiat",
    "life science",
)


def _is_biomedical(domains: Sequence[str]) -> bool:
    """Return whether *domains* describe a biomedical research area."""
    joined = " ".join(str(domain) for domain in domains).casefold()
    return any(keyword in joined for keyword in _BIOMEDICAL_DOMAIN_KEYWORDS)


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
