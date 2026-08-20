"""Deterministic M1-domain routing layered over the Qwen query planner."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
import re

from ..models import QueryIntent, SearchQuery, SearchState
from ..protocols import QueryPlannerProtocol


GENERAL_SOURCE = "serper_openalex"
GENERAL_SOURCE_FALLBACKS = (GENERAL_SOURCE, "openalex", "crossref")
OPEN_ACCESS_RESCUE_SOURCE = "openalex_oa"


def route_specialist_sources(
    domains: Sequence[str], sub_question: str
) -> list[str]:
    """Return a stable specialist-source route from M1 labels and sub-question.

    Routing is additive for interdisciplinary questions.  Crossref is the
    metadata fallback when no specialist rule matches.
    """

    text = " ".join([*map(str, domains), sub_question]).casefold()
    output: list[str] = []

    def add(*names: str) -> None:
        for name in names:
            if name not in output:
                output.append(name)

    if "mathemat" in text or "number theory" in text:
        add("zbmath", "arxiv", "crossref")
    # Word-boundary matching avoids routing biophysics/geophysics questions
    # into INSPIRE merely because their domain label contains "physics".
    if re.search(r"\bphysics\b", text) or any(term in text for term in (
        "quantum", "particle physics", "relativ", "field theory",
        "superconduct",
    )):
        add("arxiv", "inspire", "crossref")
    if any(term in text for term in (
        "astronom", "cosmolog", "universe", "galaxy", "stellar",
        "black hole", "planet", "astrophys",
    )):
        add("ads", "arxiv", "inspire")
    if any(term in text for term in (
        "medicine", "health", "clinical", "disease", "cancer", "immune",
        "vaccine", "autism", "biolog", "genom", "cell", "neuro",
    )):
        add("europe_pmc")
    if any(term in text for term in (
        "chemistry", "chemical", "molecule", "molecular", "chirality", "pigment",
    )):
        add("crossref")
        if any(term in text for term in ("life", "biolog", "cell", "drug")):
            add("europe_pmc")
    if any(term in text for term in (
        "engineering", "materials", "manufactur", "turbulence", "mars",
    )):
        add("crossref", "arxiv")
        if "mars" in text or "space" in text:
            add("ntrs")
    if any(term in text for term in (
        "information science", "computer", "artificial intelligence", "robot",
        "algorithm", "machine learning", "software",
    )):
        add("dblp", "arxiv", "crossref")
    if any(term in text for term in (
        "ecology", "climate", "environment", "species", "ocean", "population",
    )):
        add("crossref")
        if any(term in text for term in ("species", "ocean", "biolog", "population")):
            add("europe_pmc")
    if any(term in text for term in (
        "energy", "hydrogen", "fusion", "battery", "storage", "renewable",
    )):
        add("osti", "crossref")
    return output or ["crossref"]


class DomainRoutedQueryPlanner(QueryPlannerProtocol):
    """Filter Qwen output to the M1 route and enforce mandatory discovery.

    Qwen still chooses domain-aware query wording.  This wrapper makes source
    selection deterministic: Serper→OpenAlex is preferred, with OpenAlex or
    Crossref as credential-free general fallbacks.  Only relevant specialist
    APIs are exposed to execution.  Later rounds add an independent OpenAlex
    query as an open-access rescue.
    """

    tool_name = "domain_routed_query_planner"

    def __init__(
        self,
        delegate: QueryPlannerProtocol,
        enabled_sources: Sequence[str],
        planner_timeout_seconds: float = 60.0,
    ) -> None:
        self.delegate = delegate
        self.enabled_sources = {
            str(source).strip().casefold() for source in enabled_sources
            if str(source).strip()
        }
        if not self.enabled_sources.intersection(GENERAL_SOURCE_FALLBACKS):
            raise ValueError(
                "domain-routed search requires at least one general source: "
                + ", ".join(GENERAL_SOURCE_FALLBACKS)
            )
        if planner_timeout_seconds <= 0:
            raise ValueError("planner_timeout_seconds must be positive")
        self.planner_timeout_seconds = float(planner_timeout_seconds)

    def _active_sources(
        self,
        domains: Sequence[str],
        sub_question: str,
        *,
        expanded: bool,
    ) -> list[str]:
        general_source = next(
            source for source in GENERAL_SOURCE_FALLBACKS
            if source in self.enabled_sources
        )
        ordered = [general_source]
        ordered.extend(route_specialist_sources(domains, sub_question))
        if expanded:
            ordered.append(OPEN_ACCESS_RESCUE_SOURCE)
        return [
            source for source in dict.fromkeys(ordered)
            if source in self.enabled_sources
        ]

    @staticmethod
    def _query(
        *,
        source: str,
        text: str,
        round_index: int,
        suffix: str,
        purpose: str,
    ) -> SearchQuery:
        return SearchQuery(
            query_id=f"routed-{round_index + 1}-{source}-{suffix}",
            text=" ".join(text.split()),
            round_index=round_index,
            intent=(QueryIntent.REVIEW if "review" in purpose else QueryIntent.CORE),
            target_source=source,
            purpose=purpose,
            relation_to_question="Deterministic M1-domain source coverage.",
        )

    async def plan(
        self,
        sub_question: str,
        key_entities: Sequence[str] = (),
        domains: Sequence[str] = (),
        question_type: str = "",
        state: SearchState | None = None,
        supplement_entities: Sequence[str] = (),
        focus_entities: Sequence[str] = (),
    ) -> list[SearchQuery]:
        try:
            planned = await asyncio.wait_for(
                self.delegate.plan(
                    sub_question,
                    key_entities=key_entities,
                    domains=domains,
                    question_type=question_type,
                    state=state,
                    supplement_entities=supplement_entities,
                    focus_entities=focus_entities,
                ),
                timeout=self.planner_timeout_seconds,
            )
        except (asyncio.TimeoutError, TimeoutError, RuntimeError, ValueError):
            # Same deterministic portfolio used by the live benchmark when
            # Qwen query planning was unavailable.
            planned = []
        round_index = state.round_index if state is not None else 0
        expanded = question_type == "open_access_expansion"
        unavailable = {
            source.casefold() for source in (
                state.unavailable_sources if state is not None else set()
            )
        }
        active = [
            source for source in self._active_sources(
                domains, sub_question, expanded=expanded
            )
            if source not in unavailable
        ]
        general_source = next(
            (
                source for source in GENERAL_SOURCE_FALLBACKS
                if source in self.enabled_sources and source not in unavailable
            ),
            "",
        )
        if general_source:
            active = [
                general_source,
                *(
                    source for source in active
                    if source != general_source
                ),
            ]
        # The successful Science-125 benchmark used two complementary query
        # variants per phase and sent the same portfolio to the selected
        # general source and every routed specialist source.  Preserve that
        # call shape here instead of letting a source-selecting LLM silently
        # omit a backend.
        priorities = (
            ("evidence", "method", "recent", "open", "preprint")
            if expanded else
            ("mechanism", "core", "review", "foundational")
        )
        ordered_planned = sorted(
            enumerate(planned),
            key=lambda item: (
                next(
                    (
                        index for index, marker in enumerate(priorities)
                        if marker in item[1].purpose.casefold()
                    ),
                    len(priorities),
                ),
                item[0],
            ),
        )
        variants: list[str] = []
        for _, query in ordered_planned:
            text = " ".join(query.text.split())
            if text and text.casefold() not in {item.casefold() for item in variants}:
                variants.append(text)
        fallback = (
            [
                f"{sub_question} empirical evidence mechanism",
                f"{sub_question} open access preprint",
            ]
            if expanded else
            [
                sub_question,
                " ".join([*key_entities[:4], "foundational systematic review"]),
            ]
        )
        for text in fallback:
            clean = " ".join(text.split())
            if clean and clean.casefold() not in {item.casefold() for item in variants}:
                variants.append(clean)
        variants = variants[:2]

        ordered: list[SearchQuery] = []
        # The best available general source is first for every variant,
        # protecting universal discovery when a global query budget is tight.
        if general_source:
            for index, text in enumerate(variants, 1):
                ordered.append(self._query(
                    source=general_source,
                    text=text,
                    round_index=round_index,
                    suffix=f"portfolio-{index}",
                    purpose=(
                        "expanded_open_fulltext_discovery" if expanded
                        else "compact_high_citation_discovery"
                    ),
                ))
        specialist_order = [
            source for source in active if source != general_source
        ]
        if expanded and OPEN_ACCESS_RESCUE_SOURCE in specialist_order:
            specialist_order.remove(OPEN_ACCESS_RESCUE_SOURCE)
            specialist_order.insert(0, OPEN_ACCESS_RESCUE_SOURCE)
        for source in specialist_order:
            for index, text in enumerate(variants, 1):
                ordered.append(self._query(
                    source=source,
                    text=text,
                    round_index=round_index,
                    suffix=f"portfolio-{index}",
                    purpose=(
                        "open_access_rescue"
                        if source == OPEN_ACCESS_RESCUE_SOURCE
                        else "domain_specialist_discovery"
                    ),
                ))
        seen: set[tuple[str, str]] = set()
        output: list[SearchQuery] = []
        for query in ordered:
            key = (
                query.target_source.casefold(),
                " ".join(query.text.casefold().split()),
            )
            if key in seen:
                continue
            seen.add(key)
            output.append(query)
        return output
