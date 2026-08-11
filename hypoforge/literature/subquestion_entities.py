"""M2-local supplementary entity generation for atomic sub-questions.

M1 binds task-contract entities to each sub-question, and M2 consumes them
via ``search_entities_for_sub_question``.  A sub-question can, however,
imply a concept that the original question never named (e.g. a co-factor or
a method term).  This module lets M2 ask an LLM — *per sub-question, before
iterative search starts* — whether such uncovered concepts exist and, when
they do, add up to a few short noun-phrase search concepts.

Scope contract (hard requirement)
---------------------------------
Generated entities are **M2-local**: they only feed the query planner of
this sub-question's search.  They are never written back to
``problem_card`` / ``task_contract`` / ``key_entities`` and therefore never
reach M3/M4 or any later module.

Fail-safe contract
-----------------
Any LLM error or payload validation failure degrades silently to "no
supplementary entities" plus a warning event; the search itself always
proceeds with the M1 entities alone.
"""

from __future__ import annotations

import logging
from typing import Any, List, Sequence

from pydantic import BaseModel, Field

from ..observability import emit_event

logger = logging.getLogger(__name__)

_TOOL_NAME = "m2_subquestion_entities"
_MAX_NEW_ENTITIES = 3
_MAX_ENTITY_NAME_LENGTH = 80


class _NewEntityCandidate(BaseModel):
    """One candidate supplementary search concept."""

    name: str = Field(
        description=(
            "Short noun-phrase entity name (1-5 words), in its English "
            "search form when an English term exists."
        )
    )
    aliases: List[str] = Field(
        default_factory=list,
        description="Optional alternative spellings or abbreviations.",
    )
    rationale: str = Field(
        default="",
        description="Why the sub-question implies this uncovered concept.",
    )


class _EntityProposal(BaseModel):
    """Structured output of the supplementary-entity step."""

    needed: bool = Field(
        default=False,
        description=(
            "True only when the sub-question implies a genuinely new "
            "concept not covered by the listed existing entities."
        ),
    )
    new_entities: List[_NewEntityCandidate] = Field(default_factory=list)


_SYSTEM_PROMPT = (
    "You are a scientific literature search assistant. For one atomic "
    "research sub-question you are given the entities already bound to it "
    "by the upstream task analysis. Decide whether the sub-question implies "
    "additional search concepts that are NOT covered by those existing "
    "entities, and propose them only when they clearly help literature "
    "retrieval.\n\n"
    "Hard rules:\n"
    "1. Every new entity MUST be a short noun phrase (1-5 words). NEVER "
    "output a full sentence, a definition, or a clause containing verbs "
    "such as 'is', 'are', 'used', 'identified'.\n"
    "2. Never duplicate an existing entity, its aliases, or a trivial "
    "singular/plural or case variant of them.\n"
    "3. Prefer the English search form for bilingual terms.\n"
    "4. Quality over quantity: usually 0-3 new entities; set needed=false "
    "when the existing entities already cover the sub-question.\n"
    "5. Only propose concrete searchable concepts (proteins, genes, "
    "pathways, drugs, diseases, methods, materials, techniques) — never "
    "generic words such as 'mechanism', 'effect' or 'analysis'."
)

_USER_TEMPLATE = (
    "## Sub-question\n{sub_question}\n\n"
    "## Research domains\n{domains}\n\n"
    "## Existing entities already bound to this sub-question\n"
    "{existing_entities}\n\n"
    "## Task\n"
    "Does the sub-question imply any additional searchable concept NOT "
    "covered by the existing entities above? If not, return needed=false "
    "with an empty new_entities list. If yes, list at most 3 new entities "
    "as short noun phrases."
)


def normalise_entity(value: Any) -> str:
    """Case/whitespace-insensitive entity key used for dedup comparisons."""

    return " ".join(str(value or "").casefold().split())


def dedupe_new_entities(
    proposal: _EntityProposal,
    existing_terms: Sequence[str],
    max_entities: int = _MAX_NEW_ENTITIES,
) -> List[str]:
    """Deterministic safety net over the LLM proposal.

    Drops candidates that duplicate an existing entity term (normalised
    case/whitespace comparison), that look like full sentences, or that are
    too long; caps the remainder at *max_entities*.  This never relies on
    the LLM having honoured the prompt constraints.
    """

    known = {
        normalise_entity(term) for term in existing_terms if normalise_entity(term)
    }
    kept: List[str] = []
    seen: set[str] = set()
    for candidate in proposal.new_entities:
        name = " ".join(str(candidate.name or "").split()).strip('"\' ')
        norm = normalise_entity(name)
        if not norm or norm in known or norm in seen:
            continue
        if len(name) > _MAX_ENTITY_NAME_LENGTH:
            continue
        seen.add(norm)
        known.add(norm)
        kept.append(name)
        if len(kept) >= max_entities:
            break
    return kept


async def generate_subquestion_entities(
    client: Any,
    sub_question: str,
    existing_terms: Sequence[str],
    domains: Sequence[str] = (),
    max_entities: int = _MAX_NEW_ENTITIES,
) -> List[str]:
    """Generate M2-local supplementary entities for one sub-question.

    Parameters
    ----------
    client :
        LLM client exposing ``structured_chat`` (QwenClient-compatible).
        ``None`` disables the step entirely.
    sub_question :
        The atomic sub-question about to enter iterative search.
    existing_terms :
        All terms (names + aliases + selected search aliases) of the M1
        entities already bound to this sub-question.  Shown to the LLM and
        used for the deterministic dedup fallback.
    domains :
        Problem-card domains, supplied as prompt context.

    Returns
    -------
    list[str]
        Supplementary entity names, guaranteed disjoint from
        *existing_terms* under normalised comparison.  Empty on any error.
    """

    if client is None or not str(sub_question or "").strip():
        return []

    terms = [str(term).strip() for term in existing_terms if str(term).strip()]
    emit_event(
        "tool_started",
        module="m2",
        tool=_TOOL_NAME,
        status="running",
        message="M2 子问题本地实体生成开始",
        details={
            "sub_question": sub_question,
            "existing_entity_count": len(terms),
        },
    )

    try:
        payload = await client.structured_chat(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=_USER_TEMPLATE.format(
                sub_question=sub_question,
                domains=", ".join(domains) if domains else "unknown",
                existing_entities="\n".join(f"- {term}" for term in terms)
                if terms
                else "(none)",
            ),
            output_schema=_EntityProposal.model_json_schema(),
            max_tokens=1024,
            temperature=0.0,
            disable_thinking=True,
        )
        proposal = _EntityProposal.model_validate(payload)
    except Exception as exc:
        # Fail-safe: never block the search on a supplementary step failure.
        logger.warning(
            "M2 sub-question entity generation failed for %r: %s: %s",
            sub_question[:60],
            type(exc).__name__,
            exc,
        )
        emit_event(
            "tool_result",
            module="m2",
            tool=_TOOL_NAME,
            status="warning",
            message=(
                "M2 子问题本地实体生成失败，降级为仅使用已有实体："
                f"{type(exc).__name__}"
            ),
            details={
                "sub_question": sub_question,
                "existing_entity_count": len(terms),
                "new_entities": [],
            },
        )
        return []

    if not proposal.needed:
        new_entities: List[str] = []
    else:
        new_entities = dedupe_new_entities(proposal, terms, max_entities)

    emit_event(
        "tool_completed",
        module="m2",
        tool=_TOOL_NAME,
        status="completed",
        message=(
            f"M2 子问题本地实体生成完成：新增 {len(new_entities)} 个检索概念"
        ),
        details={
            "sub_question": sub_question,
            "existing_entity_count": len(terms),
            "new_entities": list(new_entities),
        },
    )
    return new_entities
