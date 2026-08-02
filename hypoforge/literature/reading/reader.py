"""Qwen-based paper reading constrained to supplied evidence identifiers."""

from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Any

from ...state import ConfidenceLevel, KnowledgeEntryType
from ...tools.qwen_client import QwenClient
from ..models import (
    EvidenceChunk,
    EvidenceLinkedKnowledge,
    PaperReadingResult,
    PaperRecord,
)
from ..protocols import PaperReaderProtocol

_SYSTEM_PROMPT = """You are a biomedical evidence extraction engine.
Use only the supplied evidence passages. Extract zero or more concise knowledge
entries in the allowed categories. Every summary and entry must cite one or more
CITABLE evidence IDs. Context-only passages may clarify adjacent text but must
never be cited. Do not infer facts that are not stated in the passages, do not
invent citations, and do not output chain-of-thought."""

_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "summary_evidence_ids": {"type": "array", "items": {"type": "string"}},
        "entries": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": [item.value for item in KnowledgeEntryType],
                    },
                    "content": {"type": "string"},
                    "confidence": {
                        "type": ["string", "null"],
                        "enum": [item.value for item in ConfidenceLevel] + [None],
                    },
                    "entities": {"type": "array", "items": {"type": "string"}},
                    "evidence_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                    },
                },
                "required": ["type", "content", "entities", "evidence_ids"],
            },
        },
    },
    "required": ["summary", "summary_evidence_ids", "entries"],
}


def _clean(value: object, limit: int) -> str:
    return " ".join(str(value or "").split())[:limit]


def _error(exc: BaseException) -> str:
    return " ".join(f"{type(exc).__name__}: {exc}".split())[:500]


class QwenPaperReader(PaperReaderProtocol):
    def __init__(self, client: QwenClient) -> None:
        self.client = client

    @staticmethod
    def _prompt(
        sub_question: str,
        paper: PaperRecord,
        evidence: list[EvidenceChunk],
    ) -> str:
        blocks = []
        for item in evidence:
            label = (
                f"[Evidence ID: {item.evidence_id}]\nCitation status: CITABLE"
                if item.citable
                else f"[CONTEXT ONLY - DO NOT CITE: {item.evidence_id}]"
            )
            blocks.append(
                "\n".join(
                    [
                        label,
                        f"Section: {item.section or 'unknown'}",
                        item.quote,
                    ]
                )
            )
        return "\n\n".join(
            [
                f"Sub-question: {_clean(sub_question, 2000)}",
                f"Paper ID: {paper.paper_id}",
                f"Title: {_clean(paper.title, 1000)}",
                "Allowed categories: "
                + ", ".join(item.value for item in KnowledgeEntryType),
                "Evidence passages:\n" + "\n\n".join(blocks),
            ]
        )

    async def read(
        self,
        sub_question: str,
        paper: PaperRecord,
        evidence: list[EvidenceChunk],
    ) -> PaperReadingResult:
        candidates = [item for item in evidence if item.paper_id == paper.paper_id]
        if not candidates:
            return PaperReadingResult(
                paper_id=paper.paper_id,
                errors=["no retrievable evidence"],
            )
        by_id = {item.evidence_id: item for item in candidates}
        citable_by_id = {
            identifier: item for identifier, item in by_id.items() if item.citable
        }
        if not citable_by_id:
            return PaperReadingResult(
                paper_id=paper.paper_id,
                errors=["no citable evidence"],
            )
        try:
            raw = await self.client.structured_chat(
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=self._prompt(sub_question, paper, candidates),
                output_schema=_OUTPUT_SCHEMA,
                max_tokens=4096,
                temperature=0.0,
                disable_thinking=True,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return PaperReadingResult(
                paper_id=paper.paper_id,
                errors=[f"paper_reader failed: {_error(exc)}"],
            )

        if not isinstance(raw, dict) or raw.get("_parse_error"):
            return PaperReadingResult(
                paper_id=paper.paper_id,
                errors=["paper_reader returned invalid structured output"],
            )

        errors: list[str] = []
        used_ids: set[str] = set()
        summary = _clean(raw.get("summary"), 2000)
        summary_ids = raw.get("summary_evidence_ids")
        if not isinstance(summary_ids, list):
            summary_ids = []
        normalized_summary_ids = [str(item) for item in summary_ids]
        if summary and normalized_summary_ids and all(
            item in citable_by_id for item in normalized_summary_ids
        ):
            used_ids.update(normalized_summary_ids)
        else:
            if summary:
                errors.append("summary rejected: invalid evidence IDs")
            summary = ""

        entries: list[EvidenceLinkedKnowledge] = []
        seen: set[tuple[str, str, tuple[str, ...]]] = set()
        rejected = 0
        raw_entries = raw.get("entries", [])
        if not isinstance(raw_entries, list):
            raw_entries = []
        for item in raw_entries:
            if not isinstance(item, dict):
                rejected += 1
                continue
            content = _clean(item.get("content"), 3000)
            evidence_ids = item.get("evidence_ids")
            normalized_ids = (
                [str(value) for value in evidence_ids]
                if isinstance(evidence_ids, list)
                else []
            )
            try:
                entry_type = KnowledgeEntryType(str(item.get("type") or ""))
                confidence_raw = item.get("confidence")
                confidence = (
                    ConfidenceLevel(str(confidence_raw))
                    if confidence_raw not in (None, "")
                    else None
                )
            except ValueError:
                rejected += 1
                continue
            if (
                not content
                or not normalized_ids
                or not all(value in citable_by_id for value in normalized_ids)
            ):
                rejected += 1
                continue
            unique_ids = list(dict.fromkeys(normalized_ids))
            key = (
                entry_type.value,
                content.casefold(),
                tuple(sorted(unique_ids)),
            )
            if key in seen:
                continue
            seen.add(key)
            digest = hashlib.sha256(
                json.dumps(key, ensure_ascii=False).encode("utf-8")
            ).hexdigest()[:16]
            entities_raw = item.get("entities", [])
            entities = (
                list(
                    dict.fromkeys(
                        _clean(value, 200)
                        for value in entities_raw
                        if _clean(value, 200)
                    )
                )
                if isinstance(entities_raw, list)
                else []
            )
            entries.append(
                EvidenceLinkedKnowledge(
                    entry_id=f"reading-{digest}",
                    entry_type=entry_type,
                    content=content,
                    confidence=confidence,
                    entities=entities,
                    evidence_ids=unique_ids,
                )
            )
            used_ids.update(unique_ids)

        if rejected:
            errors.append(f"paper_reader rejected {rejected} invalid entries")
        accepted_evidence = [
            item for item in candidates
            if item.citable and item.evidence_id in used_ids
        ]
        return PaperReadingResult(
            paper_id=paper.paper_id,
            summary=summary,
            evidence=accepted_evidence,
            knowledge_entries=entries,
            errors=errors,
        )
