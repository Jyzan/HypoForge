"""Shared, conservative entity normalization with persistent decisions.

The service separates candidate recall from merge judgment: embeddings (or a
lexical fallback) may propose similar canonical entities, but only an explicit
task alias or a cached/LLM ``same_concept`` decision can merge them.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Optional

from pydantic import BaseModel, Field

from .state import TaskContract


class NormalizedEntity(BaseModel):
    canonical_id: str
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)
    decision_source: str = "new_entity"


class EntityPairDecision(BaseModel):
    left: str
    right: str
    same_concept: bool
    rationale: str = ""
    source: str = "llm"


def clean_entity_surface(value: str) -> str:
    """Apply domain-neutral shape validation and Unicode normalization."""

    text = unicodedata.normalize("NFKC", str(value or "")).strip().casefold()
    text = re.sub(r"\s+", " ", text)
    if (
        not text
        or len(text) > 100
        or len(text.split()) > 8
        or text.count("(") != text.count(")")
        or text.count("（") != text.count("）")
        or text.endswith((".", "。", "!", "！", "?", "？", ":", "："))
    ):
        return ""
    text = text.rstrip(".,;:")
    wrappers = {("(", ")"), ("[", "]"), ("（", "）")}
    while len(text) >= 2 and (text[0], text[-1]) in wrappers:
        text = text[1:-1].strip()
    return text


def _stable_id(namespace: str, canonical_name: str) -> str:
    digest = hashlib.sha256(
        f"{namespace}|{clean_entity_surface(canonical_name)}".encode("utf-8")
    ).hexdigest()[:20]
    return f"ENT_{digest}"


def _cosine(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)


class EntityNormalizationService:
    """Normalize entities within a domain namespace and persist pair decisions."""

    schema_version = "entity-normalization/v1"

    def __init__(
        self,
        *,
        cache_dir: str | Path = "",
        namespace: str = "global",
        client: Any = None,
        embedding_backend: Any = None,
        embedding_model: str = "",
        similarity_threshold: float = 0.82,
    ) -> None:
        self.namespace = clean_entity_surface(namespace) or "global"
        self.client = client
        self.embedding_backend = embedding_backend
        self.embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL", "")
        self.similarity_threshold = max(0.0, min(1.0, similarity_threshold))
        self.records: dict[str, NormalizedEntity] = {}
        self.alias_to_id: dict[str, str] = {}
        self.pair_decisions: dict[str, EntityPairDecision] = {}
        self.cache_path: Optional[Path] = None
        if cache_dir:
            root = Path(cache_dir) / "entity_normalization"
            digest = hashlib.sha256(self.namespace.encode("utf-8")).hexdigest()[:16]
            self.cache_path = root / f"{digest}.json"
        self._load()

    @classmethod
    def from_task(
        cls,
        contract: TaskContract,
        *,
        domains: Iterable[str] = (),
        cache_dir: str | Path = "",
        client: Any = None,
        embedding_backend: Any = None,
        embedding_model: str = "",
    ) -> "EntityNormalizationService":
        namespace = "|".join(sorted(
            clean_entity_surface(domain) for domain in domains
            if clean_entity_surface(domain)
        )) or "global"
        service = cls(
            cache_dir=cache_dir,
            namespace=namespace,
            client=client,
            embedding_backend=embedding_backend,
            embedding_model=embedding_model,
        )
        service.register_task_contract(contract)
        return service

    def _load(self) -> None:
        if self.cache_path is None or not self.cache_path.exists():
            return
        try:
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
            if payload.get("schema_version") != self.schema_version:
                return
            self.records = {
                key: NormalizedEntity.model_validate(value)
                for key, value in (payload.get("records") or {}).items()
            }
            self.alias_to_id = {
                clean_entity_surface(key): str(value)
                for key, value in (payload.get("alias_to_id") or {}).items()
                if clean_entity_surface(key)
            }
            self.pair_decisions = {
                key: EntityPairDecision.model_validate(value)
                for key, value in (payload.get("pair_decisions") or {}).items()
            }
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            # A corrupt optional cache must never break the evidence pipeline.
            self.records = {}
            self.alias_to_id = {}
            self.pair_decisions = {}

    def _save(self) -> None:
        if self.cache_path is None:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": self.schema_version,
            "namespace": self.namespace,
            "records": {
                key: value.model_dump(mode="json")
                for key, value in self.records.items()
            },
            "alias_to_id": dict(self.alias_to_id),
            "pair_decisions": {
                key: value.model_dump(mode="json")
                for key, value in self.pair_decisions.items()
            },
        }
        temporary = self.cache_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(self.cache_path)

    @staticmethod
    def _pair_key(left: str, right: str) -> str:
        return "||".join(sorted((clean_entity_surface(left), clean_entity_surface(right))))

    def _upsert(
        self,
        canonical_name: str,
        aliases: Iterable[str],
        *,
        decision_source: str,
    ) -> NormalizedEntity:
        canonical = clean_entity_surface(canonical_name)
        preferred_name = re.sub(
            r"\s+",
            " ",
            unicodedata.normalize("NFKC", str(canonical_name or "")),
        ).strip().rstrip(".,;:")
        canonical_id = _stable_id(self.namespace, canonical)
        existing = self.records.get(canonical_id)
        surfaces = list(dict.fromkeys(filter(None, [
            canonical,
            *(clean_entity_surface(alias) for alias in aliases),
            *(existing.aliases if existing else []),
        ])))
        record = NormalizedEntity(
            canonical_id=canonical_id,
            canonical_name=(
                preferred_name
                if decision_source == "task_contract" or existing is None
                else existing.canonical_name
            ),
            aliases=surfaces,
            decision_source=decision_source,
        )
        self.records[canonical_id] = record
        for surface in surfaces:
            self.alias_to_id[surface] = canonical_id
        return record

    def register_task_contract(self, contract: TaskContract) -> None:
        for entity in contract.entities:
            if clean_entity_surface(entity.name):
                self.register_alias_group(
                    entity.name, entity.aliases, source="task_contract"
                )
        self._save()

    def register_alias_group(
        self,
        canonical_name: str,
        aliases: Iterable[str],
        *,
        source: str = "curated_vocabulary",
    ) -> NormalizedEntity:
        """Register an explicit identity group; later registrations take priority."""

        record = self._upsert(
            canonical_name, aliases, decision_source=source
        )
        self._save()
        return record

    async def _embedding_candidates(
        self,
        surfaces: list[str],
        canonical_names: list[str],
    ) -> dict[str, list[str]]:
        if (
            not self.embedding_backend and not self.embedding_model
        ) or not surfaces or not canonical_names:
            return {}
        try:
            backend = self.embedding_backend
            if backend is None:
                from langchain_openai import OpenAIEmbeddings

                kwargs: dict[str, Any] = {"model": self.embedding_model}
                if os.getenv("OPENAI_API_KEY"):
                    kwargs["api_key"] = os.getenv("OPENAI_API_KEY")
                if os.getenv("OPENAI_BASE_URL"):
                    kwargs["base_url"] = os.getenv("OPENAI_BASE_URL")
                backend = OpenAIEmbeddings(**kwargs)
            vectors = await backend.aembed_documents(
                [*surfaces, *canonical_names]
            )
        except Exception:
            return {}
        surface_vectors = vectors[:len(surfaces)]
        canonical_vectors = vectors[len(surfaces):]
        output: dict[str, list[str]] = {}
        for surface, vector in zip(surfaces, surface_vectors):
            scored = sorted(
                (
                    (_cosine(vector, candidate_vector), canonical)
                    for canonical, candidate_vector in zip(
                        canonical_names, canonical_vectors
                    )
                ),
                reverse=True,
            )
            output[surface] = [
                canonical for score, canonical in scored[:5]
                if score >= self.similarity_threshold
            ]
        return output

    def _lexical_candidates(
        self,
        surface: str,
        canonical_names: list[str],
    ) -> list[str]:
        scored = [
            (SequenceMatcher(None, surface, candidate).ratio(), candidate)
            for candidate in canonical_names
        ]
        return [
            candidate for score, candidate in sorted(scored, reverse=True)[:5]
            if score >= self.similarity_threshold
        ]

    async def _judge(
        self,
        requests: list[dict[str, Any]],
    ) -> dict[str, tuple[str, str]]:
        if self.client is None or not requests:
            return {}
        schema = {
            "type": "object",
            "properties": {
                "decisions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "surface": {"type": "string"},
                            "canonical_name": {"type": "string"},
                            "same_concept": {"type": "boolean"},
                            "rationale": {"type": "string"},
                        },
                        "required": [
                            "surface", "canonical_name", "same_concept", "rationale"
                        ],
                    },
                },
            },
            "required": ["decisions"],
        }
        try:
            payload = await self.client.structured_chat(
                system_prompt=(
                    "You are a domain-neutral entity identity judge. Decide only "
                    "whether two names denote the same real concept. Related, "
                    "broader, narrower, component, cause, and effect concepts are "
                    "different. Choose only from supplied canonical candidates."
                ),
                user_prompt=json.dumps(requests, ensure_ascii=False, indent=2),
                output_schema=schema,
                max_tokens=4096,
                temperature=0.0,
                disable_thinking=True,
            )
        except Exception:
            return {}
        output: dict[str, tuple[str, str]] = {}
        allowed = {
            (clean_entity_surface(item["surface"]), clean_entity_surface(candidate))
            for item in requests
            for candidate in item["candidates"]
        }
        for decision in payload.get("decisions", []) if isinstance(payload, dict) else []:
            surface = clean_entity_surface(decision.get("surface", ""))
            canonical = clean_entity_surface(decision.get("canonical_name", ""))
            if decision.get("same_concept") is True and (surface, canonical) in allowed:
                output[surface] = (canonical, str(decision.get("rationale", ""))[:500])
        return output

    async def resolve_batch(self, names: Iterable[str]) -> dict[str, NormalizedEntity]:
        original_names = list(dict.fromkeys(str(name or "") for name in names))
        surfaces = {
            original: clean_entity_surface(original) for original in original_names
        }
        result: dict[str, NormalizedEntity] = {}
        unresolved: list[str] = []
        for original, surface in surfaces.items():
            canonical_id = self.alias_to_id.get(surface)
            if canonical_id and canonical_id in self.records:
                result[original] = self.records[canonical_id]
            elif surface:
                unresolved.append(surface)

        canonical_names = sorted({
            record.canonical_name for record in self.records.values()
        })
        embedded = await self._embedding_candidates(unresolved, canonical_names)
        candidate_map: dict[str, list[str]] = {}
        for surface in unresolved:
            candidates = embedded.get(surface) or self._lexical_candidates(
                surface, canonical_names
            )
            undecided: list[str] = []
            for candidate in candidates:
                cached = self.pair_decisions.get(self._pair_key(surface, candidate))
                if cached and cached.same_concept:
                    canonical_id = self.alias_to_id.get(candidate)
                    if canonical_id:
                        self.alias_to_id[surface] = canonical_id
                        break
                if cached is None:
                    undecided.append(candidate)
            else:
                if undecided:
                    candidate_map[surface] = undecided

        judged = await self._judge([
            {"surface": surface, "candidates": candidates}
            for surface, candidates in candidate_map.items()
        ])
        for surface, candidates in candidate_map.items():
            selected = judged.get(surface)
            for candidate in candidates:
                same = bool(selected and selected[0] == candidate)
                self.pair_decisions[self._pair_key(surface, candidate)] = EntityPairDecision(
                    left=surface,
                    right=candidate,
                    same_concept=same,
                    rationale=selected[1] if same and selected else "not selected as identical",
                )
            if selected:
                canonical_id = self.alias_to_id.get(selected[0])
                if canonical_id:
                    record = self.records[canonical_id]
                    self._upsert(
                        record.canonical_name,
                        [*record.aliases, surface],
                        decision_source="llm_pair_decision",
                    )

        for original, surface in surfaces.items():
            if not surface:
                continue
            canonical_id = self.alias_to_id.get(surface)
            if not canonical_id:
                record = self._upsert(
                    surface, [surface], decision_source="new_entity"
                )
            else:
                record = self.records[canonical_id]
            result[original] = record
        self._save()
        return result
