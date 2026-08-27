"""Contracts for deterministic, auditable LLM context packs."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


ContextPurpose = Literal[
    "m4_generate",
    "m4_epistemic_audit",
    "m4_critic",
    "m4_rank",
    "m5_plan",
    "m6_logic",
    "m6_feasibility",
    "m6_sufficiency",
]


class ContextRequest(BaseModel):
    purpose: ContextPurpose
    max_input_tokens: int = Field(default=3600, gt=0)
    reserve_tokens: int = Field(default=720, ge=0)
    focus_evidence_ids: tuple[str, ...] = ()
    focus_entry_ids: tuple[str, ...] = ()


class ContextManifest(BaseModel):
    purpose: ContextPurpose
    budget_tokens: int
    estimated_tokens: int
    included_ids: list[str] = Field(default_factory=list)
    dropped_ids: list[str] = Field(default_factory=list)
    drop_reasons: dict[str, str] = Field(default_factory=dict)
    render_hash: str


class ContextPack(BaseModel):
    rendered: str
    manifest: ContextManifest
