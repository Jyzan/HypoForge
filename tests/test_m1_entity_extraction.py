"""Regression tests for M1 entity-extraction fallback used by rhetorical/state questions."""

import pytest

from hypoforge.modules.m1_problem_understanding import M1ProblemUnderstanding


class FakeClient:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = []

    async def structured_chat(self, **kwargs):
        self.calls.append(kwargs)
        return self.payloads.pop(0)


@pytest.mark.asyncio
async def test_lenient_fallback_admits_strict_audit_rejected_primary():
    """When the strict audit rejects the only plausible primary because the
    canonical name makes implicit context explicit, M1 should fall back to a
    source-verbatim best-effort entity instead of failing the whole run."""
    client = FakeClient([
        {
            "entities": [{
                "name": "Navier-Stokes existence and smoothness problem",
                "source_mention": "Navier-Stokes problem",
                "aliases": [],
                "role": "primary_object",
                "required": True,
                "extraction_reason": "",
            }],
        },
        {
            "items": [{
                "candidate_name": "Navier-Stokes existence and smoothness problem",
                "accepted": False,
                "source_mention": "Navier-Stokes problem",
                "reason": "adds inferred subtype",
            }],
            "missing_explicit_entities": ["Navier-Stokes problem"],
            "complete": False,
        },
    ])
    module = M1ProblemUnderstanding(
        mode="llm",
        coverage_max_rounds=0,
        entity_repair_attempts=0,
        requirement_repair_attempts=0,
    )
    module.client = client

    entities = await module._extract_and_audit_entities(
        "Will the Navier-Stokes problem ever be solved?"
    )

    assert len(entities) == 1
    assert entities[0].name == "Navier-Stokes existence and smoothness problem"
    assert entities[0].source_mention == "Navier-Stokes problem"
    assert entities[0].role == "primary_object"
    assert entities[0].required is True
