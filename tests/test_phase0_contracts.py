from pathlib import Path

import pytest
from pydantic import ValidationError

from hypoforge.config import PipelineConfig
from hypoforge.pipeline import PipelineRunner
from hypoforge.protocol import ModuleProtocol, SkillProtocol
from hypoforge.registry import ModuleRegistry, SkillRegistry
from hypoforge.state import (
    KnowledgeEntry,
    M2EvidenceExport,
    M2KnowledgeExport,
    M2KnowledgeRun,
    M2PaperExport,
    PipelineState,
)


class FakeM1(ModuleProtocol):
    module_name = "m1"
    module_version = "test"
    description = "test module"

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    async def __call__(self, state, config=None):
        return {"metrics": {"seen": state.input_question}}

    @classmethod
    def get_input_fields(cls):
        return ["input_question"]

    @classmethod
    def get_output_fields(cls):
        return ["metrics"]


def valid_run():
    return M2KnowledgeRun(
        sub_question="q",
        papers=[M2PaperExport(paper_id="p1", title="Paper")],
        evidence=[M2EvidenceExport(
            evidence_id="e1", paper_id="p1", chunk_id="c1", quote="quote",
            normalized_claim="claim", relevance_score=0.9,
        )],
        knowledge_entries=[KnowledgeEntry(
            id="k1", type="established_fact", content="claim",
            source_paper_id="p1", evidence_ids=["e1"]
        )],
    )


def test_m2_export_round_trip_retains_evidence():
    export = M2KnowledgeExport(runs=[valid_run()])
    restored = M2KnowledgeExport.model_validate_json(export.model_dump_json())
    assert restored.runs[0].evidence[0].evidence_id == "e1"
    assert restored.runs[0].knowledge_entries[0].evidence_ids == ["e1"]


@pytest.mark.parametrize("change", ["unknown", "cross-paper"])
def test_provenance_rejects_invalid_evidence(change):
    kwargs = valid_run().model_dump()
    if change == "unknown":
        kwargs["knowledge_entries"][0]["evidence_ids"] = ["missing"]
    else:
        kwargs["papers"].append(M2PaperExport(paper_id="p2", title="Other").model_dump())
        kwargs["evidence"][0]["paper_id"] = "p2"
    with pytest.raises(ValidationError):
        M2KnowledgeRun(**kwargs)


def test_all_yaml_configs_load():
    for path in Path("configs").glob("*.yaml"):
        if path.name == "evaluation.yaml":
            continue  # uses MasterEvaluationConfig, not PipelineConfig
        PipelineConfig.from_yaml(path)


def test_config_guards():
    with pytest.raises(ValidationError):
        PipelineConfig(grounding={"enabled": True}, search={"implementation": "automatic"})
    with pytest.raises(ValidationError):
        PipelineConfig(typo_field=True)


def test_custom_module_loader_validation():
    assert ModuleRegistry._load_module_class(
        "tests.test_phase0_contracts.FakeM1", "m1"
    ) is FakeM1
    with pytest.raises(ValueError):
        ModuleRegistry._load_module_class("tests.test_phase0_contracts.FakeM1", "m2")
    with pytest.raises(ValueError):
        ModuleRegistry._load_module_class("not-a-dotted-path", "m1")


def test_agentic_selection_builds_track_a_module():
    from hypoforge import modules  # noqa: F401
    from hypoforge.literature.adapter import AgenticM2Module
    from hypoforge.modules.m2_literature_search import M2LiteratureSearch

    config = PipelineConfig(search={"implementation": "agentic"})
    instances = ModuleRegistry.build_all(config)

    assert ModuleRegistry.get("m2") is M2LiteratureSearch
    assert isinstance(instances["m2"], AgenticM2Module)


class BeforeSkill(SkillProtocol):
    skill_name = "before_test"
    async def before(self, module_name, state):
        return {"input_question": "patched"}
    async def after(self, module_name, state_before, result, state_after):
        return {}


class CollisionSkill(BeforeSkill):
    skill_name = "collision_test"
    async def after(self, module_name, state_before, result, state_after):
        return {"metrics": {"collision": True}}


class BrokenSkill(BeforeSkill):
    skill_name = "broken_test"
    async def before(self, module_name, state):
        raise RuntimeError("broken")


@pytest.fixture(autouse=True)
def register_test_skills():
    original = dict(SkillRegistry._skill_classes)
    for skill in (BeforeSkill, CollisionSkill, BrokenSkill):
        SkillRegistry.register(skill)
    yield
    SkillRegistry._skill_classes = original


@pytest.mark.asyncio
async def test_before_patch_is_visible_to_module():
    runner = PipelineRunner(PipelineConfig(verbose=False))
    runner._skills = [BeforeSkill()]
    runner._current_run_id = "phase0-test"
    result = await runner._make_node_wrapper("m1", FakeM1())(PipelineState())
    assert result["metrics"]["seen"] == "patched"


def test_skill_instances_are_per_runner_and_unknown_rejected():
    first = SkillRegistry.build_enabled(["before_test"])[0]
    second = SkillRegistry.build_enabled(["before_test"])[0]
    assert first is not second
    with pytest.raises(ValueError, match="Unknown enabled skills"):
        SkillRegistry.build_enabled(["missing"])


@pytest.mark.asyncio
async def test_skill_collision_and_failure_policy():
    collision_runner = PipelineRunner(PipelineConfig(verbose=False, skill_fail_fast=True))
    collision_runner._skills = [CollisionSkill()]
    collision_runner._current_run_id = "phase0-collision"
    with pytest.raises(ValueError, match="cannot overwrite"):
        await collision_runner._make_node_wrapper("m1", FakeM1())(PipelineState())

    tolerant = PipelineRunner(PipelineConfig(verbose=False, skill_fail_fast=False))
    tolerant._skills = [BrokenSkill()]
    tolerant._current_run_id = "phase0-tolerant"
    result = await tolerant._make_node_wrapper("m1", FakeM1())(PipelineState())
    assert result["metrics"]["seen"] == ""

    strict = PipelineRunner(PipelineConfig(verbose=False, skill_fail_fast=True))
    strict._skills = [BrokenSkill()]
    strict._current_run_id = "phase0-strict"
    with pytest.raises(RuntimeError, match="broken"):
        await strict._make_node_wrapper("m1", FakeM1())(PipelineState())
