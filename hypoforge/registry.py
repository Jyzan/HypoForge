"""
Registry system for HypoForge.

Three registries keep the framework extensible without touching core code:

* **ModuleRegistry** — M1–M6 pipeline modules
* **ToolRegistry** — external search / data-access tools
* **SkillRegistry** — lightweight before/after middleware

Each is a class-level registry; the ``@register`` decorator is the primary
API.  A module author simply writes::

    from hypoforge.registry import ModuleRegistry

    @ModuleRegistry.register
    class MyM2Impl(ModuleProtocol):
        module_name = "m2"
        ...
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, List, Optional, Type

from .protocol import ModuleProtocol, SkillProtocol, ToolProtocol


# ============================================================================
# Module Registry
# ============================================================================

class ModuleRegistry:
    """Registry for M1–M6 pipeline modules.

    Usage::

        @ModuleRegistry.register
        class M1ProblemUnderstanding(ModuleProtocol):
            module_name = "m1"
            ...

        # Later:
        impl_cls = ModuleRegistry.get("m1")
        instance = impl_cls(**config.get_module_kwargs("m1"))
    """

    _modules: Dict[str, Type[ModuleProtocol]] = {}

    @staticmethod
    def _load_module_class(dotted_path: str, expected_name: str) -> Type[ModuleProtocol]:
        module_path, separator, class_name = dotted_path.rpartition(".")
        if not separator or not module_path or not class_name:
            raise ValueError(
                f"Invalid module class path {dotted_path!r}; expected 'package.module.ClassName'"
            )
        try:
            module = importlib.import_module(module_path)
            module_cls = getattr(module, class_name)
        except (ImportError, AttributeError) as exc:
            raise ImportError(f"Cannot load module class {dotted_path!r}: {exc}") from exc
        if not isinstance(module_cls, type) or not issubclass(module_cls, ModuleProtocol):
            raise TypeError(f"{dotted_path!r} is not a ModuleProtocol subclass")
        if module_cls.module_name != expected_name:
            raise ValueError(
                f"Module class {dotted_path!r} declares module_name="
                f"{module_cls.module_name!r}, expected {expected_name!r}"
            )
        return module_cls

    # ------------------------------------------------------------------
    # Decorator-based registration
    # ------------------------------------------------------------------

    @classmethod
    def register(cls, module_cls: Type[ModuleProtocol]) -> Type[ModuleProtocol]:
        """Decorator: register a module implementation class.

        If a module with the same ``module_name`` already exists it is
        *replaced* — this is intentional so newer implementations can
        override stubs.
        """
        name = module_cls.module_name
        if name in cls._modules:
            old = cls._modules[name]
            import warnings
            warnings.warn(
                f"ModuleRegistry: '{name}' is being replaced "
                f"({old.__name__} → {module_cls.__name__})",
                stacklevel=2,
            )
        cls._modules[name] = module_cls
        return module_cls

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    @classmethod
    def get(cls, name: str) -> Optional[Type[ModuleProtocol]]:
        """Return the registered implementation class for *name*, or None."""
        return cls._modules.get(name)

    @classmethod
    def list_all(cls) -> List[str]:
        """Return all registered module names."""
        return sorted(cls._modules.keys())

    # ------------------------------------------------------------------
    # Bulk instantiation
    # ------------------------------------------------------------------

    @classmethod
    def build_all(cls, config: Any) -> Dict[str, ModuleProtocol]:
        """Instantiate every registered module using ``config.get_module_kwargs()``.

        Parameters
        ----------
        config : PipelineConfig
            The pipeline configuration (used to resolve per-module kwargs,
            tier assignment, agentic-M2 override, and grounding config).

        Returns
        -------
        dict[str, ModuleProtocol]
            ``{"m1": <instance>, "m2": <instance>, …}``
        """
        default_tiers = {
            "m1": "base",
            "m2": "turbo",
            "m3": "plus",
            "m4": "base",
            "m5": "plus",
            "m6": "plus",
        }

        embedding_cfg = getattr(
            getattr(config, "evaluation", None), "embedding", None
        )
        if hasattr(config, "entity_embedding_model"):
            # Respect an explicit empty string: it means lexical-only entity
            # normalization. Do not silently re-enable embeddings from the
            # evaluation defaults.
            resolved_entity_embedding_model = str(
                getattr(config, "entity_embedding_model", "") or ""
            )
        else:
            resolved_entity_embedding_model = str(
                getattr(embedding_cfg, "model_name", "") or ""
            )
        instances: Dict[str, ModuleProtocol] = {}
        for name in cls.list_all():
            kwargs = dict(config.get_module_kwargs(name)) if hasattr(config, "get_module_kwargs") else {}
            override = getattr(config, "module_overrides", {}).get(name)
            module_cls = cls._modules[name]
            if override is not None and override.class_name:
                module_cls = cls._load_module_class(override.class_name, name)
            elif (
                name == "m2"
                and module_cls.__module__ == "hypoforge.modules.m2_literature_search"
                and module_cls.__name__ == "M2LiteratureSearch"
            ):
                # Keep the Pipeline-facing M2 facade in the modules package;
                # only swap its implementation for the strict contract wrapper.
                module_cls = cls._load_module_class(
                    "hypoforge.strict_contracts.StrictM2LiteratureSearch", "m2"
                )
            elif (
                name == "m3"
                and module_cls.__module__ == "hypoforge.modules.m3_evidence_graph"
                and module_cls.__name__ == "M3EvidenceGraph"
            ):
                module_cls = cls._load_module_class(
                    "hypoforge.strict_contracts.StrictM3EvidenceGraph", "m3"
                )
            elif (
                name == "m4"
                and module_cls.__module__ == "hypoforge.modules.m4_hypothesis_generation"
                and module_cls.__name__ == "M4HypothesisGeneration"
            ):
                module_cls = cls._load_module_class(
                    "hypoforge.strict_contracts.StrictM4HypothesisGeneration", "m4"
                )
            elif (
                name == "m5"
                and module_cls.__module__ == "hypoforge.modules.m5_research_plan"
                and module_cls.__name__ == "M5ResearchPlan"
            ):
                module_cls = cls._load_module_class(
                    "hypoforge.strict_contracts.StrictM5ResearchPlan", "m5"
                )
            elif (
                name == "m6"
                and module_cls.__module__ == "hypoforge.modules.m6_review_iteration"
                and module_cls.__name__ == "M6ReviewIteration"
            ):
                module_cls = cls._load_module_class(
                    "hypoforge.strict_contracts.StrictM6ReviewIteration", "m6"
                )
            tier = kwargs.pop("llm_tier", default_tiers.get(name, "base"))
            if hasattr(config, "get_llm_for_tier"):
                kwargs.setdefault("llm_config", config.get_llm_for_tier(tier))
            # M2 query generation needs a more reliable model (turbo-tier
            # returns empty output for translation tasks on some endpoints).
            if name == "m2" and hasattr(config, "get_llm_for_tier"):
                secondary_tier = getattr(config, "secondary_model_tier", "base")
                kwargs.setdefault("query_llm_config", config.get_llm_for_tier(secondary_tier))

                # Agentic M2: per-round paper budget for gap-driven
                # supplement searches (cache-first incremental flow).
                kwargs.setdefault(
                    "supplement_paper_budget",
                    getattr(config, "supplement_paper_budget", 6),
                )

            if (
                name in {"m2", "m3"}
                and module_cls.__module__ == "hypoforge.strict_contracts"
                and embedding_cfg is not None
            ):
                kwargs.setdefault(
                    "entity_embedding_model",
                    resolved_entity_embedding_model,
                )
                kwargs.setdefault(
                    "entity_embedding_base_url",
                    getattr(embedding_cfg, "base_url", ""),
                )
                kwargs.setdefault(
                    "entity_embedding_key_env",
                    getattr(embedding_cfg, "api_key_env_var", ""),
                )

            # ---- M3 grounding config injection ----
            # The top-level `grounding:` block (GroundingConfig) is the single
            # master switch for the feature; flat `grounding_*` keys in
            # module_overrides.m3.kwargs remain fine-grained tuning knobs.
            # We force the master flags from config.grounding so the config
            # validator ("grounding.enabled requires agentic M2") and the
            # module can never disagree. (The old code injected a
            # `grounding_config=` kwarg that M3EvidenceGraph silently swallowed
            # via **kwargs — a dead wire.)
            if name == "m3":
                grounding = getattr(config, "grounding", None)
                if grounding is not None:
                    kwargs["grounding_enabled"] = grounding.enabled
                    if grounding.enabled:
                        kwargs["grounding_relation_selection_mode"] = (
                            "evidence_gams" if grounding.enable_gams else "direct"
                        )

            # ---- iteration-core feature switches ----
            # M1 needs the followup-routing switch to decide search-free
            # followups; M6 needs the evidence-revisit switch to run the
            # evidence-sufficiency verdict.  Both default to False.
            if name == "m1":
                kwargs.setdefault(
                    "followup_routing", getattr(config, "followup_routing", False)
                )
            if name == "m3":
                # Gap confirmation limit + run output dir (for lossless
                # graph-round snapshots) come from the pipeline config.
                kwargs.setdefault(
                    "gap_no_improvement_limit",
                    getattr(config, "gap_no_improvement_limit", 1),
                )
                kwargs.setdefault(
                    "output_dir", getattr(config, "output_dir", "")
                )
            if name == "m6":
                kwargs.setdefault(
                    "m6_evidence_revisit",
                    getattr(config, "m6_evidence_revisit", False),
                )

            # Inject scoring weights into M4 so the composite formula has a
            # single source of truth (PipelineConfig.scoring → rubric defaults).
            if name == "m4":
                scoring = getattr(config, "scoring", None)
                if scoring is not None:
                    kwargs.setdefault("weights", dict(scoring.hypothesis_weights))
                kwargs.setdefault("interactive", getattr(config, "interactive", False))
                # Ranker uses a separate tier so the model that scores
                # hypotheses is not the same model that generated them.
                if hasattr(config, "get_llm_for_tier") and "ranker_llm_config" not in kwargs:
                    secondary_tier = getattr(config, "secondary_model_tier", "base")
                    kwargs["ranker_llm_config"] = config.get_llm_for_tier(secondary_tier)

            instances[name] = module_cls(**kwargs)
        return instances

    @classmethod
    def clear(cls) -> None:
        """Remove all registered modules (mainly for testing)."""
        cls._modules.clear()


# ============================================================================
# Tool Registry
# ============================================================================

class ToolRegistry:
    """Registry for external tools (Semantic Scholar, PubMed, …).

    Usage::

        @ToolRegistry.register
        class SemanticScholarTool(ToolProtocol):
            tool_name = "semantic_scholar"
            ...
    """

    _tools: Dict[str, Type[ToolProtocol]] = {}

    @classmethod
    def register(cls, tool_cls: Type[ToolProtocol]) -> Type[ToolProtocol]:
        cls._tools[tool_cls.tool_name] = tool_cls
        return tool_cls

    @classmethod
    def get(cls, name: str) -> Optional[Type[ToolProtocol]]:
        return cls._tools.get(name)

    @classmethod
    def list_all(cls) -> List[str]:
        return sorted(cls._tools.keys())

    @classmethod
    def clear(cls) -> None:
        cls._tools.clear()


# ============================================================================
# Skill Registry
# ============================================================================

class SkillRegistry:
    """Registry for middleware skills.

    The registry stores classes only. ``PipelineRunner`` creates fresh
    instances for each runner and reuses them only across that runner's nodes.

    Usage::

        @SkillRegistry.register
        class CitationFormatter(SkillProtocol):
            skill_name = "citation_formatter"
            ...

        # In PipelineRunner._build_graph():
        skills = SkillRegistry.build_enabled(["citation_formatter"])
    """

    _skill_classes: Dict[str, Type[SkillProtocol]] = {}

    @classmethod
    def register(cls, skill_cls: Type[SkillProtocol]) -> Type[SkillProtocol]:
        cls._skill_classes[skill_cls.skill_name] = skill_cls
        return skill_cls

    @classmethod
    def get(cls, name: str) -> Optional[Type[SkillProtocol]]:
        return cls._skill_classes.get(name)

    @classmethod
    def list_all(cls) -> List[str]:
        return sorted(cls._skill_classes)

    @classmethod
    def build_enabled(cls, names: List[str]) -> List[SkillProtocol]:
        """Build fresh instances for one PipelineRunner."""
        unknown = sorted(set(names) - set(cls._skill_classes))
        if unknown:
            raise ValueError(f"Unknown enabled skills: {unknown}")
        return [cls._skill_classes[name]() for name in names]

    @classmethod
    def clear(cls) -> None:
        cls._skill_classes.clear()
