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
            The pipeline configuration (used to resolve per-module kwargs).

        Returns
        -------
        dict[str, ModuleProtocol]
            ``{"m1": <instance>, "m2": <instance>, …}``
        """
        instances: Dict[str, ModuleProtocol] = {}
        for name in cls.list_all():
            kwargs = config.get_module_kwargs(name) if hasattr(config, "get_module_kwargs") else {}
            instances[name] = cls._modules[name](**kwargs)
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

    Usage::

        @SkillRegistry.register
        class CitationFormatter(SkillProtocol):
            skill_name = "citation_formatter"
            ...
    """

    _skills: Dict[str, Type[SkillProtocol]] = {}

    @classmethod
    def register(cls, skill_cls: Type[SkillProtocol]) -> Type[SkillProtocol]:
        cls._skills[skill_cls.skill_name] = skill_cls
        return skill_cls

    @classmethod
    def get(cls, name: str) -> Optional[Type[SkillProtocol]]:
        return cls._skills.get(name)

    @classmethod
    def list_all(cls) -> List[str]:
        return sorted(cls._skills.keys())

    @classmethod
    def clear(cls) -> None:
        cls._skills.clear()
