"""
Skill base classes and example stubs.

Skills are middleware that hook before/after module execution.
They receive ``(module_name, state)`` in ``before()`` and
``(module_name, state_before, result, state_after)`` in ``after()``,
returning a ``Dict[str, Any]`` patch to merge into the node output.

.. versionchanged:: 0.2.0
    Skill interface updated — before/after now return patch dicts instead of
    full PipelineState, and after() receives state_before/result/state_after.
"""

from typing import Any, Dict

from ..protocol import SkillProtocol
from ..registry import SkillRegistry


# ============================================================================
# Example skill: console logging of module I/O
# ============================================================================

@SkillRegistry.register
class LoggingSkill(SkillProtocol):
    """Logs the state fields being read/written by each module.

    In a real implementation this would write structured logs to
    ``output/{run_id}_execution_log.json``.  The stub is a no-op so it
    can serve as a template for Track E.
    """

    skill_name = "logging"

    async def before(self, module_name: str, state) -> Dict[str, Any]:
        return {}

    async def after(
        self,
        module_name: str,
        state_before,
        result: Dict[str, Any],
        state_after,
    ) -> Dict[str, Any]:
        return {}


# ============================================================================
# Placeholder — team members add real skills here during Track E
# ============================================================================
