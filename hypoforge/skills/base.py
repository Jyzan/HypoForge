"""
Skill base classes and example stubs.

Skills are middleware that hook before/after module execution.
"""

from ..protocol import SkillProtocol
from ..registry import SkillRegistry


# ============================================================================
# Example skill: console logging of module I/O
# ============================================================================

@SkillRegistry.register
class LoggingSkill(SkillProtocol):
    """Logs the state fields being read/written by each module."""

    skill_name = "logging"

    async def before(self, state, **kwargs):
        return state

    async def after(self, state, **kwargs):
        return state


# ============================================================================
# Placeholder — team members add real skills here during Phase 1-2
# ============================================================================
