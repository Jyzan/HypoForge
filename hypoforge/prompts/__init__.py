"""Prompt templates for HypoForge — central import hub.

Import convention::

    from hypoforge.prompts import m1_prompts
    system_prompt = m1_prompts.M1_SYSTEM_PROMPT
"""

from . import m1_prompts
from . import m3_prompts
from . import m4_prompts
from . import m5_prompts
from . import m6_prompts

__all__ = [
    "m1_prompts",
    "m3_prompts",
    "m4_prompts",
    "m5_prompts",
    "m6_prompts",
]
