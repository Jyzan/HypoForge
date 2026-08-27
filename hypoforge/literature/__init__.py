"""Compatibility alias for the canonical M2 package.

New code must import :mod:`hypoforge.modules.m2_literature`.  This loader
keeps historical deep imports working while ensuring they resolve to the
same module and class objects as the canonical namespace.
"""

from __future__ import annotations

import importlib
import pkgutil
import sys

_LEGACY_PREFIX = __name__
_CANONICAL_PREFIX = "hypoforge.modules.m2_literature"
_canonical = importlib.import_module(_CANONICAL_PREFIX)

for module_info in pkgutil.walk_packages(
    _canonical.__path__,
    prefix=f"{_CANONICAL_PREFIX}.",
):
    canonical_name = module_info.name
    canonical_module = importlib.import_module(canonical_name)
    legacy_name = canonical_name.replace(
        _CANONICAL_PREFIX,
        _LEGACY_PREFIX,
        1,
    )
    sys.modules[legacy_name] = canonical_module

sys.modules[_LEGACY_PREFIX] = _canonical
