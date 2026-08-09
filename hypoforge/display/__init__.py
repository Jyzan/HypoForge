"""Rich-powered terminal display for HypoForge.

Heavily inspired by Flash-Searcher's ``AgentLogger`` and BioDSA's
``render_utils``.  All terminal output goes through this module so the
visual style is consistent and can be customised in one place.
"""

import sys

from rich.console import Console

# ---------------------------------------------------------------------------
# Global colour palette (tune these hex values to match your brand)
# ---------------------------------------------------------------------------
COLORS = {
    "info":      "#22d3ee",
    "primary":   "#3b82f6",   # blue   — module headers, system info
    "success":   "#10b981",   # green  — completion, passing scores
    "warning":   "#f59e0b",   # amber  — conflicts, cautions
    "soft_warning": "#fdba74", # light orange — secondary analysis accents
    "soft_pink": "#f9a8d4",    # pink — metrics section accent
    "error":     "#ef4444",   # red    — errors, rejections
    "highlight": "#8b5cf6",   # purple — hypotheses, knowledge gaps
    "muted":     "#6b7280",   # grey   — secondary text, metadata
}

# Single console instance; import this everywhere
# Rich uses the stream encoding to decide whether Unicode box glyphs are safe.
# Some Windows Anaconda environments expose ``gbk`` even though the terminal
# itself supports UTF-8, which would silently downgrade rounded panels to ASCII.
# Reconfigure the terminal streams when possible, then explicitly disable
# Rich's legacy Windows renderer / safe-box substitution.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass

console = Console(
    force_terminal=True,
    legacy_windows=False,
    safe_box=False,
)

# Convenience re-exports so display callers don't need to import rich directly
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.syntax import Syntax
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich import box
from rich.text import Text
