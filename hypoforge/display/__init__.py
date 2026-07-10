"""Rich-powered terminal display for HypoForge.

Heavily inspired by Flash-Searcher's ``AgentLogger`` and BioDSA's
``render_utils``.  All terminal output goes through this module so the
visual style is consistent and can be customised in one place.
"""

from rich.console import Console

# ---------------------------------------------------------------------------
# Global colour palette (tune these hex values to match your brand)
# ---------------------------------------------------------------------------
COLORS = {
    "primary":   "#3b82f6",   # blue   — module headers, system info
    "success":   "#10b981",   # green  — completion, passing scores
    "warning":   "#f59e0b",   # amber  — conflicts, cautions
    "error":     "#ef4444",   # red    — errors, rejections
    "highlight": "#8b5cf6",   # purple — hypotheses, knowledge gaps
    "muted":     "#6b7280",   # grey   — secondary text, metadata
}

# Single console instance — import this everywhere
console = Console()

# Convenience re-exports so display callers don't need to import rich directly
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.syntax import Syntax
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich import box
from rich.text import Text
