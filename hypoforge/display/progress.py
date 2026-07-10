"""Rich progress indicators for long-running pipeline phases."""

from contextlib import contextmanager
from typing import Optional

from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
)

from . import console


@contextmanager
def spinner(label: str = "Working…"):
    """Show a spinner while an async or blocking operation runs.

    Usage::

        with spinner("Searching PubMed..."):
            results = await tool.search(query)
    """
    with console.status(f"[bold]{label}[/bold]", spinner="dots"):
        yield


@contextmanager
def progress_bar(
    total: int,
    description: str = "Processing",
    transient: bool = True,
):
    """Show a progress bar for a loop with a known total.

    Usage::

        with progress_bar(len(papers), "Extracting knowledge") as bar:
            for paper in papers:
                extract(paper)
                bar.advance(1)
    """
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("({task.completed}/{task.total})"),
        TimeElapsedColumn(),
        console=console,
        transient=transient,
    )
    task_id = progress.add_task(description, total=total)
    progress.start()

    class _Wrapper:
        def advance(self, n: int = 1):
            progress.update(task_id, advance=n)

    try:
        yield _Wrapper()
    finally:
        progress.stop()
