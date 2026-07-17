"""Progress reporting shared by the CLI jobs (ingest, chunking, sessions,
lexicon_build, annotate, topics, wrapped).

One entry point, `track`, used as a context manager around a long loop:

    with track("Embedding chunks", total=len(rows)) as p:
        for row in rows:
            ...
            p.advance()

On a terminal it draws a live rich bar with a percentage and an ETA. When
stdout is piped or redirected (including under the test suite's
redirect_stdout) it degrades to periodic plain lines -- the `  ...N/Total`
style the jobs printed by hand before this module existed -- so logs and
transcripts stay free of escape codes. PROGRESS_STYLE=off silences it.

Progress is cosmetic: `advance` never commits, and a job's commit cadence is
independent of its log cadence.
"""

import sys
import time
from contextlib import contextmanager

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

from config import PROGRESS_LOG_EVERY, PROGRESS_STYLE


def _fmt_duration(seconds):
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"


class _Null:
    """PROGRESS_STYLE=off, or a loop with nothing to report."""

    def advance(self, n=1):
        pass


# Ceiling on plain-mode lines per loop. PROGRESS_LOG_EVERY is a floor on the
# gap between lines, but a fixed gap doesn't scale: the lexicon pass streams
# ~350k messages, which at one line per 200 would bury a log under ~1700
# lines. Whichever cadence is coarser wins, so a big loop stretches its stride
# instead of shouting.
_MAX_PLAIN_LINES = 50


class _Plain:
    """Periodic newline-terminated lines, for non-terminal stdout."""

    def __init__(self, description, total, log_every):
        self.description = description
        self.total = total
        if total:
            log_every = max(log_every, -(-total // _MAX_PLAIN_LINES))
        self.log_every = max(1, log_every)
        self.done = 0
        self.logged = 0
        self.started = time.monotonic()

    def advance(self, n=1):
        self.done += n
        if self.done - self.logged < self.log_every:
            return
        self.logged = self.done
        elapsed = time.monotonic() - self.started
        if self.total:
            pct = 100 * self.done / self.total
            line = f"  ...{self.done}/{self.total} ({pct:.0f}%)"
            # Only project an ETA once the rate has some substance behind it.
            if elapsed > 5:
                remaining = (self.total - self.done) * elapsed / self.done
                line += f"  eta {_fmt_duration(remaining)}"
        else:
            line = f"  ...{self.done}  ({_fmt_duration(elapsed)})"
        print(line, flush=True)


class _Rich:
    def __init__(self, progress, task_id):
        self._progress = progress
        self._task_id = task_id

    def advance(self, n=1):
        self._progress.advance(self._task_id, n)


def _use_bar():
    if PROGRESS_STYLE == "off" or PROGRESS_STYLE == "plain":
        return False
    # 'auto': a bar only makes sense on a terminal. Ask sys.stdout at call
    # time rather than caching -- tests redirect it.
    try:
        return bool(sys.stdout.isatty())
    except (AttributeError, ValueError):
        return False


@contextmanager
def track(description, total=None, log_every=None, header=True,
          transient=False, bar_only=False, enabled=True, parent=None):
    """Report progress over a loop of `total` items (None = unknown total,
    which reports a running count and elapsed time instead).

    `enabled=False` reports nothing, for a loop that is only slow under some
    runtime condition the caller knows about.

    `header=False` suppresses the plain-mode opening line, for callers that
    already print a start line of their own.

    `transient=True` erases the bar on completion -- for an inner loop whose
    caller prints its own per-pass summary line.

    `bar_only=True` reports nothing at all in plain mode. For fine-grained
    inner loops that run many times: a live bar is a nicety on a terminal,
    but the same detail as periodic lines would bury a log.

    `parent=<handle from an enclosing track>` stacks this loop as a second
    bar under the parent's -- rich allows one live display at a time, so a
    nested track must join the enclosing one rather than start its own.
    Only meaningful when the parent is drawing a bar; a plain/off parent
    handle is inert and the normal mode selection runs.
    """
    if not enabled or PROGRESS_STYLE == "off" or total == 0:
        # total == 0: an empty loop reports nothing in either mode -- no
        # header, no flash of an empty bar.
        yield _Null()
        return

    if isinstance(parent, _Rich):
        # Nested bar: add a task to the parent's live display. transient
        # is emulated with remove_task -- rich's own flag is per-Progress,
        # and the parent's display outlives this loop.
        prog = parent._progress
        task_id = prog.add_task(description, total=total)
        try:
            yield _Rich(prog, task_id)
        finally:
            if transient:
                prog.remove_task(task_id)
        return

    if not _use_bar():
        if bar_only:
            yield _Null()
            return
        if header:
            print(f"{description}..." if total is None else f"{description}: {total}",
                  flush=True)
        yield _Plain(description, total, log_every or PROGRESS_LOG_EVERY)
        return

    columns = [
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(compact=True),
    ]
    if total is None:
        # No total: a bar and a percentage would both be lies.
        columns = [SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                   MofNCompleteColumn()]
    # console=Console() (file unset) resolves sys.stdout on each write.
    progress = Progress(*columns, console=Console(), transient=transient)
    # Progress.__exit__ stops the live region and restores the cursor, so a
    # Ctrl-C through here never leaves the terminal in a broken state.
    with progress:
        task_id = progress.add_task(description, total=total)
        yield _Rich(progress, task_id)
