"""Terminal presentation of a LunarOps run.

All run-time output of ``lunarops run`` and of the programs goes through this module so that it has one
look: a run banner, one section per program call with aligned, indented detail lines, uniform progress bars
and a closing summary::

    LunarOps  configs/lunarops_oc_residuals.yml
      7 program calls · serial · working dir .

    ▶ [1/7] IersC04EarthOrientationParameter
        samples  23,650
        output   output/eop_c04.txt
      ✔ done in 0.8 s
    ...
    ────────────────────────────────────────────────────────────
    ✔ 7 program calls completed in 4 min 48.8 s

Everything, progress included, goes to standard output in order. Progress has two modes:

* interactive (standard output is a terminal and the process is not under an MPI launcher): one line,
  redrawn in place a few times a second and cleared when the work is done;
* log (everything else: files, pipes, ``sbatch``, ``tmux ... > log``, any MPI run): a complete line every
  ``LUNAROPS_PROGRESS_INTERVAL`` seconds (default 30) and one line when the work is done, so a log file
  can be read with ``less``, ``grep`` or ``tail -f``.

Colour and Unicode symbols are used only when standard output is a terminal that supports them;
``NO_COLOR`` disables colour and ``FORCE_COLOR`` forces it. Under an MPI launcher colour is off unless
``FORCE_COLOR`` is set, because Open MPI makes rank 0's standard output look like a terminal even when
the job writes to a file.
"""

from __future__ import annotations

import math
import os
import shutil
import sys
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

INDENT = "  "
"""Indentation of a program section's detail lines (one level)."""

PROGRESS_INTERVAL_VARIABLE = "LUNAROPS_PROGRESS_INTERVAL"
"""Environment variable with the seconds between progress lines in log mode."""

_RULE_WIDTH = 60
_DEFAULT_LOG_INTERVAL_SECONDS = 30.0
_INTERACTIVE_INTERVAL_SECONDS = 0.2
_BAR_WIDTH = 24

_UNICODE_SYMBOLS = {"start": "▶", "ok": "✔", "fail": "✘", "dot": "·", "rule": "─", "arrow": "→"}
_ASCII_SYMBOLS = {"start": ">", "ok": "ok", "fail": "FAILED", "dot": "-", "rule": "-", "arrow": "->"}

_CODES = {
    "bold": "1",
    "dim": "2",
    "red": "31",
    "green": "32",
    "yellow": "33",
    "cyan": "36",
}


_MPI_LAUNCHER_VARIABLES = ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "PMIX_RANK", "MPI_LOCALNRANKS")


def _is_tty(stream) -> bool:
    try:
        return bool(stream.isatty())
    except (AttributeError, ValueError):
        return False


def _under_mpi_launcher() -> bool:
    # Open MPI gives rank 0 a pseudo-terminal for stdout even when the job output goes to a file.
    return any(name in os.environ for name in _MPI_LAUNCHER_VARIABLES)


def _use_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return _is_tty(sys.stdout) and not _under_mpi_launcher() and os.environ.get("TERM", "") != "dumb"


def _interactive() -> bool:
    return _is_tty(sys.stdout) and not _under_mpi_launcher() and os.environ.get("TERM", "") != "dumb"


def _log_interval() -> float:
    try:
        value = float(os.environ.get(PROGRESS_INTERVAL_VARIABLE, _DEFAULT_LOG_INTERVAL_SECONDS))
    except ValueError:
        return _DEFAULT_LOG_INTERVAL_SECONDS
    return value if value > 0 and math.isfinite(value) else _DEFAULT_LOG_INTERVAL_SECONDS


def _use_unicode() -> bool:
    encoding = getattr(sys.stdout, "encoding", None) or ""
    return encoding.lower().replace("-", "") in {"utf8", "utf16", "utf32"}


def _symbol(name: str) -> str:
    return (_UNICODE_SYMBOLS if _use_unicode() else _ASCII_SYMBOLS)[name]


def style(text: str, *styles: str) -> str:
    """Wrap text in ANSI styles when colour is enabled.

    Parameters
    ----------
    text : str
        Text to style.
    *styles : str
        Style names: ``bold``, ``dim``, ``red``, ``green``, ``yellow`` or ``cyan``.

    Returns
    -------
    str
        The styled text, or ``text`` unchanged when colour is disabled.
    """
    if not styles or not _use_color():
        return text
    codes = ";".join(_CODES[name] for name in styles)
    return f"\033[{codes}m{text}\033[0m"


def emit(line: str = "") -> None:
    """Write one line to standard output; an interactive progress line is cleared and redrawn below it."""
    active = _ACTIVE_PROGRESS
    if active is not None:
        active._clear()
    sys.stdout.write(line + "\n")
    if active is not None:
        active._draw()
    sys.stdout.flush()


# --------------------------------------------------------------------------------------------- formatting


def format_duration(seconds: float) -> str:
    """Format a duration as ``0.8 s``, ``2 min 05 s`` or ``1 h 02 min``."""
    if seconds < 60:
        return f"{seconds:.1f} s"
    total = round(seconds)
    if total < 3600:
        return f"{total // 60} min {total % 60:02d} s"
    return f"{total // 3600} h {total % 3600 // 60:02d} min"


def format_path(path: str | os.PathLike[str]) -> str:
    """Return a path relative to the current directory when it lies inside it, else the absolute path."""
    resolved = Path(path).expanduser().absolute()
    try:
        return str(resolved.relative_to(Path.cwd()))
    except ValueError:
        return str(resolved)


def format_value(value: object) -> str:
    """Format a summary value: integers with thousands separators, floats to four significant digits."""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return str(value)
        return f"{value:.4g}"
    if isinstance(value, os.PathLike):
        return format_path(value)
    return str(value)


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


# ------------------------------------------------------------------------------------------- run sections


def run_header(config_path: str | os.PathLike[str], *, call_count: int, mode: str, working_dir: str) -> None:
    """Print the banner of a run.

    Parameters
    ----------
    config_path : path-like
        Config file of the run.
    call_count : int
        Number of program calls.
    mode : str
        Execution mode, e.g. ``serial`` or ``MPI, 8 workers``.
    working_dir : str
        Base directory of relative output paths.
    """
    dot = style(_symbol("dot"), "dim")
    emit(f"{style('LunarOps', 'bold')}  {format_path(config_path)}")
    details = (_plural(call_count, "program call"), mode, f"working dir {format_path(working_dir)}")
    emit(INDENT + f" {dot} ".join(style(item, "dim") for item in details))


def program_header(index: int, total: int, name: str) -> None:
    """Print the heading of program call ``index`` of ``total``."""
    emit()
    counter = style(f"[{index}/{total}]", "dim")
    emit(f"{style(_symbol('start'), 'cyan')} {counter} {style(name, 'bold', 'cyan')}")


def program_footer(elapsed: float, *, ok: bool = True) -> None:
    """Print the closing line of a program call with its elapsed time."""
    if ok:
        emit(f"{INDENT}{style(_symbol('ok'), 'green')} {style('done in ' + format_duration(elapsed), 'dim')}")
    else:
        emit(f"{INDENT}{style(_symbol('fail') + ' failed after ' + format_duration(elapsed), 'red', 'bold')}")


def run_footer(call_count: int, elapsed: float, *, failed: str | None = None) -> None:
    """Print the closing summary of a run.

    Parameters
    ----------
    call_count : int
        Number of program calls that completed.
    elapsed : float
        Wall time of the run in seconds.
    failed : str, optional
        Label of the program call that failed or was interrupted; the run succeeded when None.
    """
    emit()
    emit(style(_symbol("rule") * _RULE_WIDTH, "dim"))
    if failed is None:
        message = f"{_symbol('ok')} {_plural(call_count, 'program call')} completed in {format_duration(elapsed)}"
        emit(style(message, "green", "bold"))
    else:
        message = (
            f"{_symbol('fail')} run stopped in {failed} after {format_duration(elapsed)} "
            f"({_plural(call_count, 'program call')} completed)"
        )
        emit(style(message, "red", "bold"))


# -------------------------------------------------------------------------------------- program details


def info(message: str) -> None:
    """Print a status message inside the current program section."""
    emit(f"{INDENT}{style(_symbol('dot'), 'dim')} {message}")


def warning(message: str) -> None:
    """Print a highlighted warning inside the current program section."""
    emit(f"{INDENT}{style('! ' + message, 'yellow')}")


def summary(
    values: Mapping[str, object] | None = None,
    *,
    outputs: Iterable[str | os.PathLike[str] | None] = (),
) -> None:
    """Print the result of a program as aligned ``label  value`` lines followed by its output files.

    Parameters
    ----------
    values : Mapping, optional
        Result quantities keyed by label, printed in order.
    outputs : iterable of path-like, optional
        Files the program wrote; None entries are skipped.
    """
    rows = [(label, format_value(value)) for label, value in (values or {}).items()]
    paths = [format_path(path) for path in outputs if path is not None]
    labels = [label for label, _ in rows] + (["output"] if paths else [])
    width = max((len(label) for label in labels), default=0)
    for label, text in rows:
        emit(f"{INDENT * 2}{style(label.ljust(width), 'dim')}  {text}")
    for number, path in enumerate(paths):
        label = "output" if number == 0 else ""
        emit(f"{INDENT * 2}{style(label.ljust(width), 'dim')}  {path}")


@dataclass(frozen=True, slots=True)
class Column:
    """One column of a ``Table``: header, width, alignment (``<`` or ``>``) and value format."""

    header: str
    width: int
    align: str = ">"
    format: str = ""


class Table:
    """Table printed row by row, with its header printed before the first row.

    Parameters
    ----------
    columns : sequence of Column
        Columns of the table.
    """

    def __init__(self, columns: Sequence[Column]) -> None:
        self.columns = tuple(columns)
        self._started = False

    def _line(self, cells: Sequence[str]) -> str:
        parts = [f"{cell:{column.align}{column.width}}" for column, cell in zip(self.columns, cells, strict=True)]
        return INDENT * 2 + "  ".join(parts).rstrip()

    def row(self, *values: Any) -> None:
        """Print one row; None is shown as ``-``."""
        if not self._started:
            self._started = True
            emit(style(self._line([column.header for column in self.columns]), "dim"))
        cells = [
            "-" if value is None else format(value, column.format) if column.format else format_value(value)
            for column, value in zip(self.columns, values, strict=True)
        ]
        emit(self._line(cells))


_ACTIVE_PROGRESS: Progress | None = None


class Progress:
    """Progress of a counted piece of work, shown in interactive or log mode (see the module docstring).

    Use it as an iterable wrapper (``for item in progress(items, ...)``) or as a context manager that is
    advanced with ``update``. Leaving the loop or the ``with`` block normally reports completion; an
    exception ends the progress silently, since the program section reports the failure.

    Parameters
    ----------
    iterable : iterable, optional
        Items to yield while counting them.
    total : int, optional
        Expected number of items; without it no percentage or remaining time is shown.
    desc : str
        Label of the work.
    unit : str
        Plural name of the items, e.g. ``tasks``.
    disable : bool
        Count without showing anything.
    interactive : bool, optional
        Redraw one line in place (True) or print log lines (False); detected from standard output when None.
    interval : float, optional
        Seconds between log lines; ``LUNAROPS_PROGRESS_INTERVAL`` or 30 when None.
    clock : callable, optional
        Monotonic clock in seconds, for tests.
    """

    def __init__(
        self,
        iterable: Iterable[Any] | None = None,
        *,
        total: int | None = None,
        desc: str,
        unit: str = "items",
        disable: bool = False,
        interactive: bool | None = None,
        interval: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.iterable = iterable
        self.total = total
        self.desc = desc
        self.unit = unit
        self.disable = disable
        self.count = 0
        self.interactive = _interactive() if interactive is None else interactive
        self.interval = (
            _INTERACTIVE_INTERVAL_SECONDS if self.interactive else _log_interval() if interval is None else interval
        )
        self._clock = clock
        self._started = clock()
        self._next_report = self._started + self.interval
        self._drawn = False
        self._closed = False
        self._previous_active: Progress | None = None
        if not disable and self.interactive:
            global _ACTIVE_PROGRESS
            self._previous_active, _ACTIVE_PROGRESS = _ACTIVE_PROGRESS, self
            self._draw()

    # ---------------------------------------------------------------------------------------------- usage

    def __iter__(self) -> Iterator[Any]:
        if self.iterable is None:
            raise TypeError("Progress without an iterable is advanced with update().")
        try:
            for item in self.iterable:
                yield item
                self.update()
        except BaseException:
            self.close(completed=False)
            raise
        self.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close(completed=exc_type is None)

    def update(self, n: int = 1) -> None:
        """Count ``n`` more items and report when the refresh interval has passed."""
        self.count += n
        if self.disable:
            return
        now = self._clock()
        if now < self._next_report:
            return
        self._next_report = now + self.interval
        if self.interactive:
            self._draw()
        else:
            sys.stdout.write(self._log_line(now) + "\n")
            sys.stdout.flush()

    def close(self, *, completed: bool = True) -> None:
        """End the progress; a completed log-mode progress prints its final line."""
        if self._closed:
            return
        self._closed = True
        if self.disable:
            return
        if self.interactive:
            global _ACTIVE_PROGRESS
            self._clear()
            _ACTIVE_PROGRESS = self._previous_active
            sys.stdout.flush()
        elif completed:
            sys.stdout.write(self._log_line(self._clock(), final=True) + "\n")
            sys.stdout.flush()

    # ------------------------------------------------------------------------------------------ rendering

    def _counts(self) -> str:
        if self.total is None:
            return f"{self.count:,} {self.unit}"
        return f"{self.count:,}/{self.total:,} {self.unit}"

    def _rate(self, elapsed: float) -> str:
        if elapsed <= 0 or self.count == 0:
            return "-/s"
        rate = self.count / elapsed
        return f"{rate:,.0f}/s" if rate >= 100 else f"{rate:.3g}/s"

    def _remaining(self, elapsed: float) -> str | None:
        if self.total is None or self.count == 0 or self.count >= self.total:
            return None
        return format_duration(elapsed / self.count * (self.total - self.count))

    def _fraction(self) -> float | None:
        if not self.total:
            return None
        return min(1.0, self.count / self.total)

    def _log_line(self, now: float, *, final: bool = False) -> str:
        elapsed = now - self._started
        fraction = self._fraction()
        percent = "" if fraction is None else f" ({100 * fraction:.0f}%)"
        if final:
            details = f"done in {format_duration(elapsed)}, {self._rate(elapsed)}"
        else:
            remaining = self._remaining(elapsed)
            details = f"{self._rate(elapsed)}, elapsed {format_duration(elapsed)}"
            if remaining is not None:
                details += f", ETA {remaining}"
        return f"{INDENT * 2}{self.desc}  {self._counts()}{percent}  {style(details, 'dim')}"

    def _interactive_line(self) -> str:
        elapsed = self._clock() - self._started
        fraction = self._fraction()
        parts = [INDENT * 2 + self.desc]
        if fraction is not None:
            filled = round(fraction * _BAR_WIDTH)
            full, empty = ("█", "░") if _use_unicode() else ("#", "-")
            parts.append(f"{100 * fraction:3.0f}% {full * filled}{empty * (_BAR_WIDTH - filled)}")
        parts.append(self._counts())
        details = f"{self._rate(elapsed)}  {format_duration(elapsed)}"
        remaining = self._remaining(elapsed)
        if remaining is not None:
            details += f"  ETA {remaining}"
        parts.append(details)
        line = "  ".join(parts)
        width = shutil.get_terminal_size((100, 24)).columns - 1
        return line[:width]

    def _draw(self) -> None:
        if self.disable or not self.interactive or self._closed:
            return
        sys.stdout.write("\r" + self._interactive_line() + "\033[K")
        sys.stdout.flush()
        self._drawn = True

    def _clear(self) -> None:
        if self._drawn:
            sys.stdout.write("\r\033[K")
            self._drawn = False


def progress(
    iterable: Iterable[Any] | None = None,
    *,
    total: int | None = None,
    desc: str,
    unit: str = "items",
    disable: bool = False,
) -> Progress:
    """Return a ``Progress`` in the mode that suits standard output.

    Parameters
    ----------
    iterable : iterable, optional
        Items to yield while counting them; omit to advance the progress with ``update``.
    total : int, optional
        Expected number of items.
    desc : str
        Label of the work.
    unit : str, optional
        Plural name of the items.
    disable : bool, optional
        Count without showing anything.

    Returns
    -------
    Progress
        Usable as an iterable (when ``iterable`` is given) or as a context manager.
    """
    return Progress(iterable, total=total, desc=desc, unit=unit, disable=disable)


__all__ = [
    "INDENT",
    "PROGRESS_INTERVAL_VARIABLE",
    "Column",
    "Progress",
    "Table",
    "emit",
    "format_duration",
    "format_path",
    "format_value",
    "info",
    "program_footer",
    "program_header",
    "progress",
    "run_footer",
    "run_header",
    "style",
    "summary",
    "warning",
]
