"""Discover normal-point files and dispatch native, CRD and MINI imports by content.

Native files are ``normalPoint`` artifacts. CRD files are recognized by the suffixes ``.npt``, ``.crd`` and
``.frd`` and MINI files by ``.dat``, ``.mini``, ``.dat.txt`` and ``.mini.txt``, each optionally gzipped
(``.npt.gz``, ``.crd.gz``, ``.frd.gz``, ``.dat.gz``, ``.mini.gz``). Suffixes are matched case-insensitively.
The format actually read is detected from file content; a recognized suffix only adds an expectation that
the content must meet.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from ..archive import open_ascii_source
from ..normal_points import is_normal_point_artifact, read_normal_points

SUPPORTED_MINI_SUFFIXES = (
    ".dat",
    ".mini",
    ".dat.txt",
    ".mini.txt",
    ".dat.gz",
    ".mini.gz",
)
SUPPORTED_CRD_SUFFIXES = (
    ".npt",
    ".crd",
    ".frd",
    ".npt.gz",
    ".crd.gz",
    ".frd.gz",
)


def _has_suffix(path: Path, suffixes: tuple[str, ...]) -> bool:
    name = path.name.lower()
    return any(name.endswith(suffix) for suffix in suffixes)


def is_mini_file(path: Path) -> bool:
    """Return whether a file name has a MINI suffix (``SUPPORTED_MINI_SUFFIXES``, case-insensitive)."""
    return _has_suffix(path, SUPPORTED_MINI_SUFFIXES)


def is_crd_file(path: Path) -> bool:
    """Return whether a file name has a CRD suffix (``SUPPORTED_CRD_SUFFIXES``, case-insensitive)."""
    return _has_suffix(path, SUPPORTED_CRD_SUFFIXES)


def iter_input_files(path: Path) -> Iterable[Path]:
    """Yield native ``normalPoint`` files from a file or directory.

    Parameters
    ----------
    path : Path
        A native normal-point file, or a directory searched recursively; ``~`` is expanded.

    Yields
    ------
    Path
        Native normal-point files, in sorted path order for a directory.

    Raises
    ------
    ValueError
        If ``path`` is a file that is not a native normal-point file.
    FileNotFoundError
        If ``path`` does not exist.
    """
    source = path.expanduser()
    if source.is_file():
        if not is_normal_point_artifact(source):
            raise ValueError(f"Input is not a native LunarOps normal-point file: {source}")
        yield source
        return
    if source.is_dir():
        for child in sorted(source.rglob("*")):
            if child.is_file() and is_normal_point_artifact(child):
                yield child
        return
    raise FileNotFoundError(f"Input path does not exist: {source}")


def iter_source_files(path: Path) -> Iterable[Path]:
    """Yield normal-point source files (native, CRD or MINI) from a file or directory.

    Parameters
    ----------
    path : Path
        A source file, yielded without a format check, or a directory searched recursively for native
        normal-point files and files with CRD or MINI suffixes; ``~`` is expanded.

    Yields
    ------
    Path
        Source files, in sorted path order for a directory.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    """
    source = path.expanduser()
    if source.is_file():
        yield source
        return
    if source.is_dir():
        for child in sorted(source.rglob("*")):
            if child.is_file() and (is_normal_point_artifact(child) or is_crd_file(child) or is_mini_file(child)):
                yield child
        return
    raise FileNotFoundError(f"Input path does not exist: {source}")


def _resolve(inputs, iterator) -> list[Path]:
    values = [inputs] if isinstance(inputs, (str, Path)) else list(inputs)
    seen: dict[Path, None] = {}
    for value in values:
        for path in iterator(Path(str(value))):
            seen.setdefault(path.resolve(), None)
    return sorted(seen)


def resolve_normal_point_inputs(inputs) -> list[Path]:
    """Return the native ``normalPoint`` files named by one or more paths.

    Parameters
    ----------
    inputs : str, Path or Iterable of str or Path
        Files or directories, expanded with ``iter_input_files``.

    Returns
    -------
    list of Path
        Resolved absolute paths, de-duplicated and sorted.

    Raises
    ------
    ValueError
        If a named file is not a native normal-point file.
    FileNotFoundError
        If a path does not exist.
    """
    return _resolve(inputs, iter_input_files)


def resolve_normal_point_sources(inputs) -> list[Path]:
    """Return the normal-point source files (native, CRD or MINI) named by one or more paths.

    Parameters
    ----------
    inputs : str, Path or Iterable of str or Path
        Files or directories, expanded with ``iter_source_files``.

    Returns
    -------
    list of Path
        Resolved absolute paths, de-duplicated and sorted.

    Raises
    ------
    FileNotFoundError
        If a path does not exist.
    """
    return _resolve(inputs, iter_source_files)


def detect_normal_point_format(path: str | Path) -> str:
    """Return the normal-point format of a file, detected from its first nonblank line.

    The line identifies a native file (``lunarops normalPoint``), a CRD file (starting ``H1 CRD``,
    case-insensitive) or a MINI file (``looks_like_mini_line``). A file with a CRD or MINI suffix must
    contain that format.

    Parameters
    ----------
    path : str or Path
        Source file, read as ASCII and optionally gzipped; ``~`` is expanded.

    Returns
    -------
    {"native", "crd", "mini"}
        Detected format.

    Raises
    ------
    ValueError
        If the file cannot be read as ASCII, its content is empty or unrecognized, or the content contradicts
        a recognized suffix.
    """
    from .mini import looks_like_mini_line

    source = Path(path).expanduser()
    try:
        with open_ascii_source(source) as stream:
            first = next((line.strip() for line in stream if line.strip()), "")
        if first.split()[:2] == ["lunarops", "normalPoint"]:
            detected = "native"
        elif first.upper().split()[:2] == ["H1", "CRD"]:
            detected = "crd"
        elif looks_like_mini_line(first):
            detected = "mini"
        else:
            raise ValueError("empty or unrecognized normal-point content")
        expected = "crd" if is_crd_file(source) else "mini" if is_mini_file(source) else None
        if expected is not None and expected != detected:
            raise ValueError(f"suffix expects {expected}, content declares {detected}")
        return detected
    except (OSError, UnicodeError, ValueError, EOFError) as exc:
        raise ValueError(f"{source}: format detection, line 1: {exc}") from exc


def read_normal_point_source(path: str | Path, *, crd_options=None, mini_temperature_unit="deciCelsius"):
    """Read a native, CRD or MINI normal-point file into a dataset.

    The format is chosen by ``detect_normal_point_format`` and the file is read with ``read_normal_points``,
    ``parse_crd_file`` or ``parse_mini_file``.

    Parameters
    ----------
    path : str or Path
        Source file; ``~`` is expanded.
    crd_options : dict or None, optional
        Keyword arguments for ``parse_crd_file`` (``uncertainty_policy``, ``uncertainty_two_way_s``,
        ``meteorology_max_age_s``). CRD import requires ``uncertainty_policy``.
    mini_temperature_unit : {"deciCelsius", "deciKelvin"}, optional
        Unit of the MINI temperature field.

    Returns
    -------
    NptDataset
        Normal points with UTC transmit epochs.

    Raises
    ------
    FileNotFoundError
        If ``path`` is not an existing file.
    ValueError
        If format detection or the import fails; the message names the file and the detected format.
    """
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"Normal-point source file not found: {source}")
    from .crd import parse_crd_file
    from .mini import parse_mini_file

    detected = detect_normal_point_format(source)
    try:
        if detected == "native":
            return read_normal_points(source)
        if detected == "crd":
            return parse_crd_file(source, **(crd_options or {}))
        return parse_mini_file(source, temperature_unit=mini_temperature_unit)
    except (ValueError, OSError, UnicodeError, EOFError) as exc:
        raise ValueError(f"{source}: {detected} import failed: {exc}") from exc


__all__ = [
    "SUPPORTED_CRD_SUFFIXES",
    "SUPPORTED_MINI_SUFFIXES",
    "detect_normal_point_format",
    "is_crd_file",
    "is_mini_file",
    "is_normal_point_artifact",
    "iter_input_files",
    "iter_source_files",
    "read_normal_point_source",
    "resolve_normal_point_inputs",
    "resolve_normal_point_sources",
]
