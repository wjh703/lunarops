"""Native normal-point discovery plus explicit CRD/MINI import dispatch."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

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
    return _has_suffix(path, SUPPORTED_MINI_SUFFIXES)


def is_crd_file(path: Path) -> bool:
    return _has_suffix(path, SUPPORTED_CRD_SUFFIXES)


def iter_input_files(path: Path) -> Iterable[Path]:
    """Yield native LunarOps normal-point files only."""
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
    """Yield native, CRD, or MINI sources for ``NormalPointsConvert``."""
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
    return _resolve(inputs, iter_input_files)


def resolve_normal_point_sources(inputs) -> list[Path]:
    return _resolve(inputs, iter_source_files)


def detect_normal_point_format(path: str | Path) -> str:
    """Determine one format from content; recognized suffixes impose expectations."""
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
    """Import one declared CRD/MINI/native source, without speculative fallback."""
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
    "is_crd_file",
    "is_mini_file",
    "is_normal_point_artifact",
    "iter_input_files",
    "iter_source_files",
    "read_normal_point_source",
    "detect_normal_point_format",
    "resolve_normal_point_inputs",
    "resolve_normal_point_sources",
]
