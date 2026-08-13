"""Native normal-point discovery plus explicit CRD/MINI import dispatch."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from ..normal_points import is_normal_point_artifact, read_normal_points as _read_normal_points

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


def read_normal_points(path: str | Path):
    """Read one native LunarOps normal-point text artifact."""
    return _read_normal_points(path)


def read_normal_point_source(path: str | Path):
    """Import one CRD/MINI/native source into the canonical in-memory model."""
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"Normal-point source file not found: {source}")
    from .crd import looks_like_crd_file, parse_crd_file
    from .mini import looks_like_mini_file, parse_mini_file

    if is_normal_point_artifact(source):
        return _read_normal_points(source)
    if is_crd_file(source) or looks_like_crd_file(source):
        return parse_crd_file(source)
    if is_mini_file(source) or looks_like_mini_file(source):
        return parse_mini_file(source)
    raise ValueError(f"Unsupported normal-point source format: {source}")


__all__ = [
    "SUPPORTED_CRD_SUFFIXES",
    "SUPPORTED_MINI_SUFFIXES",
    "is_crd_file",
    "is_mini_file",
    "is_normal_point_artifact",
    "iter_input_files",
    "iter_source_files",
    "read_normal_point_source",
    "read_normal_points",
    "resolve_normal_point_inputs",
    "resolve_normal_point_sources",
]
