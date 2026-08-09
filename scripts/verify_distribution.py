#!/usr/bin/env python3
"""Verify the compiled extension sources and reject legacy Fortran payloads."""

from __future__ import annotations

import argparse
import tarfile
import zipfile
from pathlib import Path


PACKAGE_REQUIRED = {
    "lunarops/_iers2010.py",
    "lunarops/_iers2010.pyi",
    "lunarops/_iers2010_core.pyx",
    "lunarops/_iers2010_tables.pxi",
    "lunarops/_normal_equations_core.pyx",
    "lunarops/_normal_equations_core.pyi",
    "lunarops/_external/iers2010/LICENSE",
}
FORTRAN_SUFFIXES = (".f", ".for", ".f77", ".f90", ".f95", ".f03", ".f08", ".pyf")


def _fortran_members(names: set[str]) -> list[str]:
    return sorted(name for name in names if name.lower().endswith(FORTRAN_SUFFIXES))


def _check_sdist(path: Path) -> None:
    with tarfile.open(path, "r:gz") as archive:
        names = set(archive.getnames())
    roots = {name.split("/", 1)[0] for name in names if "/" in name}
    if len(roots) != 1:
        raise SystemExit(f"{path}: expected one sdist root, found {sorted(roots)}")
    root = next(iter(roots))
    required = {f"{root}/{name}" for name in PACKAGE_REQUIRED}
    missing = sorted(required - names)
    if missing:
        raise SystemExit(f"{path}: missing {', '.join(missing)}")
    forbidden = _fortran_members(names)
    if forbidden:
        raise SystemExit(f"{path}: contains forbidden Fortran/f2py files: {', '.join(forbidden)}")


def _check_wheel(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
    missing = sorted(PACKAGE_REQUIRED - names)
    if missing:
        raise SystemExit(f"{path}: missing {', '.join(missing)}")
    iers_extensions = sorted(
        name
        for name in names
        if name.startswith("lunarops/_iers2010_core.") and name.endswith((".so", ".pyd"))
    )
    if not iers_extensions:
        raise SystemExit(f"{path}: missing compiled lunarops/_iers2010_core extension")
    normal_extensions = sorted(
        name
        for name in names
        if name.startswith("lunarops/_normal_equations_core.") and name.endswith((".so", ".pyd"))
    )
    if not normal_extensions:
        raise SystemExit(f"{path}: missing compiled lunarops/_normal_equations_core extension")
    forbidden = _fortran_members(names)
    if forbidden:
        raise SystemExit(f"{path}: contains forbidden Fortran/f2py files: {', '.join(forbidden)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archives", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.archives:
        if path.name.endswith(".tar.gz"):
            _check_sdist(path)
        elif path.name.endswith(".whl"):
            _check_wheel(path)
        else:
            raise SystemExit(f"unsupported archive: {path}")
        print(f"verified {path}")


if __name__ == "__main__":
    main()
