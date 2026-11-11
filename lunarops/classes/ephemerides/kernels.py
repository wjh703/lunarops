"""Listing and manifests of the SPICE kernels of an ephemeris directory."""

from __future__ import annotations

from pathlib import Path

from lunarops.fileio.archive import sha256_file


def kernel_paths(directory: Path) -> tuple[Path, ...]:
    """Return the kernels of an ephemeris directory in load order.

    Parameters
    ----------
    directory : Path
        Directory holding the kernels.

    Returns
    -------
    tuple of Path
        The ``*.bsp`` (position) kernels sorted by path, then the ``*.bpc`` (orientation) kernels sorted by path.
    """
    return (*sorted(directory.glob("*.bsp")), *sorted(directory.glob("*.bpc")))


def kernel_manifest(directory: Path) -> list[dict[str, object]]:
    """Return the manifest of the kernels of an ephemeris directory.

    Parameters
    ----------
    directory : Path
        Directory holding the kernels.

    Returns
    -------
    list of dict
        One entry per kernel in load order, with ``relativePath``, ``sha256`` and ``loadOrder``.
    """
    return [
        {"relativePath": str(path.relative_to(directory)), "sha256": sha256_file(path), "loadOrder": index}
        for index, path in enumerate(kernel_paths(directory))
    ]


__all__ = ["kernel_manifest", "kernel_paths"]
