"""Read and write processing reports and lunar-orbit metadata.

Both are structured text artifacts (a ``lunarops <artifactType>`` header followed by a YAML mapping) with the
artifact types ``processingReport`` and ``lunarOrbitMetadata``. The readers check the header and that the
payload is a mapping; they do not check its keys.
"""

from __future__ import annotations

from pathlib import Path

from .yaml_artifact import read_structured_text, write_structured_text


def write_processing_report(path, payload) -> Path:
    """Write a ``processingReport`` structured text artifact.

    Parameters
    ----------
    path : str or Path
        Target path ending in ``.txt`` or ``.txt.gz``.
    payload : Mapping of str to object
        Report content.

    Returns
    -------
    Path
        The written path.
    """
    return write_structured_text(path, "processingReport", payload)


def read_processing_report(path):
    """Read a ``processingReport`` structured text artifact.

    Parameters
    ----------
    path : str or Path
        Path ending in ``.txt`` or ``.txt.gz``.

    Returns
    -------
    dict
        Report content.
    """
    return read_structured_text(path, "processingReport")


def write_lunar_orbit_metadata(path, payload) -> Path:
    """Write a ``lunarOrbitMetadata`` structured text artifact.

    Parameters
    ----------
    path : str or Path
        Target path ending in ``.txt`` or ``.txt.gz``.
    payload : Mapping of str to object
        Orbit metadata.

    Returns
    -------
    Path
        The written path.
    """
    return write_structured_text(path, "lunarOrbitMetadata", payload)


def read_lunar_orbit_metadata(path):
    """Read a ``lunarOrbitMetadata`` structured text artifact.

    Parameters
    ----------
    path : str or Path
        Path ending in ``.txt`` or ``.txt.gz``.

    Returns
    -------
    dict
        Orbit metadata.
    """
    return read_structured_text(path, "lunarOrbitMetadata")
