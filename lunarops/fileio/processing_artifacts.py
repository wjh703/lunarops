"""Read and write the ``processingState`` restart-state artifact for LLR processing.

The artifact is structured text: a ``lunarops processingState`` header followed by a YAML mapping.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from .yaml_artifact import read_structured_text, write_structured_text


def write_processing_state(path: str | Path, payload: Mapping[str, object]) -> Path:
    """Write a ``processingState`` structured text artifact.

    Parameters
    ----------
    path : str or Path
        Target path ending in ``.txt`` or ``.txt.gz``.
    payload : Mapping of str to object
        Restart state; ``read_processing_state`` requires the keys it lists.

    Returns
    -------
    Path
        The written path.
    """
    return write_structured_text(path, "processingState", payload)


def read_processing_state(path: str | Path) -> dict[str, object]:
    """Read a ``processingState`` structured text artifact.

    Parameters
    ----------
    path : str or Path
        Path ending in ``.txt`` or ``.txt.gz``.

    Returns
    -------
    dict
        Restart state containing at least ``fingerprint``, ``parametrization``, ``reflectorPositions``,
        ``sigmaFactors`` and ``weightFactors``.

    Raises
    ------
    ValueError
        If any of these keys is missing, or the file is not a valid ``processingState`` artifact.
    """
    payload = read_structured_text(path, "processingState")
    missing = {"fingerprint", "parametrization", "reflectorPositions", "sigmaFactors", "weightFactors"} - payload.keys()
    if missing:
        raise ValueError(f"Processing state is missing field(s): {sorted(missing)}")
    return payload


__all__ = [
    "read_processing_state",
    "write_processing_state",
]
