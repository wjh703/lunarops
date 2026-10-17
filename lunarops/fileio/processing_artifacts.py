"""Typed report and restart-state artifacts for LLR processing."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np

from .yaml_artifact import read_structured_text, write_structured_text

FORMAT_VERSION = 1

def _validate_processing_state(payload: Mapping[str, object]) -> None:
    required = {
        "fingerprint",
        "parametrization",
        "reflectorPositions",
        "sigmaFactors",
        "weightFactors",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Processing state is missing field(s): {sorted(missing)}")
    fingerprint = payload["fingerprint"]
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError("Processing state fingerprint must be a non-empty string.")
    for name in ("parametrization", "reflectorPositions", "sigmaFactors", "weightFactors"):
        if not isinstance(payload[name], Mapping):
            raise TypeError(f"Processing state {name} must be a mapping.")
    for field, positive in (("sigmaFactors", True), ("weightFactors", False)):
        for key, value in cast(Mapping[str, Any], payload[field]).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                raise ValueError(f"Processing state {field}[{key!r}] must be finite and real.")
            if (positive and value <= 0) or (not positive and not 0 <= value <= 1):
                raise ValueError(f"Processing state {field}[{key!r}] is outside its valid range.")
    for key, value in cast(Mapping[Any, Any], payload["reflectorPositions"]).items():
        array = np.asarray(value, dtype=float)
        if array.shape != (3,) or not np.all(np.isfinite(array)):
            raise ValueError(f"Processing state reflector position {key!r} must be a finite vector3.")


def write_processing_state(path: str | Path, payload: Mapping[str, object]) -> Path:
    return write_structured_text(path, "processingState", payload, version=FORMAT_VERSION)


def read_processing_state(path: str | Path) -> dict[str, object]:
    payload = read_structured_text(path, "processingState", version=FORMAT_VERSION)
    _validate_processing_state(payload)
    return payload


__all__ = [
    "read_processing_state",
    "write_processing_state",
]
