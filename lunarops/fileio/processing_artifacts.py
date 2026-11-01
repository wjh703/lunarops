"""Typed report and restart-state artifacts for LLR processing."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from .yaml_artifact import read_structured_text, write_structured_text


def write_processing_state(path: str | Path, payload: Mapping[str, object]) -> Path:
    return write_structured_text(path, "processingState", payload)


def read_processing_state(path: str | Path) -> dict[str, object]:
    payload = read_structured_text(path, "processingState")
    missing = {"fingerprint", "parametrization", "reflectorPositions", "sigmaFactors", "weightFactors"} - payload.keys()
    if missing:
        raise ValueError(f"Processing state is missing field(s): {sorted(missing)}")
    return payload


__all__ = [
    "read_processing_state",
    "write_processing_state",
]
