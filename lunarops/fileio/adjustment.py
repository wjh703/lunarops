"""Typed nonlinear-adjustment report and restart-state artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np

from .structured_text import read_structured_text, write_structured_text


def write_adjustment_report(path: str | Path, payload: Mapping[str, object]) -> Path:
    return write_structured_text(path, "adjustmentReport", payload)


def read_adjustment_report(path: str | Path) -> dict[str, object]:
    return read_structured_text(path, "adjustmentReport")


def _validate_adjustment_state(payload: Mapping[str, object]) -> None:
    obsolete = {"scales", "robustFactors"} & set(payload)
    if obsolete:
        raise ValueError(f"Adjustment state contains obsolete field(s): {sorted(obsolete)}")
    required = {
        "fingerprint",
        "parametrization",
        "reflectorPositions",
        "sigmaFactors",
        "weightFactors",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Adjustment state is missing field(s): {sorted(missing)}")
    fingerprint = payload["fingerprint"]
    if (
        not isinstance(fingerprint, str)
        or len(fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in fingerprint)
    ):
        raise ValueError("Adjustment state fingerprint must be a lowercase SHA-256 digest.")
    for key in ("parametrization", "reflectorPositions", "sigmaFactors", "weightFactors"):
        if not isinstance(payload[key], Mapping):
            raise TypeError(f"Adjustment state {key} must be a mapping.")
    reflector_positions = cast(Mapping[object, object], payload["reflectorPositions"])
    sigma_factors_payload = cast(Mapping[object, object], payload["sigmaFactors"])
    weight_factors_payload = cast(Mapping[object, object], payload["weightFactors"])
    for reflector_key, values in reflector_positions.items():
        position = np.asarray(cast(Any, values), dtype=np.float64)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError(f"Adjustment state reflector position {reflector_key!r} is invalid.")
    sigma_factors = np.asarray(list(sigma_factors_payload.values()), dtype=np.float64)
    weight_factors = np.asarray(list(weight_factors_payload.values()), dtype=np.float64)
    if not np.all(np.isfinite(sigma_factors)) or np.any(sigma_factors <= 0.0):
        raise ValueError("Adjustment-state sigma factors must be positive and finite.")
    if not np.all(np.isfinite(weight_factors)) or np.any(
        (weight_factors < 0.0) | (weight_factors > 1.0)
    ):
        raise ValueError("Adjustment-state weight factors must be finite and in [0, 1].")


def write_adjustment_state(path: str | Path, payload: Mapping[str, object]) -> Path:
    _validate_adjustment_state(payload)
    return write_structured_text(path, "adjustmentState", payload)


def read_adjustment_state(path: str | Path) -> dict[str, object]:
    payload = read_structured_text(path, "adjustmentState")
    _validate_adjustment_state(payload)
    return payload


__all__ = [
    "read_adjustment_report",
    "read_adjustment_state",
    "write_adjustment_report",
    "write_adjustment_state",
]
