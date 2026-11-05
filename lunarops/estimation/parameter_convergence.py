"""Convergence test for nonlinear parameter updates."""

from __future__ import annotations

from collections.abc import Mapping


def parameter_convergence(
    updates_m: Mapping[str, float],
    default_tolerance_m: float,
    tolerance_by_block_m: Mapping[str, float],
) -> tuple[bool, dict[str, float], dict[str, float]]:
    """Compare each parametrization block's update norm with its tolerance.

    A zero tolerance accepts only a zero update.

    Parameters
    ----------
    updates_m : Mapping[str, float]
        Update norm per block identifier, in the block's update unit (metres for position and bias blocks).
    default_tolerance_m : float
        Tolerance for blocks without their own entry.
    tolerance_by_block_m : Mapping[str, float]
        Tolerance per block identifier.

    Returns
    -------
    converged : bool
        Whether every update is within its tolerance.
    tolerances : dict[str, float]
        Tolerance used per block.
    ratios : dict[str, float]
        Update divided by its tolerance per block.
    """
    tolerances = {label: float(tolerance_by_block_m.get(label, default_tolerance_m)) for label in updates_m}
    ratios = {
        label: (0.0 if value == 0.0 else float("inf"))
        if tolerances[label] == 0.0
        else float(value) / tolerances[label]
        for label, value in updates_m.items()
    }
    return all(ratio <= 1.0 for ratio in ratios.values()), tolerances, ratios


__all__ = ["parameter_convergence"]
