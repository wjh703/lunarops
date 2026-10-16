"""Compare lunarops Newtonian/EIH terms with a PlanetaryEphemeris DE430 snapshot."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from lunarops.classes.dynamics.forces import evaluate_eih_correction, evaluate_newtonian_point_mass_system

AU_M = 149_597_870_700.0
DAY_S = 86_400.0
ACCELERATION_SCALE = AU_M / DAY_S**2
POTENTIAL_SCALE = (AU_M / DAY_S) ** 2


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--rtol", type=float, default=5e-14)
    parser.add_argument("--atol", type=float, default=1e-24)
    return parser.parse_args()


def _norm_rows(values: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.asarray(values, dtype=float), axis=1)


def _report(name: str, expected: np.ndarray, actual: np.ndarray, ids: np.ndarray) -> tuple[float, float]:
    difference = actual - expected
    absolute = _norm_rows(difference)
    scale = np.maximum(_norm_rows(expected), np.finfo(float).tiny)
    relative = absolute / scale
    worst_absolute = int(np.argmax(absolute))
    worst_relative = int(np.argmax(relative))
    print(
        f"{name:12s} max_abs={absolute[worst_absolute]:.17e} "
        f"body={ids[worst_absolute]} max_rel={relative[worst_relative]:.17e} body={ids[worst_relative]}"
    )
    return float(np.max(absolute)), float(np.max(relative))


def main() -> None:
    args = _arguments()
    table = np.genfromtxt(args.snapshot, delimiter="\t", names=True, dtype=None, encoding="ascii")
    ids = np.asarray(table["body_id"], dtype=str)
    gm = np.asarray(table["gm_au3_day2"], dtype=float) * AU_M**3 / DAY_S**2
    positions = np.column_stack([table[name] for name in ("x_au", "y_au", "z_au")]) * AU_M
    velocities = np.column_stack([table[name] for name in ("vx_au_day", "vy_au_day", "vz_au_day")]) * AU_M / DAY_S
    expected_newtonian = np.column_stack(
        [table[name] for name in ("newton_x_au_day2", "newton_y_au_day2", "newton_z_au_day2")]
    ) * ACCELERATION_SCALE
    expected_potential = np.asarray(table["potential_au2_day2"], dtype=float) * POTENTIAL_SCALE
    expected_eih = np.column_stack(
        [table[name] for name in ("eih_x_au_day2", "eih_y_au_day2", "eih_z_au_day2")]
    ) * ACCELERATION_SCALE

    point_mass = evaluate_newtonian_point_mass_system(positions, gm)
    actual_eih = evaluate_eih_correction(positions, velocities, gm, point_mass, np.arange(len(ids)))
    newtonian_abs, newtonian_rel = _report(
        "newtonian", expected_newtonian, point_mass.accelerations_mps2, ids
    )
    eih_abs, eih_rel = _report("eih", expected_eih, actual_eih, ids)
    potential_difference = np.abs(point_mass.potentials_m2_s2 - expected_potential)
    potential_relative = potential_difference / np.maximum(np.abs(expected_potential), np.finfo(float).tiny)
    print(
        f"potential    max_abs={np.max(potential_difference):.17e} "
        f"max_rel={np.max(potential_relative):.17e}"
    )

    earth = int(np.flatnonzero(ids == "399")[0])
    moon = int(np.flatnonzero(ids == "301")[0])
    for name, expected, actual in (
        ("newtonian", expected_newtonian, point_mass.accelerations_mps2),
        ("eih", expected_eih, actual_eih),
    ):
        expected_relative = expected[moon] - expected[earth]
        actual_relative = actual[moon] - actual[earth]
        print(f"{name:12s} earth={actual[earth]} moon={actual[moon]}")
        print(
            f"{name:12s} delta_a_rel={actual_relative - expected_relative} "
            f"norm={np.linalg.norm(actual_relative - expected_relative):.17e} m/s^2"
        )

    failures = []
    if newtonian_abs > args.atol and newtonian_rel > args.rtol:
        failures.append("Newtonian acceleration")
    if eih_abs > args.atol and eih_rel > args.rtol:
        failures.append("EIH correction")
    if np.max(potential_relative) > args.rtol:
        failures.append("Newtonian potential")
    if failures:
        raise SystemExit("comparison failed: " + ", ".join(failures))
    print("PASS")


if __name__ == "__main__":
    main()
