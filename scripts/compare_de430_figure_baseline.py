"""Compare lunarops figure forces with native PlanetaryEphemeris DE430 terms."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyshtools

from lunarops.classes.dynamics.context import DynamicsEpochData, ForceEvaluationContext
from lunarops.classes.dynamics.force_models import FigureForce, SolarJ2Force
from lunarops.classes.dynamics.gravity import GravityField, make_gravity_coefficients
from lunarops.classes.time import Epoch, TimeScale

AU_M = 149_597_870_700.0
DAY_S = 86_400.0
ACCELERATION_SCALE = AU_M / DAY_S**2
BODY_NAMES = (
    "SUN", "MERCURY", "VENUS", "EARTH", "MOON", "MARS", "JUPITER", "SATURN", "URANUS", "NEPTUNE", "PLUTO"
)
PARTNERS = {
    "SUN": BODY_NAMES[1:],
    "EARTH": ("SUN", "MERCURY", "VENUS", "MOON", "MARS", "JUPITER"),
    "MOON": ("SUN", "MERCURY", "VENUS", "EARTH", "MARS", "JUPITER"),
}
RADII_M = {"SUN": 696_000_000.0, "EARTH": 6_378_136.3, "MOON": 1_738_000.0}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("prefix", type=Path)
    parser.add_argument("initial_conditions", type=Path)
    parser.add_argument("--atol", type=float, default=5e-18)
    return parser.parse_args()


def _initial_state(filename: Path):
    rows = [line.split() for line in filename.read_text(encoding="ascii").splitlines()[1:-3]]
    ids = np.asarray([row[0] for row in rows])
    values = np.asarray([[float(item) for item in row[1:8]] for row in rows])
    return ids, values[:, 0] * AU_M**3 / DAY_S**2, values[:, 1:4] * AU_M, values[:, 4:7] * AU_M / DAY_S


def _rotation_matrices(filename: Path):
    table = np.genfromtxt(filename, delimiter="\t", names=True, dtype=None, encoding="ascii")
    return {str(row["body"]): np.asarray(list(row)[1:], dtype=float).reshape(3, 3) for row in table}


def _fields(filename: Path, gm_by_name):
    rows = np.genfromtxt(filename, delimiter="\t", names=True, dtype=None, encoding="ascii")
    fields = {}
    for body, radius_m in RADII_M.items():
        selected = rows[rows["body"] == body]
        degree = int(np.max(selected["degree"]))
        coefficients = np.zeros((2, degree + 1, degree + 1))
        for row in selected:
            n, m = int(row["degree"]), int(row["order"])
            coefficients[:, n, m] = row["cosine_unnorm"], row["sine_unnorm"]
        normalized = pyshtools.SHCoeffs.from_array(coefficients, normalization="unnorm", csphase=1).convert(
            normalization="4pi", csphase=1
        )
        gravity_coefficients = make_gravity_coefficients(
            normalized.coeffs[0], normalized.coeffs[1],
            gm_m3_s2=gm_by_name[body], radius_m=radius_m, name=body,
        )
        fields[body] = GravityField(gravity_coefficients)
    return fields


def main() -> None:
    args = _arguments()
    ids, gm, positions, velocities = _initial_state(args.initial_conditions)
    names = BODY_NAMES + tuple(f"ASTEROID_{body_id}" for body_id in ids[len(BODY_NAMES):])
    gm_by_name = dict(zip(names, gm, strict=True))
    rotation_matrices = _rotation_matrices(Path(f"{args.prefix}_frames.tsv"))
    fields = _fields(Path(f"{args.prefix}_harmonics.tsv"), gm_by_name)
    epoch_data = DynamicsEpochData(
        epoch_tdb=Epoch(2_451_545.0, 0.0, TimeScale.TDB),
        body_names=names,
        positions_m=positions,
        velocities_mps=velocities,
        earth_fixed2inertial_matrix=rotation_matrices["EARTH"].T,
        moon_fixed2inertial_matrix=rotation_matrices["MOON"].T,
        ephemeris_earth_acceleration_mps2=np.zeros(3),
    )
    inputs = ForceEvaluationContext(names, gm)
    inputs.load_epoch_data(epoch_data)
    expected_table = np.genfromtxt(
        Path(f"{args.prefix}_accelerations.tsv"), delimiter="\t", names=True, dtype=None, encoding="ascii"
    )

    failed = False
    total_actual = np.zeros_like(positions)
    total_expected = np.zeros_like(positions)
    for body in ("SUN", "EARTH", "MOON"):
        if body == "SUN":
            solar_model = SolarJ2Force(fields[body], rotation_matrices[body])
            targets = np.asarray([name in PARTNERS[body] for name in names])
            actual = solar_model.acceleration(inputs, target_body_mask=targets)
            # Include solar recoil for the full-system baseline comparison.
            sun = names.index("SUN")
            actual[sun] = -np.sum((gm[targets] / gm[sun])[:, None] * actual[targets], axis=0)
        else:
            model = FigureForce({body: fields[body]}, {body: PARTNERS[body]}, name=f"figure_{body.lower()}")
            actual = model.acceleration(inputs, target_body_mask=np.ones(len(names), dtype=bool))
        expected = np.column_stack([expected_table[f"{body.lower()}_{axis}"] for axis in "xyz"]) * ACCELERATION_SCALE
        total_actual += actual
        total_expected += expected
        difference = np.linalg.norm(actual - expected, axis=1)
        scale = np.maximum(np.linalg.norm(expected, axis=1), np.finfo(float).tiny)
        active = np.linalg.norm(expected, axis=1) > 0
        relative = difference[active] / scale[active]
        worst = int(np.argmax(difference))
        max_relative = float(np.max(relative))
        print(
            f"{body:5s} max_abs={difference[worst]:.17e} body_id={ids[worst]} "
            f"max_rel_active={max_relative:.17e}"
        )
        earth, moon = 3, 4
        relative_difference = (actual[moon] - actual[earth]) - (expected[moon] - expected[earth])
        expected_relative = expected[moon] - expected[earth]
        relative_error = np.linalg.norm(relative_difference) / np.linalg.norm(expected_relative)
        print(
            f"{body:5s} delta_a_rel={relative_difference} norm={np.linalg.norm(relative_difference):.17e} m/s^2 "
            f"signal_rel={np.linalg.norm(expected_relative):.17e} ratio={relative_error:.17e}"
        )
        # Julia terms are isolated by subtracting two full Float64 accelerations
        # near 1e-3 m/s^2, which imposes an approximately 1e-18 m/s^2 floor.
        failed |= float(np.max(difference)) > args.atol
    total_difference = total_actual - total_expected
    total_norms = np.linalg.norm(total_difference, axis=1)
    worst = int(np.argmax(total_norms))
    earth, moon = 3, 4
    relative_difference = total_difference[moon] - total_difference[earth]
    expected_relative = total_expected[moon] - total_expected[earth]
    print(f"TOTAL max_abs={total_norms[worst]:.17e} body_id={ids[worst]}")
    print(
        f"TOTAL delta_a_rel={relative_difference} norm={np.linalg.norm(relative_difference):.17e} m/s^2 "
        f"signal_rel={np.linalg.norm(expected_relative):.17e} "
        f"ratio={np.linalg.norm(relative_difference) / np.linalg.norm(expected_relative):.17e}"
    )
    failed |= float(np.max(total_norms)) > args.atol
    if failed:
        raise SystemExit("figure comparison failed")
    print("PASS")


if __name__ == "__main__":
    main()
