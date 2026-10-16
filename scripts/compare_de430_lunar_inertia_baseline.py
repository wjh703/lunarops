"""Compare lunarops delayed lunar inertia with PlanetaryEphemeris DE430."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from lunarops.classes.dynamics.inertia import LunarDegree2GravityModel, LunarDegree2GravityParameters
from lunarops.classes.time import Epoch, TimeScale

AU_M = 149_597_870_700.0
DAY_S = 86_400.0
J2000 = Epoch(2_451_545.0, 0.0, TimeScale.TDB)


def _matrix_rows(filename: Path, key: str) -> np.ndarray:
    table = np.genfromtxt(filename, delimiter="\t", names=True, dtype=None, encoding="ascii")
    row = table[table["term"] == key]
    return np.asarray(list(row[0])[1:], dtype=float).reshape(3, 3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prefix", type=Path)
    parser.add_argument("--atol-inertia", type=float, default=2e-15)
    parser.add_argument("--atol-omega", type=float, default=2e-14)
    args = parser.parse_args()

    p = np.genfromtxt(Path(f"{args.prefix}_parameters.tsv"), delimiter="\t", names=True)
    parameters = LunarDegree2GravityParameters(
        j2_undistorted=float(p["j2"]),
        beta=float(p["beta"]),
        gamma=float(p["gamma"]),
        love_k2=float(p["love_k2"]),
        tidal_delay_days=float(p["delay_days"]),
        radius_m=float(p["radius_au"]) * AU_M,
        mean_motion_rad_s=float(p["mean_motion_rad_day"]) / DAY_S,
        attitude_derivative_step_s=float(p["difference_step_s"]),
    )

    states_table = np.genfromtxt(
        Path(f"{args.prefix}_delayed_states.tsv"), delimiter="\t", names=True, dtype=None, encoding="ascii"
    )
    scale = np.asarray([AU_M, AU_M, AU_M, AU_M / DAY_S, AU_M / DAY_S, AU_M / DAY_S])
    states = np.asarray([list(row)[1:] for row in states_table], dtype=float) * scale

    attitude_table = np.genfromtxt(
        Path(f"{args.prefix}_attitudes.tsv"), delimiter="\t", names=True, dtype=float
    )
    attitudes = {
        float(row["offset_seconds"]): np.asarray(list(row)[1:], dtype=float).reshape(3, 3)
        for row in attitude_table
    }
    delayed_epoch = J2000.shifted(-parameters.tidal_delay_days * DAY_S)

    def relative_position(epoch):
        return states[0, :3] - states[1, :3]

    def attitude(epoch):
        offset = delayed_epoch.seconds_until(epoch)
        nearest = min(attitudes, key=lambda candidate: abs(candidate - offset))
        if abs(nearest - offset) > 1e-5:
            raise ValueError(f"Unexpected attitude offset: {offset}")
        return attitudes[nearest]

    model = LunarDegree2GravityModel(
        earth_gravitational_parameter_m3_s2=8.8876924451256342e-10 * AU_M**3 / DAY_S**2,
        moon_gravitational_parameter_m3_s2=1.0931894507423740e-11 * AU_M**3 / DAY_S**2,
        earth_minus_moon_position_provider=relative_position,
        attitude_provider=attitude,
        parameters=parameters,
    )
    actual = model.evaluate(J2000)
    failures = []
    for name, actual_key in (
        ("undistorted", None),
        ("tidal", "tidal_inertia"),
        ("rotational", "rotational_inertia"),
        ("total", "normalized_inertia"),
    ):
        expected = _matrix_rows(Path(f"{args.prefix}_inertia.tsv"), name)
        value = parameters.undistorted_inertia() if actual_key is None else getattr(actual, actual_key)
        difference = np.max(np.abs(value - expected))
        print(f"{name:12s} max_abs={difference:.17e}")
        if difference > args.atol_inertia:
            failures.append(name)

    omega_table = np.genfromtxt(
        Path(f"{args.prefix}_angular_velocity.tsv"), delimiter="\t", names=True
    )
    omega_names = omega_table.dtype.names
    if omega_names is None:
        raise ValueError("Angular-velocity table has no named columns")
    expected_omega = np.array([float(omega_table[name]) for name in omega_names], dtype=float) / DAY_S
    omega_difference = actual.angular_velocity_rad_s - expected_omega
    print(
        f"angular_velocity delta={omega_difference} "
        f"norm={np.linalg.norm(omega_difference):.17e} rad/s"
    )
    if np.linalg.norm(omega_difference) > args.atol_omega:
        failures.append("angular velocity")
    if failures:
        raise SystemExit("lunar inertia comparison failed: " + ", ".join(failures))
    print("PASS")


if __name__ == "__main__":
    main()
