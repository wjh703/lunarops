"""Summarize controlled DE430 trajectory/attitude driver replacements."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from lunarops.classes.ephemerides import CalcephEphemeris, TabulatedDe430Driver
from lunarops.classes.time import Epoch, TimeScale
from lunarops.fileio.numeric_table import read_numeric_table

ROOT = Path("output/de430_hybrid")
VARIANTS = ("spk_bpc", "julia_bpc", "spk_julia", "julia_julia", "spk_bpc_de430_earth")
J2000 = Epoch(2_451_545.0, 0.0, TimeScale.TDB)


def load_orbit(name: str):
    _, columns, values, _ = read_numeric_table(ROOT / name / "lunar_orbit.dat", "lunarOrbit")
    return columns, values


def columns(values, names, available):
    return values[:, [available.index(name) for name in names]]


def norms(state):
    return np.linalg.norm(state[:, :3], axis=1), np.linalg.norm(state[:, 3:], axis=1)


def rotation_angle(matrix):
    return np.arccos(np.clip((np.trace(matrix) - 1.0) / 2.0, -1.0, 1.0))


def main() -> None:
    base_columns, base_values = load_orbit("spk_bpc")
    state_names = tuple(f"relative_{name}" for name in ("x_m", "y_m", "z_m", "vx_mps", "vy_mps", "vz_mps"))
    base_state = columns(base_values, state_names, base_columns)
    report: dict[str, Any] = {"variants": {}, "attitude": {}}
    rows: list[tuple[str, float, float, float, float, float]] = []
    for name in VARIANTS:
        available, values = load_orbit(name)
        state = columns(values, state_names, available)
        difference = state - base_state
        position, velocity = norms(difference)
        spk_rtn = columns(
            values,
            tuple(f"reference_difference_rtn_{axis}_m" for axis in ("r", "t", "n")),
            available,
        )
        metrics = {
            "final_delta_from_spk_bpc_position_m": float(position[-1]),
            "maximum_delta_from_spk_bpc_position_m": float(np.max(position)),
            "final_delta_from_spk_bpc_velocity_mps": float(velocity[-1]),
            "final_position_error_from_de430_spk_m": float(np.linalg.norm(spk_rtn[-1])),
            "maximum_position_error_from_de430_spk_m": float(np.max(np.linalg.norm(spk_rtn, axis=1))),
        }
        report["variants"][name] = metrics
        rows.append(
            (
                name,
                metrics["final_delta_from_spk_bpc_position_m"],
                metrics["maximum_delta_from_spk_bpc_position_m"],
                metrics["final_delta_from_spk_bpc_velocity_mps"],
                metrics["final_position_error_from_de430_spk_m"],
                metrics["maximum_position_error_from_de430_spk_m"],
            )
        )

    base = CalcephEphemeris(
        "../data/kernels/de430",
        lunar_relativistic_scale_convention="alreadyScaled",
        longitude_libration_correction="none",
    )
    tabulated = TabulatedDe430Driver(
        base,
        ROOT / "drivers/de430_julia",
        replace_lunar_orientation=True,
    )
    attitude_angles: list[float] = []
    earth_angles: list[float] = []
    for hour in range(169):
        epoch = J2000.shifted(hour * 3600.0)
        bpc_inertial_to_body = base.lunar_orientation.pa_to_lcrs_matrix(epoch).T
        julia_inertial_to_body = tabulated.lunar_orientation.pa_to_lcrs_matrix(epoch).T
        attitude_angles.append(rotation_angle(julia_inertial_to_body @ bpc_inertial_to_body.T))
        from lunarops.classes.dynamics.earth_gravity_orientation import de440_earth_fixed2inertial_matrix

        earth_angles.append(
            rotation_angle(
                tabulated.earth_fixed_to_inertial_matrix(epoch)
                @ de440_earth_fixed2inertial_matrix(epoch).T
            )
        )
    attitude_angle_array = np.asarray(attitude_angles)
    earth_angle_array = np.asarray(earth_angles)
    report["attitude"] = {
        "julia_vs_bpc_initial_rad": float(attitude_angle_array[0]),
        "julia_vs_bpc_final_rad": float(attitude_angle_array[-1]),
        "julia_vs_bpc_maximum_rad": float(np.max(attitude_angle_array)),
        "de430_vs_de440_earth_initial_rad": float(earth_angle_array[0]),
        "de430_vs_de440_earth_final_rad": float(earth_angle_array[-1]),
        "de430_vs_de440_earth_maximum_rad": float(np.max(earth_angle_array)),
    }
    tabulated.close()

    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "ablation_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="ascii")
    header = (
        "variant\tfinal_delta_from_spk_bpc_position_m\tmaximum_delta_from_spk_bpc_position_m\t"
        "final_delta_from_spk_bpc_velocity_mps\tfinal_position_error_from_de430_spk_m\t"
        "maximum_position_error_from_de430_spk_m"
    )
    with (ROOT / "ablation_summary.tsv").open("w", encoding="ascii") as stream:
        print(header, file=stream)
        for row in rows:
            print(row[0] + "\t" + "\t".join(f"{value:.17e}" for value in row[1:]), file=stream)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
