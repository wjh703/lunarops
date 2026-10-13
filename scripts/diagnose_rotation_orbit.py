"""Controlled DE430 asteroid/attitude audit at the same 1969 initial epoch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lunarops.classes.ephemerides import body_state_matrices, body_state_matrix
from lunarops.classes.observation_factory import ensure_registered
from lunarops.config.context import RunContext
from lunarops.config.loader import build_run_plan, load_config_file
from lunarops.fileio.numeric_table import read_numeric_table
from lunarops.programs.lunar_orbit import build_lunar_propagation_problem
from lunarops.programs.registry import get_program

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "output/rotation_orbit_audit_20261010"
MAJOR = ROOT / "output/rotation_de430_10a_20261010"
FULL = ROOT / "output/rotation_de430_10a_full_20261010"


def build(name):
    ensure_registered()
    plan = build_run_plan(load_config_file(AUDIT / f"{name}.yml"))
    config = get_program("LunarOrbitPropagation").spec.schema.resolve(plan.calls[0][1])
    context = RunContext(working_dir=ROOT)
    return build_lunar_propagation_problem(config, context), context


def acceleration_audit():
    full, full_context = build("prescribed_full")
    major, major_context = build("prescribed_major")
    rows = []
    try:
        for days in (0, 1, 3, 30, 365.25, 3652.5):
            epoch = full.initial_epoch.shifted(days * 86400)
            states = body_state_matrix(full.dynamics.ephemeris, ("EARTH", "MOON"), epoch)
            relative = states[1] - states[0]
            full_data = full.dynamics.build_epoch_data(epoch)
            major_data = major.dynamics.build_epoch_data(epoch)
            all_accel = full.dynamics.accelerations(epoch, relative, full_data)
            major_accel = major.dynamics.accelerations(epoch, relative, major_data)
            added = all_accel.total_accelerations_mps2[1] - major_accel.total_accelerations_mps2[1]
            indices = [i for i, name in enumerate(full.dynamics.body_names)
                       if name not in major.dynamics.body_names]
            delta = full_data.positions_m[indices] - states[1, :3]
            mu = full.dynamics.gravitational_parameters_m3_s2[indices]
            asteroid_accel = np.sum(mu[:, None] * delta / np.linalg.norm(delta, axis=1)[:, None]**3, axis=0)
            reference = full.dynamics.ephemeris.body_acceleration_bcrs("MOON", epoch)
            rows.append({
                "days": days, "asteroid_count": len(indices),
                "asteroid_acceleration_on_moon_mps2": asteroid_accel.tolist(),
                "full_minus_major_acceleration_mps2": added.tolist(),
                "added_minus_direct_asteroid_acceleration_norm_mps2": float(np.linalg.norm(added - asteroid_accel)),
                "full_reference_acceleration_residual_norm_mps2": float(np.linalg.norm(all_accel.total_accelerations_mps2[1] - reference)),
                "major_reference_acceleration_residual_norm_mps2": float(np.linalg.norm(major_accel.total_accelerations_mps2[1] - reference)),
            })
        report = {"full_body_count": len(full.dynamics.body_names),
                  "major_body_count": len(major.dynamics.body_names), "epochs": rows}
        (AUDIT / "acceleration_audit.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    finally:
        full_context.close()
        major_context.close()


def read_orbit(directory):
    _, columns, data, _ = read_numeric_table(directory / "lunar_orbit.dat", "lunarOrbit")
    labels = [f"relative_{axis}" for axis in ("x_m", "y_m", "z_m", "vx_mps", "vy_mps", "vz_mps")]
    return data[:, columns.index("offset_tdb_s")], data[:, [columns.index(label) for label in labels]]


def metrics(values):
    norm = np.linalg.norm(values, axis=1)
    return {"max_m": float(norm.max()), "rms_m": float(np.sqrt(np.mean(norm**2))),
            "final_m": float(norm[-1])}


def compare():
    problem, context = build("prescribed_full")
    directories = {"coupled_major": MAJOR, "coupled_full": FULL,
                   "prescribed_full": AUDIT / "prescribed_full",
                   "prescribed_major": AUDIT / "prescribed_major"}
    try:
        offsets, _ = read_orbit(FULL)
        epochs = tuple(problem.initial_epoch.shifted(float(t)) for t in offsets)
        states = body_state_matrices(problem.dynamics.ephemeris, ("EARTH", "MOON"), epochs)
        reference = states[:, 1] - states[:, 0]
    finally:
        context.close()
    errors, positions = {}, {}
    for name, directory in directories.items():
        times, orbit = read_orbit(directory)
        np.testing.assert_array_equal(times, offsets)
        positions[name] = orbit[:, :3]
        errors[name] = orbit[:, :3] - reference[:, :3]
    pairs = (("coupled_major", "prescribed_major"), ("coupled_full", "prescribed_full"))
    report = {"initial_tdb_jd": problem.initial_epoch.jd1 + problem.initial_epoch.jd2,
              "duration_julian_years": float(offsets[-1] / (365.25 * 86400)),
              "step_seconds": 2700, "samples": len(offsets),
              "reference_position_errors": {name: metrics(error) for name, error in errors.items()},
              "first_three_days_reference_position_errors": {
                  name: metrics(error[offsets <= 3 * 86400]) for name, error in errors.items()},
              "coupled_vs_prescribed_position_difference": {
                  f"{a}_vs_{b}": metrics(positions[a] - positions[b]) for a, b in pairs}}
    (AUDIT / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    years = offsets / (365.25 * 86400)
    for name, error in errors.items():
        axes[0].plot(years, np.linalg.norm(error, axis=1), label=name)
    for a, b in pairs:
        axes[1].plot(years, np.linalg.norm(positions[a] - positions[b], axis=1), label=f"{a} vs {b}")
    for ax in axes:
        ax.set_xlabel("Julian years since JD 2440400.5")
        ax.set_ylabel("Relative position difference [m]")
        ax.set_yscale("log")
        ax.grid(alpha=.25)
        ax.legend()
    axes[0].set_title("DE430 reference error: identical epoch, orbit initial state and 2700 s step")
    axes[1].set_title("Effect of integrating attitude within the same external-body selection")
    fig.savefig(AUDIT / "comparison.png", dpi=180)
    fig.savefig(AUDIT / "comparison.pdf")
    plt.close(fig)
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("acceleration", "compare"))
    args = parser.parse_args()
    (acceleration_audit if args.mode == "acceleration" else compare)()


if __name__ == "__main__":
    main()
