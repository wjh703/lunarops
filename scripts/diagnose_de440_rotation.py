"""Summarize completed DE440 coupled propagation control experiments."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from lunarops.classes.ephemerides import body_state_matrices
from lunarops.classes.ephemerides.calceph import CalcephEphemeris
from lunarops.classes.time import Epoch
from lunarops.fileio.numeric_table import read_numeric_table

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "output/rotation_de440_audit_20261011"
BASE = ROOT / "output/rotation_de440_10a_full_20261011"


def metrics(values):
    norms = np.linalg.norm(values, axis=1)
    return {"max_norm": float(norms.max()), "rms_norm": float(np.sqrt(np.mean(norms**2))),
            "final_norm": float(norms[-1])}


def read_orbit(directory):
    _, columns, data, _ = read_numeric_table(directory / "lunar_orbit.dat", "lunarOrbit")
    labels = [f"relative_{axis}" for axis in ("x_m", "y_m", "z_m", "vx_mps", "vy_mps", "vz_mps")]
    return data[:, columns.index("offset_tdb_s")], data[:, [columns.index(label) for label in labels]]


def read_rotation(directory):
    _, columns, data, _ = read_numeric_table(directory / "lunar_rotation.dat", "lunarRotation")
    return Rotation.from_quat(data[:, [columns.index(f"q_{x}") for x in "xyzw"]])


def main():
    directories = {"baseline": BASE, **{name: AUDIT / name for name in
                   ("half_step", "core_pressure", "prescribed", "no_geodetic", "degree2_offsets")}}
    offsets, _ = read_orbit(BASE)
    epochs = tuple(Epoch(2440400.5, 0, "tdb").shifted(float(t)) for t in offsets)
    with CalcephEphemeris(ROOT.parent / "data/kernels/de440",
                         lunar_relativistic_scale_convention="tdbCompatibleLunarSurface",
                         longitude_libration_correction="none") as ephemeris:
        states = body_state_matrices(ephemeris, ("EARTH", "MOON"), epochs)
        reference = states[:, 1] - states[:, 0]
        reference_rotation = Rotation.from_matrix(ephemeris.lunar_orientation.pa_to_lcrs_matrices(epochs))
    errors, orbits, rotations, reports = {}, {}, {}, {}
    for name, directory in directories.items():
        times, orbit = read_orbit(directory)
        np.testing.assert_array_equal(times, offsets)
        orbits[name] = orbit
        errors[name] = orbit[:, :3] - reference[:, :3]
        reports[name] = {"position_error_m": metrics(errors[name])}
        if name != "prescribed":
            rotations[name] = read_rotation(directory)
            rv = (reference_rotation.inv() * rotations[name]).as_rotvec()
            reports[name]["orientation_error_rad"] = metrics(rv)
            reports[name]["surface_scale_m"] = metrics(1738000 * rv)
    differences = {}
    for name in directories:
        if name == "baseline":
            continue
        differences[name] = {"position_difference_from_baseline_m": metrics(
            orbits[name][:, :3] - orbits["baseline"][:, :3])}
        if name != "prescribed":
            rv = (rotations["baseline"].inv() * rotations[name]).as_rotvec()
            differences[name]["orientation_difference_from_baseline_rad"] = metrics(rv)
            differences[name]["surface_scale_difference_from_baseline_m"] = metrics(1738000 * rv)
    report = {"reference": "DE440 BSP/BPC", "initial_tdb_jd": 2440400.5,
              "duration_julian_years": 10, "samples": len(offsets), "cases": reports,
              "differences_from_baseline": differences,
              "degree2_offset_experiment": "Explicit diagnostic adapter only; raw comment constants are not assumed to be permanent offsets."}
    (AUDIT / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    years = offsets / (365.25 * 86400)
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    for name, values in errors.items():
        axes[0].plot(years, np.linalg.norm(values, axis=1), label=name)
    for name, rotation in rotations.items():
        axes[1].plot(years, 1738000 * (reference_rotation.inv() * rotation).magnitude(), label=name)
    for ax in axes:
        ax.set_xlabel("Julian years since JD 2440400.5")
        ax.grid(alpha=.25)
        ax.legend()
    axes[0].set_ylabel("Relative orbit position error [m]")
    axes[1].set_ylabel("Attitude surface displacement scale [m]")
    fig.suptitle("DE440 controlled experiments: all external bodies retained")
    fig.savefig(AUDIT / "comparison.png", dpi=180)
    fig.savefig(AUDIT / "comparison.pdf")
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
