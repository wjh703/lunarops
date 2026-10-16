"""Controlled degree-2 comparison, including reference PA and ICRF residuals."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.signal import periodogram
from scipy.spatial.transform import Rotation

from lunarops.classes.ephemerides.calceph import CalcephEphemeris
from lunarops.classes.time import Epoch
from lunarops.fileio.numeric_table import read_numeric_table

YEAR = 365.25 * 86400
MAS = 206264806.24709636
MODES = ("baseline", "c21", "s21", "both", "negative", "mechanical")


def read(path):
    _, columns, data, meta = read_numeric_table(path, "lunarDynamicsReferenceComparison")
    return {key: data[:, i] for i, key in enumerate(columns)}, meta


def linear_fit(times, values):
    design = np.column_stack((np.ones(len(times)), times / YEAR))
    beta = np.linalg.lstsq(design, values, rcond=None)[0]
    residual = values - design @ beta
    return beta, residual


def monthly_component(times, values):
    # The final half-day endpoint is excluded: the FFT requires uniform sampling.
    stop = len(times)
    if not np.isclose(times[-1] - times[-2], times[1] - times[0], rtol=0, atol=1e-3):
        stop -= 1
    times, values = times[:stop], values[:stop]
    np.testing.assert_allclose(np.diff(times), times[1] - times[0], rtol=0, atol=1e-3)
    _, residual = linear_fit(times, values)
    frequency, power = periodogram(residual, fs=86400 / (times[1] - times[0]), axis=0)
    band = (frequency >= 1 / 35) & (frequency <= 1 / 20)
    index = np.flatnonzero(band)[np.argmax(power[band, :2].sum(axis=1))]
    days = times / 86400

    def fit(f):
        design = np.column_stack((np.ones(len(days)), days / 365.25,
                                  np.cos(2 * np.pi * f * days), np.sin(2 * np.pi * f * days)))
        beta = np.linalg.lstsq(design, values, rcond=None)[0]
        return beta, values - design @ beta

    delta = frequency[1] - frequency[0]
    optimum = minimize_scalar(lambda f: np.sum(fit(f)[1][:, :2] ** 2),
                              bounds=(frequency[index] - delta, frequency[index] + delta), method="bounded",
                              options={"xatol": 1e-12})
    beta, _ = fit(optimum.x)
    return {"period_days": float(1 / optimum.x),
            "axis_amplitude_mas": (np.hypot(beta[2], beta[3]) * MAS).tolist(),
            "samples": stop}, frequency, power


def analyze(root, kernels):
    baseline, meta = read(root / "baseline/de440_reference_comparison.dat")
    times = baseline["offset_tdb_s"]
    initial = Epoch(meta["initialTdbJd1"], meta["initialTdbJd2"], "tdb")
    epochs = tuple(initial.shifted(float(t)) for t in times)
    ephemeris = CalcephEphemeris(kernels, lunar_relativistic_scale_convention="tdbCompatibleLunarSurface",
                                longitude_libration_correction="none")
    try:
        reference = Rotation.from_matrix(ephemeris.lunar_orientation.pa_to_lcrs_matrices(epochs))
    finally:
        ephemeris.close()
    report = {"cases": {}, "scope": "10 Julian years, same initial state and all forces; diagnostic adapters only",
              "spectral_method": "daily samples; exclude irregular final endpoint; linear detrend; periodogram; joint x/y sinusoid fit in 20-35 day band",
              "surface_scale_definition": "1738000 m times attitude error angle; not an LLR range residual"}
    radius = 1738000
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), constrained_layout=True)
    for mode in MODES:
        directory = root / mode
        series, _ = read(directory / "de440_reference_comparison.dat")
        np.testing.assert_array_equal(times, series["offset_tdb_s"])
        pa = np.column_stack([series[f"orientation_error_ref_pa_{a}_rad"] for a in "xyz"])
        inertial = reference.apply(pa)
        omega_pa = np.column_stack([series[f"mantle_omega_error_ref_pa_{a}_rad_s"] for a in "xyz"])
        omega_inertial = reference.apply(omega_pa)
        # Relative rotation applied to z gives the physical pole separation.
        z = np.tile([0., 0., 1.], (len(times), 1))
        pole_difference = reference.apply(Rotation.from_rotvec(pa).apply(z) - z)
        beta, detrended = linear_fit(times, inertial)
        monthly, frequency, power = monthly_component(times, pa)
        icrf_monthly, _, _ = monthly_component(times, inertial)
        stats = json.loads((directory / "de440_reference_comparison.json").read_text())
        stats.update({"final_surface_vector_ref_pa_m": (radius * pa[-1]).tolist(),
                      "final_surface_vector_icrf_m": (radius * inertial[-1]).tolist(),
                      "icrf_linear_slope_surface_m_per_year": (radius * beta[1]).tolist(),
                      "icrf_linear_fit_residual_rms_surface_m": float(radius * np.sqrt(np.mean(np.sum(detrended ** 2, axis=1)))),
                      "max_pole_displacement_scale_m": float(radius * np.max(np.linalg.norm(pole_difference, axis=1))),
                      "monthly_pa": monthly, "monthly_icrf": icrf_monthly,
                      "sampled_surface_scale_m": {str(day): {"actual_day": float(times[np.argmin(abs(times / 86400 - day))] / 86400),
                          "value": float(series["orientation_surface_displacement_scale_m"][np.argmin(abs(times / 86400 - day))])}
                          for day in (30, 365.25, 1826.25, 3652.5)}})
        report["cases"][mode] = stats
        out = np.column_stack((times, pa, inertial, omega_pa, omega_inertial, pole_difference * radius))
        header = ["offset_tdb_s", *(f"error_pa_{a}_rad" for a in "xyz"), *(f"error_icrf_{a}_rad" for a in "xyz"),
                  *(f"omega_error_pa_{a}_rad_s" for a in "xyz"), *(f"omega_error_icrf_{a}_rad_s" for a in "xyz"),
                  *(f"pole_difference_icrf_{a}_m" for a in "xyz")]
        np.savetxt(directory / "orientation_frames.csv", out, delimiter=",", header=",".join(header), comments="", fmt="%.17g")
        axes[0, 0].plot(times / YEAR, series["orientation_surface_displacement_scale_m"], label=mode)
        axes[0, 1].plot(times / YEAR, np.sqrt(sum(series[f"relative_position_error_{a}_m"] ** 2 for a in "xyz")), label=mode)
        if mode in ("baseline", "both"):
            for i, axis in enumerate("xyz"):
                axes[1, 0].plot(times / YEAR, radius * pa[:, i], label=f"{mode} {axis}")
                axes[1, 1].plot(times / YEAR, radius * inertial[:, i], label=f"{mode} {axis}")
        if mode == "both":
            for i, axis in enumerate("xyz"):
                axes[2, 0].plot(times / YEAR, radius * inertial[:, i], label=axis)
                band = (frequency > 1 / 40) & (frequency < 1 / 15)
                axes[2, 1].plot(1 / frequency[band], power[band, i] * MAS ** 2, label=axis)
    # Linear response test: does the sum of individual changes predict both?
    c, _ = read(root / "c21/de440_reference_comparison.dat")
    s, _ = read(root / "s21/de440_reference_comparison.dat")
    b, _ = read(root / "both/de440_reference_comparison.dat")
    interaction = np.column_stack([b[f"orientation_error_ref_pa_{a}_rad"] - c[f"orientation_error_ref_pa_{a}_rad"]
                                   - s[f"orientation_error_ref_pa_{a}_rad"] + baseline[f"orientation_error_ref_pa_{a}_rad"] for a in "xyz"])
    report["c21_s21_superposition_error_surface_max_m"] = float(radius * np.max(np.linalg.norm(interaction, axis=1)))
    labels = ("Attitude angle times radius [m]", "Orbit position error [m]", "Reference PA residual times radius [m]",
              "ICRF residual times radius [m]", "Both offsets: ICRF residual times radius [m]", "Both offsets: detrended PA spectrum [mas² day]")
    for i, (ax, label) in enumerate(zip(axes.flat, labels, strict=True)):
        ax.set_ylabel(label); ax.set_xlabel("Period [day]" if i == 5 else "Julian years"); ax.grid(alpha=.25); ax.legend(fontsize=8)
    fig.savefig(root / "degree2_comparison.png", dpi=180)
    fig.savefig(root / "degree2_comparison.pdf")
    plt.close(fig)
    manifest = json.loads((root / "source_manifest.json").read_text())
    report["source_manifest_mismatches"] = [p for p, digest in manifest.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest() != digest]
    # Must reproduce the already completed full-environment baseline exactly.
    previous = Path("output/de440_missing_models_20261012/both")
    report["baseline_matches_previous_arrays"] = {}
    for filename, kind in (("lunar_orbit.dat", "lunarOrbit"), ("lunar_rotation.dat", "lunarRotation"),
                            ("lunar_acceleration_diagnostics.dat", "lunarAccelerationDiagnostics")):
        old = read_numeric_table(previous / filename, kind)
        new = read_numeric_table(root / "baseline" / filename, kind)
        report["baseline_matches_previous_arrays"][filename] = old[1] == new[1] and bool(np.array_equal(old[2], new[2]))
    (root / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: {"orbit_max_m": v["relative_position_error_m"]["max_norm"],
                         "attitude_max_m": v["orientation_surface_displacement_scale_m"]["max_norm"]} for k, v in report["cases"].items()}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("kernels", type=Path)
    args = parser.parse_args()
    analyze(args.root, args.kernels)
