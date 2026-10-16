"""Compare DE440 native constants and prescribed-BPC rotational closure.

Diagnostic only: does not change production parameters or fit initial states.
Run from the project root with ``python -m scripts.audit_de440_primary_sources``.
"""

from __future__ import annotations

import json
import re
from dataclasses import replace
from math import factorial
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation

from lunarops.classes.dynamics.context import DynamicsEpochData, DynamicsEvaluationContext
from lunarops.classes.dynamics.earth_gravity_orientation import de440_earth_fixed2inertial_matrix
from lunarops.classes.dynamics.rotation import quaternion_derivative, rotation_vector_from_euler
from lunarops.classes.ephemerides import body_state_matrix
from lunarops.classes.ephemerides.calceph import CalcephEphemeris
from lunarops.classes.time import Epoch
from lunarops.config.context import RunContext
from lunarops.fileio.mass_catalog import read_mass_catalog
from lunarops.fileio.numeric_table import read_numeric_table
from lunarops.fileio.structured_reports import read_lunar_orbit_metadata
from lunarops.programs.lunar_orbit import _build_rotation, _load_configured_gravity_fields
from scripts.check_de440_rotation_residual import omega_from_euler
from scripts.compare_de440_iau2006 import stats

ROOT = Path("output/de440_primary_source_audit_20261014")
RUN = Path("output/rotation_de440_environment_10a_full_20261014")
NATIVE = Path("../data/kernels/de440/de440_tech-comments.txt")


def trajectory_and_figures():
    """Describe the two completed runs, without adjusting their states."""
    other = Path("output/rotation_de440_environment_10a_step5400_full_20261014")
    _, columns, a, _ = read_numeric_table(RUN / "lunar_rotation.dat", "lunarRotation")
    _, other_columns, b, _ = read_numeric_table(other / "lunar_rotation.dat", "lunarRotation")
    ia = [columns.index("q_" + axis) for axis in "xyzw"]
    ib = [other_columns.index("q_" + axis) for axis in "xyzw"]
    qa, qb = Rotation.from_quat(a[:, ia]), Rotation.from_quat(b[:, ib])
    cfg = read_lunar_orbit_metadata(RUN / "lunar_orbit_metadata.txt")["config"]
    epoch0 = Epoch(cfg["initialTdbJd1"], cfg["initialTdbJd2"], "tdb")
    # Time is explicitly read from the reference comparison's documented column.
    _, _, reference_comparison, _ = read_numeric_table(
        RUN / "de440_reference_comparison.dat", "lunarDynamicsReferenceComparison"
    )
    _, _, other_comparison, _ = read_numeric_table(
        other / "de440_reference_comparison.dat", "lunarDynamicsReferenceComparison"
    )
    np.testing.assert_array_equal(reference_comparison[:, 0], other_comparison[:, 0])
    times = reference_comparison[:, 0]
    epochs = [epoch0.shifted(float(t)) for t in times]
    with CalcephEphemeris(
        cfg["ephemerides"]["directory"], lunar_relativistic_scale_convention="tdbCompatibleLunarSurface"
    ) as eph:
        reference = Rotation.from_matrix(eph.lunar_orientation.pa_to_lcrs_matrices(epochs))
    radius = cfg["lunarRotation"]["radiusM"]
    errors = radius * (qa * reference.inv()).as_rotvec()
    delta = radius * (qb * qa.inv()).as_rotvec()
    metrics, detrended, _ = stats(times / 86400, errors)
    report = {
        "current_run_icrf_surface_m": metrics,
        "direct_step_difference_max_surface_m": float(np.linalg.norm(delta, axis=1).max()),
    }
    (ROOT / "current_run_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savetxt(
        ROOT / "current_run_icrf_residual.csv",
        np.column_stack((times, errors, detrended, delta)),
        delimiter=",",
        header="offset_tdb_s,error_x_m,error_y_m,error_z_m,detrended_x_m,detrended_y_m,detrended_z_m,step_delta_x_m,step_delta_y_m,step_delta_z_m",
        comments="",
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    for axis, label in enumerate("xyz"):
        axes[0].plot(times / (365.25 * 86400), errors[:, axis] * 1000, label=label)
        axes[1].plot(times / (365.25 * 86400), delta[:, axis] * 1e6, label=label)
    axes[0].set_ylabel("BPC attitude residual, ICRF [mm]")
    axes[1].set_ylabel("5400 s minus 2700 s, ICRF [micrometre]")
    for ax in axes:
        ax.set_xlabel("Julian years")
        ax.grid(alpha=0.3)
        ax.legend()
    for extension in ("png", "pdf"):
        fig.savefig(ROOT / f"current_attitude_and_step_difference.{extension}", dpi=180)
    plt.close(fig)
    phase = np.loadtxt(ROOT / "bpc_record_phase_closure.csv", delimiter=",", skiprows=1)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True)
    for ax, start in zip(axes, np.unique(phase[:, 0]), strict=True):
        sample = phase[phase[:, 0] == start]
        for axis, label in enumerate("xyz"):
            ax.plot(sample[:, 1] * 8, sample[:, axis + 2], label=label)
        ax.set_title(f"8-day BPC interval starting at day {start / 86400:g}")
        ax.set_xlabel("Days within BPC record")
        ax.set_ylabel("Local closure [rad/s2]")
        ax.grid(alpha=0.3)
        ax.legend()
    for extension in ("png", "pdf"):
        fig.savefig(ROOT / f"bpc_second_derivative_phase.{extension}", dpi=180)
    plt.close(fig)
    return report


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    cfg = read_lunar_orbit_metadata(RUN / "lunar_orbit_metadata.txt")["config"]
    native = {
        m[1]: float(m[2].replace("D", "E"))
        for m in re.finditer(r"^([A-Z0-9]+)\s+([-+0-9.]+D[-+]\d+)\s*$", NATIVE.read_text(), re.MULTILINE)
    }
    r = cfg["lunarRotation"]
    d = r["deformation"]
    b, g, j = (native[key] for key in ("LBET", "LGAM", "J2M"))
    moments = 2 * j / (2 * b - g + b * g) * np.array([1 - b * g, 1 + g, 1 + b])
    c22 = (r["inertiaMoments"][1] - r["inertiaMoments"][0]) / 4
    checks = []

    def check(name, actual, expected):
        a, e = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
        checks.append({"name": name, "actual": a.tolist(), "expected": e.tolist(), "difference": (a - e).tolist()})

    check("Park_Eq47_49_inertia_MR2", r["inertiaMoments"], moments)
    check("initial_mantle_rad_s", r["initialAngularVelocityRadS"], [native["OMEGA" + a] / 86400 for a in "XYZ"])
    check("initial_core_rad_s", r["initialCoreAngularVelocityRadS"], [native["OMGC" + a] / 86400 for a in "XYZ"])
    check(
        "initial_quaternion",
        r["initialQuaternion"],
        rotation_vector_from_euler(native["PHI"], native["THT"], native["PSI"], 0, 0, 0)[:4],
    )
    check("radius_m", r["radiusM"], 1000 * native["AM"])
    check("love_k2", d["loveK2"], native["K2M"])
    check("delay_days", d["delayDays"], native["TAUM"])
    check("core_fraction", r["core"]["polarMomentFraction"], native["IFAC"])
    check("core_flattening", r["core"]["flattening"], native["COBLAT"])
    check("viscosity_per_s", r["core"]["viscosityPerS"], native["KVC"] * moments[2] / 86400)
    for key, symbol in (("c21Static", "C21M"), ("s21Static", "S21M"), ("s22Static", "S22M")):
        check(key, d[key], native[symbol])

    gravity = []
    for body, suffix in (("moon", "M"), ("earth", "E")):
        path = Path(cfg["gravityFields"][body]["path"])
        for line in path.read_text().splitlines():
            cells = line.split()
            if not cells or cells[0] != "gfc":
                continue
            n, m = map(int, cells[1:3])
            if n < 2:
                continue
            normal = np.sqrt((2 if m else 1) * (2 * n + 1) * factorial(n - m) / factorial(n + m))
            for kind, value in zip("CS", map(float, cells[3:5]), strict=True):
                symbol = f"J{n}{suffix}" if m == 0 and kind == "C" else f"{kind}{n}{m}{suffix}"
                expected = -native[symbol] if m == 0 and kind == "C" else native.get(symbol, 0.0)
                actual = value * normal
                gravity.append(
                    {
                        "body": body,
                        "degree": n,
                        "order": m,
                        "kind": kind,
                        "actual": actual,
                        "expected": expected,
                        "difference": actual - expected,
                    }
                )

    with RunContext(working_dir=Path.cwd()) as context:
        catalog = read_mass_catalog(context.resolve_path(cfg["inputFileMassCatalog"]))
        names = ("EARTH", "MOON", "SUN", *r["sources"][2:])
        mu = {name: catalog.require(name).gm_m3_s2 for name in names}
        unit = (1000 * native["AU"]) ** 3 / 86400**2
        lunar_gm = native["GMB"] * unit / (1 + native["EMRAT"])
        check("moon_gm_from_GMB_EMRAT", mu["MOON"], lunar_gm)
        check("earth_gm_from_GMB_EMRAT", mu["EARTH"], lunar_gm * native["EMRAT"])
        for name, symbol in (
            ("SUN", "GMS"),
            ("MERCURY BARYCENTER", "GM1"),
            ("VENUS BARYCENTER", "GM2"),
            ("MARS BARYCENTER", "GM4"),
            ("JUPITER BARYCENTER", "GM5"),
            ("SATURN BARYCENTER", "GM6"),
        ):
            check(name + "_gm", mu[name], native[symbol] * unit)
        fields = _load_configured_gravity_fields(cfg["gravityFields"], context, mu)
        rotation, initial_rotation = _build_rotation(cfg, fields, mu["MOON"], mu["EARTH"])
        epoch0 = Epoch(cfg["initialTdbJd1"], cfg["initialTdbJd2"], "tdb")
        inputs = DynamicsEvaluationContext(names, [mu[name] for name in names])
        with CalcephEphemeris(
            context.resolve_path(cfg["ephemerides"]["directory"]),
            lunar_relativistic_scale_convention="tdbCompatibleLunarSurface",
        ) as eph:

            def mantle(epoch):
                a = np.asarray(
                    eph._require_handle().orient_order(
                        epoch.jd1, epoch.jd2, eph._orientation_target, eph._angle_units, 2
                    )
                )
                w, wd = omega_from_euler(a)
                q = rotation_vector_from_euler(*a[:6])[:4]
                return q, w, wd

            class ReferenceHistory:
                def integration_sample(self, epoch):
                    states = body_state_matrix(eph, ("EARTH", "MOON"), epoch)
                    q, w, wd = mantle(epoch)
                    relative = states[1] - states[0]
                    return (
                        np.r_[relative, q, w, np.zeros(3)],
                        np.r_[relative[3:], np.zeros(3), quaternion_derivative(q, w), wd, np.zeros(3)],
                    )

            history = ReferenceHistory()

            def current(t):
                epoch = epoch0.shifted(float(t))
                states = body_state_matrix(eph, names, epoch)
                q, w, wd = mantle(epoch)
                R = eph.lunar_orientation.pa_to_lcrs_matrix(epoch)
                gm = np.array([mu[name] for name in names])
                gm[2] *= (
                    1
                    + cfg["solarGmVariation"]["fractionalRatePerYear"] * ((epoch.jd1 - 2451545.0) + epoch.jd2) / 365.25
                )
                data = DynamicsEpochData(
                    epoch,
                    names,
                    states[:, :3],
                    states[:, 3:],
                    de440_earth_fixed2inertial_matrix(epoch),
                    R,
                    np.zeros(3),
                    gravitational_parameters_m3_s2=gm,
                )
                inputs.load_evaluation(data, states[1] - states[0], history, None)
                return q, w, wd

            def core_rhs(t, wc):
                _, w, _ = current(t)
                nu = rotation.geodetic_rate(inputs)
                K = rotation.core_inertia
                cmb = rotation.core.viscosity_per_s * (wc - w)
                cmb += (K[2, 2] - K[0, 0]) * w[2] * np.array([-wc[1], wc[0], 0.0])
                return np.linalg.solve(K, -cmb - np.cross(w - nu, K @ wc))

            # Only three core variables evolve, driven by the published mantle.
            # This is a model-conditioned core, not a JPL core reference state.
            duration = cfg["durationSeconds"]
            solution = solve_ivp(
                core_rhs,
                (0, duration),
                initial_rotation[7:10],
                method="DOP853",
                rtol=1e-11,
                atol=1e-19,
                max_step=2 * 86400,
                dense_output=True,
            )
            if not solution.success:
                raise RuntimeError(solution.message)
            print("core-only prescribed-BPC propagation", solution.nfev, "RHS evaluations", flush=True)
            finer = solve_ivp(
                core_rhs,
                (0, duration),
                initial_rotation[7:10],
                method="DOP853",
                rtol=1e-12,
                atol=1e-20,
                max_step=86400,
                dense_output=True,
            )
            if not finer.success:
                raise RuntimeError(finer.message)
            probe_times = np.linspace(0, duration, 115)
            core_convergence = float(np.linalg.norm(solution.sol(probe_times) - finer.sol(probe_times), axis=0).max())
            n_kepler = np.sqrt((mu["EARTH"] + mu["MOON"]) / 384399014.0**3)
            original_parameters = rotation.deformation.parameters
            cases = ("baseline", "kepler_mean_motion", "no_static_c21_s21", "static_c22_offset", "core_pressure_spin")
            rows = []
            # 1000 s inside each 8-day polynomial, with one same-side initial probe.
            offsets = np.r_[0.0, np.arange(1000.0, duration, 32 * 86400.0)]
            for t in offsets:
                q, w, wd = current(t)
                wc = solution.sol(t)
                state = np.r_[q, w, wc]
                for case in cases:
                    parameters = original_parameters
                    if case == "kepler_mean_motion":
                        parameters = replace(parameters, mean_motion_rad_s=n_kepler)
                    if case == "no_static_c21_s21":
                        parameters = replace(parameters, c21_static=0.0, s21_static=0.0)
                    rotation.deformation.parameters = parameters
                    rotation.core = replace(
                        rotation.core, pressure_spin="core" if case == "core_pressure_spin" else "mantle"
                    )
                    inputs._figure_cache.clear()
                    inputs.lunar_inertia = rotation.deformation.evaluate(inputs)
                    if case == "static_c22_offset":
                        coeff = fields["MOON"].coefficients.coeffs[:, 2, :3].copy()
                        coeff[0, 2] += (native["C22M"] - c22) * np.sqrt(12 / 5)
                        fields["MOON"].update_degree2_coefficients(coeff)
                    evaluation = rotation.evaluate(inputs, state)
                    residual = evaluation.derivative[4:7] - wd
                    rows.append([t, cases.index(case), *residual])
            rotation.deformation.parameters = original_parameters
            rotation.core = replace(rotation.core, pressure_spin="mantle")
            # Examine several phases within the same BPC record. A derivative
            # residual at a single phase need not be a physical torque bias.
            phase_rows = []
            quadrature = []
            nodes, weights = np.polynomial.legendre.leggauss(64)
            interval = 8 * 86400.0
            starts = (0.0, 228 * interval, 455 * interval)
            for start in starts:
                phase_residuals = []
                for fraction, weight in zip((nodes + 1) / 2, weights / 2, strict=True):
                    t = start + fraction * interval
                    q, w, wd = current(t)
                    inputs.lunar_inertia = rotation.deformation.evaluate(inputs)
                    residual = rotation.evaluate(inputs, np.r_[q, w, solution.sol(t)]).derivative[4:7] - wd
                    phase_rows.append([start, fraction, *residual])
                    phase_residuals.append(residual)
                phase_residuals = np.asarray(phase_residuals)
                average = weights @ phase_residuals / 2
                lower_nodes, lower_weights = np.polynomial.legendre.leggauss(32)
                lower_residuals = []
                for fraction in (lower_nodes + 1) / 2:
                    t = start + fraction * interval
                    q, w, wd = current(t)
                    inputs.lunar_inertia = rotation.deformation.evaluate(inputs)
                    lower_residuals.append(rotation.evaluate(inputs, np.r_[q, w, solution.sol(t)]).derivative[4:7] - wd)
                lower_average = lower_weights @ np.asarray(lower_residuals) / 2
                quadrature.append(
                    {
                        "record_start_s": start,
                        "peak_norm_rad_s2": float(np.linalg.norm(phase_residuals, axis=1).max()),
                        "mean_vector_rad_s2": average.tolist(),
                        "mean_norm_rad_s2": float(np.linalg.norm(average)),
                        "32_vs_64_point_mean_difference_rad_s2": float(np.linalg.norm(average - lower_average)),
                    }
                )

    a = np.asarray(rows)
    np.savetxt(
        ROOT / "prescribed_bpc_closure.csv",
        a,
        delimiter=",",
        header="offset_tdb_s,case_id,closure_x_rad_s2,closure_y_rad_s2,closure_z_rad_s2",
        comments="",
    )
    np.savetxt(
        ROOT / "bpc_record_phase_closure.csv",
        phase_rows,
        delimiter=",",
        header="record_start_tdb_s,record_fraction,closure_x_rad_s2,closure_y_rad_s2,closure_z_rad_s2",
        comments="",
    )
    closure = {}
    baseline = a[a[:, 1] == 0, 2:]
    for index, case in enumerate(cases):
        residual = a[a[:, 1] == index, 2:]
        closure[case] = {
            "max_norm_rad_s2": float(np.linalg.norm(residual, axis=1).max()),
            "rms_norm_rad_s2": float(np.sqrt(np.mean(np.sum(residual**2, axis=1)))),
            "initial_residual_rad_s2": residual[0].tolist(),
            "max_change_from_baseline_rad_s2": float(np.linalg.norm(residual - baseline, axis=1).max()),
        }
    report = {
        "source": str(NATIVE),
        "run_metadata": str(RUN / "lunar_orbit_metadata.txt"),
        "constant_checks": checks,
        "static_gravity_checks": gravity,
        "derived_undistorted_c22": c22,
        "tech_comment_c22": native["C22M"],
        "c22_difference": c22 - native["C22M"],
        "mean_motion_rad_s": d["meanMotionRadS"],
        "kepler_probe_mean_motion_rad_s": float(n_kepler),
        "closure": closure,
        "bpc_record_quadrature": quadrature,
        "core_solver_nfev": solution.nfev,
        "closure_samples": len(offsets),
        "core_half_step_tighter_tolerance_max_difference_rad_s": core_convergence,
        "scope": "BSP/BPC-prescribed state and analytic BPC second derivatives; core propagated with baseline Eq58. Alternative cases are same-state local RHS probes, not alternate integrations or fitted solutions. BPC initial epoch is a record boundary; its analytic derivative is one-sided. JPL core history unavailable.",
    }
    (ROOT / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    trajectory_and_figures()
    print(json.dumps({"c22_difference": report["c22_difference"], "closure": closure}, indent=2), flush=True)


if __name__ == "__main__":
    main()
