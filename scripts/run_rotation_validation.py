"""Real-kernel coupled rotation validation; no observation/estimation code.

Run with the array wrapper, then use ``compare`` to produce validation.json.
All runs share DE430 initial conditions and physical sources. Step, startup,
history order and the explicitly configured seed arc are varied separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from lunarops.fileio.numeric_table import read_numeric_table
from lunarops.fileio.structured_reports import read_lunar_orbit_metadata

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output/rotation_validation_20261010"
CASES = (
    ("short_1800", 3, 1800, 225, 13, 86400),
    ("short_900", 3, 900, 112.5, 13, 86400),
    ("short_startup", 3, 1800, 112.5, 13, 86400),
    ("short_history", 3, 1800, 225, 9, 86400),
    ("short_seed", 3, 1800, 225, 13, 172800),
    ("month_5400", 30, 5400, 675, 13, 86400),
    ("month_2700", 30, 2700, 337.5, 13, 86400),
    ("year_5400", 365, 5400, 675, 13, 86400),
    ("year_2700", 365, 2700, 337.5, 13, 86400),
    ("decade_5400", 3650, 5400, 675, 13, 86400),
    ("decade_2700", 3650, 2700, 337.5, 13, 86400),
)


def run(task):
    name, days, step, startup, history_order, seed = CASES[task]
    # Explicit seed choice is encoded into a saved complete YAML per case.
    import yaml

    config = yaml.safe_load((ROOT / "configs/lunarops_orbit_rotation_deformed_de430.yml").read_text())
    config["variables"].update(
        outputDir=str(OUTPUT / name), durationSeconds=days * 86400.,
        outputStepSeconds=86400., stepSeconds=float(step), startupStepSeconds=float(startup),
    )
    config["programs"][0]["historyInterpolationOrder"] = history_order
    config["programs"][0]["lunarRotation"]["deformation"]["seedDurationSeconds"] = float(seed)
    directory = OUTPUT / name
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "config.yml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    subprocess.run([sys.executable, "-m", "lunarops", "run", str(path), "--working-dir", str(ROOT)],
                   cwd=ROOT, check=True)
    (directory / "completed").write_text("success\n")


def rotation_angle(a, b):
    qa = Rotation.from_quat(a[:, [2, 3, 4, 1]])
    qb = Rotation.from_quat(b[:, [2, 3, 4, 1]])
    return (qa.inv() * qb).magnitude()


def compare():
    data, report = {}, {"scope": "coupled dynamics only; no LLR or estimation", "runs": {}, "comparisons": {}}
    from lunarops.classes.ephemerides.calceph import CalcephEphemeris
    from lunarops.classes.time import Epoch

    ephemeris = CalcephEphemeris(ROOT.parent / "data/kernels/de430",
                                lunar_relativistic_scale_convention="alreadyScaled",
                                longitude_libration_correction="none")
    for name, days, step, startup, history_order, seed in CASES:
        directory = OUTPUT / name
        if not (directory / "completed").exists():
            raise RuntimeError(f"Required validation run has not completed: {name}")
        _, columns, rotation, _ = read_numeric_table(directory / "lunar_rotation.dat", "lunarRotation")
        _, _, orbit, _ = read_numeric_table(directory / "lunar_orbit.dat", "lunarOrbit")
        if not np.all(np.isfinite(rotation)) or not np.all(np.isfinite(orbit)):
            raise RuntimeError(f"Nonfinite result in {name}")
        data[name] = rotation, orbit
        epochs = tuple(Epoch(2440400.5, 0., "tdb").shifted(float(t)) for t in rotation[:, 0])
        reference_rotations = Rotation.from_matrix(ephemeris.lunar_orientation.pa_to_lcrs_matrices(epochs))
        calculated_rotations = Rotation.from_quat(rotation[:, [2, 3, 4, 1]])
        reference_angles = (reference_rotations.inv() * calculated_rotations).magnitude()
        metadata = read_lunar_orbit_metadata(directory / "lunar_orbit_metadata.txt")
        indices = [columns.index(f"inertia_dot_per_s_{x}") for x in ("xx", "yy", "zz")]
        report["runs"][name] = {
            "days": days, "step_s": step, "startup_step_s": startup, "history_order": history_order,
            "seed_duration_s": seed, "samples": len(rotation),
            "max_quaternion_norm_error": float(np.max(abs(np.linalg.norm(rotation[:, 1:5], axis=1) - 1))),
            "max_inertia_derivative_trace_per_s": float(np.max(abs(np.sum(rotation[:, indices], axis=1)))),
            "max_reference_orientation_difference_rad": float(np.max(reference_angles)),
            "max_reference_surface_displacement_scale_m": float(1738000 * np.max(reference_angles)),
            "max_reference_orbit_rtn_difference_m": metadata["referenceDifferenceRtnMaxAbsM"],
        }
    ephemeris.close()
    pairs = (
        ("short_1800", "short_900"), ("short_1800", "short_startup"),
        ("short_1800", "short_history"), ("short_1800", "short_seed"),
        ("month_5400", "month_2700"), ("year_5400", "year_2700"), ("decade_5400", "decade_2700"),
    )
    for first, second in pairs:
        a, oa = data[first]
        b, ob = data[second]
        np.testing.assert_array_equal(a[:, 0], b[:, 0])
        report["comparisons"][f"{first}_vs_{second}"] = {
            "max_orientation_rad": float(np.max(rotation_angle(a, b))),
            "max_surface_displacement_scale_m": float(1738000 * np.max(rotation_angle(a, b))),
            "max_mantle_omega_difference_rad_s": float(np.max(np.linalg.norm(a[:, 5:8] - b[:, 5:8], axis=1))),
            "max_core_omega_difference_rad_s": float(np.max(np.linalg.norm(a[:, 8:11] - b[:, 8:11], axis=1))),
            "max_relative_position_difference_m": float(np.max(np.linalg.norm(oa[:, 19:22] - ob[:, 19:22], axis=1))),
        }
    report["model_limitations"] = [
        "Major external bodies only; asteroid perturbations omitted.",
        "Explicit rigid-backward seed is not a fitted DE430 prehistory.",
        "Step convergence measures numerical reproducibility, not full JPL ephemeris accuracy.",
    ]
    report["reference_kernel_files"] = sorted(path.name for path in (ROOT.parent / "data/kernels/de430").iterdir()
                                              if path.suffix in (".bsp", ".bpc"))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "validation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


def reference(julia):
    """Execute extracted LOCAL Julia kernels with common SI-normalized inputs.

    This compares fixed-input kernels, not Julia's complete ephemeris solve.
    Its Earth figure term uses mantle inertia; that recorded difference is
    intentionally excluded from the angular-RHS comparison.
    """
    from types import SimpleNamespace

    from lunarops.classes.dynamics import LunarCoreParameters, LunarRotationDynamics, LunarTorqueGroup
    from lunarops.classes.dynamics.deformation import (
        LunarDeformationParameters,
        LunarInertiaEvaluation,
        deformation_tensors,
    )

    sources = ROOT.parent / "PlanetaryEphemeris.jl/src"
    inertia_path = sources / "jpl-de-430-431-earth-orientation-model.jl"
    dynamics_path = sources / "DE430/dynamicalmodel.jl"
    inertia_source, dynamics_source = inertia_path.read_text(), dynamics_path.read_text()
    functions = []
    for name in ("ITM1", "ITM2"):
        start = inertia_source.index(f"function {name}(")
        end = inertia_source.index("\nend", start) + len("\nend")
        functions.append(inertia_source[start:end])
    section = dynamics_source[dynamics_source.rindex("    Iω_x ="):]
    products = section[:section.index("    # Moon -> Earth radial")]
    start = section.index("    N_cmb_1 =")
    exchange = section[start:section.index("    # Lunar mantle physical librations", start)]
    writes = "\n".join(line for line in section.splitlines() if any(
        line.lstrip().startswith(f"dq[6N+{i}] =") for i in (4, 5, 6, 10, 11, 12)
    ))
    radius, earth_mu, moon_mu = 1738000., 3.98600435507e14, 4.902800118e12
    p = LunarDeformationParameters(.024059, 8280., 2.6616995e-6)
    total = np.diag([.392894284681978, .3929838156443403, .39314226584780865])
    core = LunarCoreParameters(.0007, .0002462390478919815, 7.446777275531163e-14, "core")
    rotation = LunarRotationDynamics(total, radius, LunarTorqueGroup(), core=core)
    rng = np.random.default_rng(430)
    rows, expected = [], []
    for _ in range(100):
        r = rng.normal(size=3)
        r *= 3.84e8 / np.linalg.norm(r)
        rd = rng.normal(size=3) * 1000
        w, wc = rng.normal(size=(2, 3)) * 2.7e-6
        wd = rng.normal(size=3) * 1e-13
        tide, spin, derivative = deformation_tensors(
            r, rd, w, wd, parameters=p, radius_m=radius, earth_mu=earth_mu, moon_mu=moon_mu,
        )
        evaluation = LunarInertiaEvaluation(total + tide + spin, derivative, tide, spin)
        inputs = SimpleNamespace(lunar_inertia=evaluation)
        result = rotation.evaluate(inputs, np.concatenate(([1, 0, 0, 0], w, wc)))
        rows.append(np.concatenate((r, rd, w, wd, wc, (evaluation.total - rotation.core_inertia).ravel(order="F"),
                                    derivative.ravel(order="F"))))
        expected.append(np.concatenate((tide.ravel(order="F"), spin.ravel(order="F"),
                                        derivative.ravel(order="F"), result.derivative[4:])))
    directory = OUTPUT / "julia_reference"
    directory.mkdir(parents=True, exist_ok=True)
    input_path, code_path, output_path = directory / "inputs.txt", directory / "reference.jl", directory / "outputs.txt"
    np.savetxt(input_path, rows)
    declarations = f"""using LinearAlgebra, DelimitedFiles
const ea=1; const mo=2; const μ=[{earth_mu}, {moon_mu}]
const R_moon={radius}; const n_moon={p.mean_motion_rad_s}; const k_2M={p.love_k2}
const normalization={moon_mu * radius**2}
"""
    body = f"""
function evaluate(row)
    r=row[1:3]; rd=row[4:6]; w=row[7:9]; wd=row[10:12]; wc=row[13:15]
    I_m_t=reshape(row[16:24],3,3); dI_m_t=reshape(row[25:33],3,3)
    I_c_t=Diagonal([{rotation.core_inertia[0,0]}, {rotation.core_inertia[1,1]}, {rotation.core_inertia[2,2]}])
    inv_I_m_t=inv(I_m_t); inv_I_c_t=inv(I_c_t)
    N=0; q=zeros(12); dq=zeros(12); q[4:6]=w; q[10:12]=wc
    k_ν={core.viscosity_per_s}; C_c_m_A_c={rotation.core_inertia[2,2]-rotation.core_inertia[0,0]}
    N_MfigM_figE_1=0.; N_MfigM_figE_2=0.; N_MfigM_figE_3=0.
    N_1_LMF=0.; N_2_LMF=0.; N_3_LMF=0.
{products}
{exchange}
{writes}
    tide=ITM1(r...)/normalization; spin=ITM2(w...)/normalization
    tensor(t)=(ITM1((r+t*rd)...)+ITM2((w+t*wd)...))/normalization
    h=0.1; d=(-tensor(2h)+8tensor(h)-8tensor(-h)+tensor(-2h))/(12h)
    return vcat(vec(tide),vec(spin),vec(d),dq[4:6],dq[10:12])
end
inputs=readdlm(ARGS[1]); results=permutedims(hcat([evaluate(row) for row in eachrow(inputs)]...))
writedlm(ARGS[2],results)
"""
    code_path.write_text(declarations + "\n".join(functions) + body)
    subprocess.run([julia, "--startup-file=no", str(code_path), str(input_path), str(output_path)], check=True)
    actual, expected = np.loadtxt(output_path), np.asarray(expected)
    np.testing.assert_allclose(actual[:, :18], expected[:, :18], rtol=2e-13, atol=1e-21)
    np.testing.assert_allclose(actual[:, 18:27], expected[:, 18:27], rtol=3e-6, atol=2e-20)
    np.testing.assert_allclose(actual[:, 27:], expected[:, 27:], rtol=3e-13, atol=1e-26)
    report = {
        "cases": len(rows), "normalization": "common whole-Moon M R², SI seconds",
        "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in (inertia_path, dynamics_path)},
        "max_tidal_tensor_difference": float(np.max(abs(actual[:, :9]-expected[:, :9]))),
        "max_spin_tensor_difference": float(np.max(abs(actual[:, 9:18]-expected[:, 9:18]))),
        "max_inertia_derivative_difference_per_s": float(np.max(abs(actual[:, 18:27]-expected[:, 18:27]))),
        "max_mantle_acceleration_difference_rad_s2": float(np.max(abs(actual[:, 27:30]-expected[:, 27:30]))),
        "max_core_acceleration_difference_rad_s2": float(np.max(abs(actual[:, 30:]-expected[:, 30:]))),
        "excluded": "Earth figure mantle-vs-total inertia difference; full Julia trajectory/time-delay reconstruction",
    }
    (directory / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "compare", "reference"))
    parser.add_argument("--task", type=int)
    parser.add_argument("--julia", default="julia")
    args = parser.parse_args()
    if args.command == "run":
        run(args.task)
    elif args.command == "compare":
        compare()
    else:
        reference(args.julia)


if __name__ == "__main__":
    main()
