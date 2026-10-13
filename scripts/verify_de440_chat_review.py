"""Check DE440 accelerations at reference states, without propagating or fitting.

The ring calculation is an explicitly approximate diagnostic: a Sun-centred
44 au circle in the J2000 mean ecliptic. It is not a JPL ring-state generator.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import erfa
import numpy as np

from lunarops.classes.dynamics.earth_gravity_orientation import de440_earth_fixed2inertial_matrix
from lunarops.classes.ephemerides import body_state_matrix
from lunarops.classes.observation_factory import ensure_registered
from lunarops.classes.time import Epoch
from lunarops.config.context import RunContext
from lunarops.config.loader import build_run_plan, load_config_file
from lunarops.programs.lunar_orbit import build_lunar_propagation_problem
from lunarops.programs.registry import get_program

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output/chat_review_verification_20261011"
AU_M = 149_597_870_700.0
RING_POINT_GM = 5.5227699716988214e-13 * AU_M**3 / 86400.0**2


def ring_acceleration(position, sun, count):
    longitude = np.arange(count) * (2 * np.pi / count)
    obliquity = np.deg2rad(23.439291111)
    ring = sun + 44 * AU_M * np.column_stack(
        (np.cos(longitude), np.sin(longitude) * np.cos(obliquity),
         np.sin(longitude) * np.sin(obliquity))
    )
    delta = ring - position
    mu = RING_POINT_GM * 36 / count
    return np.sum(mu * delta / np.linalg.norm(delta, axis=1)[:, None]**3, axis=0)


def main():
    ensure_registered()
    source = ROOT / "configs/lunarops_orbit_de440.yml"
    plan = build_run_plan(load_config_file(source))
    config = get_program("LunarOrbitPropagation").spec.schema.resolve(plan.calls[0][1])
    rows = []
    with RunContext(working_dir=ROOT) as context:
        problem = build_lunar_propagation_problem(config, context)
        dynamics = problem.dynamics
        ephemeris = dynamics.ephemeris

        def reference_history(names, epoch):
            return body_state_matrix(ephemeris, names, epoch)

        earth_mu, moon_mu = dynamics.gravitational_parameters_m3_s2[:2]
        moon_fraction = earth_mu / (earth_mu + moon_mu)
        for start_jd in (2451545.0, 2440400.5):
            for days in (0.0, 1.0, 3.0, 10.0, 30.0, 365.25, 1826.25, 3650.0):
                epoch = Epoch(start_jd, 0, "tdb").shifted(days * 86400)
                earth, moon, sun = reference_history(("EARTH", "MOON", "SUN"), epoch)
                relative = moon - earth
                data = dynamics.build_epoch_data(epoch)
                components = dynamics.accelerations(epoch, relative, data, history=reference_history)
                model = components.total_accelerations_mps2[1]
                reference = ephemeris.body_acceleration_bcrs("MOON", epoch)
                terms = {name: value[1] for name, value in components.accelerations_by_force_mps2.items()}
                tide_relative = terms["tide"] / moon_fraction
                tide_correction = terms["tide"] - tide_relative
                before = model - tide_correction
                ring = ring_acceleration(moon[:3], sun[:3], 36)
                ring_earth = ring_acceleration(earth[:3], sun[:3], 36)
                residual = model - reference
                matrix = de440_earth_fixed2inertial_matrix(epoch)
                rows.append({
                    "start_jd_tdb": start_jd, "offset_days": days,
                    "reference_moon_acceleration_mps2": reference.tolist(),
                    "model_moon_acceleration_mps2": model.tolist(),
                    "force_components_mps2": {key: value.tolist() for key, value in terms.items()},
                    "residual_before_tide_fix_mps2": (before - reference).tolist(),
                    "residual_after_tide_fix_mps2": residual.tolist(),
                    "residual_before_tide_fix_norm_mps2": float(np.linalg.norm(before - reference)),
                    "residual_after_tide_fix_norm_mps2": float(np.linalg.norm(residual)),
                    "tide_relative_acceleration_mps2": tide_relative.tolist(),
                    "tide_absolute_conversion_increment_mps2": tide_correction.tolist(),
                    "approximate_ring_moon_acceleration_mps2": ring.tolist(),
                    "approximate_ring_moon_acceleration_norm_mps2": float(np.linalg.norm(ring)),
                    "approximate_ring_moon_minus_earth_acceleration_norm_mps2": float(np.linalg.norm(ring-ring_earth)),
                    "approximate_ring_36_vs_72_norm_mps2": float(np.linalg.norm(ring - ring_acceleration(moon[:3], sun[:3], 72))),
                    "residual_with_approximate_ring_norm_mps2": float(np.linalg.norm(residual + ring)),
                    "earth_matrix_orthogonality_error": float(np.max(abs(matrix.T @ matrix - np.eye(3)))),
                    "earth_matrix_determinant": float(np.linalg.det(matrix)),
                })
        report = {
            "config": str(source.relative_to(ROOT)),
            "config_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "body_count": len(dynamics.body_names),
            "earth_gm_m3_s2": float(earth_mu), "moon_gm_m3_s2": float(moon_mu),
            "relative_to_absolute_moon_fraction": float(moon_fraction),
            "erfa_version": erfa.__version__, "ring_point_gm_m3_s2": RING_POINT_GM,
            "limitations": [
                "BSP acceleration is the second derivative of its fitted position polynomial, not the original JPL RHS.",
                "All delayed states use reference ephemeris queries; this excludes integrated-history and future-extrapolation errors.",
                "The approximate ring assumes a Sun-centred circle in the J2000 mean ecliptic; no official ring trajectories are supplied.",
                "No GMSDOT increment is applied before its units and reference epoch are established.",
                "Orthogonality with real ERFA is not an independent validation of the complete JPL Earth-pole model.",
            ],
            "epochs": rows,
        }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "reference_acceleration_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print("JD start; day; residual before/after tide fix; approximate ring; with ring [m/s^2]")
    for row in rows:
        print(f"{row['start_jd_tdb']:.1f} {row['offset_days']:8.2f} "
              f"{row['residual_before_tide_fix_norm_mps2']:.6e} "
              f"{row['residual_after_tide_fix_norm_mps2']:.6e} "
              f"{row['approximate_ring_moon_acceleration_norm_mps2']:.6e} "
              f"{row['residual_with_approximate_ring_norm_mps2']:.6e}")


if __name__ == "__main__":
    main()
