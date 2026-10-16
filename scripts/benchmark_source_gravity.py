"""Benchmark real DE440 epoch batches and assess a circular KBO approximation.

Run each thread count in a fresh process, e.g. OMP_NUM_THREADS=32
OMP_WAIT_POLICY=PASSIVE OPENBLAS_NUM_THREADS=1 python -m
scripts.benchmark_source_gravity --output output/source_gravity_32.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np
import scipy.integrate

from lunarops._dynamics_core import openmp_enabled, point_mass_symmetric, point_mass_symmetric_batch
from lunarops.classes.observation_factory import ensure_registered
from lunarops.classes.time import Epoch
from lunarops.config.context import RunContext
from lunarops.config.loader import build_run_plan, load_config_file
from lunarops.programs.lunar_orbit import build_lunar_propagation_problem
from lunarops.programs.registry import get_program


def median_seconds(function, repeats):
    function()
    samples = []
    for _ in range(repeats):
        start = perf_counter()
        function()
        samples.append(perf_counter() - start)
    return float(np.median(samples))


def benchmark(config_path, repeats, integration_days, ring_study, scalar_cache=False, ring_step_study=False):
    ensure_registered()
    config = get_program("LunarOrbitPropagation").spec.schema.resolve(
        build_run_plan(load_config_file(config_path)).calls[0][1]
    )
    if not ring_study:
        config["durationSeconds"] = max(30, integration_days) * 86400.0
    context = RunContext(working_dir=Path.cwd())
    solves = []
    original_solve = scipy.integrate.solve_ivp

    def timed_solve(*args, **kwargs):
        start = perf_counter()
        result = original_solve(*args, **kwargs)
        solves.append({"wall_s": perf_counter() - start, "nfev": result.nfev, "steps": len(result.t) - 1})
        return result

    try:
        start = perf_counter()
        with patch.object(scipy.integrate, "solve_ivp", timed_solve):
            problem = build_lunar_propagation_problem(config, context)
        dynamics = problem.dynamics
        report = {
            "threads": os.environ.get("OMP_NUM_THREADS"),
            "openmp_enabled": openmp_enabled(),
            "build_problem_s": perf_counter() - start,
            "ring_solve_ivp": solves,
            "body_count": len(dynamics.body_names),
            "batch_epochs": 64,
            "scalar_cache": scalar_cache,
        }
        epochs = tuple(problem.initial_epoch.shifted(i * config["stepSeconds"]) for i in range(64))
        data = dynamics.build_epoch_data_batch(epochs)
        positions = np.ascontiguousarray([item.positions_m[2:] for item in data])
        gm = np.ascontiguousarray([item.gravitational_parameters_m3_s2[2:] for item in data])

        def scalar():
            return tuple(point_mass_symmetric(r, mu) for r, mu in zip(positions, gm, strict=True))

        reference = scalar()
        actual = point_mass_symmetric_batch(positions, gm)
        for index, (acceleration, potential) in enumerate(reference):
            np.testing.assert_array_equal(actual[0][index], acceleration)
            np.testing.assert_array_equal(actual[1][index], potential)
        report["kernel_bitwise_equal"] = True
        report["scalar_kernel_batch_s"] = median_seconds(scalar, repeats)
        report["native_kernel_batch_s"] = median_seconds(lambda: point_mass_symmetric_batch(positions, gm), repeats)
        report["epoch_batch_s"] = median_seconds(lambda: dynamics.build_epoch_data_batch(epochs), repeats)
        if integration_days:
            if scalar_cache:
                def cache_batch(positions, integrated_body_count, gravitational_parameters_m3_s2):
                    return tuple(
                        dynamics.force_group.build_point_mass_gravity_cache(r, integrated_body_count, mu)
                        for r, mu in zip(positions, gravitational_parameters_m3_s2, strict=True)
                    )

                dynamics.force_group.build_point_mass_gravity_cache_batch = cache_batch
            start = perf_counter()
            trajectory = problem.integrator.integrate(
                dynamics, problem.initial_epoch, problem.initial_vector, integration_days * 86400.0,
                seed_duration_s=problem.seed_duration_s,
                external_history_provider=problem.external_history,
            )
            report["integration_days"] = integration_days
            report["integration_s"] = perf_counter() - start
            report["integration_final_state"] = trajectory.vector(problem.initial_epoch.shifted(integration_days * 86400.0)).tolist()
            sample_states = np.asarray([
                trajectory.vector(problem.initial_epoch.shifted(float(offset)))
                for offset in np.linspace(0, integration_days * 86400.0, 301)
            ])
            report["integration_301_sample_sha256"] = hashlib.sha256(sample_states.tobytes()).hexdigest()
        if ring_study:
            belt = dynamics.kuiper_belt_ring
            points_error, velocity_error, force_error = [], [], []
            ring_samples = []
            for offset in np.linspace(0, problem.duration_s, 101):
                epoch = problem.initial_epoch.shifted(float(offset))
                sun, moon = dynamics.ephemeris.body_state_matrix_bcrs(("SUN", "MOON"), epoch)
                numerical_r, numerical_v = belt.states(epoch, sun[:3], sun[3:])
                ring_samples.append((epoch, sun, moon, numerical_r, numerical_v))
                circular_r, circular_v = belt.circular_states(epoch, sun[:3], sun[3:])
                points_error.append(float(np.max(np.linalg.norm(numerical_r - circular_r, axis=1))))
                velocity_error.append(float(np.max(np.linalg.norm(numerical_v - circular_v, axis=1))))

                def force(r, target=moon[:3]):
                    delta = r - target
                    return np.sum(belt.point_gm_m3_s2 * delta / np.linalg.norm(delta, axis=1)[:, None]**3, axis=0)

                force_error.append(float(np.linalg.norm(force(numerical_r) - force(circular_r))))
            report["circular_ring_comparison"] = {
                "duration_s": problem.duration_s,
                "samples": 101,
                "max_point_position_difference_m": max(points_error),
                "max_point_velocity_difference_mps": max(velocity_error),
                "max_moon_acceleration_difference_mps2": max(force_error),
                "free_double_integral_scale_m": 0.5 * max(force_error) * problem.duration_s**2,
                "note": "Acceleration-only diagnostic; double-integral scale is not a coupled orbit error bound. EIH not compared here.",
            }
            if ring_step_study:
                ring_steps = []
                source_count = len(dynamics.body_names) - belt.point_count
                lower = min(0.0, belt._backward_solution.t_min if belt._backward_solution is not None else 0.0)
                upper = max(0.0, belt._solution.t_max if belt._solution is not None else 0.0)
                reference = Epoch(belt.reference_jd_tdb, 0.0, "tdb")
                for days in (28, 56):
                    alternate = replace(belt)
                    evaluations = []

                    def larger_step_solve(*args, cap_seconds=days * 86400.0, counts=evaluations, **kwargs):
                        kwargs["max_step"] = cap_seconds
                        result = original_solve(*args, **kwargs)
                        counts.append(result.nfev)
                        return result

                    start = perf_counter()
                    with patch.object(scipy.integrate, "solve_ivp", larger_step_solve):
                        alternate.prepare(
                            dynamics.ephemeris, dynamics.body_names[:source_count],
                            dynamics.gravitational_parameters_m3_s2[:source_count],
                            reference.shifted(lower), upper - lower, dynamics.solar_gm_variation,
                        )
                    elapsed = perf_counter() - start
                    differences, acceleration_differences = [], []
                    for epoch, sun, moon, numerical_r, _ in ring_samples:
                        candidate_r, _ = alternate.states(epoch, sun[:3], sun[3:])
                        differences.append(float(np.max(np.linalg.norm(candidate_r - numerical_r, axis=1))))
                        delta = candidate_r - moon[:3]
                        baseline_delta = numerical_r - moon[:3]
                        candidate_force = np.sum(belt.point_gm_m3_s2 * delta / np.linalg.norm(delta, axis=1)[:, None]**3, axis=0)
                        baseline_force = np.sum(belt.point_gm_m3_s2 * baseline_delta / np.linalg.norm(baseline_delta, axis=1)[:, None]**3, axis=0)
                        acceleration_differences.append(float(np.linalg.norm(candidate_force - baseline_force)))
                    ring_steps.append({
                        "max_step_days": days, "prepare_s": elapsed, "nfev": sum(evaluations),
                        "max_point_position_difference_m": max(differences),
                        "max_moon_acceleration_difference_mps2": max(acceleration_differences),
                        "note": "Compared to 14-day cap at 101 epochs; no coupled integration/EIH validation.",
                    })
                report["ring_max_step_comparison"] = ring_steps
        return report
    finally:
        context.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/lunarops_orbit_rotation_de440_environment_10a.yml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--integration-days", type=float, default=0)
    parser.add_argument("--ring-study", action="store_true")
    parser.add_argument("--ring-step-study", action="store_true", help="Also compare 28/56-day DOP853 caps")
    parser.add_argument("--scalar-cache", action="store_true", help="Use scalar source caches during integration as a control")
    args = parser.parse_args()
    report = benchmark(
        args.config, args.repeats, args.integration_days, args.ring_study or args.ring_step_study,
        args.scalar_cache, args.ring_step_study,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
