"""Controlled static-field and mechanical-inertia hypotheses for DE440."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lunarops.classes.dynamics.inertia import _degree2_coefficients_from_inertia
from lunarops.classes.dynamics.rotation import LunarRotationDynamics
from lunarops.classes.observation_factory import ensure_registered
from lunarops.config.context import RunContext
from lunarops.config.loader import build_run_plan, load_config_file
from lunarops.programs.lunar_orbit import lunar_orbit_propagation
from lunarops.programs.registry import get_program

C21 = 2.2779569194479859e-10
S21 = 1.3343450358392620e-9
MODES = ("baseline", "c21", "s21", "both", "negative", "mechanical")


def inertia_offset():
    result = np.zeros((3, 3))
    result[0, 2] = result[2, 0] = -C21
    result[1, 2] = result[2, 1] = -S21
    return result


def field_offset(mode):
    result = _degree2_coefficients_from_inertia(inertia_offset())
    if mode == "c21":
        result[1, 1] = 0
    elif mode == "s21":
        result[0, 1] = 0
    elif mode == "negative":
        result *= -1
    elif mode in ("baseline", "mechanical"):
        result.fill(0)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--mode", choices=MODES, default="both")
    args = parser.parse_args()
    from lunarops.programs import lunar_orbit

    original_build = lunar_orbit._build_rotation

    def build_rotation(config, fields, moon_mu, earth_mu):
        rotation, initial = original_build(config, fields, moon_mu, earth_mu)
        assert rotation.deformation.field is fields["MOON"]
        for torque in rotation.torque_group.models:
            if hasattr(torque, "field"):
                assert torque.field is fields["MOON"]
        if args.mode == "mechanical":
            total = rotation.total_inertia + inertia_offset()
            total.setflags(write=False)
            rotation.deformation.undistorted_inertia = total
            rotation = LunarRotationDynamics(
                total, rotation.radius_m, rotation.torque_group,
                core=rotation.core, geodetic_precession=rotation.geodetic_precession,
                deformation=rotation.deformation,
            )
        report = {
            "mode": args.mode,
            "unnormalized_constants": {"C21M": C21, "S21M": S21},
            "independent_normalized_field_offset": field_offset(args.mode).tolist(),
            "undistorted_inertia": rotation.total_inertia.tolist(),
            "deformation_base_matches_mechanics": bool(np.array_equal(
                rotation.total_inertia, rotation.deformation.undistorted_inertia)),
            "core_inertia": rotation.core_inertia.tolist(),
            "gravity_force_and_figure_torques_use_assembly_moon_field": True,
            "initial_state_unchanged": initial.tolist(),
            "scope": "controlled diagnostic hypothesis; production defaults unchanged",
        }
        output = Path(config["outputFileRotation"]).parent
        output.mkdir(parents=True, exist_ok=True)
        (output / "degree2_experiment.json").write_text(json.dumps(report, indent=2) + "\n")
        return rotation, initial

    lunar_orbit._build_rotation = build_rotation
    ensure_registered()
    plan = build_run_plan(load_config_file(args.config))
    config = get_program("LunarOrbitPropagation").spec.schema.resolve(plan.calls[0][1])
    parameters = config["lunarRotation"]["deformation"]
    sign = -1 if args.mode == "negative" else 1
    parameters["c21Static"] = sign * C21 if args.mode in ("c21", "both", "negative") else 0.0
    parameters["s21Static"] = sign * S21 if args.mode in ("s21", "both", "negative") else 0.0
    parameters["s22Static"] = 0.0
    with RunContext(working_dir=Path(__file__).resolve().parents[1]) as context:
        lunar_orbit_propagation(config, context)


if __name__ == "__main__":
    main()
