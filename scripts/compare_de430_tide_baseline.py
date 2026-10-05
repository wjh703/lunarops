"""Compare lunarops Earth tide with a native PlanetaryEphemeris DE430 term."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from lunarops.classes.dynamics.forces import EarthTideModel, EarthTideParameters
from lunarops.classes.time import Epoch, TimeScale

AU_M = 149_597_870_700.0
DAY_S = 86_400.0
ACCELERATION_SCALE = AU_M / DAY_S**2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prefix", type=Path)
    parser.add_argument("initial_conditions", type=Path)
    parser.add_argument("earth_frame", type=Path)
    parser.add_argument("--atol", type=float, default=5e-18)
    args = parser.parse_args()

    initial = [line.split() for line in args.initial_conditions.read_text(encoding="ascii").splitlines()[1:-3]]
    by_id = {row[0]: np.asarray([float(value) for value in row[1:]], dtype=float) for row in initial}
    gm = {body_id: values[0] * AU_M**3 / DAY_S**2 for body_id, values in by_id.items()}
    current = {body_id: values[1:4] * AU_M for body_id, values in by_id.items()}

    delayed_table = np.genfromtxt(
        Path(f"{args.prefix}_delayed_states.tsv"), delimiter="\t", names=True, dtype=None, encoding="ascii"
    )
    delayed = {}
    for order in range(3):
        rows = delayed_table[delayed_table["order"] == order]
        delayed[order] = {
            str(row["body_id"]): np.asarray(list(row)[3:], dtype=float)
            * np.asarray([AU_M, AU_M, AU_M, AU_M / DAY_S, AU_M / DAY_S, AU_M / DAY_S])
            for row in rows
        }
    delay_days = np.asarray([delayed_table[delayed_table["order"] == order][0]["delay_days"] for order in range(3)])

    def history(names, epoch):
        offset_days = (float(epoch.jd1 + epoch.jd2) - 2_451_545.0)
        order = int(np.argmin(np.abs(offset_days + delay_days)))
        if abs(offset_days + delay_days[order]) > 1e-9:
            raise ValueError(f"Unexpected delayed epoch offset: {offset_days}")
        ids = {"EARTH": "399", "MOON": "301", "SUN": "10"}
        return np.asarray([delayed[order][ids[name]] for name in names])

    parameters_table = np.genfromtxt(
        Path(f"{args.prefix}_parameters.tsv"), delimiter="\t", names=True, dtype=float
    )
    p = parameters_table
    parameters = EarthTideParameters(
        k20=float(p["k20"]),
        k21=float(p["k21"]),
        k22=float(p["k22"]),
        tau_orb_days=(float(delay_days[0]), float(delay_days[1]), float(delay_days[2])),
        tau_rot_days=(float(p["tau0_rot_days"]), float(p["tau1_rot_days"]), float(p["tau2_rot_days"])),
        earth_rotation_rate_rad_s=float(p["earth_rate_rad_day"]) / DAY_S,
    )
    frame_table = np.genfromtxt(args.earth_frame, delimiter="\t", names=True, dtype=None, encoding="ascii")
    earth_row = frame_table[frame_table["body"] == "EARTH"][0]
    earth_frame = np.asarray(list(earth_row)[1:], dtype=float).reshape(3, 3)
    model = EarthTideModel(
        gm["399"],
        gm["301"],
        history,
        earth_radius_m=float(p["earth_radius_au"]) * AU_M,
        tide_parameters=parameters,
        tide_raisers=("MOON", "SUN"),
        tide_raiser_gm={"MOON": gm["301"], "SUN": gm["10"]},
    )
    actual = model.relative_acceleration(
        Epoch(2_451_545.0, 0.0, TimeScale.TDB), current["399"], current["301"], earth_frame
    )
    acceleration_table = np.genfromtxt(
        Path(f"{args.prefix}_accelerations.tsv"), delimiter="\t", names=True, dtype=None, encoding="ascii"
    )
    ids = np.asarray(acceleration_table["body_id"], dtype=str)
    native = np.column_stack([acceleration_table[f"tide_{axis}"] for axis in "xyz"]) * ACCELERATION_SCALE
    earth = int(np.flatnonzero(ids == "399")[0])
    moon = int(np.flatnonzero(ids == "301")[0])
    expected = native[moon] - native[earth]
    difference = actual - expected
    print(f"native_relative={expected}")
    print(f"lunarops_relative={actual}")
    print(
        f"delta_a_rel={difference} norm={np.linalg.norm(difference):.17e} m/s^2 "
        f"ratio={np.linalg.norm(difference) / np.linalg.norm(expected):.17e}"
    )
    if np.linalg.norm(difference) > args.atol:
        raise SystemExit("Earth tide comparison failed")
    print("PASS")


if __name__ == "__main__":
    main()
