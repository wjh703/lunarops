"""Measure DE430 Earth-tide error caused by ABMD delayed-state extrapolation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from lunarops.classes.dynamics.earth_gravity_orientation import de430_earth_fixed2inertial_matrix
from lunarops.classes.dynamics.forces import EarthTideModel, EarthTideParameters
from lunarops.classes.dynamics.integrators import _interpolate_regular
from lunarops.classes.ephemerides import CalcephEphemeris, body_state_matrix
from lunarops.classes.time import Epoch, TimeScale

DAY_S = 86_400.0
J2000 = Epoch(2_451_545.0, 0.0, TimeScale.TDB)
PARAMETERS = EarthTideParameters(
    k20=0.335,
    k21=0.32,
    k22=0.32,
    tau_orb_days=(0.0640, -0.044, -0.1000),
    tau_rot_days=(0.0, 0.007363219022804189, 0.002535297863338872),
    earth_rotation_rate_rad_s=7.29211514670698e-5,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel-dir", type=Path, default=Path("../data/kernels/de430"))
    parser.add_argument("--years", type=float, default=10.0)
    parser.add_argument("--sample-days", type=float, default=10.0)
    parser.add_argument("--step-seconds", type=float, default=5400.0)
    parser.add_argument("--order", type=int, default=13)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _state(ephemeris: CalcephEphemeris, names, epoch) -> np.ndarray:
    return body_state_matrix(ephemeris, tuple(names), epoch)


def _approximate_provider(ephemeris, evaluation_epoch, step_s, order, history_lag_steps):
    endpoint = evaluation_epoch.shifted(-history_lag_steps * step_s)
    offsets = step_s * np.arange(-order, 1, dtype=float)
    nodes = np.column_stack(
        [
            _state(ephemeris, ("EARTH", "MOON"), endpoint.shifted(float(offset))).reshape(-1)
            for offset in offsets
        ]
    )

    def provider(names, epoch):
        query = float(endpoint.seconds_until(epoch))
        integrated = _interpolate_regular(offsets, nodes, query, order, extrapolate=True).reshape(2, 6)
        by_name = {"EARTH": integrated[0], "MOON": integrated[1]}
        external = [name for name in names if name not in by_name]
        external_states = dict(
            zip(external, _state(ephemeris, tuple(external), epoch), strict=True)
        ) if external else {}
        by_name.update(external_states)
        return np.asarray([by_name[name] for name in names])

    return provider, endpoint


def _summarize(values: np.ndarray) -> tuple[float, float, float]:
    return float(np.max(values)), float(np.sqrt(np.mean(values**2))), float(np.median(values))


def main() -> None:
    args = _arguments()
    ephemeris = CalcephEphemeris(
        args.kernel_dir,
        lunar_relativistic_scale_convention="alreadyScaled",
        longitude_libration_correction="none",
    )
    try:
        # Use the DE430 GMs carried by the native J2000 initial-condition file.
        ic = Path(__file__).parents[2] / "PlanetaryEphemeris.jl/data/de430ic_2000Jan1.txt"
        rows = [line.split() for line in ic.read_text(encoding="ascii").splitlines()[1:-3]]
        au_m = 149_597_870_700.0
        gm = {row[0]: float(row[1]) * au_m**3 / DAY_S**2 for row in rows}
        model = EarthTideModel(
            gm["399"],
            gm["301"],
            lambda names, epoch: body_state_matrix(ephemeris, tuple(names), epoch),
            earth_inertial2fixed_matrix_provider=lambda epoch: de430_earth_fixed2inertial_matrix(epoch).T,
            earth_radius_m=6_378_136.3,
            tide_parameters=PARAMETERS,
            tide_raisers=("MOON", "SUN"),
            tide_raiser_gm={"MOON": gm["301"], "SUN": gm["10"]},
        )
        duration_s = args.years * 365.25 * DAY_S
        sample_step_s = args.sample_days * DAY_S
        sample_offsets = np.arange(DAY_S, duration_s + 0.5 * sample_step_s, sample_step_s)
        records = []
        for history_lag_steps in (0, 1):
            acceleration_errors: list[float] = []
            relative_state_errors: list[list[float]] = [[], [], []]
            endpoint_extrapolation_hours: list[list[float]] = [[], [], []]
            for offset in sample_offsets:
                epoch = J2000.shifted(float(offset))
                current = _state(ephemeris, ("EARTH", "MOON"), epoch)
                frame = de430_earth_fixed2inertial_matrix(epoch).T
                exact = model.relative_acceleration(epoch, current[0, :3], current[1, :3], frame)
                approximate_history, endpoint = _approximate_provider(
                    ephemeris, epoch, args.step_seconds, args.order, history_lag_steps
                )
                approximate = model.relative_acceleration(
                    epoch, current[0, :3], current[1, :3], frame, history=approximate_history
                )
                acceleration_errors.append(np.linalg.norm(approximate - exact))
                for tide_order, delay_days in enumerate(PARAMETERS.tau_orb_days):
                    delayed_epoch = epoch.shifted(-delay_days * DAY_S)
                    estimated = approximate_history(("EARTH", "MOON"), delayed_epoch)
                    truth = _state(ephemeris, ("EARTH", "MOON"), delayed_epoch)
                    error = (estimated[1, :3] - estimated[0, :3]) - (truth[1, :3] - truth[0, :3])
                    relative_state_errors[tide_order].append(np.linalg.norm(error))
                    endpoint_extrapolation_hours[tide_order].append(
                        endpoint.seconds_until(delayed_epoch) / 3600.0
                    )
            acceleration_error_array = np.asarray(acceleration_errors)
            maximum, rms, median = _summarize(acceleration_error_array)
            acceleration_summary = maximum, rms, median
            mode = "accepted-output" if history_lag_steps == 0 else "PECEC-RHS"
            print(
                f"{mode}: samples={len(sample_offsets)} tide_accel_error_mps2 "
                f"max={maximum:.17e} rms={rms:.17e} median={median:.17e}"
            )
            for tide_order, errors in enumerate(relative_state_errors):
                error_array = np.asarray(errors)
                maximum, rms, median = _summarize(error_array)
                hours = endpoint_extrapolation_hours[tide_order][0]
                print(
                    f"  order={tide_order} tau_days={PARAMETERS.tau_orb_days[tide_order]:+.6f} "
                    f"query_from_history_endpoint_h={hours:+.6f} rel_position_error_m "
                    f"max={maximum:.9e} rms={rms:.9e} median={median:.9e}"
                )
                records.append((mode, tide_order, hours, maximum, rms, median, *acceleration_summary))
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("w", encoding="ascii") as stream:
                stream.write(
                    "mode\torder\tquery_from_endpoint_h\tmax_position_m\trms_position_m\t"
                    "median_position_m\tmax_acceleration_mps2\trms_acceleration_mps2\t"
                    "median_acceleration_mps2\n"
                )
                for record in records:
                    stream.write("\t".join(map(str, record)) + "\n")
    finally:
        ephemeris.close()


if __name__ == "__main__":
    main()
