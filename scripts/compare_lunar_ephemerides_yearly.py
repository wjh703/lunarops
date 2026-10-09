"""Yearly DE440/INPOP21a/EPM2021 lunar relative-state comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from lunarops.classes.ephemerides import CalcephEphemeris
from lunarops.classes.time import Epoch, TimeScale


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--step-days", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    root = args.root
    dirs = {
        "DE440": root / "data/kernels/de440",
        "INPOP21a": root / "data/kernels/inpop21a",
        "EPM2021": root / "data/kernels/epm2021",
    }
    duration_days = 315360000.0 / 86400.0
    start_jd = 2451545.0
    count = int(np.floor(duration_days / args.step_days)) + 1
    epochs = [Epoch(start_jd + i * args.step_days, 0.0, TimeScale.TDB) for i in range(count)]
    providers = {
        name: CalcephEphemeris(
            path,
            lunar_relativistic_scale_convention="tdbCompatibleLunarSurface",
            longitude_libration_correction="none",
        )
        for name, path in dirs.items()
    }
    try:
        states = {name: np.empty((count, 6)) for name in providers}
        for i, epoch in enumerate(epochs):
            for name, eph in providers.items():
                earth = eph.body_state_bcrs("EARTH", epoch)
                moon = eph.body_state_bcrs("MOON", epoch)
                states[name][i] = np.r_[moon.position_m - earth.position_m, moon.velocity_mps - earth.velocity_mps]
    finally:
        for eph in providers.values():
            eph.close()
    ref = states["DE440"]
    years = np.floor(np.arange(count) * args.step_days / 365.25).astype(int) + 2000
    lines = ["year,n,INPOP21a_pos_rms_m,INPOP21a_pos_max_m,EPM2021_pos_rms_m,EPM2021_pos_max_m,INPOP21a_vel_rms_mps,EPM2021_vel_rms_mps"]
    for year in sorted(set(years)):
        mask = years == year
        row = [year, int(mask.sum())]
        for name in ("INPOP21a", "EPM2021"):
            dp = np.linalg.norm(states[name][mask, :3] - ref[mask, :3], axis=1)
            row += [float(np.sqrt(np.mean(dp * dp))), float(np.max(dp))]
        for name in ("INPOP21a", "EPM2021"):
            dv = np.linalg.norm(states[name][mask, 3:] - ref[mask, 3:], axis=1)
            row += [float(np.sqrt(np.mean(dv * dv)))]
        lines.append(",".join(str(x) for x in row))
    text = "\n".join(lines) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
