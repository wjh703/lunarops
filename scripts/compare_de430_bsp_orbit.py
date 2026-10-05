"""Compare a LunarOps DE430 propagation with DE430 BSP Earth-Moon states."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from lunarops.classes.ephemerides import CalcephEphemeris
from lunarops.classes.time import Epoch, TimeScale
from lunarops.fileio.numeric_table import read_numeric_table

J2000_TDB_JD = 2_451_545.0


def _state(values: np.ndarray, columns: tuple[str, ...], body: str) -> np.ndarray:
    labels = tuple(f"{body}_{name}" for name in ("x_m", "y_m", "z_m", "vx_mps", "vy_mps", "vz_mps"))
    return values[:, [columns.index(label) for label in labels]]


def _metrics(error: np.ndarray) -> dict[str, object]:
    position_norm = np.linalg.norm(error[:, :3], axis=1)
    velocity_norm = np.linalg.norm(error[:, 3:], axis=1)
    return {
        "initial_position_m": float(position_norm[0]),
        "final_position_m": float(position_norm[-1]),
        "maximum_position_m": float(np.max(position_norm)),
        "rms_position_m": float(np.sqrt(np.mean(position_norm**2))),
        "final_velocity_mps": float(velocity_norm[-1]),
        "maximum_velocity_mps": float(np.max(velocity_norm)),
        "rms_velocity_mps": float(np.sqrt(np.mean(velocity_norm**2))),
        "final_position_xyz_m": error[-1, :3].tolist(),
        "final_velocity_xyz_mps": error[-1, 3:].tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("lunarops_orbit", type=Path)
    parser.add_argument("kernel_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()

    _, columns, values, _ = read_numeric_table(args.lunarops_orbit, "lunarOrbit")
    offsets = values[:, columns.index("offset_tdb_s")]
    if np.any(np.diff(offsets) <= 0.0):
        raise ValueError("Orbit output epochs must be strictly increasing")
    integrated_relative = _state(values, columns, "relative")

    start = time.perf_counter()
    ephemeris = CalcephEphemeris(
        args.kernel_directory,
        lunar_relativistic_scale_convention="alreadyScaled",
        longitude_libration_correction="none",
    )
    try:
        states = {}
        for name in ("EARTH", "MOON"):
            states[name] = np.asarray(
                [
                    np.r_[state.position_m, state.velocity_mps]
                    for state in (
                        ephemeris.body_state_bcrs(
                            name,
                            Epoch(J2000_TDB_JD, float(offset) / 86400.0, TimeScale.TDB),
                        )
                        for offset in offsets
                    )
                ]
            )
    finally:
        ephemeris.close()

    bsp_relative = states["MOON"] - states["EARTH"]
    error = integrated_relative - bsp_relative
    radial = bsp_relative[:, :3] / np.linalg.norm(bsp_relative[:, :3], axis=1)[:, None]
    normal = np.cross(bsp_relative[:, :3], bsp_relative[:, 3:])
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    transverse = np.cross(normal, radial)
    rtn = np.column_stack([np.einsum("ij,ij->i", error[:, :3], axis) for axis in (radial, transverse, normal)])

    report = {
        "reference": "DE430 BSP, CALCEPH, SSB-centered Earth/Moon states",
        "sample_count": len(offsets),
        "duration_days": float(offsets[-1] / 86400.0),
        "comparison_seconds": time.perf_counter() - start,
        "relative_state_error": _metrics(error),
        "relative_rtn_position_error_m": {
            "final": rtn[-1].tolist(),
            "maximum_absolute": np.max(np.abs(rtn), axis=0).tolist(),
            "rms": np.sqrt(np.mean(rtn**2, axis=0)).tolist(),
        },
    }
    args.output_directory.mkdir(parents=True, exist_ok=True)
    report_path = args.output_directory / "de430_bsp_comparison.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="ascii")
    print(json.dumps(report, indent=2))
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
