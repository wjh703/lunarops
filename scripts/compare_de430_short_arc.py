"""Compare LunarOps and PlanetaryEphemeris.jl DE430 short-arc states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from lunarops.fileio.numeric_table import read_numeric_table

STATE_NAMES = ("earth", "moon", "emb", "relative")


def _state(table: np.ndarray, columns: tuple[str, ...], prefix: str) -> np.ndarray:
    names = (
        f"{prefix}_x_m",
        f"{prefix}_y_m",
        f"{prefix}_z_m",
        f"{prefix}_vx_mps",
        f"{prefix}_vy_mps",
        f"{prefix}_vz_mps",
    )
    return table[:, [columns.index(name) for name in names]]


def _metrics(error: np.ndarray) -> dict[str, float | list[float]]:
    position_norm = np.linalg.norm(error[:, :3], axis=1)
    velocity_norm = np.linalg.norm(error[:, 3:], axis=1)
    return {
        "initial_position_norm_m": float(position_norm[0]),
        "final_position_norm_m": float(position_norm[-1]),
        "maximum_position_norm_m": float(np.max(position_norm)),
        "rms_position_norm_m": float(np.sqrt(np.mean(position_norm**2))),
        "initial_velocity_norm_mps": float(velocity_norm[0]),
        "final_velocity_norm_mps": float(velocity_norm[-1]),
        "maximum_velocity_norm_mps": float(np.max(velocity_norm)),
        "rms_velocity_norm_mps": float(np.sqrt(np.mean(velocity_norm**2))),
        "final_cartesian_state_error": error[-1].tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("lunarops_orbit", type=Path)
    parser.add_argument("planetary_ephemeris", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    _, lunar_columns, lunar_values, _ = read_numeric_table(args.lunarops_orbit, "lunarOrbit")
    planetary = np.genfromtxt(args.planetary_ephemeris, delimiter="\t", names=True)
    planetary_columns = tuple(planetary.dtype.names or ())
    planetary_values = np.column_stack([planetary[name] for name in planetary_columns])

    lunar_offsets = lunar_values[:, lunar_columns.index("offset_tdb_s")]
    planetary_offsets = planetary_values[:, planetary_columns.index("offset_tdb_s")]
    if lunar_offsets.shape != planetary_offsets.shape or not np.allclose(
        lunar_offsets, planetary_offsets, rtol=0.0, atol=1e-8
    ):
        raise ValueError("The two integrations do not have identical output epochs")

    errors: dict[str, np.ndarray] = {}
    report: dict[str, Any] = {
        "reference": "PlanetaryEphemeris.jl DE430!",
        "duration_days": float(lunar_offsets[-1] / 86400.0),
        "sample_count": len(lunar_offsets),
        "states": {},
    }
    for name in STATE_NAMES:
        lunar_state = _state(lunar_values, lunar_columns, name)
        planetary_state = _state(planetary_values, planetary_columns, name)
        errors[name] = lunar_state - planetary_state
        report["states"][name] = _metrics(errors[name])

    reference_relative = _state(planetary_values, planetary_columns, "relative")
    radial = reference_relative[:, :3] / np.linalg.norm(reference_relative[:, :3], axis=1)[:, None]
    normal = np.cross(reference_relative[:, :3], reference_relative[:, 3:])
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    transverse = np.cross(normal, radial)
    relative_rtn = np.column_stack(
        [
            np.einsum("ij,ij->i", errors["relative"][:, :3], axis)
            for axis in (radial, transverse, normal)
        ]
    )
    report["relative_rtn"] = {
        "final_m": relative_rtn[-1].tolist(),
        "maximum_absolute_m": np.max(np.abs(relative_rtn), axis=0).tolist(),
        "rms_m": np.sqrt(np.mean(relative_rtn**2, axis=0)).tolist(),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "short_arc_comparison.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="ascii")

    columns = ["offset_tdb_s"]
    blocks = [lunar_offsets]
    for name in STATE_NAMES:
        columns.extend(f"{name}_{component}_error" for component in ("x_m", "y_m", "z_m", "vx_mps", "vy_mps", "vz_mps"))
        blocks.extend(errors[name].T)
    columns.extend(("relative_r_error_m", "relative_t_error_m", "relative_n_error_m"))
    blocks.extend(relative_rtn.T)
    np.savetxt(
        args.output_dir / "short_arc_state_errors.tsv",
        np.column_stack(blocks),
        delimiter="\t",
        header="\t".join(columns),
        comments="",
        fmt="%.17e",
    )

    try:
        import matplotlib.pyplot as plt

        days = lunar_offsets / 86400.0
        figure, axes = plt.subplots(2, 1, figsize=(8.5, 7.0), sharex=True, constrained_layout=True)
        axes[0].plot(days, relative_rtn[:, 0], label="Radial")
        axes[0].plot(days, relative_rtn[:, 1], label="Transverse")
        axes[0].plot(days, relative_rtn[:, 2], label="Normal")
        axes[0].set_ylabel("Position error (m)")
        axes[0].grid(alpha=0.3)
        axes[0].legend()
        for name in STATE_NAMES:
            axes[1].plot(days, np.linalg.norm(errors[name][:, :3], axis=1), label=name)
        axes[1].set_yscale("symlog", linthresh=1e-6)
        axes[1].set_xlabel("TDB days since J2000")
        axes[1].set_ylabel("3D position error (m)")
        axes[1].grid(alpha=0.3)
        axes[1].legend()
        figure.savefig(args.output_dir / "short_arc_position_errors.png", dpi=180)
        plt.close(figure)
    except ImportError:
        pass

    relative = report["states"]["relative"]
    print(f"samples={len(lunar_offsets)} duration_days={lunar_offsets[-1] / 86400.0:.6f}")
    print(f"relative_final_position_m={relative['final_position_norm_m']:.17e}")
    print(f"relative_max_position_m={relative['maximum_position_norm_m']:.17e}")
    print(f"relative_final_velocity_mps={relative['final_velocity_norm_mps']:.17e}")
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
