"""Plot ten-year LunarOps orbit errors against the DE430/DE440 ephemerides.

One figure is written for each ephemeris.  Each figure contains:

1. Radial, transverse, and normal position errors;
2. Position-error magnitude;
3. Velocity-error magnitude.

The two figures deliberately use identical axes limits so their error levels
can be compared directly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lunarops.fileio.numeric_table import read_numeric_table

POSITION_RTN_LIMIT_M = (-0.2, 0.2)
POSITION_NORM_LIMIT_M = (1.0e-6, 0.2)
VELOCITY_NORM_LIMIT_UMPS = (1.0e-6, 0.4)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--de430",
        type=Path,
        default=Path("output/de430/lunar_orbit.dat"),
        help="DE430 LunarOps orbit table.",
    )
    parser.add_argument(
        "--de440",
        type=Path,
        default=Path("output/de440/lunar_orbit.dat"),
        help="DE440 LunarOps orbit table.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/ephemeris_comparison"),
        help="Directory for the comparison figures.",
    )
    parser.add_argument("--dpi", type=int, default=200, help="PNG resolution (default: 200).")
    return parser.parse_args()


def _load_errors(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    artifact_type, columns, values, _ = read_numeric_table(path, "lunarOrbit")
    if artifact_type != "lunarOrbit":
        raise ValueError(f"Unexpected artifact type in {path}: {artifact_type!r}")

    column_index = {name: index for index, name in enumerate(columns)}

    def column(name: str) -> np.ndarray:
        if name not in column_index:
            raise ValueError(f"Missing {name!r} in {path}")
        return values[:, column_index[name]]

    offsets = column("offset_tdb_s")
    if len(offsets) < 2 or np.any(np.diff(offsets) <= 0.0):
        raise ValueError(f"Orbit epochs must be strictly increasing: {path}")

    rtn = np.column_stack([column(f"reference_difference_rtn_{axis}_m") for axis in "rtn"])
    velocity_xyz = np.column_stack(
        [column(f"reference_difference_velocity_{axis}_mps") for axis in "xyz"]
    )
    if not np.all(np.isfinite(rtn)) or not np.all(np.isfinite(velocity_xyz)):
        raise ValueError(f"Non-finite ephemeris differences found in {path}")

    years = offsets / (365.0 * 86400.0)
    position_norm = np.linalg.norm(rtn, axis=1)
    velocity_norm_umps = np.linalg.norm(velocity_xyz, axis=1) * 1.0e6
    return years, rtn, np.column_stack([position_norm, velocity_norm_umps])


def _plot_one(
    ephemeris_name: str,
    years: np.ndarray,
    rtn: np.ndarray,
    norms: np.ndarray,
    output_path: Path,
    dpi: int,
) -> None:
    figure, axes = plt.subplots(
        3,
        1,
        figsize=(8.5*0.9, 9.0*0.9),
        sharex=True,
        constrained_layout=True,
    )

    colors = ("tab:blue", "tab:orange", "tab:green")
    labels = ("Radial", "Transverse", "Normal")
    for index, (color, label) in enumerate(zip(colors, labels)):
        axes[0].plot(years, rtn[:, index], color=color, linewidth=0.85, label=label)
    axes[0].axhline(0.0, color="black", linewidth=0.6, alpha=0.65)
    axes[0].set_ylim(*POSITION_RTN_LIMIT_M)
    axes[0].set_ylabel("RTN position error [m]", fontsize=14)
    axes[0].legend(loc="upper left", frameon=False, fontsize=14, ncol=3)

    position_norm = np.ma.masked_less_equal(norms[:, 0], 0.0)
    axes[1].semilogy(years, position_norm, color="tab:purple", linewidth=0.9)
    axes[1].set_ylim(*POSITION_NORM_LIMIT_M)
    axes[1].set_ylabel("Position error norm [m]", fontsize=14)

    velocity_norm = np.ma.masked_less_equal(norms[:, 1], 0.0)
    axes[2].semilogy(years, velocity_norm, color="tab:red", linewidth=0.9)
    axes[2].set_ylim(*VELOCITY_NORM_LIMIT_UMPS)
    axes[2].set_ylabel(r"Velocity error norm [$\mu$m s$^{-1}$]", fontsize=14)
    axes[2].set_xlabel("Time since J2000 [year]", fontsize=14)

    for axis in axes:
        axis.tick_params(axis="both", which="major", labelsize=14)
        axis.grid(True, which="major", linewidth=0.6, alpha=0.55)
        axis.grid(True, which="minor", linewidth=0.35, alpha=0.25)
        axis.set_xlim(float(years[0]), float(years[-1]))

    # Keep the ephemeris identity visible without adding a figure title.
    axes[0].text(
        0.99,
        0.94,
        ephemeris_name,
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        fontsize=14,
        fontweight="bold",
    )
    figure.savefig(output_path, dpi=dpi, facecolor="white")
    plt.close(figure)


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for ephemeris_name, input_path in (("DE430", args.de430), ("DE440", args.de440)):
        years, rtn, norms = _load_errors(input_path)
        output_path = args.output_dir / f"{ephemeris_name.lower()}_ephemeris_comparison.png"
        _plot_one(ephemeris_name, years, rtn, norms, output_path, args.dpi)
        print(output_path)


if __name__ == "__main__":
    main()
