"""Plot the cumulative DE440 lunar acceleration breakdown over a 10-year arc.

The three Cartesian components of each acceleration term are combined into
one curve by plotting its vector magnitude.  Five independent figures are
written, with force terms added in the requested order:

1. Newtonian
2. Newtonian + post-Newtonian
3. Previous terms + figure
4. Previous terms + lunar inertia
5. Previous terms + SRP + Lense-Thirring

The default input is the DE440 diagnostic file produced by LunarOps.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lunarops.fileio.numeric_table import read_numeric_table

GROUP_LABELS = {
    "newtonian": "Newtonian",
    "post_newtonian": "Post-Newtonian",
    "figure": "Figure",
    "lunar_degree2_gravity": "Lunar inertia tensor",
    "srp": "Solar radiation pressure",
    "lt": "Lense-Thirring",
}

SEQUENCES = (
    ("01_newtonian", ("newtonian",)),
    ("02_newtonian_post_newtonian", ("newtonian", "post_newtonian")),
    (
        "03_newtonian_post_newtonian_figure",
        ("newtonian", "post_newtonian", "figure"),
    ),
    (
        "04_newtonian_post_newtonian_figure_lunar_degree2_gravity",
        ("newtonian", "post_newtonian", "figure", "lunar_degree2_gravity"),
    ),
    (
        "05_all_terms",
        ("newtonian", "post_newtonian", "figure", "lunar_degree2_gravity", "srp", "lt"),
    ),
)

LABEL_OFFSETS = {
    "newtonian": 1.70,
    "post_newtonian": 1.55,
    "figure": 1.55,
    "lunar_degree2_gravity": 1.65,
    "srp": 0.72,
    "lt": 1.35,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("output/de440/lunar_acceleration_diagnostics.dat"),
        help="LunarOps acceleration diagnostics table.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/de440/acceleration_plots"),
        help="Directory for the five PNG figures.",
    )
    parser.add_argument("--dpi", type=int, default=200, help="PNG resolution (default: 200).")
    return parser.parse_args()


def _load_accelerations(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    artifact_type, columns, values, _ = read_numeric_table(path, "lunarAccelerationDiagnostics")
    if artifact_type != "lunarAccelerationDiagnostics":
        raise ValueError(f"Unexpected artifact type in {path}: {artifact_type!r}")

    column_index = {name: index for index, name in enumerate(columns)}
    time_column = "offset_tdb_s"
    if time_column not in column_index:
        raise ValueError(f"Missing {time_column!r} column in {path}")

    offsets = values[:, column_index[time_column]]
    if len(offsets) < 2 or np.any(np.diff(offsets) <= 0.0):
        raise ValueError("Diagnostic epochs must be strictly increasing")

    accelerations: dict[str, np.ndarray] = {}
    for group in GROUP_LABELS:
        names = [f"{group}_moon_{axis}_mps2" for axis in "xyz"]
        missing = [name for name in names if name not in column_index]
        if missing:
            raise ValueError(f"Missing {group} columns in {path}: {missing}")
        vector = values[:, [column_index[name] for name in names]]
        if not np.all(np.isfinite(vector)):
            raise ValueError(f"Non-finite values found in {group} acceleration")
        accelerations[group] = np.linalg.norm(vector, axis=1)

    # The propagation duration is defined as 3650 days, so this convention
    # makes the configured ten-year arc end at exactly 10 on the x axis.
    years = offsets / (365.0 * 86400.0)
    return years, accelerations


def _plot_sequence(
    years: np.ndarray,
    accelerations: dict[str, np.ndarray],
    name: str,
    groups: tuple[str, ...],
    output_path: Path,
    dpi: int,
    y_limits: tuple[float, float],
) -> None:
    figure, axis = plt.subplots(figsize=(8*0.8, 5.8*0.8), constrained_layout=True)
    plotted_lines = {}
    for group in groups:
        (line,) = axis.plot(
            years,
            accelerations[group],
            linewidth=0.9,
        )
        plotted_lines[group] = line

    axis.set_yscale("log", nonpositive="clip")
    axis.set_ylim(*y_limits)
    label_x = float(years[-1]) - 0.04
    axis.set_xlim(float(years[0]), float(years[-1]))
    axis.set_xlabel("Time since J2000 [year]", fontsize=15)
    axis.set_ylabel(r"Acceleration magnitude [m s$^{-2}$]", fontsize=15)
    axis.tick_params(axis="both", which="major", labelsize=15)
    axis.grid(True, which="major", linewidth=0.6, alpha=0.55)
    axis.grid(True, which="minor", linewidth=0.35, alpha=0.25)
    for group in groups:
        line_value = float(np.interp(label_x, years, accelerations[group]))
        axis.text(
            label_x,
            line_value * LABEL_OFFSETS[group],
            GROUP_LABELS[group],
            color=plotted_lines[group].get_color(),
            fontsize=16,
            ha="right",
            va="center",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.25},
        )
    figure.savefig(output_path, dpi=dpi, facecolor="white")
    plt.close(figure)


def _fifth_figure_limits(years: np.ndarray, accelerations: dict[str, np.ndarray]) -> tuple[float, float]:
    """Return the autoscaled log limits for the all-terms (fifth) figure."""
    all_terms = SEQUENCES[-1][1]
    figure, axis = plt.subplots()
    for group in all_terms:
        axis.plot(years, accelerations[group])
    axis.set_yscale("log", nonpositive="clip")
    limits = axis.get_ylim()
    plt.close(figure)
    return float(limits[0]), float(limits[1])


def main() -> None:
    args = _parse_args()
    years, accelerations = _load_accelerations(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    y_limits = _fifth_figure_limits(years, accelerations)

    for name, groups in SEQUENCES:
        output_path = args.output_dir / f"de440_acceleration_{name}.png"
        _plot_sequence(years, accelerations, name, groups, output_path, args.dpi, y_limits)
        print(output_path)


if __name__ == "__main__":
    main()
