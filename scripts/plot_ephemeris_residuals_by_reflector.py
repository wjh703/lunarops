#!/usr/bin/env python3
"""Compare ephemeris residuals and matched differences by lunar reflector."""

from __future__ import annotations

import argparse
import math
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from lunarops.fileio.observation_results import read_observation_results


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "output"
DEFAULT_RESULTS = {
    "INPOP21a": OUTPUT_DIR / "oc_residuals_inpop21a.txt",
    "DE440": OUTPUT_DIR / "oc_residuals_de440.txt",
    "EPM2021": OUTPUT_DIR / "oc_residuals_epm2021.txt",
}
REFLECTOR_ORDER = ("APOLLO11", "APOLLO14", "APOLLO15", "LUNOKHOD1", "LUNOKHOD2")
COLORS = {"INPOP21a": "#1f77b4", "DE440": "#d62728", "EPM2021": "#2ca02c"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for label, default in DEFAULT_RESULTS.items():
        parser.add_argument(f"--{label.lower()}", type=Path, default=default, metavar="FILE")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR / "ephemeris_residual_comparison" / "reflectors",
    )
    parser.add_argument("--residual-limit-cm", type=float, default=None, metavar="CM")
    parser.add_argument("--difference-limit-cm", type=float, default=None, metavar="CM")
    return parser.parse_args()


def parse_epoch(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"invalid observation epoch {value!r}")
    epoch = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return (epoch if epoch.tzinfo is not None else epoch.replace(tzinfo=UTC)).astimezone(UTC)


def observation_key(row: dict[str, object]) -> tuple[object, ...]:
    return (
        row.get("source"),
        row.get("normal_point_index"),
        row.get("station_id"),
        row.get("reflector_id"),
    )


def load_results(path: Path) -> dict[str, dict[tuple[object, ...], tuple[datetime, float]]]:
    grouped: dict[str, dict[tuple[object, ...], tuple[datetime, float]]] = {}
    for row in read_observation_results(path):
        if row.get("status") != "ok" or row.get("light_time_converged") is not True:
            continue
        value = row.get("oc_one_way_m")
        reflector = row.get("reflector_name")
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            continue
        if not isinstance(reflector, str) or not reflector:
            continue
        grouped.setdefault(reflector, {})[observation_key(row)] = (
            parse_epoch(row.get("obs_time_utc")),
            float(value) * 100.0,
        )
    if not grouped:
        raise ValueError(f"{path} contains no successful finite observations")
    return grouped


def symmetric_limit(values: np.ndarray, configured: float | None) -> float:
    if configured is not None:
        if not math.isfinite(configured) or configured <= 0.0:
            raise ValueError("plot limits must be finite and positive")
        return configured
    return max(1.0, math.ceil(float(np.percentile(np.abs(values), 99.5)) / 5.0) * 5.0)


def matched_difference(
    candidate: dict[tuple[object, ...], tuple[datetime, float]],
    reference: dict[tuple[object, ...], tuple[datetime, float]],
) -> tuple[list[datetime], np.ndarray]:
    keys = sorted(candidate.keys() & reference.keys(), key=lambda key: candidate[key][0])
    return (
        [candidate[key][0] for key in keys],
        np.asarray([candidate[key][1] - reference[key][1] for key in keys], dtype=float),
    )


def render_reflector(
    reflector: str,
    results: dict[str, dict[str, dict[tuple[object, ...], tuple[datetime, float]]]],
    output_path: Path,
    residual_limit_cm: float | None,
    difference_limit_cm: float | None,
) -> None:
    available = {label: grouped[reflector] for label, grouped in results.items() if reflector in grouped}
    residual_values = np.asarray([value for rows in available.values() for _, value in rows.values()])
    residual_limit = symmetric_limit(residual_values, residual_limit_cm)

    reference = available.get("INPOP21a")
    differences: dict[str, tuple[list[datetime], np.ndarray]] = {}
    if reference is not None:
        for label in ("DE440", "EPM2021"):
            if label in available:
                differences[label] = matched_difference(available[label], reference)
    difference_values = np.concatenate([values for _, values in differences.values()])
    difference_limit = symmetric_limit(difference_values, difference_limit_cm)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
    for label, rows in available.items():
        ordered = sorted(rows.values(), key=lambda item: item[0])
        epochs = [item[0] for item in ordered]
        values = np.asarray([item[1] for item in ordered])
        axes[0].scatter(epochs, values, s=2, alpha=0.3, color=COLORS[label], label=label, rasterized=True)
        print(
            f"{reflector} {label}: N={len(values):,}, mean={np.mean(values):.3f} cm, "
            f"RMS={np.sqrt(np.mean(values**2)):.3f} cm, median={np.median(values):.3f} cm"
        )

    for label, (epochs, values) in differences.items():
        axes[1].scatter(
            epochs,
            values,
            s=2,
            alpha=0.35,
            color=COLORS[label],
            label=f"{label} - INPOP21a",
            rasterized=True,
        )
        print(
            f"{reflector} {label}-INPOP21a: N={len(values):,}, mean={np.mean(values):.3f} cm, "
            f"RMS={np.sqrt(np.mean(values**2)):.3f} cm, median={np.median(values):.3f} cm, "
            f"min={np.min(values):.3f} cm, max={np.max(values):.3f} cm"
        )

    axes[0].set_title(f"{reflector} LLR residual comparison")
    axes[0].set_ylabel("O-C (cm)")
    axes[0].set_ylim(-residual_limit, residual_limit)
    axes[1].set_ylabel("Matched difference (cm)")
    axes[1].set_xlabel("Observation epoch (UTC)")
    axes[1].set_ylim(-difference_limit, difference_limit)
    axes[1].xaxis.set_major_locator(mdates.YearLocator(base=5))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for axis in axes:
        axis.axhline(0.0, color="black", linewidth=0.7)
        axis.grid(True, alpha=0.25)
        axis.legend(loc="upper right")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"wrote {output_path}")


def main() -> int:
    args = parse_args()
    paths = {"INPOP21a": args.inpop21a, "DE440": args.de440, "EPM2021": args.epm2021}
    results = {label: load_results(path) for label, path in paths.items()}
    reflectors = set().union(*(grouped.keys() for grouped in results.values()))
    ordered = [name for name in REFLECTOR_ORDER if name in reflectors]
    ordered.extend(sorted(reflectors - set(ordered)))
    for reflector in ordered:
        render_reflector(
            reflector,
            results,
            args.output_dir / f"oc_residuals_{reflector.lower()}.png",
            args.residual_limit_cm,
            args.difference_limit_cm,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
