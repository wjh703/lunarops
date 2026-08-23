#!/usr/bin/env python3
"""Compare DE440 residuals from the two lunar relativistic scale conventions."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Mapping

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from lunarops.fileio.observation_results import read_observation_results


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "de440_lunar_scale_convention_comparison"
DEFAULT_ALREADY_SCALED = PROJECT_ROOT / "output" / "oc_residuals_de440_alreadyScaled.txt"
DEFAULT_TDB_COMPATIBLE = (
    PROJECT_ROOT / "output" / "oc_residuals_de440_tdbCompatibleLunarSurface.txt"
)
STATION_ORDER = (
    "MCDONALD",
    "MLRS1",
    "MLRS2",
    "HALEAKALA",
    "GRASSE",
    "MATERA",
    "APOLLO",
    "WETTZELL",
)


@dataclass(frozen=True)
class Observation:
    epoch: datetime
    station_name: str
    oc_one_way_cm: float


@dataclass(frozen=True)
class Difference:
    epoch: datetime
    station_name: str
    already_scaled_cm: float
    tdb_compatible_cm: float

    @property
    def delta_cm(self) -> float:
        return self.tdb_compatible_cm - self.already_scaled_cm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--already-scaled",
        type=Path,
        default=DEFAULT_ALREADY_SCALED,
        metavar="FILE",
        help=(
            "DE440 result generated with lunarRelativisticScaleConvention: alreadyScaled "
            f"(default: {DEFAULT_ALREADY_SCALED})"
        ),
    )
    parser.add_argument(
        "--tdb-compatible-lunar-surface",
        type=Path,
        default=DEFAULT_TDB_COMPATIBLE,
        metavar="FILE",
        help=(
            "DE440 result generated with lunarRelativisticScaleConvention: tdbCompatibleLunarSurface "
            f"(default: {DEFAULT_TDB_COMPATIBLE})"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--y-limit-cm",
        type=float,
        default=None,
        metavar="CM",
        help="symmetric y-axis limit for convention differences; default uses the 99.5th percentile",
    )
    return parser.parse_args()


def parse_epoch(value: object, *, path: Path) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{path}: obs_time_utc must be text, got {value!r}")
    try:
        epoch = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{path}: invalid obs_time_utc {value!r}") from exc
    return (epoch if epoch.tzinfo is not None else epoch.replace(tzinfo=UTC)).astimezone(UTC)


def observation_key(row: Mapping[str, object], *, path: Path) -> tuple[object, ...]:
    fields = ("normal_point_index", "station_id", "reflector_id", "obs_time_utc")
    key = tuple(row.get(field) for field in fields)
    if any(value is None for value in key):
        raise ValueError(f"{path}: successful row is missing an observation identity field")
    return key


def load_observations(path: Path) -> dict[tuple[object, ...], Observation]:
    rows = read_observation_results(path)
    observations: dict[tuple[object, ...], Observation] = {}
    for row in rows:
        if row.get("status") != "ok" or row.get("light_time_converged") is not True:
            continue
        value = row.get("oc_one_way_m")
        station_name = row.get("station_name")
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            continue
        if not isinstance(station_name, str) or not station_name:
            raise ValueError(f"{path}: successful row has invalid station_name {station_name!r}")
        key = observation_key(row, path=path)
        if key in observations:
            raise ValueError(f"{path}: duplicate successful observation key {key!r}")
        observations[key] = Observation(
            epoch=parse_epoch(row["obs_time_utc"], path=path),
            station_name=station_name,
            oc_one_way_cm=float(value) * 100.0,
        )
    if not observations:
        raise ValueError(f"{path}: contains no successful finite observations")
    return observations


def align_observations(
    already_scaled: Mapping[tuple[object, ...], Observation],
    tdb_compatible: Mapping[tuple[object, ...], Observation],
) -> list[Difference]:
    common = already_scaled.keys() & tdb_compatible.keys()
    if not common:
        raise ValueError("the two result files have no successful observations in common")
    differences: list[Difference] = []
    for key in common:
        already = already_scaled[key]
        tdb = tdb_compatible[key]
        if already.epoch != tdb.epoch or already.station_name != tdb.station_name:
            raise ValueError(f"inconsistent identity metadata for observation {key!r}")
        differences.append(
            Difference(
                epoch=already.epoch,
                station_name=already.station_name,
                already_scaled_cm=already.oc_one_way_cm,
                tdb_compatible_cm=tdb.oc_one_way_cm,
            )
        )
    differences.sort(key=lambda item: item.epoch)
    return differences


def station_order(differences: list[Difference]) -> list[str]:
    stations = {item.station_name for item in differences}
    return [station for station in STATION_ORDER if station in stations] + sorted(stations - set(STATION_ORDER))


def y_limit(values: np.ndarray, configured: float | None) -> float:
    if configured is not None:
        if not math.isfinite(configured) or configured <= 0.0:
            raise ValueError("--y-limit-cm must be finite and positive")
        return configured
    percentile = float(np.percentile(np.abs(values), 99.5))
    return max(0.01, math.ceil(percentile * 100.0) / 100.0)


def summary(values: np.ndarray) -> dict[str, float | int]:
    return {
        "count": int(values.size),
        "mean_cm": float(np.mean(values)),
        "median_cm": float(np.median(values)),
        "rms_cm": float(np.sqrt(np.mean(values**2))),
        "min_cm": float(np.min(values)),
        "max_cm": float(np.max(values)),
    }


def write_summary(path: Path, differences: list[Difference]) -> None:
    by_station: dict[str, list[float]] = defaultdict(list)
    for item in differences:
        by_station[item.station_name].append(item.delta_cm)
    rows = [("ALL", summary(np.asarray([item.delta_cm for item in differences], dtype=float)))]
    rows.extend((station, summary(np.asarray(by_station[station], dtype=float))) for station in station_order(differences))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("station", "count", "mean_cm", "median_cm", "rms_cm", "min_cm", "max_cm"),
        )
        writer.writeheader()
        for station, values in rows:
            writer.writerow({"station": station, **values})


def render_all(differences: list[Difference], output_path: Path) -> None:
    epochs = [item.epoch for item in differences]
    already = np.asarray([item.already_scaled_cm for item in differences], dtype=float)
    tdb = np.asarray([item.tdb_compatible_cm for item in differences], dtype=float)
    delta = tdb - already
    limit = y_limit(delta, None)
    dates = mdates.date2num(epochs)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
    axes[0].scatter(dates, already, s=2, alpha=0.25, label="alreadyScaled", color="#d62728", rasterized=True)
    axes[0].scatter(
        dates,
        tdb,
        s=2,
        alpha=0.25,
        label="tdbCompatibleLunarSurface",
        color="#1f77b4",
        rasterized=True,
    )
    axes[0].axhline(0.0, color="black", linewidth=0.7)
    axes[0].set_ylabel("O-C (cm)")
    axes[0].set_title("DE440 residuals by lunar relativistic scale convention")
    axes[0].legend(loc="upper right", frameon=True)
    axes[1].scatter(dates, delta, s=2, alpha=0.35, color="#2ca02c", rasterized=True)
    axes[1].axhline(0.0, color="black", linewidth=0.7)
    axes[1].set_ylim(-limit, limit)
    axes[1].set_ylabel("TDB-compatible - already-scaled (cm)")
    axes[1].set_xlabel("Observation epoch (UTC)")
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.xaxis.set_major_locator(mdates.YearLocator(base=5))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def render_station(
    station: str,
    differences: list[Difference],
    output_path: Path,
    configured_y_limit: float | None,
) -> None:
    data = [item for item in differences if item.station_name == station]
    dates = mdates.date2num([item.epoch for item in data])
    delta = np.asarray([item.delta_cm for item in data], dtype=float)
    limit = y_limit(delta, configured_y_limit)
    stats = summary(delta)
    fig, ax = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
    ax.scatter(dates, delta, s=3, alpha=0.4, color="#2ca02c", rasterized=True)
    ax.axhline(0.0, color="black", linewidth=0.7)
    ax.set_ylim(-limit, limit)
    ax.set_title(
        f"{station}: tdbCompatibleLunarSurface - alreadyScaled "
        f"(N={stats['count']:,}, RMS={stats['rms_cm']:.4f} cm)"
    )
    ax.set_ylabel("One-way O-C difference (cm)")
    ax.set_xlabel("Observation epoch (UTC)")
    ax.xaxis.set_major_locator(mdates.YearLocator(base=5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, alpha=0.25)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    already_scaled = load_observations(args.already_scaled)
    tdb_compatible = load_observations(args.tdb_compatible_lunar_surface)
    differences = align_observations(already_scaled, tdb_compatible)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"aligned {len(differences):,} observations; "
        f"alreadyScaled-only={len(already_scaled) - len(differences):,}; "
        f"tdbCompatibleLunarSurface-only={len(tdb_compatible) - len(differences):,}"
    )
    all_delta = np.asarray([item.delta_cm for item in differences], dtype=float)
    print("ALL", summary(all_delta))
    for station in station_order(differences):
        values = np.asarray([item.delta_cm for item in differences if item.station_name == station], dtype=float)
        print(station, summary(values))
        render_station(
            station,
            differences,
            args.output_dir / f"delta_{station.lower()}.png",
            args.y_limit_cm,
        )
    render_all(differences, args.output_dir / "de440_lunar_scale_conventions.png")
    write_summary(args.output_dir / "summary.csv", differences)
    print(f"wrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
