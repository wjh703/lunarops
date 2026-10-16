"""Compare the J2000 orbit regression with matched 1969-start controls."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from lunarops.fileio.numeric_table import read_numeric_table

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "output/de440_regression_epoch_audit_20261011"
OLD = ROOT / "output/regression_20261009/orbit_de440/current"


def read_case(directory):
    _, columns, values, metadata = read_numeric_table(directory / "lunar_orbit.dat", "lunarOrbit")
    times = values[:, columns.index("offset_tdb_s")]
    rtn = values[:, [columns.index(f"reference_difference_rtn_{axis}_m") for axis in "rtn"]]
    norm = np.linalg.norm(rtn, axis=1)
    daily = times % 86400 == 0
    return columns, values, times, norm, {
        "initial_tdb_jd": metadata["initialTdbJd1"] + metadata["initialTdbJd2"],
        "samples": len(values),
        "duration_days": float(times[-1] / 86400),
        "maximum_rtn_components_m": np.abs(rtn).max(axis=0).tolist(),
        "maximum_position_norm_m": float(norm.max()),
        "rms_position_norm_m": float(np.sqrt(np.mean(norm**2))),
        "maximum_at_offset_days": float(times[norm.argmax()] / 86400),
        "daily_maximum_position_norm_m": float(norm[daily].max()),
        "first_3650_days_maximum_position_norm_m": float(norm[times <= 3650 * 86400].max()),
        "all_values_finite": bool(np.all(np.isfinite(values))),
    }


def main():
    directories = {
        "old_j2000": OLD,
        "current_j2000_exact": AUDIT / "j2000_exact",
        "current_1969_epoch_only": AUDIT / "epoch_1969_only",
        "current_1969_prescribed_2700s": ROOT / "output/rotation_de440_audit_20261011/prescribed",
        "current_1969_coupled_2700s": ROOT / "output/rotation_de440_10a_full_20261011",
    }
    loaded = {name: read_case(directory) for name, directory in directories.items()}
    old_columns, old_values, *_ = loaded["old_j2000"]
    new_columns, new_values, *_ = loaded["current_j2000_exact"]
    assert old_columns == new_columns
    np.testing.assert_array_equal(old_values[:, 0], new_values[:, 0])
    old_config = yaml.safe_load((OLD / "config.yml").read_text())["programs"][0]
    new_config = yaml.safe_load((AUDIT / "j2000_exact.yml").read_text())["programs"][0]
    assert old_config == new_config
    epoch_config = yaml.safe_load((AUDIT / "epoch_1969_only.yml").read_text())["programs"][0]
    changed_keys = [key for key in old_config if old_config[key] != epoch_config[key]]
    assert changed_keys == ["initialTdbJd1"]
    _, old_acc_columns, old_acc, _ = read_numeric_table(
        OLD / "lunar_acceleration_diagnostics.dat", "lunarAccelerationDiagnostics"
    )
    _, new_acc_columns, new_acc, _ = read_numeric_table(
        AUDIT / "j2000_exact/lunar_acceleration_diagnostics.dat", "lunarAccelerationDiagnostics"
    )
    assert old_acc_columns == new_acc_columns
    coarse_columns, coarse, coarse_times, *_ = loaded["current_1969_epoch_only"]
    fine_columns, fine, fine_times, *_ = loaded["current_1969_prescribed_2700s"]
    common, coarse_indices, fine_indices = np.intersect1d(
        coarse_times, fine_times, assume_unique=True, return_indices=True
    )
    position_labels = [f"relative_{axis}_m" for axis in "xyz"]
    difference = (
        coarse[coarse_indices][:, [coarse_columns.index(label) for label in position_labels]]
        - fine[fine_indices][:, [fine_columns.index(label) for label in position_labels]]
    )
    difference_norm = np.linalg.norm(difference, axis=1)
    manifest = json.loads((ROOT / "output/regression_20261009/manifest.json").read_text())
    inputs = {}
    for path, old_hash in manifest["inputs_sha256"].items():
        if "de440" in path.lower() or "sb441" in path.lower():
            with Path(path).open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            inputs[path] = {"old_sha256": old_hash, "current_sha256": digest, "unchanged": digest == old_hash}
    report = {
        "job_id": "44962429",
        "j2000_program_config_identical": True,
        "epoch_control_changed_program_keys": changed_keys,
        "j2000_orbit_all_columns_exact": bool(np.array_equal(old_values, new_values)),
        "j2000_orbit_all_columns_maximum_absolute_difference": float(np.abs(new_values - old_values).max()),
        "j2000_acceleration_all_columns_exact": bool(np.array_equal(old_acc, new_acc)),
        "j2000_acceleration_all_columns_maximum_absolute_difference": float(np.abs(new_acc - old_acc).max()),
        "prescribed_1969_5400_vs_2700s_common_samples": {
            "samples": len(common),
            "duration_days": float(common[-1] / 86400),
            "maximum_position_difference_m": float(difference_norm.max()),
            "rms_position_difference_m": float(np.sqrt(np.mean(difference_norm**2))),
        },
        "de440_inputs": inputs,
        "cases": {name: case[-1] for name, case in loaded.items()},
    }
    (AUDIT / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    for name, (_, _, times, norm, _) in loaded.items():
        ax.plot(times / 86400, norm, label=name, linewidth=1)
    ax.set_xlabel("Days since each run's initial epoch (different calendar arcs)")
    ax.set_ylabel("Moon-relative position error against DE440 [m]")
    ax.set_title("Matched DE440 orbit regression and start-epoch controls")
    ax.grid(alpha=.25)
    ax.legend()
    fig.savefig(AUDIT / "comparison.png", dpi=180)
    fig.savefig(AUDIT / "comparison.pdf")
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
