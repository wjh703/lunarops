"""Measure advanced Earth-tide interpolation conditioning on exact BSP samples.

These are input extrapolation errors, not propagated trajectory errors.
No production dynamics, integration state or physical parameters are changed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lunarops.classes.dynamics.trajectory import _regular_barycentric_weights, interpolate_regular
from lunarops.classes.ephemerides import body_state_matrices, body_state_matrix
from lunarops.classes.ephemerides.calceph import CalcephEphemeris
from lunarops.classes.time import Epoch

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "output/rotation_de440_audit_20261011"


def main():
    rows = []
    with CalcephEphemeris(ROOT.parent / "data/kernels/de440",
                         lunar_relativistic_scale_convention="tdbCompatibleLunarSurface",
                         longitude_libration_correction="none") as ephemeris:
        for step in (5400., 2700., 1350.):
            for day in (30., 365.25, 1826.25):
                origin = Epoch(2440400.5, 0, "tdb").shifted(day * 86400)
                nodes = np.arange(-13, 1) * step
                target = .20982243826832672 * 86400 + step
                states = body_state_matrices(ephemeris, ("EARTH", "MOON", "SUN"),
                                             tuple(origin.shifted(float(t)) for t in nodes))
                reference = body_state_matrix(ephemeris, ("EARTH", "MOON", "SUN"), origin.shifted(target))
                for order in (3, 5, 7, 9, 13):
                    count = order + 1
                    coordinate = (target - nodes[-count]) / step
                    factors = _regular_barycentric_weights(count) / (coordinate - np.arange(count))
                    weights = factors / factors.sum()
                    row = {"step_s": step, "day": day, "order": order,
                           "future_seconds": target, "weight_l1": float(abs(weights).sum())}
                    for label, body in (("moon", 1), ("sun", 2)):
                        relative = (states[:, body] - states[:, 0]).T
                        interpolated = interpolate_regular(nodes, relative, target, order, extrapolate=True)
                        difference = interpolated[:3] - (reference[body, :3] - reference[0, :3])
                        row[f"relative_{label}_position_extrapolation_error_m"] = float(np.linalg.norm(difference))
                    rows.append(row)
    report = {"scope": "BSP sample extrapolation only; no propagated integration error",
              "reason_for_future_epoch": "Negative DE440 Earth tide orbital delay plus one RHS step",
              "rows": rows}
    (AUDIT / "tide_extrapolation_probe.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
