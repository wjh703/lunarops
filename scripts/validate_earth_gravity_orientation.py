"""Audit ERFA substitutions against saved matrices and published DE pole equations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import erfa
import numpy as np
from scipy.spatial.transform import Rotation

from lunarops.classes.dynamics import earth_gravity_orientation as orientation
from lunarops.classes.time import Epoch
from lunarops.fileio.numeric_table import read_numeric_table


def angles(days, model):
    t = days / 36525
    rate, a, b = (6962890.549, -17.206262, 9.205348) if model == "de430" else (6962890.539, -17.1996, 9.2025)
    omega = (450160.280 - rate*t + 7.455*t*t + .008*t**3) * erfa.DAS2R
    eps = (84381.448 - 46.815*t - .00059*t*t + .001813*t**3) * erfa.DAS2R
    return eps, a*erfa.DAS2R*np.sin(omega), b*erfa.DAS2R*np.cos(omega)


def active(axis, angle):
    vectors = np.zeros((len(angle), 3))
    vectors[:, axis] = angle
    return Rotation.from_rotvec(vectors)


def paper_poles(days, model):
    """Closed-form pole of date, then independent active-rotation composition."""
    eps, dp, de = angles(days, model)
    poles = np.column_stack((np.sin(dp)*np.sin(eps+de),
                            np.cos(dp)*np.sin(eps+de)*np.cos(eps)-np.cos(eps+de)*np.sin(eps),
                            np.cos(dp)*np.sin(eps+de)*np.sin(eps)+np.cos(eps+de)*np.cos(eps)))
    years, t = days/365.25, days/36525
    if model == "de430":
        x = (.005675420332289347 + .0002768991557448355*years)*erfa.DAS2R
        y = (-.01702265691498953 - .001211859121655924*years)*erfa.DAS2R
        poles = active(0, x).apply(active(1, y).apply(poles))
        zeta = (2306.2181*t+.30188*t*t+.017998*t**3)*erfa.DAS2R
        theta = (2004.3109*t-.42665*t*t-.041833*t**3)*erfa.DAS2R
        z = (2306.2181*t+1.09468*t*t+.018203*t**3)*erfa.DAS2R
        return active(2, -zeta).apply(active(1, theta).apply(active(2, -z).apply(poles)))
    x = np.full(len(days), .005343991604414805*erfa.DAS2R)
    y = np.full(len(days), -.017128824840317532*erfa.DAS2R)
    poles = active(0, x).apply(active(1, y).apply(poles))
    n, k = erfa.ltpequ(2000+years), erfa.ltpecl(2000+years)
    equinox = np.cross(n, k); equinox /= np.linalg.norm(equinox, axis=1)[:, None]
    y_axis = np.cross(n, equinox); y_axis /= np.linalg.norm(y_axis, axis=1)[:, None]
    return np.einsum("nij,nj->ni", np.stack((equinox, y_axis, n), axis=2), poles)


def pole_separation(a, b):
    return np.arctan2(np.linalg.norm(np.cross(a, b), axis=1), np.einsum("ij,ij->i", a, b))


def validate(root):
    before = np.load(root / "before_matrices.npz")
    dates = before["tdb_jd"]
    days = dates - 2451545
    report = {"samples_per_model": len(dates), "julian_epoch_range": [float(2000+days.min()/365.25), float(2000+days.max()/365.25)],
              "erfa_version": erfa.__version__, "models": {},
              "time_scale": "TDB argument mandated by JPL; ERFA routines used as mathematical kernels without TT conversion"}
    for model in ("de430", "de440"):
        provider = getattr(orientation, model + "_earth_fixed2inertial_matrix")
        current = np.array([provider(Epoch(2451545., float(day), "tdb")) for day in days])
        reference = paper_poles(days, model)
        pole_error = pole_separation(current[:, :, 2], reference)
        entry = {"published_pole_max_vector_difference": float(np.max(np.linalg.norm(current[:, :, 2]-reference, axis=1))),
                 "published_pole_max_angle_rad": float(pole_error.max()),
                 "old_vs_new_max_matrix_component": float(abs(current-before[model]).max()),
                 "old_vs_new_max_pole_angle_arcsec": float(pole_separation(current[:, :, 2], before[model][:, :, 2]).max()/erfa.DAS2R),
                 "max_orthogonality_component": float(abs(current.transpose(0, 2, 1)@current-np.eye(3)).max()),
                 "max_det_minus_one": float(abs(np.linalg.det(current)-1).max())}
        assert pole_error.max() < 2e-15
        # Isolate ERFA library substitution from correcting the DE430 coefficients.
        eps, dp, de = angles(days, "de440")
        nutation = erfa.numat(eps, dp, de)
        if model == "de430":
            x = (.005675420332289347+.0002768991557448355*days/365.25)*erfa.DAS2R
            y = (-.01702265691498953-.001211859121655924*days/365.25)*erfa.DAS2R
            correction = erfa.ry(y, erfa.rx(x, np.broadcast_to(np.eye(3), (len(days), 3, 3))))
            erfa_only = (nutation@correction@erfa.pmat76(2451545., days)).transpose(0, 2, 1)
            entry["erfa_substitution_only_max_matrix_component"] = float(abs(erfa_only-before[model]).max())
            entry["printed_nutation_correction_max_pole_angle_arcsec"] = float(pole_separation(current[:, :, 2], erfa_only[:, :, 2]).max()/erfa.DAS2R)
            assert entry["erfa_substitution_only_max_matrix_component"] < 2e-15
        else:
            entry["erfa_substitution_only_max_matrix_component"] = entry["old_vs_new_max_matrix_component"]
            assert entry["old_vs_new_max_matrix_component"] < 2e-15
        full = erfa.numat(eps, *erfa.nut80(2451545., days))
        model_eps, model_dp, model_de = angles(days, model)
        model_nutation = erfa.numat(model_eps, model_dp, model_de)
        entry["full_nut80_vs_jpl_single_term_max_pole_arcsec"] = float(pole_separation(full[:, 2], model_nutation[:, 2]).max()/erfa.DAS2R)
        report["models"][model] = entry
    report["erfa_sources"] = [{"url": f"https://github.com/liberfa/erfa/blob/v2.0.1/src/{name}.c",
                               "sha256": hashlib.sha256((root/f"erfa_{name}.c").read_bytes()).hexdigest()}
                              for name in ("ltp", "pmat76", "numat", "obl80", "nut80")]
    manifest = json.loads((root/"source_manifest.json").read_text())
    report["source_manifest_mismatches"] = [p for p, digest in manifest.items()
                                           if hashlib.sha256(Path(p).read_bytes()).hexdigest() != digest]
    assert not report["source_manifest_mismatches"]
    (root/"matrix_validation.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


def compare(root, years):
    paths = [("de430", root/"de430"/"before", root/"de430"/"after"),
             ("de440", root/"de440"/"before", root/"de440"/"after")]
    if years:
        paths = [("de430", root/"de430_before_10a", root/"de430_after_10a"),
                 ("de440", root.parent/"de440_pavlov_eq16_fix_20261012/de440_10a", root/"de440_after_10a")]
    report = {}
    for model, before, after in paths:
        summary = {}
        for name, kind in (("lunar_orbit.dat", "lunarOrbit"), ("lunar_rotation.dat", "lunarRotation"),
                           ("lunar_acceleration_diagnostics.dat", "lunarAccelerationDiagnostics")):
            old = read_numeric_table(before/name, kind); new = read_numeric_table(after/name, kind)
            assert old[1] == new[1]
            np.testing.assert_array_equal(old[2][:, 0], new[2][:, 0])
            columns = new[1]; difference = new[2]-old[2]
            entry = {"rows": len(new[2]), "finite": bool(np.isfinite(new[2]).all()),
                     "max_abs_difference_by_column": dict(zip(columns, np.max(abs(difference), axis=0).tolist(), strict=True))}
            if kind == "lunarOrbit":
                indices = [columns.index(f"relative_{a}_m") for a in "xyz"]
                entry["max_relative_position_difference_m"] = float(np.linalg.norm(difference[:, indices], axis=1).max())
            if kind == "lunarRotation":
                indices = [columns.index(f"q_{a}") for a in "xyzw"]
                error = (Rotation.from_quat(old[2][:, indices]).inv()*Rotation.from_quat(new[2][:, indices])).as_rotvec()
                entry["max_attitude_angle_difference_rad"] = float(np.linalg.norm(error, axis=1).max())
                entry["max_attitude_surface_difference_m"] = entry["max_attitude_angle_difference_rad"]*1738000
            summary[name] = entry
        if years:
            summary["reference_before"] = json.loads((before/f"{model}_reference_comparison.json").read_text())
            summary["reference_after"] = json.loads((after/f"{model}_reference_comparison.json").read_text())
        report[model] = summary
    (root/("ten_year_regression.json" if years else "short_regression.json")).write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({m: {name: {k: v for k, v in r.items() if k.startswith(("max_relative", "max_attitude"))}
                          for name, r in s.items() if name.endswith(".dat")} for m, s in report.items()}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--compare", choices=("short", "ten_year"))
    args = parser.parse_args()
    if args.compare:
        compare(args.root, args.compare == "ten_year")
    else:
        validate(args.root)
