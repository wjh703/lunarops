"""Build the Earth zonal gravity field and the J2(t) polynomial of the lunar dynamics model.

    python scripts/build_earth_zonal_field.py GOCO2025s.gfc gsfc_slr_c20_long_term_rl07.txt OUTPUT.gfc
        [--k20 0.335] [--max-degree 5]

The lunar orbit feels only the zonal Earth field (the tesseral terms average out within a day, as in DE440,
Park et al. 2021). The field written here combines:

- J2 from the NASA GSFC SLR C20 series (Loomis et al. 2025): the TSVD MM estimates plus the AOD1B 28-day means
  (the total C20 signal) are fitted, with equal weights, by J2(T) = J2 + a T + b T^2 plus annual and semiannual
  terms, T in Julian years from J2000 (the parabola of the JPL ephemerides and of the IAU 2006J2 precession,
  Liu & Huang 2025). The printed a and b are linearPerYear and quadraticPerYear2 of earthJ2TimeVariation; their
  uncertainties are 3 sigma scaled by the post-fit scatter, which is mostly geophysical and correlated, so they
  are optimistic.
- C30 ... C(max-degree)0 from GOCO2025s (Oehlinger et al. 2025): its static term plus its trend at J2000.

Both sources give zero-tide C20, which includes the permanent deformation by the mean Sun and Moon. The tide
model of the integration (k20 raised by the Moon and the Sun) adds that deformation itself, so the field is
written tide-free with the same k20: C20_tf = C20_zt - A0 H0 k20 (IERS Conventions 2010, Eq. 6.14), with
A0 = 4.4228e-8 and H0 = -0.31460 m. The higher zonals have no permanent-tide part.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

A0, H0 = 4.4228e-8, -0.31460
J2000_MJD = 51544.5
DAYS_PER_YEAR = 365.25


def read_goco(path: Path) -> tuple[dict, dict]:
    """Header and the zonal (static, trend, t0 MJD) terms of an ICGEM time-variable field."""
    header, zonal = {}, {}
    in_data = False
    for line in path.read_text().splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "end_of_head":
            in_data = True
        elif not in_data and fields[0] in ("earth_gravity_constant", "radius", "tide_system", "norm", "modelname"):
            header[fields[0]] = fields[1]
        elif in_data and fields[0] in ("gfc", "gfct", "trnd") and fields[2] == "0":
            n = int(fields[1])
            term = zonal.setdefault(n, {"static": 0.0, "trend": 0.0, "t0": J2000_MJD})
            if fields[0] == "trnd":
                term["trend"] = float(fields[3])
            else:
                term["static"] = float(fields[3])
                if fields[0] == "gfct":
                    y, m, d = fields[7][:4], fields[7][4:6], fields[7][6:]
                    term["t0"] = float(np.datetime64(f"{y}-{m}-{d}").astype("datetime64[D]").astype(int)) + 40587.0
    if header.get("tide_system") != "zero_tide" or header.get("norm") != "fully_normalized":
        raise SystemExit(f"{path}: expected a fully normalized zero-tide field")
    return header, zonal


def fit_j2(path: Path):
    """Parabola of the total GSFC J2 about J2000: (J2, a, b), their 3-sigma errors, rms, span."""
    rows = [line.split() for line in path.read_text().splitlines() if line[:2] in ("19", "20") and line[4] == "."]
    data = np.array(rows, dtype=float)
    year, c20 = data[:, 0], data[:, 4] + data[:, 7] * 1e-10
    j2 = -np.sqrt(5.0) * c20
    t = year - 2000.0
    w = 2 * np.pi * t
    design = np.column_stack([np.ones_like(t), t, t**2, np.cos(w), np.sin(w), np.cos(2 * w), np.sin(2 * w)])
    x, *_ = np.linalg.lstsq(design, j2, rcond=None)
    residual = j2 - design @ x
    variance = residual @ residual / (len(t) - design.shape[1])
    sigma = np.sqrt(np.diag(np.linalg.inv(design.T @ design)) * variance)
    return x[:3], 3 * sigma[:3], np.sqrt(variance), (year[0], year[-1])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("goco", type=Path)
    p.add_argument("c20", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--k20", type=float, default=0.335)
    p.add_argument("--max-degree", type=int, default=5)
    a = p.parse_args()
    header, zonal = read_goco(a.goco)
    (j2_zero_tide, rate, quadratic), (e_j2, e_rate, e_quadratic), rms, span = fit_j2(a.c20)
    c20_tide_free = -j2_zero_tide / np.sqrt(5.0) - A0 * H0 * a.k20
    coefficients = {2: c20_tide_free}
    for n in range(3, a.max_degree + 1):
        term = zonal[n]
        coefficients[n] = term["static"] + term["trend"] * (J2000_MJD - term["t0"]) / DAYS_PER_YEAR
    lines = [
        "Earth zonal field of the LunarOps lunar dynamics model (scripts/build_earth_zonal_field.py).",
        (
            f"C20: tide-free (k20 = {a.k20}) J2 at J2000 of a parabola fitted to {a.c20.name}"
            f" ({span[0]:.2f}-{span[1]:.2f}); zero-tide J2 {j2_zero_tide:.10e} +- {e_j2:.1e} (3 sigma)."
        ),
        (
            f"J2(T) = J2 + linearPerYear T + quadraticPerYear2 T^2: linearPerYear {rate:.4e} +- {e_rate:.1e},"
            f" quadraticPerYear2 {quadratic:.4e} +- {e_quadratic:.1e} (3 sigma), post-fit rms {rms:.2e}."
        ),
        f"C30-C{a.max_degree}0: {header['modelname']} static plus trend at J2000.",
        "",
        "begin_of_head ================================================================",
        "product_type            gravity_field",
        "body                    earth",
        "modelname               LUNAROPS_EARTH_ZONAL",
        f"earth_gravity_constant  {header['earth_gravity_constant']}",
        f"radius                  {header['radius']}",
        f"max_degree              {a.max_degree}",
        "errors                  no",
        "norm                    fully_normalized",
        "tide_system             tide_free",
        "end_of_head ==================================================================",
        f"gfc 0 0  {1.0: .16e}  {0.0: .16e}  0.0  0.0",
    ]
    lines += [f"gfc {n} 0 {c: .16e}  {0.0: .16e}  0.0  0.0" for n, c in coefficients.items()]
    a.output.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[1:4]))
    print(f"earthJ2TimeVariation: linearPerYear {rate:.4e}, quadraticPerYear2 {quadratic:.4e}")
    print(f"wrote {a.output}")


if __name__ == "__main__":
    main()
