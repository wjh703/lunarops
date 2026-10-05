"""Spherical-harmonic gravity fields and ICGEM coefficient loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lunarops._dynamics_core import nonspherical_gravity_accelerations
from lunarops.base.array_validation import finite_array


@dataclass(slots=True)
class GravityCoefficients:
    """Fully normalized ``4pi`` spherical-harmonic coefficients.

    ``coeffs[0, n, m]`` stores Cnm and ``coeffs[1, n, m]`` stores Snm.
    Coefficients are expressed with ``csphase=1`` and include C00=1.
    """

    _coeffs: np.ndarray
    gm_m3_s2: float
    radius_m: float
    name: str | None = None

    def __post_init__(self) -> None:
        coeffs = np.asarray(self._coeffs, dtype=float)
        if coeffs.ndim != 3 or coeffs.shape[0] != 2 or coeffs.shape[1] != coeffs.shape[2]:
            raise ValueError("coeffs must have shape (2, degree+1, degree+1)")
        if not np.all(np.isfinite(coeffs)):
            raise ValueError("coeffs must be finite")
        degree = coeffs.shape[1] - 1
        if not np.isclose(coeffs[0, 0, 0], 1.0, rtol=0.0, atol=1e-15):
            raise ValueError("coeffs must use C00=1")
        if degree >= 1 and np.max(np.abs(coeffs[:, 1, :2])) > 1e-15:
            raise ValueError("gravity fields must not contain degree-1 terms")
        gm = float(self.gm_m3_s2)
        radius = float(self.radius_m)
        if not np.isfinite(gm) or gm <= 0.0:
            raise ValueError("gm_m3_s2 must be positive and finite")
        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError("radius_m must be positive and finite")
        self._coeffs = np.ascontiguousarray(coeffs.copy())
        self.gm_m3_s2 = gm
        self.radius_m = radius

    @property
    def coeffs(self) -> np.ndarray:
        result = self._coeffs.view()
        result.setflags(write=False)
        return result

    @property
    def normalization(self) -> str:
        return "4pi"

    @property
    def csphase(self) -> int:
        return 1

    def copy(self) -> GravityCoefficients:
        return GravityCoefficients(self._coeffs, self.gm_m3_s2, self.radius_m, self.name)

    def _update_degree2(self, values: np.ndarray) -> None:
        values = finite_array(values, shape=(2, 3), name="degree2_coefficients")
        if self._coeffs.shape[1] - 1 < 2:
            raise ValueError("gravity field must support degree 2")
        self._coeffs[:, 2, :3] = values


def _parse_icgem(path: Path, *, max_degree: int | None, gm_override: float | None, radius_override: float | None, name: str | None) -> GravityCoefficients:
    header: dict[str, str] = {}
    rows: list[tuple[int, int, float, float]] = []
    in_header = False
    for line in path.read_text(encoding="ascii").splitlines():
        text = line.strip()
        if not text:
            continue
        if text.startswith("begin_of_head"):
            in_header = True
            continue
        if text.startswith("end_of_head"):
            in_header = False
            continue
        fields = text.split()
        if in_header:
            if len(fields) >= 2:
                header[fields[0].lower()] = fields[1]
            continue
        if fields[0].lower() not in {"gfc", "gfct"} or len(fields) < 5:
            continue
        degree, order = int(fields[1]), int(fields[2])
        if max_degree is None or degree <= max_degree:
            rows.append((degree, order, float(fields[3]), float(fields[4])))
    if not rows:
        raise ValueError(f"No ICGEM gfc coefficients found in {path}")
    degree = max(row[0] for row in rows)
    coeffs = np.zeros((2, degree + 1, degree + 1), dtype=float)
    for n, m, cosine, sine in rows:
        coeffs[0, n, m] = cosine
        coeffs[1, n, m] = sine
    gm_text = header.get("earth_gravity_constant", header.get("gravity_constant"))
    radius_text = header.get("radius")
    gm = float(gm_override) if gm_override is not None else (float(gm_text) if gm_text is not None else None)
    radius = float(radius_override) if radius_override is not None else (float(radius_text) if radius_text is not None else None)
    if gm is None or radius is None:
        raise ValueError(f"ICGEM file {path} must define gravity constant and radius")
    field_name = name if name is not None else header.get("modelname", path.stem)
    return GravityCoefficients(coeffs, gm, radius, field_name)


def load_gravity_field(
    file_path, *, file_format="icgem", max_degree=None, gm_m3_s2=None, reference_radius_m=None, name=None
) -> GravityCoefficients:
    """Load fully normalized ICGEM coefficients without an external gravity package."""
    if str(file_format).lower() not in {"icgem", "gfc"}:
        raise ValueError("Only ICGEM .gfc gravity files are supported")
    return _parse_icgem(
        Path(file_path),
        max_degree=None if max_degree is None else int(max_degree),
        gm_override=None if gm_m3_s2 is None else float(gm_m3_s2),
        radius_override=None if reference_radius_m is None else float(reference_radius_m),
        name=name,
    )


class GravityField:
    """Cartesian adapter for a fully normalized non-spherical gravity field."""

    def __init__(self, coefficients: GravityCoefficients):
        if not isinstance(coefficients, GravityCoefficients):
            raise TypeError("coefficients must be a GravityCoefficients object")
        self.coefficients = coefficients

    @property
    def gm_m3_s2(self) -> float:
        return self.coefficients.gm_m3_s2

    @property
    def radius_m(self) -> float:
        return self.coefficients.radius_m

    def update_degree2_coefficients(self, coefficients) -> None:
        values = finite_array(coefficients, shape=(2, 3), name="degree2_coefficients")
        self.coefficients._update_degree2(values)

    def nonspherical_acceleration(self, relative_position_m) -> np.ndarray:
        x = finite_array(relative_position_m, shape=(3,), name="relative_position_m")
        if np.linalg.norm(x) == 0.0:
            raise ValueError("relative_position_m must be nonzero")
        return self.nonspherical_accelerations(x.reshape(1, 3))[0]

    def nonspherical_accelerations(self, relative_positions_m) -> np.ndarray:
        x = np.asarray(relative_positions_m, dtype=float)
        if x.ndim != 2 or x.shape[1] != 3:
            raise ValueError("relative_positions_m must have shape (N,3)")
        x = finite_array(x, shape=x.shape, name="relative_positions_m")
        if len(x) == 0:
            return np.empty((0, 3), dtype=float)
        if np.any(np.linalg.norm(x, axis=1) == 0.0):
            raise ValueError("relative_positions_m must be nonzero")
        return np.asarray(nonspherical_gravity_accelerations(
            np.ascontiguousarray(x),
            np.ascontiguousarray(self.coefficients.coeffs),
            self.gm_m3_s2,
            self.radius_m,
        ), dtype=float)

    # Internal aliases are intentionally absent; callers use the explicit
    # non-spherical terminology.


def make_gravity_coefficients(cosine, sine, *, gm_m3_s2, radius_m, name=None) -> GravityCoefficients:
    """Construct fully normalized ``4pi`` coefficients with C00=1."""
    c = np.asarray(cosine, dtype=float)
    s = np.asarray(sine, dtype=float)
    if c.ndim != 2 or c.shape != s.shape or c.shape[0] != c.shape[1]:
        raise ValueError("cosine and sine must be square arrays of the same shape")
    if not np.all(np.isfinite(c)) or not np.all(np.isfinite(s)):
        raise ValueError("cosine and sine coefficients must be finite")
    coeffs = np.stack((c, s))
    return GravityCoefficients(coeffs, gm_m3_s2, radius_m, name)


def make_j2_coefficients(*, gm_m3_s2, radius_m, j2, name=None) -> GravityCoefficients:
    """Return a degree-2 gravity field equivalent to an axisymmetric J2."""
    if not np.isfinite(j2):
        raise ValueError("j2 must be finite")
    cosine = np.zeros((3, 3), dtype=float)
    sine = np.zeros_like(cosine)
    cosine[0, 0] = 1.0
    cosine[2, 0] = -float(j2) / np.sqrt(5.0)
    return make_gravity_coefficients(
        cosine,
        sine,
        gm_m3_s2=gm_m3_s2,
        radius_m=radius_m,
        name="axisymmetric J2" if name is None else name,
    )


def ensure_gravity_field(value: GravityField | GravityCoefficients) -> GravityField:
    return value if isinstance(value, GravityField) else GravityField(value)
