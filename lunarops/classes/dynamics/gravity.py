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
    source_metadata: dict | None = None

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

    def _update_degree2(self, values: np.ndarray) -> None:
        values = finite_array(values, shape=(2, 3), name="degree2_coefficients")
        if self._coeffs.shape[1] - 1 < 2:
            raise ValueError("gravity field must support degree 2")
        self._coeffs[:, 2, :3] = values


def _parse_icgem(
    path: Path, *, max_degree, gm_override, radius_override, name, allow_missing_coefficients
) -> GravityCoefficients:
    header: dict[str, str] = {}
    rows: dict[tuple[int, int], tuple[float, float]] = {}
    in_header = False
    ended_header = False
    for line_no, line in enumerate(path.read_text(encoding="ascii").splitlines(), 1):
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        fields = text.split()
        tag = fields[0].lower()
        if tag in {"gfct", "trnd", "dot", "asin", "acos"}:
            raise ValueError(f"{path}: line {line_no}: unsupported time-variable ICGEM record {tag}")
        if tag == "begin_of_head":
            if in_header or ended_header:
                raise ValueError(f"{path}: line {line_no}: duplicate ICGEM header")
            in_header = True
            continue
        if tag == "end_of_head":
            # ICGEM permits headers without an explicit begin_of_head marker.
            in_header = False
            ended_header = True
            continue
        if not ended_header:
            if in_header or tag in {
                "product_type",
                "modelname",
                "earth_gravity_constant",
                "gravity_constant",
                "radius",
                "max_degree",
                "errors",
                "norm",
                "tide_system",
                "format",
                "body",
            }:
                if len(fields) < 2 or tag in header:
                    raise ValueError(f"{path}: line {line_no}: incomplete or duplicate header field {tag}")
                header[tag] = fields[1]
            continue
        if tag != "gfc" or len(fields) < 5:
            raise ValueError(f"{path}: line {line_no}: expected static gfc coefficient")
        try:
            n, m = int(fields[1]), int(fields[2])
            c, v = (float(value.replace("D", "E").replace("d", "e")) for value in fields[3:5])
            if n < 0 or not 0 <= m <= n or not np.all(np.isfinite((c, v))):
                raise ValueError("invalid degree/order or non-finite coefficient")
            if (n, m) in rows:
                raise ValueError(f"duplicate coefficient ({n}, {m})")
            rows[n, m] = (c, v)
        except ValueError as exc:
            raise ValueError(f"{path}: line {line_no}: {exc}") from exc
    if not ended_header or not rows:
        raise ValueError(f"{path}: missing ICGEM header or static coefficients")
    if header.get("norm") != "fully_normalized":
        raise ValueError(f"{path}: norm must explicitly be fully_normalized")
    tide_system = header.get("tide_system")
    if tide_system not in {"zero_tide", "tide_free", "mean_tide"}:
        raise ValueError(f"{path}: missing or unsupported tide_system {tide_system!r}")
    try:
        declared_degree = int(header["max_degree"])
    except (KeyError, ValueError) as exc:
        raise ValueError(f"{path}: max_degree must be declared as an integer") from exc
    actual_degree = max(n for n, _ in rows)
    if declared_degree < 0 or actual_degree != declared_degree:
        raise ValueError(f"{path}: declared max_degree {declared_degree} differs from data degree {actual_degree}")
    degree = declared_degree if max_degree is None else max_degree
    if degree > declared_degree:
        raise ValueError(f"{path}: requested degree {degree} exceeds declared max_degree {declared_degree}")
    # Degree 1 vanishes in the centre-of-mass frame by contract; it can be omitted.
    expected = {(0, 0)} | {(n, m) for n in range(2, degree + 1) for m in range(n + 1)}
    missing = sorted(expected - rows.keys())
    if missing and not allow_missing_coefficients:
        raise ValueError(f"{path}: missing coefficients through degree {degree}: {missing[:20]}")
    coeffs = np.zeros((2, degree + 1, degree + 1), dtype=float)
    for (n, m), (c, v) in rows.items():
        if n <= degree:
            coeffs[:, n, m] = (c, v)

    def value(text):
        return float(text.replace("D", "E").replace("d", "e"))

    gm_text = header.get("earth_gravity_constant", header.get("gravity_constant"))
    radius_text = header.get("radius")
    if gm_text is None or radius_text is None:
        raise ValueError(f"{path}: gravity constant and radius must be declared even when overridden")
    gm = value(gm_text) if gm_override is None else gm_override
    radius = value(radius_text) if radius_override is None else radius_override
    return GravityCoefficients(
        coeffs,
        gm,
        radius,
        name or header.get("modelname", path.stem),
        {
            "path": str(path.resolve()),
            "norm": header["norm"],
            "tideSystem": tide_system,
            "declaredDegree": declared_degree,
            "loadedDegree": degree,
            "allowMissingCoefficients": allow_missing_coefficients,
            "missingCoefficients": missing,
            "gmSource": "file" if gm_override is None else "explicit override",
            "radiusSource": "file" if radius_override is None else "explicit override",
            "fileGmM3S2": value(gm_text),
            "fileRadiusM": value(radius_text),
        },
    )


def load_gravity_field(
    file_path,
    *,
    file_format="icgem",
    max_degree=None,
    gm_m3_s2=None,
    reference_radius_m=None,
    name=None,
    allow_missing_coefficients=False,
) -> GravityCoefficients:
    """Load static fully normalized ICGEM; sparse fields require explicit opt-in."""
    if file_format != "icgem":
        raise ValueError("Only ICGEM .gfc gravity files are supported")
    if max_degree is not None and (isinstance(max_degree, bool) or not isinstance(max_degree, int) or max_degree < 0):
        raise ValueError("max_degree must be a nonnegative integer")
    if not isinstance(allow_missing_coefficients, bool):
        raise TypeError("allow_missing_coefficients must be boolean")
    return _parse_icgem(
        Path(file_path),
        max_degree=max_degree,
        gm_override=None if gm_m3_s2 is None else float(gm_m3_s2),
        radius_override=None if reference_radius_m is None else float(reference_radius_m),
        name=name,
        allow_missing_coefficients=allow_missing_coefficients,
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


    def nonspherical_accelerations(self, relative_positions_m) -> np.ndarray:
        x = np.asarray(relative_positions_m, dtype=float)
        if x.ndim != 2 or x.shape[1] != 3:
            raise ValueError("relative_positions_m must have shape (N,3)")
        x = finite_array(x, shape=x.shape, name="relative_positions_m")
        if len(x) == 0:
            return np.empty((0, 3), dtype=float)
        if np.any(np.linalg.norm(x, axis=1) == 0.0):
            raise ValueError("relative_positions_m must be nonzero")
        return np.asarray(
            nonspherical_gravity_accelerations(
                np.ascontiguousarray(x),
                np.ascontiguousarray(self.coefficients.coeffs),
                self.gm_m3_s2,
                self.radius_m,
            ),
            dtype=float,
        )

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
