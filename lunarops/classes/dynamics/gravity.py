"""Spherical-harmonic gravity fields and ICGEM coefficient loading.

Coefficients are fully normalised (``4π`` normalisation, Condon-Shortley
phase excluded) and stored as ``coeffs[0, n, m] = C̄nm`` and
``coeffs[1, n, m] = S̄nm`` with ``C̄00 = 1``. Accelerations are evaluated by
the native kernel :func:`lunarops._dynamics_core.nonspherical_gravity_accelerations`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lunarops._dynamics_core import nonspherical_gravity_accelerations


@dataclass(slots=True)
class GravityCoefficients:
    """Validated fully normalised spherical-harmonic coefficients.

    Attributes
    ----------
    coeffs : numpy.ndarray
        Coefficients of shape ``(2, L + 1, L + 1)`` with ``[0, n, m] = C̄nm``
        and ``[1, n, m] = S̄nm``, ``4π`` normalisation, no Condon-Shortley
        phase (``csphase=1``), ``C̄00 = 1`` and no degree-1 terms. They are
        passed as the first constructor argument (``_coeffs``) and stored as
        a contiguous private copy.
    gm_m3_s2 : float
        Gravitational parameter of the body in m³ s⁻².
    radius_m : float
        Reference radius of the expansion in m.
    name : str or None, optional
        Model name.
    source_metadata : dict or None, optional
        Provenance of a loaded file (path, tide system, degrees, overrides).

    Raises
    ------
    ValueError
        If the array has the wrong shape, contains non-finite values, has
        ``C̄00 ≠ 1`` or nonzero degree-1 terms, or if ``gm_m3_s2`` or
        ``radius_m`` is not positive and finite.
    """

    _coeffs: np.ndarray
    gm_m3_s2: float
    radius_m: float
    name: str | None = None
    source_metadata: dict | None = None

    def __post_init__(self) -> None:
        """Validate and copy the coefficients."""
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
        """numpy.ndarray: Coefficient array of shape ``(2, L + 1, L + 1)`` (a view)."""
        result = self._coeffs.view()
        return result


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
    max_degree=None,
    gm_m3_s2=None,
    reference_radius_m=None,
    name=None,
    allow_missing_coefficients=False,
) -> GravityCoefficients:
    """Load a static, fully normalised gravity field in ICGEM format.

    Parameters
    ----------
    file_path : str or path-like
        ICGEM ``.gfc`` file with ``norm fully_normalized`` and a declared
        ``tide_system``. Time-variable records are rejected.
    max_degree : int or None, optional
        Truncation degree; the declared ``max_degree`` when ``None``.
    gm_m3_s2 : float or None, optional
        GM in m³ s⁻² overriding the file's gravity constant.
    reference_radius_m : float or None, optional
        Reference radius in m overriding the file's radius.
    name : str or None, optional
        Model name; the file's ``modelname`` (or file stem) when ``None``.
    allow_missing_coefficients : bool, default False
        Accept files that omit coefficients up to the loaded degree; missing
        coefficients are zero. Degree 1 may always be omitted.

    Returns
    -------
    GravityCoefficients
        Coefficients with ``source_metadata`` describing the file and any
        overrides.

    Raises
    ------
    ValueError
        If the header or coefficient records are malformed, duplicated,
        time-variable or inconsistent with the declared degree, the
        normalisation or tide system is unsupported, required coefficients
        are missing, or ``max_degree`` exceeds the declared degree.
    """
    return _parse_icgem(
        Path(file_path),
        max_degree=max_degree,
        gm_override=None if gm_m3_s2 is None else float(gm_m3_s2),
        radius_override=None if reference_radius_m is None else float(reference_radius_m),
        name=name,
        allow_missing_coefficients=allow_missing_coefficients,
    )


class GravityField:
    """Cartesian acceleration evaluator for a spherical-harmonic field.

    Accelerations are expressed in the body-fixed axes of the coefficients.

    Parameters
    ----------
    coefficients : GravityCoefficients
        Field coefficients.

    Attributes
    ----------
    coefficients : GravityCoefficients
        The field coefficients.
    """

    def __init__(self, coefficients: GravityCoefficients):
        self.coefficients = coefficients
        degree = coefficients.coeffs.shape[1] - 1
        self._normalization = np.zeros((degree + 1, degree + 1))
        for n in range(degree + 1):
            for m in range(n + 1):
                self._normalization[n, m] = math.sqrt(
                    (2 * n + 1) * (2 if m else 1) * math.exp(math.lgamma(n - m + 1) - math.lgamma(n + m + 1))
                )
        if degree >= 1:
            self._normalization[1, 0] = math.sqrt(6)


    @property
    def normalization_table(self) -> np.ndarray:
        """numpy.ndarray: Full-normalisation factors ``N̄nm``, shape ``(L + 1, L + 1)``."""
        return self._normalization

    @property
    def gm_m3_s2(self) -> float:
        """float: Gravitational parameter of the field in m³ s⁻²."""
        return self.coefficients.gm_m3_s2

    @property
    def radius_m(self) -> float:
        """float: Reference radius of the field in m."""
        return self.coefficients.radius_m

    def coefficients_with_degree2(self, degree2) -> np.ndarray:
        """Return a copy of the coefficients with degree 2 replaced.

        Parameters
        ----------
        degree2 : array_like
            Fully normalised degree-2 coefficients, shape ``(2, 3)``: rows C̄
            and S̄, columns orders 0–2.

        Returns
        -------
        numpy.ndarray
            New coefficient array of shape ``(2, L + 1, L + 1)``; the field
            itself is unchanged.
        """
        coefficients = self.coefficients.coeffs.copy()
        coefficients[:, 2, :3] = degree2
        return coefficients

    def coefficients_with_c20(self, c20: float) -> np.ndarray:
        """Return a copy of the coefficients with C̄20 replaced.

        Parameters
        ----------
        c20 : float
            Fully normalised C̄20.

        Returns
        -------
        numpy.ndarray
            New coefficient array of shape ``(2, L + 1, L + 1)``.
        """
        coefficients = self.coefficients.coeffs.copy()
        coefficients[0, 2, 0] = c20
        return coefficients

    def nonspherical_accelerations(self, relative_positions_m, coefficients=None) -> np.ndarray:
        """Return the non-spherical accelerations at several field points.

        The central (degree-0) term is excluded.

        Parameters
        ----------
        relative_positions_m : array_like
            Field-point positions relative to the body centre of mass in the
            body-fixed axes of the field, m, shape ``(N, 3)``.
        coefficients : numpy.ndarray or None, optional
            Replacement coefficients of the field's degree, shape
            ``(2, L + 1, L + 1)``; the field's own coefficients when ``None``.

        Returns
        -------
        numpy.ndarray
            Accelerations in the same axes, m s⁻², shape ``(N, 3)``.

        Raises
        ------
        ValueError
            If the array shapes are inconsistent.
        """
        x = np.asarray(relative_positions_m, dtype=float)
        return np.asarray(
            nonspherical_gravity_accelerations(
                np.ascontiguousarray(x),
                np.ascontiguousarray(self.coefficients.coeffs if coefficients is None else coefficients),
                self.gm_m3_s2,
                self.radius_m,
                self._normalization,
            ),
            dtype=float,
        )


def make_gravity_coefficients(cosine, sine, *, gm_m3_s2, radius_m, name=None) -> GravityCoefficients:
    """Build validated coefficients from separate cosine and sine arrays.

    Parameters
    ----------
    cosine : array_like
        Fully normalised C̄nm, square array of shape ``(L + 1, L + 1)``, with
        ``C̄00 = 1``.
    sine : array_like
        Fully normalised S̄nm, same shape as ``cosine``.
    gm_m3_s2 : float
        Gravitational parameter in m³ s⁻².
    radius_m : float
        Reference radius in m.
    name : str or None, optional
        Model name.

    Returns
    -------
    GravityCoefficients
        Validated coefficients.

    Raises
    ------
    ValueError
        If the arrays are not square, differ in shape, contain non-finite
        values, or fail the :class:`GravityCoefficients` checks.
    """
    c = np.asarray(cosine, dtype=float)
    s = np.asarray(sine, dtype=float)
    if c.ndim != 2 or c.shape != s.shape or c.shape[0] != c.shape[1]:
        raise ValueError("cosine and sine must be square arrays of the same shape")
    if not np.all(np.isfinite(c)) or not np.all(np.isfinite(s)):
        raise ValueError("cosine and sine coefficients must be finite")
    coeffs = np.stack((c, s))
    return GravityCoefficients(coeffs, gm_m3_s2, radius_m, name)


def make_j2_coefficients(*, gm_m3_s2, radius_m, j2, name=None) -> GravityCoefficients:
    """Return a degree-2 field equivalent to an axisymmetric J2.

    Parameters
    ----------
    gm_m3_s2 : float
        Gravitational parameter in m³ s⁻².
    radius_m : float
        Reference radius in m.
    j2 : float
        Unnormalised J2; the field has ``C̄20 = -J2 / √5``.
    name : str or None, optional
        Model name; ``"axisymmetric J2"`` when ``None``.

    Returns
    -------
    GravityCoefficients
        Degree-2 coefficients with ``C̄00 = 1`` and ``C̄20 = -J2 / √5``.

    Raises
    ------
    ValueError
        If ``j2`` is not finite.
    """
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
