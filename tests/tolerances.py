"""Named floating-point tolerances shared by the test suite.

Values computed through libm, ERFA, BLAS or the compiled extensions differ by a
few ulp between operating systems, compilers and FMA settings, so tests never
compare such values bit for bit. Each tolerance below is far below the physical
significance of its quantity and above the spread measured with an FMA build
(``-ffp-contract=fast -O3 -march=native``). Model or discretization limits of a
particular comparison (finite differences, step convergence, an external code)
are declared next to that test as a named ``Tolerance`` with their reason.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Tolerance:
    name: str
    atol: float = 0.0
    rtol: float = 0.0
    unit: str = ""


# Epoch differences and time-scale conversions: one ulp of a two-part JD day fraction is up to 4.8e-12 s.
TIME_S = Tolerance("TIME_S", atol=1e-11, unit="s")
# Light-time solutions and round-trip times (1e-13 s is 30 micrometres one way).
LIGHT_TIME_S = Tolerance("LIGHT_TIME_S", atol=1e-13, unit="s")
# Double-precision displacement and delay models (DEHANT, pole tides, FCUL delays).
DISPLACEMENT_M = Tolerance("DISPLACEMENT_M", atol=1e-12, unit="m")
# HARDISP keeps float32 intermediates of the reference code; an FMA build differs by 1.4e-9 m.
HARDISP_M = Tolerance("HARDISP_M", atol=2e-9, unit="m")
# Rotation-matrix elements, pole coordinates and orientation angles.
ANGLE_RAD = Tolerance("ANGLE_RAD", atol=1e-14, unit="rad")
# IERS high-frequency EOP series in microarcseconds or microseconds.
EOP_UAS = Tolerance("EOP_UAS", atol=1e-10, unit="uas")
# Dimensionless quantities and relative comparisons of double-precision results.
REL = Tolerance("REL", rtol=1e-12)
# Native dynamics RHS against the NumPy oracle, relative to the compared vector's norm.
DYNAMICS_REL = Tolerance("DYNAMICS_REL", rtol=1e-10)
# Least-squares parameters, and their covariance, sigma0 and redundancies.
ESTIMATE_REL = Tolerance("ESTIMATE_REL", rtol=1e-10)
COVARIANCE_REL = Tolerance("COVARIANCE_REL", rtol=1e-8)


def assert_close(actual, expected, tol: Tolerance, *, err_msg: str = "") -> None:
    """Element-wise ``|actual - expected| <= atol + rtol * |expected|``."""
    np.testing.assert_allclose(
        np.asarray(actual, dtype=float), np.asarray(expected, dtype=float),
        rtol=tol.rtol, atol=tol.atol, err_msg=f"{tol.name} {tol.unit} {err_msg}".strip(),
    )


def assert_vector_close(actual, expected, tol: Tolerance, *, err_msg: str = "") -> None:
    """Compare vectors along the last axis by ``|delta| <= atol + rtol * |expected|``.

    Use for physical vectors whose small components are not separately significant.
    """
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    if actual.size == 0:
        return
    error = np.linalg.norm(actual - expected, axis=-1)
    limit = tol.atol + tol.rtol * np.linalg.norm(expected, axis=-1)
    worst = np.unravel_index(np.argmax(error - limit), np.shape(error)) if np.ndim(error) else ()
    assert np.all(error <= limit), (
        f"{tol.name}: |delta|={np.max(error):.3e} exceeds {np.asarray(limit)[worst]:.3e} {tol.unit} {err_msg}".strip()
    )
