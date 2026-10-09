"""Time-varying lunar degree-2 gravity from prescribed deformation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from lunarops.classes.time import require_tdb_epoch

from .gravity import GravityCoefficients, GravityField


@dataclass(frozen=True, slots=True)
class LunarDegree2GravityParameters:
    """Parameters for the delayed lunar degree-2 gravity response."""

    # DE440 BSP comment-area constants, not the mean observed quadrupole.
    j2_undistorted: float = 2.0321436013500001e-4
    beta: float = 6.3122040737914229e-4
    gamma: float = 2.2778898477433167e-4
    love_k2: float = 0.02419
    tidal_delay_days: float = 0.14519388258636395
    radius_m: float = 1738000.0
    mean_motion_rad_s: float = 2.6616995e-6
    attitude_derivative_step_s: float = 30.0
    include_tidal_deformation: bool = True
    include_rotational_deformation: bool = True

    def __post_init__(self) -> None:
        values = [
            getattr(self, key)
            for key in self.__dataclass_fields__
            if key not in {"include_tidal_deformation", "include_rotational_deformation"}
        ]
        if not np.all(np.isfinite(values)):
            raise ValueError("Lunar degree-2 gravity parameters must be finite.")
        if not isinstance(self.include_tidal_deformation, bool) or not isinstance(
            self.include_rotational_deformation, bool
        ):
            raise TypeError("Lunar deformation switches must be boolean.")
        if not (0 < self.gamma < self.beta < 1) or self.j2_undistorted <= 0:
            raise ValueError("Lunar deformation requires 0 < gamma < beta < 1 and positive J2.")
        if self.love_k2 < 0 or self.tidal_delay_days < 0:
            raise ValueError("Lunar deformation delay and Love number must be nonnegative.")
        if min(self.radius_m, self.mean_motion_rad_s, self.attitude_derivative_step_s) <= 0:
            raise ValueError("Invalid lunar deformation or attitude parameters.")

    def undistorted_inertia(self) -> np.ndarray:
        """Return the normalized undeformed lunar inertia tensor."""
        b, g = self.beta, self.gamma
        factor = 2 * self.j2_undistorted / (2 * b - g + b * g)
        return np.diag([factor * (1 - b * g), factor * (1 + g), factor * (1 + b)])


@dataclass(frozen=True, slots=True)
class LunarDegree2GravityEvaluation:
    """Diagnostic result from one lunar degree-2 gravity evaluation."""

    normalized_inertia: np.ndarray
    tidal_inertia: np.ndarray
    rotational_inertia: np.ndarray
    angular_velocity_rad_s: np.ndarray
    degree2_gravity_coefficients: GravityCoefficients

    def __post_init__(self) -> None:
        for name, shape in (
            ("normalized_inertia", (3, 3)),
            ("tidal_inertia", (3, 3)),
            ("rotational_inertia", (3, 3)),
            ("angular_velocity_rad_s", (3,)),
        ):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != shape or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be a finite array with shape {shape}")
            value = np.ascontiguousarray(value.copy())
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if not isinstance(self.degree2_gravity_coefficients, GravityCoefficients):
            raise TypeError("degree2_gravity_coefficients must be GravityCoefficients")


# Coefficient order is (C20, C21, C22, S20, S21, S22), and inertia order is
# (Ixx, Iyy, Izz, Ixy, Ixz, Iyz).
_DEGREE2_GRAVITY_FROM_INERTIA = np.array(
    (
        (1 / (2 * np.sqrt(5)), 1 / (2 * np.sqrt(5)), -1 / np.sqrt(5), 0, 0, 0),
        (0, 0, 0, 0, -np.sqrt(3 / 5), 0),
        (-np.sqrt(12 / 5) / 4, np.sqrt(12 / 5) / 4, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, -np.sqrt(3 / 5)),
        (0, 0, 0, -np.sqrt(12 / 5) / 2, 0, 0),
    ),
    dtype=float,
)
_DEGREE2_GRAVITY_FROM_INERTIA.setflags(write=False)


def _inertia_components(inertia: np.ndarray) -> np.ndarray:
    tensor = np.asarray(inertia, dtype=float)
    if tensor.shape != (3, 3) or not np.all(np.isfinite(tensor)):
        raise ValueError("Inertia must be a finite 3x3 tensor.")
    if not np.allclose(tensor, tensor.T, rtol=0, atol=1e-15):
        raise ValueError("Inertia must be symmetric.")
    return np.asarray(
        (tensor[0, 0], tensor[1, 1], tensor[2, 2], tensor[0, 1], tensor[0, 2], tensor[1, 2]),
        dtype=float,
    )


def degree2_gravity_coefficients_from_inertia(
    inertia, *, gravitational_parameter_m3_s2, radius_m
) -> GravityCoefficients:
    """Convert a normalized inertia tensor to fully normalized degree-2 coefficients."""
    coefficients = np.zeros((2, 3, 3), dtype=float)
    coefficients[0, 0, 0] = 1.0
    coefficients[:, 2, :3] = (
        _DEGREE2_GRAVITY_FROM_INERTIA @ _inertia_components(inertia)
    ).reshape(2, 3)
    return GravityCoefficients(coefficients, float(gravitational_parameter_m3_s2), float(radius_m))


def _degree2_coefficients_from_inertia(inertia) -> np.ndarray:
    return (
        _DEGREE2_GRAVITY_FROM_INERTIA @ _inertia_components(inertia)
    ).reshape(2, 3)


class LunarDegree2GravityModel:
    """Compute delayed lunar degree-2 gravity from prescribed orbit and attitude."""

    def __init__(
        self,
        earth_gravitational_parameter_m3_s2,
        moon_gravitational_parameter_m3_s2,
        earth_minus_moon_position_provider: Callable,
        attitude_provider: Callable,
        *,
        parameters: LunarDegree2GravityParameters | None = None,
    ) -> None:
        if not np.all(
            np.isfinite((earth_gravitational_parameter_m3_s2, moon_gravitational_parameter_m3_s2))
        ) or min(earth_gravitational_parameter_m3_s2, moon_gravitational_parameter_m3_s2) <= 0:
            raise ValueError("Lunar deformation GMs must be positive and finite.")
        if not callable(earth_minus_moon_position_provider) or not callable(attitude_provider):
            raise TypeError("Lunar degree-2 gravity requires position and attitude providers.")
        self.earth_gravitational_parameter_m3_s2 = float(earth_gravitational_parameter_m3_s2)
        self.moon_gravitational_parameter_m3_s2 = float(moon_gravitational_parameter_m3_s2)
        self.earth_minus_moon_position_provider = earth_minus_moon_position_provider
        self.attitude_provider = attitude_provider
        self.parameters = parameters if parameters is not None else LunarDegree2GravityParameters()
        self._attitude_cache: dict[tuple[float, float, object], np.ndarray] = {}
        self._static_coefficients: GravityCoefficients | None = None
        self._degree2_correction_field: GravityField | None = None

    def attitude(self, epoch):
        require_tdb_epoch(epoch)
        return np.array(self._cached_attitude(epoch), copy=True)

    def _cached_attitude(self, epoch):
        key = (float(epoch.jd1), float(epoch.jd2), epoch.scale)
        matrix = self._attitude_cache.get(key)
        if matrix is not None:
            return matrix
        matrix = np.asarray(self.attitude_provider(epoch), dtype=float)
        if (
            matrix.shape != (3, 3)
            or not np.all(np.isfinite(matrix))
            or not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-12, rtol=0)
            or not np.isclose(np.linalg.det(matrix), 1.0, atol=1e-12, rtol=0)
        ):
            raise ValueError("Lunar attitude must be a proper orthogonal matrix.")
        matrix = np.array(matrix, dtype=float, copy=True)
        matrix.setflags(write=False)
        # Keep the cache bounded without making cache policy part of the model state.
        if len(self._attitude_cache) >= 1024:
            self._attitude_cache.clear()
        self._attitude_cache[key] = matrix
        return matrix

    def angular_velocity(self, epoch):
        """Return PA-frame angular velocity in radians per TDB second."""
        require_tdb_epoch(epoch)
        return self._angular_velocity(epoch)

    def _angular_velocity(self, epoch):
        h = self.parameters.attitude_derivative_step_s
        derivative = (
            self._cached_attitude(epoch.shifted(-2 * h))
            - 8 * self._cached_attitude(epoch.shifted(-h))
            + 8 * self._cached_attitude(epoch.shifted(h))
            - self._cached_attitude(epoch.shifted(2 * h))
        ) / (12 * h)
        skew = self._cached_attitude(epoch).T @ derivative
        skew = (skew - skew.T) / 2
        return np.array([skew[2, 1], skew[0, 2], skew[1, 0]])

    def _evaluate_components(self, epoch, *, earth_minus_moon_position_provider=None):
        require_tdb_epoch(epoch)
        p = self.parameters
        delayed = epoch.shifted(-p.tidal_delay_days * 86400)
        provider = (
            self.earth_minus_moon_position_provider
            if earth_minus_moon_position_provider is None
            else earth_minus_moon_position_provider
        )
        earth_minus_moon = np.asarray(provider(delayed), dtype=float)
        if earth_minus_moon.shape != (3,) or not np.all(np.isfinite(earth_minus_moon)):
            raise ValueError("Relative Earth-minus-Moon position must be a finite three-vector")
        r = self._cached_attitude(delayed).T @ earth_minus_moon
        distance = np.linalg.norm(r)
        if not np.isfinite(distance) or distance == 0:
            raise ValueError("Delayed Earth-Moon separation must be nonzero and finite.")
        w = self._angular_velocity(delayed)
        tide = (
            -p.love_k2
            * (self.earth_gravitational_parameter_m3_s2 / self.moon_gravitational_parameter_m3_s2)
            * (p.radius_m / distance) ** 3
            * (np.outer(r / distance, r / distance) - np.eye(3) / 3)
        )
        if not p.include_tidal_deformation:
            tide = np.zeros((3, 3))
        spin_tensor = np.outer(w, w) - np.eye(3) * (w @ w - p.mean_motion_rad_s**2) / 3
        spin_tensor[2, 2] -= p.mean_motion_rad_s**2
        spin = p.love_k2 * p.radius_m**3 / (3 * self.moon_gravitational_parameter_m3_s2) * spin_tensor
        if not p.include_rotational_deformation:
            spin = np.zeros((3, 3))
        total = p.undistorted_inertia() + tide + spin
        return total, tide, spin, w

    def evaluate(self, epoch, *, earth_minus_moon_position_provider=None) -> LunarDegree2GravityEvaluation:
        total, tide, spin, angular_velocity = self._evaluate_components(
            epoch, earth_minus_moon_position_provider=earth_minus_moon_position_provider
        )
        return LunarDegree2GravityEvaluation(
            normalized_inertia=total,
            tidal_inertia=tide,
            rotational_inertia=spin,
            angular_velocity_rad_s=angular_velocity,
            degree2_gravity_coefficients=degree2_gravity_coefficients_from_inertia(
                total,
                gravitational_parameter_m3_s2=self.moon_gravitational_parameter_m3_s2,
                radius_m=self.parameters.radius_m,
            ),
        )

    def _ensure_gravity_field_cache(self, static_coefficients: GravityCoefficients) -> None:
        if static_coefficients is self._static_coefficients:
            return
        if not isinstance(static_coefficients, GravityCoefficients):
            raise TypeError("Static lunar gravity must be GravityCoefficients.")
        if static_coefficients.normalization != "4pi" or static_coefficients.csphase != 1:
            raise ValueError("Static lunar gravity must use 4pi/csphase=1.")
        if not np.isclose(static_coefficients.radius_m, self.parameters.radius_m, rtol=0, atol=1e-9):
            raise ValueError("Static lunar gravity and degree-2 parameters must use the same radius.")
        increment_array = np.zeros_like(static_coefficients.coeffs)
        increment_array[0, 0, 0] = 1.0
        increment_coefficients = GravityCoefficients(
            increment_array, static_coefficients.gm_m3_s2, static_coefficients.radius_m, static_coefficients.name
        )
        self._static_coefficients = static_coefficients
        self._degree2_correction_field = GravityField(increment_coefficients)

    def degree2_gravity_correction_field(
        self, epoch, static_coefficients, *, earth_minus_moon_position_provider=None
    ) -> GravityField:
        """Return only the dynamic-minus-static lunar degree-2 field."""
        self._ensure_gravity_field_cache(static_coefficients)
        total, _tide, _spin, _angular_velocity = self._evaluate_components(
            epoch, earth_minus_moon_position_provider=earth_minus_moon_position_provider
        )
        dynamic_degree2 = _degree2_coefficients_from_inertia(total)
        static_degree2 = np.asarray(static_coefficients.coeffs[:, 2, :3], dtype=float)
        assert self._degree2_correction_field is not None
        self._degree2_correction_field.update_degree2_coefficients(dynamic_degree2 - static_degree2)
        return self._degree2_correction_field
