"""Time-varying lunar degree-2 gravity from prescribed deformation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from lunarops.classes.time import require_tdb_epoch

from .deformation import LunarDeformationParameters, _degree2_coefficients_from_inertia, deformation_tensors


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
    include_tidal_deformation: bool = True
    include_rotational_deformation: bool = True
    # Unnormalized static offsets in Pavlov et al. (2016), Eq. (16).
    c21_static: float = 0.0
    s21_static: float = 0.0
    s22_static: float = 0.0

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
        if min(self.radius_m, self.mean_motion_rad_s) <= 0:
            raise ValueError("Invalid lunar deformation or attitude parameters.")

    def undistorted_inertia(self) -> np.ndarray:
        """Return the normalized undeformed lunar inertia tensor."""
        b, g = self.beta, self.gamma
        factor = 2 * self.j2_undistorted / (2 * b - g + b * g)
        return np.diag([factor * (1 - b * g), factor * (1 + g), factor * (1 + b)])


class LunarDegree2GravityModel:
    """Compute delayed lunar degree-2 gravity from prescribed orbit and attitude."""

    def __init__(
        self,
        earth_gravitational_parameter_m3_s2,
        moon_gravitational_parameter_m3_s2,
        earth_minus_moon_position_provider: Callable,
        attitude_provider: Callable,
        *,
        angular_velocity_provider: Callable,
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
        self.angular_velocity_provider = angular_velocity_provider
        self.parameters = parameters if parameters is not None else LunarDegree2GravityParameters()
        self._response = LunarDeformationParameters(
            self.parameters.love_k2, self.parameters.tidal_delay_days * 86400,
            self.parameters.mean_motion_rad_s,
            self.parameters.include_tidal_deformation, self.parameters.include_rotational_deformation,
            self.parameters.c21_static, self.parameters.s21_static, self.parameters.s22_static,
        )
        self._attitude_cache: dict[tuple[float, float, object], np.ndarray] = {}


    def _cached_attitude(self, epoch):
        key = (float(epoch.jd1), float(epoch.jd2), epoch.scale)
        matrix = self._attitude_cache.get(key)
        if matrix is not None:
            return matrix
        matrix = np.asarray(self.attitude_provider(epoch), dtype=float)
        matrix.setflags(write=False)
        # Keep the cache bounded without making cache policy part of the model state.
        if len(self._attitude_cache) >= 1024:
            self._attitude_cache.clear()
        self._attitude_cache[key] = matrix
        return matrix


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
        r = self._cached_attitude(delayed).T @ earth_minus_moon
        w = self.angular_velocity_provider(delayed)
        tide, spin, _ = deformation_tensors(
            r, np.zeros(3), w, np.zeros(3), parameters=self._response,
            radius_m=p.radius_m, earth_mu=self.earth_gravitational_parameter_m3_s2,
            moon_mu=self.moon_gravitational_parameter_m3_s2,
        )
        total = p.undistorted_inertia() + tide + spin
        return total, tide, spin, w

    def update_gravity_field(self, epoch, field, *, earth_minus_moon_position_provider=None):
        """Replace degree 2 in the shared Moon field, preserving higher degrees."""
        total, _, _, _ = self._evaluate_components(
            epoch, earth_minus_moon_position_provider=earth_minus_moon_position_provider,
        )
        p = self.parameters
        field.update_degree2_coefficients(_degree2_coefficients_from_inertia(
            total, c21_static=p.c21_static, s21_static=p.s21_static, s22_static=p.s22_static,
        ))
