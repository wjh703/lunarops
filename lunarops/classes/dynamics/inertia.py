"""Delayed lunar quadrupole with prescribed PA attitude (Park 2021, Eq. 54).

Inertia tensors are divided by M_moon R_moon**2. This avoids introducing G
or an independently rounded lunar mass into the gravity coefficients.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lunarops.classes.ephemerides import require_tdb_epoch


@dataclass(frozen=True, slots=True)
class LunarInertiaParameters:
    # DE440 BSP comment-area constants, not the mean observed quadrupole.
    j2_undistorted: float = 2.0321436013500001e-4
    beta: float = 6.3122040737914229e-4
    gamma: float = 2.2778898477433167e-4
    love_k2: float = 0.02419
    delay_days: float = 0.14519388258636395
    radius_m: float = 1738000.0
    mean_motion_rad_s: float = 2.6616995e-6
    attitude_difference_step_s: float = 30.0
    include_tidal_deformation: bool = True
    include_rotational_deformation: bool = True

    def __post_init__(self):
        values = [
            getattr(self, key)
            for key in self.__dataclass_fields__
            if key not in {"include_tidal_deformation", "include_rotational_deformation"}
        ]
        if not np.all(np.isfinite(values)):
            raise ValueError("Lunar inertia parameters must be finite.")
        if not isinstance(self.include_tidal_deformation, bool) or not isinstance(
            self.include_rotational_deformation, bool
        ):
            raise TypeError("Lunar deformation switches must be boolean.")
        if not (0 < self.gamma < self.beta < 1) or self.j2_undistorted <= 0:
            raise ValueError("Lunar inertia requires 0 < gamma < beta < 1 and positive J2.")
        if self.love_k2 < 0 or min(self.radius_m, self.mean_motion_rad_s, self.attitude_difference_step_s) <= 0:
            raise ValueError("Invalid lunar deformation or attitude parameters.")

    def undistorted(self):
        b, g = self.beta, self.gamma
        factor = 2 * self.j2_undistorted / (2 * b - g + b * g)
        return np.diag([factor * (1 - b * g), factor * (1 + g), factor * (1 + b)])


def degree2_from_inertia(inertia, *, gm_m3_s2, radius_m):
    """Return a degree-2 ``SHGravCoeffs`` field from ``I/(M R**2)``."""
    import pyshtools

    tensor = np.asarray(inertia, float)
    if tensor.shape != (3, 3) or not np.all(np.isfinite(tensor)):
        raise ValueError("Inertia must be a finite 3x3 tensor.")
    if not np.allclose(tensor, tensor.T, rtol=0, atol=1e-15):
        raise ValueError("Inertia must be symmetric.")
    c = np.zeros((2, 3, 3))
    c[0, 2, 0] = (tensor[0, 0] + tensor[1, 1] - 2 * tensor[2, 2]) / 2
    c[0, 2, 1], c[1, 2, 1] = -tensor[0, 2], -tensor[1, 2]
    c[0, 2, 2], c[1, 2, 2] = (tensor[1, 1] - tensor[0, 0]) / 4, -tensor[0, 1] / 2
    coefficients = pyshtools.SHCoeffs.from_array(c, normalization="unnorm", csphase=1).convert(
        normalization="4pi", csphase=1
    )
    result = pyshtools.SHGravCoeffs.from_array(
        coefficients.coeffs,
        gm=float(gm_m3_s2),
        r0=float(radius_m),
        normalization="4pi",
        csphase=1,
        set_degree0=True,
    )
    return result


class LunarInertiaModel:
    """Dynamic lunar gravity using delayed orbit and prescribed mantle attitude.

    The PA-to-inertial matrix and its differentiated angular velocity are
    prescribed. Mantle/core Euler equations are not integrated by this class.
    """

    def __init__(self, earth_gm, moon_gm, initial_history, pa2inertial_provider, *, parameters=None):
        if not np.all(np.isfinite([earth_gm, moon_gm])) or min(earth_gm, moon_gm) <= 0:
            raise ValueError("Lunar deformation GMs must be positive and finite.")
        if not callable(initial_history) or not callable(pa2inertial_provider):
            raise TypeError("Lunar inertia requires history and attitude providers.")
        self.earth_gm, self.moon_gm = earth_gm, moon_gm
        self.initial_history = initial_history
        self.pa2inertial_provider = pa2inertial_provider
        self.parameters = parameters if parameters is not None else LunarInertiaParameters()

    def attitude(self, epoch):
        require_tdb_epoch(epoch)
        matrix = np.asarray(self.pa2inertial_provider(epoch), float)
        if (
            matrix.shape != (3, 3)
            or not np.all(np.isfinite(matrix))
            or not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-12, rtol=0)
            or not np.isclose(np.linalg.det(matrix), 1.0, atol=1e-12, rtol=0)
        ):
            raise ValueError("Lunar attitude must be a proper orthogonal matrix.")
        return matrix

    def angular_velocity(self, epoch):
        """Five-point attitude derivative; return PA components in rad/TDB s."""
        h = self.parameters.attitude_difference_step_s
        derivative = (
            self.attitude(epoch.shifted(-2 * h))
            - 8 * self.attitude(epoch.shifted(-h))
            + 8 * self.attitude(epoch.shifted(h))
            - self.attitude(epoch.shifted(2 * h))
        ) / (12 * h)
        skew = self.attitude(epoch).T @ derivative
        skew = (skew - skew.T) / 2
        return np.array([skew[2, 1], skew[0, 2], skew[1, 0]])

    def evaluate(self, epoch, *, history=None):
        require_tdb_epoch(epoch)
        p = self.parameters
        delayed = epoch.shifted(-p.delay_days * 86400)
        provider = self.initial_history if history is None else history
        states = np.asarray(provider(("EARTH", "MOON"), delayed), dtype=float)
        if states.shape != (2, 6) or not np.all(np.isfinite(states)):
            raise ValueError("Lunar inertia history must return a finite (2,6) state matrix")
        earth, moon = states[:, :3]
        r = self.attitude(delayed).T @ (earth - moon)
        distance = np.linalg.norm(r)
        if not np.isfinite(distance) or distance == 0:
            raise ValueError("Delayed Earth-Moon separation must be nonzero and finite.")
        w = self.angular_velocity(delayed)
        tide = (
            -p.love_k2
            * (self.earth_gm / self.moon_gm)
            * (p.radius_m / distance) ** 3
            * (np.outer(r / distance, r / distance) - np.eye(3) / 3)
        )
        if not p.include_tidal_deformation:
            tide = np.zeros((3, 3))
        spin_tensor = np.outer(w, w) - np.eye(3) * (w @ w - p.mean_motion_rad_s**2) / 3
        spin_tensor[2, 2] -= p.mean_motion_rad_s**2
        spin = p.love_k2 * p.radius_m**3 / (3 * self.moon_gm) * spin_tensor
        if not p.include_rotational_deformation:
            spin = np.zeros((3, 3))
        total = p.undistorted() + tide + spin
        return {
            "normalized_inertia": total,
            "tidal_inertia": tide,
            "rotational_inertia": spin,
            "delayed_angular_velocity_rad_s": w,
            "degree2": degree2_from_inertia(total, gm_m3_s2=self.moon_gm, radius_m=p.radius_m),
        }

    def field(self, epoch, static_coefficients, *, history=None):
        import pyshtools

        if not isinstance(static_coefficients, pyshtools.SHGravCoeffs):
            raise TypeError("Static lunar harmonics must be SHGravCoeffs.")
        if static_coefficients.normalization != "4pi" or static_coefficients.csphase != 1:
            raise ValueError("Static lunar harmonics must use 4pi/csphase=1.")
        if not np.isclose(static_coefficients.r0, self.parameters.radius_m, rtol=0, atol=1e-9):
            raise ValueError("Static lunar harmonics and inertia parameters must use the same radius.")
        result = self.evaluate(epoch, history=history)
        field = static_coefficients.copy()
        field.coeffs[:, 2, :3] = result["degree2"].coeffs[:, 2, :3]
        return field, result
