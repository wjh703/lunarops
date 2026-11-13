"""Type declarations of the compiled lunar dynamics kernels.

The extension provides the Newtonian point-mass sums, the spherical-harmonic
gravity accelerations and the fused lunar right-hand side
:class:`LunarRhsKernel`. All quantities are SI: positions in m, velocities
in m s⁻¹, accelerations in m s⁻², potentials in m² s⁻² and gravitational
parameters in m³ s⁻².
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

def point_mass_symmetric(
    positions: NDArray[np.float64],
    gravitational_parameters: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the mutual Newtonian accelerations and potentials of N bodies.

    Each pair is evaluated once and contributes to both bodies.

    Parameters
    ----------
    positions : numpy.ndarray
        C-contiguous positions in m, shape ``(N, 3)``, in any common frame.
    gravitational_parameters : numpy.ndarray
        GM in m³ s⁻², shape ``(N,)``.

    Returns
    -------
    acceleration : numpy.ndarray
        Acceleration of each body due to all others, m s⁻², shape ``(N, 3)``.
    potential : numpy.ndarray
        Newtonian potential ``Σ_j GM_j / r_ij`` at each body, m² s⁻²,
        shape ``(N,)``; positive by convention.

    Raises
    ------
    ValueError
        If the shapes are inconsistent or two bodies coincide.
    """

def point_mass_symmetric_batch(
    positions: NDArray[np.float64],
    gravitational_parameters: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return mutual Newtonian accelerations and potentials at E epochs.

    Parameters
    ----------
    positions : numpy.ndarray
        C-contiguous positions in m, shape ``(E, N, 3)``.
    gravitational_parameters : numpy.ndarray
        GM in m³ s⁻² per epoch, shape ``(E, N)``.

    Returns
    -------
    acceleration : numpy.ndarray
        Accelerations in m s⁻², shape ``(E, N, 3)``.
    potential : numpy.ndarray
        Potentials in m² s⁻², shape ``(E, N)``.

    Raises
    ------
    ValueError
        If the shapes are inconsistent or two bodies coincide at any epoch.
    """

def nonspherical_gravity_accelerations(
    positions: NDArray[np.float64],
    coefficients: NDArray[np.float64],
    gravitational_parameter: float,
    reference_radius: float,
    norm: np.ndarray,
) -> NDArray[np.float64]:
    """Return non-spherical spherical-harmonic accelerations at field points.

    The degree-0 (central) term is excluded. Unnormalised Legendre
    intermediates limit the usable degree to about 150.

    Parameters
    ----------
    positions : numpy.ndarray
        Field points relative to the body centre of mass in the body-fixed
        axes of the coefficients, m, shape ``(N, 3)``.
    coefficients : numpy.ndarray
        Fully normalised (4π) coefficients ``[C̄, S̄]``, shape
        ``(2, L + 1, L + 1)``.
    gravitational_parameter : float
        GM of the body in m³ s⁻².
    reference_radius : float
        Reference radius of the expansion in m.
    norm : numpy.ndarray
        Full-normalisation factors, shape ``(L + 1, L + 1)``.

    Returns
    -------
    numpy.ndarray
        Accelerations in the body-fixed axes, m s⁻², shape ``(N, 3)``.

    Raises
    ------
    ValueError
        If the shapes are inconsistent or ``gravitational_parameter`` or
        ``reference_radius`` is not positive.
    """

def adams_accumulate(
    step: float,
    state: NDArray[np.float64],
    state_low: NDArray[np.float64],
    derivative: NDArray[np.float64],
    derivative_low: NDArray[np.float64],
    weight: NDArray[np.float64],
    weight_low: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return ``y + step * sum_j weight[j] * f[:, j]`` in double-double arithmetic.

    Parameters
    ----------
    step : float
        Signed step in s.
    state, state_low : numpy.ndarray
        High and low parts of ``y``, shape ``(d,)``.
    derivative, derivative_low : numpy.ndarray
        High and low parts of ``f``, shape ``(d, m)``.
    weight, weight_low : numpy.ndarray
        High and low parts of the weights, shape ``(m,)``.

    Returns
    -------
    high, low : numpy.ndarray
        High and low parts of the result, each of shape ``(d,)``.

    Raises
    ------
    ValueError
        If the shapes are inconsistent.
    """

def hermite_sample(
    states: NDArray[np.float64],
    derivatives: NDArray[np.float64],
    first_time: float,
    step: float,
    offset: float,
    state_out: NDArray[np.float64],
    derivative_out: NDArray[np.float64],
) -> int:
    """Evaluate the Hermite interpolant of regularly spaced nodes and its derivative into the outputs.

    Compiled form of ``lunarops.classes.dynamics.history.interpolate_state_derivative``; the sums are
    reordered, so results agree with it to a few ulp of the summed terms.

    Parameters
    ----------
    states, derivatives : numpy.ndarray
        Node states and their time derivatives, shape ``(d, n)`` with ``1 <= n <= 32``; strided views are
        accepted.
    first_time : float
        Offset of the first node in s.
    step : float
        Node spacing in s (ignored for one node).
    offset : float
        Evaluation offset in s.
    state_out, derivative_out : numpy.ndarray
        Outputs, shape ``(d,)``.

    Returns
    -------
    int
        Index of the node within 1e-13 grid units of ``offset``, whose values are copied unchanged, or -1.

    Raises
    ------
    ValueError
        If the shapes are inconsistent or ``n`` is outside 1..32.
    """

def lunar_deformation(
    state: NDArray[np.float64],
    state_derivative: NDArray[np.float64],
    undistorted_inertia: NDArray[np.float64],
    love_k2: float,
    mean_motion_rad_s: float,
    include_tidal: bool,
    include_rotational: bool,
    radius_m: float,
    earth_mu: float,
    moon_mu: float,
    c21_static: float,
    s21_static: float,
    s22_static: float,
    total_out: NDArray[np.float64],
    derivative_out: NDArray[np.float64],
    tidal_out: NDArray[np.float64],
    rotational_out: NDArray[np.float64],
    degree2_out: NDArray[np.float64],
) -> None:
    """Delayed lunar inertia, its rate, its tidal and rotational parts and the degree-2 coefficients.

    Compiled form of ``euler_matrix``, ``deformation_tensors`` and ``_degree2_coefficients_from_inertia``
    (Folkner et al. 2014, Eqs. 40-42; Park et al. 2021, Eqs. 47-55; Pavlov et al. 2016, Eq. 16).

    Parameters
    ----------
    state : numpy.ndarray
        Integration state at ``t - tau``: relative position (m) and velocity (m/s) in ICRF, 3-1-3 Euler
        angles (rad), mantle angular velocity in PA (rad/s); at least 12 entries.
    state_derivative : numpy.ndarray
        Its time derivative; entries 9:12 (angular acceleration) are used.
    undistorted_inertia : numpy.ndarray
        Undeformed inertia in PA axes normalised by ``M R^2``, shape ``(3, 3)``, C-contiguous.
    love_k2, mean_motion_rad_s, include_tidal, include_rotational : float, float, bool, bool
        Response parameters (``LunarDeformationParameters``).
    radius_m, earth_mu, moon_mu : float
        Lunar radius (m) and Earth and Moon GM (m^3 s^-2).
    c21_static, s21_static, s22_static : float
        Unnormalised static degree-2 offsets added to the gravity coefficients only.
    total_out, derivative_out, tidal_out, rotational_out : numpy.ndarray
        Outputs, shape ``(3, 3)``; ``derivative_out`` is the rate of ``tidal + rotational`` (s^-1).
    degree2_out : numpy.ndarray
        Output fully normalised degree-2 coefficients, shape ``(2, 3)`` (rows C, S; orders 0-2).
    """

class LunarRhsKernel:
    """Fused right-hand side of the Moon-minus-Earth translation and rotation.

    :meth:`set_epoch` loads the prescribed inputs of one epoch,
    :meth:`set_step` loads the candidate-independent delayed data of one
    integration step, and :meth:`derivative` then evaluates any number of
    candidate states without holding the GIL. The kernel owns its
    workspaces, so one instance must not be used by several threads at once.

    Parameters
    ----------
    cfg : dict
        Kernel configuration produced by
        :func:`lunarops.classes.dynamics.fused_rhs.compile_fused_rhs`.
    """

    def __init__(self, cfg: dict[str, Any]) -> None: ...
    @property
    def state_dimension(self) -> int:
        """int: State length: 6 for orbit only, 12 with rotation, 15 with a core."""
    def set_epoch(self, batch: Any, index: int) -> None:
        """Load the prescribed inputs of one epoch.

        Parameters
        ----------
        batch : EpochBatch
            Stacked prescribed inputs.
        index : int
            Row of the epoch in ``batch``.
        """
    def set_step(
        self,
        tide_vectors: NDArray[np.float64],
        raiser_gravitational_parameters: NDArray[np.float64],
        inertia_total: NDArray[np.float64],
        inertia_derivative: NDArray[np.float64],
        moon_coefficients: NDArray[np.float64],
    ) -> None:
        """Load the candidate-independent data of one integration step.

        Parameters
        ----------
        tide_vectors : numpy.ndarray
            Delayed Earth-to-raiser vectors in Earth gravity-field axes, m,
            shape ``(n_raisers, 3, 3)`` (raiser, tidal order, axis).
        raiser_gravitational_parameters : numpy.ndarray
            Raiser GM in m³ s⁻², shape ``(n_raisers,)``.
        inertia_total : numpy.ndarray
            Whole-Moon inertia in PA axes normalised by ``M R²``,
            shape ``(3, 3)``.
        inertia_derivative : numpy.ndarray
            Its time derivative in s⁻¹, shape ``(3, 3)``.
        moon_coefficients : numpy.ndarray
            Moon field coefficients for this step, shape ``(2, L + 1, L + 1)``.

        Raises
        ------
        ValueError
            If the tide payload does not match the configured raiser count.
        """
    def split_derivative(
        self, y: NDArray[np.float64], y_low: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return the derivative of a double-double state as a double-double pair.

        The Earth-Moon term is evaluated in double-double arithmetic on
        ``y + y_low`` and the third-body terms in a cancellation-free form;
        all other terms use ``y`` alone.

        Parameters
        ----------
        y, y_low : numpy.ndarray
            High and low parts of the integration state, see :meth:`derivative`.

        Returns
        -------
        derivative, derivative_low : numpy.ndarray
            High and low parts of the time derivative.

        Raises
        ------
        ValueError
            As for :meth:`derivative`.
        """
    def derivative(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the state derivative of one candidate state.

        Parameters
        ----------
        y : numpy.ndarray
            Integration state of length :attr:`state_dimension`: Moon-minus-
            Earth position (m) and velocity (m s⁻¹) with ICRF axes, then the
            PA-to-inertial 3-1-3 Euler angles (rad), the mantle and the core
            angular velocities in PA axes (rad s⁻¹).

        Returns
        -------
        numpy.ndarray
            Time derivative of ``y`` in TDB seconds.

        Raises
        ------
        ValueError
            If the state length is wrong, the Euler angles are not finite
            or ``sin θ`` is zero, two bodies coincide, an Earth-tide vector is zero, or the
            mantle inertia is singular.
        """
    def terms(self, y: NDArray[np.float64]) -> dict[str, Any]:
        """Return the derivative together with every individual term.

        Parameters
        ----------
        y : numpy.ndarray
            Integration state of length :attr:`state_dimension`.

        Returns
        -------
        dict
            ``"forces"``: Moon accelerations in m s⁻² with ICRF axes,
            shape ``(8, 3)``, rows point mass, EIH, solar J2, Earth figure,
            Moon figure, Earth tide, Lense-Thirring, radiation pressure;
            ``"earth_forces"``: Earth accelerations of the same terms, so the
            relative acceleration is ``forces - earth_forces``;
            ``"torques"``: torques in PA axes in s⁻², shape ``(n_torques, 3)``;
            ``"rotation"``: viscous core-mantle torque, pressure torque and
            geodetic precession rate, shape ``(3, 3)``; ``"inertia"``: total
            inertia, mantle inertia and inertia rate, shape ``(3, 3, 3)``;
            ``"derivative"``: the state derivative.

        Raises
        ------
        ValueError
            Under the same conditions as :meth:`derivative`.
        """
