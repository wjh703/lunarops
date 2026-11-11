"""BCRS body states evaluated directly from the Chebyshev records of SPK kernels."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Self

import numpy as np

from lunarops._ephemeris_core import (  # pyright: ignore[reportMissingModuleSource]
    ChebyshevLink,
    EphemerisCoverageError,
    evaluate_states,
)
from lunarops.classes.ephemerides.body_ids import body_name, naif_id
from lunarops.classes.ephemerides.ephemeris_provider import BodyState
from lunarops.classes.frames.coordinate_scaling import LunarRelativisticScale, LunarRelativisticScaleConvention
from lunarops.classes.time import Epoch, TimeScale, TimeScaleConverter, require_tdb_epoch
from lunarops.fileio.daf import DafFile, DafSegment, chebyshev_layout

from .kernels import calceph_kernel_paths
from .lunar_orientation import (
    EulerAngleLunarOrientation,
    LongitudeLibrationCorrection,
    normalize_longitude_libration_correction,
)

_J2000_TT = Epoch(2_451_545.0, 0.0, TimeScale.TT)
_J2000_JD = 2_451_545.0
_SECONDS_PER_DAY = 86_400.0
_KM_TO_M = 1000.0
_ICRF_FRAME = 1
_MAX_CHAIN = 16


def split_epoch_seconds(jd1, jd2) -> tuple[np.ndarray, np.ndarray]:
    """Return two-part TDB seconds past J2000 of two-part Julian Dates.

    The whole days of both parts go into ``a`` (exact whole seconds); the fractional days are added and
    scaled into ``f``, so that ``a + f`` is the epoch and ``|f| < 2`` days regardless of how the Julian
    Date is split.

    Parameters
    ----------
    jd1, jd2 : array_like
        Two-part Julian Dates (TDB); broadcast together.

    Returns
    -------
    a, f : ndarray of float64
        Whole and fractional seconds past J2000 (JD 2451545.0 TDB), C-contiguous, one-dimensional.
    """
    if isinstance(jd1, (float, int)) and isinstance(jd2, (float, int)):
        return _split_scalar(float(jd1), float(jd2))
    parts = np.broadcast_arrays(np.asarray(jd1, dtype=float), np.asarray(jd2, dtype=float))
    first, second = parts[0].ravel(), parts[1].ravel()
    if not (np.isfinite(first).all() and np.isfinite(second).all()):
        raise ValueError("Julian-date parts must be finite.")
    whole_first = np.floor(first)
    whole_second = np.floor(second)
    a = ((whole_first - _J2000_JD) + whole_second) * _SECONDS_PER_DAY
    f = ((first - whole_first) + (second - whole_second)) * _SECONDS_PER_DAY
    return a, f


def _split_scalar(first: float, second: float) -> tuple[np.ndarray, np.ndarray]:
    # Same arithmetic as split_epoch_seconds without numpy's per-call overhead.
    if not (math.isfinite(first) and math.isfinite(second)):
        raise ValueError("Julian-date parts must be finite.")
    whole_first, whole_second = math.floor(first), math.floor(second)
    a = ((whole_first - _J2000_JD) + whole_second) * _SECONDS_PER_DAY
    f = ((first - whole_first) + (second - whole_second)) * _SECONDS_PER_DAY
    return np.array([a]), np.array([f])


def coverage_partition(segments: Sequence[tuple[float, float, int]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Partition segment coverage into elementary intervals owned by the highest-priority segment.

    Parameters
    ----------
    segments : sequence of (start, end, priority)
        Closed coverage intervals; a larger priority wins where segments overlap.

    Returns
    -------
    bounds : ndarray, shape (n + 1,)
        Increasing interval boundaries.
    owners : ndarray of int32, shape (n,)
        Index into ``segments`` of each interval's owner, -1 where no segment covers it.
    priorities : ndarray of int32, shape (n,)
        Priority of each interval's owner, -1 for gaps.

    Raises
    ------
    ValueError
        If no segment has a positive length.
    """
    usable = [(index, start, end, priority) for index, (start, end, priority) in enumerate(segments) if end > start]
    if not usable:
        raise ValueError("no segment with a positive coverage interval")
    bounds = np.unique(np.asarray([value for _, start, end, _ in usable for value in (start, end)], dtype=float))
    owners = np.full(bounds.size - 1, -1, dtype=np.int32)
    priorities = np.full(bounds.size - 1, -1, dtype=np.int32)
    for index, start, end, priority in sorted(usable, key=lambda item: -item[3]):
        lo, hi = np.searchsorted(bounds, start), np.searchsorted(bounds, end)
        free = owners[lo:hi] == -1
        owners[lo:hi][free] = index
        priorities[lo:hi][free] = priority
    # Merge neighbouring intervals of the same owner.
    keep = np.concatenate(([True], owners[1:] != owners[:-1]))
    return np.append(bounds[:-1][keep], bounds[-1]), owners[keep], priorities[keep]


class ChebyshevEphemeris:
    """SPK-kernel ephemeris of BCRS body states evaluated from the kernels' Chebyshev records.

    All ``.bsp`` position kernels and ``.bpc`` orientation kernels of a directory are opened in the load
    order of the CALCEPH backend (``.bsp`` sorted by path, then ``.bpc``). Later kernels, and later
    segments within a kernel, take precedence where coverage overlaps, as in SPICE. Body states are
    relative to the solar-system barycentre on ICRF axes, in metres, m/s and m/s^2, at TDB epochs; each
    body is the sum of the links of its centre chain down to the barycentre (for example Moon = Moon
    relative to the Earth-Moon barycentre + Earth-Moon barycentre relative to the solar-system barycentre).

    Epochs are converted to two-part seconds past J2000 (see :func:`split_epoch_seconds`), so results do not
    depend on how a Julian Date is split between its two parts.

    SPK segments of types 2, 3 and 20 and PCK segments of types 2 and 20 on frame 1 (J2000/ICRF) are
    evaluated; a body whose chain needs another type or frame raises when it is first requested. The
    ``.bpc`` kernels must describe exactly one orientation target relative to ICRF; its 3-1-3 Euler angles
    give :attr:`lunar_orientation`.

    Parameters
    ----------
    kernel_directory : str or Path
        Directory containing at least one ``.bsp`` and one ``.bpc`` kernel.
    lunar_relativistic_scale_convention : LunarRelativisticScaleConvention or str
        Coordinate scale of the lunar positions in the kernels (``tdbCompatibleLunarSurface`` or
        ``alreadyScaled``); exposed as ``relativistic_scale``.
    longitude_libration_correction : LongitudeLibrationCorrection or str or None, optional
        Periodic correction added to the kernel's longitude libration; ``None`` means ``"none"``.

    Attributes
    ----------
    lunar_orientation : EulerAngleLunarOrientation
        Lunar principal-axis orientation from the ``.bpc`` kernels.

    Raises
    ------
    ValueError
        If the directory does not exist, lacks ``.bsp`` or ``.bpc`` kernels, a kernel is not a valid SPK or
        binary PCK DAF file, or the orientation segments do not describe exactly one target relative to
        ICRF with supported types.
    """

    def __init__(
        self,
        kernel_directory: str | Path,
        *,
        lunar_relativistic_scale_convention: LunarRelativisticScaleConvention | str,
        longitude_libration_correction: LongitudeLibrationCorrection | str | None = None,
    ) -> None:
        directory = Path(kernel_directory).expanduser()
        if not directory.is_dir():
            raise ValueError(f"SPICE kernel directory not found: {directory}")
        kernels = calceph_kernel_paths(directory)
        position_kernels = tuple(path for path in kernels if path.suffix == ".bsp")
        orientation_kernels = tuple(path for path in kernels if path.suffix == ".bpc")
        if not position_kernels or not orientation_kernels:
            raise ValueError(f"SPICE kernel directory must contain .bsp and .bpc kernels: {directory}")
        self._source_path = directory
        self._scale = LunarRelativisticScale.from_convention(lunar_relativistic_scale_convention)
        self._correction = normalize_longitude_libration_correction(longitude_libration_correction)
        self._files: list[DafFile] | None = [DafFile(path) for path in position_kernels]
        # (file, segment, priority) per (target, centre); priority increases with load order.
        self._segments: dict[int, dict[int, list[tuple[DafFile, DafSegment, int]]]] = {}
        priority = 0
        for daf in self._files:
            if daf.kind != "spk":
                raise ValueError(f"Position kernel is not an SPK: {daf.path}")
            for segment in daf.segments:
                center = int(segment.center)  # type: ignore[arg-type]
                self._segments.setdefault(segment.target, {}).setdefault(center, []).append((daf, segment, priority))
                priority += 1
        self._links: dict[tuple[int, int], ChebyshevLink] = {}
        self._plans: dict[int, list[tuple[ChebyshevLink, float]]] = {}
        self._target_ids: dict[str, int] = {}
        self._resolved_bodies: dict[tuple, tuple[list[tuple[ChebyshevLink, float]], ...]] = {}
        self._orientation_plan = [(self._orientation_link(orientation_kernels, priority), 1.0)]
        j2000_tdb = (
            TimeScaleConverter().tt2tdb(_J2000_TT)
            if self._correction is LongitudeLibrationCorrection.INPOP21A
            else None
        )
        self.lunar_orientation = EulerAngleLunarOrientation(
            self._lunar_angles_rad, self._lunar_angles_many_rad, self._correction, j2000_tdb
        )

    @property
    def source_path(self) -> Path:
        """Directory containing the SPICE kernels."""
        return self._source_path

    @property
    def relativistic_scale(self) -> LunarRelativisticScale:
        """Lunar relativistic coordinate scaling selected for the kernels' lunar positions."""
        return self._scale

    def _require_open(self) -> None:
        if self._files is None:
            raise ValueError("ChebyshevEphemeris is closed.")

    def _orientation_link(self, paths: Sequence[Path], first_priority: int) -> ChebyshevLink:
        entries: list[tuple[DafFile, DafSegment, int]] = []
        for path in paths:
            daf = DafFile(path)
            if daf.kind != "pck":
                raise ValueError(f"Orientation kernel is not a binary PCK: {path}")
            base = first_priority + len(entries)
            entries.extend((daf, segment, base + index) for index, segment in enumerate(daf.segments))
        targets = sorted({segment.target for _, segment, _ in entries})
        if len(targets) != 1:
            raise ValueError(f"Expected one lunar orientation target, found {targets}.")
        frames = sorted({segment.frame for _, segment, _ in entries})
        if frames != [_ICRF_FRAME]:
            raise ValueError(f"Lunar orientation must be relative to ICRF frame 1, found {frames}.")
        return _build_link(targets[0], _ICRF_FRAME, entries, 1.0, "PCK orientation")

    def _link(self, target: int, center: int) -> ChebyshevLink:
        link = self._links.get((target, center))
        if link is None:
            link = _build_link(target, center, self._segments[target][center], _KM_TO_M, "SPK")
            self._links[(target, center)] = link
        return link

    def lunar_euler_jd(self, jd1, jd2, *, order: int = 1) -> np.ndarray:
        """Return the 3-1-3 Euler angles of the lunar PA frame relative to ICRF at two-part TDB Julian Dates.

        The ICRF-to-PA rotation is ``Rz(psi) Rx(theta) Rz(phi)``. No longitude-libration correction is
        applied (see :attr:`lunar_orientation`).

        Parameters
        ----------
        jd1, jd2 : array_like
            Two-part TDB Julian Dates, broadcast together to ``(n_epochs,)``.
        order : {0, 1, 2}, default 1
            Highest time derivative returned.

        Returns
        -------
        ndarray, shape (n_epochs, 3 * (order + 1))
            (phi, theta, psi) in rad, then their rates in rad/s, then their second derivatives in rad/s^2.

        Raises
        ------
        ValueError
            If an epoch lies outside the orientation coverage.
        """
        self._require_open()
        a, f = split_epoch_seconds(jd1, jd2)
        try:
            return evaluate_states([self._orientation_plan], a, f, order)[:, 0]
        except EphemerisCoverageError as exc:
            raise _coverage_error(exc, a, f, "PCK coverage of the lunar orientation", self._orientation_plan) from None

    def _lunar_angles_rad(self, epoch_tdb: Epoch) -> np.ndarray:
        epoch = require_tdb_epoch(epoch_tdb, name="epoch_tdb")
        return self.lunar_euler_jd(epoch.jd1, epoch.jd2)[0]

    def _lunar_angles_many_rad(self, epochs) -> np.ndarray:
        jd1, jd2 = _jd_arrays(epochs)
        return self.lunar_euler_jd(jd1, jd2)

    def _plan(self, name: str, target: int) -> list[tuple[ChebyshevLink, float]]:
        plan = self._plans.get(target)
        if plan is not None:
            return plan
        plan = []
        current = target
        while current != 0:
            centers = self._segments.get(current)
            if not centers:
                if current == target:
                    barycenter = naif_id(f"{name} BARYCENTER") if f"{name} BARYCENTER" in _BARYCENTER_NAMES else None
                    hint = f"; try {name} BARYCENTER" if barycenter in self._segments else ""
                    raise KeyError(f"No SPK position record for {name!r} (NAIF {target}){hint}.")
                raise KeyError(f"No SPK position record for NAIF {current}, needed in the chain of {name!r}.")
            if len(centers) != 1:
                raise ValueError(
                    f"NAIF {current} has segments relative to several centres {sorted(centers)}; "
                    "time-dependent centre chains are not supported."
                )
            if len(plan) >= _MAX_CHAIN:
                raise ValueError(f"Centre chain of {name!r} does not reach the solar-system barycentre.")
            center = next(iter(centers))
            plan.append((self._link(current, center), 1.0))
            current = center
        self._plans[target] = plan
        return plan

    def _resolve(self, bodies: tuple) -> tuple[list[tuple[ChebyshevLink, float]], ...]:
        resolved = self._resolved_bodies.get(bodies)
        if resolved is None:
            plans = []
            for body in bodies:
                name = body_name(body)
                target = self._target_ids.get(name)
                if target is None:
                    target = naif_id(name)
                    if target is None:
                        raise KeyError(f"Unknown ephemeris body name: {body!r}")
                plans.append(self._plan(name, target))
                self._target_ids[name] = target
            resolved = tuple(plans)
            self._resolved_bodies[bodies] = resolved
        return resolved

    def body_states_jd(self, bodies: Sequence[str], jd1, jd2, *, order: int = 1) -> np.ndarray:
        """Return barycentric states of several bodies at two-part TDB Julian Dates.

        Parameters
        ----------
        bodies : sequence of str
            Body identifiers (see ``canonical_body_id``).
        jd1, jd2 : array_like
            Two-part TDB Julian Dates, broadcast together to ``(n_epochs,)``.
        order : {0, 1, 2}, default 1
            Highest time derivative: 0 position, 1 position and velocity, 2 also acceleration.

        Returns
        -------
        ndarray, shape (n_epochs, n_bodies, 3 * (order + 1))
            Position (m), then velocity (m/s), then acceleration (m/s^2), on ICRF axes.

        Raises
        ------
        KeyError
            If a body is unknown or has no position record in its centre chain.
        ValueError
            If an epoch lies outside the coverage of a needed segment, or a needed segment has an unsupported
            type or is not on ICRF.
        """
        self._require_open()
        bodies = tuple(bodies)
        plans = self._resolve(bodies)
        a, f = split_epoch_seconds(jd1, jd2)
        try:
            return evaluate_states(list(plans), a, f, order)
        except EphemerisCoverageError as exc:
            raise _coverage_error(
                exc, a, f, f"SPK coverage of {bodies[exc.body_index]!r}", plans[exc.body_index]
            ) from None

    def body_state_bcrs(self, body: str, epoch_tdb: Epoch) -> BodyState:
        """Return the barycentric BCRS state of ``body`` at a TDB epoch, in m and m/s on ICRF axes."""
        values = self.body_state_matrix_bcrs((body,), epoch_tdb)[0]
        return BodyState(values[:3], values[3:])

    def body_position_bcrs(self, body: str, epoch_tdb: Epoch) -> np.ndarray:
        """Return the barycentric BCRS position of ``body`` at a TDB epoch, shape (3,), in metres."""
        epoch = require_tdb_epoch(epoch_tdb, name="epoch_tdb")
        return self.body_states_jd((body,), epoch.jd1, epoch.jd2, order=0)[0, 0]

    def body_state_matrix_bcrs(self, bodies, epoch_tdb: Epoch) -> np.ndarray:
        """Return the barycentric states of several bodies at one TDB epoch.

        Parameters
        ----------
        bodies : sequence of str
            Body identifiers (see ``canonical_body_id``).
        epoch_tdb : Epoch
            Epoch on the TDB scale.

        Returns
        -------
        ndarray, shape (n_bodies, 6)
            Rows (x, y, z, vx, vy, vz) on ICRF axes, in m and m/s.
        """
        epoch = require_tdb_epoch(epoch_tdb, name="epoch_tdb")
        return self.body_states_jd(bodies, epoch.jd1, epoch.jd2)[0]

    def body_state_matrices_bcrs(self, bodies, epochs) -> np.ndarray:
        """Return the barycentric states of several bodies at several TDB epochs.

        Parameters
        ----------
        bodies : sequence of str
            Body identifiers (see ``canonical_body_id``).
        epochs : iterable of Epoch
            Epochs on the TDB scale.

        Returns
        -------
        ndarray, shape (n_epochs, n_bodies, 6)
            States (x, y, z, vx, vy, vz) relative to the solar-system barycentre on ICRF axes, in m and m/s.

        Raises
        ------
        ValueError
            If an epoch is not on the TDB scale or outside the kernel coverage.
        KeyError
            If a body is unknown or has no position record in the kernels.
        """
        jd1, jd2 = _jd_arrays(epochs)
        return self.body_states_jd(bodies, jd1, jd2)

    def body_accelerations_bcrs(self, body: str, epochs) -> np.ndarray:
        """Return the barycentric acceleration of one body at several TDB epochs.

        Parameters
        ----------
        body : str
            Body identifier (see ``canonical_body_id``).
        epochs : iterable of Epoch
            Epochs on the TDB scale.

        Returns
        -------
        ndarray, shape (n_epochs, 3)
            Second TDB derivative of the barycentric position on ICRF axes, in m/s^2.
        """
        jd1, jd2 = _jd_arrays(epochs)
        return np.ascontiguousarray(self.body_states_jd((body,), jd1, jd2, order=2)[:, 0, 6:9])

    def close(self) -> None:
        """Release the kernels; further state requests raise ``ValueError``."""
        self._files = None
        self._orientation_plan = []
        self._segments = {}
        self._links = {}
        self._plans = {}
        self._resolved_bodies = {}

    def __enter__(self) -> Self:
        """Return the ephemeris for use as a context manager."""
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        """Close the ephemeris on leaving the context."""
        self.close()


_BARYCENTER_NAMES = frozenset(
    f"{name} BARYCENTER" for name in ("MERCURY", "VENUS", "MARS", "JUPITER", "SATURN", "URANUS", "NEPTUNE", "PLUTO")
)


def _build_link(target: int, center: int, entries, scale: float, label: str) -> ChebyshevLink:
    records, init, interval, positions, velocities, coverage = [], [], [], [], [], []
    for daf, segment, priority in entries:
        where = f"{label} segment of NAIF {target} relative to {center}"
        if segment.frame != _ICRF_FRAME:
            raise ValueError(f"{where} is on frame {segment.frame}, not ICRF (1): {daf.path}")
        try:
            layout = chebyshev_layout(daf, segment)
        except ValueError as exc:
            raise ValueError(f"{where}: {exc}") from None
        records.append(layout.records)
        init.append(layout.init_s)
        interval.append(layout.interval_s)
        positions.append(layout.position_count)
        velocities.append(layout.velocity_count)
        coverage.append((segment.start_s, segment.end_s, priority))
    bounds, owners, priorities = coverage_partition(coverage)
    return ChebyshevLink(
        target, center, records, init, interval, positions, velocities, bounds, owners, priorities, scale
    )


def _coverage_error(exc, a: np.ndarray, f: np.ndarray, what: str, plan) -> ValueError:
    epoch = exc.epoch_index
    jd = _J2000_JD + (a[epoch] + f[epoch]) / _SECONDS_PER_DAY
    link = next(link for link, _ in plan if (link.target, link.center) == (exc.target, exc.center))
    first, last = (_J2000_JD + value / _SECONDS_PER_DAY for value in link.coverage_s)
    return ValueError(
        f"Epoch JD(TDB) {jd:.9f} is outside the {what} "
        f"(NAIF {exc.target} relative to {exc.center}; this link covers JD {first:.1f}..{last:.1f})."
    )


def _jd_arrays(epochs) -> tuple[np.ndarray, np.ndarray]:
    epochs = tuple(require_tdb_epoch(epoch, name="epoch_tdb") for epoch in epochs)
    return (
        np.fromiter((epoch.jd1 for epoch in epochs), dtype=float, count=len(epochs)),
        np.fromiter((epoch.jd2 for epoch in epochs), dtype=float, count=len(epochs)),
    )


__all__ = ["ChebyshevEphemeris", "coverage_partition", "split_epoch_seconds"]
