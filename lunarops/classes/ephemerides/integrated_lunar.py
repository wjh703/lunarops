"""Ephemeris whose Earth-Moon relative motion and lunar orientation come from a propagation.

``LunarOrbitPropagation`` integrates the Moon-minus-Earth state and the lunar
rotation about the Earth-Moon barycentre of an ephemeris and writes them as
the ``lunarOrbit`` and ``lunarRotation`` tables. :class:`IntegratedLunarEphemeris`
wraps that ephemeris and answers every query from it, except that the Earth
and the Moon are placed about its barycentre from the propagated relative
state, and that the lunar orientation is the propagated one. The observation
model then ranges to the propagated Moon without any other change.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar

import numpy as np

from lunarops.base.interpolation import interpolate_regular_grid
from lunarops.classes.time import Epoch, TimeScale, require_tdb_epoch
from lunarops.config.schema import ConfigSchema, class_config, integer, path
from lunarops.fileio.mass_catalog import read_mass_catalog
from lunarops.fileio.numeric_table import read_numeric_table

from .body_ids import body_name
from .ephemeris_provider import BodyState
from .lunar_orientation import EulerAngleLunarOrientation, LongitudeLibrationCorrection

_EMB = "EARTH MOON BARYCENTER"
_RELATIVE_COLUMNS = tuple(f"relative_{axis}" for axis in ("x_m", "y_m", "z_m", "vx_mps", "vy_mps", "vz_mps"))
_ANGLE_COLUMNS = tuple(f"euler_{angle}_rad" for angle in ("phi", "theta", "psi"))
_RATE_COLUMNS = tuple(f"mantle_omega_pa_{axis}_rad_s" for axis in "xyz")


class _RegularTable:
    """Lagrange interpolation of the columns of a regularly sampled table.

    See :func:`lunarops.base.interpolation.interpolate_regular_grid`; offsets up to 1e-6 s beyond the ends are
    extrapolated with the end stencil.
    """

    def __init__(self, offsets: np.ndarray, values: np.ndarray, order: int, name: str) -> None:
        if len(offsets) < 2:
            raise ValueError(f"The {name} table needs at least two rows")
        self._offsets = np.ascontiguousarray(offsets, dtype=float)
        self._first = float(offsets[0])
        self._step = float(offsets[1] - offsets[0])
        self._last = float(offsets[-1])
        if self._step <= 0 or not np.allclose(np.diff(offsets), self._step, rtol=0.0, atol=1e-9 * self._step):
            raise ValueError(f"The {name} table is not sampled on a regular, increasing grid")
        self._values = np.ascontiguousarray(values, dtype=float)
        self._count = min(order + 1, len(offsets))
        self._name = name

    @property
    def span(self) -> tuple[float, float]:
        return self._first, self._last

    def __call__(self, offset: float) -> np.ndarray:
        if not self._first - 1e-6 <= offset <= self._last + 1e-6:
            raise ValueError(
                f"Offset {offset:.3f} s is outside the {self._name} table, which covers "
                f"{self._first:.3f} to {self._last:.3f} s from its initial epoch"
            )
        return interpolate_regular_grid(self._offsets, self._values, offset, self._count)[0]


class IntegratedLunarEphemeris:
    """Ephemeris with the propagated Earth-Moon relative state and lunar orientation.

    Parameters
    ----------
    ephemeris : Ephemeris
        Ephemeris the propagation used: the source of the Earth-Moon barycentre and of every other body.
    orbit_path : str or Path
        ``lunarOrbit`` table of the propagation.
    rotation_path : str or Path
        ``lunarRotation`` table of the same propagation.
    earth_gm_m3_s2, moon_gm_m3_s2 : float
        Earth and Moon GM of the propagation; only their ratio is used, to place the two bodies about the
        barycentre as the propagation does.
    interpolation_order : int, default 13
        Order of the Lagrange interpolation of the tables.

    Raises
    ------
    ValueError
        If the tables lack the relative state, the Euler angles or the mantle angular velocity, are not
        regularly sampled, or differ in their initial epoch or sampling.

    Notes
    -----
    The Euler angles are the 3-1-3 angles (phi, theta, psi) of the integration state, PA to ICRF axes
    ``Rz(phi) Rx(theta) Rz(psi)``, the convention of the binary PCK kernels; their rates follow from the
    mantle angular velocity (Folkner et al. 2014, Eq. 14). Queries outside the propagated span raise.
    Accelerations are those of the wrapped ephemeris.
    """

    SCHEMA: ClassVar[ConfigSchema] = ConfigSchema(
        fields=(
            class_config(
                "ephemerides",
                "ephemerides",
                required=True,
                description=(
                    "Ephemeris the propagation used; it supplies the Earth-Moon barycentre and every other body."
                ),
            ),
            path(
                "inputFileOrbit",
                required=True,
                non_empty=True,
                description="`lunarOrbit` table written by `LunarOrbitPropagation`.",
            ),
            path(
                "inputFileRotation",
                required=True,
                non_empty=True,
                description="`lunarRotation` table written by the same propagation.",
            ),
            path(
                "inputFileMassCatalog",
                required=True,
                non_empty=True,
                description=(
                    "Mass catalog of the propagation; the Earth/Moon GM ratio places the two bodies about the "
                    "barycentre."
                ),
            ),
            integer(
                "interpolationOrder",
                default=13,
                ge=1,
                description="Order of the Lagrange interpolation of the propagated tables.",
            ),
        ),
        type_name="integratedLunar",
    )

    @classmethod
    def from_config(cls, cfg, ctx):
        """Create the ephemeris from its validated configuration section."""
        catalog = read_mass_catalog(ctx.resolve_path(cfg["inputFileMassCatalog"]))
        return cls(
            ctx.create_class("ephemerides", cfg["ephemerides"]),
            ctx.resolve_path(cfg["inputFileOrbit"]),
            ctx.resolve_path(cfg["inputFileRotation"]),
            catalog.require("EARTH").gm_m3_s2,
            catalog.require("MOON").gm_m3_s2,
            interpolation_order=cfg["interpolationOrder"],
        )

    def __init__(
        self,
        ephemeris,
        orbit_path: str | Path,
        rotation_path: str | Path,
        earth_gm_m3_s2: float,
        moon_gm_m3_s2: float,
        *,
        interpolation_order: int = 13,
    ) -> None:
        self._inner = ephemeris
        total = float(earth_gm_m3_s2) + float(moon_gm_m3_s2)
        self._moon_fraction = float(moon_gm_m3_s2) / total
        self._earth_fraction = float(earth_gm_m3_s2) / total
        orbit = self._read(orbit_path, "lunarOrbit", _RELATIVE_COLUMNS)
        rotation = self._read(rotation_path, "lunarRotation", (*_ANGLE_COLUMNS, *_RATE_COLUMNS))
        if orbit[0] != rotation[0]:
            raise ValueError("The orbit and rotation tables have different initial epochs")
        self._initial_epoch = Epoch(orbit[0][0], orbit[0][1], TimeScale.TDB)
        self._relative = _RegularTable(orbit[1], orbit[2], interpolation_order, "lunarOrbit")
        self._rotation = _RegularTable(rotation[1], rotation[2], interpolation_order, "lunarRotation")
        self.lunar_orientation = EulerAngleLunarOrientation(
            self._angles, self._angles_many, LongitudeLibrationCorrection.NONE, None
        )

    @staticmethod
    def _read(table_path, artifact_type: str, names: tuple[str, ...]):
        _, columns, values, metadata = read_numeric_table(table_path, artifact_type)
        missing = [name for name in ("offset_tdb_s", *names) if name not in columns]
        if missing:
            raise ValueError(f"The {artifact_type} table {table_path} lacks the columns {missing}")
        if "initialTdbJd1" not in metadata or "initialTdbJd2" not in metadata:
            raise ValueError(f"The {artifact_type} table {table_path} does not state its initial epoch")
        offsets = values[:, columns.index("offset_tdb_s")]
        selected = values[:, [columns.index(name) for name in names]]
        return (float(metadata["initialTdbJd1"]), float(metadata["initialTdbJd2"])), offsets, selected

    # ------------------------------------------------------------- propagated

    @property
    def initial_epoch(self) -> Epoch:
        """Initial TDB epoch of the propagation."""
        return self._initial_epoch

    @property
    def span_s(self) -> tuple[float, float]:
        """Offsets from :attr:`initial_epoch`, in s, between which both tables answer."""
        orbit, rotation = self._relative.span, self._rotation.span
        return max(orbit[0], rotation[0]), min(orbit[1], rotation[1])

    def _offset(self, epoch_tdb: Epoch) -> float:
        return float(self._initial_epoch.seconds_until(require_tdb_epoch(epoch_tdb, name="epoch_tdb")))

    def _angles(self, epoch_tdb: Epoch) -> np.ndarray:
        phi, theta, psi, wx, wy, wz = self._rotation(self._offset(epoch_tdb))
        phi_dot = (wx * np.sin(psi) + wy * np.cos(psi)) / np.sin(theta)
        theta_dot = wx * np.cos(psi) - wy * np.sin(psi)
        return np.array([phi, theta, psi, phi_dot, theta_dot, wz - phi_dot * np.cos(theta)])

    def _angles_many(self, epochs) -> np.ndarray:
        return np.asarray([self._angles(epoch) for epoch in epochs]).reshape(-1, 6)

    # ------------------------------------------------------------- ephemeris

    @property
    def source_path(self) -> Path | None:
        """Source of the wrapped ephemeris."""
        return self._inner.source_path

    @property
    def relativistic_scale(self):
        """Lunar relativistic coordinate scaling of the wrapped ephemeris."""
        return self._inner.relativistic_scale

    def relative_state_matrix(self, target: str, center: str, epoch_tdb: Epoch) -> np.ndarray:
        """Return the state of ``target`` relative to ``center``; the propagated one for the Moon and the Earth."""
        pair = (body_name(target), body_name(center))
        if pair == ("MOON", "EARTH"):
            return self._relative(self._offset(epoch_tdb))
        if pair == ("EARTH", "MOON"):
            return -self._relative(self._offset(epoch_tdb))
        states = self.body_state_matrix_bcrs(pair, epoch_tdb)
        return states[0] - states[1]

    def relative_state_matrices(self, target: str, center: str, epochs: Sequence[Epoch]) -> np.ndarray:
        """Return :meth:`relative_state_matrix` at several epochs, shape ``(n_epochs, 6)``."""
        return np.asarray([self.relative_state_matrix(target, center, epoch) for epoch in epochs]).reshape(-1, 6)

    def body_state_matrix_bcrs(self, bodies: Sequence[str], epoch_tdb: Epoch) -> np.ndarray:
        """Return the BCRS states of ``bodies`` at one TDB epoch, shape ``(n_bodies, 6)``.

        The Earth and the Moon are placed about the barycentre of the wrapped ephemeris from the propagated
        relative state; every other body is that of the wrapped ephemeris.
        """
        names = tuple(body_name(body) for body in bodies)
        if "EARTH" not in names and "MOON" not in names:
            return self._inner.body_state_matrix_bcrs(names, epoch_tdb)
        others = tuple(dict.fromkeys(name for name in names if name not in ("EARTH", "MOON")))
        states = np.asarray(self._inner.body_state_matrix_bcrs((_EMB, *others), epoch_tdb), dtype=float)
        relative = self._relative(self._offset(epoch_tdb))
        rows = {name: states[1 + index] for index, name in enumerate(others)}
        rows["EARTH"] = states[0] - self._moon_fraction * relative
        rows["MOON"] = states[0] + self._earth_fraction * relative
        return np.asarray([rows[name] for name in names])

    def body_state_matrices_bcrs(self, bodies: Sequence[str], epochs: Sequence[Epoch]) -> np.ndarray:
        """Return :meth:`body_state_matrix_bcrs` at several epochs, shape ``(n_epochs, n_bodies, 6)``."""
        names = tuple(body_name(body) for body in bodies)
        if "EARTH" not in names and "MOON" not in names:
            return self._inner.body_state_matrices_bcrs(names, epochs)
        return np.asarray([self.body_state_matrix_bcrs(names, epoch) for epoch in epochs]).reshape(-1, len(names), 6)

    def body_state_bcrs(self, body: str, epoch_tdb: Epoch) -> BodyState:
        """Return the BCRS state of one body at one TDB epoch."""
        state = self.body_state_matrix_bcrs((body,), epoch_tdb)[0]
        return BodyState(state[:3].copy(), state[3:].copy())

    def close(self) -> None:
        """Do nothing: the wrapped ephemeris is closed by the context that created it."""
