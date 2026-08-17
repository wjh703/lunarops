"""CALCEPH implementation of the LLR ephemeris interface."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lunarops.base.array_validation import readonly_matrix3x3
from lunarops.classes.relativistic import (
    LunarRelativisticScaleConvention,
    l_b_minus_l_l_for_convention,
    normalize_lunar_relativistic_scale_convention,
)
from lunarops.classes.time import Epoch, TimeScale, TimeScaleConverter

from .base import (
    BodyState,
    Ephemeris,
    LongitudeLibrationCorrectionType,
    require_tdb_epoch,
)
from .longitude_libration import (
    LongitudeLibrationCorrectionModel,
    make_longitude_libration_correction_model,
    normalize_longitude_libration_correction_type,
)

_BODY_ID_BY_NAME = {
    "SSB": 0,
    "SOLAR SYSTEM BARYCENTER": 0,
    "MERCURY BARYCENTER": 1,
    "VENUS BARYCENTER": 2,
    "EARTH MOON BARYCENTER": 3,
    "EARTH BARYCENTER": 3,
    "MARS BARYCENTER": 4,
    "JUPITER BARYCENTER": 5,
    "SATURN BARYCENTER": 6,
    "URANUS BARYCENTER": 7,
    "NEPTUNE BARYCENTER": 8,
    "PLUTO BARYCENTER": 9,
    "SUN": 10,
    "MOON": 301,
    "EARTH": 399,
}
_J2000_TT_JD1 = 2451545.0
_J2000_TT_JD2 = 0.0


def _passive_rotation_z(angle_rad: float) -> np.ndarray:
    cosine, sine = np.cos(angle_rad), np.sin(angle_rad)
    return np.array(
        [[cosine, sine, 0.0], [-sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )


def _passive_rotation_x(angle_rad: float) -> np.ndarray:
    cosine, sine = np.cos(angle_rad), np.sin(angle_rad)
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, cosine, sine], [0.0, -sine, cosine]],
        dtype=float,
    )


class CalcephEphemeris(Ephemeris):
    """INPOP/DE binary ephemeris read through :mod:`calcephpy`."""

    _LIBRATION_TARGET = 15
    _TT_MINUS_TDB_TARGET = 16

    def __init__(
        self,
        ephemeris_file: str | Path,
        *,
        lunar_relativistic_scale_convention: LunarRelativisticScaleConvention | str,
        longitude_libration_correction_type: (LongitudeLibrationCorrectionType | str | None) = None,
    ) -> None:
        path = Path(ephemeris_file).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"CALCEPH ephemeris file not found: {path}")
        try:
            from calcephpy import CalcephBin, Constants
        except (ImportError, OSError) as exc:  # pragma: no cover
            raise ImportError(
                f"The CALCEPH ephemeris requires a working calcephpy installation. Original import error: {exc}"
            ) from exc

        self._source_file = path
        self._lunar_relativistic_scale_convention = normalize_lunar_relativistic_scale_convention(
            lunar_relativistic_scale_convention
        )
        self._l_b_minus_l_l = l_b_minus_l_l_for_convention(
            self._lunar_relativistic_scale_convention
        )
        self._longitude_libration_correction_type = normalize_longitude_libration_correction_type(
            longitude_libration_correction_type
        )
        self._longitude_libration_correction_model: LongitudeLibrationCorrectionModel = (
            make_longitude_libration_correction_model(self._longitude_libration_correction_type)
        )
        self._j2000_tdb: Epoch | None = None
        self._handle = CalcephBin.open(str(path))
        self._state_units = Constants.UNIT_KM + Constants.UNIT_SEC + Constants.USE_NAIFID
        self._angle_units = Constants.UNIT_RAD + Constants.UNIT_SEC

    @property
    def source_file_path(self) -> Path:
        return self._source_file

    @property
    def l_b_minus_l_l(self) -> float:
        return self._l_b_minus_l_l

    @property
    def lunar_relativistic_scale_convention(
        self,
    ) -> LunarRelativisticScaleConvention:
        return self._lunar_relativistic_scale_convention

    @property
    def longitude_libration_correction_type(
        self,
    ) -> LongitudeLibrationCorrectionType:
        return self._longitude_libration_correction_type

    def _require_open_handle(self):
        if self._handle is None:
            raise RuntimeError("CALCEPH ephemeris is closed.")
        return self._handle

    def close(self) -> None:
        handle = self._handle
        self._handle = None
        if handle is not None:
            try:
                handle.close()
            except Exception as exc:  # pragma: no cover - backend cleanup detail
                raise RuntimeError("CALCEPH ephemeris close() failed.") from exc

    def body_state_bcrs(self, body_name: str, epoch_tdb: Epoch) -> BodyState:
        epoch_tdb = require_tdb_epoch(epoch_tdb, name="epoch_tdb")
        normalized_body_name = str(body_name).strip().upper()
        try:
            target_id = _BODY_ID_BY_NAME[normalized_body_name]
        except KeyError:
            raise KeyError(f"Unknown CALCEPH body name: {body_name!r}") from None
        values = self._require_open_handle().compute_unit(
            epoch_tdb.jd1,
            epoch_tdb.jd2,
            target_id,
            0,
            self._state_units,
        )
        state = np.asarray(values, dtype=float)
        if state.size < 6:
            raise RuntimeError(
                f"CALCEPH returned {state.size} state values for {normalized_body_name}; expected at least six."
            )
        return BodyState(
            position_m=state[:3] * 1000.0,
            velocity_mps=state[3:6] * 1000.0,
        )

    def _lunar_orientation_angles_rad(self, epoch_tdb: Epoch) -> np.ndarray:
        epoch_tdb = require_tdb_epoch(epoch_tdb, name="epoch_tdb")
        values = self._require_open_handle().compute_unit(
            epoch_tdb.jd1,
            epoch_tdb.jd2,
            self._LIBRATION_TARGET,
            0,
            self._angle_units,
        )
        angles = np.asarray(values, dtype=float)
        if angles.size < 3:
            raise RuntimeError("CALCEPH libration target returned fewer than three angles.")
        result = np.array(angles[:3], dtype=float, copy=True)
        result.setflags(write=False)
        return result

    def _j2000_tdb_epoch(self) -> Epoch:
        if self._j2000_tdb is None:
            converter = TimeScaleConverter()
            self._j2000_tdb = converter.tt2tdb(Epoch(_J2000_TT_JD1, _J2000_TT_JD2, TimeScale.TT))
        return self._j2000_tdb

    def longitude_libration_correction_rad(self, epoch_tdb: Epoch) -> float:
        epoch_tdb = require_tdb_epoch(epoch_tdb, name="epoch_tdb")
        if self.longitude_libration_correction_type is LongitudeLibrationCorrectionType.NONE:
            return 0.0
        return self._longitude_libration_correction_model.correction_rad(
            epoch_tdb,
            j2000_epoch_tdb=self._j2000_tdb_epoch(),
        )

    def pa2lcrs_matrix(self, epoch_tdb: Epoch) -> np.ndarray:
        epoch_tdb = require_tdb_epoch(epoch_tdb, name="epoch_tdb")
        phi, theta, psi = self._lunar_orientation_angles_rad(epoch_tdb)
        psi += self.longitude_libration_correction_rad(epoch_tdb)
        lcrs_to_pa_matrix = _passive_rotation_z(psi) @ _passive_rotation_x(theta) @ _passive_rotation_z(phi)
        return readonly_matrix3x3(
            lcrs_to_pa_matrix.T,
            name="pa2lcrs_matrix",
        )

    def target16_tdb_minus_tt_s(self, epoch_tdb: Epoch) -> float:
        """Read target 16 for diagnostic comparison with ERFA only."""

        epoch_tdb = require_tdb_epoch(epoch_tdb, name="epoch_tdb")
        try:
            values = self._require_open_handle().compute_unit(
                epoch_tdb.jd1,
                epoch_tdb.jd2,
                self._TT_MINUS_TDB_TARGET,
                0,
                self._angle_units,
            )
        except Exception as exc:
            raise RuntimeError(
                f"CALCEPH failed while reading target 16 (TT−TDB) at jd=({epoch_tdb.jd1}, {epoch_tdb.jd2})."
            ) from exc
        values = np.asarray(values, dtype=float)
        if values.size < 1 or not np.isfinite(values[0]):
            raise RuntimeError("CALCEPH target 16 returned an invalid TT−TDB value.")
        # CALCEPH target 16 stores TT−TDB; this diagnostic returns TDB−TT.
        return -float(values[0])


def load_calceph_ephemeris(
    ephemeris_file: str | Path,
    *,
    lunar_relativistic_scale_convention: LunarRelativisticScaleConvention | str,
    longitude_libration_correction_type: (LongitudeLibrationCorrectionType | str | None) = None,
) -> CalcephEphemeris:
    return CalcephEphemeris(
        ephemeris_file,
        lunar_relativistic_scale_convention=lunar_relativistic_scale_convention,
        longitude_libration_correction_type=longitude_libration_correction_type,
    )


__all__ = ["CalcephEphemeris", "load_calceph_ephemeris"]
