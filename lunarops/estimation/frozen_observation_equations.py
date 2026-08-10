"""Frozen observation-equation models and normal-equation conversion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence, cast

import numpy as np

from lunarops.classes.time import Epoch, TimeScale
from lunarops.base.parameter_name import ParameterName, parameter_unit
from lunarops.base.serialization import plain_data
from lunarops.classes.observation.equations import ObservationEquation
from lunarops.classes.parametrization.base import ParametrizationList


@dataclass(frozen=True, slots=True, eq=False)
class FrozenObservationEquations:
    """Immutable fixed-linearization rows used by later estimation programs."""

    parameter_names: tuple[ParameterName, ...]
    parameter_units: tuple[str, ...]
    design: np.ndarray
    reduced_observations: np.ndarray
    sigmas: np.ndarray
    identities: tuple[int, ...]
    sources: tuple[str, ...]
    epochs: tuple[Epoch, ...]
    station_keys: tuple[str, ...]
    reflector_keys: tuple[str, ...]
    light_time_converged: tuple[bool, ...]
    wavelengths_nm: tuple[float | None, ...]
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        names = tuple(self.parameter_names)
        if not all(isinstance(name, ParameterName) for name in names):
            raise TypeError("Frozen observation parameter names must be ParameterName objects.")
        units = tuple(str(unit).strip() for unit in self.parameter_units)
        design = np.array(self.design, dtype=float, copy=True)
        observations = np.array(self.reduced_observations, dtype=float, copy=True).reshape(-1)
        sigmas = np.array(self.sigmas, dtype=float, copy=True).reshape(-1)
        count = observations.size
        sequences = (
            self.identities,
            self.sources,
            self.epochs,
            self.station_keys,
            self.reflector_keys,
            self.light_time_converged,
            self.wavelengths_nm,
        )
        if design.shape != (count, len(names)):
            raise ValueError("Frozen observation design shape is inconsistent.")
        if sigmas.size != count or any(len(values) != count for values in sequences):
            raise ValueError("Frozen observation row arrays must have equal length.")
        if len(units) != len(names) or any(not unit for unit in units) or len(set(names)) != len(names):
            raise ValueError("Frozen observation parameter names/units are inconsistent.")
        if any(
            isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) for value in self.identities
        ):
            raise TypeError("Frozen observation identities must be integers.")
        identities = tuple(int(value) for value in self.identities)
        if len(set(identities)) != count:
            raise ValueError("Frozen observation identities must be unique integers.")
        if not np.all(np.isfinite(design)) or not np.all(np.isfinite(observations)):
            raise ValueError("Frozen observation equations must be finite.")
        if not np.all(np.isfinite(sigmas)) or np.any(sigmas <= 0.0):
            raise ValueError("Frozen observation sigmas must be positive and finite.")
        epochs = tuple(self.epochs)
        for epoch in epochs:
            if not isinstance(epoch, Epoch):
                raise TypeError("Frozen observation epochs must be Epoch objects.")
            epoch.require_scale(TimeScale.UTC, name="observation epoch")
        sources = tuple(str(value).strip() for value in self.sources)
        station_keys = tuple(str(value).strip() for value in self.station_keys)
        reflector_keys = tuple(str(value).strip() for value in self.reflector_keys)
        if any(not value for values in (sources, station_keys, reflector_keys) for value in values):
            raise ValueError("Frozen observation source/station/reflector names must not be empty.")
        if any(not isinstance(value, (bool, np.bool_)) for value in self.light_time_converged):
            raise TypeError("Frozen observation convergence flags must be booleans.")
        light_time_converged = tuple(bool(value) for value in self.light_time_converged)
        wavelengths: list[float | None] = []
        for raw_value in self.wavelengths_nm:
            value = None if raw_value is None else float(raw_value)
            if value is not None and (not np.isfinite(value) or value <= 0.0):
                raise ValueError("Frozen observation wavelengths must be positive and finite.")
            wavelengths.append(value)
        metadata = cast(dict[str, object], plain_data(dict(self.metadata)))
        compatibility = metadata.get("compatibility")
        if (
            not isinstance(compatibility, str)
            or len(compatibility) != 64
            or any(character not in "0123456789abcdef" for character in compatibility)
        ):
            raise ValueError("Frozen observation equations require a lowercase SHA-256 compatibility fingerprint.")
        design.setflags(write=False)
        observations.setflags(write=False)
        sigmas.setflags(write=False)
        object.__setattr__(self, "parameter_names", names)
        object.__setattr__(self, "parameter_units", units)
        object.__setattr__(self, "design", design)
        object.__setattr__(self, "reduced_observations", observations)
        object.__setattr__(self, "sigmas", sigmas)
        object.__setattr__(self, "identities", identities)
        object.__setattr__(self, "sources", sources)
        object.__setattr__(self, "epochs", epochs)
        object.__setattr__(self, "station_keys", station_keys)
        object.__setattr__(self, "reflector_keys", reflector_keys)
        object.__setattr__(self, "light_time_converged", light_time_converged)
        object.__setattr__(self, "wavelengths_nm", tuple(wavelengths))
        object.__setattr__(self, "metadata", metadata)

    @classmethod
    def from_equations(
        cls,
        equations: Sequence[ObservationEquation],
        parametrization: ParametrizationList,
        *,
        source_by_identity: Mapping[int, str] | None = None,
        metadata: Mapping[str, object],
    ) -> "FrozenObservationEquations":
        rows = list(equations)
        if not rows:
            raise ValueError("Cannot freeze an empty observation-equation sequence.")
        names = parametrization.parameter_names()
        source_map = source_by_identity or {}
        raw_identities = tuple(equation.observation_id for equation in rows)
        if any(isinstance(identity, bool) or not isinstance(identity, int) for identity in raw_identities):
            raise TypeError("Frozen observation equations require integer observation IDs.")
        identities = cast(tuple[int, ...], raw_identities)
        return cls(
            parameter_names=tuple(names),
            parameter_units=tuple(parameter_unit(name) for name in names),
            design=np.vstack([parametrization.design_row(equation) for equation in rows]),
            reduced_observations=np.asarray([parametrization.reduced_observation(equation) for equation in rows]),
            sigmas=np.asarray([equation.sigma_one_way_m for equation in rows]),
            identities=tuple(identities),
            sources=tuple(str(source_map.get(identity, "unknown")) for identity in identities),
            epochs=tuple(equation.transmit_epoch_utc for equation in rows),
            station_keys=tuple(equation.station_key for equation in rows),
            reflector_keys=tuple(equation.reflector_key for equation in rows),
            light_time_converged=tuple(equation.light_time_converged for equation in rows),
            wavelengths_nm=tuple(equation.wavelength_nm for equation in rows),
            metadata=dict(metadata),
        )

    def normal_equations(self):
        """Build a weighted normal-equation system from the frozen rows."""
        from .normal_equations import NormalEquations

        with np.errstate(over="raise", divide="raise", invalid="raise"):
            weights = 1.0 / (self.sigmas * self.sigmas)
            weighted_design = weights[:, None] * self.design
            matrix = self.design.T @ weighted_design
            rhs = self.design.T @ (weights * self.reduced_observations)
            lpl = float(np.dot(weights, self.reduced_observations**2))
        matrix = 0.5 * (matrix + matrix.T)
        if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(rhs)) or not np.isfinite(lpl):
            raise FloatingPointError("Frozen observation normal-equation accumulation produced non-finite values.")
        return NormalEquations(
            parameter_names=list(self.parameter_names),
            parameter_units=list(self.parameter_units),
            N=matrix,
            W=rhs,
            lPl=lpl,
            obs_count=len(self.identities),
            meta={**dict(self.metadata), "source": "FrozenObservationEquations"},
        )


__all__ = ["FrozenObservationEquations"]
