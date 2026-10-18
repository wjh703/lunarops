"""Dataset orchestration for LLR measurement evaluation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import cast

import numpy as np
from tqdm import tqdm as _tqdm  # type: ignore[import-untyped]

from lunarops.classes.time import format_time_with_utc_offset, validate_utc_offset_hours

from .equations import ObservationEquation, ObservationResultDetail
from .measurement import LlrObservationModel
from .normal_points import NptDataset
from .resolver import ObservationCatalogSelection, ObservationResolver, ResolvedObservation


@dataclass(frozen=True, slots=True)
class ObservationProcessingOptions:
    station_identifier: str | None = None
    reflector_identifier: str | None = None
    min_elevation_deg: float = 0.0
    low_elevation_policy: str = "rejectClamped"
    utc_offset_hours: float = 0.0
    include_reflector_position_partials: bool = False
    show_progress: bool = False
    progress_description: str | None = None

    def __post_init__(self) -> None:
        if self.low_elevation_policy not in {"rejectClamped", "allowFlagged"}:
            raise ValueError("low_elevation_policy must be rejectClamped or allowFlagged")
        min_elevation = float(self.min_elevation_deg)
        if not np.isfinite(min_elevation):
            raise ValueError("min_elevation_deg must be finite.")
        object.__setattr__(self, "min_elevation_deg", min_elevation)
        object.__setattr__(self, "utc_offset_hours", validate_utc_offset_hours(self.utc_offset_hours))

    @property
    def catalog_selection(self) -> ObservationCatalogSelection:
        return ObservationCatalogSelection(self.station_identifier, self.reflector_identifier)

    def with_progress(
        self,
        description: str | None,
        *,
        enabled: bool | None = None,
    ) -> ObservationProcessingOptions:
        return replace(
            self,
            progress_description=description,
            show_progress=self.show_progress if enabled is None else bool(enabled),
        )


class LlrObservationProcessor:
    def __init__(self, resolver: ObservationResolver, observation_model: LlrObservationModel) -> None:
        self.resolver = resolver
        self.model_state = resolver.model_state
        self.observation_model = observation_model

    def _with_progress(
        self,
        observations: Iterable[ResolvedObservation],
        *,
        total: int,
        options: ObservationProcessingOptions,
    ) -> Iterable[ResolvedObservation]:
        if not options.show_progress or total <= 0:
            return observations
        description = options.progress_description or "LLR observations"
        return _tqdm(
            observations,
            total=total,
            desc=description,
            unit="np",
            dynamic_ncols=True,
            smoothing=0.1,
        )

    def _resolved_observations(
        self,
        dataset: NptDataset,
        options: ObservationProcessingOptions,
    ) -> Iterable[ResolvedObservation]:
        observations = self.resolver.resolve_all(dataset.records, options.catalog_selection)
        return self._with_progress(
            observations,
            total=len(observations),
            options=options,
        )

    def equations(
        self,
        dataset: NptDataset,
        *,
        options: ObservationProcessingOptions | None = None,
    ) -> list[ObservationEquation]:
        options = options or ObservationProcessingOptions()
        evaluations = [
            self.observation_model.evaluate(
                observation,
                min_elevation_deg=options.min_elevation_deg,
                low_elevation_policy=options.low_elevation_policy,
                include_reflector_position_partials=(options.include_reflector_position_partials),
            )
            for observation in self._resolved_observations(dataset, options)
        ]
        return [evaluation.equation for evaluation in evaluations if not evaluation.below_elevation_limit]

    def rows(
        self,
        dataset: NptDataset,
        *,
        options: ObservationProcessingOptions | None = None,
        detail: ObservationResultDetail = ObservationResultDetail.STANDARD,
    ) -> list[dict[str, object]]:
        options = options or ObservationProcessingOptions()
        rows: list[dict[str, object]] = []
        for observation in self._resolved_observations(dataset, options):
            evaluation = self.observation_model.evaluate(
                observation,
                min_elevation_deg=options.min_elevation_deg,
                low_elevation_policy=options.low_elevation_policy,
                include_reflector_position_partials=(options.include_reflector_position_partials),
                result_detail=detail,
            )
            if evaluation.below_elevation_limit:
                continue
            epoch_utc = evaluation.equation.transmit_epoch_utc
            rows.append(
                {
                    "obs_time_utc": format_time_with_utc_offset(epoch_utc, utc_offset_hours=0.0, precision=9),
                    **cast(dict[str, object], evaluation.result_row),
                    "obs_time_local": format_time_with_utc_offset(
                        epoch_utc,
                        utc_offset_hours=options.utc_offset_hours,
                        precision=9,
                    ),
                }
            )
        return rows


__all__ = ["LlrObservationProcessor", "ObservationProcessingOptions"]
