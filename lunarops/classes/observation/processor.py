"""Dataset orchestration for LLR measurement evaluation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace

import numpy as np
from tqdm import tqdm as _tqdm  # type: ignore[import-untyped]

from .normal_points import NptDataset

from .equations import ObservationEquation, ObservationResultDetail
from .measurement import LlrObservationModel
from .resolver import ObservationCatalogSelection, ObservationResolver, ResolvedObservation


@dataclass(frozen=True, slots=True)
class ObservationProcessingOptions:
    station_identifier: str | None = None
    reflector_identifier: str | None = None
    min_elevation_deg: float = 0.0
    include_reflector_position_partials: bool = False
    show_progress: bool = False
    progress_description: str | None = None

    def __post_init__(self) -> None:
        min_elevation = float(self.min_elevation_deg)
        if not np.isfinite(min_elevation):
            raise ValueError("min_elevation_deg must be finite.")
        object.__setattr__(self, "min_elevation_deg", min_elevation)

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
        if not isinstance(resolver, ObservationResolver):
            raise TypeError("resolver must be an ObservationResolver.")
        if not isinstance(observation_model, LlrObservationModel):
            raise TypeError("observation_model must be an LlrObservationModel.")
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
                include_reflector_position_partials=(options.include_reflector_position_partials),
                result_detail=detail,
            )
            if evaluation.below_elevation_limit:
                continue
            row = evaluation.result_row
            if row is None:
                raise RuntimeError("Measurement row was not generated.")
            rows.append(row)
        return rows


__all__ = ["LlrObservationProcessor", "ObservationProcessingOptions"]
