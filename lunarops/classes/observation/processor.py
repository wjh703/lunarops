"""Dataset orchestration for LLR measurement evaluation."""

from __future__ import annotations

from collections.abc import Iterable
from typing import cast

from tqdm import tqdm as _tqdm  # type: ignore[import-untyped]

from lunarops.classes.time import format_time_with_utc_offset

from .equations import EquationBatch, ObservationOptions, ObservationResultDetail
from .measurement import LlrObservationModel
from .normal_points import NptDataset
from .resolver import ObservationResolver, ResolvedObservation


class LlrObservationProcessor:
    """Evaluate normal-point datasets with the LLR observation model.

    Observations below the elevation limit (see ``ObservationOptions``) are left out of both equations and
    result rows.

    Parameters
    ----------
    resolver : ObservationResolver
        Catalog resolver; its ``model_state`` is exposed as ``self.model_state``.
    observation_model : LlrObservationModel
        Single-observation model.
    """

    def __init__(self, resolver: ObservationResolver, observation_model: LlrObservationModel) -> None:
        self.resolver = resolver
        self.model_state = resolver.model_state
        self.observation_model = observation_model

    def _with_progress(
        self,
        observations: Iterable[ResolvedObservation],
        *,
        total: int,
        options: ObservationOptions,
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
        options: ObservationOptions,
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
        options: ObservationOptions,
    ) -> EquationBatch:
        """Return the linearized equations of a dataset.

        Parameters
        ----------
        dataset : NptDataset
            Normal points to evaluate.
        options : ObservationOptions
            Catalog overrides, elevation limits, partials to compute and progress display.

        Returns
        -------
        EquationBatch
            One row per observation above the elevation limit, including rows whose light time did not converge.

        Raises
        ------
        ValueError
            If any record cannot be resolved in the catalogs.
        """
        evaluations = [
            self.observation_model.evaluate(
                observation,
                options,
            )
            for observation in self._resolved_observations(dataset, options)
        ]
        return EquationBatch.concatenate([evaluation.equation for evaluation in evaluations if not evaluation.below_elevation_limit])

    def rows(
        self,
        dataset: NptDataset,
        *,
        options: ObservationOptions,
        detail: ObservationResultDetail = ObservationResultDetail.STANDARD,
    ) -> list[dict[str, object]]:
        """Return the per-observation result rows of a dataset.

        Parameters
        ----------
        dataset : NptDataset
            Normal points to evaluate.
        options : ObservationOptions
            Catalog overrides, elevation limits, partials to compute, local-time offset and progress display.
        detail : ObservationResultDetail, default ``STANDARD``
            Detail level of the rows.

        Returns
        -------
        list[dict[str, object]]
            One row per observation above the elevation limit, starting with ``obs_time_utc`` and ending with
            ``obs_time_local`` (transmit epoch, nanosecond precision).

        Raises
        ------
        ValueError
            If any record cannot be resolved in the catalogs.
        """
        rows: list[dict[str, object]] = []
        for observation in self._resolved_observations(dataset, options):
            evaluation = self.observation_model.evaluate(
                observation,
                options,
                result_detail=detail,
            )
            if evaluation.below_elevation_limit:
                continue
            epoch_utc = evaluation.equation.epoch(0)
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


__all__ = ["LlrObservationProcessor", "ObservationOptions"]
