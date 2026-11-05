"""Observation screening that defines the fixed observation domain of an LLR adjustment.

Screening keeps light-time-converged observations, rejects gross pre-fit residuals, assigns variance
components and rejects implausibly small reported a-priori sigmas.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from lunarops.classes.observation.equations import EquationBatch
from lunarops.classes.parametrization.base import ParametrizationList
from lunarops.estimation.adjustment_plan import (
    AccuracyScreeningSettings,
    ResidualScreening,
    VarianceComponentSettings,
)
from lunarops.estimation.adjustment_result_models import LlrAdjustmentObservationDomain, ObsKey
from lunarops.estimation.variance_component_groups import assign_variance_components


def prefit_gross_threshold(
    station: str,
    default: float | None,
    by_station: Mapping[str, float | None] | None,
) -> float | None:
    """Return the gross-error threshold of a station.

    Parameters
    ----------
    station : str
        Station key.
    default : float or None
        Default threshold, in metres.
    by_station : Mapping[str, float or None] or None
        Per-station overrides, in metres; an override of ``None`` disables the station's screening.

    Returns
    -------
    float or None
        Threshold in metres, or ``None`` when screening is disabled.
    """
    overrides = by_station or {}
    if station in overrides:
        value = overrides[station]
        return None if value is None else float(value)
    return None if default is None else float(default)


def prefit_gross_rejections(
    equations: EquationBatch,
    parametrization: ParametrizationList,
    *,
    threshold_m: float | None,
    threshold_by_station_m: Mapping[str, float | None] | None,
) -> dict[ObsKey, float]:
    """Return the observations whose absolute reduced pre-fit residual exceeds their station threshold.

    Parameters
    ----------
    equations : EquationBatch
        Observation equations.
    parametrization : ParametrizationList
        Parametrization, already set up, whose ``reduce`` gives the reduced one-way O-C residuals.
    threshold_m : float or None
        Default threshold, in metres.
    threshold_by_station_m : Mapping[str, float or None] or None
        Per-station overrides, in metres.

    Returns
    -------
    dict[ObsKey, float]
        Reduced one-way residual in metres per rejected observation identity, in batch order.
    """
    rejected: dict[ObsKey, float] = {}
    residuals = parametrization.reduce(equations)
    thresholds = {
        station: prefit_gross_threshold(station, threshold_m, threshold_by_station_m)
        for station in equations.stations.tolist()
    }
    for station, threshold in thresholds.items():
        if threshold is not None:
            mask = (equations.station == station) & (np.abs(residuals) > threshold)
            rejected.update(zip(equations.observation_id[mask].tolist(), residuals[mask].tolist()))
    return {identity: rejected[identity] for identity in equations.observation_id if identity in rejected}


def reject_implausible_apriori_accuracies(
    equations: EquationBatch,
    assignments: Mapping[ObsKey, str],
    *,
    minimumOneWayM: float,
    minimum_group_median_fraction: float,
) -> tuple[EquationBatch, dict[ObsKey, dict[str, object]], dict[str, dict[str, object]]]:
    """Reject observations whose reported a-priori sigma is below the validity limit of their component.

    The limit of a variance component is ``max(minimumOneWayM, minimum_group_median_fraction * median)``, with
    the median of the one-way a-priori sigmas of the component's observations.

    Parameters
    ----------
    equations : EquationBatch
        Observation equations; ``sigma`` is the one-way a-priori sigma in metres.
    assignments : Mapping[ObsKey, str]
        Variance-component identifier per observation identity.
    minimumOneWayM : float
        Absolute lower limit, in metres.
    minimum_group_median_fraction : float
        Lower limit as a fraction of the component median.

    Returns
    -------
    retained : EquationBatch
        Observations at or above their limit.
    records : dict[ObsKey, dict[str, object]]
        Per observation: component, reported sigma and limit (metres), ``status`` (``RETAINED`` or
        ``REJECTED``) and ``reason``.
    groups : dict[str, dict[str, object]]
        Per component: median reported sigma, limit (metres) and observation, rejected and retained counts.
    """
    component_ids = np.asarray([assignments[key] for key in equations.observation_id])
    group_diagnostics: dict[str, dict[str, object]] = {}
    limits = np.empty(len(equations))
    for component_id in dict.fromkeys(component_ids.tolist()):
        mask = component_ids == component_id
        median = float(np.median(equations.sigma[mask]))
        limit = max(float(minimumOneWayM), float(minimum_group_median_fraction) * median)
        limits[mask] = limit
        rejected_count = int(np.count_nonzero(equations.sigma[mask] < limit))
        count = int(np.count_nonzero(mask))
        group_diagnostics[component_id] = {
            "medianReportedSigmaM": median, "minimumValidSigmaM": limit,
            "observationCount": count, "rejectedCount": rejected_count, "retainedCount": count - rejected_count,
        }
    rejected = equations.sigma < limits
    records = {
        identity: {
            "componentId": component_id, "reportedSigmaM": reported, "minimumValidSigmaM": limit,
            "status": "REJECTED" if invalid else "RETAINED",
            "reason": "APRIORI_SIGMA_BELOW_VALIDITY_LIMIT" if invalid else None,
        }
        for identity, component_id, reported, limit, invalid in zip(
            equations.observation_id.tolist(), component_ids.tolist(), equations.sigma.tolist(),
            limits.tolist(), rejected.tolist(), strict=True,
        )
    }
    return equations[~rejected], records, group_diagnostics


def screen_observations(
    equations: EquationBatch,
    parametrization: ParametrizationList,
    *,
    residual: ResidualScreening,
    reported_sigma: AccuracyScreeningSettings,
    variance_components: VarianceComponentSettings,
) -> LlrAdjustmentObservationDomain:
    """Create the fixed observation domain used by all estimate steps.

    The steps are: keep light-time-converged observations, set up the parametrization on them, reject gross
    pre-fit residuals, assign variance components, and reject implausibly small a-priori sigmas.

    Parameters
    ----------
    equations : EquationBatch
        Observation equations at the a-priori parameter values.
    parametrization : ParametrizationList
        Parametrization; it is set up on the converged observations.
    residual : ResidualScreening
        Gross-error thresholds.
    reported_sigma : AccuracyScreeningSettings
        A-priori sigma limits.
    variance_components : VarianceComponentSettings
        Variance-component groups.

    Returns
    -------
    LlrAdjustmentObservationDomain
        Retained identities, rejections, component assignments, screening records and the immutable
        metadata (station, reflector, transmit epoch, wavelength) of each retained observation.

    Raises
    ------
    ValueError
        If no observation converged, every observation is rejected, or an observation matches no or several
        variance components.
    """
    converged = equations[equations.converged]
    if not converged:
        raise ValueError("screenObservations has no light-time-converged observations.")

    parametrization.setup(converged)
    gross_rejected = prefit_gross_rejections(
        converged,
        parametrization,
        threshold_m=residual.maximumAbsoluteM,
        threshold_by_station_m=residual.maximumAbsoluteByStationM,
    )
    gross_retained = converged[~np.isin(converged.observation_id, np.fromiter(gross_rejected, dtype=object))]
    assignments = assign_variance_components(gross_retained, variance_components.components)
    retained, sigma_records, sigma_groups = reject_implausible_apriori_accuracies(
        gross_retained,
        assignments,
        minimumOneWayM=reported_sigma.minimumOneWayM,
        minimum_group_median_fraction=reported_sigma.minimumFractionOfGroupMedian,
    )
    if not retained:
        raise ValueError("screenObservations rejected every observation.")
    retained_keys = set(retained.observation_id.tolist())
    return LlrAdjustmentObservationDomain(
        gross_rejected=gross_rejected,
        retained_keys=retained_keys,
        assignments={key: value for key, value in assignments.items() if key in retained_keys},
        accuracy_records=sigma_records,
        accuracy_groups=sigma_groups,
        observation_signatures={
            identity: (station, reflector, retained.epoch(index), wavelength)
            for index, (identity, station, reflector, wavelength) in enumerate(zip(
                retained.observation_id.tolist(), retained.station.tolist(), retained.reflector.tolist(),
                retained.wavelength_nm.tolist(), strict=True,
            ))
        },
    )


__all__ = [
    "prefit_gross_rejections",
    "prefit_gross_threshold",
    "reject_implausible_apriori_accuracies",
    "screen_observations",
]
