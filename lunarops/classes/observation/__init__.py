"""LLR observation modelling, processing and linearized observation equations.

The package provides normal-point records (``NptRecord``, ``NptDataset``), station and reflector catalogs, the
two-way light-time solver (``LightTimeSolver``), the single-observation model that turns a normal point into a
one-way O-C residual in metres with its partial derivatives (``LlrObservationModel``), the dataset processor that
collects these into a columnar ``EquationBatch``, and time-grid pointing and visibility prediction
(``LlrObservationPredictor``).
"""

from .catalogs import ReflectorRecord, StationRecord, resolve_catalog_key
from .equations import EquationBatch, ObservationResultDetail
from .light_time import (
    LightTimeLeg,
    LightTimeRequest,
    LightTimeSolution,
    LightTimeSolver,
    UplinkSolution,
)
from .measurement import LlrObservationModel
from .normal_points import NptDataset, NptRecord, combine_npt_datasets, parse_time_filter
from .prediction import (
    LlrObservationPredictor,
    PredictionCriteria,
    PredictionMeteorology,
    build_visibility_windows,
)
from .processor import LlrObservationProcessor, ObservationOptions
from .resolver import (
    ObservationCatalogSelection,
    ObservationCatalogState,
    ObservationResolver,
    ResolvedObservation,
)

__all__ = [
    "EquationBatch",
    "LightTimeLeg",
    "LightTimeRequest",
    "LightTimeSolution",
    "LightTimeSolver",
    "LlrObservationModel",
    "LlrObservationPredictor",
    "LlrObservationProcessor",
    "NptDataset",
    "NptRecord",
    "ObservationCatalogSelection",
    "ObservationCatalogState",
    "ObservationOptions",
    "ObservationResolver",
    "ObservationResultDetail",
    "PredictionCriteria",
    "PredictionMeteorology",
    "ReflectorRecord",
    "ResolvedObservation",
    "StationRecord",
    "UplinkSolution",
    "build_visibility_windows",
    "combine_npt_datasets",
    "parse_time_filter",
    "resolve_catalog_key",
]
