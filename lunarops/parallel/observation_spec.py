"""Observation specs: the self-contained description of an observation model sent to MPI workers.

A spec holds the resolved observation-model configs, the working directory, the station and reflector
catalogs and preloaded shared data (the Earth-orientation table of a ``file`` Earth-rotation source), so that
each worker can build the same observation processor or predictor as the master.
"""

from __future__ import annotations

from lunarops.config.context import ClassCache


def _prepare_shared_resources(merged: dict, context) -> dict:
    """Load a ``file`` Earth-orientation table once on the master so workers receive it with the spec."""
    resources: dict = {}
    earth_rotation_config = merged.get("earthRotation")
    if earth_rotation_config is not None and earth_rotation_config["type"] == "file":
        earth_orientation = context.create_class(
            "earthRotation",
            earth_rotation_config,
            cache=True,
        )
        resources["earthRotation"] = earth_orientation
    return resources


def make_observation_spec(
    config: dict,
    context,
    *,
    station_catalog=None,
    reflector_catalog=None,
) -> dict:
    """Build the observation spec of a program config.

    Parameters
    ----------
    config : dict
        Resolved program config.
    context : RunContext
        Run context of the master.
    station_catalog : dict, optional
        Station catalog to send instead of the one read from the config.
    reflector_catalog : dict, optional
        Reflector catalog to send instead of the one read from the config.

    Returns
    -------
    dict
        Spec with ``specId``, ``programConfig``, ``workingDir``, ``stationCatalog``, ``reflectorCatalog`` and
        ``sharedResources``.
    """
    from lunarops.classes.observation.assembly import resolve_observation_assembly

    assembly = resolve_observation_assembly(
        context,
        config,
        station_catalog=station_catalog,
        reflector_catalog=reflector_catalog,
    )
    return {
        "specId": context.next_observation_spec_id(),
        "programConfig": assembly.model_configs,
        "workingDir": str(context.working_dir),
        "stationCatalog": assembly.station_catalog,
        "reflectorCatalog": assembly.reflector_catalog,
        "sharedResources": _prepare_shared_resources(
            assembly.model_configs,
            context,
        ),
    }


def _worker_context(spec: dict, shared_class_cache: ClassCache | None = None):
    from lunarops.config.context import RunContext

    return RunContext(
        working_dir=spec.get("workingDir", "."),
        mpi_resources=spec.get("sharedResources"),
        class_cache=shared_class_cache,
    )


def build_worker_processor(spec: dict, shared_class_cache: ClassCache | None = None):
    """Build the observation processor of a spec on a worker.

    Parameters
    ----------
    spec : dict
        Observation spec from ``make_observation_spec``.
    shared_class_cache : ClassCache, optional
        Worker-wide cache of model instances shared between specs.

    Returns
    -------
    tuple of (RunContext, LlrObservationProcessor)
        Worker run context and processor.
    """
    from lunarops.classes.observation.assembly import build_observation_processor

    context = _worker_context(spec, shared_class_cache)
    processor = build_observation_processor(
        context,
        spec["programConfig"],
        station_catalog=spec["stationCatalog"],
        reflector_catalog=spec["reflectorCatalog"],
    )
    return context, processor


def build_worker_observation_runtime(
    spec: dict,
    shared_class_cache: ClassCache | None = None,
    *,
    context=None,
):
    """Build the observation runtime (frames, light-time solver, catalogs) of a spec on a worker.

    Parameters
    ----------
    spec : dict
        Observation spec from ``make_observation_spec``.
    shared_class_cache : ClassCache, optional
        Worker-wide cache of model instances shared between specs.
    context : RunContext, optional
        Existing worker context; a new one is created from the spec when None.

    Returns
    -------
    tuple of (RunContext, ObservationRuntime)
        Worker run context and observation runtime.
    """
    from lunarops.classes.observation.assembly import build_observation_runtime

    worker_context = context if context is not None else _worker_context(spec, shared_class_cache)
    runtime = build_observation_runtime(
        worker_context,
        spec["programConfig"],
        station_catalog=spec["stationCatalog"],
        reflector_catalog=spec["reflectorCatalog"],
    )
    return worker_context, runtime


def snapshot_catalog_state(model_state) -> dict:
    """Return the current reflector positions and station states of an observation model state.

    Parameters
    ----------
    model_state : object
        Observation model state.

    Returns
    -------
    dict
        ``reflectorPositions`` (Moon PA frame, m) and ``stationStates`` (ITRF, m).
    """
    return {
        "reflectorPositions": model_state.reflector_positions_pa_m(),
        "stationStates": model_state.station_states_itrf_m(),
    }


def apply_catalog_state(processor, catalog_state: dict | None) -> None:
    """Apply a snapshot from ``snapshot_catalog_state`` to the model state of a processor.

    Parameters
    ----------
    processor : LlrObservationProcessor
        Processor whose model state is updated.
    catalog_state : dict or None
        Snapshot with ``reflectorPositions`` and ``stationStates``; nothing is applied when empty or None.
    """
    if not catalog_state:
        return
    positions = catalog_state.get("reflectorPositions") or {}
    if positions:
        processor.model_state.apply_reflector_positions_pa_m(positions)
    station_states = catalog_state.get("stationStates") or {}
    if station_states:
        processor.model_state.apply_station_states_itrf_m(station_states)


__all__ = [
    "apply_catalog_state",
    "build_worker_observation_runtime",
    "build_worker_processor",
    "make_observation_spec",
    "snapshot_catalog_state",
]
