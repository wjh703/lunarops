from __future__ import annotations


def _prepare_shared_resources(merged: dict, context) -> dict:
    resources: dict = {}
    earth_rotation_config = merged.get("earthRotation")
    if earth_rotation_config is not None and earth_rotation_config["type"] == "file":
        earth_orientation = context.create_class(
            "earthRotation",
            earth_rotation_config,
            cache=True,
        )
        resources["earthRotation"] = earth_orientation.to_mpi_payload()
    return resources


def make_observation_spec(
    config: dict,
    context,
    *,
    station_catalog=None,
    reflector_catalog=None,
) -> dict:
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


def _worker_context(spec: dict, shared_class_cache: dict | None = None):
    from lunarops.config.context import RunContext

    return RunContext(
        working_dir=spec.get("workingDir", "."),
        mpi_resources=spec.get("sharedResources"),
        class_cache=shared_class_cache,
        owns_class_cache=shared_class_cache is None,
    )


def build_worker_processor(spec: dict, shared_class_cache: dict | None = None):
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
    shared_class_cache: dict | None = None,
    *,
    context=None,
):
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
    return {
        "reflectorPositions": model_state.reflector_positions_pa_m(),
        "stationStates": model_state.station_states_itrf_m(),
    }


def apply_catalog_state(processor, catalog_state: dict | None) -> None:
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
