"""Picklable observation specifications shared by serial and MPI execution."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# observation spec: everything a worker needs to build its own observation processor
# ---------------------------------------------------------------------------


def _prepare_shared_resources(merged: dict, context) -> dict:
    """Build rank-0-only immutable resources for one observation spec.

    Text tables are parsed once on rank 0 and serialized as compact arrays.
    Process-local native handles (notably CALCEPH) are intentionally excluded;
    they cannot be safely pickled or shared between MPI processes.
    """
    resources: dict = {}
    earth_rotation_config = merged.get("earthRotation")
    if earth_rotation_config is not None:
        from lunarops.classes.observation_factory import ensure_registered

        if earth_rotation_config["type"] == "file":
            ensure_registered()
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
    """Resolve one picklable observation specification on rank 0.

    The complete specification is broadcast once to every worker and then
    referenced by ``specId`` in individual tasks.  This avoids repeatedly
    pickling catalogs and EOP arrays for every small chunk.
    """
    from lunarops.classes.observation_factory import resolve_observation_assembly

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
    from lunarops.classes.observation_factory import build_observation_processor

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
    """Build prediction services without requiring an observation range-bias model."""
    from lunarops.classes.observation_factory import build_observation_runtime

    worker_context = context if context is not None else _worker_context(spec, shared_class_cache)
    runtime = build_observation_runtime(
        worker_context,
        spec["programConfig"],
        station_catalog=spec["stationCatalog"],
        reflector_catalog=spec["reflectorCatalog"],
    )
    return worker_context, runtime


def snapshot_catalog_state(model_state) -> dict:
    """Pickle-light snapshot of the mutable per-iteration model state."""
    return {
        "reflectorPositions": model_state.reflector_positions_pa_m(),
    }


def apply_catalog_state(processor, catalog_state: dict | None) -> None:
    if not catalog_state:
        return
    positions = catalog_state.get("reflectorPositions") or {}
    if positions:
        processor.model_state.apply_reflector_positions_pa_m(positions)


__all__ = [
    "apply_catalog_state",
    "build_worker_observation_runtime",
    "build_worker_processor",
    "make_observation_spec",
    "snapshot_catalog_state",
]
