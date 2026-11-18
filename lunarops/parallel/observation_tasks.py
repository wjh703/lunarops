"""Observation and prediction work distributed over the workers of a task runtime.

The master broadcasts an observation spec (:mod:`lunarops.parallel.observation_spec`) and streams chunks of
normal points or of prediction epochs; ``distributed_observations`` and ``distributed_prediction_rows`` put
the results back in input order. Both work with either runtime (``TaskRuntime``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict

from lunarops.parallel.runtime import TaskRuntime


def distributed_observations(
    runtime: TaskRuntime,
    spec: dict,
    datasets,
    options,
    *,
    chunksize: int = 8,
    catalog_state: dict | None = None,
    rows: bool,
    output_level: str = "standard",
    progress_desc: str = "residuals",
    quiet: bool = False,
) -> dict[str, list]:
    """Evaluate normal points on the workers of a task runtime.

    The datasets are sent once with the spec (if the spec has none yet); each task then names a chunk of at
    most ``chunksize`` consecutive normal points. ``catalog_state`` sends the current reflector and station
    states with every task.

    Parameters
    ----------
    runtime : TaskRuntime
        Task runtime of the master.
    spec : dict
        Observation spec from ``make_observation_spec``.
    datasets : dict of str to NptDataset
        Datasets keyed by source name.
    options : ObservationOptions
        Observation options.
    chunksize : int, optional
        Normal points per task. Default 8.
    catalog_state : dict, optional
        Snapshot from ``snapshot_catalog_state`` applied by the workers before evaluating.
    rows : bool
        Return result-table rows (True) or observation equations (False).
    output_level : str, optional
        Column set of the rows: ``standard`` or ``full``.
    progress_desc : str, optional
        Progress label.
    quiet : bool, optional
        Report no progress.

    Returns
    -------
    dict of str to list or EquationBatch
        Per source: rows sorted by normal-point index, or one ``EquationBatch`` sorted by observation ID.
    """
    spec.setdefault("datasets", datasets)
    spec_id = runtime.broadcast_spec("observationSpec", spec)
    common = {
        "specId": spec_id,
        "options": asdict(options),
        "catalogState": catalog_state,
        "returnRows": rows,
        "outputLevel": output_level,
    }
    chunk = int(chunksize)
    tasks: list[dict] = []
    for source_name, dataset in datasets.items():
        records = dataset.records
        for start in range(0, len(records), chunk):
            tasks.append(
                common
                | {
                    "sourceName": str(source_name),
                    "startIndex": start,
                    "recordCount": min(chunk, len(records) - start),
                }
            )
    results = runtime.map_tasks(
        "observation_equations",
        tasks,
        progress=None if quiet else progress_desc,
        unit="normal points",
        sizes=[task["recordCount"] for task in tasks],
    )
    items_by_source: dict[str, list] = {str(name): [] for name in datasets}
    for result in results:
        source_name = result["sourceName"]
        items = result["items"]
        if rows:
            items_by_source[source_name].extend(items)
        else:
            items_by_source[source_name].append(items)
    for items in items_by_source.values():
        if rows:
            items.sort(key=lambda row: int(row["normal_point_index"]))
    if not rows:
        import numpy as np

        from lunarops.classes.observation.equations import EquationBatch

        return {
            source: batch[np.argsort(batch.observation_id, kind="stable")]
            for source, chunks in items_by_source.items()
            for batch in (EquationBatch.concatenate(chunks),)
        }
    return items_by_source


def distributed_prediction_rows(
    runtime: TaskRuntime,
    spec: dict,
    epochs: Sequence,
    *,
    step_seconds: float,
    station: str,
    reflector: str,
    criteria: dict,
    meteorology: dict,
    utc_offset_hours: float,
    chunksize: int = 8,
    progress_desc: str = "prediction",
    quiet: bool = False,
) -> list[dict[str, object]]:
    """Evaluate prediction rows on the workers of a task runtime.

    The epochs must form a grid ``epochs[0] + k * step_seconds``: each task sends only its start index and
    count, and the workers rebuild the epochs from the first epoch and the step.

    Parameters
    ----------
    runtime : TaskRuntime
        Task runtime of the master.
    spec : dict
        Observation spec from ``make_observation_spec``; the prediction settings are added to it.
    epochs : sequence of Epoch
        UTC grid epochs.
    step_seconds : float
        Grid spacing, in seconds.
    station : str
        Station catalog key.
    reflector : str
        Reflector catalog key.
    criteria : dict
        Fields of ``PredictionCriteria``.
    meteorology : dict
        Fields of ``PredictionMeteorology``.
    utc_offset_hours : float
        Civil-time offset from UTC of the local-time columns, in hours.
    chunksize : int, optional
        Epochs per task. Default 8.
    progress_desc : str, optional
        Progress label.
    quiet : bool, optional
        Report no progress.

    Returns
    -------
    list of dict
        Prediction rows in epoch order.
    """
    spec["prediction"] = {
        "station": station,
        "reflector": reflector,
        "criteria": criteria,
        "meteorology": meteorology,
        "utcOffsetHours": utc_offset_hours,
    }
    spec_id = runtime.broadcast_spec("prediction", spec)
    chunk = int(chunksize)
    epoch_values = list(epochs)
    payloads = []
    for start in range(0, len(epoch_values), chunk):
        selected = epoch_values[start : start + chunk]
        payloads.append(
            {
                "specId": spec_id,
                "startEpoch": epoch_values[0],
                "startIndex": start,
                "recordCount": len(selected),
                "stepSeconds": step_seconds,
            }
        )
    results = runtime.map_tasks(
        "prediction",
        payloads,
        progress=None if quiet else progress_desc,
        unit="epochs",
        sizes=[payload["recordCount"] for payload in payloads],
    )
    rows: list[tuple[int, dict[str, object]]] = []
    for result in results:
        for item in result["items"]:
            rows.append((int(item["index"]), item["row"]))
    rows.sort(key=lambda item: item[0])
    return [row for _, row in rows]


__all__ = ["distributed_observations", "distributed_prediction_rows"]
