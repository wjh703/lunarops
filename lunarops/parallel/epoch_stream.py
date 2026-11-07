"""Computation of lunar-dynamics epoch data on MPI worker ranks."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import cast

from lunarops.classes.dynamics.epoch_data import EpochBatch, EpochInputs
from lunarops.classes.time import Epoch


class MpiEpochSource:
    """Source of epoch data (ephemeris-derived inputs of the dynamics) computed by MPI workers.

    Creating the source broadcasts the epoch inputs and the ephemeris config to every worker, which opens
    the ephemeris once.

    Parameters
    ----------
    runtime : MpiRuntime
        MPI runtime of the master rank.
    inputs : EpochInputs
        Description of the per-epoch data to compute.
    ephemeris_config : dict
        Resolved ``ephemerides`` class config.
    working_dir : str or Path
        Base directory for relative paths on the workers.
    spec_id : str
        Identifier of the broadcast spec, unique within the run.
    """

    def __init__(self, runtime, inputs: EpochInputs, ephemeris_config: dict, working_dir, spec_id: str) -> None:
        self._runtime = runtime
        self._spec_id = runtime.broadcast_spec(
            "epochInputs",
            {
                "specId": spec_id,
                "inputs": inputs,
                "ephemerides": ephemeris_config,
                "workingDir": str(working_dir),
            },
        )

    def stream(self, initial_epoch: Epoch, offset_batches: Sequence[Sequence[float]]) -> Iterator[EpochBatch]:
        """Yield the epoch data of each batch of offsets, in batch order.

        Parameters
        ----------
        initial_epoch : Epoch
            Reference epoch of the offsets (TDB).
        offset_batches : sequence of sequence of float
            Offsets from ``initial_epoch``, in seconds; one worker task per batch.

        Yields
        ------
        EpochBatch
            Epoch data of one batch.
        """
        payloads = [
            {"specId": self._spec_id, "initialEpoch": initial_epoch, "offsets": list(offsets)}
            for offsets in offset_batches
        ]
        yield from cast(Iterator[EpochBatch], self._runtime.stream_tasks("epoch_data", payloads))
