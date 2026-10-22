"""Lunar epoch-data production on MPI worker ranks."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import cast

from lunarops.classes.dynamics.epoch_data import EpochArrays, EpochInputs
from lunarops.classes.time import Epoch


class MpiEpochSource:
    """Epoch-data source that farms batches out to MPI workers, in integration order.

    Workers open their own ephemeris from the same resolved class config and
    run the same :meth:`EpochInputs.arrays`, so the results equal the serial ones bit for bit.
    """

    def __init__(self, runtime, inputs: EpochInputs, ephemeris_config: dict, working_dir, spec_id: str) -> None:
        self._runtime = runtime
        self._spec_id = str(spec_id)
        runtime.prepare_epoch_inputs(
            {
                "specId": self._spec_id,
                "inputs": inputs,
                "ephemerides": ephemeris_config,
                "workingDir": str(working_dir),
            }
        )

    def stream(self, initial_epoch: Epoch, offset_batches: Sequence[Sequence[float]]) -> Iterator[EpochArrays]:
        payloads = [
            {"specId": self._spec_id, "initialEpoch": initial_epoch, "offsets": list(offsets)}
            for offsets in offset_batches
        ]
        yield from cast(Iterator[EpochArrays], self._runtime.stream_tasks("epoch_data", payloads))
