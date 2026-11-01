from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import cast

from lunarops.classes.dynamics.epoch_data import EpochBatch, EpochInputs
from lunarops.classes.time import Epoch


class MpiEpochSource:

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

    def stream(self, initial_epoch: Epoch, offset_batches: Sequence[Sequence[float]]) -> Iterator[EpochBatch]:
        payloads = [
            {"specId": self._spec_id, "initialEpoch": initial_epoch, "offsets": list(offsets)}
            for offsets in offset_batches
        ]
        yield from cast(Iterator[EpochBatch], self._runtime.stream_tasks("epoch_data", payloads))
