"""Array-oriented epoch and force-model data records."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from lunarops.classes.time import Epoch

from .deformation import LunarInertiaEvaluation


class StateHistoryProvider(Protocol):
    def states(self, body_names: Sequence[str], epoch: Epoch) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class StepData:
    """Candidate-independent data shared by every right-hand-side evaluation of one ABM step."""

    force_payloads: Mapping[str, object]
    lunar_inertia: LunarInertiaEvaluation | None = None
