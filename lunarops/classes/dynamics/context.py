"""Array-oriented epoch and force-model data records."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from lunarops.classes.time import Epoch

from .deformation import LunarInertiaEvaluation

StateHistoryProvider = Callable[[Sequence[str], Epoch], np.ndarray]


@dataclass(frozen=True, slots=True)
class StepData:
    """Candidate-independent data shared by every right-hand-side evaluation of one ABM step."""

    force_payloads: Mapping[str, object]
    lunar_inertia: LunarInertiaEvaluation | None = None
