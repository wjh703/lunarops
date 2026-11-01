"""Run defaults are declared under ``shared:``.

Identical cached model declarations share one instance for the run."""

from __future__ import annotations

import json
from collections.abc import Mapping
from contextlib import ExitStack
from itertools import count
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

if TYPE_CHECKING:
    from lunarops.parallel.mpi import MpiRuntime


from .class_configs import validate_global_class_configs
from .registry import create


class ClassCache(dict[str, Any]):
    def __init__(self):
        super().__init__()
        self.resources = ExitStack()

    def close(self):
        self.resources.close()
        self.clear()


_UNSET = object()


class RunContext:
    def __init__(
        self,
        *,
        shared_class_configs: Mapping[str, Any] | None = None,
        observation_model_configs: Mapping[str, Any] | None = None,
        working_dir: str | Path | None = None,
        runtime: MpiRuntime | None = None,
        mpi_resources: Mapping[str, object] | None = None,
        class_cache: ClassCache | None = None,
    ) -> None:
        self.shared_class_configs: dict[str, Any] = dict(shared_class_configs or {})
        self.observation_model_configs: dict[str, Any] = dict(observation_model_configs or {})
        self.working_dir = Path(working_dir or ".").expanduser().resolve()
        self.runtime = runtime
        self.mpi_resources: dict[str, object] = dict(mpi_resources or {})
        self._cache = class_cache if class_cache is not None else ClassCache()
        self._resources = ExitStack()
        if class_cache is None:
            self._resources.callback(self._cache.close)
        self._observation_spec_sequence = count(1)

    @property
    def class_configs(self) -> dict[str, Any]:
        return {**self.shared_class_configs, **self.observation_model_configs}

    def create_class(
        self,
        category: str,
        config=_UNSET,
        *,
        cache: bool = True,
    ):
        """Instantiate a resolved declaration, optionally reusing the run cache.

        With ``cache=True`` (default) identical (category, config) pairs share
        one instance for the lifetime of the run; this is how the CALCEPH
        ephemeris or Earth-orientation source is opened once and reused by every program.
        """
        if config is _UNSET:
            config = self.class_configs[category]
        key = category + ":" + json.dumps(config, sort_keys=True, default=str)
        if cache and key in self._cache:
            return self._cache[key]
        instance = create(category, config, self)
        stack = self._cache.resources if cache else self._resources
        close = getattr(instance, "close", None)
        if close is not None:
            stack.callback(close)
        if cache:
            self._cache[key] = instance
        return instance

    def class_config(self, category: str, program_config: dict, key: str | None = None):
        key = category if key is None else key
        if key in program_config:
            return program_config[key]
        if key in self.observation_model_configs:
            return self.observation_model_configs[key]
        return self.shared_class_configs.get(category)

    def resolve_defaults(self) -> None:
        from lunarops.classes.observation.assembly import observation_model_schema

        self.shared_class_configs = validate_global_class_configs(self.shared_class_configs, path="shared")
        self.observation_model_configs = observation_model_schema().resolve(
            self.observation_model_configs,
            path="observationModel",
        )

    def resolve_path(self, value) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = self.working_dir / path
        return path.resolve()

    def next_observation_spec_id(self) -> str:
        return f"{id(self)}:{next(self._observation_spec_sequence)}"

    def close(self) -> None:
        self._resources.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
