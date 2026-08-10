"""Run context shared across programs in one config run.

Analogue of GROOPS' global config elements: heavyweight objects (ephemeris
backend, frame system, catalogs, IERS table) are declared once under
``globals:`` in the config and lazily constructed on first use; subsequent
programs in the same run reuse the same instance.
"""

from __future__ import annotations

import json
import hashlib
from collections.abc import Mapping, MutableMapping
from copy import deepcopy
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from lunarops.parallel.mpi import MpiRuntime

from .registry import resolve_class_config, validate_global_class_configs
from lunarops.resource_lifecycle import close_resources


_UNSET = object()


def _canonical_value(value: Any) -> Any:
    """Convert YAML-like values into a stable, type-preserving JSON value."""
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(value[key])
            for key in sorted(value, key=lambda item: str(item))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, Path):
        return {"__path__": str(value)}
    if isinstance(value, (datetime, date)):
        return {"__date__": value.isoformat()}
    if isinstance(value, Enum):
        return {"__enum__": f"{type(value).__name__}:{value.value}"}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return {"__repr__": repr(value), "__type__": type(value).__qualname__}


def _config_key(category: str, config: Mapping[str, Any]) -> str:
    payload = json.dumps(
        {"category": category, "config": _canonical_value(config)},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class RunContext:
    """Own run configuration, runtime services, and cached class instances."""

    def __init__(
        self,
        *,
        global_class_configs: Mapping[str, Any] | None = None,
        working_dir: str | Path | None = None,
        runtime: MpiRuntime | None = None,
        mpi_resources: Mapping[str, object] | None = None,
        class_cache: MutableMapping[str, Any] | None = None,
        owns_class_cache: bool | None = None,
    ) -> None:
        if global_class_configs is not None and not isinstance(global_class_configs, Mapping):
            raise TypeError("global_class_configs must be a mapping.")
        if global_class_configs is not None and any(not isinstance(key, str) for key in global_class_configs):
            raise TypeError("global_class_configs keys must be strings.")
        if working_dir is not None and not isinstance(working_dir, (str, Path)):
            raise TypeError("working_dir must be a path string or Path.")
        if class_cache is not None and not isinstance(class_cache, MutableMapping):
            raise TypeError("class_cache must be a mutable mapping.")
        if owns_class_cache is not None and not isinstance(owns_class_cache, bool):
            raise TypeError("owns_class_cache must be a boolean or None.")
        self.global_class_configs: Dict[str, Any] = deepcopy(dict(global_class_configs or {}))
        self.working_dir = Path(working_dir or ".").expanduser().resolve()
        self.runtime = runtime
        self.mpi_resources: Dict[str, object] = dict(mpi_resources or {})
        self._cache: MutableMapping[str, Any] = class_cache if class_cache is not None else {}
        # A supplied cache is shared state by default.  Its owner must decide
        # when the shared resources are closed; a worker context must never
        # clear a cache still used by another observation spec.
        self._owns_class_cache = class_cache is None if owns_class_cache is None else owns_class_cache
        self._transient_resources: list[Any] = []
        self._cache_lock = RLock()
        self._observation_spec_sequence = 0
        self._closed = False

    # -- class instantiation ------------------------------------------------
    def create_class(
        self,
        category: str,
        config=_UNSET,
        *,
        cache: bool = True,
        factory_context=None,
        cache_namespace: str = "",
    ):
        """Instantiate a class; omitted config falls back to ``globals:``.

        With ``cache=True`` (default) identical (category, config) pairs share
        one instance for the lifetime of the run — this is how the CALCEPH
        ephemeris or Earth-orientation source is opened once and reused by every program.
        """
        with self._cache_lock:
            if not isinstance(category, str) or not category.strip():
                raise ValueError("Class categories must be non-empty strings.")
            category = category.strip()
            if self._closed:
                raise RuntimeError("RunContext is closed.")
            if not isinstance(cache, bool):
                raise TypeError("cache must be a boolean.")
            if not isinstance(cache_namespace, str):
                raise TypeError("cache_namespace must be a string.")
            if config is _UNSET:
                if category not in self.global_class_configs:
                    raise KeyError(
                        f"Program requires class {category!r} but neither the program "
                        f"config nor the run 'globals:' section defines it."
                    )
                config = self.global_class_configs[category]
            resolved_config, registered = resolve_class_config(category, config)
            target_context = self if factory_context is None else factory_context
            if not cache:
                instance = registered.factory(resolved_config, target_context)
                self._transient_resources.append(instance)
                return instance
            key = f"{cache_namespace}:{_config_key(category, resolved_config)}"
            if key not in self._cache:
                self._cache[key] = registered.factory(resolved_config, target_context)
            return self._cache[key]

    def class_config(self, category: str, program_config: dict, key: Optional[str] = None):
        """Return the class config for *category*: program entry overrides globals."""
        if not isinstance(program_config, Mapping):
            raise TypeError("program_config must be a mapping.")
        if not isinstance(category, str) or not category.strip():
            raise ValueError("Class categories must be non-empty strings.")
        category = category.strip()
        if key is None:
            key = category
        if not isinstance(key, str) or not key.strip():
            raise ValueError("Class configuration keys must be non-empty strings.")
        if key in program_config:
            return deepcopy(program_config[key])
        if category not in self.global_class_configs:
            return None
        return deepcopy(self.global_class_configs[category])

    def validate_globals(self) -> dict[str, Any]:
        """Validate global class declarations without constructing their instances."""
        with self._cache_lock:
            if self._closed:
                raise RuntimeError("RunContext is closed.")
            self.global_class_configs = validate_global_class_configs(self.global_class_configs)
            return deepcopy(self.global_class_configs)

    # -- paths ---------------------------------------------------------------
    def resolve_path(self, value) -> Path:
        if not isinstance(value, (str, Path)):
            raise TypeError(f"Path value must be a string or Path, got {value!r}.")
        if not str(value).strip():
            raise ValueError("Path values must not be empty.")
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = self.working_dir / path
        return path.resolve()

    def next_observation_spec_id(self) -> str:
        with self._cache_lock:
            if self._closed:
                raise RuntimeError("RunContext is closed.")
            self._observation_spec_sequence += 1
            return f"{id(self)}:{self._observation_spec_sequence}"

    def close(self) -> None:
        with self._cache_lock:
            if self._closed:
                return
            self._closed = True
            resources = tuple(self._transient_resources)
            self._transient_resources.clear()
            if self._owns_class_cache:
                resources += tuple(self._cache.values())
                self._cache.clear()
        close_resources(resources, owner="run-context")

    def __enter__(self) -> "RunContext":
        with self._cache_lock:
            if self._closed:
                raise RuntimeError("RunContext is closed.")
            return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
