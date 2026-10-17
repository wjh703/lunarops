"""Run context shared across programs in one config run.

Analogue of GROOPS' global config elements: heavyweight objects (ephemeris
backend, frame system, catalogs, IERS table) are declared once under
``shared:`` in the config and lazily constructed on first use; subsequent
programs in the same run reuse the same instance.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from collections.abc import Iterable, Mapping, MutableMapping
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

if TYPE_CHECKING:
    from lunarops.parallel.mpi import MpiRuntime


from .registry import instantiate, validate_global_class_configs

logger = logging.getLogger(__name__)


def close_resource(resource: Any, *, owner: str) -> dict[str, str] | None:
    """Close one optional resource and report cleanup failures."""
    close = getattr(resource, "close", None)
    if not callable(close):
        return None
    try:
        close()
    except Exception as exc:
        logger.warning(
            "Failed to close %s resource %s",
            owner,
            type(resource).__name__,
            exc_info=True,
        )

        return {
            "owner": owner,
            "resourceType": type(resource).__name__,
            "errorType": type(exc).__name__,
            "message": str(exc),
        }
    return None


def close_resources(resources: Iterable[Any], *, owner: str, strict: bool = False) -> list[dict[str, str]]:
    """Close each unique resource, logging but isolating failures."""
    failures = []
    seen: set[int] = set()
    for resource in resources:
        identity = id(resource)
        if identity in seen:
            continue
        seen.add(identity)
        failure = close_resource(resource, owner=owner)
        if failure is not None:
            failures.append(failure)
    if strict and failures:
        raise RuntimeError(f"Resource cleanup failed: {failures}")
    return failures



_UNSET = object()


def _canonical_value(value: Any) -> Any:
    """Convert YAML-like values into a stable, type-preserving JSON value."""
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("Configuration mapping keys must be strings.")
        return {key: _canonical_value(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, Path):
        return {"__path__": str(value)}
    if isinstance(value, (datetime, date)):
        return {"__date__": value.isoformat()}
    if isinstance(value, Enum):
        return {"__enum__": f"{type(value).__name__}:{value.value}"}
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Configuration values must be finite.")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Cannot canonicalize configuration value of type {type(value).__qualname__}.")


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
        shared_class_configs: Mapping[str, Any] | None = None,
        observation_model_configs: Mapping[str, Any] | None = None,
        working_dir: str | Path | None = None,
        runtime: MpiRuntime | None = None,
        mpi_resources: Mapping[str, object] | None = None,
        class_cache: MutableMapping[str, Any] | None = None,
        owns_class_cache: bool | None = None,
    ) -> None:
        self.shared_class_configs: dict[str, Any] = dict(shared_class_configs or {})
        self.observation_model_configs: dict[str, Any] = dict(observation_model_configs or {})
        self.working_dir = Path(working_dir or ".").expanduser().resolve()
        self.runtime = runtime
        self.mpi_resources: dict[str, object] = dict(mpi_resources or {})
        self._cache: MutableMapping[str, Any] = class_cache if class_cache is not None else {}
        # A supplied cache is shared state by default.  Its owner must decide
        # when the shared resources are closed; a worker context must never
        # clear a cache still used by another observation spec.
        self._owns_class_cache = class_cache is None if owns_class_cache is None else owns_class_cache
        self._transient_resources: list[Any] = []
        self._observation_spec_sequence = 0
        self._closed = False
        self.cleanup_failures: list[dict[str, str]] = []

    @property
    def class_configs(self) -> dict[str, Any]:
        return {**self.shared_class_configs, **self.observation_model_configs}

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
        """Instantiate a resolved declaration, optionally reusing the run cache.

        With ``cache=True`` (default) identical (category, config) pairs share
        one instance for the lifetime of the run; this is how the CALCEPH
        ephemeris or Earth-orientation source is opened once and reused by every program.
        """
        if config is _UNSET:
            config = self.class_configs[category]
        target_context = self if factory_context is None else factory_context
        if not cache:
            instance = instantiate(category, config, target_context)
            self._transient_resources.append(instance)
            return instance
        key = f"{cache_namespace}:{_config_key(category, config)}"
        if key not in self._cache:
            self._cache[key] = instantiate(category, config, target_context)
        return self._cache[key]

    def class_config(self, category: str, program_config: dict, key: str | None = None):
        """Select a resolved program declaration or its run default."""
        key = category if key is None else key
        if key in program_config:
            return program_config[key]
        if key in self.observation_model_configs:
            return self.observation_model_configs[key]
        return self.shared_class_configs.get(category)

    def resolve_defaults(self) -> None:
        """Resolve run defaults once before assembling any program."""
        from lunarops.classes.observation.assembly import observation_model_schema

        self.shared_class_configs = validate_global_class_configs(self.shared_class_configs, path="shared")
        self.observation_model_configs = observation_model_schema().resolve(
            self.observation_model_configs,
            path="observationModel",
        )

    # -- paths ---------------------------------------------------------------
    def resolve_path(self, value) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = self.working_dir / path
        return path.resolve()

    def next_observation_spec_id(self) -> str:
        self._observation_spec_sequence += 1
        return f"{id(self)}:{self._observation_spec_sequence}"

    def close(self, *, strict: bool = True) -> None:
        if self._closed:
            return
        self._closed = True
        resources = tuple(self._transient_resources)
        self._transient_resources.clear()
        if self._owns_class_cache:
            resources += tuple(self._cache.values())
            self._cache.clear()
        self.cleanup_failures.extend(close_resources(resources, owner="run-context"))
        if strict and self.cleanup_failures:
            raise RuntimeError(f"Run-context cleanup failed: {self.cleanup_failures}")

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close(strict=exc_type is None)
