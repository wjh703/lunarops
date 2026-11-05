"""Run context shared by all program calls of one run.

The context holds the run-level model configs (``shared:`` and ``observationModel:`` sections), the working
directory for relative paths and the optional MPI runtime. Model classes created through the context with
identical declarations share one instance for the run, and their resources are closed with the context.
"""

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
    """Cache of model class instances keyed by category and serialized config.

    Attributes
    ----------
    resources : contextlib.ExitStack
        Close callbacks of the cached instances.
    """

    def __init__(self):
        super().__init__()
        self.resources = ExitStack()

    def close(self):
        """Close every cached instance and empty the cache."""
        self.resources.close()
        self.clear()


class RunContext:
    """Run-level state passed to every program call and model class factory.

    The context is a context manager; leaving it closes the resources it owns.

    Parameters
    ----------
    shared_class_configs : Mapping, optional
        Configs of the ``shared:`` section, keyed by category.
    observation_model_configs : Mapping, optional
        Configs of the ``observationModel:`` section, keyed by category.
    working_dir : str or Path, optional
        Base directory for relative paths; the current directory when None.
    runtime : MpiRuntime, optional
        MPI runtime of the master rank; None for serial runs.
    mpi_resources : Mapping, optional
        Preloaded model data sent with an observation spec to MPI workers, keyed by category (for example the
        Earth-orientation table under ``earthRotation``); model factories use it instead of reading files.
    class_cache : ClassCache, optional
        Existing cache of model instances to share; it is not closed with this context. A private cache is
        created and closed with the context when None.

    Attributes
    ----------
    shared_class_configs : dict
        Shared class configs, keyed by category.
    observation_model_configs : dict
        Observation-model class configs, keyed by category.
    working_dir : Path
        Absolute base directory for relative paths.
    runtime : MpiRuntime or None
        MPI runtime, or None.
    mpi_resources : dict
        Preloaded model data keyed by category.
    """

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
        """Shared and observation-model class configs merged; observation-model entries take precedence."""
        return {**self.shared_class_configs, **self.observation_model_configs}

    def create_class(
        self,
        category: str,
        config,
        *,
        cache: bool = True,
    ):
        """Instantiate a resolved class declaration, optionally reusing the run cache.

        With ``cache=True`` identical (category, config) pairs share one instance for the lifetime of the
        cache, so that, for example, a CALCEPH ephemeris or an Earth-orientation source is opened once and
        used by every program. An instance with a ``close`` method is closed with the cache, or with this
        context when not cached.

        Parameters
        ----------
        category : str
            Class category, such as ``ephemerides``.
        config : dict
            Resolved class declaration with a ``type`` entry.
        cache : bool, optional
            Reuse and store the instance in the run cache. Default True.

        Returns
        -------
        object
            Model class instance.
        """
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
        """Return the class config that applies to a program call.

        The program config entry takes precedence over the ``observationModel:`` entry, which takes
        precedence over the ``shared:`` entry of the category.

        Parameters
        ----------
        category : str
            Class category, used for the ``shared:`` lookup.
        program_config : dict
            Resolved program config.
        key : str, optional
            Key looked up in the program and observation-model configs; ``category`` when None.

        Returns
        -------
        object
            Class config.

        Raises
        ------
        KeyError
            If no section defines the config.
        """
        key = category if key is None else key
        if key in program_config:
            return program_config[key]
        if key in self.observation_model_configs:
            return self.observation_model_configs[key]
        return self.shared_class_configs[category]

    def resolve_defaults(self) -> None:
        """Validate the ``shared:`` and ``observationModel:`` configs and fill in their defaults."""
        from lunarops.classes.observation.assembly import observation_model_schema

        self.shared_class_configs = validate_global_class_configs(self.shared_class_configs, path="shared")
        self.observation_model_configs = observation_model_schema().resolve(
            self.observation_model_configs,
            path="observationModel",
        )

    def resolve_path(self, value) -> Path:
        """Return a path as an absolute path; relative paths are taken from ``working_dir``.

        Parameters
        ----------
        value : str or Path
            Path, ``~`` expanded.

        Returns
        -------
        Path
            Absolute, resolved path.
        """
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = self.working_dir / path
        return path.resolve()

    def next_observation_spec_id(self) -> str:
        """Return an identifier for a new observation spec, unique within this context."""
        return f"{id(self)}:{next(self._observation_spec_sequence)}"

    def close(self) -> None:
        """Close the resources owned by this context, including its private class cache."""
        self._resources.close()

    def __enter__(self) -> Self:
        """Return the context itself."""
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Close the context."""
        self.close()
