"""Process-wide cache of loaded displacement coefficient files."""

from functools import lru_cache


@lru_cache(maxsize=16)
def coefficient_file(factory, path):
    """Return ``factory(path)``, cached by ``(factory, path)`` for up to 16 coefficient files."""
    return factory(path)
