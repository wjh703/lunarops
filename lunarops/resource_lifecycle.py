"""Small lifecycle helpers shared by run contexts and execution backends."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

logger = logging.getLogger(__name__)


def close_resource(resource: Any, *, owner: str) -> dict[str, str] | None:
    """Close one optional resource and report cleanup failures."""
    close = getattr(resource, "close", None)
    if not callable(close):
        return
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


__all__ = ["close_resource", "close_resources"]
