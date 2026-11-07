"""Native ``massCatalog`` of gravitational parameters used by dynamics programs.

The file is a structured text artifact (a ``lunarops massCatalog`` header followed by a YAML mapping) with
the keys ``name``, ``source``, ``gmUnit`` (always ``m3/s2``), ``bodyCount`` and ``bodies``. Each body entry
holds ``bodyId``, ``name``, ``ephemerisTarget`` (NAIF ID), ``gmM3S2`` (GM in m^3/s^2), ``groups`` and
``source``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np

from lunarops.classes.ephemerides.body_ids import body_name, naif_id

from .yaml_artifact import read_structured_text, write_structured_text


@dataclass(frozen=True, slots=True)
class BodyMass:
    """Gravitational parameter of one body, with its catalog groups and ephemeris target.

    Values are normalized on construction: ``body_id`` to its canonical body name, ``gm_m3_s2`` and
    ``ephemeris_target`` to float and int, ``groups`` stripped and de-duplicated in order, ``name`` stripped
    (defaulting to ``body_id``) and ``source`` stripped.

    Attributes
    ----------
    body_id : str
        Canonical body identifier.
    gm_m3_s2 : float
        Gravitational parameter GM, in m^3/s^2; positive and finite.
    groups : tuple of str
        Non-empty group labels used to select bodies; at least one.
    ephemeris_target : int
        NAIF ID of the body in the ephemeris; must equal the canonical NAIF ID when the body has one.
    name : str
        Display name.
    source : str
        Provenance of the GM value.

    Raises
    ------
    ValueError
        If GM is not positive and finite, no non-empty group is given, or ``ephemeris_target`` contradicts
        the canonical NAIF ID.
    """

    body_id: str
    gm_m3_s2: float
    groups: tuple[str, ...]
    ephemeris_target: int
    name: str = ""
    source: str = ""

    def __post_init__(self) -> None:
        """Validate and normalize the fields."""
        body_id = body_name(self.body_id)
        gm = float(self.gm_m3_s2)
        groups = tuple(dict.fromkeys(str(value).strip() for value in self.groups))
        if not np.isfinite(gm) or gm <= 0:
            raise ValueError(f"Body {body_id!r} requires a positive finite GM")
        if not groups or any(not value for value in groups):
            raise ValueError(f"Body {body_id!r} requires at least one non-empty group")
        target = int(self.ephemeris_target)
        canonical_target = naif_id(body_id)
        if canonical_target is not None and canonical_target != target:
            raise ValueError(
                f"Body {body_id!r} has canonical NAIF ID {canonical_target}, not ephemeris target {target}"
            )
        object.__setattr__(self, "body_id", body_id)
        object.__setattr__(self, "gm_m3_s2", gm)
        object.__setattr__(self, "groups", groups)
        object.__setattr__(self, "ephemeris_target", target)
        object.__setattr__(self, "name", self.name.strip() or body_id)
        object.__setattr__(self, "source", self.source.strip())


class MassCatalog:
    """Named catalog of body gravitational parameters.

    Parameters
    ----------
    bodies : Iterable of BodyMass
        Catalog bodies, in catalog order.
    name : str
        Catalog name; stripped and non-empty.
    source : str, optional
        Catalog provenance; stripped.

    Attributes
    ----------
    name : str
        Catalog name.
    source : str
        Catalog provenance.
    bodies : tuple of BodyMass
        Bodies in catalog order.
    by_id : Mapping of str to BodyMass
        Read-only lookup by canonical body identifier.

    Raises
    ------
    ValueError
        If the catalog is empty, body identifiers or ephemeris targets repeat, or the name is empty.
    """

    def __init__(self, bodies: Iterable[BodyMass], *, name: str, source: str = "") -> None:
        records = tuple(bodies)
        by_id = {record.body_id: record for record in records}
        targets = [record.ephemeris_target for record in records]
        if not records:
            raise ValueError("Mass catalog must not be empty")
        if len(by_id) != len(records):
            raise ValueError("Mass catalog body IDs must be unique")
        if len(set(targets)) != len(targets):
            raise ValueError("Mass catalog ephemeris targets must be unique")
        self.name = str(name).strip()
        self.source = str(source).strip()
        if not self.name:
            raise ValueError("Mass catalog name must not be empty")
        self.bodies = records
        self.by_id: Mapping[str, BodyMass] = MappingProxyType(by_id)

    def require(self, body_id: str) -> BodyMass:
        """Return the body with the given identifier.

        Parameters
        ----------
        body_id : str
            Body identifier; normalized to its canonical body name.

        Returns
        -------
        BodyMass
            Catalog entry of the body.

        Raises
        ------
        KeyError
            If the catalog has no such body.
        """
        key = body_name(body_id)
        try:
            return self.by_id[key]
        except KeyError as exc:
            raise KeyError(f"Mass catalog {self.name!r} has no body {key!r}") from exc

    def select(self, *, groups: Iterable[str] = (), body_ids: Iterable[str] = ()) -> tuple[BodyMass, ...]:
        """Return the bodies that belong to any requested group or are requested by identifier.

        Parameters
        ----------
        groups : Iterable of str, optional
            Group labels; surrounding whitespace is stripped.
        body_ids : Iterable of str, optional
            Body identifiers; normalized to canonical body names.

        Returns
        -------
        tuple of BodyMass
            Selected bodies in catalog order, each once.

        Raises
        ------
        KeyError
            If a requested body or group is not in the catalog.
        """
        selected_groups = {str(value).strip() for value in groups}
        selected_ids = {body_name(value) for value in body_ids}
        unknown_ids = selected_ids - set(self.by_id)
        if unknown_ids:
            raise KeyError(f"Mass catalog has no requested bodies: {sorted(unknown_ids)}")
        known_groups = {group for body in self.bodies for group in body.groups}
        unknown_groups = selected_groups - known_groups
        if unknown_groups:
            raise KeyError(f"Mass catalog has no requested groups: {sorted(unknown_groups)}")
        return tuple(
            body for body in self.bodies if body.body_id in selected_ids or selected_groups.intersection(body.groups)
        )


def write_mass_catalog(catalog: MassCatalog, path: str | Path) -> Path:
    """Write a ``massCatalog`` structured text artifact.

    Parameters
    ----------
    catalog : MassCatalog
        Catalog to write; GM values are written in m^3/s^2.
    path : str or Path
        Target path ending in ``.txt`` or ``.txt.gz``.

    Returns
    -------
    Path
        The written path.
    """
    payload = {
        "name": catalog.name,
        "source": catalog.source,
        "gmUnit": "m3/s2",
        "bodyCount": len(catalog.bodies),
        "bodies": [
            {
                "bodyId": body.body_id,
                "name": body.name,
                "ephemerisTarget": body.ephemeris_target,
                "gmM3S2": body.gm_m3_s2,
                "groups": list(body.groups),
                "source": body.source,
            }
            for body in catalog.bodies
        ],
    }
    return write_structured_text(path, "massCatalog", payload)


def read_mass_catalog(path: str | Path) -> MassCatalog:
    """Read and validate a ``massCatalog`` structured text artifact.

    Parameters
    ----------
    path : str or Path
        Path ending in ``.txt`` or ``.txt.gz``.

    Returns
    -------
    MassCatalog
        Catalog with GM values in m^3/s^2.

    Raises
    ------
    ValueError
        If ``gmUnit`` is not ``m3/s2``, ``bodyCount`` differs from the number of bodies, or a body or the
        catalog fails ``BodyMass`` or ``MassCatalog`` validation.
    TypeError
        If ``bodies`` is not a list, a body entry is not a mapping, or ``name`` or ``source`` is not a string.
    """
    payload = read_structured_text(path, "massCatalog")
    if payload.get("gmUnit") != "m3/s2":
        raise ValueError("Mass catalog canonical GM unit must be m3/s2")
    raw_bodies = payload.get("bodies")
    if not isinstance(raw_bodies, list):
        raise TypeError("Mass catalog bodies must be a sequence")
    bodies = []
    for index, raw in enumerate(raw_bodies):
        if not isinstance(raw, dict):
            raise TypeError(f"Mass catalog bodies[{index}] must be a mapping")
        bodies.append(
            BodyMass(
                body_id=raw["bodyId"],
                gm_m3_s2=raw["gmM3S2"],
                groups=tuple(raw["groups"]),
                ephemeris_target=raw["ephemerisTarget"],
                name=raw.get("name", ""),
                source=raw.get("source", ""),
            )
        )
    if payload.get("bodyCount") != len(bodies):
        raise ValueError("Mass catalog bodyCount does not match its records")
    name = payload.get("name")
    source = payload.get("source", "")
    if not isinstance(name, str) or not isinstance(source, str):
        raise TypeError("Mass catalog name and source must be strings")
    return MassCatalog(bodies, name=name, source=source)


__all__ = ["BodyMass", "MassCatalog", "read_mass_catalog", "write_mass_catalog"]
