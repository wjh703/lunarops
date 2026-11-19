"""Optical path delay inside the solid corner-cube prisms of the lunar retroreflectors."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from lunarops.config.schema import ConfigSchema, number, sequence, string

#: Front-face-to-vertex depth h of the fused-silica prisms, in mm (Williams & Boggs 2020, Sec. 3.18).
JPL_2020_CORNER_CUBE_DEPTHS = (
    {"reflector": "APOLLO11", "depthMm": 27.5},
    {"reflector": "APOLLO14", "depthMm": 27.5},
    {"reflector": "APOLLO15", "depthMm": 27.5},
    {"reflector": "LUNOKHOD1", "depthMm": 43.5},
    {"reflector": "LUNOKHOD2", "depthMm": 43.5},
)
#: Refractive index of the fused silica (Heraeus), as in Williams & Boggs (2020), Sec. 3.18.
FUSED_SILICA_REFRACTIVE_INDEX = 1.4555

_REFLECTOR_SCHEMA = ConfigSchema(
    fields=(
        string("reflector", required=True, non_empty=True, description="Reflector catalog key."),
        number("depthMm", required=True, gt=0.0, description="Front-face-to-vertex depth h of the prisms, in mm."),
    )
)


class CornerCubeDelay:
    """One-way path delay ``n h`` from the front face of a reflector array to the prism vertex and back.

    A ray entering a solid corner cube travels the depth ``h`` from the front face to the vertex in glass
    of refractive index ``n``, so seen from outside the bounce happens ``n h`` behind the front face. Adding
    ``n h`` to each leg refers the reflector coordinates to the centre of the array at the front face, the
    convention of the JPL reflector coordinates. With n = 1.4555, this is 40.0 mm (Apollo, h = 27.5 mm) and
    63.3 mm (Lunokhod, h = 43.5 mm) per leg. The longer paths of off-axis rays and the wavelength dependence of
    ``n`` (below 1 mm) are ignored.

    Parameters
    ----------
    reflectors : Sequence[Mapping[str, Any]]
        Per reflector ``reflector`` (catalog key) and ``depthMm``.
    refractive_index : float
        Group refractive index of the prism glass.

    Raises
    ------
    ValueError
        If a reflector is listed twice, or (when evaluated) a reflector is not listed.

    References
    ----------
    .. [1] Williams, J. G., & Boggs, D. H. (2020). The JPL lunar laser range model 2020. JPL IOM 335N-20-01,
       Sec. 3.18, Eq. (20).
    """

    SCHEMA: ClassVar[ConfigSchema] = ConfigSchema(
        fields=(
            sequence(
                "reflectors",
                item_kind="mapping",
                item_nested=_REFLECTOR_SCHEMA,
                min_items=1,
                default=[dict(item) for item in JPL_2020_CORNER_CUBE_DEPTHS],
                description="Prism depth per reflector. Defaults to Williams & Boggs (2020), Sec. 3.18.",
            ),
            number(
                "refractiveIndex",
                default=FUSED_SILICA_REFRACTIVE_INDEX,
                gt=1.0,
                description="Refractive index n of the prism glass.",
            ),
        ),
        type_name="cornerCube",
    )

    @classmethod
    def from_config(cls, cfg, ctx):
        """Create the model from its configuration."""
        return cls(cfg["reflectors"], refractive_index=float(cfg["refractiveIndex"]))

    def __init__(
        self,
        reflectors: Sequence[Mapping[str, Any]] = JPL_2020_CORNER_CUBE_DEPTHS,
        *,
        refractive_index: float = FUSED_SILICA_REFRACTIVE_INDEX,
    ) -> None:
        self.delays_m: dict[str, float] = {}
        for item in reflectors:
            key = str(item["reflector"])
            if key in self.delays_m:
                raise ValueError(f"cornerCube lists reflector {key!r} twice.")
            self.delays_m[key] = refractive_index * 1.0e-3 * float(item["depthMm"])

    def path_delay_m(self, reflector_key: str | None) -> float:
        """Return the one-way path delay of a reflector, in metres.

        Raises
        ------
        ValueError
            If the reflector is not listed.
        """
        try:
            return self.delays_m[str(reflector_key)]
        except KeyError:
            raise ValueError(
                f"cornerCube has no prism depth for reflector {reflector_key!r}; known: {sorted(self.delays_m)}."
            ) from None


__all__ = ["FUSED_SILICA_REFRACTIVE_INDEX", "JPL_2020_CORNER_CUBE_DEPTHS", "CornerCubeDelay"]
