"""Convert and merge IERS Earth-orientation products."""

from __future__ import annotations

from pathlib import Path

from lunarops.classes.frames.earth_orientation import read_iers_c04, read_iers_rapid
from lunarops.config.context import RunContext
from lunarops.fileio.earth_orientation import (
    read_earth_orientation_parameter,
    write_earth_orientation_parameter,
)
from lunarops.programs.registry import ArtifactSlot, ProgramSpec, program


def _output(config: dict, context: RunContext) -> Path:
    return context.resolve_path(config["outputFileEarthOrientationParameter"])


@program(
    ProgramSpec(
        name="IersC04EarthOrientationParameter",
        summary="Convert an IERS 20 C04 daily file to the native Earth-orientation format.",
        inputs=(ArtifactSlot("inputFileC04", "ExternalEarthOrientationParameterFile"),),
        outputs=(ArtifactSlot("outputFileEarthOrientationParameter", "EarthOrientationParameterFile"),),
    )
)
def iers_c04_earth_orientation_parameter(config: dict, context: RunContext):
    samples = read_iers_c04(context.resolve_path(config["inputFileC04"]))
    output = _output(config, context)
    write_earth_orientation_parameter(samples, output)
    print(f"[IersC04EarthOrientationParameter] {len(samples)} sample(s) -> {output}")
    return output


@program(
    ProgramSpec(
        name="IersRapidEarthOrientationParameter",
        summary="Convert IERS finals2000A Bulletin-A rapid and prediction values to native EOP.",
        inputs=(ArtifactSlot("inputFileFinals2000A", "ExternalEarthOrientationParameterFile"),),
        outputs=(ArtifactSlot("outputFileEarthOrientationParameter", "EarthOrientationParameterFile"),),
    )
)
def iers_rapid_earth_orientation_parameter(config: dict, context: RunContext):
    samples = read_iers_rapid(context.resolve_path(config["inputFileFinals2000A"]))
    output = _output(config, context)
    write_earth_orientation_parameter(samples, output)
    print(f"[IersRapidEarthOrientationParameter] {len(samples)} sample(s) -> {output}")
    return output


@program(
    ProgramSpec(
        name="EarthOrientationParameterMerge",
        summary="Merge C04 and Bulletin-A EOP, preferring C04 on overlapping days.",
        inputs=(
            ArtifactSlot("inputFileC04EarthOrientationParameter", "EarthOrientationParameterFile"),
            ArtifactSlot("inputFileRapidEarthOrientationParameter", "EarthOrientationParameterFile"),
        ),
        outputs=(ArtifactSlot("outputFileEarthOrientationParameter", "EarthOrientationParameterFile"),),
    )
)
def earth_orientation_parameter_merge(config: dict, context: RunContext):
    c04 = read_earth_orientation_parameter(
        context.resolve_path(config["inputFileC04EarthOrientationParameter"])
    )
    rapid = read_earth_orientation_parameter(
        context.resolve_path(config["inputFileRapidEarthOrientationParameter"])
    )
    by_mjd = {sample.mjd_utc: sample for sample in rapid}
    by_mjd.update({sample.mjd_utc: sample for sample in c04})
    samples = tuple(by_mjd[mjd] for mjd in sorted(by_mjd))
    output = _output(config, context)
    write_earth_orientation_parameter(samples, output)
    overlap = len(set(sample.mjd_utc for sample in c04) & set(sample.mjd_utc for sample in rapid))
    print(
        f"[EarthOrientationParameterMerge] {len(samples)} sample(s), "
        f"{overlap} C04-over-rapid overlap(s) -> {output}"
    )
    return output


__all__ = [
    "earth_orientation_parameter_merge",
    "iers_c04_earth_orientation_parameter",
    "iers_rapid_earth_orientation_parameter",
]
