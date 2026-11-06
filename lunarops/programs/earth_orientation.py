"""Programs that convert and merge IERS Earth-orientation parameter (EOP) products.

The IERS 20 C04 series and the IERS finals2000A (Bulletin A rapid service and prediction) file are converted
to ``earthOrientationParameter`` artifacts, daily samples tagged with their UTC MJD, and the two can be merged
into one series.
"""

from __future__ import annotations

from pathlib import Path

from lunarops import console
from lunarops.classes.frames.earth_orientation import read_iers_c04, read_iers_rapid
from lunarops.config.context import RunContext
from lunarops.fileio.earth_orientation import (
    read_earth_orientation_parameter,
    write_earth_orientation_parameter,
)
from lunarops.programs.registry import ArtifactSlot, program, program_spec


def _output(config: dict, context: RunContext) -> Path:
    return context.resolve_path(config["outputFileEarthOrientationParameter"])


@program(
    program_spec(
        name="IersC04EarthOrientationParameter",
        summary="Convert an IERS 20 C04 daily file to the native Earth-orientation format.",
        inputs=(
            ArtifactSlot(
                "inputFileC04",
                "ExternalEarthOrientationParameterFile",
                external=True,
                description="IERS 20 C04 daily EOP file.",
            ),
        ),
        outputs=(
            ArtifactSlot(
                "outputFileEarthOrientationParameter",
                "earthOrientationParameter",
                description="Output native EOP file with source and quality columns.",
            ),
        ),
    )
)
def iers_c04_earth_orientation_parameter(config: dict, context: RunContext):
    """Convert an IERS 20 C04 daily EOP file to an ``earthOrientationParameter`` artifact.

    Reads ``inputFileC04`` and writes ``outputFileEarthOrientationParameter``. See the program reference
    (``docs/reference/programs.md``) for every configuration key.

    Parameters
    ----------
    config : dict
        Resolved program config.
    context : RunContext
        Run context.

    Returns
    -------
    Path
        Path of the written EOP file.
    """
    samples = read_iers_c04(context.resolve_path(config["inputFileC04"]))
    output = _output(config, context)
    write_earth_orientation_parameter(samples, output)
    console.summary({"samples": len(samples)}, outputs=[output])
    return output


@program(
    program_spec(
        name="IersRapidEarthOrientationParameter",
        summary="Convert IERS finals2000A Bulletin-A rapid and prediction values to native EOP.",
        inputs=(
            ArtifactSlot(
                "inputFileFinals2000A",
                "ExternalEarthOrientationParameterFile",
                external=True,
                description="IERS finals2000A file (Bulletin A rapid service and prediction).",
            ),
        ),
        outputs=(
            ArtifactSlot(
                "outputFileEarthOrientationParameter",
                "earthOrientationParameter",
                description="Output native EOP file with source and quality columns.",
            ),
        ),
    )
)
def iers_rapid_earth_orientation_parameter(config: dict, context: RunContext):
    """Convert an IERS finals2000A file to an ``earthOrientationParameter`` artifact.

    Reads the Bulletin A rapid-service and prediction values of ``inputFileFinals2000A`` and writes
    ``outputFileEarthOrientationParameter``. See the program reference (``docs/reference/programs.md``) for
    every configuration key.

    Parameters
    ----------
    config : dict
        Resolved program config.
    context : RunContext
        Run context.

    Returns
    -------
    Path
        Path of the written EOP file.
    """
    samples = read_iers_rapid(context.resolve_path(config["inputFileFinals2000A"]))
    output = _output(config, context)
    write_earth_orientation_parameter(samples, output)
    console.summary({"samples": len(samples)}, outputs=[output])
    return output


@program(
    program_spec(
        name="EarthOrientationParameterMerge",
        summary="Merge C04 and Bulletin-A EOP, preferring C04 on overlapping days.",
        inputs=(
            ArtifactSlot(
                "inputFileC04EarthOrientationParameter",
                "earthOrientationParameter",
                description="Native EOP file converted from C04.",
            ),
            ArtifactSlot(
                "inputFileRapidEarthOrientationParameter",
                "earthOrientationParameter",
                description="Native EOP file converted from finals2000A.",
            ),
        ),
        outputs=(
            ArtifactSlot(
                "outputFileEarthOrientationParameter",
                "earthOrientationParameter",
                description="Output merged EOP file; C04 values take precedence on days present in both inputs.",
            ),
        ),
    )
)
def earth_orientation_parameter_merge(config: dict, context: RunContext):
    """Merge a C04 and a rapid EOP series into one series sorted by UTC MJD.

    Reads the ``earthOrientationParameter`` artifacts ``inputFileC04EarthOrientationParameter`` and
    ``inputFileRapidEarthOrientationParameter``; on days present in both, the C04 sample is kept. Writes
    ``outputFileEarthOrientationParameter``. See the program reference (``docs/reference/programs.md``) for
    every configuration key.

    Parameters
    ----------
    config : dict
        Resolved program config.
    context : RunContext
        Run context.

    Returns
    -------
    Path
        Path of the written EOP file.
    """
    c04 = read_earth_orientation_parameter(context.resolve_path(config["inputFileC04EarthOrientationParameter"]))
    rapid = read_earth_orientation_parameter(context.resolve_path(config["inputFileRapidEarthOrientationParameter"]))
    by_mjd = {sample.mjd_utc: sample for sample in rapid}
    by_mjd.update({sample.mjd_utc: sample for sample in c04})
    samples = tuple(by_mjd[mjd] for mjd in sorted(by_mjd))
    output = _output(config, context)
    write_earth_orientation_parameter(samples, output)
    overlap = len({sample.mjd_utc for sample in c04} & {sample.mjd_utc for sample in rapid})
    console.summary({"samples": len(samples), "overlap days (C04 used)": overlap}, outputs=[output])
    return output


__all__ = [
    "earth_orientation_parameter_merge",
    "iers_c04_earth_orientation_parameter",
    "iers_rapid_earth_orientation_parameter",
]
