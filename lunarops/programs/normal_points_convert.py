"""Import of CRD, MINI and LunarOps normal-point files into one ``normalPoint`` file."""

from __future__ import annotations

from lunarops import console
from lunarops.config.context import RunContext
from lunarops.config.schema import number, string
from lunarops.programs.registry import ArtifactSlot, program, program_spec


def _validate_conversion(config, path):
    """Require ``crdUncertaintyPolicy`` for inputs named as CRD/FRD and ``crdUncertaintyTwoWayS`` for ``fixed``."""
    declared_crd = any(
        str(value).casefold().endswith((".crd", ".frd", ".crd.gz", ".frd.gz"))
        for value in config["inputFilesNormalPoints"]
    )
    policy = config.get("crdUncertaintyPolicy")
    if declared_crd and policy is None:
        raise ValueError(f"{path}.crdUncertaintyPolicy is required for declared CRD/FRD inputs.")
    if policy == "fixed" and config.get("crdUncertaintyTwoWayS") is None:
        raise ValueError(f"{path}.crdUncertaintyTwoWayS is required for fixed uncertainty.")
    return config


@program(
    program_spec(
        name="NormalPointsConvert",
        summary="Import CRD, MINI, or native normal points into one canonical file.",
        inputs=(
            ArtifactSlot(
                "inputFilesNormalPoints",
                "ExternalNormalPointFile",
                external=True,
                many=True,
                description="CRD v2, MINI or native normal-point files, or directories searched recursively for them.",
            ),
        ),
        outputs=(
            ArtifactSlot("outputFileNormalPoints", "normalPoint", description="Output canonical normal-point file."),
            ArtifactSlot(
                "outputFileImportReport",
                "normalPointImportReport",
                description="Output report with source checksums, record counts and rejected lines.",
            ),
        ),
        validator=_validate_conversion,
        fields=(
            string(
                "datasetName",
                default="normal-points",
                non_empty=True,
                description="Name stored in the header of the output file.",
            ),
            string(
                "crdUncertaintyPolicy",
                choices=("binRmsAsSigma", "fixed"),
                description=(
                    "Source of the two-way uncertainty of CRD normal points: `binRmsAsSigma` uses the bin RMS of "
                    "record 11, `fixed` uses `crdUncertaintyTwoWayS`. Required for CRD input."
                ),
            ),
            number(
                "crdUncertaintyTwoWayS",
                gt=0,
                description=(
                    "Two-way uncertainty assigned to every CRD normal point under the `fixed` policy, in seconds."
                ),
            ),
            number(
                "crdMeteorologyMaxAgeSeconds",
                default=1800.0,
                ge=0,
                description=(
                    "Largest allowed time between a CRD normal point and the nearest meteorological record, in seconds."
                ),
            ),
            string(
                "miniTemperatureUnit",
                default="deciCelsius",
                choices=("deciCelsius", "deciKelvin"),
                description="Unit of the MINI temperature field: `deciCelsius` (0.1 °C) or `deciKelvin` (0.1 K).",
            ),
        ),
    )
)
def normal_points_convert(config: dict, context: RunContext):
    """Read normal-point source files and write them as one ``normalPoint`` file.

    Reads every supported file found under ``inputFilesNormalPoints`` (the output file itself is skipped),
    applying the CRD uncertainty and meteorology settings and the MINI temperature unit, and writes the
    combined records to ``outputFileNormalPoints`` and a ``normalPointImportReport`` with the SHA-256 digest,
    record counts and import issues of each source to ``outputFileImportReport``. See the program reference
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
        Path of the written normal-point file.

    Raises
    ------
    FileNotFoundError
        If no supported input file is found.
    """
    from lunarops.classes.observation.normal_points import combine_npt_datasets
    from lunarops.fileio.archive import sha256_file
    from lunarops.fileio.formats.normal_point_sources import (
        read_normal_point_source,
        resolve_normal_point_sources,
    )
    from lunarops.fileio.normal_points import write_normal_points
    from lunarops.fileio.yaml_artifact import write_structured_text

    output = context.resolve_path(config["outputFileNormalPoints"])
    sources = [
        path
        for path in resolve_normal_point_sources(
            [context.resolve_path(value) for value in config["inputFilesNormalPoints"]]
        )
        if path.resolve() != output.resolve()
    ]
    if not sources:
        raise FileNotFoundError("NormalPointsConvert found no supported input files.")
    datasets = [
        read_normal_point_source(
            path,
            crd_options={
                "uncertainty_policy": config.get("crdUncertaintyPolicy"),
                "uncertainty_two_way_s": config.get("crdUncertaintyTwoWayS"),
                "meteorology_max_age_s": config["crdMeteorologyMaxAgeSeconds"],
            },
            mini_temperature_unit=config["miniTemperatureUnit"],
        )
        for path in sources
    ]
    combined = combine_npt_datasets(
        datasets,
        name=str(config["datasetName"]),
    )
    write_normal_points(combined, output)
    report = {
        "sourceCount": len(sources),
        "recordCount": len(combined),
        "inputRecordCount": combined.n_input_records,
        "invalidRecordCount": combined.n_invalid_records,
        "sources": [
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "recordCount": len(dataset),
                "inputRecordCount": dataset.n_input_records,
                "invalidRecordCount": dataset.n_invalid_records,
                "issues": dataset.import_issues,
                "semantics": dataset.import_metadata,
            }
            for path, dataset in zip(sources, datasets)
        ],
    }
    write_structured_text(
        context.resolve_path(config["outputFileImportReport"]),
        "normalPointImportReport",
        report,
    )
    console.summary({"normal points": len(combined), "input files": len(sources)}, outputs=[output])
    return output


__all__ = ["normal_points_convert"]
