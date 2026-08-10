"""Persist and consume fixed-linearization LLR observation equations."""

from __future__ import annotations

from lunarops.config.context import RunContext
from lunarops.llr_workflow import (
    build_equation_source,
    build_parametrization,
    build_processor,
    load_datasets,
    model_compatibility_fingerprint,
)
from lunarops.programs.registry import ArtifactSlot, ProgramSpec, program
from lunarops.programs.specs import observation_fields


@program(
    ProgramSpec(
        name="LlrObservationEquations",
        summary="Freeze LLR design rows and reduced observations at one model state.",
        inputs=(ArtifactSlot("inputFilesNormalPoints", "NormalPointFile", many=True),),
        outputs=(ArtifactSlot("outputFileObservationEquations", "ObservationEquationFile"),),
        fields=observation_fields(parametrized=True),
    )
)
def llr_observation_equations(config: dict, context: RunContext):
    from lunarops.fileio.observation_equation_file import (
        write_observation_equations,
    )
    from lunarops.estimation.observation_equations import FrozenObservationEquations

    datasets = load_datasets(config, context)
    processor = build_processor(config, context)
    parametrization = build_parametrization(config, context)
    equations = build_equation_source(config, context, datasets, processor)(1)
    parametrization.setup(equations, processor.model_state)
    source_by_identity = {
        int(record.index): source for source, dataset in datasets.items() for record in dataset.records
    }
    ephemeris = processor.observation_model.ephemeris
    frozen = FrozenObservationEquations.from_equations(
        equations,
        parametrization,
        source_by_identity=source_by_identity,
        metadata={
            "sources": sorted(datasets),
            "ephemeris": str(ephemeris.source_file_path),
            "lunar_relativistic_scale_convention": ephemeris.lunar_relativistic_scale_convention.value,
            "l_b_minus_l_l": ephemeris.l_b_minus_l_l,
            "compatibility": model_compatibility_fingerprint(config, context),
        },
    )
    output = context.resolve_path(config["outputFileObservationEquations"])
    write_observation_equations(frozen, output)
    print(
        f"[LlrObservationEquations] {len(frozen.identities)} row(s), "
        f"{len(frozen.parameter_names)} parameter(s) -> {output}"
    )
    return frozen


@program(
    ProgramSpec(
        name="ObservationEquationsToNormals",
        summary="Accumulate a frozen observation-equation file into normal equations.",
        inputs=(ArtifactSlot("inputFileObservationEquations", "ObservationEquationFile"),),
        outputs=(ArtifactSlot("outputFileNormalEquations", "NormalEquationFile"),),
    )
)
def observation_equations_to_normals(config: dict, context: RunContext):
    from lunarops.fileio.observation_equation_file import read_observation_equations
    from lunarops.fileio.normal_equation_file import write_normal_equations

    frozen = read_observation_equations(context.resolve_path(config["inputFileObservationEquations"]))
    normals = frozen.normal_equations()
    output = context.resolve_path(config["outputFileNormalEquations"])
    write_normal_equations(normals, output)
    print(
        f"[ObservationEquationsToNormals] {normals.obs_count} row(s), "
        f"{len(normals.parameter_names)} parameter(s) -> {output}"
    )
    return normals


__all__ = ["llr_observation_equations", "observation_equations_to_normals"]
