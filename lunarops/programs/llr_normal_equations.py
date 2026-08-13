"""Build and store LLR normal equations at one linearization point."""

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
        name="LlrNormalEquations",
        summary="Build normal equations at one fixed LLR linearization.",
        inputs=(ArtifactSlot("inputFilesNormalPoints", "NormalPointFile", many=True),),
        outputs=(ArtifactSlot("outputFileNormalEquations", "NormalEquationFile"),),
        fields=observation_fields(parametrized=True),
    )
)
def llr_normal_equations(config: dict, context: RunContext):
    from lunarops.estimation.linearization import (
        build_normal_equations_streaming,
    )
    from lunarops.fileio.normal_equations import write_normal_equations

    datasets = load_datasets(config, context)
    parametrization = build_parametrization(config, context)
    processor = build_processor(config, context)
    ephemeris = processor.observation_model.ephemeris

    equation_source = build_equation_source(config, context, datasets, processor)
    equations = equation_source(1)
    parametrization.setup(equations, processor.model_state)
    names = parametrization.parameter_names()
    x0 = parametrization.reference_values_for(names)
    normals = build_normal_equations_streaming(
        equations,
        parametrization,
        parameter_names=names,
        x0=x0,
        sources=sorted(datasets),
        ephemeris=ephemeris.source_file_path,
        lunar_relativistic_scale_convention=ephemeris.lunar_relativistic_scale_convention.value,
        l_b_minus_l_l=ephemeris.l_b_minus_l_l,
        compatibility=model_compatibility_fingerprint(config, context),
    )

    out = context.resolve_path(config["outputFileNormalEquations"])
    write_normal_equations(normals, out)
    print(f"[LlrNormalEquations] {normals.obs_count} obs, {len(names)} parameters -> {out}")
    return normals


__all__ = ["llr_normal_equations"]
