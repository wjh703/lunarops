"""Normal-equation accumulation and solution programs."""

from __future__ import annotations

from lunarops.config.context import RunContext
from lunarops.programs.registry import ArtifactSlot, ProgramSpec, program


@program(
    ProgramSpec(
        name="NormalsAccumulate",
        summary="Align parameters by structured name and add normal equations.",
        inputs=(ArtifactSlot("inputFilesNormalEquations", "NormalEquationFile", many=True),),
        outputs=(ArtifactSlot("outputFileNormalEquations", "NormalEquationFile"),),
    )
)
def normals_accumulate(config: dict, context: RunContext):
    from lunarops.fileio.normal_equation_file import read_normal_equations, write_normal_equations

    paths = [context.resolve_path(value) for value in config["inputFilesNormalEquations"]]
    total = read_normal_equations(paths[0])
    for path in paths[1:]:
        total = total.add(read_normal_equations(path))
    output = context.resolve_path(config["outputFileNormalEquations"])
    if any(path.resolve() == output.resolve() for path in paths):
        raise ValueError("NormalsAccumulate output must not also be an input.")
    write_normal_equations(total, output)
    print(
        f"[NormalsAccumulate] {len(paths)} system(s), {total.obs_count} observation(s), "
        f"{len(total.parameter_names)} parameter(s) -> {output}"
    )
    return total


@program(
    ProgramSpec(
        name="NormalsSolve",
        summary="Solve normal equations into a typed solution, covariance, and report.",
        inputs=(ArtifactSlot("inputFileNormalEquations", "NormalEquationFile"),),
        outputs=(
            ArtifactSlot("outputFileSolution", "ParameterVectorFile"),
            ArtifactSlot("outputFileCovariance", "CovarianceMatrixFile"),
            ArtifactSlot("outputFileReport", "SolutionReportFile"),
        ),
    )
)
def normals_solve(config: dict, context: RunContext):
    import numpy as np

    from lunarops.estimation.adjustment_options import PARAMETER_UNCERTAINTY_SIGMA_MULTIPLIER
    from lunarops.estimation.parameter_products import CovarianceMatrix, ParameterVector
    from lunarops.fileio.normal_equation_file import read_normal_equations
    from lunarops.fileio.parameters import (
        write_covariance,
        write_parameter_vector,
    )
    from lunarops.fileio.structured_text import write_structured_text

    normals = read_normal_equations(context.resolve_path(config["inputFileNormalEquations"]))
    values, cofactor, sigma0 = normals.solve()
    diagonal = np.maximum(np.diag(cofactor), 0.0)
    one_sigma = np.sqrt(diagonal)
    covariance_values = cofactor
    covariance_kind = "cofactor"
    if sigma0 is not None:
        one_sigma = sigma0 * one_sigma
        covariance_values = sigma0 * sigma0 * cofactor
        covariance_kind = "posteriorCovariance"
    solution = ParameterVector(
        parameter_names=tuple(normals.parameter_names),
        values=values,
        units=tuple(normals.parameter_units),
        uncertainties=PARAMETER_UNCERTAINTY_SIGMA_MULTIPLIER * one_sigma,
        kind="correction",
        uncertainty_sigma_multiplier=PARAMETER_UNCERTAINTY_SIGMA_MULTIPLIER,
    )
    covariance = CovarianceMatrix(
        tuple(normals.parameter_names),
        covariance_values,
        tuple(normals.parameter_units),
        covariance_kind,
    )
    solution_path = context.resolve_path(config["outputFileSolution"])
    covariance_path = context.resolve_path(config["outputFileCovariance"])
    report_path = context.resolve_path(config["outputFileReport"])
    write_parameter_vector(solution, solution_path)
    write_covariance(covariance, covariance_path)
    write_structured_text(
        report_path,
        "normalEquationSolutionReport",
        {
            "observationCount": normals.obs_count,
            "parameterCount": len(normals.parameter_names),
            "degreesOfFreedom": normals.obs_count - len(normals.parameter_names),
            "sigma0Post": sigma0,
            "parameterUncertaintySigmaMultiplier": (PARAMETER_UNCERTAINTY_SIGMA_MULTIPLIER),
            "solutionFile": str(solution_path),
            "covarianceFile": str(covariance_path),
        },
    )
    sigma_text = "undefined" if sigma0 is None else f"{sigma0:.6g}"
    print(f"[NormalsSolve] {len(values)} parameter(s), sigma0={sigma_text} -> {solution_path}")
    return solution


__all__ = ["normals_accumulate", "normals_solve"]
