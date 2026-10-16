"""SLURM regression runs and numerical comparisons before lunar rotation work."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from lunarops.config.loader import build_run_plan
from lunarops.fileio.archive import decode_token, open_text_reader, parse_header, sha256_file
from lunarops.fileio.covariance import read_covariance
from lunarops.fileio.normal_equations import read_normal_equations
from lunarops.fileio.numeric_table import read_numeric_table
from lunarops.fileio.observation_results import read_observation_results
from lunarops.fileio.parameter_vectors import read_parameter_vector
from lunarops.fileio.prediction_results import (
    PREDICTION_ARTIFACT_TYPE,
    WINDOW_ARTIFACT_TYPE,
    read_prediction_results,
    read_prediction_windows,
)
from lunarops.fileio.structured_reports import read_lunar_orbit_metadata, read_processing_report

ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "output/regression_20261009"
MANIFEST = RUN_ROOT / "manifest.json"
BASELINE = RUN_ROOT / "baseline_code"
IGNORED_REPORT_KEYS = {"fingerprint", "elapsedSeconds", "performanceSeconds"}


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def environment_info():
    return {
        "python": platform.python_version(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "pyerfa", "PyYAML", "mpi4py", "calcephpy", "Cython")
        },
        "openmpi": subprocess.check_output(["mpirun", "--version"], text=True).splitlines()[0],
        "threads_per_rank": 1,
    }


def prepare():
    for revision, code in (("e4f8caa", BASELINE), ("c6cd6f8", RUN_ROOT / "pre_simplification_code")):
        if not code.exists():
            subprocess.run(["git", "worktree", "add", "--detach", str(code), revision], cwd=ROOT, check=True)
        for kernel in (ROOT / "lunarops").glob("*.so"):
            target = code / "lunarops" / kernel.name
            if not target.exists():
                target.symlink_to(kernel)
    cases = []
    llr = [
        ("estimate_igg3", "estimation", "lunarops_reflector_bias_adjustment_detailed.yml"),
        ("estimate_direct", "estimation", "lunarops_reflector_bias_adjustment_detailed.yml"),
        ("prediction_grid", "prediction", "lunarops_observation_prediction.yml"),
        ("prediction_a15", "prediction", "lunarops_observation_prediction_a15_comparison_202505.yml"),
        ("prediction_nightly", "prediction", "lunarops_observation_prediction_nightly_20260825_to_20260925.yml"),
        ("residual_inpop_corrected", "residual", "lunarops_oc_residuals.yml"),
        ("residual_inpop", "residual", "lunarops_oc_residuals_inpop21a.yml"),
        ("residual_epm", "residual", "lunarops_oc_residuals_epm2021.yml"),
        ("residual_de440", "residual", "lunarops_oc_residuals_de440.yml"),
        ("residual_de440_scaled", "residual", "lunarops_oc_residuals_de440_alreadyScaled.yml"),
        ("residual_de440_tdb", "residual", "lunarops_oc_residuals_de440_tdbCompatibleLunarSurface.yml"),
        ("estimate_igg3_extended", "estimation", "lunarops_reflector_bias_adjustment_detailed.yml"),
    ]
    for name, family, filename in llr:
        config = yaml.safe_load((ROOT / "configs" / filename).read_text())
        config["variables"]["dataDir"] = str(ROOT.parent / "data")
        if family == "residual" and not any(p["program"] == "NormalPointsConvert" for p in config["programs"]):
            config["programs"].insert(
                3,
                {
                    "program": "NormalPointsConvert",
                    "inputFilesNormalPoints": [str(ROOT.parent / "data/polac/TOTALOBS6924.DAT")],
                    "outputFileNormalPoints": "{outputDir}/normal_points.txt",
                    "outputFileImportReport": "{outputDir}/normal_point_import_report.txt",
                    "datasetName": "totalobs6924",
                },
            )
        if name == "estimate_direct":
            for program in config["programs"]:
                for step in program.get("processingSteps", []):
                    if step["type"] == "estimate":
                        step["robustWeighting"] = {"model": "directRejection", "k0": 3.0}
        if family == "estimation":
            for program in config["programs"]:
                for step in program.get("processingSteps", []):
                    if step.get("name") == "joint":
                        step["maxIterationCount"] = 20 if name == "estimate_igg3_extended" else 10
        cases.append({"name": name, "family": family, "source": filename, "config": config})
    for model in ("de430", "de440"):
        historical = ROOT / "output" / f"{model}_10y_current_20261005"
        metadata = read_lunar_orbit_metadata(historical / "lunar_orbit_metadata.txt")
        program = metadata["config"]
        program["program"] = "LunarOrbitPropagation"
        program["ephemerides"]["directory"] = str((ROOT / program["ephemerides"]["directory"]).resolve())
        program["inputFileMassCatalog"] = str((ROOT / program["inputFileMassCatalog"]).resolve())
        for field in program["gravityFields"].values():
            field["path"] = str((ROOT / field["path"]).resolve())
        for key in ("outputFileOrbit", "outputFileAccelerationDiagnostics", "outputFileMetadata"):
            program[key] = "{outputDir}/" + Path(program[key]).name
        cases.append(
            {
                "name": f"orbit_{model}",
                "family": "orbit",
                "historical": str(historical),
                "config": {"variables": {"outputDir": ""}, "programs": [program]},
            }
        )
    inputs = set()
    for case in cases:
        config = case.pop("config")
        case["runs"] = {}
        revisions = (
            ("baseline", "current", "pre_simplification") if case["family"] == "residual" else ("baseline", "current")
        )
        for revision in revisions:
            directory = RUN_ROOT / case["name"] / revision
            directory.mkdir(parents=True, exist_ok=True)
            config["variables"]["outputDir"] = str(directory)
            path = directory / "config.yml"
            path.write_text(yaml.safe_dump(config, sort_keys=False))
            case["runs"][revision] = {"config": str(path), "output": str(directory)}
        plan = build_run_plan(config)

        def collect(value):
            if isinstance(value, dict):
                for item in value.values():
                    collect(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    collect(item)
            elif isinstance(value, str):
                candidate = Path(value).expanduser()
                if candidate.is_file():
                    inputs.add(candidate.resolve())
                elif candidate.is_dir() and "kernels" in candidate.parts:
                    inputs.update(p.resolve() for p in candidate.iterdir() if p.suffix in {".bsp", ".bpc"})

        collect(plan.shared)
        collect(plan.observation_model)
        collect(list(plan.calls))
    payload = {
        "current_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "baseline_commit": "e4f8caa",
        "baseline_code": str(BASELINE),
        "pre_simplification_commit": "c6cd6f8",
        "cases": cases,
        "inputs_sha256": {str(p): sha256_file(p) for p in sorted(inputs)},
        "kernels_sha256": {p.name: sha256_file(p) for p in sorted((ROOT / "lunarops").glob("*.so"))},
        "environment": environment_info(),
    }
    write_json(MANIFEST, payload)
    print(f"Prepared {len(cases)} cases, {len(inputs)} hashed input files: {MANIFEST}")


def run(case, revision, mpi):
    code = {"baseline": BASELINE, "current": ROOT, "pre_simplification": RUN_ROOT / "pre_simplification_code"}[revision]
    config = case["runs"][revision]["config"]
    python = str(ROOT / ".venv/bin/python")
    environment = {
        **os.environ,
        "PYTHONPATH": str(code),
        "PYTHONUNBUFFERED": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    start = time.time()
    (Path(case["runs"][revision]["output"]) / "run.json").unlink(missing_ok=True)
    subprocess.run(
        [python, "-m", "lunarops", "validate", config, "--working-dir", str(ROOT)],
        cwd=code,
        env=environment,
        check=True,
    )
    command = [python, "-m", "lunarops", "run", config, "--working-dir", str(ROOT)]
    if mpi:
        command = ["mpirun", "-n", "32", "--map-by", "core", "--bind-to", "core", *command, "--mpi"]
    subprocess.run(command, cwd=code, env=environment, check=True)
    write_json(
        Path(case["runs"][revision]["output"]) / "run.json",
        {
            "case": case["name"],
            "revision": revision,
            "elapsed_seconds": time.time() - start,
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "ranks": 32 if mpi else 1,
            "node": os.environ.get("SLURM_JOB_NODELIST"),
            "command": command,
        },
    )


def arrays(a, b, *, atol=0.0, rtol=0.0):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return {"passed": False, "shape": [list(a.shape), list(b.shape)]}
    difference = np.abs(a - b)
    return {
        "passed": bool(np.all(np.isfinite(a)) and np.all(np.isfinite(b)) and np.allclose(a, b, atol=atol, rtol=rtol)),
        "exact": bool(np.array_equal(a, b)),
        "maximum_absolute_difference": float(difference.max(initial=0)),
        "shape": list(a.shape),
        "atol": atol,
        "rtol": rtol,
    }


def records(a, b, *, excluded=(), coordinate_atol=0.0):
    if len(a) != len(b):
        return {"passed": False, "counts": [len(a), len(b)]}
    fields = (set(a[0]) if a else set()) - set(excluded)
    other = (set(b[0]) if b else set()) - set(excluded)
    if fields != other:
        return {"passed": False, "field_difference": sorted(fields ^ other)}
    results = {}
    for field in sorted(fields):
        av, bv = [r[field] for r in a], [r[field] for r in b]
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in av + bv):
            atol = coordinate_atol if "_itrf_" in field else 0.0
            results[field] = arrays(av, bv, atol=atol)
        else:
            results[field] = {"passed": av == bv, "exact": av == bv}
    return {"passed": all(r["passed"] for r in results.values()), "rows": len(a), "fields": results}


def legacy_prediction(path):
    """Read historical v4 prediction/v3 windows only for this offline comparison."""
    with path.open() as stream:
        next(stream)
        count = int(next(stream).split()[1])
        fields = next(stream).split()[1:]
        assert next(stream).strip() == "data"
        rows = []
        for line in stream:
            row = {}
            for key, token in zip(fields, line.split(), strict=True):
                if key == "observable":
                    value = token == "1"
                elif key == "sample_count":
                    value = int(token)
                elif key.endswith(("_m", "_deg", "_s")):
                    value = float(token)
                else:
                    value = decode_token(token)
                row[key] = value
            rows.append(row)
    assert count == len(rows)
    return rows


def legacy_parameters(path):
    """Read the data rows of the archived parameterVector formats offline."""
    with path.open() as stream:
        for line in stream:
            if line.strip() == "data":
                break
        return {decode_token(fields[0]): float(fields[2]) for line in stream if (fields := line.split())}


def report_data(value):
    if isinstance(value, dict):
        return {
            k: report_data(v)
            for k, v in value.items()
            if k not in IGNORED_REPORT_KEYS and not k.startswith("outputFile")
        }
    if isinstance(value, list):
        return [report_data(v) for v in value]
    return value


def read_report(path, artifact_type):
    if artifact_type == "processingReport":
        return read_processing_report(path)
    # These full observation reports contain tens of MB of YAML. Use the
    # installed libyaml safe parser for offline comparisons.
    with open_text_reader(path) as stream:
        parse_header(stream, artifact_type, expected_version=1)
        return yaml.load(stream, Loader=yaml.CSafeLoader)


def compare(case):
    roots = [Path(case["runs"][revision]["output"]) for revision in ("baseline", "current")]
    if not all((p / "run.json").exists() for p in roots):
        return None
    old, new = roots
    result = {"case": case["name"], "family": case["family"], "checks": {}}
    checks = result["checks"]
    if case["family"] == "orbit":
        for filename in ("lunar_orbit.dat", "lunar_acceleration_diagnostics.dat"):
            _, ac, av, _ = read_numeric_table(old / filename)
            _, bc, bv, _ = read_numeric_table(new / filename)
            assert ac == bc
            checks[filename] = arrays(av, bv)
            _, hc, hv, _ = read_numeric_table(Path(case["historical"]) / filename)
            assert hc == bc
            # Historical long-arc summation changes can accumulate sub-mm roundoff.
            checks["historical_" + filename] = arrays(hv, bv, atol=0.01 if filename == "lunar_orbit.dat" else 1e-12)
        metadata = read_lunar_orbit_metadata(new / "lunar_orbit_metadata.txt")
        result["ephemeris_rtn_rms_m"] = metadata["referenceDifferenceRtnRmsM"]
        result["ephemeris_rtn_maximum_m"] = metadata["referenceDifferenceRtnMaxAbsM"]
    elif case["family"] == "prediction":
        readers = {PREDICTION_ARTIFACT_TYPE: read_prediction_results, WINDOW_ARTIFACT_TYPE: read_prediction_windows}
        for path in sorted(new.glob("*.txt")):
            with path.open() as stream:
                artifact_type = stream.readline().split()[1]
            if artifact_type not in readers:
                continue
            reader = readers[artifact_type]
            a, b = reader(old / path.name), reader(path)
            checks[path.name] = records(a, b)
            historical = ROOT / "output" / path.name
            if historical.exists():
                # Archived coordinates have six decimals; earlier evaluation
                # ordering can move the rounded last digit by one micrometre.
                # Angles, ranges, epochs and visibility remain exact checks.
                checks["historical_" + path.name] = records(legacy_prediction(historical), b, coordinate_atol=2e-6)
    elif case["family"] == "residual":
        path = next(new.glob("oc_residuals*"))
        b = read_observation_results(path)
        checks[path.name] = records(read_observation_results(old / path.name), b, excluded={"source"})
        early = Path(case["runs"]["pre_simplification"]["output"])
        if (early / "run.json").exists():
            checks["pre_simplification_" + path.name] = records(
                read_observation_results(early / path.name), b, excluded={"source"}
            )
        result["status_counts"] = dict(Counter(str(row.get("status", row.get("processing_status"))) for row in b))
        historical = ROOT / "output" / path.name
        if historical.exists():
            try:
                checks["historical_" + path.name] = records(
                    read_observation_results(historical), b, excluded={"source", "obs_time_local"}
                )
            except ValueError as exc:
                result["historical_format_note"] = str(exc)
        values = np.array([row["oc_one_way_m"] for row in b if row.get("oc_one_way_m") is not None])
        result["oc_rms_m"] = float(np.sqrt(np.mean(values**2)))
    else:
        solution = "reflector_bias_solution_igg3.txt"
        a, b = read_parameter_vector(old / solution), read_parameter_vector(new / solution)
        vector = b
        assert a.parameter_names == b.parameter_names and a.units == b.units
        checks["absolute_estimates"] = arrays(a.values, b.values)
        checks["parameter_uncertainties"] = arrays(a.uncertainties, b.uncertainties)
        a, b = (
            read_covariance(old / "reflector_bias_covariance_igg3"),
            read_covariance(new / "reflector_bias_covariance_igg3"),
        )
        checks["covariance"] = arrays(a.matrix, b.matrix)
        covariance = b.matrix
        checks["covariance_symmetry"] = arrays(covariance, covariance.T)
        checks["uncertainty_convention"] = arrays(
            vector.uncertainties, vector.uncertainty_sigma_multiplier * np.sqrt(np.diag(covariance)), rtol=1e-14
        )
        a, b = (
            read_normal_equations(old / "reflector_bias_adjustment_normals_igg3"),
            read_normal_equations(new / "reflector_bias_adjustment_normals_igg3"),
        )
        for field in ("N", "W", "x0"):
            checks["normals_" + field] = arrays(getattr(a, field), getattr(b, field))
        checks["normal_statistics"] = {"passed": a.lPl == b.lPl and a.obs_count == b.obs_count}
        checks["serialization_state"] = arrays(vector.values, b.x0)
        reports = [read_report(p / "reflector_bias_processing_report_igg3.txt", "processingReport") for p in roots]
        checks["scientific_report"] = {"passed": report_data(reports[0]) == report_data(reports[1])}
        report = reports[1]
        sigma0 = report["summary"]["sigma0Post"]
        scale = 1.0 if sigma0 is None else sigma0**2
        checks["covariance_inverse"] = arrays(b.N @ covariance / scale, np.eye(len(vector.values)), atol=1e-8)
        checks["full_rank"] = {"passed": report["summary"]["rank"] == len(vector.values)}
        residual_name = "reflector_bias_postfit_residuals_igg3.txt.gz"
        checks["postfit_residuals"] = records(
            read_observation_results(old / residual_name),
            read_observation_results(new / residual_name),
            excluded={"source"},
        )
        checks["converged"] = {"passed": report["converged"]}
        result["termination_reason"] = report["terminationReason"]
        result["summary"] = report["summary"]
        result["parameter_estimates"] = [
            {"name": str(name), "value_m": float(value), "uncertainty_m": float(uncertainty)}
            for name, value, uncertainty in zip(
                vector.parameter_names, vector.values, vector.uncertainties, strict=True
            )
        ]
        tag = "directRejection" if case["name"] == "estimate_direct" else "igg3"
        historical = legacy_parameters(ROOT / "output" / f"reflector_bias_adjustment_solution_{tag}.txt")
        differences = {
            str(name): float(value - historical[str(name)])
            for name, value in zip(vector.parameter_names, vector.values, strict=True)
        }
        # Archived fits use different screening/processing settings; this is
        # a reported difference, not an equivalence assertion.
        result["historical_solution"] = {"same_configuration": False, "differences_m": differences}
        result["global_residuals"] = report["globalResiduals"]
        result["remaining_correction_m"] = report["finalRemainingCorrection"]
        result["remaining_by_block_m"] = {
            "reflectorPosition": max(
                abs(report["finalRemainingCorrection"][str(name)])
                for name in vector.parameter_names
                if name.parameter_type.startswith("position.")
            ),
            "stationRangeBias": max(
                abs(report["finalRemainingCorrection"][str(name)])
                for name in vector.parameter_names
                if name.parameter_type == "rangeBias"
            ),
        }
        if report["converged"]:
            thresholds = report["adjustmentIterations"][-1]["convergenceThresholdByParametrizationM"]
            checks["remaining_stationarity"] = {
                "passed": all(value <= thresholds[key] for key, value in result["remaining_by_block_m"].items())
            }
        result["joint_iteration_updates"] = [
            {
                "iteration": item["iteration"],
                "updates_m": item["appliedUpdateByBlockM"],
                "converged": item["parameterConverged"],
            }
            for item in report["adjustmentIterations"]
        ]
        if case["name"] == "estimate_igg3_extended":
            initial = json.loads((RUN_ROOT / "estimate_igg3/comparison.json").read_text())
            checks["iteration_budget_preserves_prefix"] = {
                "passed": result["joint_iteration_updates"][:10] == initial["joint_iteration_updates"]
            }
        result["estimate_steps"] = [
            {"name": step["name"], "summary": step["summary"]}
            for step in report["processingSteps"]
            if step["type"] == "estimate"
        ]
    result["passed"] = bool(checks) and all(item["passed"] for item in checks.values())
    current_checks = [v for k, v in checks.items() if not k.startswith("historical_")]
    result["current_regression_passed"] = bool(current_checks) and all(v["passed"] for v in current_checks)
    equivalent_checks = [v for k, v in checks.items() if not k.startswith("historical_") and k != "converged"]
    result["source_equivalence_passed"] = bool(equivalent_checks) and all(v["passed"] for v in equivalent_checks)
    historical_checks = [v for k, v in checks.items() if k.startswith("historical_")]
    result["historical_passed"] = all(v["passed"] for v in historical_checks) if historical_checks else None
    write_json(RUN_ROOT / case["name"] / "comparison.json", result)
    print(f"{case['name']}: {'PASS' if result['passed'] else 'FAIL'}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "compare"))
    parser.add_argument("--family", choices=("orbit", "mpi"))
    parser.add_argument("--task", type=int)
    parser.add_argument(
        "--early",
        action="store_true",
        help="Run the six residual cases with c6cd6f8 before the simplification batches.",
    )
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
        return
    manifest = json.loads(MANIFEST.read_text())
    if args.action == "run":
        if args.early:
            cases = [case for case in manifest["cases"] if case["family"] == "residual"]
            case, revision = cases[args.task], "pre_simplification"
        else:
            cases = [case for case in manifest["cases"] if (case["family"] == "orbit") == (args.family == "orbit")]
            case = cases[args.task // 2]
            revision = ("baseline", "current")[args.task % 2]
        run(case, revision, mpi=args.family == "mpi")
        compare(case)
    else:
        results = [compare(case) for case in manifest["cases"]]
        historical_files = set()
        for case, result in zip(manifest["cases"], results, strict=True):
            if result is None:
                continue
            directory = Path(case["historical"]) if case["family"] == "orbit" else ROOT / "output"
            historical_files.update(
                directory / key.removeprefix("historical_") for key in result["checks"] if key.startswith("historical_")
            )
            if case["family"] == "estimation":
                tag = "directRejection" if case["name"] == "estimate_direct" else "igg3"
                historical_files.add(directory / f"reflector_bias_adjustment_solution_{tag}.txt")
        manifest["historical_sha256"] = {str(p): sha256_file(p) for p in sorted(historical_files)}
        write_json(MANIFEST, manifest)
        runs_complete = all(
            (Path(run["output"]) / "run.json").exists() for case in manifest["cases"] for run in case["runs"].values()
        )
        write_json(
            RUN_ROOT / "summary.json",
            {
                "cases": results,
                "complete": runs_complete and all(r is not None for r in results),
                "passed": all(r is not None and r["passed"] for r in results),
                "current_regression_passed": runs_complete
                and all(r is not None and r["current_regression_passed"] for r in results),
                "source_equivalence_passed": runs_complete
                and all(r is not None and r["source_equivalence_passed"] for r in results),
            },
        )


if __name__ == "__main__":
    main()
