"""Outer loop of the LunarOps lunar ephemeris: integrate, fit, and start the next integration from the fit.

    python scripts/lunar_ephemeris_loop.py RUNDIR [--parameters FILE] [--first N] [--iterations N]
        [--stop-sigma S] [--set NAME=VALUE ...] [--integration-workers N] [--fit-workers N] [--update-configs]

Iteration n integrates configs/lunarops_lunar_ephemeris_integration.yml into RUNDIR/iter<n> from the parameters
of fit n-1 (iteration --first: --parameters, default configs/lunar_dynamics_parameters_j2000.txt), then fits
configs/lunarops_lunar_ephemeris_fit.yml to that orbit into RUNDIR/fit<n>. The fit corrects the orbit through the
variational partials, a linearization about the integrated orbit, so the loop repeats until every dynamical
correction is below --stop-sigma formal (1-sigma) errors. The other parameters enter the range linearly and are
re-estimated in full by every fit. An existing iteration directory with complete products is reused, so an
interrupted loop resumes. --set NAME=VALUE is passed to both runs (a variable of base/dynamics.yml such as
earthPoleModel applies to the integration only).

With --update-configs the last fit becomes the a priori of the configs: its parameters file is written to
configs/lunar_dynamics_parameters_j2000.txt, and its reflector and station coordinates, libration amplitudes and
h2 replace the block between the "a priori" markers of configs/base/llr_fit.yml.

Run from the repository root (relative paths of the configs resolve against it).
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
INTEGRATION = ROOT / "configs/lunarops_lunar_ephemeris_integration.yml"
FIT = ROOT / "configs/lunarops_lunar_ephemeris_fit.yml"
FIT_BASE = ROOT / "configs/base/llr_fit.yml"
PARAMETERS = ROOT / "configs/lunar_dynamics_parameters_j2000.txt"
INTEGRATION_ONLY = {"earthPoleModel", "stepSeconds", "startupStepSeconds", "correctorIterations"}


def read_table(path: Path) -> tuple[dict, list[list[str]]]:
    """Header fields and data rows of a LunarOps table file."""
    lines = path.read_text().splitlines()
    header = {}
    for line in lines[: lines.index("data")]:
        if line.startswith("header "):
            header = yaml.safe_load(line[len("header ") :])
    return header, [line.split() for line in lines[lines.index("data") + 1 :] if line]


def dynamics_corrections(fit: Path) -> list[tuple[float, str]]:
    """|correction| / sigma of every dynamical parameter of a fit, largest first."""
    header, rows = read_table(fit / "reflector_bias_solution_igg3.txt")
    multiplier = float(header.get("uncertaintySigmaMultiplier", 1.0))
    ratios = []
    for name, _unit, value, uncertainty in rows:
        if ":dynamics." in name:
            ratios.append(
                (abs(float(value)) / (float(uncertainty) / multiplier), name.split(":dynamics.")[1].rstrip(":"))
            )
    return sorted(ratios, reverse=True)


def run(config: Path, log: Path, workers: int, settings: list[str]) -> None:
    command = [sys.executable, "-m", "lunarops", "run", str(config), "--working-dir", str(ROOT)]
    command += ["--workers", str(workers)] if workers > 1 else []
    for setting in settings:
        command += ["--set", setting]
    with log.open("w") as stream:
        subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)


def a_priori_block(fit: Path) -> str:
    """The "a priori" variables of base/llr_fit.yml from the products of a fit."""
    state = yaml.safe_load((fit / "reflector_bias_processing_state_igg3.txt").read_text().split("\n", 1)[1])
    blocks = state["parametrization"]
    h2 = float(blocks["lunarLoveNumber"]["h2"]["updated"])
    lines = [
        f"  # BEGIN a priori (scripts/lunar_ephemeris_loop.py --update-configs) from {fit.parent.name}/{fit.name}",
        "  # Lunar radial Love number h2.",
        f"  moonH2: {h2:.6f}",
        "  # Reflector PA coordinates (m).",
        "  reflectorCoordinates:",
    ]
    for key, *xyz in read_table(fit / "reflector_catalog_estimated_igg3.txt")[1]:
        lines.append(f"    - {{key: {key}, xyzM: [{', '.join(f'{float(v):.4f}' for v in xyz)}]}}")
    lines += ["  # Station ITRF positions (m) at positionEpochUtc and velocities (m/yr).", "  stationCoordinates:"]
    for key, x, y, z, vx, vy, vz, epoch in read_table(fit / "station_catalog_estimated_igg3.txt")[1]:
        position = ", ".join(f"{float(v):.4f}" for v in (x, y, z))
        velocity = ", ".join(f"{float(v):.4f}" for v in (vx, vy, vz))
        lines.append(
            f'    - {{key: {key}, xyzM: [{position}], velocityMPerYear: [{velocity}], positionEpochUtc: "{epoch}"}}'
        )
    terms = yaml.safe_load(FIT_BASE.read_text())["variables"]["librationTerms"]
    lines += ["  # Empirical longitude-libration terms (mas).", "  librationTerms:"]
    for term in terms:
        saved = blocks["empiricalLibration"][term["name"]]
        delaunay = ", ".join(f"{k}: {v}" for k, v in term["delaunay"].items())
        lines.append(
            f"    - {{name: {term['name']}, axis: {term['axis']}, delaunay: {{{delaunay}}}, "
            f"cosMas: {float(saved['cosMas']):.4f}, sinMas: {float(saved['sinMas']):.4f}, "
            f"estimate: {str(term.get('estimate', True)).lower()}, "
            f"estimateSin: {str(term.get('estimateSin', False)).lower()}}}"
        )
    lines.append("  # END a priori")
    return "\n".join(lines)


def update_configs(fit: Path) -> None:
    text = FIT_BASE.read_text()
    pattern = re.compile(r"  # BEGIN a priori.*?  # END a priori", re.DOTALL)
    if not pattern.search(text):
        raise SystemExit(f"{FIT_BASE}: no a priori markers")
    FIT_BASE.write_text(pattern.sub(lambda _: a_priori_block(fit), text, count=1))
    lines = (fit / "lunar_dynamics_parameters_updated.txt").read_text().split("\n", 1)
    document = yaml.safe_load(lines[1])
    document["name"] = f"LunarOps lunar ephemeris: converged solution {fit.parent.name}/{fit.name}"
    PARAMETERS.write_text(lines[0] + "\n" + yaml.safe_dump(document, sort_keys=False, width=200))
    print(f"updated {FIT_BASE.relative_to(ROOT)} and {PARAMETERS.relative_to(ROOT)} from {fit}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("rundir", type=Path)
    p.add_argument("--parameters", type=Path, default=PARAMETERS)
    p.add_argument("--first", type=int, default=0)
    p.add_argument("--iterations", type=int, default=6)
    p.add_argument("--stop-sigma", type=float, default=0.5)
    p.add_argument("--set", action="append", default=[], metavar="NAME=VALUE")
    p.add_argument("--integration-workers", type=int, default=4)
    p.add_argument("--fit-workers", type=int, default=28)
    p.add_argument("--update-configs", action="store_true")
    a = p.parse_args()
    rundir = a.rundir.resolve()
    rundir.mkdir(parents=True, exist_ok=True)
    fit_settings = [s for s in a.set if s.split("=", 1)[0] not in INTEGRATION_ONLY]
    parameters = a.parameters.resolve()
    fit = None
    for n in range(a.first, a.first + a.iterations):
        orbit, fit = rundir / f"iter{n}", rundir / f"fit{n}"
        if not (orbit / "lunar_dynamics_partials.dat").exists():
            orbit.mkdir(exist_ok=True)
            print(f"iter{n}: integration from {parameters}", flush=True)
            run(
                INTEGRATION,
                rundir / f"iter{n}.log",
                a.integration_workers,
                [f"outputDir={orbit}", f"parametersFile={parameters}", *a.set],
            )
        if not (fit / "lunar_dynamics_parameters_updated.txt").exists():
            fit.mkdir(exist_ok=True)
            print(f"fit{n}", flush=True)
            run(
                FIT,
                rundir / f"fit{n}.log",
                a.fit_workers,
                [f"outputDir={fit}", f"orbitDir={orbit}", f"parametersFile={parameters}", *fit_settings],
            )
        ratios = dynamics_corrections(fit)
        print(
            f"fit{n}: largest dynamical correction {ratios[0][0]:.2f} sigma ({ratios[0][1]}); "
            + ", ".join(f"{name} {ratio:.2f}" for ratio, name in ratios[1:6]),
            flush=True,
        )
        parameters = fit / "lunar_dynamics_parameters_updated.txt"
        if ratios[0][0] < a.stop_sigma:
            print(f"converged at fit{n}", flush=True)
            break
    else:
        print(f"not converged after {a.iterations} iterations", flush=True)
    if a.update_configs and fit is not None:
        update_configs(fit)


if __name__ == "__main__":
    main()
