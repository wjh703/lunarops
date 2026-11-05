# Testing

## Running the tests

```bash
uv run -m pytest tests -q
uv run -m pytest tests -m "not slow" -q        # skip long integrations
uv run -m pytest tests/observation -q          # one area
```

| Marker | Meaning |
|---|---|
| `mpi` | starts `mpirun`; skipped without an MPI launcher and mpi4py |
| `requires_data` | needs scientific data outside the repository; skipped when absent |
| `slow` | integrates long arcs or many steps (more than about half a second) |

## Layout

`tests/` mirrors the package: `time`, `iers`, `frames`, `dynamics`,
`observation`, `estimation`, `fileio`, `config`, `parallel` and `packaging`.
Shared synthetic inputs live in `tests/support/`: static and analytic
ephemerides, a synthetic observation pipeline, BLQ, CRD and MINI samples, and
the independent NumPy implementation of the lunar dynamics used as an oracle.
Test modules never import each other.

## Principles

**Every physical test names its reference.** References are IERS source-code
examples, the frozen grid of the IERS Fortran routines
(`tests/data/iers_fortran_baseline.npz`), ERFA validation values, Orekit,
analytic formulas, or an independent DOP853 or method-of-steps integration.
A test that compares two LunarOps code paths says so in its name. Analytic
partial derivatives are checked against central differences of the full
synthetic observation model.

**Floating-point comparisons use named tolerances.** Values that pass through
libm, ERFA, BLAS or the compiled extensions are never compared bit for bit.
They use the tolerances of `tests/tolerances.py` (`TIME_S`, `LIGHT_TIME_S`,
`DISPLACEMENT_M`, `HARDISP_M`, `ANGLE_RAD`, `EOP_UAS`, `REL`, `DYNAMICS_REL`,
`ESTIMATE_REL`, `COVARIANCE_REL`). When a comparison is limited by a model or
discretization (finite differences, step-size convergence, an external code),
the test declares a named `Tolerance` with its reason next to it. Exact
comparison is used for integers, strings, booleans, values parsed from text,
file and pickle round trips, and repeated evaluation of the same code on the
same inputs.

**Tests use public behaviour only.** They do not patch private functions,
count calls or cache hits, or read third-party internals. Fakes enter through
the public injection points: ephemeris, Earth orientation, MPI runtime and a
program's factory.

## Platform dependence

The tolerances are calibrated against an extension build with
`-ffp-contract=fast -O3 -march=native`. That build differs from the release
flags (`-ffp-contract=off`, `/fp:strict` on MSVC) by up to 1.4·10⁻⁹ m in
HARDISP, whose reference code keeps single-precision intermediates. CI runs the
suite on `ubuntu-22.04` and `ubuntu-latest`; the latter also runs the `mpi`
tests with Open MPI.

## Output regression

Scenario-level regression compares the products of two code versions on the
same frozen inputs:

```bash
python scripts/compare_outputs.py BASELINE_DIR NEW_DIR --rtol 1e-12 --atol 0 [--skip GLOB]
```

`compare_outputs.py` reads both sides with the artifact readers, so binary
tables, matrices and reports are compared as values with the given relative
and absolute tolerances, not as bytes. Provenance fields such as checksums,
paths and timings are excluded; `--skip` leaves out files matching a glob.
`scripts/run_regression_suite.py` prepares, submits (Slurm) and compares a
fixed set of orbit and MPI residual scenarios against a stored baseline
(`prepare`, `run`, `compare`).

Use physical acceptance limits, for example a relative orbit position
difference below 10⁻¹⁰, residual differences below 1 µm and parameter
differences below 10⁻⁶ of their formal sigma. Binary checksums, fingerprints,
timings and the round-off of condition numbers are not physical measures.
Tolerances never replace the rank and convergence criteria of the estimation.

## Static checks

```bash
uv run ruff check lunarops tests scripts
uv run --group typecheck mypy
python scripts/generate_reference_docs.py --check
```
