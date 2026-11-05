# Development and IERS backend

## Build

The project uses setuptools and Cython to build the private extensions
`lunarops._iers2010_core` and `lunarops._dynamics_core`. The IERS Python
facade is `lunarops._iers2010`; the single production dynamics RHS is compiled
by `FusedRhs`. Normal-equation assembly and solving use NumPy. No Fortran
compiler or f2py step is required.

```bash
uv sync --extra build --extra test --extra mpi --extra dynamics
uv pip install --no-build-isolation --editable .
uv run -m pytest tests -q
uv run ruff check lunarops tests scripts
uv run --group typecheck mypy
uv run -m build
uv run scripts/verify_distribution.py dist/*.tar.gz dist/*.whl
```

The non-isolated editable install keeps Cython, NumPy, and setuptools available
in the persistent development environment. Standard distribution builds use
the isolated requirements declared in `pyproject.toml`. `setup.py` only
describes the Cython extensions; use the PEP 517 commands above instead of
running `python setup.py install` directly.

## Backend boundary

`lunarops/_iers2010.py` validates inputs and delegates UTC/TAI/TT conversion,
leap seconds, and IAU 2003 fundamental arguments to ERFA. The compiled Cython
core implements FCUL, orthotide EOP, libration corrections, DEHANT solid-Earth
tides, and HARDISP ocean-loading synthesis.

Columnar `EquationBatch` data becomes a `DenseLinearization`. Parametrizations
write design blocks into A, and weighted NumPy products form N, W and lPl.
`NormalEquations` caches its symmetric matrix, spectrum, Cholesky factor and
covariance after assembly. Numerical-rank rejection remains strict; scipy is
only a dynamics extra. Python owns the ABM loop and accepted history; the
compiled dynamics core supplies the physical RHS and its diagnostics.
MPI startup fixes OpenBLAS, OpenMP, and MKL to one native thread per rank before
loading NumPy or model modules. Rank 0 solves the current small normal systems
with the same single-thread policy; large distributed solvers are out of scope.

The application layer continues to own C04 interpolation, BLQ parsing,
coordinate conversion, units, and model composition. Solid-Earth pole tide,
ocean pole-tide grid interpolation, and ERFA frame matrices remain Python
implementations by design.

## Conventions that must not drift

- C04 is the observed Earth-orientation series. `RG_ZONT2` is not applied, and
  observed `dX/dY` is not augmented with `FCNNUT`.
- Native boundaries document time scale, frame, units, and signs explicitly.
- `FCUL_ZD_HPA` receives pressure and water-vapour pressure in hPa.
- Production HARDISP calls use irregular scalar light-time epochs (`n=1`).
- A regular HARDISP series cannot cross a UTC offset transition.
- Atmospheric loading is outside the current backend scope.

## Source and licensing

The algorithms and tables are derived from IERS Conventions v1.3.0. The pinned
upstream archive had size 63,646,900 bytes and SHA-256
`5f6215b74d22cf53c5f8c40804db091f5ea2cafdaa5e131b8a9ca87c0fb43ea1`.
The former source tree is not retained. Derived routine names use the
`lunarops_` prefix. Wheels include the `.pyx` and `.pxi` sources.

## Testing

`tests/` mirrors the library: `time`, `iers`, `frames`, `dynamics`, `observation`,
`estimation`, `fileio`, `config`, `parallel` and `packaging`. Shared synthetic
inputs live in `tests/support/` (static or analytic ephemerides, the synthetic
observation pipeline, BLQ/CRD/MINI samples and the NumPy dynamics oracle); test
modules never import each other.

- **References.** Each physical test names its reference: IERS source-header
  examples, the frozen IERS Fortran grid (`tests/data/iers_fortran_baseline.npz`),
  ERFA validation vectors, Orekit, analytic formulas, or an independent DOP853 or
  method-of-steps integration. Tests that compare two LunarOps paths say so in
  their name. Analytic partials are checked against central differences of the
  full synthetic observation model.
- **Floating-point comparisons.** Values computed through libm, ERFA, BLAS or the
  compiled extensions are never compared bit for bit. They use the named
  tolerances of `tests/tolerances.py` (`TIME_S`, `LIGHT_TIME_S`,
  `DISPLACEMENT_M`, `HARDISP_M`, `ANGLE_RAD`, `EOP_UAS`, `REL`, `DYNAMICS_REL`,
  `ESTIMATE_REL`, `COVARIANCE_REL`). A model or discretization limit of one
  comparison (finite differences, step convergence, an external code) is declared
  next to that test as a named `Tolerance` with its reason. Exact comparison is
  kept for integers, strings, booleans, values parsed from text, file and pickle
  round trips, and repeated evaluation of the same code on the same inputs.
- **Public behaviour only.** Tests do not patch private functions, count calls or
  cache hits, or read third-party internals. Fakes enter through public injection
  points (ephemeris, Earth orientation, MPI runtime, a program's factory).
- **Markers.** `mpi` tests start `mpirun` and are skipped without an MPI launcher
  and mpi4py; `requires_data` tests need data outside the repository; `slow`
  marks integrations of more than about half a second.
- **Platform check.** The tolerances are calibrated against an extension build with
  `-ffp-contract=fast -O3 -march=native`, which differs from the release flags
  (`-ffp-contract=off`, `/fp:strict` on MSVC) by up to 1.4e-9 m in HARDISP, whose
  reference code keeps float32 intermediates. CI runs the suite on `ubuntu-22.04`
  and `ubuntu-latest`; the latter also runs the `mpi` tests with Open MPI.

`scripts/verify_distribution.py` requires the Cython sources and rejects
Fortran/f2py files.

## TDB-TT

Production TT/TDB conversion uses ERFA's `dtdb`, with no ephemeris dependency.
The station-dependent term is
`dtdb(TDB, UT1, longitude, u, v) - dtdb(TDB, 0, 0, 0, 0)`.

## Output regression

`scripts/compare_outputs.py BASELINE_DIR NEW_DIR` compares typed scientific
artifacts with explicit `--rtol/--atol` tolerances and provenance exclusions.
Use the same frozen scenario on both sides. Refactoring acceptance separates
Phase 0 orbit/integrated rotation, accepted native diagnostics and the Phase 4
numeric observation/estimation references; see
[the validation protocol](refactoring/ELEGANCE_PLAN.md) and
[recorded results](refactoring/PROGRESS.md). Numerical tolerances never replace
strict rank rejection or convergence criteria.
