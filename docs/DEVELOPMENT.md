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

## Verification

Tests cover published source vectors, the frozen pre-removal differential grid,
leap and validity boundaries, signs and frames, end-to-end light-time effects,
MPI imports, and distribution contents. `scripts/verify_distribution.py`
requires the Cython sources and rejects Fortran/f2py files.

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
