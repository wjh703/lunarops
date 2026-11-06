# Installation

## Requirements

- Python 3.14
- A C compiler (the package builds two Cython extensions)
- [uv](https://docs.astral.sh/uv/) to manage the environment
- Optional: an MPI implementation (Open MPI or MPICH) for parallel runs

The runtime dependencies are NumPy, [pyERFA](https://github.com/liberfa/pyerfa),
[CALCEPH](https://www.imcce.fr/inpop/calceph) (`calcephpy`) and PyYAML.
No Fortran compiler is needed.

## Development install

Clone the repository, then create the environment and build the extensions in
place:

```bash
uv sync --extra build --extra test --extra dynamics --extra mpi
uv pip install --no-build-isolation --editable .
```

The extras are:

| Extra | Adds | Needed for |
|---|---|---|
| `build` | Cython, setuptools | building the extensions |
| `test` | pytest, pytest-cov | running the test suite |
| `dynamics` | SciPy | `LunarOrbitPropagation` (DOP853 start-up, Kuiper-belt ring orbits) |
| `mpi` | mpi4py | `--mpi` runs |
| `plot` | Matplotlib | plotting results outside the package |

The editable install uses `--no-build-isolation` so that the Cython, NumPy and
setuptools of the environment compile the extensions. Rebuild after changing a
`.pyx` or `.pxi` file by running the same `uv pip install` command again.

## OpenMP build

The dynamics extension can use OpenMP to evaluate the point-mass interactions
of many bodies in parallel. The default build is serial. On Linux, enable
OpenMP at build time:

```bash
LUNAROPS_ENABLE_OPENMP=1 uv pip install --no-build-isolation --editable .
```

Check the result with:

```bash
python -c "from lunarops._dynamics_core import openmp_enabled; print(openmp_enabled())"
```

Both builds use the same summation order, so they give the same results. See
[Parallel execution](../user-guide/parallel-execution.md) for thread settings.

## Building distributions

```bash
uv run -m build
uv run scripts/verify_distribution.py dist/*.tar.gz dist/*.whl
```

`verify_distribution.py` checks that the source distribution and wheels
contain the Cython sources.

## Checking the installation

```bash
python -m lunarops list-programs
uv run -m pytest tests -q
```

Some tests need external scientific data (marked `requires_data`) or an MPI
launcher (marked `mpi`); they are skipped when these are absent. See
[Testing](../developer/testing.md).
