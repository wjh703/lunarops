# Building

LunarOps is a setuptools project with two Cython extensions:

| Extension | Source | Content |
|---|---|---|
| `lunarops._iers2010_core` | `_iers2010_core.pyx`, `_iers2010_tables.pxi` | IERS Conventions routines: FCUL troposphere, ocean-tide and libration EOP corrections, solid Earth tides, HARDISP ocean-loading synthesis |
| `lunarops._dynamics_core` | `_dynamics_core.pyx`, `_lunar_rhs.pxi`, `_double_double.pxi` | fused lunar dynamics right-hand side, body-pair point-mass loop, gravity-field evaluation, double-double Adams accumulation |

`lunarops/_iers2010.py` is the Python interface to the IERS routines: it
validates inputs and delegates UTC/TAI/TT conversion, leap seconds and the
IAU 2003 fundamental arguments to ERFA. Typed stubs (`.pyi`) describe both
extensions for type checkers and the API reference.

## Development build

```bash
uv sync --extra build --extra test --extra mpi --extra dynamics
uv pip install --no-build-isolation --editable .
```

The non-isolated editable install compiles with the environment's Cython,
NumPy and setuptools. Run it again after changing a `.pyx` or `.pxi` file.
`setup.py` only declares the extensions; use the PEP 517 commands, not
`python setup.py install`.

## Compiler flags

The release build compiles with `-ffp-contract=off` (`/fp:strict` on MSVC) so
that multiply–add rounding matches the frozen scientific references in the test
suite. The extensions are single-threaded; parallelism comes from MPI worker
processes, not from threads inside the compiled code.

## Distributions

```bash
uv run -m build
uv run scripts/verify_distribution.py dist/*.tar.gz dist/*.whl
```

`verify_distribution.py` checks that the source distribution and the wheels
contain the Cython sources (`.pyx`, `.pxi`) and the stubs.

## Static checks

```bash
uv run ruff check lunarops tests scripts
uv run --group typecheck mypy
```

Ruff also enforces NumPy-style docstrings (`pydocstyle` rules) on the package;
see [Documentation](docs.md).

## Scope of the compiled IERS layer

The compiled layer covers the routines whose cost or reference behaviour
matters: FCUL zenith delay and mapping (pressure and water-vapour pressure in
hPa), ocean-tide and libration EOP variations, the DEHANT solid-tide
displacement and HARDISP. Production HARDISP calls use one irregular epoch at
a time (`n = 1`); a regular HARDISP series cannot cross a UTC offset
transition. The solid Earth pole tide, the ocean pole-tide grid interpolation,
BLQ parsing, coordinate conversion and the ERFA frame matrices are Python.
Atmospheric loading is not modelled.

## Source notice for the IERS routines

The routines and coefficient tables in `_iers2010_core.pyx` and
`_iers2010_tables.pxi` implement the algorithms of the IERS Conventions (2010)
software collection, version 1.3.0, released by the IERS Conventions Centre.
They are a separate implementation written for LunarOps, not the original IERS
software: routine names carry the `lunarops_` prefix and the code has been
restructured for Cython. The IERS Conventions Centre is not responsible for
this implementation. The source archive used as the reference had a size of
63 646 900 bytes and SHA-256
`5f6215b74d22cf53c5f8c40804db091f5ea2cafdaa5e131b8a9ca87c0fb43ea1`.
