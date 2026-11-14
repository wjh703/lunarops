# LunarOps

LunarOps is a configuration-driven processing system for Lunar Laser Ranging
(LLR). It covers the whole chain from raw normal points to scientific products:

- **Data preparation**: import of CRD v2 and MINI normal points, IERS C04 and
  Bulletin-A Earth orientation, and station, reflector and mass catalogs.
- **Observation modelling**: two-way light-time solution in the barycentric
  frame with IERS 2010 station displacements, tropospheric and Shapiro delays,
  lunar solid tides and range-bias corrections.
- **Parameter estimation**: nonlinear weighted least squares for reflector,
  station, range-bias and libration parameters, with robust weighting,
  variance-component estimation and restartable processing state.
- **Observation prediction**: pointing geometry and visibility windows for
  observing campaigns.
- **Lunar dynamics**: propagation of the Moon's orbit relative to the Earth,
  optionally coupled with mantle and fluid-core rotation, using a compiled
  right-hand side and a fixed-step Adams–Bashforth–Moulton integrator.

Every run is described by a YAML file that lists programs and their inputs and
outputs. All products are typed, self-describing native files.

## Installation

LunarOps needs Python 3.14, a C compiler and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra build --extra test --extra dynamics --extra mpi
uv pip install --no-build-isolation --editable .
```

## First run

```bash
python -m lunarops list-programs
python -m lunarops describe-program LlrResiduals
python -m lunarops validate configs/lunarops_oc_residuals.yml
python -m lunarops run configs/lunarops_oc_residuals.yml
python -m lunarops run configs/lunarops_oc_residuals.yml --workers 8
mpirun -n 8 python -m lunarops run configs/lunarops_oc_residuals.yml --mpi
```

## Documentation

The full manual lives in [`docs/`](docs/index.md). To build and browse it
locally:

```bash
uv sync --group docs
uv run mkdocs serve
```
