# LunarOps documentation

LunarOps is a configuration-driven Lunar Laser Ranging processor with a
GROOPS-inspired split between files, classes, parametrizations, programs, and
estimation.

## Start here

| Task | Document |
|---|---|
| Run a program or choose an output | [PROGRAMS.md](PROGRAMS.md) |
| Prepare MINI, CRD, or canonical LunarOps inputs | [INPUTS.md](INPUTS.md) |
| Configure reflector and station-bias adjustment | [ADJUSTMENT.md](ADJUSTMENT.md) |
| Understand module boundaries and extension points | [ARCHITECTURE.md](ARCHITECTURE.md) |
| Review the GROOPS-inspired file and program contract | [GROOPS_FILE_PROGRAM_DESIGN.md](GROOPS_FILE_PROGRAM_DESIGN.md) |
| Build and validate the IERS Cython extension | [DEVELOPMENT.md](DEVELOPMENT.md) |
| Review the completed ERFA/Cython migration | [IERS_CYTHON_MIGRATION.md](IERS_CYTHON_MIGRATION.md) |

## Minimal command line

```bash
python -m lunarops list-programs
python -m lunarops describe-program LlrResiduals
python -m lunarops describe-config
python -m lunarops validate configs/lunarops_oc_residuals.yml
python -m lunarops list-classes
python -m lunarops run configs/lunarops_oc_residuals.yml
python -m lunarops run configs/lunarops_oc_residuals.yml --mpi
```

`--set name=value` overrides a value in the config `variables` section. Paths
are resolved relative to the config working directory unless they are absolute.

## Program and file model

External MINI/CRD data enters through `NormalPointsConvert`. Model and
estimation programs consume typed text artifacts; dense matrix payloads use
typed binary files. Each task declares its input/output slots and accepted
configuration keys in the program registry.

## Current conventions

- Runtime epochs are explicit two-part `Epoch` values with `UTC`, `TT`, or
  `TDB` scale.
- Normal-point uncertainty comes from the input record; it is not replaced by
  a station lookup table.
- Range-bias corrections in `globals.rangeBias` are deterministic forward
  corrections. Estimated `stationRangeBias` parameters are separate.
- The production Earth-orientation path uses explicit IERS C04 data, ERFA,
  and the private `lunarops._iers2010` extension.
