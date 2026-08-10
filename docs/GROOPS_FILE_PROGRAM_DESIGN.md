# GROOPS-aligned file and program design

## Status and scope

This document is the implemented LunarOps data and program contract. It is not a
migration plan. The refactor intentionally has no compatibility layer: old
program names, old configuration keys, and the former JSON Lines, CSV, JSON,
NPZ, and sidecar writers/readers are not part of LunarOps.

YAML remains the configuration language. YAML scenario files select programs,
classes, variables, and paths; they are not scientific data artifacts.

CRD and MINI remain supported as external source formats, but only through the
explicit `NormalPointsConvert` import boundary. Every downstream program reads
native typed artifacts.

## Design rules

1. Every reusable scientific boundary has a declared artifact type.
2. ASCII is the default encoding. It uses `.txt` or `.txt.gz` and can be
   inspected without a project-specific tool.
3. Dense numerical payloads may use `.dat` or `.dat.gz`. Their binary header
   records dtype, shape, and matrix kind.
4. Compression changes storage only, never logical schema.
5. The first non-comment line of every text artifact is:

   ```text
   lunarops <artifactType> version=20260728
   ```

6. Floating-point values use 17 significant decimal digits and readers reject
   non-finite values.
7. Time values carry an explicit scale; normal-point and observation-equation
   epochs are two-part UTC Julian dates.
8. Parameter-bearing products always carry structured `ParameterName` values
   and units. Column position alone is never parameter identity.
9. Multi-file products are atomically published directory groups with payload
   checksums.
10. A program has explicit typed input and output slots. There is no generic
    untyped `outputFile` key.

## Native artifact catalog

| Logical type | Canonical form | Main producer | Main consumer |
|---|---|---|---|
| `NormalPointFile` | `normalPoints.txt[.gz]` | conversion/filter/concatenate | LLR model programs |
| `ObservationResultFile` | `observationResults.txt[.gz]` | `LlrResiduals` | statistics/QC |
| `StationCatalogFile` | `stations.txt[.gz]` | `CatalogCreate` | observation model |
| `ReflectorCatalogFile` | `reflectors.txt[.gz]` | catalog/apply solution | observation model |
| `ParameterVectorFile` | `solution.txt[.gz]` | solve/adjustment | apply solution |
| `MatrixFile` | `.txt[.gz]` or `.dat[.gz]` | matrix tools | matrix tools |
| `NormalEquationFile` | directory group | equation programs | accumulate/solve |
| `ObservationEquationFile` | directory group | `LlrObservationEquations` | equation accumulation |
| `CovarianceMatrixFile` | directory group | solve/adjustment | analysis |
| `AdjustmentReportFile` | `report.txt[.gz]` | `LlrAdjustment` | audit/human inspection |
| `AdjustmentStateFile` | `state.txt[.gz]` | `LlrAdjustment` | adjustment resume |
| `NormalPointStatisticsFile` | `normalPointStatistics.txt[.gz]` | normal-point statistics | audit/QC |
| `ObservationResultStatisticsFile` | `observationResultStatistics.txt[.gz]` | result statistics | audit/QC |
| `ModelStateFile` | `modelState.txt[.gz]` | apply solution | model setup/audit |
| `ImportReportFile` | `importReport.txt[.gz]` | normal-point conversion | audit/QC |

## Scalar and row encoding

Whitespace separates fields. Comments begin with `#`. Text tokens use percent
encoding, so names may contain spaces without quoting ambiguity. `~` denotes a
missing optional value. Counts are declared before row payloads and verified by
readers.

### NormalPointFile

The header declares dataset name, UTC scale, record count, original input
count, and invalid source-record count. Each row contains:

```text
jd1_utc jd2_utc station reflector rtt_s uncertainty_two_way_s
pressure_hPa temperature_K humidity_percent wavelength_nm index
station_code reflector_code
```

All mandatory physical values are positive and finite; humidity is in
`[0, 100]`. Station and reflector identities are compact tokens without
whitespace (`Apollo11`, not `Apollo 11`). Record uncertainty is the sole formal
observation uncertainty.

### ObservationResultFile

The file carries its own table schema. Each field declaration contains name,
scalar type (`bool`, `int`, `float`, or `text`), and unit. Rows from multiple
inputs include a `source` column. This supports both compact and full residual
tables without an untyped map per line.

### Catalog files

Station rows carry key, name, ITRF position, ITRF velocity, position epoch, and
aliases. Reflector rows carry key, name, Moon PA-frame position, and aliases.
The file header fixes the frame (`ITRF` or `MOON_PA`).

### ParameterVectorFile

Each row contains structured parameter name, unit, value, and optional
uncertainty. `hasUncertainty` declares whether the fourth column is populated,
and `uncertaintySigmaMultiplier` declares its sigma multiplier. Solutions
produced by `NormalsSolve` and `LlrAdjustment` publish parameter uncertainty at
3 sigma. `vectorKind` is either `correction` or `estimate`; consumers must
respect that semantic when applying values. Covariance artifacts retain their
standard cofactor or posterior 1-sigma-squared meaning and are not multiplied
by nine.

### Structured reports and state

Reports, statistics, and restart state use a constrained YAML scalar/container
grammar after their LunarOps type header. This keeps nested diagnostics readable
while retaining a declared outer artifact type. Non-finite scalars and opaque
Python objects are rejected.

## Matrix encoding

`MatrixFile` supports `dense`, `vector`, and `lowerSymmetric` matrix kinds with
`float64` or `int64` values. Symmetric matrices store one triangle. Binary files
start with the `LLRMTX01` magic and a fixed-width little-endian header carrying
version, dtype, kind, dimensions, and payload count.

`MatrixConvert` changes only ASCII/binary encoding. It preserves matrix kind,
shape, dtype, and numerical values.

## File groups

### NormalEquationFile

```text
normalEquations/
  info.txt
  normalMatrix.dat.gz
  rightHandSide.dat.gz
  parameterNames.txt
```

`info.txt` carries observation count, `lPl`, parameter count, payload names,
SHA-256 checksums, and scientific metadata. Addition aligns systems by
structured parameter name and rejects inconsistent units or compatibility
metadata.

### ObservationEquationFile

```text
observationEquations/
  info.txt
  metadata.txt
  parameterNames.txt
  observations.txt
  rowPointers.dat.gz
  columnIndices.dat.gz
  designValues.dat.gz
  observationVector.dat.gz
  sigmas.dat.gz
```

The design matrix uses CSR payloads. Observation rows retain integer identity,
source, UTC epoch, station, reflector, convergence flag, and wavelength. All
payloads are checksummed. Accumulating this file must produce the same normal
equations as direct accumulation at the same linearization.

### CovarianceMatrixFile

```text
covariance/
  info.txt
  covariance.dat.gz
  parameterNames.txt
```

The covariance kind distinguishes a cofactor matrix from posterior covariance.
Names and units are inseparable from the symmetric matrix.

## Declarative program contract

Registration stores a `ProgramSpec`, not only a callable:

```python
ProgramSpec(
    name="LlrNormalEquations",
    summary="Build normal equations at one fixed LLR linearization.",
    inputs=(
        ArtifactSlot("inputFilesNormalPoints", "NormalPointFile", many=True),
    ),
    outputs=(
        ArtifactSlot("outputFileNormalEquations", "NormalEquationFile"),
    ),
    fields=(
        FieldSpec(name="combineInputs", kind="boolean", default=False),
        FieldSpec(name="mpi", kind="mapping", nested=mpi_schema),
    ),
)
```

`ArtifactSlot` describes file products; `fields` describes every non-artifact
option. There is no second `required_keys`/`optional_keys` declaration. The
single generated `ConfigSchema` applies defaults and rejects unknown keys
before execution. The registry then rejects absent required slots, wrong path
cardinality, wrong filename encoding, nonexistent inputs, and an artifact
header that does not match the declared slot. Static validation tracks outputs
produced earlier in a scenario, so a complete not-yet-run graph can be
validated.

```bash
python -m lunarops list-programs
python -m lunarops describe-program LlrNormalEquations
python -m lunarops validate config.yml
```

## Program graph

```text
CRD / MINI / native source
          |
          v
 NormalPointsConvert
          |
          v
 NormalPointFile <--- NormalPointsConcatenate / NormalPointsFilter
       |     |
       |     +--> LlrResiduals --> ObservationResultFile
       |                              |
       |                              +--> ObservationResultsStatistics
       |
       +--> LlrNormalEquations ------------------------+
       |                                               |
       +--> LlrObservationEquations                    v
                 |                          NormalEquationFile
                 v                                     |
        ObservationEquationFile                       |
                 |                                     |
                 +--> ObservationEquationsToNormals ---+
                                                       |
                                    NormalsAccumulate -+
                                                       |
                                                       v
                                                 NormalsSolve
                                               /       |       \
                                      solution   covariance   report

NormalPointFile + models + parametrization --> LlrAdjustment
                                                  |  |  |  |  |
                                             report state solution covariance normals

ParameterVector + ReflectorCatalog --> LlrApplySolution --> catalog + model state
```

## Registered programs

| Program | Required typed inputs | Required outputs |
|---|---|---|
| `NormalPointsConvert` | external normal-point sources | normal points and import report |
| `NormalPointsConcatenate` | normal-point files | normal points |
| `NormalPointsFilter` | normal points | filtered normal points |
| `NormalPointsStatistics` | normal-point files | statistics |
| `LlrResiduals` | normal-point files | observation results |
| `LlrNormalEquations` | normal-point files | normal equations |
| `LlrObservationEquations` | normal-point files | frozen equations |
| `ObservationEquationsToNormals` | frozen equations | normal equations |
| `NormalsAccumulate` | normal-equation files | normal equations |
| `NormalsSolve` | normal equations | solution, covariance, report |
| `LlrAdjustment` | normal-point files, optional state | report, state, solution, covariance, normals |
| `CatalogCreate` | none | station and reflector catalogs |
| `LlrApplySolution` | solution and reflector catalog | reflector catalog and model state |
| `ObservationResultsStatistics` | observation results | statistics |
| `MatrixConvert` | matrix | matrix in the other selected encoding |

## Adjustment restart contract

Adjustment state contains current reflector positions, parametrization state,
variance-component scales, robust factors, last stage, convergence status, and
a SHA-256 scientific fingerprint. The fingerprint covers the resolved program
settings, global model settings, and referenced input/model file contents.
Resume is rejected when the fingerprint differs.

The report is audit output and is never used as restart state. The solution is
a typed absolute estimate; covariance and final weighted normal equations are
published independently.

## Validation invariants

- Readers reject unknown archive versions and wrong artifact types.
- Counts, dimensions, CSR structure, checksums, units, time scales, and frames
  are validated before returning an object.
- Duplicate parameter names and catalog keys are rejected.
- Normal matrices and covariance matrices must be finite and symmetric.
- Normal-equation addition is name-aligned and unit-aware.
- Artifact publication is atomic for individual files and directory groups.
- A program cannot read its own output as an input for destructive conversion
  or accumulation operations.

## Explicit exclusions

LunarOps does not reproduce the full GROOPS format catalog. Satellite orbit,
gravity-field, GNSS receiver/transmitter, and generic platform formats should
be added only when an LunarOps scientific program owns their semantics. External
presentation export belongs in a separately named adapter, not in a canonical
producer.
