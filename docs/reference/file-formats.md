# File formats

Every file that LunarOps writes is a typed native artifact. Its first header
line names the artifact type, so a reader can identify the content
independently of the file name. Readers check the type and the fields they
use; there is no automatic conversion between versions or formats.

## Encodings

| Suffix | Encoding | Used for |
|---|---|---|
| `.txt`, `.txt.gz` | UTF-8 text | record tables, catalogs, EOP, reports, state |
| `.dat`, `.dat.gz` | binary `LUNARTBL` table | time series and matrices |
| no suffix | directory | normal equations, covariance |

A `.gz` suffix selects gzip compression; the content is the same. JSON, CSV
and NPZ files are not native artifacts.

### Text header

A text artifact starts with

```text
lunarops <artifactType>
```

Blank lines and lines starting with `#` before the header are ignored; in
record data, text after `#` is a comment.

### Record tables

Catalogs, normal points, EOP, parameter names and vectors, observation results
and predictions share one layout:

```text
lunarops stationCatalog
header {"frame": "ITRF"}
fieldCount 8
field key text 1
field x_m float m
field y_m float m
field z_m float m
field vx_m_per_year float m/yr
field vy_m_per_year float m/yr
field vz_m_per_year float m/yr
field epoch text 1
recordCount 2
data
APOLLO -1.46399890790000000e+06 ...
GRASSE 4.58169216860000000e+06 ...
```

- `header` is a JSON mapping with the domain information of the file: frame,
  time scale, data-set counts, import provenance, uncertainty conventions.
- Each `field` line gives the name, the kind (`bool`, `int`, `float`, `text`)
  and the unit (`1` for dimensionless).
- Values are separated by whitespace. Text tokens are percent-encoded, so they
  never contain spaces; an empty value is written `~` and a literal `~` as
  `%7E`. Floats use 17 significant digits in scientific notation unless a
  column declares its own precision (prediction tables do).

### Structured reports

Reports, processing state, orbit metadata and the `info.txt` of
normal-equation directories are a text header followed by a YAML mapping.
Keys are lowerCamelCase.

### Binary tables

`LUNARTBL` files store a float64 matrix with named columns:

| Bytes | Content |
|---|---|
| 8 | magic `LUNARTBL` |
| 4 | header length `h` (uint32, little-endian) |
| 8 | row count `n` (uint64) |
| 4 | column count `m` (uint32) |
| `h` | JSON header: `artifactType`, `columns`, `metadata` |
| `8 n m` | values, little-endian float64, row-major |

All values must be finite. A `matrix` artifact has one column, `value`, and
`metadata.kind` `vector` or `lowerSymmetric`; a symmetric matrix of
`metadata.rows` rows stores its lower triangle in NumPy `tril_indices` order.

## Artifact types

| Artifact type | Encoding | Written by | Content |
|---|---|---|---|
| `normalPoint` | records | `NormalPointsConvert` | normal points with UTC epoch, two-way time and uncertainty, meteorology |
| `normalPointImportReport` | structured | `NormalPointsConvert` | sources, checksums, counts, rejected lines |
| `stationCatalog` | records | `StationCatalogCreate`, `LlrProcessing` | ITRF positions, velocities, epochs |
| `reflectorCatalog` | records | `ReflectorCatalogCreate`, `LlrProcessing` | Moon PA positions |
| `massCatalog` | structured | `MassCatalogCreate` | body IDs, GM, groups, ephemeris targets |
| `earthOrientationParameter` | records | EOP programs | daily EOP with source and quality |
| `observationResult` | records | `LlrResiduals`, `LlrProcessing` | per-normal-point residuals and diagnostics |
| `processingReport` | structured | `LlrProcessing` | adjustment result, diagnostics, publication quality |
| `processingState` | structured | `LlrProcessing` | restart state and scientific fingerprint |
| `parameterName` | records | `LlrProcessing` | parameter names and units |
| `parameterVector` | records | `LlrProcessing` | parameter names, units, values, uncertainties |
| `normalEquationInfo`, `normalEquationMetadata` | structured | `LlrProcessing` | normal-equation counts, statistics, conventions |
| `covarianceInfo` | structured | `LlrProcessing` | covariance dimensions |
| `matrix` | binary | `LlrProcessing` | normal matrix, right-hand side, `x0`, covariance |
| `observationPrediction` | records | prediction programs | per-epoch pointing and observability |
| `predictionWindow` | records | prediction programs | visibility windows |
| `lunarOrbit` | binary | `LunarOrbitPropagation` | orbit time series |
| `lunarAccelerationDiagnostics` | binary | `LunarOrbitPropagation` | acceleration breakdown |
| `lunarRotation` | binary | `LunarOrbitPropagation` | rotation time series |
| `lunarOrbitMetadata` | structured | `LunarOrbitPropagation` | resolved configuration, sources, statistics |

## Selected layouts

### Normal points

| Field | Kind | Unit |
|---|---|---|
| `jd1_utc`, `jd2_utc` | float | d (two-part UTC Julian date of transmission) |
| `station_name`, `reflector_name` | text | — |
| `round_trip_time_s` | float | s |
| `uncertainty_two_way_s` | float | s |
| `pressure_hpa` | float | hPa |
| `temperature_k` | float | K |
| `humidity_percent` | float | % |
| `wavelength_nm` | float | nm |
| `index` | int | — |
| `station_code`, `reflector_code` | text | — (source codes) |

The header records `timeScale: UTC`, the data-set name, the input and invalid
record counts and the import metadata.

### Earth orientation

`mjdUtc` (d), `xpArcsec`, `ypArcsec`, `ut1MinusUtcSec`, `lodSec`, `dxArcsec`,
`dyArcsec`, `source`, `quality`, `sourceSha256`. Every row has all ten values;
`quality` is `final`, `rapid`, `prediction` or `unspecified`.

### Normal-equation and covariance directories

A normal-equation directory holds `info.txt` (`normalEquationInfo`),
`metadata.txt` (`normalEquationMetadata`), `parameterNames.txt` and the
`matrix` files `normalMatrix.dat.gz`, `rightHandSide.dat.gz` and `x0.dat.gz`.
The system is stored in the absolute form `N x = W` at the linearization
point `x0`. A covariance directory holds `info.txt` (`covarianceInfo`),
`parameterNames.txt` and the lower-symmetric `covariance.dat.gz`.

### Processing report

The top-level keys `converged`, `terminationReason`, `publicationQuality`,
`settings`, `summary`, `parameters` and `observations` are always present.

- `converged` is true exactly when `terminationReason` is `CONVERGED`.
- `publicationQuality` has `formalProductEligible`, `status` (`passed` or
  `diagnostic`) and `reasons`.
- Each entry of `parameters` has a unique `name`, a `unit`, the `finalValue`
  and the formal uncertainty.
- Each entry of `observations` has the `observationId`, `stationId`,
  `epochUtc`, `lightTimeConverged`, `aprioriSigmaM`, the current-state residual
  and its sigma, the weight factor and the variance component.
- `scientificInputs.resources` lists every input file with its SHA-256 (and,
  for ephemeris directories, every kernel with its load order), together with
  `codeSha256` and the package versions.

A missing key and an explicit `null` mean different things: `null` appears only
where a value is legitimately absent, such as a disabled screening threshold or
an output that was not written.
