# Inputs and data contracts

## Normal points

External source data enters through one explicit program:

```text
MINI or CRD/FRD -> NormalPointsConvert -> NormalPointFile
```

The canonical file is typed ASCII (`.txt` or `.txt.gz`). It stores station and
reflector identity, a two-part UTC transmit epoch, two-way light time, two-way
uncertainty, pressure, temperature, humidity, wavelength, and source codes.

Every downstream LLR program accepts `inputFileNormalPoints` or
`inputFilesNormalPoints` according to its declared cardinality. It does not
auto-detect external formats.

`NptRecord.uncertainty_two_way_s` is the only observation uncertainty input.
Estimation uses `0.5 * c * uncertainty_two_way_s` as one-way range sigma.
CRD v2 normal-point record `11` is accepted only for ground-transmit
epoch-event `2`, H4 normal-point data type `1`, and H2 UTC realization `3`, `4`
or `7`. Unsupported events are rejected with their source line and event code;
they are never silently reinterpreted as transmit times.

H1 must declare version 2; record `11` has exactly 14 fields including SNR.
H2 station name and pad ID must agree with the declared source mapping; H3
must identify a supported target exactly. Structural and identity errors abort
with the source line; invalid normal-point rows are rejected and counted.
Historical CRD v1 cannot be imported directly. Obtain a version 2 source or
convert it with an independent tool outside LunarOps. MINI remains a separate
explicit source product. Neither channel is a fallback for a failed parser.
Imported records use exact station and reflector IDs in subsequent catalogs.

CRD conversion requires explicit `crdUncertaintyPolicy`: `binRmsAsSigma` uses
the source bin RMS as the two-way sigma by the caller's declared precision
model; `fixed` requires a positive `crdUncertaintyTwoWayS` in seconds. No
sqrt(N) reduction is inferred. Source bin RMS, window duration, return count,
event code and selected precision policy remain available in the import report
and canonical metadata, separately from the uncertainty used for weighting.
Meteorology uses the nearest absolute UTC sample within
`crdMeteorologyMaxAgeSeconds` (default 1800 s), including across midnight.
MINI temperature units are explicit (`miniTemperatureUnit`: `deciCelsius`,
default, or `deciKelvin`); authoritative batch units still require confirmation.

## Time

`Epoch(jd1, jd2, scale)` is the only runtime scalar time type. Normal-point and
observation-equation files require UTC. Model boundaries explicitly convert to
TT or TDB through ERFA; ephemeris target 16 is not used for time conversion.

Configuration intervals use `[start, endExclusive)`. `null` means no bound.
Observation configuration `startTime` and `endTime` may be civil local times
when `utcOffsetHours` is set; they are normalized to UTC before filtering and
model evaluation. Native normal-point, EOP, catalog, and observation-equation
artifacts remain UTC-only. Human-facing observation and processing outputs may
include a separate local timestamp alongside the UTC value.

## Ephemerides

The `calceph` backend accepts a SPICE kernel directory containing at least one
`.bsp` position kernel and one `.bpc` lunar-orientation kernel. Only those
binary kernels are loaded. LunarOps discovers the unique BPC orientation
target through CALCEPH and requires its records to use ICRF (SPICE frame code
1, historically labeled `J2000`).

```yaml
ephemerides:
  type: calceph
  directory: "{dataDir}/kernels/inpop21a"
  lunarRelativisticScaleConvention: alreadyScaled
  longitudeLibrationCorrection: inpop21a
```

## Catalogs

Each `LlrResiduals` or `LlrProcessing` call consumes native files through
`inputFileStationCatalog` and `inputFileReflectorCatalog`. The native files use
ITRF and Moon PA headers. A station entry contains `key` and either `xyzM` or
the WGS84 geodetic triplet `longitudeDeg`, `latitudeDeg`, and `heightM`.
Longitude is positive east and latitude is positive north, both in degrees;
`heightM` is ellipsoidal height in metres.
The catalog creator converts geodetic input to ITRF XYZ before writing the
native file. A station entry may also include `velocityMPerYear` and
`positionEpochUtc`; a reflector entry contains only `key` and `xyzM`.

```yaml
programs:
  - program: StationCatalogCreate
    outputFileStationCatalog: output/stations.txt
    stationCoordinates:
      - {key: APOLLO, xyzM: [-1463998.9, -5166632.8, 3435012.9], velocityMPerYear: [0, 0, 0]}
      - {key: PRIVATE, longitudeDeg: 116.391, latitudeDeg: 39.907, heightM: 45.0}
  - program: ReflectorCatalogCreate
    outputFileReflectorCatalog: output/reflectors.txt
    reflectorCoordinates:
      - {key: APOLLO11, xyzM: [1591966.6, 690699.5, 21003.8]}
  - program: LlrResiduals
    inputFilesNormalPoints: [output/normalPoints.txt.gz]
    outputFileObservationResults: output/residuals.txt
    inputFileStationCatalog: output/stations.txt
    inputFileReflectorCatalog: output/reflectors.txt
```

`StationCatalogCreate` and `ReflectorCatalogCreate` accept either an external
coordinate source file or their respective inline sequence, and write typed
native files. `LlrProcessing.writeResults` can publish an updated reflector
catalog after estimation.

## Configuration

Run configuration remains YAML (`.yml` or `.yaml`). It is distinct from native
scientific artifacts, even where structured report payloads use YAML scalar
syntax after an LunarOps type header.


2026-10-13 严格输入合同与配置迁移说明见 [审计整改记录](AUDIT_REMEDIATION_20261013.md)。CRD 转换必须显式指定 σ 策略，预报必须指定气象策略；EOP v2 保留质量来源；旧科学指纹不用于新版本续跑。

当前版本与拒绝规则见 [CURRENT_DATA_CONTRACT.md](CURRENT_DATA_CONTRACT.md)。
