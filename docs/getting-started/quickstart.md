# Quickstart

This walkthrough computes O−C residuals for a set of normal points. It covers
the steps every LLR analysis shares: prepare the Earth orientation, convert the
normal points, create the station and reflector catalogs, and evaluate the
observation model. The complete runnable scenario is
`configs/lunarops_oc_residuals.yml`.

## 1. Collect the input data

| Data | Typical file | Used for |
|---|---|---|
| SPICE kernels of a planetary/lunar ephemeris (`.bsp` and `.bpc`) | `kernels/inpop21a/` | body states and lunar orientation |
| IERS 20 C04 daily EOP | `eopc04.1962-now.txt` | Earth orientation of the past |
| IERS finals2000A | `finals2000A.all` | recent and predicted Earth orientation |
| Ocean-loading coefficients (BLQ) | `fes2022b.txt` | ocean tidal loading |
| Ocean pole-tide coefficients | `opoleloadcoefcmcor.txt` | ocean pole-tide loading |
| Normal points (CRD v2 or MINI) | `*.crd`, `*.npt` | observations |

[Input data](../user-guide/input-data.md) describes each source.

## 2. Write the configuration

A run configuration has three sections: `variables`, `models` and `programs`.

```yaml
variables:
  dataDir: ~/llr/data
  outputDir: ./output

models:
  ephemerides:
    type: calceph
    directory: "{dataDir}/kernels/inpop21a"
    lunarRelativisticScaleConvention: alreadyScaled
    longitudeLibrationCorrection: inpop21a
  earthRotation:
    type: file
    file: "{outputDir}/eop_merged.txt"
  troposphere: {type: mendesPavlis}
  relativity: {type: iersShapiro}
  stationDisplacement:
    - type: iers2010SolidEarthTide
    - type: iers2010PoleTide
    - type: iers2010OceanPoleTide
      coefficientFile: "{dataDir}/auxiliary/opoleloadcoefcmcor.txt"
    - type: iers2010OceanTidalLoading
      coefficientFile: "{dataDir}/auxiliary/fes2022b.txt"
      model: FES2022b
  reflectorDisplacement: {type: lunarSolidTide}
  rangeBias: {type: inpop21a}

programs:
  # Earth orientation: C04 for the past, Bulletin A for recent days.
  - program: IersC04EarthOrientationParameter
    inputFileC04: "{dataDir}/auxiliary/eopc04.1962-now.txt"
    outputFileEarthOrientationParameter: "{outputDir}/eop_c04.txt"
  - program: IersRapidEarthOrientationParameter
    inputFileFinals2000A: "{dataDir}/auxiliary/finals2000A.all"
    outputFileEarthOrientationParameter: "{outputDir}/eop_rapid.txt"
  - program: EarthOrientationParameterMerge
    inputFileC04EarthOrientationParameter: "{outputDir}/eop_c04.txt"
    inputFileRapidEarthOrientationParameter: "{outputDir}/eop_rapid.txt"
    outputFileEarthOrientationParameter: "{outputDir}/eop_merged.txt"

  # Normal points: external formats enter only through this program.
  - program: NormalPointsConvert
    inputFilesNormalPoints: ["{dataDir}/crd/"]
    crdUncertaintyPolicy: binRmsAsSigma
    outputFileNormalPoints: "{outputDir}/normal_points.txt"
    outputFileImportReport: "{outputDir}/normal_point_import_report.txt"

  # Catalogs: station ITRF positions and reflector Moon-PA positions.
  - program: StationCatalogCreate
    outputFileStationCatalog: "{outputDir}/stations.txt"
    stationCoordinates:
      - {key: APOLLO, xyzM: [-1463998.9079, -5166632.7663, 3435012.8921],
         velocityMPerYear: [-0.0139, -0.0003, -0.0023], positionEpochUtc: "2000-01-01T00:00:00"}
  - program: ReflectorCatalogCreate
    outputFileReflectorCatalog: "{outputDir}/reflectors.txt"
    reflectorCoordinates:
      - {key: APOLLO15, xyzM: [1554678.3071, 98095.5262, 765005.2077]}

  # Residuals.
  - program: LlrResiduals
    inputFilesNormalPoints: ["{outputDir}/normal_points.txt"]
    inputFileStationCatalog: "{outputDir}/stations.txt"
    inputFileReflectorCatalog: "{outputDir}/reflectors.txt"
    outputLevel: standard
    outputFileObservationResults: "{outputDir}/residuals.txt"
```

Placeholders such as `{dataDir}` are replaced by the values in `variables`.
The model classes in `models` apply to every program
that needs them; a program entry can override any of them. The station and
reflector keys in the catalogs must match the identities in the normal points
exactly.

## 3. Validate

```bash
python -m lunarops validate quickstart.yml
```

`validate` resolves variables, checks every program's fields and class
declarations, and checks that each input file either exists or is produced by
an earlier program of the same run. For existing native inputs it reads only
the header, to check the artifact type; no model is evaluated. Add
`--resolve` to print the fully expanded configuration with all defaults.

## 4. Run

```bash
python -m lunarops run quickstart.yml
```

Or distribute the normal points over MPI workers (at least two ranks):

```bash
mpirun -n 8 python -m lunarops run quickstart.yml --mpi
```

## 5. Read the results

`residuals.txt` is a typed text table with one row per normal point. Its
header lists every column with its unit. The main columns are the transmit
epoch (`obs_time_utc`), station and reflector, the observed and computed
round-trip times (`observed_rtt_s`, `computed_rtt_s`), the one-way residual
`oc_one_way_m`, its a-priori sigma, the elevation and a `status` such as
`ok`, `below_elevation_limit` or `light_time_not_converged`. `outputLevel: full`
adds every delay, correction and event time. See
[Observation model](../user-guide/observation-model.md) and
[File formats](../reference/file-formats.md).

## Next steps

- Estimate reflector coordinates and station biases with `LlrProcessing`:
  [Parameter estimation](../user-guide/estimation.md).
- Predict the next observing night: [Observation prediction](../user-guide/prediction.md).
- Integrate the lunar orbit: [Lunar orbit and rotation](../user-guide/lunar-dynamics.md).
