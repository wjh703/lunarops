# Observation prediction

`LlrObservationPrediction` evaluates a time grid for one station and one
reflector and reports where to point and when the target is observable. It
uses the same light-time solution as the residual computation, restricted to
the outgoing leg: for each transmit epoch `t1` it solves for the bounce epoch
`t2` at the reflector.

```yaml
- program: LlrObservationPrediction
  inputFileStationCatalog: "{outputDir}/stations.txt"
  inputFileReflectorCatalog: "{outputDir}/reflectors.txt"
  outputFilePrediction: "{outputDir}/prediction.txt"
  outputFileWindows: "{outputDir}/prediction_windows.txt"
  stationName: APOLLO
  reflectorName: APOLLO15
  startTime: "2024-09-18T00:00:00"
  endTime: "2024-09-18T23:55:00"
  utcOffsetHours: 8
  stepSeconds: 300
  meteorologyPolicy: nominal
  minElevationDeg: 20
  minReflectorElevationDeg: 0
  maxSunElevationDeg: -6
  allowedElongationRangesDeg: [{startDeg: 30, endDeg: 330}]
```

`configs/lunarops_observation_prediction.yml` is a complete scenario including
the EOP preparation.

## Time grid

The grid runs from `startTime` to `endTime` in steps of `stepSeconds`. Naive
times are civil times at `utcOffsetHours`. Output timestamps carry three
fractional-second digits, so `startTime` and `stepSeconds` must be whole
milliseconds; the computation itself keeps full precision. The Earth
orientation table must cover the grid; with Bulletin A predictions in the
merged EOP file, future dates can be predicted.

## Observability

A grid sample is observable when all four criteria hold:

| Criterion | Option | Quantity |
|---|---|---|
| Elevation | `minElevationDeg` | topocentric elevation of the reflector at the station |
| Reflector illumination geometry | `minReflectorElevationDeg` | elevation of the incoming beam above the reflector's local lunar horizon |
| Darkness | `maxSunElevationDeg` | Sun elevation at the station |
| Lunar phase | `allowedElongationRangesDeg` | mean elongation `D` of the Moon from the Sun (ERFA `fad03`); a range whose `startDeg` exceeds `endDeg` wraps through 0° |

The tropospheric model needs weather: `meteorologyPolicy: nominal` uses
900 hPa, 285 K, 25 % relative humidity and 532 nm for any value not given,
`measured` requires `pressureHpa`, `temperatureK`, `relativeHumidityPercent`
and `wavelengthNm`.

## Outputs

The prediction table (`observationPrediction`) has one row per grid epoch:

| Column | Unit | Meaning |
|---|---|---|
| `utc_t1`, `local_t1` | — | transmit epoch, UTC and local, millisecond resolution |
| `station`, `reflector` | — | catalog keys |
| `station_itrf_x/y/z_m` | m | station ITRF position at `t1` |
| `reflector_itrf_x/y/z_m` | m | reflector position at `t2`, in the ITRF |
| `range_up_geometric_m` | m | geometric uplink range |
| `azimuth_deg`, `elevation_deg` | deg | pointing at the station |
| `observable` | — | all criteria met |

The window table (`predictionWindow`) joins adjacent observable samples into
windows with `start_utc`, `end_utc`, their local equivalents, the sample count
and the duration in seconds.

The program predicts geometry and coarse visibility only. It does not compare
with CPF predictions and adds no atmospheric refraction, pointing model or
telescope control output.

## Merging prediction grids

Campaigns often consist of separate nights. Run `LlrObservationPrediction` per
night (for example with a `loop`) and merge the results:

```yaml
- program: LlrObservationPredictionMerge
  inputFilesPrediction: ["{outputDir}/night_1.txt", "{outputDir}/night_2.txt"]
  outputFilePrediction: "{outputDir}/campaign.txt"
  outputFileWindows: "{outputDir}/campaign_windows.txt"
  stepSeconds: 300
```

The inputs must belong to one station and reflector and must not share
epochs. The rows are sorted by `utc_t1` and the windows are rebuilt with
`stepSeconds`: samples further apart start a new window.

## MPI

With `--mpi`, grid epochs are sent to the workers in blocks of `mpi.chunksize`.
Rank 0 restores the chronological order, builds the windows and writes both
files.
