# Observation model

`LlrResiduals`, `LlrProcessing` and `LlrObservationPrediction` share one
observation model. For each normal point it computes the round-trip time a
station clock would measure for a light pulse sent to a lunar reflector and
back, and compares it with the observed value.

## Computed observable

For a normal point with UTC transmit epoch `t1` the model solves for the
bounce epoch `t2` at the reflector and the receive epoch `t3` at the station:

1. **Transmit state.** The station's catalog position is propagated with its
   velocity to `t1` and displaced by the station displacement models (ITRF).
   It is rotated to the GCRS with the Earth orientation and transformed to the
   BCRS. `t1` is converted to TDB, including the station-dependent `dtdb`
   term.
2. **Bounce state.** The reflector's PA coordinates plus the reflector
   displacement are rotated to the LCRS with the ephemeris lunar orientation
   at `t2` and transformed to the BCRS.
3. **Light-time legs.** Each leg's travel time is the BCRS geometric distance
   divided by `c`, plus the relativistic (Shapiro) delay and the tropospheric
   delay of that leg. The receive state at `t3` is formed like the transmit
   state.
4. **Iteration.** `t2` and `t3` are iterated until the round-trip time changes
   by less than 10⁻¹² s, at most 12 times. A normal point that does not
   converge is reported with status `light_time_not_converged` and excluded
   from estimation.
5. **Clock interval.** The coordinate (TDB) interval `t3 − t1` is converted to
   the TT interval measured by the station clock. For transmit epochs from
   1968-02-01 to 1971-12-31, when the UTC second differed in rate from the SI
   second, the interval is also divided by `1 + 3·10⁻⁸`.
6. **Range bias.** The deterministic range-bias correction of the station and
   date, in centimetres of two-way light distance, is subtracted from the
   computed round-trip time.

The station displacement at the receive epoch is evaluated in the first two
iterations and then held fixed. Station coordinates, event times and rotations
use the exact epoch of every iteration.

The residual is reported one-way: `oc_one_way_m = 0.5 c (observed_rtt_s − computed_rtt_s)`.

## Model components

All components are model classes, selected in `observationModel`, `shared` or
on the program entry (see [Configuration](configuration.md#model-classes-shared-observationmodel-and-program-entries)).
Options are listed in the [model class reference](../reference/classes.md).

### Troposphere — `mendesPavlis`

The optical zenith delay and mapping function of Mendes & Pavlis (IERS
Conventions 2010, Ch. 9.1), evaluated per leg from the normal point's pressure,
temperature, humidity and wavelength. Below 3° elevation the model is outside
its validity range; the delay is then evaluated at 3° and the normal point is
marked as clamped. `lowElevationPolicy` decides what happens to clamped normal
points: `rejectClamped` (default) excludes them, `allowFlagged` keeps them with
the flag.

### Relativistic delay — `iersShapiro`

The one-way gravitational delay of IERS Conventions (2010), Eq. 11.17, for the
Sun, the Earth, the Moon, Jupiter and Saturn as point masses, evaluated for
each leg.

### Station displacement

The configured list is summed in the ITRF:

| Type | Effect | Reference |
|---|---|---|
| `iers2010SolidEarthTide` | solid Earth tides | IERS Conventions (2010), Ch. 7.1.1 |
| `iers2010PoleTide` | solid Earth pole tide | Ch. 7.1.4 |
| `iers2010OceanPoleTide` | ocean pole-tide loading | Ch. 7.1.5 |
| `iers2010OceanTidalLoading` | ocean tidal loading from BLQ coefficients | Ch. 7.1.2 (HARDISP) |

### Reflector displacement — `lunarSolidTide`

The degree-2 lunar body tide raised by the Earth and the Sun, with radial and
horizontal Love numbers `h2` and `l2`, applied in the LCRS (Pavlov et al.,
2016).

### Range bias

`none`, the built-in `inpop21a` table, or a user `table` of station and
interval corrections (see [Input data](input-data.md#range-bias-tables)).

## Elevation limits

`minElevationDeg` excludes normal points whose uplink or downlink vacuum
elevation at the station is below the limit. Excluded normal points stay in the
residual table with status `below_elevation_limit`.

## Selecting data

`startTime`/`endTime` restrict the transmit epochs (`[start, end)`). The
station and reflector of each normal point are looked up in the catalogs by
their exact names; `overrideStation` and `overrideReflector` force one catalog
key for all normal points.

## Output columns

`LlrResiduals` writes an `observationResult` table with one row per normal
point.

**Standard level**

| Column | Unit | Meaning |
|---|---|---|
| `obs_time_utc`, `obs_time_local` | — | transmit epoch in UTC and local civil time |
| `normal_point_index` | — | index of the normal point in the input |
| `station_id`, `station_name` | — | source station code and catalog key |
| `reflector_id`, `reflector_name` | — | source reflector code and catalog key |
| `observed_rtt_s`, `computed_rtt_s` | s | observed and computed round-trip time |
| `oc_one_way_m` | m | one-way residual |
| `observation_sigma_one_way_m` | m | a-priori one-way sigma |
| `elevation_up_deg` | deg | uplink elevation at the station |
| `light_time_converged` | — | light-time iteration converged |
| `status` | — | `ok`, `below_elevation_limit` or `light_time_not_converged` |

With `includeReflectorDesign: true`, the columns `design_reflector_pa_x/y/z`
add the partial derivatives of the one-way residual with respect to the
reflector PA coordinates.

**Full level** adds the meteorological inputs, the computed round-trip time
before the range bias, the TDB coordinate interval and the TT−TDB and UTC rate
corrections, the event-closure residuals of each leg, the range-bias lookup
and its components, the longitude-libration correction and lunar scale
convention, the TDB transmit, bounce and receive epochs, the geometric range,
gravitational and tropospheric delay of each leg, the troposphere elevations
and clamp flags, the downlink elevation, and the station and reflector
displacements. The file header declares every column with its unit.
