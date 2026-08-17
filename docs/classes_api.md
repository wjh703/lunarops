# LunarOps classes API

本文档描述 lunarops/classes/ 当前版本的非私有 Python API。配置文件是模型选择和参数输入的主要入口；Python 调用者通常只需要使用各包的 __init__.py 导出，不应依赖以下划线开头的辅助函数或属性。

## 配置入口

### lunarops.classes

| 名称 | 参数 | 功能 |
|---|---|---|
| Epoch | jd1, jd2, scale | 二段儒略日标量历元。 |
| TimeScale | UTC、TT、TDB | 显式时间尺度枚举。 |
| TimeScaleConverter | 无 | 基于 ERFA 的 UTC、TT、TDB 时间尺度转换。 |
| ObservationAssembly | model_configs, station_catalog, reflector_catalog | 保存解析后的观测模型配置和目录。 |
| ensure_registered() | 无 | 注册内置配置工厂，幂等调用。 |
| resolve_observation_assembly(context, program_config, *, station_catalog=None, reflector_catalog=None) | 运行上下文、程序配置、可选目录 | 合并 globals 与程序级模型配置并加载目录。 |
| build_observation_processor(context, program_config, *, station_catalog=None, reflector_catalog=None) | 同上 | 创建完整 LLR 观测处理器。 |

内置配置类别为 ephemerides、earthRotation、troposphere、relativity、stationDisplacement、reflectorDisplacement、rangeBias 和 parametrization。类别的 type 和选项由 YAML 配置提供。

### 配置工厂

| 类别 | 内置 type | 主要配置参数 |
|---|---|---|
| ephemerides | calceph | file, lunarRelativisticScaleConvention, longitudeLibrationCorrection |
| earthRotation | iersC04 | file, duplicateMjdPolicy |
| troposphere | none, mendesPavlis | 无额外参数 |
| relativity | none, iersShapiro | 无额外参数；iersShapiro 使用观测上下文的 ephemeris |
| stationDisplacement | none, iers2010SolidEarthTide, iers2010PoleTide, iers2010OceanPoleTide, iers2010OceanTidalLoading | 配置值是非空列表，所有列出的模型自动相加；海潮模型使用 coefficientFile 和可选 model |
| reflectorDisplacement | none, lunarSolidTide | h2, l2, moonRadiusM |
| rangeBias | none, inpop21a, table | table 使用 file 或 biases 二选一 |
| parametrization | reflectorPosition, stationRangeBias | 见参数化章节 |

## Delays

模块：lunarops.classes.delays

GravitationalDelay 是抽象接口：path_delay_m(transmitter_bcrs_m, receiver_bcrs_m, epoch_tdb) -> float。输入为发射端、接收端 BCRS 米制位置和 TDB 历元，返回单程引力路径延迟（米）。

Iers2010ShapiroDelay(ephemeris) 实现 IERS 2010 点质量 Shapiro 延迟，方法为 path_delay_m(transmitter_bcrs_m, receiver_bcrs_m, epoch_tdb)。

ZeroGravitationalDelay 的 path_delay_m(...) 恒返回 0.0，用于 relativity: none。

TroposphereInput 字段为 elevation_rad、pressure_hpa、temperature_k、relative_humidity_percent、latitude_rad、height_m、wavelength_um。

TroposphereDelay 是抽象接口：elevation_floor_rad 属性和 slant_delay_m(data: TroposphereInput) -> float。

Iers2010MendesPavlisTroposphere 实现 Mendes-Pavlis 模型，提供 elevation_floor_rad 和 slant_delay_m(data)。ZeroTroposphereDelay 的 elevation_floor_rad 为 None，slant_delay_m(data) 恒返回 0.0。

## Displacement

模块：lunarops.classes.displacement

StationDisplacementInput(reference_position_itrf_m, epoch_utc, station_id=None) 是测站位移输入；ReflectorDisplacementInput(reference_position_lcrs_m, epoch_tdb) 是反射器位移输入。StationDisplacement 提供 displacement_itrf_m(data)，ReflectorDisplacement 提供 displacement_lcrs_m(data)。ZeroStationDisplacement 和 ZeroReflectorDisplacement 返回零向量。CompositeStationDisplacement(components) 将多个测站模型求和。

Iers2010SolidEarthTide(frame_system) 提供 displacement_itrf_m(data)。Iers2010SolidEarthPoleTide(earth_orientation_provider) 提供 evaluate(data) 和 displacement_itrf_m(data)。Iers2010OceanPoleTide(grid, earth_orientation_provider) 提供 evaluate(data) 和 displacement_itrf_m(data)。Iers2010OceanTidalLoading(catalog) 提供 evaluate(data) 和 displacement_itrf_m(data)。LunarSolidTide(ephemeris, h2=0.0423, l2=0.0107, moon_radius_m=1737400.0) 提供 displacement_lcrs_m(data)。

OceanPoleTideGrid(coefficient_file) 提供 info 属性和 coefficients_at(reference_position_itrf_m)。OceanPoleTideCoefficients、OceanPoleTideGridInfo、OceanPoleTideResult 分别表示插值系数、网格元数据和模型结果。

OceanTidalLoadingCatalog(coefficient_file) 提供 info、station_ids、coefficients_for(station_id)。OceanTidalLoadingCoefficients、OceanTidalLoadingCatalogInfo、OceanTidalLoadingResult 分别表示 BLQ 测站记录、目录元数据和模型结果。

PolarWobble、PoleTideResult 是极移和极潮结果数据类。secular_pole_2018_arcsec(epoch_utc) 返回秒差 (x, y)，polar_wobble(epoch_utc, earth_orientation_provider) 返回 PolarWobble。

GeodeticPosition 提供 latitude_deg、longitude_deg 属性。工具函数 enu2itrf(enu_m, latitude_rad, longitude_rad)、itrf2geodetic(station_itrf_m)、itrf2geocentric(station_itrf_m)、local_up_unit_itrf(station_itrf_m) 完成地面站坐标转换。

## Ephemerides

模块：lunarops.classes.ephemerides

Ephemeris 抽象接口包含 source_file_path、body_state_bcrs(body_name, epoch_tdb)、body_position_bcrs(body_name, epoch_tdb)、pa2lcrs_matrix(epoch_tdb)、longitude_libration_correction_type、longitude_libration_correction_rad(epoch_tdb)、l_b_minus_l_l、lunar_relativistic_scale_convention 和 close()。

BodyState(position_m, velocity_mps) 是不可变 BCRS 状态，位置单位米、速度单位米/秒。

CalcephEphemeris(ephemeris_file, *, lunar_relativistic_scale_convention, longitude_libration_correction_type=None) 是 CALCEPH/INPOP/DE 实现。公开属性为 source_file_path、l_b_minus_l_l、lunar_relativistic_scale_convention、longitude_libration_correction_type；公开方法为 body_state_bcrs、pa2lcrs_matrix、longitude_libration_correction_rad、close()。target16_tdb_minus_tt_s() 仅用于将 target-16 与 ERFA 进行诊断比较；load_calceph_ephemeris(...) 不再要求 target-16。

require_tdb_epoch(epoch, name="epoch") 要求 Epoch 且尺度为 TDB。LongitudeLibrationCorrectionType 的值为 none、inpop21a。normalize_longitude_libration_correction_type(value) 和 make_longitude_libration_correction_model(correction_type) 显式选择月球经度修正模型。

## Frames

模块：lunarops.classes.frames

PolarMotion、CelestialPoleOffsets、EarthOrientationSample 是不可变地球定向数据类。EarthOrientationProvider 抽象接口包含 source_file_path、polar_motion(epoch_utc)、celestial_pole_offsets(epoch_utc)、ut1_minus_utc_s(epoch_utc)、close()。

TabulatedEarthOrientation(samples, *, source_file_path=None, duplicate_mjd_policy="error") 提供 from_columns(...)、to_mpi_payload()、from_mpi_payload(payload)、source_file_path、duplicate_mjd_policy、mjd_utc_range、samples，以及 polar_motion、ut1_minus_utc_s、celestial_pole_offsets。read_iers_eop(eop_file) 解析 IERS C04/FINALS；load_iers_eop(eop_file, *, duplicate_mjd_policy="error") 读取并构造表。

TerrestrialFrameTransform(earth_orientation_provider) 提供 ut1_jd(epoch_utc)、tdb_topocentric_arguments(position_itrf_m, epoch_utc)、gcrs2itrf_matrix(epoch_utc)、gcrs2itrf(position_gcrs_m, epoch_utc)、itrf2gcrs(position_itrf_m, epoch_utc)。前两者以 C04 和高频 EOP 构造 ERFA `dtdb` 所需的 UT1 与测站参数。LunarFrameTransform(ephemeris) 提供 pa2lcrs(position_pa_m, epoch_tdb)、lcrs2pa(position_lcrs_m, epoch_tdb)。RelativisticFrameTransform(ephemeris) 提供 external_gravitational_potential_m2_s2(...)、gcrs2bcrs、bcrs2gcrs、lcrs2bcrs、bcrs2lcrs、lcrs2gcrs、gcrs2lcrs。

ReferenceFrameSystem(ephemeris, earth_orientation_provider) 是组合门面，提供上述框架转换和 external_gravitational_potential_m2_s2，并持有 time_scale_converter、terrestrial_transform、lunar_transform、relativistic_transform。

HighFrequencyEopCorrection 保存海潮和章动两部分的高频修正；delta_xp_arcsec、delta_yp_arcsec、delta_ut1_s 是合并后的属性。ocean_tide_eop_correction(epoch_utc, *, background_ut1_minus_utc_s)、earth_rotation_libration_eop_correction(epoch_tt_or_tdb)、high_frequency_eop_correction(epoch_utc, *, background_ut1_minus_utc_s) 返回对应修正。

## Time scale

模块：lunarops.classes.time

Epoch(jd1, jd2, scale) 是唯一的运行时标量时间类型，提供 from_isot、from_calendar、from_date_seconds、shifted、seconds_until、date_iso、to_datetime 和 isot。TimeScale 枚举包含 UTC、TT 和 TDB；utc2tt 与 tt2utc 提供不依赖星历的 UTC/TT 转换。

TimeScaleConverter() 以 ERFA `dtdb` 完成 TDB/TT 转换，不依赖星历。TdbTopocentricArguments 保存 UT1 日分数、站经度、到自转轴距离和赤道北向距离。tdb_minus_tt_s(epoch_tdb, *, topocentric_arguments=None) 可直接传入这些参数；tdb2tt、tt2tdb、convert 的 topocentric_observer 参数为接收 UTC 并返回这些参数的回调。TDB 到 TT 在原始 TDB 历元直接计算；为取得站心 UTC 参数会先作一次地心预估。TT 到 TDB 以 TDB 为自变量固定点迭代；达到上限时采用最后一次结果。

## Observation

ObservationResultDetail 值为 standard、full，parse(value) 解析配置值。ObservationEquation 保存 observed_minus_computed_one_way_m、sigma_one_way_m、design_partials、observation_id、station_key、reflector_key、transmit_epoch_utc、light_time_converged=True、wavelength_nm=None。

TroposphereEnvironment(...) 提供 troposphere_input(elevation_rad)。LightTimeRequest(...) 提供 station_reference_itrf_at(epoch_utc)。LightTimeLeg(...) 提供 path_length_m、travel_time_s；troposphere_elevation_used_rad 默认 None，troposphere_elevation_clamped 默认 False。LightTimeSolution 保存三次 TDB 事件历元、各项延迟、位置、迭代次数和 light_time_converged。LightTimeSolver(frame_system, gravitational_delay_model, troposphere_delay_model, station_displacement_model, reflector_displacement_model) 提供 solve(request)。

LlrObservationModel(frame_system, light_time_solver, range_bias_model) 提供 ephemeris 属性和 evaluate(resolved_observation, *, min_elevation_deg, include_reflector_position_partials=False, result_detail=None)。LlrObservationEvaluation 保存 equation、result_row=None、below_elevation_limit=False。

ObservationCatalogSelection(station_identifier=None, reflector_identifier=None) 描述筛选条件。ResolvedObservation 保存目录解析后的 normal point、站点、反射器和历元，并提供 transmit_epoch_utc、station_identity_candidates。ObservationCatalogState(station_catalog, reflector_catalog) 提供 reflector_positions_pa_m()、apply_reflector_positions_pa_m(positions)。ObservationResolver(model_state) 提供 resolve(record, selection=None) 和 resolve_all(records, selection=None)。

ObservationProcessingOptions(station_identifier=None, reflector_identifier=None, min_elevation_deg=0.0, include_reflector_position_partials=False, show_progress=False, progress_description=None) 提供 catalog_selection 和 with_progress(description, *, enabled=None)。LlrObservationProcessor(resolver, observation_model) 提供 equations(dataset, *, options=None) 和 rows(dataset, *, options=None, detail=ObservationResultDetail.STANDARD)。

## Parametrization

Parametrization 是参数块基类，接口为 block_id、setup(equations, model_state)、parameter_names()、parameter_count、design_columns(eq)、design_entries(eq)、reduce_observation(eq)、apply_update(delta)、max_update_norm(delta)、state()、initial_update(equations, *, weight_cap, maximum_iterations)。ParametrizationList(blocks) 将参数块拼接，提供 blocks、setup、parameter_names、select_blocks、parameter_count、design_entries、design_row、design_value、reduced_observation、split、update_norms、apply_update、state、initial_update、matched_parameter_names。

ReflectorPositionParametrization(*, reflectors=None) 估计反射器 PA 坐标改正，提供 from_config、setup、parameter_names、design_columns、design_entries、apply_update、max_update_norm、state。

StationBiasInterval(station, start, end_exclusive, name=None) 提供 key、active_at(epoch)。parse_station_bias_intervals(config_value)、canonical_station_for_equation(eq)、active_station_bias_interval_keys(intervals, eq, *, requested=None) 是对应工具函数。StationRangeBiasParametrization(*, stations=None, per="station", intervals=None) 支持 per=station 或 station+interval，提供 from_config、setup、parameter_names、design_columns、design_entries、reduce_observation、initial_update、apply_update、state。

## Range bias

RangeBiasRequest(station_identifiers, observation_epoch_utc) 描述查表请求。RangeBiasCorrection(model_label, lookup) 提供 correction_two_way_cm、correction_two_way_m、correction_round_trip_time_s、correction_one_way_m、apply_to_computed_round_trip_time_s(computed_round_trip_time_s)。RangeBiasModel 抽象方法为 evaluate(request)；ZeroRangeBiasModel 返回零修正；TableRangeBiasModel(bias_table) 提供 model_label 和 evaluate(request)。

RangeBiasComponent、RangeBiasLookup、AdditiveRangeBiasTable 是表数据、查表结果和表容器。AdditiveRangeBiasTable 提供 first_table_station_id、lookup、active_components、total_correction_two_way_cm、coverage_intervals_by_station、from_mapping。builtin_additive_range_bias_table(name) 返回内置表；load_additive_range_bias_table(path) 从文件读取表。RangeBiasLookupStatus 的值为 matched、explicit_zero、station_not_in_table、outside_coverage，表示匹配、显式零、站点不在表内和日期超出覆盖范围。

## Relativistic constants

LunarRelativisticScaleConvention 的值为 tdbCompatibleLunarSurface 和 alreadyScaled。前者采用 L_B-L_L 的月球表面坐标时尺度；其中 L_L = 3.139054e-11 由月球重力和自转导出，类似地球的 L_G，不是 DE440 内核元数据。normalize_lunar_relativistic_scale_convention(value) 解析配置；l_b_minus_l_l_for_convention(convention) 返回实际使用的无量纲 L_B-L_L 项。其余 L_*、GM_* 和外部势天体元组是物理常量，不是运行时配置接口；GM_BY_BODY 为只读映射。L_L 的来源见 Turyshev et al. (2024) 的 [Table 2](https://doi.org/10.3847/1538-4357/adcc18)。

## 设计约定

1. 配置字段使用类别名和 type 选择模型；计算类不根据文件名猜测模型。
2. 单位写入名称：_m、_s、_arcsec、_deg、_rad、_cm 分别表示米、秒、角秒、度、弧度、厘米。
3. Epoch 参数名包含时间尺度后缀，例如 epoch_utc、epoch_tdb；坐标参数名包含参考系后缀，例如 position_itrf_m、position_bcrs_m。
4. classes 包级导出是稳定入口；以下划线开头的函数、缓存字段和解析辅助函数属于实现细节。
5. 配置校验负责选择模型、检查键和解析文件；数值模型仍负责物理量的有限性、维度和单位不变量。
6. 同一 category/type 不能被静默覆盖；扩展模型如需替换已有类型，必须在注册时显式传入 replace=True。
7. 每次 build_observation_processor() 都有独立的模型缓存命名空间；ephemeris 和 Earth orientation 仍由 RunContext 共享缓存。

## 完整签名索引

本节逐项列出 `lunarops/classes/` 的非私有类、模块函数和公开方法，便于逐项核查参数。`fields:` 表示冻结数据类的构造参数；未列出的下划线名称是实现细节。属性以 `property` 标记，`self`/`cls` 省略。

### `delays`

```text
GravitationalDelay.path_delay_m(transmitter_bcrs_m: ArrayLike, receiver_bcrs_m: ArrayLike, epoch_tdb: Epoch)
ZeroGravitationalDelay.path_delay_m(transmitter_bcrs_m, receiver_bcrs_m, epoch_tdb: Epoch)
TroposphereInput fields: elevation_rad: float, pressure_hpa: float, temperature_k: float,
    relative_humidity_percent: float, latitude_rad: float, height_m: float, wavelength_um: float
TroposphereDelay.elevation_floor_rad property
TroposphereDelay.slant_delay_m(data: TroposphereInput)
ZeroTroposphereDelay.slant_delay_m(data: TroposphereInput)
Iers2010ShapiroDelay(ephemeris: Ephemeris)
Iers2010ShapiroDelay.path_delay_m(transmitter_bcrs_m: ArrayLike, receiver_bcrs_m: ArrayLike, epoch_tdb: Epoch)
Iers2010MendesPavlisTroposphere.elevation_floor_rad property
Iers2010MendesPavlisTroposphere.slant_delay_m(data: TroposphereInput)
```

### `displacement`

```text
StationDisplacementInput fields: reference_position_itrf_m: np.ndarray, epoch_utc: Epoch, station_id: str | None = None
ReflectorDisplacementInput fields: reference_position_lcrs_m: np.ndarray, epoch_tdb: Epoch
StationDisplacement.displacement_itrf_m(data: StationDisplacementInput)
ReflectorDisplacement.displacement_lcrs_m(data: ReflectorDisplacementInput)
ZeroStationDisplacement.displacement_itrf_m(data: StationDisplacementInput)
ZeroReflectorDisplacement.displacement_lcrs_m(data: ReflectorDisplacementInput)
CompositeStationDisplacement(components: Sequence[StationDisplacement])
CompositeStationDisplacement.displacement_itrf_m(data: StationDisplacementInput)
LunarSolidTide(ephemeris: Ephemeris, h2: float = LUNAR_H2, l2: float = LUNAR_L2,
    moon_radius_m: float = MOON_REFERENCE_RADIUS_M)
LunarSolidTide.displacement_lcrs_m(data: ReflectorDisplacementInput)
OceanPoleTideCoefficients fields: latitude_rad: float, longitude_rad: float, ellipsoidal_height_m: float,
    radial: complex, north: complex, east: complex
OceanPoleTideGridInfo fields: coefficient_file: Path, latitude_nodes: int, longitude_nodes: int,
    latitude_min_deg: float, latitude_max_deg: float, longitude_min_deg: float, longitude_max_deg: float,
    latitude_step_deg: float, longitude_step_deg: float
OceanPoleTideResult fields: displacement_itrf_m: np.ndarray, displacement_enu_m: np.ndarray,
    coefficients: OceanPoleTideCoefficients, wobble: PolarWobble
OceanPoleTideGrid(coefficient_file: str | Path)
OceanPoleTideGrid.info property
OceanPoleTideGrid.coefficients_at(reference_position_itrf_m: ArrayLike)
Iers2010OceanPoleTide(grid: OceanPoleTideGrid, earth_orientation_provider: EarthOrientationProvider)
Iers2010OceanPoleTide.evaluate(data: StationDisplacementInput)
Iers2010OceanPoleTide.displacement_itrf_m(data: StationDisplacementInput)
OceanTidalLoadingCoefficients fields: station_id: str, source_station_name: str,
    amplitudes_m: np.ndarray, phases_deg: np.ndarray
OceanTidalLoadingCatalogInfo fields: coefficient_file: Path, station_count: int,
    tidal_model: str | None, center_of_mass_correction: bool | None
OceanTidalLoadingResult fields: displacement_itrf_m: np.ndarray, displacement_enu_m: np.ndarray,
    displacement_up_south_west_m: np.ndarray, coefficients: OceanTidalLoadingCoefficients
OceanTidalLoadingCatalog(coefficient_file: str | Path)
OceanTidalLoadingCatalog.info property
OceanTidalLoadingCatalog.station_ids property
OceanTidalLoadingCatalog.coefficients_for(station_id: object)
Iers2010OceanTidalLoading(catalog: OceanTidalLoadingCatalog)
Iers2010OceanTidalLoading.evaluate(data: StationDisplacementInput)
Iers2010OceanTidalLoading.displacement_itrf_m(data: StationDisplacementInput)
PolarWobble fields: xp_arcsec: float, yp_arcsec: float, secular_x_arcsec: float, secular_y_arcsec: float,
    m1_arcsec: float, m2_arcsec: float
PolarWobble.m1_rad property
PolarWobble.m2_rad property
PoleTideResult fields: displacement_itrf_m: np.ndarray, displacement_enu_m: np.ndarray,
    wobble: PolarWobble, geocentric_latitude_rad: float, longitude_rad: float
secular_pole_2018_arcsec(epoch_utc: Epoch)
polar_wobble(epoch_utc: Epoch, earth_orientation_provider: EarthOrientationProvider)
Iers2010SolidEarthPoleTide(earth_orientation_provider: EarthOrientationProvider)
Iers2010SolidEarthPoleTide.evaluate(data: StationDisplacementInput)
Iers2010SolidEarthPoleTide.displacement_itrf_m(data: StationDisplacementInput)
Iers2010SolidEarthTide(frame_system: ReferenceFrameSystem)
Iers2010SolidEarthTide.displacement_itrf_m(data: StationDisplacementInput)
GeodeticPosition fields: latitude_rad: float, longitude_rad: float, ellipsoidal_height_m: float
GeodeticPosition.latitude_deg property
GeodeticPosition.longitude_deg property
enu2itrf(enu_m: ArrayLike, *, latitude_rad: float, longitude_rad: float)
itrf2geodetic(station_itrf_m: ArrayLike)
itrf2geocentric(station_itrf_m: ArrayLike)
local_up_unit_itrf(station_itrf_m: ArrayLike)
```

### `ephemerides`

```text
require_tdb_epoch(epoch: Epoch, *, name: str = "epoch")
LongitudeLibrationCorrectionType enum: none, inpop21a
BodyState fields: position_m: np.ndarray, velocity_mps: np.ndarray
Ephemeris.source_file_path property
Ephemeris.body_state_bcrs(body_name: str, epoch_tdb: Epoch)
Ephemeris.body_position_bcrs(body_name: str, epoch_tdb: Epoch)
Ephemeris.pa2lcrs_matrix(epoch_tdb: Epoch)
Ephemeris.longitude_libration_correction_type property
Ephemeris.longitude_libration_correction_rad(epoch_tdb: Epoch)
Ephemeris.l_b_minus_l_l property
Ephemeris.lunar_relativistic_scale_convention property
Ephemeris.close()
CalcephEphemeris(ephemeris_file: str | Path, *,
    lunar_relativistic_scale_convention: LunarRelativisticScaleConvention | str,
    longitude_libration_correction_type: LongitudeLibrationCorrectionType | str | None = None)
CalcephEphemeris.source_file_path property
CalcephEphemeris.l_b_minus_l_l property
CalcephEphemeris.lunar_relativistic_scale_convention property
CalcephEphemeris.longitude_libration_correction_type property
CalcephEphemeris.close()
CalcephEphemeris.body_state_bcrs(body_name: str, epoch_tdb: Epoch)
CalcephEphemeris.longitude_libration_correction_rad(epoch_tdb: Epoch)
CalcephEphemeris.pa2lcrs_matrix(epoch_tdb: Epoch)
CalcephEphemeris.target16_tdb_minus_tt_s(epoch_tdb: Epoch) [diagnostic only]
load_calceph_ephemeris(ephemeris_file: str | Path, *,
    lunar_relativistic_scale_convention: LunarRelativisticScaleConvention | str,
    longitude_libration_correction_type: LongitudeLibrationCorrectionType | str | None = None)
normalize_longitude_libration_correction_type(value: LongitudeLibrationCorrectionType | str | None)
LongitudeLibrationCorrectionModel.correction_rad(epoch_tdb: Epoch, *, j2000_epoch_tdb: Epoch)
ZeroLongitudeLibrationCorrection.correction_rad(epoch_tdb: Epoch, *, j2000_epoch_tdb: Epoch)
Inpop21aLongitudeLibrationCorrection.correction_rad(epoch_tdb: Epoch, *, j2000_epoch_tdb: Epoch)
make_longitude_libration_correction_model(correction_type: LongitudeLibrationCorrectionType | str | None)
```

### `frames`

```text
PolarMotion fields: xp_arcsec: float, yp_arcsec: float
CelestialPoleOffsets fields: dx_arcsec: float, dy_arcsec: float
EarthOrientationSample fields: mjd_utc: float, xp_arcsec: float, yp_arcsec: float,
    ut1_minus_utc_s: float, dx_arcsec: float = 0.0, dy_arcsec: float = 0.0
EarthOrientationProvider.source_file_path property
EarthOrientationProvider.polar_motion(epoch_utc: Epoch)
EarthOrientationProvider.celestial_pole_offsets(epoch_utc: Epoch)
EarthOrientationProvider.ut1_minus_utc_s(epoch_utc: Epoch)
EarthOrientationProvider.close()
TabulatedEarthOrientation(samples: Sequence[EarthOrientationSample], *, source_file_path: str | Path | None = None,
    duplicate_mjd_policy: DuplicateMjdPolicy = "error")
TabulatedEarthOrientation.from_columns(mjd_utc: ArrayLike, xp_arcsec: ArrayLike, yp_arcsec: ArrayLike,
    ut1_minus_utc_s: ArrayLike, dx_arcsec: ArrayLike | None = None, dy_arcsec: ArrayLike | None = None, *,
    source_file_path: str | Path | None = None, duplicate_mjd_policy: DuplicateMjdPolicy = "error")
TabulatedEarthOrientation.to_mpi_payload()
TabulatedEarthOrientation.from_mpi_payload(payload: Mapping[str, object])
TabulatedEarthOrientation.source_file_path property
TabulatedEarthOrientation.duplicate_mjd_policy property
TabulatedEarthOrientation.mjd_utc_range property
TabulatedEarthOrientation.samples property
TabulatedEarthOrientation.polar_motion(epoch_utc: Epoch)
TabulatedEarthOrientation.ut1_minus_utc_s(epoch_utc: Epoch)
TabulatedEarthOrientation.celestial_pole_offsets(epoch_utc: Epoch)
read_iers_eop(eop_file: str | Path)
load_iers_eop(eop_file: str | Path, *, duplicate_mjd_policy: DuplicateMjdPolicy = "error")
HighFrequencyEopCorrection fields: ocean_delta_xp_arcsec: float = 0.0,
    ocean_delta_yp_arcsec: float = 0.0, ocean_delta_ut1_s: float = 0.0,
    libration_delta_xp_arcsec: float = 0.0, libration_delta_yp_arcsec: float = 0.0,
    libration_delta_ut1_s: float = 0.0, libration_delta_lod_s_per_day: float = 0.0
HighFrequencyEopCorrection.delta_xp_arcsec property
HighFrequencyEopCorrection.delta_yp_arcsec property
HighFrequencyEopCorrection.delta_ut1_s property
ocean_tide_eop_correction(epoch_utc: Epoch, *, background_ut1_minus_utc_s: float)
earth_rotation_libration_eop_correction(epoch_tt_or_tdb: Epoch)
high_frequency_eop_correction(epoch_utc: Epoch, *, background_ut1_minus_utc_s: float)
LunarFrameTransform(ephemeris: Ephemeris)
LunarFrameTransform.pa2lcrs(position_pa_m: ArrayLike, epoch_tdb: Epoch)
LunarFrameTransform.lcrs2pa(position_lcrs_m: ArrayLike, epoch_tdb: Epoch)
ReferenceFrameSystem(ephemeris: Ephemeris, earth_orientation_provider: EarthOrientationProvider)
ReferenceFrameSystem.itrf2gcrs(position_itrf_m: ArrayLike, epoch_utc: Epoch)
ReferenceFrameSystem.gcrs2itrf(position_gcrs_m: ArrayLike, epoch_utc: Epoch)
ReferenceFrameSystem.pa2lcrs(position_pa_m: ArrayLike, epoch_tdb: Epoch)
ReferenceFrameSystem.lcrs2pa(position_lcrs_m: ArrayLike, epoch_tdb: Epoch)
ReferenceFrameSystem.gcrs2bcrs(position_gcrs_m: ArrayLike, epoch_tdb: Epoch)
ReferenceFrameSystem.bcrs2gcrs(position_bcrs_m: ArrayLike, epoch_tdb: Epoch)
ReferenceFrameSystem.lcrs2bcrs(position_lcrs_m: ArrayLike, epoch_tdb: Epoch)
ReferenceFrameSystem.bcrs2lcrs(position_bcrs_m: ArrayLike, epoch_tdb: Epoch)
ReferenceFrameSystem.lcrs2gcrs(position_lcrs_m: ArrayLike, epoch_tdb: Epoch)
ReferenceFrameSystem.gcrs2lcrs(position_gcrs_m: ArrayLike, epoch_tdb: Epoch)
ReferenceFrameSystem.external_gravitational_potential_m2_s2(center_body_name: str, epoch_tdb: Epoch,
    perturbing_body_names: Iterable[str])
RelativisticFrameTransform(ephemeris: Ephemeris)
RelativisticFrameTransform.external_gravitational_potential_m2_s2(center_body_name: str, epoch_tdb: Epoch,
    perturbing_body_names: Iterable[str])
RelativisticFrameTransform.gcrs2bcrs(position_gcrs_m: ArrayLike, epoch_tdb: Epoch)
RelativisticFrameTransform.bcrs2gcrs(position_bcrs_m: ArrayLike, epoch_tdb: Epoch)
RelativisticFrameTransform.lcrs2bcrs(position_lcrs_m: ArrayLike, epoch_tdb: Epoch)
RelativisticFrameTransform.bcrs2lcrs(position_bcrs_m: ArrayLike, epoch_tdb: Epoch)
RelativisticFrameTransform.lcrs2gcrs(position_lcrs_m: ArrayLike, epoch_tdb: Epoch)
RelativisticFrameTransform.gcrs2lcrs(position_gcrs_m: ArrayLike, epoch_tdb: Epoch)
TerrestrialFrameTransform(earth_orientation_provider: EarthOrientationProvider)
TerrestrialFrameTransform.gcrs2itrf_matrix(epoch_utc: Epoch)
TerrestrialFrameTransform.gcrs2itrf(position_gcrs_m: ArrayLike, epoch_utc: Epoch)
TerrestrialFrameTransform.itrf2gcrs(position_itrf_m: ArrayLike, epoch_utc: Epoch)
```

### `observation` 和 `observation_factory`

```text
ObservationResultDetail enum: standard, full
ObservationResultDetail.parse(value: object)
ObservationEquation fields: observed_minus_computed_one_way_m: float, sigma_one_way_m: float,
    design_partials: Mapping[str, np.ndarray], observation_id: Hashable, station_key: str,
    reflector_key: str, transmit_epoch_utc: Epoch, light_time_converged: bool = True,
    wavelength_nm: float | None = None
TroposphereEnvironment fields: pressure_hpa: float, temperature_k: float,
    relative_humidity_percent: float, latitude_rad: float, ellipsoidal_height_m: float, wavelength_um: float
TroposphereEnvironment.troposphere_input(elevation_rad: float)
LightTimeRequest fields: reflector_reference_pa_m: np.ndarray, transmit_epoch_utc: Epoch,
    troposphere_environment: TroposphereEnvironment,
    station_reference_itrf_at_utc: Callable[[Epoch], ArrayLike], station_key: str
LightTimeRequest.station_reference_itrf_at(epoch_utc: Epoch)
LightTimeLeg fields: geometric_range_m: float, gravitational_path_delay_m: float,
    tropospheric_path_delay_m: float, vacuum_elevation_rad: float,
    troposphere_elevation_used_rad: float | None = None, troposphere_elevation_clamped: bool = False
LightTimeLeg.path_length_m property
LightTimeLeg.travel_time_s property
LightTimeSolution fields: transmit_epoch_tdb: Epoch, bounce_epoch_tdb: Epoch, receive_epoch_tdb: Epoch,
    computed_observable_round_trip_time_s: float, tdb_coordinate_round_trip_time_s: float,
    tt_minus_tdb_interval_correction_s: float, pre_1972_utc_rate_offset: float,
    uplink: LightTimeLeg, downlink: LightTimeLeg,
    station_displacement_transmit_itrf_m: np.ndarray, station_displacement_receive_itrf_m: np.ndarray,
    reflector_displacement_bounce_pa_m: np.ndarray, station_bcrs_transmit_m: np.ndarray,
    station_bcrs_receive_m: np.ndarray, reflector_bcrs_bounce_m: np.ndarray,
    iteration_count: int, light_time_converged: bool
LightTimeSolver(frame_system: ReferenceFrameSystem, *, gravitational_delay_model: GravitationalDelay,
    troposphere_delay_model: TroposphereDelay, station_displacement_model: StationDisplacement,
    reflector_displacement_model: ReflectorDisplacement)
LightTimeSolver.solve(request: LightTimeRequest)
LlrObservationEvaluation fields: equation: ObservationEquation, result_row: dict[str, object] | None = None,
    below_elevation_limit: bool = False
LlrObservationModel(frame_system: ReferenceFrameSystem, light_time_solver: LightTimeSolver,
    range_bias_model: RangeBiasModel)
LlrObservationModel.ephemeris property
LlrObservationModel.evaluate(resolved_observation: ResolvedObservation, *, min_elevation_deg: float,
    include_reflector_position_partials: bool = False, result_detail: ObservationResultDetail | None = None)
ObservationProcessingOptions fields: station_identifier: str | None = None, reflector_identifier: str | None = None,
    min_elevation_deg: float = 0.0, include_reflector_position_partials: bool = False,
    show_progress: bool = False, progress_description: str | None = None
ObservationProcessingOptions.catalog_selection property
ObservationProcessingOptions.with_progress(description: str | None, *, enabled: bool | None = None)
LlrObservationProcessor(resolver: ObservationResolver, observation_model: LlrObservationModel)
LlrObservationProcessor.equations(dataset: NptDataset, *, options: ObservationProcessingOptions | None = None)
LlrObservationProcessor.rows(dataset: NptDataset, *, options: ObservationProcessingOptions | None = None,
    detail: ObservationResultDetail = ObservationResultDetail.STANDARD)
ObservationCatalogSelection fields: station_identifier: str | None = None, reflector_identifier: str | None = None
ResolvedObservation fields: normal_point: NptRecord, station_key: str, station: StationRecord,
    reflector_key: str, reflector: ReflectorRecord
ResolvedObservation.transmit_epoch_utc property
ResolvedObservation.station_identity_candidates property
ObservationCatalogState(station_catalog: Mapping[str, StationRecord], reflector_catalog: Mapping[str, ReflectorRecord])
ObservationCatalogState.reflector_positions_pa_m()
ObservationCatalogState.apply_reflector_positions_pa_m(positions_pa_m_by_key: Mapping[str, Sequence[float]])
ObservationResolver(model_state: ObservationCatalogState)
ObservationResolver.resolve(normal_point: NptRecord, catalog_selection: ObservationCatalogSelection | None = None)
ObservationResolver.resolve_all(normal_points: Sequence[NptRecord],
    catalog_selection: ObservationCatalogSelection | None = None)
ObservationAssembly fields: model_configs: dict, station_catalog: Mapping[str, StationRecord],
    reflector_catalog: Mapping[str, ReflectorRecord]
ensure_registered()
resolve_observation_assembly(context, program_config: dict, *, station_catalog = None, reflector_catalog = None)
build_observation_processor(context, program_config: dict, *, station_catalog = None, reflector_catalog = None)
```

### `parametrization`

```text
Parametrization.block_id property
Parametrization.setup(equations: Sequence[ObservationEquation], model_state)
Parametrization.parameter_names()
Parametrization.parameter_count property
Parametrization.design_columns(eq: ObservationEquation)
Parametrization.design_entries(eq: ObservationEquation)
Parametrization.reduce_observation(eq: ObservationEquation)
Parametrization.apply_update(delta: np.ndarray)
Parametrization.max_update_norm(delta: np.ndarray)
Parametrization.state()
Parametrization.initial_update(equations: Sequence[ObservationEquation], *, weight_cap: float, maximum_iterations: int)
ParametrizationList(blocks: Sequence[Parametrization])
ParametrizationList.blocks property
ParametrizationList.setup(equations: Sequence[ObservationEquation], model_state)
ParametrizationList.parameter_names()
ParametrizationList.select_blocks(selectors: Sequence[str])
ParametrizationList.parameter_count property
ParametrizationList.design_entries(eq: ObservationEquation)
ParametrizationList.design_row(eq: ObservationEquation)
ParametrizationList.design_value(eq: ObservationEquation, coefficients: np.ndarray)
ParametrizationList.reduced_observation(eq: ObservationEquation)
ParametrizationList.split(delta: np.ndarray)
ParametrizationList.update_norms(delta: np.ndarray)
ParametrizationList.apply_update(delta: np.ndarray)
ParametrizationList.state()
ParametrizationList.initial_update(equations: Sequence[ObservationEquation], *, weight_cap: float, maximum_iterations: int)
ParametrizationList.matched_parameter_names(eq: ObservationEquation)
ReflectorPositionParametrization(*, reflectors: Sequence[str] | None = None)
ReflectorPositionParametrization.from_config(config: dict, context)
ReflectorPositionParametrization.setup(equations: Sequence[ObservationEquation], model_state: ObservationCatalogState)
ReflectorPositionParametrization.parameter_names()
ReflectorPositionParametrization.design_columns(eq: ObservationEquation)
ReflectorPositionParametrization.design_entries(eq: ObservationEquation)
ReflectorPositionParametrization.apply_update(delta: np.ndarray)
ReflectorPositionParametrization.max_update_norm(delta: np.ndarray)
ReflectorPositionParametrization.state()
StationBiasInterval fields: station: str, start: date, end_exclusive: date | None, name: str | None = None
StationBiasInterval.key property
StationBiasInterval.active_at(epoch: Epoch)
parse_station_bias_intervals(config_value: object)
canonical_station_for_equation(eq: ObservationEquation)
active_station_bias_interval_keys(intervals: Sequence[StationBiasInterval], eq: ObservationEquation, *,
    requested: set[str] | None = None)
StationRangeBiasParametrization(*, stations: Sequence[str] | None = None, per: str = "station",
    intervals: Sequence[Mapping[str, object]] | None = None)
StationRangeBiasParametrization.from_config(config: dict, context)
StationRangeBiasParametrization.setup(equations: Sequence[ObservationEquation], model_state)
StationRangeBiasParametrization.parameter_names()
StationRangeBiasParametrization.design_columns(eq: ObservationEquation)
StationRangeBiasParametrization.design_entries(eq: ObservationEquation)
StationRangeBiasParametrization.reduce_observation(eq: ObservationEquation)
StationRangeBiasParametrization.initial_update(equations: Sequence[ObservationEquation], *,
    weight_cap: float, maximum_iterations: int)
StationRangeBiasParametrization.apply_update(delta: np.ndarray)
StationRangeBiasParametrization.state()
```

### `range_bias`、`relativistic` 和 `time`

```text
RangeBiasRequest fields: station_identifiers: tuple[str, ...], observation_epoch_utc: Epoch
RangeBiasCorrection fields: model_label: str, lookup: RangeBiasLookup
RangeBiasCorrection.correction_two_way_cm property
RangeBiasCorrection.correction_two_way_m property
RangeBiasCorrection.correction_round_trip_time_s property
RangeBiasCorrection.correction_one_way_m property
RangeBiasCorrection.apply_to_computed_round_trip_time_s(computed_round_trip_time_s: float)
RangeBiasModel.evaluate(request: RangeBiasRequest)
ZeroRangeBiasModel.evaluate(request: RangeBiasRequest)
TableRangeBiasModel(bias_table: AdditiveRangeBiasTable)
TableRangeBiasModel.model_label property
TableRangeBiasModel.evaluate(request: RangeBiasRequest)
RangeBiasLookupStatus enum: matched, explicit_zero, station_not_in_table, outside_coverage
RangeBiasComponent fields: station_id: str, start_date_utc: date, end_date_exclusive_utc: date,
    correction_two_way_cm: float, source: str | None = None
RangeBiasComponent.from_config_row(row: object, *, table_source: str | None = None)
RangeBiasComponent.active_on(observation_date_utc: date)
RangeBiasLookup fields: requested_station_identifiers: tuple[str, ...], matched_station_id: str | None,
    observation_date_utc: date, active_components: tuple[RangeBiasComponent, ...], status: RangeBiasLookupStatus
RangeBiasLookup.correction_two_way_cm property
RangeBiasLookup.sources property
AdditiveRangeBiasTable fields: components: tuple[RangeBiasComponent, ...], source: str | None = None
AdditiveRangeBiasTable.first_table_station_id(station_identifiers: Sequence[str])
AdditiveRangeBiasTable.lookup(station_identifiers: Sequence[str], observation_epoch_utc: Epoch)
AdditiveRangeBiasTable.active_components(station_identifiers: Sequence[str], observation_epoch_utc: Epoch)
AdditiveRangeBiasTable.total_correction_two_way_cm(station_identifiers: Sequence[str], observation_epoch_utc: Epoch)
AdditiveRangeBiasTable.coverage_intervals_by_station()
AdditiveRangeBiasTable.from_mapping(config_mapping: Mapping[str, object], *, source_path: str | Path | None = None)
builtin_additive_range_bias_table(name: str)
load_additive_range_bias_table(path: str | Path)
LunarRelativisticScaleConvention enum: tdbCompatibleLunarSurface, alreadyScaled
normalize_lunar_relativistic_scale_convention(value: LunarRelativisticScaleConvention | str)
l_b_minus_l_l_for_convention(convention: LunarRelativisticScaleConvention | str)
TdbTopocentricArguments(ut1_fraction_of_day: float, longitude_rad: float,
    distance_from_spin_axis_km: float, north_of_equatorial_plane_km: float)
TimeScaleConverter()
TimeScaleConverter.utc2tt(epoch: Epoch)
TimeScaleConverter.tt2utc(epoch: Epoch)
TimeScaleConverter.tdb_minus_tt_s(epoch_tdb: Epoch, *,
    topocentric_arguments: TdbTopocentricArguments | None = None)
TimeScaleConverter.tdb2tt(epoch_tdb: Epoch, *,
    topocentric_observer: TdbTopocentricArgumentsProvider | None = None)
TimeScaleConverter.tt2tdb(epoch_tt: Epoch, *,
    topocentric_observer: TdbTopocentricArgumentsProvider | None = None)
TimeScaleConverter.convert(epoch: Epoch, scale: TimeScale | str, *,
    topocentric_observer: TdbTopocentricArgumentsProvider | None = None)
```
