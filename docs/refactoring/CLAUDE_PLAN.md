# LunarOps 全面 Review 与分批重构计划

## Context

`llrops_project_dev/lunarops` 是开发中的 LLR 数据处理与月球动力学代码，约 28.8k 行库代码，另有 9.6k 行测试、7.5k 行脚本，大量由 LLM 生成。本次以独立第三方身份审查，不参考 docs。

### 核心诊断

1. **处处校验。**
   - 共有 1256 处 `raise ValueError/TypeError`，约每 23 行一处。
   - 同一个值通常被 schema、settings、plan、solver、内部 dataclass 依次校验 3–9 次。例如 GM>0 被检查了 9 次，adjustment 配置被校验约 6 次。
   - 热路径上同样如此：每个 NP 调用 `finite_array` 约 1240 次，是 cProfile 中自耗时第一的函数。
2. **存在大量仅剩测试在用、甚至完全无人调用的代码**，例如：
   - 流式法方程整条链路加 `_normal_equations_core.pyx`；
   - 942 行的 `report_serialization.py`，内容只是机械地把 snake_case 键改成 camelCase；
   - scalar 与 batch 两套并行实现；
   - 各种 `xxx is None` 兜底分支。
3. **围绕历次 AI 审查形成了一套附属物**：AST 守卫脚本、按审查日期命名的契约测试、40 篇审查文档、根目录 37 个 sbatch。
4. **性能上的主要浪费是重复计算，不是算法本身慢。**
   - 动力学：Earth figure 力算了 7 个目标却只用 1 个；"仅 2 阶增量"场按全阶跑核函数；RHS 内反复校验。
   - 观测：每个 NP 约 494 次星历查询、42 次 `xys06a`、12 次完整测站位移计算。
   - 报告与指纹：同一份 manifest 每次运行重复哈希 3 次；用纯 Python 的 PyYAML 输出大报告。

### 已与用户确认

- 分批实施，每批跑完全量测试和回归对比。
- 允许浮点舍入级差异：动力学 1e-10 相对误差；残差和参数在 1e-9 m 量级。
- 可以删除：审查契约测试和 AST 守卫、一次性对比与诊断脚本、根目录 sbatch、旧格式和旧版本的读写兼容。

### 统一原则

- **只在边界处校验**：配置 schema、文件读取器、对外的公开构造函数。
- 内部 dataclass 和热路径中不再做 `isinstance`、`finite_array` 复制、重复的 shape 检查。
- 只有一个实现的 Protocol/ABC、纯转发的包装函数、`_UNSET`/`None` 兜底分支，在确认无生产调用方后一律删除。

---

## 阶段 0：建立基线（不改代码）

- 在当前 HEAD (`b87c9ca`) 上用 `scripts/run_regression_suite.py` 跑以下几项，产物存到 `output/baseline_b87c9ca/`：
  - DE440 1 年轨道加转动积分；
  - `configs/lunarops_oc_residuals_de440.yml` 的残差与平差；
  - 一个 prediction 配置。
- 同时记录墙钟时间和 cProfile 前 30 项，作为性能基线。
- 每个阶段结束后：
  - `uv run pytest`；
  - 用回归脚本对比基线（轨道 RMS 差、残差最大差、参数及其 sigma 差）；
  - 单独提交一个 commit。

## 阶段 1：仓库清理（低风险，先做）

- **删除审查契约测试和 AST 守卫**：
  - `tests/test_audit_20261013.py`、`tests/test_second_review_contracts.py`、`tests/test_zero_compatibility_contracts.py`；
  - `scripts/check_current_contract.py`；
  - `.github/workflows/ci.yml` 中的 "Guard current data contracts" 步骤和专门跑 second_review 的步骤。
  - 这些测试里如有覆盖真实物理或 IO 行为的用例（例如 CRD 解析拒绝非法行），挪到对应模块的测试文件里再删。
- **一次性脚本**：
  - `scripts/` 只保留 `run_regression_suite.py`、`verify_distribution.py`、`benchmark_source_gravity.py`，以及被保留测试 import 的模块（`compare_tdb_tt_topocentric.py`、`run_de440_degree2_probe.py` 等）。
  - 其余全部删除，包括已经坏掉的 `analyze_de430_tide_history.py`，它 import 了一个不存在的函数。
  - 引用被删脚本的测试一并删除，例如 `tests/test_de440_residual_audit.py`。
- **sbatch**：根目录 37 个 `run_*.sbatch` 中保留 3–4 个常用的（`run_lunarops.sbatch`、`run_regression_*.sbatch`），移入 `jobs/`，其余删除。
- **uv.lock**：从 `.gitignore` 中去掉 `uv.lock` 并提交它（uv 项目应锁版本）。
- **docs**：带日期的审查和审计类 md/json 共约 25 篇，建议移入 `docs/archive/`。只做这一项，不改内容。

## 阶段 2：删除冗余兜底与死代码（预计 −3.5k～4k 行）

每一项都已用 grep 确认生产代码无调用方；只被测试使用的，同步删掉或改写对应测试。

### estimation/（约 −1.5k 行）

- **`report_serialization.py`（942 行）整体删除。**
  - 由 `adjustment_reporting.py` 和 solver 在构造字典时直接写 camelCase 键。
  - 分量 ID、block ID 这类标识符映射的键保持原样，不能套用通用转换器。
  - `fileio/report_schema.py` 继续作为唯一的输出声明。
- **删除无人使用的流式法方程链路：**
  - `linearization.build_normal_equations_streaming`；
  - `NormalEquations.accumulate_sparse_rows`、`_normalize_sparse_row`、`accumulate`、`add`、`_assert_compatible`、`zeros`；
  - `_normal_equations_core.pyx/.pyi`；
  - `setup.py` 中对应的 Extension，以及 `verify_distribution.py` 中相关的检查。
- **`adjustment_solver.py`：**
  - 删除对构造参数的 `isinstance`/`callable` 检查。
  - 删除 warm-start 的重复校验，挪到 `read_processing_state` 里做。
  - 删除 `dense is None` 守卫，以及永远为 False 的 `_retained_keys is None` / `final_solution is None` 分支。
  - 删除 `observation_domain is None` 分支，它重新实现了一遍 `screen_observations`；`observation_domain` 改为必填参数。
  - 修复 `parametrization.setup()` 被调用两次、names 与 references 顺序错位的问题（`adjustment_solver.py:336,365-367`）。
- **合并以下模块的重复校验**：`robust_weights`、`parameter_convergence`、`sigma_factor_estimator`、`variance_component_groups.from_config`。
  - `robust_weights.py` 收缩为一个函数 `robust_factors(model, k0, k1, z) -> ndarray`。
  - `parameter_convergence` 收缩为一个函数。
- **其余：**
  - `normal_equation_solver.py`：删除与 `NormalEquations` 重复的 shape、对称性、有限性检查。
  - `normal_equations.from_linearized_statistics`：不再构造两次对象。
  - `DenseLinearization.__post_init__`：去掉对 m×n 设计矩阵的复制。

### config/ 与 fileio/ 与 base/（约 −900 行）

- `config/__init__.py` 只保留 docstring。它是 95 行无人 import 的 re-export，还带来急切导入。
- `config/expressions.py`：删除零调用的 `substitute` 和 `evaluate_condition`，合并 `resolve_value` 与 `substitute_resolved`。
- `config/schema.py`：
  - 删除约 110 行针对开发者常量的元校验；
  - 删除从未被设置的字段 `examples`、`item_choices`；
  - 减少反复的 deepcopy。
- `config/registry.py`：
  - 删除 `create_list`、`_registry_category`、`_GLOBAL_SCALAR_FIELDS`，以及只有测试在用的 `create` 和 `class_descriptions`；
  - scope 只保留 shared/非 shared 两态。
- `config/overrides.py`：改为直接用 `yaml.safe_load` 解析标量。
- `adjustment_config.py`：
  - 删除 `_PROCESSING_STEP_KEYS` 全局检查（已被按类型检查覆盖）；
  - 删除到不了的"过时 section"拒绝逻辑（program schema 已先行拦截）；
  - 删除 `defaults.k1 if ...` 这个死表达式。
- **旧格式兼容（用户已同意删除）：**
  - `catalog_sources.py`：删除 `"coordinates"` 别名，以及"native catalog 当作坐标源"的路径。
  - `catalogs.load_*_catalog`：删除 Mapping 和 `"builtin"` 分支。
  - `earth_orientation` reader：删除 `duplicate_mjd_policy`。
- `formats/crd.py`：
  - 删除零调用的 `crd_sessions_to_npt_records`、`looks_like_crd_file`、重复定义的 `CRD_SUFFIXES`；
  - 删除对解析器已保证内容的二次校验；
  - 去掉 `_CrdObservation` 这一中间层。
- `formats/mini.py`：
  - 删除 `_parse_int(required=True)` 之后的 8 个 assert；
  - 删除 `MiniRecord` 的 8 个从未被读取的字段；
  - 由 `parse_mini_line` 直接构造 `NptRecord`，不再经过 MiniRecord→NptRecord→replace 三次对象构造。
- `fileio/matrix.py`：删除只有测试在用的文本编码路径和 `matrix_kind`，约 −100 行。
- 三份 `_open_text` 合并为 `fileio/archive.py` 中的一份。
- `base/`：
  - 删除 `strings_to_names`；
  - 别名函数 `readonly_vector3`、`matrix3x3`、`catalog_vector3` 内联到调用处；
  - `canonical_station_id` 实际只做校验，重命名为 `validate_station_id`。

### classes/observation 及周边、programs、parallel（约 −900 行）

- `light_time.py`：
  - 删除零调用的 `station_position_itrf_m`；
  - 合并 `TroposphereEnvironment` 和 `TroposphereInput`；
  - 去掉 `LightTimeLeg` 及各 solution 类的 `__post_init__` 校验；
  - `require_scale` 只在 `LightTimeRequest` 里检查一次；
  - 删除 `_IterationState` 中的冗余字段；
  - 1968、1972 两个 `Epoch` 改为模块常量。
- `observation_factory.py`：
  - 删除 `_UNSET_CONFIG`、`mpi_resources`、`_PathResolver`；
  - `getattr(ctx, ..., None)` 改为直接访问属性；
  - 16 处重复出现的 `scope="shared"` 用 `functools.partial` 统一。
- displacement、delays、range_bias、parametrization：删除每次调用都执行的 `isinstance`、`vector3`、`finite_array` 和 result dataclass 复制。
  - `range_bias` 中"长度必须为 1 的标识符元组"改为 `station_id: str`。
  - parametrization 只保留 `design_entries` 一套接口，删除 dense 回退和 `design_value`。
- `epoch.py`：删除 `to_datetime`、`isot(converter=)` 分支，三层转发的 `utc2tt` 合并为一层。
- `catalogs.py`：删除 6 个零调用的 geodetic 访问器。
- `normal_points.py`：删除 `observed_round_trip_time_s` 别名，以及逐条记录的 `isinstance`。
- `cli.py`：删除到不了的 `size < 2` 分支。
- `programs/registry.py`：删除 `validate_program_config` 别名、`allowed_keys`、未使用的属性。
- `parallel/mpi.py`：
  - 删除 `as` 别名形式的再导出和恒为真的 `has_workers` 分支；
  - 合并两个 get-processor 辅助函数。
- `classes/__init__.py` 清空。实测仅 `import lunarops.classes.time` 就会连带导入整个观测包，耗时 134 ms。
- `llr_workflow.py`：
  - 删除 `build_processor` 转发函数、重复的 schema 检查、陈旧的 key（`inputFileNormalPoints`、`inputFileAdjustmentState`）；
  - 修复 bug：fingerprint 未排除 `inputFileProcessingState`（`llr_processing.py:70` 已经排除了）。

### classes/dynamics、ephemerides、frames、programs/lunar_orbit.py（约 −600 行）

- `lunar_dynamics.py`：
  - `ephemeris` 改为必填参数；
  - `build_epoch_data` 和 `_make_epoch_data` 改为调用 `build_epoch_data_batch((epoch,))[0]`；
  - 删除 `build_history_provider` 中生产代码到不了的 `cached_states is None` 和 `external_provider` 分支，以及空操作的 `try/except KeyError` 重抛；
  - `rigid_seed_derivatives` 并入 `derivatives(rigid=True)`。
- `force_models.py`：
  - 删除 `NewtonianPointMassForce` 和 `EihPointMassForce` 中重复保存的 `body_parameters`，以及死掉的 `_mu` 回退（约 −30 行）；
  - 删除只有一个元素的 `for index in (MOON,)` 循环；
  - 删除 `ensure_gravity_field` 的类型强转。
- `forces.py`：
  - 删除 `earth_inertial2fixed_matrix_provider` 和 `state_provider`，生产代码从不使用；
  - 删除 `PointMassGravityEvaluation` 和 `PointMassGravityCache` 每次调用时的复制与有限性检查；
  - 单 epoch 版本的 `build_point_mass_gravity_cache` 改为调用 batch 版本。
- `context.py`：`DynamicsEpochData.__post_init__` 不再做重复的规范化、复制和正交性检查。确实需要校验的外部构造，单独提供 `from_external` 构造函数。
- `inertia.py`：删除零调用的 `attitude()`、`angular_velocity()`，只有测试在用的 `evaluate()`，以及恒为假的 normalization 检查。
- `trajectory.py`：删除零调用的 `orientation_states`、只有测试在用的 `matrices` 和 `vector(extrapolate=)`。
- `gravity.py`：删除零调用的 `nonspherical_acceleration`。
- `torques.py`：删除未被引用的 `LunarTorqueModel` Protocol。
- `lunar_orbit.py`：
  - 删除约 60–80 行与 schema 重复的 `isinstance(dict)`、长度、键检查；
  - 删除 `figure_partners` 的大写键回退；
  - 删除 3 处重复的 `__import__("scipy")` 守卫；
  - `dict(all_bodies)` 只构造一次（目前构造了 6 次）。
- `ephemeris_provider.py`：
  - batch 方法纳入 `Ephemeris` Protocol，删除 `getattr(..., None)` 回退；
  - 删除 `BcrsAccelerationProvider`；
  - `calceph.py` 的 scalar 方法改为调用 batch 实现。
- `frames/`：
  - `earth_orientation.py` 删除零调用的 `quality_coverage`、到不了的 `AssertionError`，合并 `__init__` 与 `from_columns` 的重复代码；
  - `high_frequency_eop.py` 删除对内部值的类型和有限性检查；
  - `_iers2010.py` 删除 `_vector3`、`fundarg` 别名、未使用的返回值。
- `rotation.py`：硬编码的 `299792458.0` 改用 `base.constants.C2`。`SECONDS_PER_DAY` 和 `86400` 统一从 `base.constants` 引用。

## 阶段 3：架构调整

1. **类注册表归位。**
   - `observation_factory.py` 实际上是全局类注册表（动力学和配置模块都在 import 它），而且位置不对。
   - 拆分为两部分：
     - `classes/builtin_registry.py`：存放所有 `register_factory`；
     - `classes/observation/assembly.py`：`build_observation_runtime/processor/assembly`，合并现有的 `observation/configuration.py`。
   - 程序导入移到 `programs.registry.ensure_builtin_programs`。`config/fingerprints.py` 不再依赖 `lunarops.cli._import_programs`，以消除 config→cli 的反向依赖。
2. **adjustment 三个模块合一。**
   - `adjustment_config.py`、`adjustment_settings.py`、`adjustment_plan.py` 合并为 `estimation/adjustment_plan.py`。
   - 拆出 `ResidualScreening(max_abs, by_station)` 和 `EstimateControl`；删除 `plan.settings`，它与 `steps[0]` 重复。
   - 字段名直接使用配置名，删除 `to_report_settings` 改名层。目前同一个旋钮有三个名字，例如 `estimateVarianceFactors` / `adjust_sigma0` / `estimate_variance_factors`。
   - `uncertainty_conventions.py` 只有 9 行，并入 `parameter_products.py`。
3. **动力学精简。**
   - Earth 重力极模型分散在 `orientation.py`、`earth_gravity_orientation.py`、`earth_iau2006_orientation.py` 三个文件，合并为 `earth_figure_orientation.py`，内含 `{name: fn}` 表。
   - 潮汐张量和自转张量在 `inertia.py:209-240` 与 `deformation.py:44-72` 各实现了一遍，参数类也重复。统一为一套，由 `inertia` 调用 `deformation` 的张量函数。
   - J2(t) 多项式在 `force_models.py:413` 和 `torques.py:91` 各实现一遍，且数值精度不同。抽成一个函数，采用 `(jd1-ref)+jd2` 形式。
   - 删除 `LunarDegree2GravityCorrectionForce`：把动态 C2x/S2x 直接写入共享的 Moon 场（与 `LunarDeformation` 的做法一致），只跑一次核函数。
   - 三维旋转矩阵统一使用 `erfa.rx/rz`，删除 `forces._passive_rotation_z` 及另外两份欧拉角实现。
   - 去掉硬编码的 EARTH=0、MOON=1 索引与 `body_indices` 字典查找的混用：构造时一次性确定索引。
   - `solar_environment` 在 frozen dataclass 上用 `object.__setattr__` 修改字段，改为由 `prepare()` 返回新对象。
   - `build_lunar_propagation_problem` 是 360 行的长函数，拆为 `_build_forces`、`_build_tide`、`_build_ring`。
4. **frames 与 relativistic。**
   - `ReferenceFrameSystem` 是纯转发 facade（13 个一行委托）。删除它，或只保留几个真正需要的组合方法；`frames/lunar.py` 内联。
   - `frames/relativistic.py` 中 6 个几乎相同的变换方法，改为由两个参数化的私有函数实现（约 −55 行）。
   - `classes/relativistic/` 包与 `frames/relativistic.py` 重名，把该包的约 60 行逻辑并入 `frames/coordinate_scaling.py`。
   - GM 目前有两个来源：观测侧用 `relativistic/constants.py` 硬编码的 GM，动力学侧用 mass catalog。改为统一来源，或至少在报告中显式记录。
5. **命名和接口一致性。**
   - 配置项 `stationName`/`reflectorName` 实际作用是覆盖每个 NP 的测站，而不是过滤。重命名为 `overrideStation`/`overrideReflector`，或改为过滤语义。这是正确性风险，下面单独问用户。
   - 低仰角策略默认值不一致：schema 为 `rejectClamped`，代码为 `allowFlagged`，需统一。
   - `resource_lifecycle.py` 并入 `config/context.py`。
   - `cli.py` 不再导入 IO 层的私有符号 `_PREDICTION_FIELDS`。
6. **配置中从未变化过的旋钮**（`computeResiduals`、`estimateVarianceFactors`、`estimateRobustWeights` 恒为 true），保留但不再为其维护单独的分支测试。是否删除见末尾待定项。

## 阶段 4：性能优化（按收益排序，只做简单的改法）

### 动力学积分（RHS）

1. **Earth figure 力只对 MOON 一个目标求值**（`force_models.py:345-370`）。目前算 7 个目标后丢弃 6 个，J2(t) 增量又全部重算一遍：共 14 个核函数点，实际只需 1 个。J2(t) 改为对 C20 打补丁后只跑一次。
2. **lunar degree-2 增量不再按全阶计算**：用阶段 3 的写入共享场方案，或至少截断为 `(2,3,3)`（`inertia.py:270`）。
3. **`ForceModel.acceleration` 只返回 Moon 的 3 维向量**，不再每个力分配一个 (N,3) 数组（N≈48）。
4. **RHS 内不做任何校验**：删除 body 顺序的三次比较、r/v/μ 的复制与有限性检查、旋转矩阵正交性检查。
5. **`_dynamics_core.pyx` 球谐核：**
   - 归一化因子与位置无关，每个场预计算一次，不再对每个点、每个 (n,m) 调用 `lgamma`；
   - `cos(mλ)`、`sin(mλ)` 改用递推；
   - 把 `pow(sqrt(x),5)` 和 `rho0_2*0.0` 这类表达式提出循环。
   - 另外，非归一化 Legendre 在约 150 阶以上会溢出。目前用不到高阶，记录为已知限制，本次不改。
6. **小项：**
   - Adams 校正中的 `column_stack @ beta` 改为预计算基底；
   - `solar_environment.bodies` 不再每次重建 36 个对象；
   - Kuiper 环用 `OdeSolution` 对一批 epoch 一次性求值；
   - `lunar_orbit` 输出循环向量化，用 `body_state_matrices`。
7. **月球角速度**：目前用 5 次 `orient_unit` 做有限差分。CALCEPH 已返回角速率（`calceph.py:244` 取到 (6,N) 后丢掉了后 3 行），改用解析角速率。这会带来 1e-12 量级的数值变化，属于用户允许范围；同时删除 `attitude_derivative_step_s` 配置项。

### 观测与光时（每个 NP）

1. **每个 epoch 的星历缓存**：对 `body_state_matrix_bcrs` 按 `(jd1, jd2)` 做小型 LRU 缓存，`frames/relativistic.py` 和 Shapiro 改用 batch 查询（约 494 次 → 约 40 次）。
2. **`gcrs2itrf_matrix` 按 UTC epoch 缓存**（最近几个 epoch）：42 次 `xys06a` → 约 4 次。`solid_earth_tide` 对 Sun 和 Moon 只计算一次矩阵。
3. **测站状态计算从 12 次降到约 4 次：**
   - topocentric observer 用参考位置计算 dtdb 参数（对米级位移不敏感）；
   - 不再重复调用 `_station_state_from_tdb`；
   - 循环结束后不再重算 receive、reflector 状态和 TT 转换。
4. **按测站缓存**：HARDISP 按 `(station, 整数 UTC 秒)` 缓存；海洋极潮系数、测站大地坐标按测站缓存。
5. **文件数据按路径缓存**：`cache_namespace=f"observation:{spec_id}"` 导致每个 spec、每个 worker 都重新解析 BLQ 和 0.5° 海洋极潮网格。改为按文件路径缓存（`observation_factory.py:596`）。
6. **`_erfa_call` 不再每次进入 `warnings.catch_warnings()`**，只在解析时使用。
7. **reflector 偏导只在选中 `reflectorPosition` 参数时计算**（`llr_workflow.py:188`），并复用 solver 已有的 Moon 状态。

### 平差、报告与 IO

1. **`scientific_input_manifest` 每次运行只算一次**（目前 `llr_processing.py` 中算了 3 次），并按 `(path, size, mtime_ns)` 缓存 `sha256_file`。
2. **YAML 改用 `yaml.CSafeDumper` 和 `CSafeLoader`**（`fileio/yaml_artifact.py`），报告只走一次 `plain_data` 和一次校验。
3. **solver 内部：**
   - 每次求解只算一次 `eigvalsh`、rank、cond、update_norms（目前每次外迭代约 23 次）；
   - `weight_factors` 和 residual 改为与 `dense.identities` 对齐的数组，不再是字典；
   - 法方程用 `B = A*sqrt(w)[:,None]; N = B.T @ B`（走 syrk，结果天然对称）；
   - Cholesky 只求一次逆，替代 4 次 LU solve；
   - observation domain 不再深拷贝。
4. **`assign_variance_components`**：每个方程只计算一次日期，并先按测站匹配。
5. **CRD**：每个 session 预计算气象 epoch。
6. **prediction 表**：去掉 float→str→float 往返转换。
7. **MPI**：
   - 数据集随 spec 广播一次，之后每次迭代只发索引区间，不再每次迭代重新 pickle 所有 `NptRecord`；
   - prediction 任务直接发送起始索引和步长，不再发送 ISO 字符串。

### 暂不做的

把所有 NP 的光时迭代整体向量化可能带来 10 倍以上提升，但需要重写 solver，会明显降低可读性。等阶段 4 前几项完成、重新 profile 之后再评估。如果要做，从 prediction 开始，因为其 epoch 等间隔且相互独立。

---

## 关键文件

- **estimation**：`adjustment_solver.py`、`normal_equations.py`、`linearization.py`、`report_serialization.py`（删）、`adjustment_{config,settings,plan}.py`（合并）
- **observation**：`classes/observation_factory.py`（拆分）、`classes/observation/light_time.py`、`classes/frames/{relativistic,terrestrial,earth_orientation}.py`
- **dynamics**：`classes/dynamics/{force_models,forces,lunar_dynamics,context,inertia,deformation}.py`、`_dynamics_core.pyx`、`programs/lunar_orbit.py`
- **config 与 IO**：`config/{schema,registry,fingerprints,__init__}.py`、`fileio/{yaml_artifact,structured_reports,matrix}.py`、`fileio/formats/{crd,mini}.py`
- **构建与 CI**：`setup.py`、`.github/workflows/ci.yml`、`.gitignore`

## 验证

1. **每个阶段结束后：**
   - `uv run pytest`（当前 696 个用例；阶段 1、2 删除契约测试和只测死代码的测试后，数量会下降）；
   - `uv run ruff check lunarops`；
   - `uv run mypy`。
2. **数值回归**：用 `scripts/run_regression_suite.py` 与阶段 0 的基线对比。
   - 轨道与转动：位置差的 RMS 和最大值，阈值为相对 1e-10，或 10 年弧段上 <1 mm。
   - 残差：逐 NP 最大差 <1e-6 m。
   - 平差参数：差值 < 1e-6 × 各自的 sigma。
   - 若有超出，逐项定位：只有月球角速度改为解析值这一项预期会有 1e-12 级变化。
3. **性能**：阶段 4 每项完成后重跑基线配置，记录墙钟时间和 cProfile 前 30 项。期望：
   - 轨道 RHS 明显下降（figure 力的核函数点数降为原来的约 1/7）；
   - 单 NP 耗时降为原来的 1/3～1/6；
   - 报告写出时间降为原来的 1/10 左右。
4. **打包**：
   - `uv run -m build` 后用 `scripts/verify_distribution.py` 检查产物，确认去掉 `_normal_equations_core` 后 wheel 正常；
   - 在仅含 `.venv` 的环境中跑一次 `lunarops` CLI 冒烟测试（`lunarops describe`、`lunarops run <config>`）。

## 待定项（实施时按默认方案执行，用户可否决）

- `stationName`/`reflectorName`：默认重命名为 `overrideStation`/`overrideReflector`，保持覆盖语义不变。
- `computeResiduals`、`estimateVarianceFactors`、`estimateRobustWeights` 三个开关：默认保留。
