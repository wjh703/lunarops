# LunarOps 简洁化重构计划（第三轮）

本计划交给 Codex 实施，执行时以本文为准。

## 0. 现状与目标

- **起点**：`dev` 分支 `ee2b98b`。
  - 库代码（py/pyx/pxi）共 26.0k 行，测试 9.2k 行。
  - `pytest tests` 共 520 项，8 s 内全部通过，ruff 干净。
  - 已完成的工作：`CLAUDE_PLAN.md` 的 phase 0–4、fused Cython RHS、MPI 星历预取、de Sitter 进动。
- **仍然存在的问题**（源自四份只读审计，以及对 tudatpy/GROOPS 的对照）：
  1. 校验分散在各层。同一个值依次被 schema、validator、builder、dataclass `__post_init__` 和消费方检查。例如 `utcOffsetHours` 被检查 8 处，`finite_array` 拷贝出现在逐 NP 的热路径上。
  2. 同一物理有两套实现。动力学的 Python 参考力模型和 Cython kernel 并存，`fused_rhs.py` 遇到不支持的配置会**静默** `return None`，退回慢路径。
  3. 存在多套平行的表示和管线：epoch 数据有三层；`solve`/`solve_uplink` 分开实现；`integrate`/`_integrate_delayed` 分开实现；MPI 有两个调度器和一套握手协议；fileio 中约 10 套手写的 header 解析器。
  4. 输出端存在"自校验"：报告 schema 双向校验、写入后读回再做哈希、写入时复查求解器不变量。
  5. 有一层没有消费方的 GUI/JSON Schema 描述层：`UiHints`、`presentation.py`、`catalog.py`。
  6. 估计链路逐条进行：每个方程生成 Python `(col, val)` 列表，参数化接口有 9 个方法，大量样板代码。
- **目标**：
  - 库代码降到约 18–19k 行（减少约 7k 行）。
  - 除标注【数值】的条目外，输出逐位不变。
  - 残差的逐 NP 耗时从 18.6 ms 降到 10 ms 以下，启动时间从 7.3 s 降到 3 s 以下。

> 文中的 `file:line` 均对应 `ee2b98b`。实施前先用 grep 重新定位，行号漂移后以符号名为准。

## 1. 终态设计原则（参考 GROOPS / tudat）

1. **每个类别只有一个窄接口。** 参照 GROOPS 的 `Ephemerides`，它只有 `position/ephemeris/origin` 三个方法。接口只提供批量签名（`times: ndarray`），不同时保留 scalar 和 batch 两套实现。
2. **schema 与构造写在一起。** 参照 GROOPS 的 `readConfig`：每个类在自己的模块里并列声明 `SCHEMA` 和 `from_config(cfg, ctx)`，注册表只是一张 `(category, type) -> 类` 的表，构造路径只有一条。
3. **只在边界做校验。** 边界只有三处：`ConfigSchema.resolve`、文件 reader、CLI 入口。内部代码信任调用方。参照值：GROOPS 的 `slr/` 加 `parametrizationGravity/` 共 5870 行，只有约 10 处 `throw`。
4. **组合式参数化，按切片写入设计矩阵。** 参照 GROOPS 的 `computeIndices` 与 `designMatrix(A.column(index))`。`ParametrizationList` 只计算一次列偏移，每个块向 `A[:, slice]` 批量写入。
5. **参数是模型的句柄。** 参照 tudat 的 `setParameterValue`：`update(dx)` 直接修改模型状态，不另设一层 model_state 字典做中转。
6. **处理步骤操作一个共享 State。** 参照 GROOPS 的 `slrProcessingStep`：estimate、writeResiduals、selectParametrizations 等步骤都读写同一个 `ProcessingState`。
7. **MPI 只保留少量原语。** 参照 GROOPS 的 `Parallel::forEach/reduceSum`：`broadcast_spec`、`stream_tasks` 加上 `gather`/`reduce`。
8. **不照搬 tudat 的 settings→factory 双层体系和几千行的 switch 工厂，也不照搬 GROOPS 的宏和 try/RETHROW。**

## 2. 给实施者（Codex）的硬性规则

以下规则专门约束"过度防御"倾向，违反任意一条都视为该提交不合格。

1. **不新增防御代码。**
   - 禁止新增内部 `isinstance` 检查、`x is None` 兜底、`max(norm, 1e-30)` 一类的钳位、`np.array(..., copy=True)` 防御拷贝、`setflags(write=False)`，以及对内部数据的 shape/finite 复查。
   - 例外只有三处：config schema、文件 reader、CLI 入口。
2. **不支持的组合直接抛错，不做回退。**
   - 出现不支持的配置时，写一句 `raise ValueError("…原因…")`。
   - 禁止"失败就悄悄改走另一条路径"。
3. **不保留兼容层。**
   - 删除 API 时，同步修改所有调用方和测试。
   - 不留别名、deprecated 包装、`_UNSET` 哨兵，也不写旧格式的读取分支。
4. **一个实现不配一个抽象。**
   - 只有一个实现的 ABC/Protocol 要删除。
   - 例外：测试里有 fake 实现需要它作类型标注，此时保留 Protocol，但去掉 `runtime_checkable` 和 isinstance 检查。
5. **注释只写物理、约定、单位和符号。**
   - 删除"vXX 中曾经……""为兼容……""审查要求……"这类历史叙述。
   - docstring 不复述函数名。
6. **数值不变是默认要求。**
   - 没有【数值】标签的条目，回归必须逐位一致。
   - 带【数值】标签的条目各自单独提交，且必须过 §3 的门限。不得放宽门限，不得修改收敛规则。
7. **不改范围之外的东西。**
   - 不做本计划以外的风格整理、重命名或格式化。
   - 不修改物理模型的选择。例如生产配置中的 `pressureSpin`、`c22Static`，用户还在决定，不要动。
8. **架构约束。**
   - 禁止使用本地 `ProcessPoolExecutor`，所有并行都走 `lunarops/parallel/mpi.py`。
   - ABM 主循环留在 Python，不移入 Cython。
9. **提交粒度。**
   - 每个编号条目（如 3.1）单独提交一次。
   - commit message 写明：改了什么、删减的行数（`git diff --stat`），以及验证结果（逐位一致，或最大差值）。
10. **遇到计划之外的情况就停下。** 如果删除某段代码时发现了计划没有预料到的生产调用方，或者回归没有通过，先停下并记录在 `PROGRESS.md`，不要自行发明兜底来绕过。
    - 细化（2026-11-01）：如果某个"死代码"删除目标实际上有生产调用方，只跳过这一个目标，把它记录到 PROGRESS 的"待修订"列表，同一条目的其余目标和后续条目照常继续。
    - 只有以下两种情况需要停下整个流程：回归未通过；计划描述的改法本身不可行（比如依赖的库行为与计划假设不符）。
11. **resolved config 只能包含 YAML 原生类型**（`str/int/float/bool/None/list/dict`）。类型化对象只在 builder 中构造，不写回 config。原因：resolved config 要参与 scientific fingerprint 计算，并写入报告。见 1.4a。Phase 2 的 schema 改造（2.2、2.3、2.6）同样遵守这一条。
12. **不重排格式。**
    - 修改参数时，保持原有的换行方式：原来是一行的，只要不超过 120 列就继续写成一行。
    - 不得为了"整齐"把调用拆成每个参数一行。
    - 新建模块是为了收拢代码：它必须替代掉更多行数。
    - 每个提交都在 message 里写明净行数；某个条目预期减行却净增时，要在 PROGRESS 里说明原因。

## 3. 验证协议

### 3.1 每个提交都要跑

```bash
cd llrops_project_dev
LUNAROPS_ENABLE_OPENMP=1 .venv/bin/python setup.py build_ext --inplace   # 只在改动了 .pyx/.pxi 时需要
.venv/bin/python -m pytest tests -q
.venv/bin/ruff check lunarops tests scripts
.venv/bin/mypy                     # 错误数不得高于 phase 0 的记录
```

### 3.2 每个 phase 结束时跑回归场景

基线在 phase 0 生成，存放在 `../refactor_work/elegance_baseline/`。

| 场景 | 配置 | 说明 |
|---|---|---|
| R1 轨道+转动 | `configs/lunarops_orbit_rotation_de440_environment_10a.yml`，改为 30 天 | 动力学主路径 |
| R2 纯轨道 | `configs/lunarops_orbit_de440.yml`，改为 30 天 | 非转动路径 |
| R3 残差 | `configs/lunarops_oc_residuals_de440.yml`，取 2019-01 至 2019-03 | 观测主路径 |
| R4 预测 | `configs/lunarops_observation_prediction.yml` | 预测路径 |
| R5 平差（快） | `configs/lunarops_parameter_estimation_de440.yml`，2019-01-01 至 2019-07-01，4 个 MPI rank；`empiricalLibration` 删掉 `tau_F-l_2190d`、`tau_2F-2l_1095d`、`tau_lp_365d`、`Isigma_l_27.6d` 四项；`stationPosition.fixedStations` 改为 `[APOLLO, WETTZELL]`；其余参数和处理步骤不变 | 估计 + MPI，每个 phase 都跑 |
| R5L 平差（长） | 同一配置，2012-01-01 至 2020-01-01，全部参数，16 个 MPI rank（经 Slurm 提交） | 只在 phase 0、5、6 和最终验收时跑 |
| R6 长弧 | `configs/lunarops_orbit_rotation_de440_environment_10a.yml`，完整 10 年，用 OpenMP 32 线程或 `jobs/*_mpi.sbatch` | 只在 phase 0、3、6 和最终验收时跑 |

> 基线修订（2026-11-01，用户已批准）：最初的 R5 是 6 个月弧段、带全部参数，第一次联合估计时报出法方程秩为 35/37。诊断发现，6 个月内 τ 的 2190 d 和 1095 d 项相当于常数加漂移，与 5 个反射器的 x 坐标（对应极轴旋转）线性相关，365 d 项也接近相关。这是场景设计不当，不是代码缺陷。修订后，R5 在短弧段中去掉这三项，R5L 则用长弧段覆盖全部参数。R5L 如果仍然缺秩，就停下来报告，不要再改场景。
>
> 第二次修订（2026-11-01）：只去掉三项 τ 的 R5（commit 4f8794c）已满秩（31/31），但联合估计 20 次迭代仍未收敛，原因有两个，都是 6 个月弧段的病态：
> - `rho_l_27.6d` 和 `Isigma_l_27.6d` 的相关系数为 0.9987，形式误差约 12–16 m，而收敛门限是 0.05 m；
> - WETTZELL 只有 2018 年起的少量 NP，它的 x 坐标与 range bias 的相关系数为 0.9988，在鲁棒权翻转时引起约 2 cm 的极限环。
>
> 已在 /tmp 中实测修订后的 R5：联合估计 16 次迭代收敛，`formalProductEligible: true`，solution、covariance 和 catalog 全部写出，4 个 rank 用时约 152 s。reflector 中间步骤达到 8 次迭代上限的情况在 R5 和 R5L 中都存在，属于中间步骤，不作为通过与否的判据。

- 弧段缩短：CLI 的 `--set` 只能覆盖 `variables:` 段，配置里的时间段是 variable 时用它。否则把派生配置放到 `../refactor_work/elegance_baseline/configs/`。这类派生配置在 phase 0 一次性建好，后续不再修改。
- 比较用 `scripts/compare_outputs.py <baseline> <new>`，默认 `--rtol 0 --atol 0`。
- 不要拿 `/data4/wangjinghuan/refactor_baseline_20261021/` 当本轮基线。de Sitter 进动是在那份基线之后才加入的。

### 3.3 【数值】条目的门限

只有标注了【数值】的条目可以使用以下门限：

- 月球位置差 ≤ 0.01 mm（30 天）；转动角差 ≤ 1e-12 rad。
- 残差差 ≤ 1e-9 m。
- 平差：迭代次数相同，参数差 ≤ 1e-6 σ，σ 的相对差 ≤ 1e-9。

### 3.4 记录

- 每个 phase 结束后，在 `docs/refactoring/PROGRESS.md` 追加一节，内容包括：
  - 提交列表；
  - 行数变化（`cloc` 或 `wc -l`）；
  - R1–R6（以及当期跑了的 R5L）的结果；
  - 墙钟时间；
  - R3 的 cProfile top-20。
- 产物放在 `../refactor_work/elegance_<phase>/`。

## Phase 0：基线（不改代码）

1. 从 `dev` 拉出分支 `refactor/elegance`。
2. 按 §3.2 跑 R1–R6 和 R5L，产物存入 `../refactor_work/elegance_baseline/`。R3 要用更新后的 EOP 重新生成。原来的 `configs/R5.yml` 改名为 `R5_rejected_rankdeficient.yml` 留档，然后新建 `R5.yml` 和 `R5L.yml`。
3. 记录以下数据，写进 `PROGRESS.md`：
   - 墙钟时间；
   - R3 的 cProfile top-30；
   - mypy 错误数；
   - `wc -l` 统计的库行数和测试行数。

## Phase 1：清除内部防御代码（预计 −900 行，数值不变）

纯机械删除。每一类改动单独提交一次。

**1.1 精简 `base/array_validation.py`**
- 删除没有调用方的 `rotation_matrix`，以及多余的空行。
- `parameter_vector` 和 `state_matrix` 改为在调用处直接写 `np.asarray(x, float)`。
- `finite_array` 和 `vector3` 只能在 schema、reader 和公开构造函数里使用。

**1.2 从热路径和内部 dataclass 中移除 `finite_array`/`require_*`/isinstance 等检查**

涉及的位置：
- `classes/observation/light_time.py`：`LightTimeRequest`，约第 81–87 行。
- `classes/displacement/base.py`：两个 `*DisplacementInput`，约第 15–47 行。
- `classes/delays/base.py`：`TroposphereInput`，约第 44–69 行。另外，`light_time.py` 中通过 `dataclasses.replace` 重建 `TroposphereInput` 的写法，改成直接传入仰角。
- `classes/observation/equations.py`：`ObservationEquation.__post_init__`。
- `classes/observation/normal_points.py`：`NptRecord.__post_init__`。其中的 isinstance、函数内局部 import 和反射器代码映射，移到 `fileio/formats/` 导入器。
- `classes/dynamics/rotation.py`：`LunarRotationState`。
- `classes/ephemerides/ephemeris_provider.py`：`BodyState.__post_init__`。
- `classes/frames/earth_orientation.py`：`EarthOrientation.__post_init__`。读 EOP 时它被调用约 17 万次，共耗时 1.5 s。逐行校验改为由 reader 对整张数组校验一次。
- `estimation/normal_equations.py`：`NormalEquations.__init__` 中约 50 行的类型、shape、finite、对称性检查。
- `classes/dynamics/gravity.py`：约第 178–196 行的类型检查。
- `classes/dynamics/lunar_dynamics.py`：约第 41–70 行与 builder 重复的检查。
- `classes/dynamics/integrators.py`：`IntegratorSettings.__post_init__` 只保留"startup 步长为整数比"这一项检查，其余都已由 schema 覆盖。
- `programs/registry.py`：`ProgramSpec.__post_init__`。
- `base/parameter_name.py`：`ParameterName.__post_init__`。

**1.3 删除重复的默认值**

schema 已经填好的默认值，下游一律写 `cfg["key"]`。
- 用 grep `config.get(` 和 `.get("…", ` 找出调用处，审计时统计到 23 处。
- `(config.get("mpi") or {}).get("chunksize", 8)` 出现 3 次，统一改为 `config["mpi"]["chunksize"]`，需要时在 schema 中补默认值。
- `programs/lunar_orbit.py` 里的 `_build_tide` 使用了 `tide_config.get("k20", defaults.k20)` 这类写法；`earthJ2TimeVariation` 的默认值写了两份，硬编码的 `earthRadiusM` 移入 schema。
- `llr_workflow.load_datasets` 重新检查了类型和非空，`build_parametrization` 检查了 `not blocks`。这两处 schema 都已覆盖，删除。

**1.4 `utcOffsetHours` 只在 schema 里校验一次**
- 校验内容：`ge/le` 范围，并且必须是整分钟。
- 删除下游的 7 处重复校验，审计列出的位置包括 `specs.py:205`、`llr_observation_prediction.py:84`、`adjustment_solver.py:85`、`adjustment_reporting.py:172/271`、`prediction.py:126`、`processor.py:40`。
- ~~对于由 validator 解析出的时间，把 `Epoch` 写回 resolved config，builder 不再重复解析。~~ 这一条已作废，改为下面的 1.4a。

**1.4a 修订（2026-11-01）：resolved config 只能包含原生类型**

- 背景：1.4（15efab9）把 `Epoch` 写回了 resolved config，导致 R5 在 `_scientific_fingerprint` 中执行 `yaml.safe_dump` 时报 `TimeScale` 的 RepresenterError。resolved config 有三个用途：交给 builder、计算 scientific fingerprint、写入报告，因此它必须保持纯数据。
- 新增不变量，同时补入 §2，作为第 11 条硬性规则：resolved config 只能包含 `str/int/float/bool/None/list/dict`。类型化对象（`Epoch`、`Path`、dataclass 等）只能在 builder 中由 resolved config 构造，不得写回 config。
- 改法：
  1. `specs.validate_observation_time_config` 只做两件事：整分钟检查（可以把 `utcOffsetHours` 归一化成 float，float 本身就是原生类型），以及试解析 `startTime/endTime` 以便尽早报错。`startTime/endTime` 保持用户输入的原始字符串，不写回。
  2. 在 `programs/specs.py` 新增唯一的入口 `observation_time_window(config) -> tuple[Epoch | None, Epoch | None]`。它调用 `parse_time_with_utc_offset`，validator 和 builder 共用它。
     - 预测程序 `_validate_config` 和 `build_prediction_problem` 改为调用它。
     - `llr_workflow.load_datasets` 中 `utc_offset_hours == 0.0` 与非零两个分支合并为一处调用它。前提是 `filter_time` 接收 `Epoch` 和接收字符串时结果逐位一致；R3/R5 回归会验证这一点，不一致时保留原来的两个分支。
     - 一次运行中同一字符串会解析 2 次，开销可以忽略。
  3. 新增测试：对 `configs/` 下每个生产配置执行 `build_run_plan` 和 `resolve_defaults`，然后对每个程序的 resolved config 断言 `yaml.safe_dump(plain_data(cfg))` 成功。这个测试同时防止今后其他字段再违反这条不变量。
- 验收：
  - R1–R5 逐位一致。
  - R5 报告的 `fingerprint:` 字段必须与 Phase 0 R5 的值完全相同，Phase 0 的值是 `7ca03426c0e4…5038c`，记录在 `elegance_baseline/R5/reflector_bias_processing_report_igg3.txt`。如果 fingerprint 不同，先确认 manifest 中的源码哈希是否随代码改动而变化。源码哈希变化是预期内的，这时只比对 config 部分的哈希，并在 PROGRESS 中说明。`compare_outputs.py` 会忽略哈希类字段，所以这一项要单独比对，写进 PROGRESS。
  - 完成后，Phase 1 的收尾验收视为通过。

**1.5 精简 `classes/time/epoch.py`（471 行，目标约 280 行）**
- 5 个比较方法改用 `functools.total_ordering` 生成。
- 三个 civil 构造函数（`from_isot`、`from_calendar`、`from_date_seconds`）的 TDB 拒绝和秒范围检查，统一收进 `_civil_to_jd` 一处。
  - 第一次修订（2026-11-01）：已实测 ERFA 的 `dtf2d` 接受 TDB，所以原计划"交给 ERFA 报错"的写法不成立。
  - TDB 拒绝继续保留在 `_civil_to_jd` 里，现在就在那里。三个构造函数各自重复的 TDB 判断全部删除。报错文案统一为一条："TDB has no civil constructor; use Epoch(jd1, jd2, 'tdb') or TimeScaleConverter."
  - 秒的检查：`_civil_to_jd` 开头加一行 `if not math.isfinite(second): raise ValueError(...)`。NaN 在 ERFA 中只会产生 RuntimeWarning，不会报错。
  - 越界交给 ERFA：负秒由 `dtf2d` 抛 ErfaError；秒数超过当天长度时抛 ErfaWarning，现有代码已把这个 warning 当作错误处理。`dtf2d` 能识别闰秒日，比现有的 `[0, 61)` 判断更准确。因此删除 `from_isot`、`from_calendar` 里的 `[0, 61)` 检查。
  - `from_date_seconds` 的 `day_length_limit` 检查保留。它决定 `23:59:60+` 的拆分方式，是逻辑分支，不是重复校验。
  - 契约变化：在非闰秒日传入 `second ∈ [60, 61)`，原来能通过，修订后会报错。这一处是更严格、更正确。相应的测试改为断言报错。
- `parse_time_with_utc_offset` 复用 `_parse_isot` 的 offset 解析逻辑。
- `_erfa_call` 不再通过 getattr 取字符串名，改为直接调用。
- 新增内部快速构造 `_from_normalized(scale, jd1, jd2)`：跳过 `TimeScale.parse` 和 finite 检查，但保留归一化。`shifted()` 改为使用它。结果必须逐位一致。

**1.6 删除死代码**

删除前先 grep，确认只有测试在用。对应的测试一并删除。
- `normal_matrix_condition`、`normal_matrix_rank`、`coverage_intervals_by_station`。
- `OceanPoleTideGridInfo` 和 `OceanPoleTideGrid.info`：删除前先 grep 确认没有生产调用方。
- `OceanTidalLoadingCatalogInfo`（第一次修订）：这个类不是死代码。`builtin_registry._station_ocean_tidal_loading` 用 `catalog.info.tidal_model` 检查配置中的模型名和 BLQ 文件头是否一致，这是合理的边界校验，需要保留。
  - 处理方式：删除整个 Info dataclass，把 `tidal_model: str | None` 改成 `OceanTidalLoadingCatalog` 的普通属性，调用处改为 `catalog.tidal_model`。
  - 只被测试使用的 `coefficient_file`、`station_count`、`center_of_mass_correction` 一起删除，测试改为断言 `catalog.tidal_model`，或直接用 `len(catalog)`。
  - 模型名不一致时的报错行为保持不变。
- `registry.register()` 装饰器（第一次修订）：仍有 4 个生产参数化在使用，分别是 `reflector_position.py`、`station_position.py`、`station_range_bias.py`、`empirical_libration.py`。Phase 1 不处理，留到 2.3 统一改为表驱动注册时一起删除。
  - 2.3 中这 4 个类要提供 `SCHEMA` 和 `from_config`，并登记进 `builtin_registry` 的表；之后删除 `register()`、`hasattr(cls, "from_config")` 双路径，以及 `cls._registry_type`。`_registry_type` 的读取方要先 grep 出来，改为读表或读类常量 `TYPE`。
- `UnknownClassError`：如果只在测试中使用，就删除，调用处改为抛 `KeyError`/`ValueError`；如果有生产调用方，就保留。
- `OceanTidalLoadingResult`、`OceanPoleTideResult`、`PoleTideResult`、`PolarWobble`：对应的 `evaluate()` 直接返回 `ndarray`。
- `OrientationProvider.from_scalar`，这是一个恒等 classmethod。
- `classes/dynamics/deformation.py` 中的 `_validated_inertia` 和 `degree2_gravity_coefficients_from_inertia`（在导出列表里，但没有生产调用方）。
- 内部类型不再从包里导出：`LlrObservationEvaluation`、`ObservationAssembly`。
- `classes/dynamics/__init__.py` 当前导出约 45 个名字，收窄到 programs 和测试实际 import 的那些。

**1.7 清理注释和 docstring**

按 §2.5 的规则清理，可以一个提交完成。

## Phase 2：配置、注册表、程序层（预计 −2.0k 行，数值不变）

**2.1 删除 GUI/presentation 描述层**
- 要删除的内容：`UiHints`（`config/schema.py` 约第 29–58 行，共 37 处调用）、`FieldSpec.describe/json_schema`、`config/presentation.py`、`config/catalog.py`、`registry.class_json_schema`、`ProgramSpec.describe/json_schema`，以及 CLI 的 `describe-config` 和 `describe-artifacts` 两个子命令。
- 保留 `describe-program`，只输出字段名、类型、默认值和 description。
- 只服务于描述功能的字段 `allow_variable_reference`、`freeform_identifiers` 也一并删除。删除前先 grep，确认 resolve 逻辑不依赖它们。
- `tests/test_config_contracts.py` 中对应的测试随之删除。

**2.2 简化 FieldSpec 的标志位**
- 现状：`allow_none=False` 显式出现了 175 次，`allow_none=True` 只有 18 次。
- 默认值改为 `allow_none=False`；字符串字段默认要求非空。
- 数值约束统一改用 `gt/ge/lt/le` 关键字，替换 `minimum` 加 `minimum_exclusive` 的组合。
- 合并 `_validate_sequence` 和 `class_list` 中重复的 min/max/non_empty 逻辑。
- 删除 `ConfigSchema.validate()`，它只是 `resolve` 的别名。
- `__post_init__` 里用正则检查 camelCase 的逻辑，改成一个测试：遍历所有已注册的 schema，逐一检查。

**2.3 每个类自己声明 schema，注册表改为表驱动**
- 删除 `classes/builtin_registry.py` 中集中手写的 `ConfigSchema` 和大约 12 个一行的 `_zero_xxx(cfg, ctx)` 包装函数。
- 每个模型类在自己的模块内声明 `SCHEMA = ConfigSchema(...)` 和 `@classmethod from_config(cls, cfg, ctx)`。
- `builtin_registry.py` 只保留一张表 `{(category, type): "module:Class"}`，按需懒加载对应模块。
- 删除 `register()` 装饰器及其中 `hasattr(cls, "from_config")` 的双路径判断，构造只走一条路径。四个参数化类的迁移方式见 1.6 的第一次修订。
- 所有 `type: none` / `zero` 模型统一由表中的同一个 `Zero` 条目处理，删掉逐个编写的 Zero 类。如果某个 Zero 类需要返回特定 shape 的零数组，则保留该类，但不再给它配包装函数。

**2.4 收窄 registry 和 RunContext 的 API**
- `config/registry.py` 只保留 `register_factory`、`resolve(category, cfg)`、`create(category, cfg, ctx)` 三个入口。删除 `_normalize_category`、`_normalize_type_name`、`normalize_class_config`（含其中的 deepcopy）、`validate_class_config`、`get_class_declaration`、`instantiate`。
- `config/context.py`：
  - 删除 `create_class` 的 `factory_context` 和 `cache_namespace` 参数。observation assembly 改为直接传入依赖，与 4.6 一起修改。
  - cache 的所有权规则简化为：谁创建谁关闭。用 `contextlib.ExitStack` 管理关闭流程，删除 `close_resource(s)` 返回的失败字典和 `cleanup_failures` 的 strict 模式。
  - 缓存键改为 `json.dumps(cfg, sort_keys=True, default=str)`，删除 `_canonical_value`。
  - `next_observation_spec_id` 改用 `itertools.count`。

**2.5 精简程序注册表 `programs/registry.py`**
- `ProgramSpec` 改为普通的 frozen dataclass。
- 删除 `RegisteredProgram` 和 `get_program` 里的 isinstance 检查。
- 删除 `_validate_program_artifacts_resolved`（约第 210–275 行），它的作用是运行前预读文件头。每个 reader 已经会检查自己的文件头，这一步只在 `lunarops validate` 中保留。
- `ArtifactSlot.artifact_type` 直接使用文件头里的类型名，例如 `"normalPoint"`。删除 `_TEXT_ARTIFACT_HEADERS` 和 `_BINARY_ARTIFACT_TYPES` 这两张映射表。
- `cmd_validate` 和 `cmd_resolve_config` 合并成一个命令，同时删除 `cli.py` 里对 `processing` 的特判。

**2.6 `processingSteps` 改用 schema 解析**
- 当前情况：
  - `programs/specs.py` 把 `processingSteps` 声明为不带约束的 mapping 序列。
  - `estimation/adjustment_plan.py` 用约 330 行手写的 `_mapping/_number/_string/_reject_unknown` 重新实现了一遍 ConfigSchema。
  - 各 dataclass 的 `__post_init__` 又校验了第三遍。
- 改法：
  - 每种 step 写一个嵌套 `ConfigSchema`，以 `type` 作为判别字段，复用 `class_list`/choice 机制。
  - dataclass 只保留数据，不做校验。
  - 删除 `VarianceComponentDefinition.from_config` 与 `__post_init__` 之间重复的检查。
  - `SelectParametrizationsStep` 的选择器只在 schema 中校验一次，`ParametrizationList.select_blocks` 不再重复校验。

**2.7 零碎项**
- `config/overrides._parse_value` 用 `yaml.safe_load(text)` 一行实现。
- `fileio/catalog_sources.py` 反向 import 了 `programs.specs`，还重复校验了 inline 条目。inline 条目的转换移到 programs 层，fileio 只负责读文件。
- `_catalog_create.create_catalog`（10 个参数）与两个各约 60 行的 catalog 程序合并到一个文件中，或者直接内联。
- `base/station_identity.validate_station_id` 只在 CRD/MINI 导入边界调用。`prefit_gross_threshold` 中逐方程调用的那一处删除。

**2.8 Fingerprint 只计算一次**（第一次修订，2026-11-01）

原计划中有两条是错的，现已撤回。Codex 的审计 `../refactor_work/elegance_phase2/audit_2_8.json` 证实了这一点：
- `_scientific_fingerprint` 的投影**不是**冗余代码。它定义了 fingerprint 的科学载荷：类型化的 variance 设置、只保留 screen/select/estimate 步骤（排除输出步骤，所以改输出路径不会破坏 restart），以及规范化的默认值。删掉它，R5 的 fingerprint 就会变。
- `_schema_paths` 的递归是必要的。改成非递归后，R1 会漏掉 `ephemerides[0].directory` 和 `gravityFields.{earth,moon}.path`。

修订后的做法：
1. **载荷不变，只搬家。** `llr_processing._scientific_fingerprint` 移到 `config/fingerprints.py`，改名为 `processing_scientific_payload(config, plan) -> dict`。`model_compatibility_fingerprint` 的排除键集合也移过去，作为模块常量 `COMPATIBILITY_EXCLUDED_KEYS`。两个 fingerprint 的公开入口都放在 `fingerprints.py` 中，共用同一个 `_digest(payload, manifest)`。`llr_workflow.py` 和 `llr_processing.py` 只负责调用。
2. **保留 `_schema_paths` 的递归。** 只做局部整理，不改变它收集的资源集合。
3. **每个文件最多哈希一次。** 在 `RunContext` 上加一个按 `(resolved path, size, mtime_ns)` 缓存 sha256 的字典，也可以复用现有的 `sha256_file` lru_cache，但要确认缓存键包含 size 和 mtime。所有 manifest 构建（包括源码哈希）都走这个缓存。
   - 两个 fingerprint 的 manifest 资源子集本来就不同，不要强行合并成一个 manifest 对象，共享的是逐文件的哈希。
4. 验收：
   - 用 `audit_2_8.py` 复现：R5 的 fingerprint 必须仍为 `7ca03426…5038c`（manifest 固定为 Phase 0 的版本），R1 的 manifest 资源集合不变。
   - R1–R5 逐位一致。
   - 在 PROGRESS 中记录一次 R5 运行里 sha256 被调用的次数和总耗时，修改前后各记一次。

**2.2a 撤回 2.2 的格式膨胀**（第一次修订，2026-11-01）

2.2（9b74eb4）净增 493 行，几乎全部来自格式变化。原本一行写完的 `number("c21Static", default=..., allow_none=False)`，删掉 `allow_none` 之后反而被拆成了 4–5 行，`config/lunar_models.py` 一个文件就增加了 290 行。
- 处理：凡是在 120 列以内放得下的 `number/boolean/string/...` 调用，一律收回成一行。只改格式，AST 必须与改动前完全一致；Codex 在 1.7 已经用过这种比对方法。
- 预期：减少约 400 行。
- 以后遵守 §2 第 12 条。

**2.9 拆分 `programs/lunar_orbit.py`（842 行，目标约 450 行）**
- `build_lunar_propagation_problem`（约第 348–503 行）拆到 `classes/dynamics/assembly.py`，按 `build_forces / build_rotation / build_tides` 三个函数组织。力项名称和默认 partner 表也移过去。
- `solar_force_terms` 中 `(x or {}).get(..., True)` 这类重复推导的写法改为直接读取 schema 默认值。
- 删除只为检查一个不使用的 import 而写的 scipy 导入检查（约第 349–352 行）。
- metadata 中的大段常量描述字符串提到模块级常量。
- 程序文件只保留编排逻辑和文件写出。输出部分的向量化放在 8.6。

## Phase 3：动力学单一实现（预计 −1.3k 行）

**3.1 Cython kernel 作为唯一的 RHS 实现**（主积分逐位不变）

步骤：
1. 在 `_lunar_rhs.pxi` 的 `LunarRhsKernel` 中新增 `terms(y)`。它把 `_eval`/`_rotation` 中各项累加之前的值写到预分配的缓冲区：
   - 力：`(n_force, 3)`；
   - 力矩：`(n_torque, 3)`；
   - 另外写出 cmb_viscous、cmb_pressure、nu、ω̇ 和 inertia。

   累加本身的代码不动，保证主路径逐位一致。
2. 两处诊断改用 `terms()`：`programs/lunar_orbit.py` 中的力分解诊断（约第 576 行）和转动表（约第 804 行）。
3. `force_models.py`、`torques.py` 中的 Force/Torque 类只保留参数，改成 frozen dataclass，删除 `acceleration()`/`torque()`。`compile_fused_rhs` 读取这些参数记录。
4. `fused_rhs.py` 中 13 处静默的 `return None` 全部改成 `raise ValueError("unsupported: …")`。
5. 删除 `lunar_dynamics.py` 的 `fused_rhs` 参数，以及 `LunarDynamics.accelerations/_load_evaluation/rotation_evaluation` 中的 Python 回退路径。
6. Python 参考物理（当前 `force_models` 中的 acceleration、`torques`、`rotation.evaluate/geodetic_rate`、`forces.relative_acceleration` 等）挪到 `tests/reference_dynamics.py`，作为 `test_fused_rhs.py` 的独立 oracle，不再属于库代码。
7. 删除以下类和函数：
   - `context.py` 中的 `DynamicsEvaluationContext`；
   - `LunarForceGroup.compute_accelerations/_contribution/acceleration_terms`；
   - `LunarTorqueGroup.evaluate`；
   - `RotationEvaluation`、`AccelerationComponents`、`PointMassGravityEvaluation`；
   - `_vector.py`；
   - `_dynamics_core.pyx` 中只供 Python 路径调用的 def 包装：`point_mass_with_cache`、`eih_correction`、`earth_tide_relative_acceleration`。如果 tests oracle 需要这些函数，就在 tests 里用 numpy 重写。

验收：
- R1、R2 轨道表、积分得到的转动状态（四元数、mantle/core ω、四元数范数误差）逐位一致。
- 诊断表（力分解、转动表中的各力矩项、CMB、nu、ω̇、inertia）按"项差 / 所属总量"判定：
  - 力分解：每行每项 `|Δterm|₂ / |Σ全部力项|₂ ≤ 1e-15`；
  - 力矩/ω̇：每行每项 `|Δterm|₂ / |ω̇_mantle|₂ ≤ 1e-15`（力矩列单位已是 per_s2，与 ω̇ 同量纲）；
  - inertia：`|ΔK|_max / |K|_max ≤ 1e-15`；
  - 参考值为 0 的分量结果必须仍为 0。
  - PROGRESS 中同时报告逐元素相对差最大值，仅作记录，不作判据。
- `terms()` 记录分支不得拖慢主路径：R1 连跑 3 次取中位数，相对同机当次跑的 `7fb1774` 中位数不超过 +3%（单次 30 天运行的抖动约 3%：原型 11.80 s，恢复后 12.13 s）。
- 3.1 的 R1/R2 诊断表从此以 kernel 输出为准：验收后把新的诊断表另存为 `../refactor_work/elegance_phase3/diagnostic_reference/`，3.2–3.5 的诊断表对它逐位比较（轨道表仍对 phase 0 逐位比较）。

> 修订（2026-11-01）：原门限"逐元素相对差 ≤1e-15"不可行。旧诊断值来自 numpy 参考实现，新值来自 kernel，两者是同一公式的不同求和顺序。Earth J2 力矩的 `cross(u, K·u)` 中 K 近对角、对角元差 ~1e-4，存在 ~1e4 倍抵消放大，逐元素相对差达 2.1e-12；但绝对差 1.3e-33，占 |ω̇| 的 7e-19。潮汐力最大差占总加速度的 3e-23。见 PROGRESS "Phase 3 — 3.1 diagnostic gate failed"。被拒的原型（`../refactor_work/elegance_phase3/rejected_3_1_terms.patch`）按修订后的判据可通过，可直接以它为起点继续 3.1 步骤 3–7；它的 `force_indices` 动态挂属性要改为 `FusedRhs` 的正式字段。

**3.2 epoch 数据合并为一层**（逐位一致）
- 目前同一份数据有三种形态：`EpochInputs.arrays()` 返回的 camelCase dict、`records()` 产生的逐 epoch 对象 `DynamicsEpochData`/`PointMassGravityCache`，以及 `FusedRhs._load_epoch` 拆开后的参数。三者合并为一个 `EpochBatch` dataclass。
- `EpochBatch` 的字段全部是堆叠数组，不允许为 None；不用的字段填零（kernel 本来就是这样处理的）。
- kernel 新接口为 `kernel.set_epoch(batch, i)`。`_EpochDataPrefetch` 和 DOP853 的 stage 都通过索引访问 batch，不再为每个 stage 创建对象。
- 保留 `SerialEpochSource`/`MpiEpochSource` 的 `.stream()` 接口，这是 MPI 约束要求的。

**3.3 history 合并**（【数值】允许 1 ulp 差异）
- "Moon = Earth + relative，其余天体读星历"这段逻辑现在有两份：`lunar_dynamics.build_history_provider`（约第 109–132 行）和 `forces._delayed_states`（约第 136–144 行）。合并成 `DelayedHistory.states(names, epoch)`。
- 启动阶段使用的星历 history 也实现这个接口。删除 `forces.py` 约第 154 行的 `getattr(history, "relative_state", None)` 鸭子类型分支。
- 合并后启动段的 Moon 位置按 earth+(moon−earth) 计算，会有 ulp 级差异，在 PROGRESS 中注明。

**3.4 去掉状态包装类**
- 删除 `classes/dynamics/state.py` 中的 `MoonRelativeState` 和 `BarycentricEarthMoonState`。`lunar_orbit.py` 中的 3 处使用直接改用数组运算。

**3.5 积分器合并两条路径**（逐位一致，ABM 仍留在 Python）
- `integrators.py` 中的 `integrate` 和 `_integrate_delayed`（约第 196–372 行）重复了 DOP853 启动、prefetch、node buffer、诊断和网格循环。重构成两段：
  - `seed`：后向 warmup，或者显式的刚体 seed；
  - 统一的主循环。
- seed 阶段中逐点调用的 `build_epoch_data`（约第 328 行）改为批量调用，batch 的内容和顺序与原来一致。

验收：phase 3 结束时跑一次 R6（10 年），与 phase 0 的 R6 对比。除 3.3 的 ulp 级差异外应逐位一致；即便有差异，10 年月球位置差也应 ≤ 0.01 mm。

## Phase 4：观测链路精简（预计 −700 行，数值不变）

**4.1 光行时只保留一条实现**
- `light_time.py` 里的 `solve` 和 `solve_uplink` 有约 70 行重复，下行腿（约第 482–516 行）是上行腿 `_uplink_state`（约第 377–400 行）的复制。
  - 抽出一个 `_leg(...)`，接口统一为 `solve(request, *, two_way=True)`。prediction 调用 `two_way=False`。
- 闭包加 tuple 写法的 `_fixed_point` 改成显式 `for` 循环。迭代顺序和收敛判据保持不变，结果必须逐位一致。
- 6 个 dataclass 精简到 3 个：`LightTimeRequest`、`UplinkSolution`、`LightTimeSolution(UplinkSolution)`。

**4.2 测量模型与偏导**
- `measurement.py` 约第 85–129 行：两个偏导函数各自算了一遍上下行单位向量，改为共用。删除 `max(norm, 1e-30)` 一类的钳位。
- `_add_full_diagnostics` 是一个 100 行的手写 dict，改成声明式字段表 `(name, unit, getter)`，同时供残差表和报告使用。
- `processor.equations()` 和 `rows()` 都转发同一组 6 个 kwargs，改为传一个 `ObservationOptions` 对象。
- ~~`LightTimeRequest.from_station` 按测站缓存大地坐标~~ 撤销（2026-11-01）：`itrf_xyz_at(epoch_utc)` 按测站速度外推，大地坐标随历元变化，按站缓存会改变对流层输入（R3 探针：高度差最大 0.420 mm，30° 仰角延迟差 4.97e-10 m）。保持逐 NP 计算，不做任何缓存。见 PROGRESS "stopped at 4.2"。

**4.3 【数值，需用户确认后执行】测站偏导的接收时刻**
- `_station_position_partial_itrf` 用 transmit + TDB RTT 近似接收时刻的 UTC，而 solution 里已经有精确的 receive UTC。改用精确值会导致偏导发生微小变化。
- 这一项单独提交，并且必须过 §3.3 的门限。

**4.4 range bias 收敛到一个类型**
- 目前用了 7 个类型：Request、Correction、Lookup、LookupStatus、ABC、ZeroModel、TableModel。改为只保留 `RangeBiasTable`：
  - `lookup(station, date) -> (cm, components)`；
  - 空表就是 zero model；
  - diagnostics 只在 FULL 输出时生成。

**4.5 合并重复的"测站 + 日期区间"**
- `RangeBiasComponent`、`StationBiasInterval`、`VarianceComponentDefinition` 三处各实现了一遍"测站 + [start, end) 区间"，日期解析也有三份：`_parse_date`、`_iso_date`、`_date_text`。
- 合并为 `base/station_interval.py` 中的 `StationInterval`，提供 `contains(station, date)` 和 `parse`。

**4.6 delays、displacement 与 assembly**
- 删除只有一个实现的 `GravitationalDelay`/`TroposphereDelay` ABC。
- `type: none` 的处理方式：在组合层直接不加入该分量，替代 Zero* 类。
- `classes/observation/assembly.py` 中的 `_ObservationDependencies` 和 factory_context 机制改为显式依赖注入。这一项与 2.4 配套。
- `NptDataset` 不再对元数据做 deepcopy；`filter_time` 在没有过滤条件时直接返回原对象。

## Phase 5：估计链路列式化（预计 −900 行）

**5.1 新增列式 `EquationBatch`，替代逐条 `ObservationEquation`**
- 字段全部是数组：`oc, sigma, station_idx, reflector_idx, epoch_jd1/jd2, t_receive...`。
- 偏导以固定的 `(n, 3)` 或 `(n, k)` 数组字段存放，不再使用"名字 → ndarray"字典。
- MPI worker 直接返回 batch 的分片，主进程用 `np.concatenate` 拼接，传输量会减少。
- 验收：`DenseLinearization` 的 A、l、P 与原实现逐位一致。拼接顺序必须与原来的 `np.vstack` 顺序相同。

**5.2 参数化采用 GROOPS 式最小契约**

```python
class Parametrization(Protocol):
    def setup(self, batch: EquationBatch) -> list[ParameterName]: ...
    def design(self, batch: EquationBatch, out: np.ndarray) -> None: ...  # 写入 A[:, 本块列切片]
    def x0(self) -> np.ndarray: ...
    def update(self, dx: np.ndarray) -> None: ...                          # 直接改写模型状态
    def reduce(self, batch: EquationBatch) -> np.ndarray:                 # 默认: design @ x
        ...
```

- `ParametrizationList` 只负责三件事：计算一次列偏移（类比 `computeIndices`）、分发列切片、`select`。删除 `_ensure_layout` 和 `matched_parameter_names` 的重复计算。
- 线性块（bias、libration）的 `reduce_observation` 由默认实现给出。
- 删除各块里重复的样板：
  - requested 去重（schema 已经做了）；
  - 5 处 `_model_state is None` 的 RuntimeError；
  - `_index_by_key`；
  - 缺少偏导块时抛出的 KeyError。
- `empirical_libration.py` 约第 181–196 行：目前每个方程、每个项都调用一次 ERFA 基本参数，并且在 design、reduce 和报告中各算一遍，共三遍。改成按 epoch 数组一次算好，缓存在 batch 上。
- 【数值】按块写入 A 时，如果求和顺序与原来的逐行 tuple 方式不同，必须过 §3.3 的门限。能保持逐位一致时优先保持。

**5.3 求解器与报告分离**
- 拆分 `adjustment_solver.py` 中约 310 行的 `run()`（约第 241–552 行）：
  - 算法部分只返回 `AdjustmentState(x, sigma, weights, iterations, normals)`；
  - 报告由 `adjustment_reporting.build_report(state, plan)` 单独生成。
- 删除重复的东西：
  - 把 domain 的 6 个字段拷贝到私有属性的代码，以及空转发的 `_domain()`；
  - 每次内迭代 3 次的 `weight_factor_summary`，改为只算 1 次；
  - 重复出现 4 次的 `ObsKey` 别名定义，保留一份。
- `LlrAdjustmentEstimateResult` 与 `LlrAdjustmentResult` 合并。
- `state()` 直接输出 camelCase，删除 `report_state` 里的改名补丁。
- `variance_component_records` 中每个分量都扫一遍全表，改为一次 groupby。

**5.4 残差和报告使用同一套命名**
- 删除 `llr_processing._write_residuals` 中 26 行的 camelCase→snake_case 硬编码映射。
- 统一由 4.2 的声明式字段表生成残差行，报告直接复用这些行。

**5.5 删除输出端的自校验**（约 −700 行）
- `fileio/report_schema.py`（537 行）：写出时直接 `yaml.safe_dump(plain_data(payload))`。reader 只检查 artifact type 头，以及实际会读取的键。
- `structured_reports.validate_processing_report`（约 90 行）：formal product 的判定只在 `LlrAdjustmentResult.require_formal_product()` 中做一次。
- 删除 `config/lunar_models.LUNAR_ORBIT_SNAPSHOT_SCHEMA` 与 `validate_config_snapshot`（约第 263–321 行），以及 `processing_artifacts._validate_processing_state`。
- 保留 `read_processing_report` 的签名，`tests/test_regressions.py` 和 `scripts/run_regression_suite.py` 依赖它。

Phase 5 验收：
- R3、R4 逐位一致。
- R5 和 R5L 的参数、σ、残差逐位一致，或在 5.2 的【数值】门限内。
- 报告 YAML 的内容等价，只允许键的顺序不同。

## Phase 6：MPI 收敛（830 → 约 350 行，数值不变）

目标是精简 `lunarops/parallel/mpi.py`。

**rank 0 端 API**

```python
class MpiRuntime:
    def broadcast_spec(self, kind: str, spec) -> int: ...
    def stream_tasks(self, kind: str, payloads, *, progress: str | None = None) -> Iterator: ...  # 每个 worker 同时最多 2 个 batch 在途
    def map_tasks(self, kind, payloads, **kw) -> list:  # = list(stream_tasks(...))
        ...
    def shutdown(self) -> None: ...
```

**worker 端**
- 提供 `worker_loop()`，状态放在 `WorkerState(specs, contexts, processors, predictors)` 中。
- 收到 spec 后立即构建 processor。初始化耗时用 `comm.gather(elapsed)` 汇总。

**要删除的部分**
- 第二个调度器：手写的 `map_tasks` 实现，以及手写的进度打印（约第 514–596 行）。进度改用 tqdm。
- 初始化握手：`TAG_INITIALIZE_SPEC`、`TAG_READY`，约第 362–442 行和第 467–500 行。
- 内部协议的防御检查：spec ID mismatch、invalid command、duplicate response、结果 shape 的 TypeError。
- 用元组键做的 worker cache，以及扫描这些键的 `_close_worker_contexts`。
- `_handle_prediction` 中的两层 get/None 回退。
- docstring 中关于 v24/v31 的历史叙述。

**要合并的部分**
- `prepare_observation_spec` 与 `prepare_epoch_inputs`（两者逐行相同，约第 274–307 行）合并成一个函数。
- `mpi_observation_equations` 与 `mpi_observation_rows` 合并为 `mpi_observations(..., rows: bool)`。在 5.1 之后，这个函数返回 `EquationBatch` 分片。

**tag**
- 只保留 3 个：`TASK`、`RESULT`、`STOP`。

**测试**
- 重写 `tests/test_mpi_spec_broadcast.py`：只测公开 API 的行为，不测内部协议。

**验收**
- R1 使用 MPI 预取，R5 使用 4 rank，R5L 使用 16 rank，三者都必须逐位一致。
- R6 MPI 版（`jobs/rotation_de440_environment_10a_mpi.sbatch`）与 phase 0 结果逐位一致。

## Phase 7：fileio 统一（预计 −500 行）【文件格式变更，需用户确认后执行】

- 新增一个通用原语 `write_records(path, artifact_type, header: dict, fields, rows)` 及对应的 `read_records(...)`。下列文件改用它：station/reflector catalog（两套近乎相同的实现合并为一套）、`parameter_names`、`parameter_vectors`、`normal_points`、`earth_orientation`。`result_table.py` 可以作为它的实现基础。
- `normal_equations`、`covariance` 的 `info.txt` 改为 YAML，复用现有的 `yaml_artifact`。
- 删除写完立即读回再算 sha256 的自校验（`normal_equations.py` 约第 54–56、101–105 行，`covariance.py` 约第 33–61 行）。
- 合并 `numeric_table.py` 与 `matrix.py` 两种二进制格式：matrix 写成"单列"或"下三角"的 numeric table。
- 影响：旧产物需要重新生成，包括 `output/regression_*` 下的参考文件。`output/regression_20261009` 中存放的是 version-1 的 `normalEquationMetadata`，导致 `run_regression_suite.py compare` 失败。这个问题在本步重新生成参考产物后一并解决。

## Phase 8：性能（每项单独提交）

### 逐位一致的项（可直接做）

**8.1 OceanPoleTideGrid 读取改用 numpy**
- 把 `OceanPoleTideGrid._read` 中逐行解析 26 万行的纯 Python 代码，换成 `np.loadtxt` 或 `np.genfromtxt` 加 `comments`/`usecols`，再 reshape。
- 预期耗时从 3.7 s 降到约 0.3 s。
- 注意：原实现会静默跳过字段数少于 8 的行和无法解析的文本头行。新实现必须给出完全相同的行集合。可以先用 `str.split` 过滤，再一次性交给 `np.array`。

**8.2 calceph 历元数组处理合并**
- 三个批量函数都在逐 epoch 调用 `require_tdb_epoch`，并各自提取一遍 jd1/jd2。合并成一个 `_jd_arrays(epochs)`。
- 删除对 CALCEPH 返回 shape 的 RuntimeError 检查。
- 3.3 之后动力学不再使用标量 LRU 加 RLock 缓存。先 grep 确认观测侧也不再依赖它，然后删除。

**8.3 EOP 插值一次完成**
- 把 `TabulatedEarthOrientation` 中按量逐个调用的 5 次 `np.interp` 和重复的 scale、范围检查（约第 222–262 行），改为一个 `interpolate(epoch)`，对堆叠的各列逐列插值。
- 逐列 interp 的结果不会变。
- `to_mpi_payload`/`from_mpi_payload` 和 `__new__` 技巧一并删除，直接使用 pickle。

**8.4 同一 NP 内的星历和变换复用**
- `lunar_solid_tide.py` 约第 43 行每次调用都新建 `RelativisticFrameTransform`，改为由调用方传入或缓存。
- 同一 TDB 历元下的 Earth/Moon/Sun 状态只查询一次。当前每个 NP 约 92 次星历查询，其中 relativistic transform 约占 33 次。
- 只要传入的历元完全相同，查询结果就逐位一致。
- 目标是把查询降到每个 NP 50 次以下。

**8.5 `lunar_orbit` 输出循环去掉逐行对象**
- 这一步只去掉每行的 `BodyState`/`MoonRelativeState` 构造和两次 `shifted`，算术顺序不变。

### 【数值】项（需用户确认后执行）

每一项单独提交，必须过 §3.3 的门限。不过就回滚，并在 PROGRESS 中记录。

**8.6 `lunar_orbit` 的 RTN 和 EMB 输出完全向量化**
- 约 8.8 万行，预期差异约 1e-16 相对。

**8.7 接收端测站位移复用**
- 光行时迭代到第 2 轮之后，复用上一轮的位移，或者按 1 µs 取整做缓存。
- 测站计算的开销可以减半，预期差异约 1e-12 m。

**8.8 法方程求解**
- 当前做法：`solve_normal_equations` 同时跑 eigvalsh（O(n³)）、Cholesky 和单位阵双求解，矩阵被对称化 3 次。
- 改法：
  - 改用 `scipy.linalg.cho_factor/cho_solve`；
  - 对称化只做 1 次；
  - 条件数只在最终报告时计算；
  - `NormalEquations` 缓存分解结果，solve、sigma、covariance 共用同一份因子（GROOPS 状态机）。
- 注意：PROGRESS.md 记录过类似的求逆改动，当时因未过门限被撤回。这次必须严格按门限执行，不得放宽。
- 依赖：scipy 目前只是 `dynamics` optional extra。如果不打算把 scipy 移进核心依赖，这一项只做"缓存 numpy Cholesky 因子、只对称化 1 次、延后计算条件数"这几步。

**8.9（可选，下一轮再说）跨 NP 向量化光行时**
- 所有 NP 的迭代次数约为 4。
- 本轮只评估可行性，不实施。

### 性能目标

在 R3 上实测：
- 逐 NP 耗时 < 10 ms，当前为 18.6 ms；
- 启动时间 < 3 s，当前为 7.3 s。

## Phase 9：文档与测试收尾

**9.1 文档**
- `docs/ARCHITECTURE.md`、`docs/LUNAR_DYNAMICS_ARCHITECTURE.md`、`docs/PROGRAMS.md`、`docs/INPUTS.md`：按终态改写，删掉已被移除的概念。
- `docs/DEVELOPMENT.md`：其中仍提到已删除的 `_normal_equations_core` 和流式法方程，需要修正。
- `docs/classes_api.md`（492 行）：如果它是手写的 API 清单，就删除，改由 `lunarops list-classes` / `describe-program` 提供同样的信息。
- `docs/DE440_FORCE_MODEL_AUDIT_20261017.md`：移到 `docs/archive/`。
- `docs/LUNAR_ROTATION_DEVELOPMENT.md`（824 行）：保留物理推导和约定，历次调试过程移到 archive。

**9.2 测试**
- 删除只测内部实现细节、且对应 API 已删除的测试。
- 物理参考值、单位/符号、IO 往返和回归类测试必须保留。
- 测试数量可以下降，但下列模块的物理覆盖不能下降：
  - 力、力矩、潮汐、光行时、位移、EOP、参数化偏导（数值差分对拍）。
- 新增 2 个测试：
  - schema 命名测试（见 2.2）；
  - "fused 不支持的组合会抛 ValueError"（见 3.1）。

**9.3 最终验收**
- 跑一遍 §3.1 全套检查、R1–R6 和 R5L。
- 把最终行数、耗时对比表、所有【数值】项的最大差值写入 PROGRESS。
- 用 `uv run -m build` 和 `scripts/verify_distribution.py` 验证分发包。

## 执行顺序与依赖

```text
P0 → P1 → P2 ─┬→ P4 → P5 → P6
              └→ P3 ────────┘
                         → P8(逐位项) → [用户确认] → P7, P8(数值项), 4.3 → P9
```

- P2.4 和 P4.6 要配套修改。
- P5.1 必须先于 P6 的 `mpi_observations` 合并。
- P3.3 必须先于 P8.2 删除标量缓存。

| Phase | 预计删减 | 数值 |
|---|---|---|
| 1 防御代码 | −900 | 逐位 |
| 2 配置/注册/程序 | −2000 | 逐位 |
| 3 动力学 | −1300 | 逐位（3.3 允许 ulp） |
| 4 观测 | −700 | 逐位（4.3 除外） |
| 5 估计 | −900（含 5.5） | 逐位或门限内 |
| 6 MPI | −480 | 逐位 |
| 7 fileio | −500 | 格式变更 |
| 8 性能 | ±0 | 分项 |
| 合计 | 约 −6.8k | |

## 需要用户确认的事项（Codex 执行到这里先停下）

1. **Phase 7 的文件格式变更**：旧产物不再兼容，需要重新生成 `output/regression_*` 的参考文件。
2. **数值项**：4.3、8.6、8.7、8.8 都需要确认。如果 8.8 要引入 scipy，还需要确认是否把 scipy 加入核心依赖。
3. **门限取值**：§3.3 的数值，尤其是残差 1e-9 m、参数 1e-6 σ。
4. **不在本计划范围内**：生产配置中的 `pressureSpin` 和 `c22Static` 选择，由用户另行决定。

