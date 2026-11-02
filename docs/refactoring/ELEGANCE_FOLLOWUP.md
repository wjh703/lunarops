# LunarOps 简洁化重构：补充修订（第三轮收尾）

本文交给 Codex 实施。起点为 `refactor/elegance` 分支 `269ea29`。

- 规则沿用 `ELEGANCE_PLAN.md` 的 §2（硬性规则 1–12）和 §3（验证协议），本文不重复。
- 除标注【数值】的条目外，输出必须逐位不变；本文没有【数值】条目。
- 回归基线：
  - 轨道、转动：对比 Phase 6 已验收的快照；
  - R3、R5、R5L：对比 Phase 4 的数值参考；
  - R1/R2 诊断表：对比 `elegance_phase3/diagnostic_reference/`。
- 每个编号条目单独提交，commit message 写净行数和验证结果。
- 记录：在 `PROGRESS.md` 追加 "Elegance follow-up" 一节。
- 行号对应 `269ea29`，行号漂移时以符号名为准。

## F1 MPI：主进程异常时不得挂死（正确性，优先做）

现象：
- 消费方在 `MpiRuntime.stream_tasks`（`parallel/mpi.py:175-217`）的迭代中途抛错，例如积分器报错，或 `integrators.py:180` 的 `zip(strict=True)` 不匹配。
- 此时生成器仍被 traceback 引用，它的 `finally`（排空 RESULT、`Waitall`）不会在 `cli.py:91-96` 的 `runtime.shutdown()` 之前执行。
- `shutdown()` 发出 STOP 时，worker 手上还有排队中的 TASK，并且阻塞在发送大块 RESULT 上，mpirun 可能挂住。

改法：
1. `MpiRuntime` 记录当前活跃的 stream。`shutdown()` 先 `close()` 这个 stream，触发其 `finally` 排空，再发 STOP。
2. `broadcast_spec` 被调用时如果有 stream 未结束，`raise RuntimeError("broadcast_spec while a task stream is open")`。这是内部协议约束，不属于防御兜底。
3. `worker_loop` 中，内层 `try` 之外的失败（解包消息、未知 `kind`）目前会让该 rank 静默退出，主进程永远阻塞在 `recv`。把这一步纳入 `try`，并回送错误：
   - 能取到 `taskId` 时回送 `(True, taskId, traceback)`；
   - 取不到时调用 `comm.Abort(1)`。

测试：
- 在 `tests/test_mpi_spec_broadcast.py` 现有 mpirun 测试旁加一个用例：4 rank，消费方读到第 2 个结果后抛错，断言进程在超时（例如 60 s）内以非零码退出，且不挂住。
- 没有 mpirun 时跳过，与现有用例一致。

验收：R1（MPI 预取）、R5（4 rank）逐位一致。

## F2 5.1 收尾：EquationBatch 真正按列使用（预计净减）

现状：
- `measurement.py:202` 每个 NP 调一次 `EquationBatch.single`。
- `EquationBatch.__iter__/__getitem__`（`equations.py:134-143`）被生产代码用来逐行切出单行 batch，每次切行约 16 次花式索引拷贝。

改法：
1. 以下逐行循环改为列运算（布尔掩码、`np.unique(..., return_inverse=True)` 分组、整列 `tolist()`）：
   - `adjustment_solver.py:75-89`（`_equations`）和 `:252`；
   - `adjustment_preprocessing.py:43, 66, 84`；
   - `variance_component_groups.py:43`；
   - `adjustment_reporting.py:30, 78, 155-194`。
2. 删除 `EquationBatch.__iter__`。`__getitem__` 只保留"布尔掩码或索引数组 → 子 batch"一种语义，测试按需调整。
3. `EquationBatch.single` 的逐 NP 构造本条可以保留。processor 输出的拼接点只做一次 `concatenate`（Phase 6 已经如此），不要改动拼接顺序。

约束：
- 筛选、分组后的行顺序必须与原逐行循环完全相同。法方程累加顺序变了就不再逐位一致，所以凡是涉及求和的地方先确认顺序。
- 报告和残差行仍然用 `column.tolist()`/`.item()` 转成 Python 原生类型，与 5.1 的补充要求一致。

验收：
- R5、R5L 的参数、σ、残差，以及 29 个捕获的线性化 A/l/P，都与 Phase 4 逐位一致。
- R5 墙钟不慢于 `269ea29` 的 +3%。

## F3 3.5 收尾：积分器只保留一个主体

现状：`integrators.py` 中 `integrate` 与 `_integrate_delayed`（`:184-329`）重复了以下几段：
- order-1 步数检查（`:217` / `:322`）；
- `_EpochDataPrefetch` 构建（`:229` / `:294`）；
- 初始 `node_diagnostic` 调用（`:253` / `:318`）；
- `LunarTrajectory` 构造（`:258-264` / `:326-329`）。

改法：
- 拆出 `_seed_backward(...)` 与 `_seed_rigid(...)`，二者都返回 `(nodes, history, initial_batch, dop853_steps, trajectory_history)`。
- `integrate` 只按 `deformation is not None` 选择 seed，之后是唯一的公共主体：检查、诊断、`_main_grid`、构造 trajectory。
- 各自特有的检查（delay、前向积分、seed 覆盖范围）留在 `_seed_rigid` 中。

约束：
- ABM 主循环留在 Python。
- 两条路径的调用顺序和 batch 内容保持不变。

验收：R1、R2 轨道表、转动状态、诊断表逐位一致；R6 留到 F10 时再跑。

## F4 删除兼容哨兵与静默回退（§2 规则 2/3）

- `config/context.py:32,64,74`：删除 `_UNSET`，`create_class(category, config, *, cache=True)` 的 `config` 改为必填。生产调用方已全部显式传参，先 grep 确认，测试同步修改。
- `config/context.py:94`：`class_config` 末尾的 `shared_class_configs.get(category)` 改为 `[category]`。如果有合法的"缺省为 None"调用方，就在 schema 中声明默认值，不在这里兜底。
- `classes/time/epoch.py:380-383`：`parse_time_with_utc_offset` 对空字符串静默返回 None，改为由 schema 的非空约束保证，函数内不再处理空串。先 grep 调用方，确认 None 输入只来自 schema 已允许为 None 的字段。如果确实需要"可选时间"，由调用方写 `None if x is None else parse(...)`。
- `programs/processor.py:61,78`：`options or ObservationOptions()` 中 `options` 改为必填。

## F5 删除残留的内部防御代码（§2 规则 1）

逐处删除，每删一处重跑相关测试。不新增任何替代检查。

估计链路：
- `estimation/parameter_products.py:23-82`：删除 isinstance、`copy=True`、`setflags`、finite 和对称性复查，改为纯 dataclass。
  - 文件边界的校验只保留在 `fileio/parameter_vectors.py` 和 `fileio/covariance.py` 的 reader 中；reader 已有的检查不重复添加。
- `estimation/normal_equation_solver.py:49`（不会走到的 `condition is None` 分支）和 `:85-99`（setflags、finite 复查）。
- `estimation/normal_equations.py:50-65`：`x0/meta/units` 的 `is None` 默认值改为由调用方传入。
- `estimation/normal_equations.py:89`：covariance 的第二次对称化。删除前先确认它不影响输出；若影响逐位一致就保留，并在 PROGRESS 中说明。

动力学：
- `classes/dynamics/rotation.py:64`：`finite_array(..., copy=True, readonly=True)` 改为 `np.asarray(total_inertia, float)`。
- `classes/dynamics/rotation.py:24-25`：热路径上逐步执行的四元数 finite/非零检查。
- 下列位置的 `setflags(write=False)` 和防御拷贝：
  - `earth_figure_orientation.py:25,75,83,97,108,110,118`
  - `assembly.py:140,478`
  - `trajectory.py:79-90`
  - `integrators.py:125`
  - `forces.py:42`
  - `epoch_data.py:20`
  - `lunar_dynamics.py:55`
  - `deformation.py:32`
  - `gravity.py:52`
- `classes/dynamics/gravity.py:178-181`：参数的 isinstance 检查。
- `classes/dynamics/forces.py:89-91`：与 `assembly.py:272` 重复的潮汐源检查，保留 assembly 一处。
- `integrators.py:200-202`：`integrate()` 的 shape 检查。
- `tests/test_orbit_problem.py:163`：断言数组只读的测试，随 setflags 一并删除。

星历、坐标系：
- `classes/ephemerides/calceph.py`：
  - 删除 `_require_handle` 及其 6 处调用，直接使用 `self._handle`；
  - `:186` 的 `np.array(..., copy=True)` 改为直接返回；
  - `:168,191,207` 的空长度保护。
- `classes/frames/earth_orientation.py:94` 的 `setflags`。

观测、位移：
- `classes/observation/equations.py:28`：`parse` 中的 isinstance。
- `programs/llr_processing.py:108`：对 `matched_parameter_names` 的 isinstance，改为按输出级别直接格式化。
- `ocean_tidal_loading.py:30-39` 与 `ocean_pole_tide.py:116-117`：确认是否就在文件 reader 中。是就保留；不是就删除。

## F6 删除 schema 已覆盖的重复默认值与类型转换（1.3 / 2.6 收尾）

- `classes/dynamics/assembly.py:85-88, 120, 264, 302, 323, 332, 347-349, 402, 418`：
  - 统一写 `cfg["key"]`；
  - `bodyForceTerms`、`solarJ2`、`maxDegree`、`radiusM`、`name` 缺默认值的，在 schema 中补默认值（可以是 `None`）；
  - 删除 `min_items=1` 已覆盖的非空复查。
- `classes/dynamics/assembly.py:273-286, 317-319` 和 `programs/lunar_orbit.py:48,51`、`programs/specs.py:91-92`：删除对 schema 已保证为数值的字段的 `float(...)` 再转换和 `.get()`。
- `estimation/adjustment_plan.py:12-60`：
  - dataclass 字段不再带默认值（20.0、3、1e-2、1.5、6.0 等），只由 `config/processing.py` 的 schema 提供；
  - 删除 `:160,164,172` 的 `float()`、`:146-148,180` 的 `.get()`，以及 `:100` 的 `robust_weighting or RobustWeightSettings()`；
  - 未知 step 类型补 `else: raise ValueError(...)`。
- `config/processing.py:10-17,21,29-31,67,72,138`：精简 `_numbers`（目前对 None 校验了两次），删除对已校验值的再次 `strip()` 和 `float()`。
- `config/schema.py:49-52` 与 `:236-239`：重复的 null 检查只保留一处。
- `config/fingerprints.py:131,145`：`showProgress`、`mpi` 两个排除键写了两份，改为只传 processing 专有的额外键。

验收：R1–R5 逐位一致；R5 的 config fingerprint 与 Phase 0 的值 `7ca03426…5038c` 相同，按 1.4a 的方法比对。

**F6 修订（2026-11-02，见 PROGRESS "F6 STOP"）**

原文要求给可选字段"补默认值（可以是 `None`）"，这一条是错的。resolved config 会原样写进 `lunar_orbit_metadata.txt` 和报告，"缺省"和"`null`"是两种不同的配置，所以 R1/R2 元数据多出了 `radiusM: null`。另外，`assembly.py:458` 依赖 `"bodyForceTerms" in config` 来报告 `source: explicit/default`，补上默认值会让这个判断失效。

修订后的规则：
- **不新增任何 `default=None`。** 可选且没有默认值的字段（`allow_none=True`，未声明 default），resolved config 中保持缺省。读取处继续写 `cfg.get("key")`，这不算重复默认值，不改。
  - 涉及的字段：`maxDegree`、`radiusM`、`name`、`bodyForceTerms`、`solarJ2`、`k1`、`endExclusive`、`wavelengthMinNm`、`wavelengthMaxExclusiveNm`、`startTime`、`endTime`。
- **只删除真正重复的默认值。** 判定标准是：`.get("key", 字面量)` 中的字面量与 schema 已声明的 default 相同时，改为 `cfg["key"]`。
  - 不满足这个条件的 `.get`，以及默认值依赖其他配置推导出来的情况（例如 `bodyForceTerms` 的 `default_terms`），都保留。
  - `name=entry.get("name", body_id)` 同理保留。
- `adjustment_plan.py` 的 dataclass 默认值：只删除 schema 中已声明相同 default 的字段。如果某个字段在 schema 里没有 default，就保留它的 dataclass 默认值，并在 PROGRESS 中列出。
- F6 原文中的以下改动不受影响，照常做：`float()`/`strip()` 再转换、`min_items=1` 的非空复查、`robust_weighting or ...`、未知 step 类型的 `else: raise`、`_numbers` 精简、schema 重复的 null 检查、fingerprint 排除键合并。
- 可以从留档的 `rejected_F6_defaults.patch` 出发，去掉其中所有新增的 `default=None` 及其对应的 `cfg[...]` 改写。

验收（补充）：
- R1–R5 的完整产物比较（44 项）全部通过，包括元数据中的 config 段，不允许任何键的增减。
- 加一个测试：对 `configs/` 下每个生产配置做 resolve，断言 resolved config 中不存在用户没写、而 schema 也没有声明非 None 默认值的键。实现方式是：遍历 schema，凡是 `default is None` 且用户输入里没有的字段，在结果中必须缺省。

## F7 程序层零碎项（2.5 / 2.7 收尾）

- `programs/registry.py`：
  - `:114-116`：删除 `isinstance(name, str)` 和无用的 `key = name`；
  - `:177-187`：删除内联写死的 External* 和 binary 类型集合，改为 `ArtifactSlot` 上的 `binary: bool` 字段（或 reader 自带属性），只保留一处真相；
  - `:35-69`：`ProgramSpec.__post_init__` 中用 `object.__setattr__` 构造 schema 的写法，改为模块级工厂函数，`ProgramSpec` 成为普通 frozen dataclass。
- `config/overrides.py:182-183,189-191`：内联 `_parse_value`（直接 `yaml.safe_load`），删除 `isinstance(pair, str)` 和 `pairs or []`。
- `cli.py:197`：让 `available()` 始终返回 dict，删除 isinstance 分支。
- `programs/catalog_create.py` 与 `programs/mass_catalog_create.py`：按原计划 2.7 合并到一个文件。如果合并后不减行，就不合并，并在 PROGRESS 中说明原因。

## F8 格式和小型死代码（§2 规则 12）

- `classes/time/epoch.py:83-115, 197-205, 225-233, 267-275, 318-323`：
  - 一个参数一行、但单行能放进 120 列的调用，收回成一行；
  - 修正 `:84-91` 的缩进。
  - 只改格式，AST 必须与改动前完全一致，比对方法同 2.2a。预期减少约 60 行。
- `classes/dynamics/_vector.py`：删除整个文件，`deformation.py:137` 改用 `np.cross`。先写一个临时脚本，对随机输入断言二者逐位一致，然后跑 R1。不一致就保留原文件，并在 PROGRESS 中记录。
- `classes/dynamics/fused_rhs.py:99`：`_tide_step` 返回的矩阵没人使用，删除这个返回值。
- `classes/dynamics/fused_rhs.py:25-34,283` 与 `_lunar_rhs.pxi:195`：`force_indices` 依赖 `_FORCE_TYPES` 的顺序，这个顺序必须与 kernel 中写死的 8 行 `force_terms` 一致。在两处各加一行注释说明对应关系，不改代码。
- `classes/dynamics/history.py:151`：`DelayedHistory.__init__` 没有调用父类初始化而是手动设置 `_ephemeris`，改为 `super().__init__(system.ephemeris)`。
- `classes/dynamics/epoch_data.py:47-49`：删除多余的空行和未使用的 `ephemeris=None` 参数。
- `classes/observation/station_range_bias.py:109-110`：内联的 `[start, end)` 判断改为调用 `StationInterval`（需要的话给它加一个向量化的 `contains`）。

## F9 性能小项（逐位一致）

- **8.1 补完**：`ocean_pole_tide.py:72-84` 仍是逐行 `float()` 加 try/except，耗时 0.79 s，目标 0.3 s。
  - 改为：用 `str.split` 过滤出字段数 ≥ 8 且首字段能解析为数值的行，然后对这些行一次 `np.array(..., float)` 或 `np.loadtxt(usecols=range(8))`。
  - 行集合必须与现实现完全相同，加一个测试断言这一点。
- **5.2 剩余逐行计算**：
  - `station_range_bias.py:108` 每次 design/reduce 都逐行构造 `Epoch` 并转成 `date_iso` 字符串；
  - `station_position.py:104-105` 逐行计算速度对应的年数；
  - `empirical_libration` 的 `_basis` 在 design 和 reduce 中各算一遍。
  - 改为在 `EquationBatch` 上用 `cached_property` 缓存日期列、年数列和 basis。
- **`calceph.py`**：`np.empty` 之后再 `result[:] = values.T` 多了一次拷贝，改为直接返回 `values.T * 1000.0`。前提是乘法顺序和结果完全相同，先验证。
- **`fileio/result_table.py:107-110`**（可选）：每列预先建好一个 formatter，`infer_fields` 不再逐值 isinstance，也不重复 `isfinite`。写出的文件必须逐字节相同。

不在本轮范围：启动时间 < 3 s（目前 6.55 s，瓶颈在 normal point 转换、ASCII IO 和 EOP 预处理）。如果要追，另立计划。

## F10 文档与已知格式问题

- `docs/CURRENT_DATA_CONTRACT.md:37` 还引用已删除的 `fileio/report_schema.py`，改为指向当前实现。
- `fileio/earth_orientation.py:13-21`：EOP 列名匹配不到单位后缀，每列单位都被写成 `"1"`，需要按列传入正确单位（`field_units`）。
  - **这会改变文件头，属于格式变更：先停下，等用户确认后再做。** 获批后，同步重新生成受影响的参考产物。

## 最终验收

- 跑一遍 §3.1 全套检查，以及 R1–R6 和 R5L（R5L 与 R6 经 Slurm 提交）。
- 与上面列出的基线逐位一致。
- 在 PROGRESS 中记录：
  - 各条目净行数，以及库代码总行数（起点 20,716）；
  - R5 墙钟，以及 F9 前后 R3 每个 NP 的耗时；
  - 两项残留计数在本轮前后的变化：`git grep -c 'setflags(write' -- lunarops` 和内部 `isinstance(`（排除 fileio、config、cli）。
