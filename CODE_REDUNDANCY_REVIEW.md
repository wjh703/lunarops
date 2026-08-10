# LunarOps 代码冗余与复杂度 Review

## Review 范围

本次 review 完全以实际代码、测试和运行行为为依据，不使用仓库 `docs/` 目录中的架构说明、迁移说明或设计目标作为判断依据。

外部设计对照主要参考：

- [Orekit ObservedMeasurement](https://www.orekit.org/site-orekit-latest/apidocs/org/orekit/estimation/measurements/ObservedMeasurement.html)
- [Orekit EstimationModifier](https://www.orekit.org/site-orekit-latest/apidocs/org/orekit/estimation/measurements/EstimationModifier.html)
- [GROOPS Config files](https://groops-devs.github.io/groops/html/general.configFiles.html)

结论：当前物理模型和文件解析部分总体比较直接，主要冗余集中在观测对象传递、调整求解、配置解析和资源管理四个外围区域。

## 实施状态

本轮已按 review 结论一次性完成重构，不保留旧接口兼容层：

- adjustment 只保留 dense 重加权路径；固定线性化的 `LlrNormalEquations` 继续使用流式法方程累积。
- 合并 observation model、reducer 和 equation builder 为 `LlrObservationModel.evaluate()`；`ObservationEquation` 只保留估计字段，诊断数据仅在 row 输出时创建。
- 统一参数块的 registry ID，并将 range-bias 初值计算移入对应参数块。
- 配置解析直接使用 `LlrAdjustmentSettings` 四个子设置对象的默认值，settings 也由该对象序列化，消除默认值漂移和手工搬运。
- 最终报告复用已有 covariance 和 sigma0，不再重复求解法方程；报告组装从 solver 主流程移出。
- 串行与 MPI 按业务用途统一返回 row 或 equation；MPI worker 显式持有并关闭 context。
- 删除领域对象的资源关闭转发、动态 factory service locator、无消费者的 wrapper/property/helper。

本次删除旧的 model/reducer/metadata/streaming-adjustment API，调用方需要直接使用新接口；这是有意的非兼容变更。

## 1. 调整求解器承担了过多职责

严重程度：高

[`LlrAdjustmentSolver.run()`](lunarops/estimation/adjustment_solver.py#L508) 长 549 行，同时负责：

- 初始方程计算与粗差剔除
- 不确定度预处理
- bias 初值计算
- IGGIII 权更新
- Helmert VCE
- 参数更新与非线性重线性化
- 收敛判断
- 性能统计
- 最终结果和报表组装

dense/streaming 分支又分别出现在：

- [`_solve_linearized()`](lunarops/estimation/adjustment_solver.py#L197)
- [`_standardized_residuals()`](lunarops/estimation/adjustment_solver.py#L288)
- [`_update_scales()`](lunarops/estimation/adjustment_solver.py#L384)
- [`HelmertVceEstimator.estimate()` 和 `estimate_dense()`](lunarops/estimation/helmert_vce.py#L132)

当前 streaming backend 仍然先持有全部 `ObservationEquation`，只是避免物化完整设计矩阵，并非端到端流式处理。

### 建议

优先根据真实数据的内存基准只保留一种 adjustment backend。当前实际配置使用 dense；固定线性化的 `LlrNormalEquations` 可以继续保留 streaming。

如果两种 backend 确实都需要，应封装为统一的 `Linearization` 对象，让 solver 不再包含多组 `if dense is not None` 分支。

不要只是把 549 行 `run()` 拆成多个私有函数。应先删除双路径和报表拼装，复杂度才会真正减少。

## 2. 观测链路最终退化成超大 metadata

严重程度：高

[`build_observation_equation()`](lunarops/classes/observation/equations.py#L185) 一次性构造 66 个 metadata 字段。`ObservationEquation` 同时负责：

- 估计所需的观测方程
- 输入校验和数组复制
- metadata 冻结
- MPI 传输对象
- 标准和完整结果表格投影

当前数据链路为：

```text
NptRecord
  -> ResolvedObservation
  -> LightTimeRequest
  -> LightTimeSolution
  -> LlrPrediction
  -> ObservationReduction
  -> ObservationEquation
  -> output row
```

其中 `LlrPrediction` 和 `ObservationReduction` 没有真正的外部消费者，只被 processor、equation builder 和测试使用。

### 建议

保留 `ObservationResolver` 与 `LightTimeSolver`，合并以下部分为一次测量评估：

- `LlrObservationModel`
- `LlrObservationReducer`
- `build_observation_equation()`

`ObservationEquation` 只保存估计真正需要的数据：

- residual
- sigma
- partials
- identity
- station/reflector/component keys
- epoch

完整诊断只在 full-output 边界生成，不应永久附着在每个估计方程上。

Orekit 值得借鉴的是“一个 measurement contract，加可选 modifier”，而不是照搬其 Java 类层级。

## 3. 调整配置存在多份事实来源

严重程度：中高

同一组选项分别维护在：

- [`adjustment_config.py` 的配置键集合和解析器](lunarops/estimation/adjustment_config.py#L122)
- [`LlrAdjustmentSettings` 及四个子设置对象](lunarops/estimation/adjustment_settings.py)
- 各子设置对象的 `__post_init__()` 校验（同一文件）
- [`LlrAdjustmentSolver` 的 settings 搬运](lunarops/estimation/adjustment_solver.py#L849)

这已经导致默认值漂移：

- VCE 的 `maximum_stochastic_iterations` 默认值现在只在 `VarianceComponentEstimationSettings` 中定义，为 `8`。

另外，[`stage.apply(options)`](lunarops/estimation/adjustment_config.py#L391) 的返回值被直接丢弃，仅仅通过 `replace()` 构造临时 options 来触发校验。这种隐式校验方式不直观。

### 建议

按照现有配置区段拆成少量嵌套 dataclass：

- adjustment
- initialization
- robust estimation
- VCE

每个配置对象自己实现 `from_config()` 和校验，结果中的 settings 直接由配置对象序列化生成。

不建议仅为这个问题引入 Pydantic；普通 dataclass 足够。

## 4. 参数块存在多套身份系统

严重程度：中

同一个参数块当前可能使用以下名称：

- 注册名称：`reflectorPosition`
- Python 类名：`ReflectorPositionParametrization`
- 带索引名称：`0:ReflectorPositionParametrization`
- 参数类型：`position.x`、`position.y`、`position.z`

[`ParametrizationList.select_blocks()`](lunarops/classes/parametrization/base.py#L116) 使用具体 Python 类名进行 stage 选择，因此重命名类会直接破坏用户配置。

`apply_update()`、`state()` 和 solver 又分别生成不同形式的 block label，当前测试甚至显式依赖 candidate/applied update 使用不同 key 的行为。

### 建议

每个参数块只保留一个稳定的 `id` 或 `type_name`，统一用于：

- registry
- stage 选择
- convergence tolerance
- state 输出
- candidate/applied update 报告

现有 `@register` 已经为类设置 `_registry_type`，可以直接以此为基础收口，不需要新增另一套命名机制。

## 5. 通用 estimator 泄漏 range-bias 特例

严重程度：中

虽然 solver 接收通用 `ParametrizationList`，但核心流程仍通过字符串识别 range bias：

- [`_bias_indices()`](lunarops/estimation/adjustment_preprocessing.py#L113) 判断 `name.type == "rangeBias"`
- [`observation_records()`](lunarops/estimation/adjustment_reporting.py) 再次判断 `name.type == "rangeBias"`
- [`build_observation_equation()`](lunarops/classes/observation/equations.py#L284) 无条件创建 `station_range_bias=[1.0]` partial

但 [`StationRangeBiasParametrization.design_entries()`](lunarops/classes/parametrization/station_range_bias.py#L265) 已经在 partial 缺失时默认使用 `1.0`，因此每个 observation 上保存这个数组是完全冗余的。

[`KNOWN_PARAMETER_TYPES`](lunarops/base/parameter_name.py#L22) 还预先注册了多种尚未实现的参数类型。新增参数块时仍需要修改 base 层白名单，削弱了 parametrization 的扩展性。

### 建议

- 将 range-bias 初值计算移入对应参数块，或作为该参数块的显式初始化步骤。
- 删除 observation 上恒为 `[1.0]` 的 station-range-bias partial。
- 删除尚未实现的预留参数类型。
- 考虑取消全局 parameter-type 白名单，只校验参数名非空、格式合法且唯一。

## 6. 资源所有权被表达了多次

严重程度：中

资源关闭当前同时存在于：

- [`RunContext.close()`](lunarops/config/context.py#L94)
- [`ReferenceFrameSystem.owns_ephemeris`](lunarops/classes/frames/reference_frame_system.py#L18)
- [`LlrObservationModel`](lunarops/classes/observation/measurement.py#L33)
- [`LlrObservationProcessor.close()`](lunarops/classes/observation/processor.py#L85)
- [`close_cached_objects()`](lunarops/parallel/worker_cache.py#L10) 的递归 cache 遍历

生产 factory 始终为 `ReferenceFrameSystem` 传入 `owns_ephemeris=False`，所以程序中调用的 `processor.close()` 实际基本是空操作。

### 建议

确定唯一所有权规则：注入的资源统一由 `RunContext` 或 worker context 管理，领域对象不关闭注入依赖。

MPI worker 应保存创建 processor 时使用的 context，并在 worker 退出时只调用一次 `context.close()`。这样可以删除：

- `owns_ephemeris` flag
- processor/model 两层 close 转发
- worker cache 的递归资源查找

这也更接近 GROOPS 的实际原则：program 保持独立，数据通过文件流转，而不是跨 program 共享复杂可变对象图。

## 7. Factory context 是隐式 service locator

严重程度：中低

[`_ObservationFactoryContext`](lunarops/classes/observation_factory.py#L53) 包装 `RunContext`，同时动态挂载：

- ephemeris
- earth orientation
- frames
- MPI resources
- path resolver
- cache namespace

各 factory 通过 `ctx.ephemeris`、`ctx.frames`、`ctx.mpi_resources` 等隐式属性获取依赖，并使用 `hasattr()` 或 `getattr()` 判断能力。添加一个新模型时，需要了解 context 的隐藏属性和 processor 的构建顺序。

### 建议

如果保留 registry，应传入一个字段明确、带类型的最小 dependency object。更简单的方案是让 observation factory 中的构造函数直接接收所需依赖，不再通过通用 context 查找。

## 8. 可立即清理的明显冗余

以下修改风险较低，可先完成：

1. [`parameter_records()`](lunarops/estimation/adjustment_reporting.py) 会重新求解已经求解过的最终法方程。应直接传入已有的 covariance、delta 和 sigma0。
2. [`LlrObservationModel.evaluate()`](lunarops/classes/observation/measurement.py#L75) 先计算一次 station position 和大地坐标，再复用于对流层环境与诊断输出。
3. uncertainty QC 已经单独保存在 solver 中，又在 preprocessing 和每次 relinearization 时复制进 equation metadata。运行时没有消费者，应删除 metadata 副本。
4. 删除每个 observation 上恒为 `[1.0]` 的 `station_range_bias` partial。
5. [`adjustment_solver.py`](lunarops/estimation/adjustment_solver.py#L22) 直接调用 `adjustment_reporting` 的纯函数，不再保留结果模块转发包装。
6. `ParametrizationList.apply_update()` 与 `state()` 重复实现 block label 生成，应统一为一个实现。
7. `ObservationModelState.from_catalogs()` 只是单行构造转发，没有提供额外语义，可以直接调用构造函数。
8. `NptDataset.n_valid_records` 与 `len(dataset)` 完全等价且无生产调用，可以删除。

## 推荐处理顺序

### 第一阶段：无行为变化的收缩

- 删除转发方法和重复求解
- 删除重复 QC metadata
- 删除恒定 station-bias partial
- 合并 geodetic 重复计算
- 统一 block ID 和 label
- 统一 serial/MPI residual 返回类型

### 第二阶段：收口 adjustment

- 用真实数据确认只保留 dense，或建立统一 backend contract
- 将最终报表组装移出求解循环
- 合并配置默认值、解析和序列化来源
- 将 range-bias 特殊初始化移出通用 solver

### 第三阶段：收缩观测与资源层

- 合并 model、reducer 和 equation builder
- 将完整 diagnostics 移到输出边界
- 确立 RunContext/worker context 的唯一资源所有权
- 删除 processor/model/frame 的 close 转发链

每个阶段都应保持现有数值测试通过，避免同时重构数值算法和对象结构。

## 验证结果

执行：

```bash
.venv/bin/pytest -q
```

重构前结果：

```text
209 passed in 3.30s
```

当前执行完整测试：

```text
328 passed in 3.62s
```

当前环境没有安装 Ruff，因此未执行 Ruff。没有执行真实多 rank MPI 运行。
