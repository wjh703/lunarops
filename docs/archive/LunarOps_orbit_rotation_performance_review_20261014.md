# LunarOps 轨道与自转积分性能审计及 Cython 优化建议

**审计对象**：`lunarops_current_source_20261014.zip`（目前会话最后一个源码快照）  
**审计日期**：2026-10-14  
**审计范围**：地月轨道传播、月球刚体/核幔自转、形变时滞积分、历表输入、力与力矩评估及既有 Cython 内核。  
**交付方式**：只提供性能分析、整改路径和验证标准；未修改原始项目源码。  
**证据范围**：源码静态审阅、独立编译与微基准、既有配置检查；不等于缺少实际 CALCEPH/ERFA 数据依赖条件下的全链路 10 年传播剖析。

## 1. 结论摘要

**项目已经使用 Cython，不应再重复移植已有内核。** `lunarops/_dynamics_core.pyx` 已以 Cython 实现 `point_mass_symmetric`、`point_mass_with_cache`、`eih_correction`、`earth_tide_relative_acceleration` 和 `nonspherical_gravity_accelerations`；`setup.py:18–32` 已配置编译。当前性能工作的核心是：

1. **长弧/众多摄动天体：** 优先处理 `build_epoch_data_batch()` 中的外部天体—外部天体辅助重力预计算；在一个 Cython 调用中处理一批独立历元，并在完成 CALCEPH 串行批量读取后用 OpenMP 按历元并行。注意必须保持 EIH 所需完整牛顿辅助量，不能以“月球目标不需要”为由删掉源天体间相互作用。
2. **自转/高频 RHS：** 采用 Cython 小型数值核替换 `np.cross`、四元数导数/姿态矩阵和 3×3 惯量求解造成的小数组、高频 Python 调用，特别针对 `LunarRotationDynamics.evaluate()` 和力矩函数。
3. **球谐力：** 优化**现有** Cython 中与历元无关的球谐归一化常数；其次考虑经验证的经度三角函数递推、半径幂递推和局部工作区复用。已完成归一化预计算的隔离微基准，有实测收益但**不是全轨道加速倍数**。
4. **延迟形变：** 将 `interpolate_state_derivative()` 的小矩阵构造/加权多项式移到单一 Cython 核，保持因果和历史节点/导数的一致性。
5. **最后才优化 ABM 外环：** `_compute_abm_step()` 运行每步的预测及两次校正，并额外对最终接受状态计算导数。最后这次 RHS 是必要的状态一致性保障，不能为提速直接删除。

总体原则：**一个公开计算算法 + 少量专业 Cython 数值核；不增加旧版兼容层和静默科学降级路径。**

## 2. 实际调用链与已具备的优化

```text
AdamsBashforthMoultonIntegrator.integrate()
 ├─  DOP853 startup (fixed steps; only startup/seed)
 ├─  LunarDynamics.build_epoch_data_batch(epochs)  # 每批 64 历元
 │   ├─  CALCEPH batched position/velocity, Earth acceleration
 │   ├─  prescribed position / frame / GM
 │   └─  point_mass_symmetric(prescribed subsystem)  [Cython]
 └─  _compute_abm_step(): predictor + corrector iterations
     ├─ LunarDynamics.derivatives(epoch, state, cachedEpochData)
     │  ├─ Python ForceGroup dispatch
     │  ├─ point_mass_with_cache(...)                 [Cython]
     │  ├─ eih_correction(target=MOON)                [Cython]
     │  ├─ nonspherical_gravity_accelerations(...)   [Cython]
     │  ├─ Earth tide compiled kernel + delayed-history queries
     │  └─ integrated rotation (when enabled)
     │     ├─ delayed inertia (when enabled)
     │     ├─ torque group + small NumPy arrays
     │     ├─ 3x3 solve for mantle/core
     │     └─ quaternion derivative
     └─ accepted-state RHS (not removable without changing numerical semantics)
```

已存在并建议保留的优化：CALCEPH `compute_unit/compute_order` 的 epoch-array 批量接口；64 历元的 `_epoch_batches()`；已分离的 `DynamicsEpochData` 与可变 `DynamicsEvaluationContext`；固定历元外部天体牛顿辅助缓存；缓存只在同一候选状态有效的 figure 加速度；编译内核的 `nogil`；长期轨迹历史的局部插值窗；每日诊断输出与高频节点存储分离。这些都是较成熟的设计，不应作为“冗余代码”删除。

## 3. 可复核的性能证据

### 3.1 测试环境与边界

在隔离目录编译从压缩包取出的 `_dynamics_core.pyx`，并建立一个**仅预计算静态球谐归一化数组**的临时试验模块，其他重力方程未变。Python 3.13.5、NumPy 2.3.5、Cython 3.2.4、GCC 14.2.0。每项预热后取 5 次重复计时的中位数。测试脚本与试验产物留在隔离工作目录，不是生产代码或发布补丁。

对球谐阶数 0、1、2、5、6、12、20、点数 0/1/2/8、普通位置与极区/原点边界，共执行 **56 组输入一致性检查**。全部通过；原核与缓存归一化核的最大绝对加速度差为 **2.710505431213761×10⁻²⁰ m/s²**。这只能支持“该隔离优化在上述输入上一致”，不能替代真实长期轨道积分或不同编译器测试。

| 球谐阶数 | 作用点数量 | 原有 Cython 核（µs/调用） | 归一化预计算试验核（µs/调用） | 核级加速比 |
|:--:|:--:|--:|--:|--:|
| 5 | 1 | 3.822 | 3.523 | 1.08× |
| 5 | 2 | 5.016 | 3.992 | 1.26× |
| 5 | 10 | 13.956 | 9.297 | 1.50× |
| 6 | 1 | 4.511 | 3.529 | 1.28× |
| 6 | 2 | 5.997 | 4.403 | 1.36× |
| 6 | 10 | 17.842 | 10.346 | 1.72× |
| 12 | 10 | 48.691 | 27.640 | 1.76× |

**解释**：真实 DE440 配置中的地球场是 5 阶、月球场是 6 阶，典型每次力项所需的作用点远少于 10。因此不能声称总轨道加速 1.7×；应以真实每次 figure 受力点个数、figure 核调用占总 RHS 时间的比例估算。归一化优化还须与时变二阶系数同时验证。

### 3.2 数百天体辅助计算微基准

按当前质量目录解析，**384 个有质量天体**（11 个 `de440_major` + 373 个 `sb441`），包括地球和月球，因此当前分析条件下独立规定天体数为 382。每个历元，完整源—源牛顿相互作用具有 **72,771 个唯一天体对**。10 年示例使用 315,360,000 s、5,400 s 固定步长，共 58,400 个主步骤，源—源对运算的理论数量级为约 42.50 亿次；这只是总操作规模，不是 CPU 性能测量，也不是所有力项调用次数。

在不使用真实历表、以不重合的合成位置和正 GM 构建输入的隔离测试中，现有 Cython 算法测得：

| 总天体数 | 外部—外部预计算（µs/历元） | 候选态的缓存牛顿补算（µs/RHS） | 单目标月球 EIH（µs/RHS） |
|:--:|--:|--:|--:|
| 12 | 1.874 | 1.972 | 1.646 |
| 347 | 355.492 | 8.594 | 5.631 |
| 380 | 429.582 | 9.230 | 6.143 |

以上时间不包含 CALCEPH、历史插值、球谐、自转与 Python 力项组装。它说明大质量目录下的外部—外部辅助计算确实值得单独优化/并行，也说明 **重新用 Cython 写“EIH 月球一行”未必是第一优先级**。

### 3.3 小数组开销的独立提示

对随机 SPD 3×3 矩阵与 3 分量向量进行了无物理意义的孤立调用计时：`np.linalg.solve(3×3)` 约 3.76 µs，已经预存的逆矩阵乘向量约 0.99 µs，`np.cross(3D,3D)` 约 14.41 µs；标量 Python 三维叉积约 1.05 µs。此组数据只表明**小型运算的 Python/NumPy 接口开销不可忽视**；生产代码不得据此将时变、病态矩阵求解无条件替换为显式求逆。对于已知正对角、固定的月核惯量，直接按各分量除以对角元则有明确的数学依据。

## 4. 优先级 P1：外部天体辅助系统按历元批量 Cython/OpenMP 并行

**定位**：`classes/dynamics/lunar_dynamics.py:100–114,125–179`；`classes/dynamics/forces.py:309–359`；`_dynamics_core.pyx:10–136`；`classes/dynamics/integrators.py:180–185`。

**现状**：`build_epoch_data_batch()` 已批量获取 CALCEPH 输入，但仍逐历元 `_make_epoch_data()`，每次独立调用 `_dynamics_core.point_mass_symmetric()` 构造规定天体之间的加速度与势。对于几百个外部小天体，这一计算至少是 O(N²)，而不同网格历元的源—源计算彼此独立。

**建议改变的不是物理模型，而是数据执行方式**：

- 使用形如 `(batch, prescribedBodyCount, 3)` 的连续位置张量、`(batch, bodyCount, 3)` 的输出张量；保持相同 GM、点对顺序和公式。
- 编写一种规范的 `point_mass_symmetric_batch` Cython 核（名称建议），在单次 Python→Cython 边界调用中处理完整批次；先验证单线程输出与既有逐历元 Cython 输出。
- 完成 CALCEPH 句柄锁保护下的批量读取**之后**，再对**不同历元**进行 `prange` 并行。每个线程独占其历元的输出切片和累加变量，按天体对的顺序仍然串行、确定性累加，避免 `prange` 直接跨对而造成写冲突和浮点求和顺序变化。
- 对 1/2/4/8 线程、batch=16/32/64/128 的组合真实测量；默认线程数不能盲取全部逻辑核数，须与 BLAS、MPI、多任务并行协调。无收益时选择单线程的同一正式实现，而不是保留第二套旧代码。

**科学约束**：EIH 使用每个摄动天体的牛顿辅助加速度与势，不能简单删除规定天体彼此作用，也不能直接用历表的总加速度替代这些低阶项（将混入不同阶次/力项）；新缓存必须与候选状态缓存保持一致。与之相关的 64 历元批量策略已存在，应利用而非另造重复缓存层。

**优先级判断**：数百天体长弧可能有较高收益；只有主要天体、很短积分或 CALCEPH 时间占绝大部分时收益有限。最终以各工作负载热点占比决定。

## 5. 优先级 P1：改进现有球谐 Cython 核，而不是重复移植

**定位**：`_dynamics_core.pyx:331–431`；`classes/dynamics/gravity.py:212–248`（具体读取前以源码为准）；`force_models.py:319–345`。

**已证实的冗余**：每次求值都为每个位置、每个 `(n,m)` 计算 4π fully-normalized 系数，包括 `exp(lgamma(...))`；但这个归一化常数只依赖阶数 `(n,m)`，与 epoch、体轴坐标、时变 Cnm/Snm 均无关。

**修改方向**：

1. 在创建或切换球谐核阶数时一次性预计算归一化矩阵（或稳定的完全归一化递推系数），存于只读 C 连续缓冲；实际重力函数只读。更新月球动态二阶系数时不必重算归一化。
2. 将 `cos(m*lon)/sin(m*lon)` 每个 `(n,m)` 的重复调用改为一次 `sin(lon)/cos(lon)` 后有界阶次递推，注意相位定义 `csphase=1` 与球谐全归一化约定不能改变。
3. 将 `pow(R/r,n)` 改为连续阶次幂递推，并将密集矩阵清零工作限制在实际使用的 `(n,m)` 区域；对 5/6 阶这种小阶数，优先避免优化带来复杂维护成本。
4. 时变二阶校正继续以动态减静态的增量计算，不能叠加两次静态 C20；刚体与动态变形字段共享同一个物理重力场对象的既有语义必须保持。

**验收**：0/1/2/5/6/12/20 阶、稀疏项、所有 5 个二阶系数、接近极轴、原点拒绝或定义行为、超大/超小半径、时变系数、动态月自转耦合。应比较加速度及其轨道积累影响，不只比较单次 RHS。关闭 `-ffast-math`，不允许静默改变系数归一化/相位。

## 6. 优先级 P1：自转小型数值核与惯量求解

**定位**：`classes/dynamics/rotation.py:21–44,143–211`；`classes/dynamics/torques.py:26–28,42–98`；`classes/dynamics/lunar_dynamics.py:279–290`。

**现状**：四元数归一化和矩阵构造、`np.cross`、`np.linalg.norm`、3×3 小矩阵乘法、`np.linalg.solve`、临时 `np.zeros()`/`np.array()`、`RotationEvaluation` 对象和逐力矩 `dict`，在每次校正的 RHS 都重复执行。

**建议**：

- 用 Cython 静态大小数组（局部 `double[3]` / `double[3][3]` / `double[4]`，或连续 typed memoryview）实现**一组**自转 RHS 小核，内部同时求叉积、四元数导数、惯量乘法和月核/幔角加速度；每次 RHS 返回小向量，不再逐个产生 Python 临时数组。
- **固定对角的月核惯量**在构造器中已严格确定，直接分量除以经过正定性检查的对角元；无需重复调用 `np.linalg.solve(core_inertia, ...)`。
- **刚性固定月幔惯量**可以在建模时一次性分解（如正定 Cholesky），并重用；**时变形变的月幔惯量**必须根据每个候选状态重求解，建议 Cython 专用数值稳定的 3×3 SPD 分解及正定性检查，不要缓存旧惯量的逆矩阵。
- 力矩组的“正常传播快路径”只返回三分量总力矩；只有观测/诊断采样时再构造按模型名分类的 `terms` 字典。可类比 `LunarForceGroup.compute_accelerations(collect_terms=False)` 的已有做法。
- `PointMassGravityTorque`、`EarthFigureTorque`、`LunarFigureTorque` 可以减少当前每项的多次 3D NumPy 调用；但需要复核坐标系、作用/反作用与力臂定义，并保留已存在的候选态 figure 缓存。

**注意**：姿态四元数只在矩阵求值时归一化，不能为性能方便而修改接受节点的积分状态；核幔压强转矩 `pressureSpin=core/mantle` 的物理分支与DE430/DE440约定不能改。

## 7. 优先级 P1/P2：延迟变形和 Hermite 插值的 Cython 化

**定位**：`classes/dynamics/history.py:16–54,139–230`；`classes/dynamics/deformation.py:44–109`；`classes/dynamics/integrators.py:324–410`。

**现状**：`interpolate_state_derivative()` 为每次调用分配 `count×count` 广播数组，构造逆差、基函数与导数，用多次短向量乘法求同一 Hermite 多项式的状态和导数。`DelayedHistory.integration_sample()` 在形变每次 RHS 中高频使用，还要处理启动历史和本体动力学历史的分段。

**建议**：将固定 13 阶以内的 Hermite 权重与导数在 Cython 的小工作区计算一次，再对整个 13/16 维状态批量加权；共用权重计算，杜绝多份短小 NumPy 临时数组。`_hermite_grid(count)` 已有缓存，直接沿用。外部延迟状态也可在复用同一个插值核前检查时间窗与轴对应关系。

**不能改变**：同一 Hermite 多项式同时定义 `state` 与 `state_dot`；`seed` 和正式前向积分的历史窗不得跨界；月球形变使用过去接受节点、存在潮汐负延迟（会产生超前平移查询）的规则不能被“统一缓存”误改；数值误差在极近网格节点与插值区间边界均需验证。

## 8. 优先级 P2：避免 Python 力项装配中的 O(N) 小对象

**定位**：`classes/dynamics/force_models.py:86–101,204–265,319–345`；`classes/dynamics/lunar_dynamics.py:261–277`；`classes/dynamics/context.py:93–135`。

`LunarForceGroup.compute_accelerations()` 在普通 RHS 中为每项分配 `(N,3)` 的数组，随后只抽取月球行。由于项目主要只积分 Moon（Earth 与其他天体的轨道来自历表），可以为生产 RHS 使用“目标 Moon 三分量加速度”单一计算接口，诊断路径再分解来源；EIH 仍保留全部天体的下阶辅助量。提前在构造阶段编译必要的整数索引、力项清单与对象句柄。尽量避免在 RHS 内反复构造 `np.asarray([moon_index])`、字典、`np.zeros_like(positions)`、`np.concatenate(parts)`。

先做 profiling 和内存分配火焰图；**该修改涉及接口变化较多，必须以实测收益为依据，且不可删除 Newtonian/EIH 的全体辅助数据**。

## 9. 优先级 P2：历表与积分器层面的机会/限制

- **CALCEPH 已经向量化。** `ephemerides/calceph.py:163–185,202–219` 已对 epochs 使用 `compute_unit` / `compute_order`，受同一句柄的 `RLock` 保护，别误判为“每次每个天体独立 Python 调用”的未优化实现。`build_epoch_data_batch()` 读取成功以后再并行独立数值计算，而不是直接并发访问同一可变 CALCEPH 句柄。
- **预插值星历不是无代价加速。** TudatPy 官方提供 `interpolated_spice`，用内存与插值误差交换查询成本；LunarOps 如果引入，必须验证星历位置/速度/加速度的误差经十年轨道积分和 LLR 可观测量放大后的预算，再决定是否使用。不能把精度较差的插值作为隐藏兜底。
- **ABM 已复用网格数据。** `_compute_abm_step()` 在两次校正后额外重算接受态导数是必要的数值一致性；不要删除。小型 `np.column_stack` 和步进多项式最终可以交给 Cython，但应排在 RHS 主要热点之后。改变步长、ABM 阶数、校正次数或 DOP853 启动策略属于数值算法方案变更，必须做独立积分误差与长期稳定性比较，不应算作纯粹代码加速。
- **SciPy `LowLevelCallable` 并不是 solve_ivp 通用 RHS 接口。** 官方列出的低层回调支持主要在 `quad`、若干 `ndimage` 函数，不能据此给现有 DOP853/ABM 直接塞 Cython 函数指针来绕过 Python 框架。若要真正无 Python 回调，需把固定步骤/固定数据布局的求解循环一并设计为 Cython 内核。
- **并行粒度**：同一单条耦合轨道和延迟自转不能将积分步骤随意并行；可以跨独立的 epoch source-cache 批次、独立弧段（仅在边界状态已定义）、独立参数试算案例并行。线程安全要覆盖月球动态重力场的可变系数、`DynamicsEvaluationContext`、历史缓存以及历表句柄，不可共享同一个可变求解对象做并行轨道。

## 10. 分阶段实施顺序和性能验收

| 阶段 | 变更 | 价值判断 | 主要风险 | 必过验证 |
|---|---|---|---|---|
| 0 | 固定 3 种代表性工作负载并采集壁钟/CPU/profile | 必须先做 | 低 | 记录 CALCEPH、源—源、RHS、figure、EIH、rotation、history、I/O 的累计耗时与调用次数 |
| 1 | 批量 Cython 计算源—源牛顿辅助量；再测 OpenMP 按 epoch 并行 | 对数百天体优先 | 中 | 对照旧核的势、加速度、EIH 与整弧积分 |
| 2 | 既有球谐核静态归一化预计算、工作区与三角递推 | 已有局部测量证据 | 低～中 | 全归一化/相位、极点、稀疏系数、动态二阶场 |
| 3 | 自转小核、固定 core 直接解、动态 mantle 3×3 SPD 解、力矩诊断分离 | 对耦合自转优先 | 中～高 | 核幔两种 spin 分支、四元数符号、正定性、转矩符号 |
| 4 | Cython 化延迟 Hermite 权重与加权求和 | 对月幔延迟优先 | 高 | 严格分段历史、导数一致、延迟与潮汐因果规则 |
| 5 | Python→Cython 边界融合、目标体专用力项装配、ABM 短向量操作 | 依 profile 决定 | 中～高 | 唯一规范 RHS、诊断一致、所有积分模式数值等价 |

**三类必要性能样本**：A：仅主要天体、6 维轨道，测轻量力项与 CALCEPH；B：`de440_major + sb441`、6 维轨道，测数百天体辅助计算；C：DE440/DE430 10 年轨道 + 月幔/液核自转 + 延迟形变，测 13/16 维 RHS、转矩与历史插值。每类先跑短弧低成本性能回归，再跑全弧正式科学对照。**每组固定配置、星历 SHA-256、GM 表、线程数、CPU 亲和性、编译器标志、Python/NumPy/Cython/CALCEPH 版本和对照结果。**

**整体速度只接受由真实程序 wall time 复现的收益**：不能把 `1.7×` 的 Cython 球谐微核加速当成整个 LLR 传播的 `1.7×`；报告 `T_total`、`T_epochData`、`T_RHS`、`T_rotation`、`T_history`、峰值 RSS，并记录每项占比。必要时用 Cython `-a` 注释查看 Python API 交互，用 Linux `perf` 对 native 部分采样；不要在最终优化编译开启大开销的 trace。

**科学验收**：力、力矩、形变张量、姿态矩阵、各时标、延迟值、系数归一化、状态维度与受力体集合不变。须保留未优化版本的脱离生产的回归基准（例如验证数据 / 测试夹具），而不是让**生产发布包**永久保留旧算法兼容开关。最终分开核验近地月心状态误差、姿态误差、投影至反射器的位移误差、LLR 光时/残差变化、十年弧末长期偏差，阈值由项目测距和动力学误差预算制定，不能事后放宽容差。不同 CPU/编译器的浮点逐位一致不必强求，但必须有独立、预先声明的物理验收标准。

## 11. 暂时不建议采取的“捷径”

- 全仓 `.py → .pyx`、只加 `@cython.boundscheck(False)` 就宣称加速，或把 Python 对象放进 `nogil` 循环。
- 删除 EIH 所需的全体牛顿辅助，或把历表的总加速度当成下阶点质量项。
- 直接将 3×3 时变惯量求解替换为显式逆矩阵；忽略退化/正定性检查。
- 对完整耦合轨道步骤做 `prange`，导致时滞历史与候选态缓存的非因果并发。
- 更换星历插值模型、降低精度、放大步长或减少 ABM 校正次数却宣称“仅性能优化”。
- 为性能加入第二套永久运行路径、旧参数别名、兼容读取或静默降级，违反既定的零历史兼容原则。
- 为几十至数百个积分体的当前架构盲目引入 GPU；其数据准备、复杂的候选态依赖与 PCIe 往返开销可能抵消收益，必须先测 CPU 编译核。

## 12. 与成熟软件的对应借鉴及官方资料

1. **Cython**：Typed Memoryviews、`nogil`、`prange`/OpenMP 与 `cythonize(annotate=True)`，对应数据布局、native 实现和 CPU 并行。
   - https://docs.cython.org/en/latest/src/userguide/memoryviews.html
   - https://docs.cython.org/en/latest/src/userguide/parallelism.html
   - https://docs.cython.org/en/latest/src/quickstart/cythonize.html
2. **TudatPy**：`interpolated_spice` 为已评估误差的星历插值提供设计参考，不能无审计地用于毫米级 LLR。
   - https://py.api.tudat.space/en/latest/dynamics/environment_setup/ephemeris.html
3. **CALCEPH**：`CalcephBin.compute_order()` 已支持多 JD 数组和加速度阶次，适于已有的批次模式。
   - https://calceph.imcce.fr/docs/latest/html/python/calceph.multiple.pythonusage.html
4. **SciPy**：DOP853 采用显式八阶 Runge–Kutta；LowLevelCallable 支持范围有限，不应误认为可直接插入 `solve_ivp`。
   - https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.DOP853.html
   - https://docs.scipy.org/doc/scipy/reference/generated/scipy.LowLevelCallable.html

**最终建议**：先完成标准化 profile，再依次实施“Cython 规定天体批量辅助核 → 静态球谐归一化与递推 → 自转小型 Cython 数值核 → 延迟 Hermite Cython 核”，最后根据真实热点考虑 ABM 进一步融合。整个改造只允许性能改善，不得改变物理模型、不保留生产兼容层。
