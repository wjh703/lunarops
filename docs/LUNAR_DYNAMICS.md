# 月球轨道动力学技术说明

## 范围

当前轨道程序使用 6 维地月相对平动积分。Earth 的 BCRS 位置、速度和加速度由
历表规定，Moon 由 `Earth + Moon/Earth relative` 重建；太阳和行星也使用历表给定的
外部状态，因此这是一个混合受迫子系统，不是完整 DE440 重积分。仓库仍保留
生产路径只积分 6 维 Moon/Earth 相对状态。地幔/流核自转联立和变分方程暂未实现。

所有动力学模块统一使用 SI 单位、TDB 时间和 SSB/ICRF 坐标。积分状态为
生产轨道程序的积分状态是 `[r_M/E, v_M/E]`。RHS 使用
`a_M(model) - a_E(ephemeris)`；Moon 的 BCRS 状态在输出和历史查询时由历表 Earth
状态加相对状态重建。力模型可以独立启用，模型组成完全由配置决定。

## 模块边界

- `state.py`：物理状态和积分状态之间的显式转换。
- `bodies.py`：积分天体、外部给定天体及其 GM。
- `context.py`：连续数组形式的 `DynamicsEpochData`、动力学工作区和批量结果。
- `fileio/mass_catalog.py`：统一的天体 GM 目录及其原生 artifact 读写。
- `forces.py`：牛顿、EIH、潮汐和太阳力的数学内核。
- `gravity.py`：`GravityCoefficients` 文件加载、球谐场适配、J2 构造。
- `earth_gravity_orientation.py`：DE440 长期岁差、18.6 年章动和拟合极轴偏置。
- `force_models.py`：可组合的力模型和加速度汇总。
- `lunar_dynamics.py`：规定 Earth 下的 Moon/Earth 相对状态和 RHS。
- `integrators.py`：固定步长 modified ABMD、DOP853 启动和规则节点局部高阶插值。
- `programs/lunar_orbit.py`：配置解析、模型装配、传播和诊断输出。
- `fileio/numeric_table.py`：轨道和加速度诊断的紧凑二进制表。

## 力模型

`NewtonianPointMassForce` 对完整 `newtonianSystem` 计算 Newtonian 加速度和势，包括
所有由历表给定的外部天体。外部天体之间的项在每个历元预计算一次，同一历元
的校正迭代只更新 Earth/Moon 相关项。`EihPointMassForce` 使用这份完整辅助系统，但只
计算 `bodyForceTerms` 中启用 `eih_1pn` 的目标行。当前只计算 Moon 行；Earth
加速度直接取历表，
373 个小天体不参与状态积分，但各自的 Newtonian 辅助加速度仍包含所有其他
有质量天体。figure、潮汐或其他非点质量项不会直接代入 EIH 辅助量。
完整 Newtonian 系统由 Cython 对称 pair 内核计算，每个天体对只计算一次距离，
同时更新两端加速度和势，并避免构造 `(N,N,3)` 临时数组。

Earth/Moon 静态场由 `gravityFields.earth` 和 `gravityFields.moon` 文件提供，
文件加载后得到 `GravityCoefficients`，由 `GravityField` 计算体固连系中的非球形
加速度。J2 也通过同一 `GravityCoefficients` 路径表示。Earth 动力学极轴由 DE440
长期岁差、18.6 年章动及 `ROTEX/ROTEY` 构造，不读取观测建模使用的 EOP/ITRF
变换；Moon 姿态来自历表 PA 矩阵。

`FigureForce` 必须为每个扩展体显式声明 point-mass interaction partners。
`LunarForceGroup` 只为 Moon 编译目标掩码；partner 集合只定义每个扩展体的受力对象。
即使 Sun 和主要行星使用规定轨道，Moon figure 与这些天体相互作用产生的 Moon
反作用仍会进入积分。Earth figure 只需计算对 Moon 的直接作用；Earth 对其他天体
figure 相互作用产生的 Earth recoil 已包含在规定 Earth 历表中。
旧的 `mu_by_body` 构造接口已删除。

`EarthTideForce` 在规定 Earth 模式下直接返回完整的 Moon/Earth 相对潮汐加速度；
`LunarDegree2GravityCorrectionForce` 只计算动态减静态的 degree-2 增量，避免重复评估静态
月球高阶场。太阳相对论项拆分为
`LenseThirringForce` 与 `SolarRadiationPressureForce`。
可选太阳 Lense--Thirring 与辐射压。

## 配置原则

每个物理效应由自身配置项控制：

- `includeEih: true` 启用 EIH；
- `solarJ2` 非空启用太阳 J2；
- `gravityFields` 非空启用 Earth/Moon 静态球谐场；
- `earthJ2TimeVariation` 以 J2000 TDB 为参考历元配置 Earth J2 时间多项式；
  DE430 只含一次项，DE440 含 `linearPerYear*T + quadraticPerYear2*T^2`，二次项
  是多项式系数，不乘 `1/2`；
- `earthTide` 非空启用地球潮汐；
- `lunarDegree2Gravity` 非空启用动态月球二阶系数；
- `solarRelativistic` 非空启用太阳相对论/辐射压项。
- `stepSeconds` 是 ABMD 固定主步长，论文配置为 5400 s（1/16 日）；
- `integratorOrder` 是 Adams--Bashforth--Moulton 阶数，论文配置为 13；
- `correctorIterations: 2` 实现论文的 PECEC 流程；
- `startupStepSeconds` 是八阶 Dormand--Prince 启动步长，论文配置为 675 s，即主步长的 1/8；
- `historyInterpolationOrder` 控制延迟历史的局部插值阶数；`trajectoryInterpolationOrder` 控制输出轨迹查询插值阶数。
- `accelerationDiagnosticsStepSeconds` 控制分项加速度诊断间隔，必须是主积分步长的整数倍；为空时不重复计算诊断。

规定 Earth 模式的 `LunarForceGroup` 将 acceleration target 固定为 `MOON`；
`bodyForceTerms` 直接声明 Moon 的力项序列，因为 Earth
加速度来自历表。`point_mass` 是 EIH 的前置项，
因此启用 `eih_1pn` 时必须同时启用 `point_mass`。程序在每次 RHS 中先缓存
Newtonian 点质量辅助系统，再计算目标天体的 EIH，最后累加所有启用的加速度数组。
规定 Earth 模式中，`tide` 直接给出完整的 Moon/Earth 相对潮汐加速度。
`lunar_degree2_gravity` 是时变月球二阶场相对静态月球场的加速度增量。

加速度诊断固定汇总为七组：`newtonian`、`post_newtonian`、`figure`、
`tide`、`lunar_degree2_gravity`、`srp` 和 `lt`。其中 `figure` 汇总太阳 J2、静态 Earth
figure 与静态 Moon figure；`lunar_degree2_gravity` 单独保存月球时变惯量张量产生的增量。
每组输出 modeled Moon 三轴分量。历表 Earth 的总加速度无法按本程序的七个力组
可靠拆分，因此不再输出伪造的 Earth/EMB/relative 分项。

GM 不再内嵌于轨道配置。先运行 `MassCatalogCreate`，它读取
`configs/lunarops_mass_catalog.yml` 中的主天体和 SB441-N373 数值、统一转换为
`m3/s2` 并生成 `massCatalog` artifact。轨道程序通过 `inputFileMassCatalog`
读取该目录，以 `externalBodyGroups` 和 `externalBodyIds` 选择规定轨道源。
Earth 和 Moon 不接受两份独立 GM；`MassCatalogCreate` 从 DE440 的 canonical
`GMB` 与 `EMRAT` 统一推导两者，确保 EMB 状态变换、潮汐和所有力模型使用
完全相同的质量参数。
球谐文件头中的 GM 只作为输入文件元数据；加载时会由同一 mass catalog 数值
覆盖，轨道配置不再提供逐场 `gmM3S2` 覆盖接口。

目录中的 `bodyId` 使用 NAIF 语义的规范名称。`MARS` 表示行星中心
NAIF 499，`MARS BARYCENTER` 表示火星系统质心 NAIF 4，二者不会互相回退。
`EMB` 和 NAIF 常用名 `EARTH BARYCENTER` 均规范化为更明确的
`EARTH MOON BARYCENTER`（NAIF 3）。
DE440 对 Mars 至 Pluto 只提供系统质心，因而当前外部摄动体使用
`MARS BARYCENTER` 至 `PLUTO BARYCENTER`。Mercury 和 Venus 同时提供本体中心
与系统质心；本配置为保持行星外部摄动体的一致语义，也使用其质心记录。

## 数值积分

积分器支持正、负积分时长和乱序查询，但不允许普通轨迹查询超出弧段。传播时长
必须是固定主步长的整数倍，以保持严格规则网格。生产模式不构造 dense-output
多项式；轨迹节点通过局部高阶插值支持查询。外部历表按积分历元批量查询并使用
CALCEPH 预取缓存。ABMD 启动使用八阶 Dormand--Prince 小步积分生成前 `k-1` 个
主网格节点，启动期间的延迟状态来自参考历表。启动后由 `LunarNodeBuffer` 保存节点和
导数，由 `DelayedHistory` 负责局部 Lagrange 插值；body-name 混合查询由
`LunarDynamics` 处理。

积分器在生成规则节点时调用 `NodeDiagnosticCallback` 诊断回调。每日诊断直接复用该节点的
`DynamicsEpochData`，只保存 modeled Moon 的分项加速度，不再重复构造 Earth 诊断
动力学。输出拆分为 `lunarOrbit` 二进制表、
`lunarAccelerationDiagnostics` 二进制表和轻量 `lunarOrbitMetadata` 文本 artifact。

启动步长在完整 384 体模型上以 675 s 为基准进行了全年对照。1350 s 的月地相对
位置最大差约 1.26 mm；2700 s 的 EMB 最大差约 9.31 mm；5400 s 的月地相对位置
最大差约 1.58 mm。按所有主状态位置差均小于 1 mm 的保守标准，生产默认值仍为
675 s。更大启动步长只减少固定启动成本，不影响长弧每节点成本，因此没有放宽标准。
