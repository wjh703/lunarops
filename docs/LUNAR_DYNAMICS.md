# 月球轨道动力学技术说明

## 范围

当前实现的是 Earth/Moon 12 维联立平动积分。太阳和行星使用历表给定的
外部状态，因此这是一个混合受迫子系统，不是完整 DE440 重积分。地幔/流核
自转联立和变分方程暂未实现。

所有动力学模块统一使用 SI 单位、TDB 时间和 SSB/ICRF 坐标。积分状态为
`[r_B, v_B, r_M/E, v_M/E]`，其中 `B` 是 Earth--Moon barycenter，
`r_M/E = r_M-r_E`。力模型可以独立启用，模型组成完全由配置决定。

## 模块边界

- `state.py`：物理状态和积分状态之间的显式转换。
- `bodies.py`：积分天体、外部给定天体及其 GM。
- `context.py`：连续数组形式的 `PreparedEpoch`、动力学工作区和批量结果。
- `fileio/mass_catalog.py`：统一的天体 GM 目录及其原生 artifact 读写。
- `forces.py`：牛顿、EIH、潮汐和太阳力的数学内核。
- `gravity.py`：`pyshtools.SHGravCoeffs` 文件加载、球谐场适配、J2 构造。
- `earth_orientation.py`：DE440 长期岁差、18.6 年章动和拟合极轴偏置。
- `force_models.py`：可组合的力模型和加速度汇总。
- `earth_moon.py`：Earth/Moon 加速度到 EMB/相对坐标的投影。
- `integrators.py`：固定步长 modified ABMD 及 Runge--Kutta/RKN、
  Stoermer--Cowell、Gauss--Jackson 接口；轨迹查询使用规则节点局部高阶插值。
- `delayed.py`：延迟项的分段推进和历史缓存插值。
- `programs/lunar_orbit.py`：配置解析、模型装配、传播和诊断输出。
- `fileio/numeric_table.py`：轨道和加速度诊断的紧凑二进制表。

## 力模型

`PointMassForce` 对完整 `newtonianSystem` 计算 Newtonian 加速度和势，包括
所有由历表给定的外部天体。外部天体之间的项在每个历元预计算一次，同一历元
的校正迭代只更新 Earth/Moon 相关项。`EihForce` 使用这份完整辅助系统，但只
计算 `bodyForceTerms` 中启用 `eih_1pn` 的目标行。当前目标为 Earth 和 Moon；
373 个小天体不参与状态积分，但各自的 Newtonian 辅助加速度仍包含所有其他
有质量天体。figure、潮汐或其他非点质量项不会直接代入 EIH 辅助量。
完整 Newtonian 系统由 Cython 对称 pair 内核计算，每个天体对只计算一次距离，
同时更新两端加速度和势，并避免构造 `(N,N,3)` 临时数组。

Earth/Moon 静态场由 `gravityFields.earth` 和 `gravityFields.moon` 文件提供，
文件加载后得到 `SHGravCoeffs`，由 `GravityField` 计算体固连系中的非球形
加速度。J2 也通过同一 `SHGravCoeffs` 路径表示。Earth 动力学极轴由 DE440
长期岁差、18.6 年章动及 `ROTEX/ROTEY` 构造，不读取观测建模使用的 EOP/ITRF
变换；Moon 姿态来自历表 PA 矩阵。

`FigureForce` 必须为每个扩展体显式声明 point-mass interaction partners。
partner 集合与 `bodyForceTerms` 产生的 acceleration target mask 相互独立，因而
即使 Sun 和主要行星使用规定轨道，它们对 Earth/Moon figure 的反作用仍会进入
积分。旧的 `mu_by_body` 构造接口已删除。

`EarthTideForce` 使用延迟历表状态和 Love 数；`LunarInertiaFigureForce`
可用延迟轨道和规定的 PA 姿态更新月球二阶系数；太阳相对论项拆分为
`LenseThirringForce` 与 `SolarRadiationPressureForce`。
可选太阳 Lense--Thirring 与辐射压。

## 配置原则

每个物理效应由自身配置项控制：

- `includeEih: true` 启用 EIH；
- `solarJ2` 非空启用太阳 J2；
- `gravityFields` 非空启用 Earth/Moon 静态球谐场；
- `earthTide` 非空启用地球潮汐；
- `lunarInertia` 非空启用动态月球二阶系数；
- `solarRelativistic` 非空启用太阳相对论/辐射压项。
- `stepSeconds` 是 ABMD 固定主步长，论文配置为 5400 s（1/16 日）；
- `integratorOrder` 是 Adams--Bashforth--Moulton 阶数，论文配置为 13；
- `correctorIterations: 2` 实现论文的 PECEC 流程；
- `startupStepSeconds` 是八阶 Dormand--Prince 启动步长，论文配置为 675 s，即主步长的 1/8；
- `interpolationOrder` 控制规则积分节点上的局部历史插值阶数，默认与 ABM 同为 13。
- `accelerationDiagnosticsStepSeconds` 控制分项加速度诊断间隔；为空时不重复计算诊断。
- `preparationBatchSize` 控制规则节点和启动 stage 的有界双缓冲预取批量；
- `preparationWorkers` 控制释放 GIL 后的逐历元 Newtonian 并行准备线程数。

`bodyForceTerms` 为每个天体声明启用的力项。`point_mass` 是 EIH 的前置项，
因此启用 `eih_1pn` 时必须同时启用 `point_mass`。程序在每次 RHS 中先缓存
Newtonian 点质量辅助系统，再计算目标天体的 EIH，最后累加所有启用的加速度数组。
`tide` 是 Earth--Moon 耦合项，配置必须对 Earth 和 Moon 同时启用或同时关闭。

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
主网格节点，启动期间的延迟状态来自参考历表。启动后只在固定容量环形缓存中保留
局部规则节点，并进行 Lagrange 插值或外推；外部天体继续由参考历表批量提供。

积分器在接受规则节点时调用 observer。每日诊断直接复用该节点的
`PreparedEpoch`，只保存 Earth、Moon、EMB 和 Moon-Earth 的分项投影，不再保存
外部天体的辅助数组。输出拆分为 `lunarOrbit` 二进制表、
`lunarAccelerationDiagnostics` 二进制表和轻量 `lunarOrbitMetadata` 文本 artifact。

启动步长在完整 384 体模型上以 675 s 为基准进行了全年对照。1350 s 的月地相对
位置最大差约 1.26 mm；2700 s 的 EMB 最大差约 9.31 mm；5400 s 的月地相对位置
最大差约 1.58 mm。按所有主状态位置差均小于 1 mm 的保守标准，生产默认值仍为
675 s。更大启动步长只减少固定启动成本，不影响长弧每节点成本，因此没有放宽标准。
