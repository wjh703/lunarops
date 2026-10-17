# DE430／DE440 Chat review 实证核对

日期：2026-10-11。核对上传代码包的 Chat review、当前源码、本地原始论文、
真实 DE430／DE440 BSP/BPC 和已有对照输出。范围为平动／自转动力学，
没有接入 LLR、参数估计或变分方程。

后续已完成修复后的十年 DE440 联合积分，见
[复跑报告](DE440_REVIEW_FIX_10A_REGRESSION.md)：最大轨道误差
`45.866 cm → 45.394 cm`，姿态最大月面位移尺度仍约 `2.323 m`。
下文“本次没有提交新十年作业”指最初 review 核对阶段。

2026-10-12 后续：[KBO 环与时变太阳 GM 实现／验证](DE440_MISSING_MODELS_VALIDATION.md)
已完成独立数值核对和四组十年消融，两项同时开启后轨道最大误差
`45.394 cm → 2.812 cm`，姿态残差仍约 `2.323 m`。
下表“尚未实现”描述 2026-10-11 的核对快照。

## 逐项结论

| Review 判断 | 本次核对结论 | 当前处理 |
|---|---|---|
| DE440 的 36 点 KBO ring 缺失 | 确认。论文 §2.2／§3.1 和 BSP 注释常数均有依据；当前 384 天体不含环点 | 增加独立近似加速度诊断；生产环模型尚未实现 |
| GMSDOT 缺失 | 确认当前没有时变太阳 GM 路径，原始注释给出非零 GMSDOT | 尚未编码；先确定时间单位、归一化及 GM 的参考历元 |
| 规定地球情况下，漏掉环不能只按地月差分效应评估 | 确认，这是现有运动方程直接决定的 | 修正此前关于小项共模抵消的判断；补做两个起始历元的诊断 |
| Earth tide 相对量直接装入 Moon 绝对加速度 | 确认；结合论文的 EMB 不变约定，可以明确推导月球份额 | 已修正 Python 力模型边界，保留 Cython 相对量内核 |
| 纯轨道 ABM 接受节点导数不一致 | 确认，独立简谐振子重现 | 已统一重算最终接受状态 RHS |
| 负轨道时滞触发远距离高阶外推 | 确认算法与病态权重风险；现有探针不能直接给出真实积分历史误差 | 尚未替换历史算法；保留负时滞物理参数 |
| C21/S21 与米级姿态差高度相关 | 确认已有受控实验；常数如何进入正式模型仍待核实 | 不将 field-only 加常数实验升格为物理修复 |
| 地球极轴缺少完整独立验收 | 确认。当前环境有真实 ERFA，但缺少 JPL 极轴真值 | 记录真实 ERFA 下的几何检查，仍不宣称模型验收完成 |
| DE430 已有 J5E | 确认本地官方 BSP 注释有非零 J5E，配置最高 5 阶合理 | 保留现有配置；记录原始论文文字与定版常数的差别 |

Review 所报简化模型的 `0.919 m` 环效应、`5.82 cm` GMSDOT 效应没有随文
提供完整实验代码和初值。本次没有独立重现这些数值，也不把它们当作实际
LunarOps 十年误差或各缺项的验收标准。

## 规定地球的方程与环模型缺口

当前积分状态是月球相对地球六状态，但其加速度来自

```text
relative_acceleration = model_moon_barycentric_acceleration
                        - ephemeris_earth_barycentric_acceleration
```

因此遗漏一个外力时，误差是漏掉的 Moon 绝对加速度；只有双方使用相同
开关并分别计算两边模型时，才可按 Moon-minus-Earth 的差分效应评价。
现有方程中补环应加月球绝对项，不再手动减一次地球环项。GMSDOT 同理。
此前“小项主要共模抵消”的量级判断不适用于这个混合方程。

[Park 2021](<../../document/Park 等 - 2021 - The JPL Planetary and Lunar Ephemerides DE440 and DE441.pdf>)
§2.2 明确给出 36 个等质量点、44 au、黄道平面的环；§3.1 说明其进入
DE440 摄动。原始注释中的 `MA8201…MA8236` 均为
`5.5227699716988214e-13 au³/day²`，每点 SI GM 为
`2.4768842184000977e11 m³/s²`。当前质量表和装配流程只有 11 个主要天体
加 373 个 SB441 摄动体，共 384 个，没有这 36 点的轨迹或环力模型。

本次诊断只采用以太阳为中心、J2000 平黄道中的等间距 44 au 圆环，
尚未声称复现 JPL 环点的完整状态演化。在 16 个实际参考历元上：

| 近似环诊断量 | 范围 |
|---|---:|
| Moon 绝对加速度模长 | 2.295–2.384 × 10⁻¹⁵ m/s² |
| Moon-minus-Earth 加速度模长 | 5.652–6.403 × 10⁻¹⁸ m/s² |
| 固定总 GM，36 点与 72 点的 Moon 加速度差 | ≤ 4.64 × 10⁻²⁹ m/s² |

这支持 review 的方程层判断，但瞬时加速度不能直接换算为十年轨道差。
正式实现仍需明确环的中心、参考平面和状态约定，并检查其参与牛顿／EIH
辅助量的范围，避免只在一条计算路径中补项。

`GMSDOT=-9e-14` 在 DE440 原始积分注释中明确存在；Newtonian、EIH 的
GM 数组、SolarJ2 重力场和 SolarSourceParameters 目前均固定。注释同时列出
`JDEPOC=2440400.5`，但不能据此直接断定 GMS 的时间变化以它为零点，
也不能仅凭数值假设是以 J2000 为基准的年相对变化率。这里保留为需要
原始实现／常数说明确认的模型缺口，不写猜测性公式进生产配置。

## 已修正的地球潮汐边界

[Folkner DE430／DE431](<../../document/Folkner 等 - The Planetary and Lunar Ephemerides DE430 and DE431.pdf>)
p.14 式 (32) 明确把该量定义为 Moon 相对 Earth 的加速度，p.15 说明月球
引起的 Earth tides 不改变 EMB，且忽略 Sun-raised tides 的质心效应。
Park 2021 p.7 式 (36) 包含相同的 `(m_E+m_M)/m_E` 因子，并重申这两项
质心约定。因此对这部分模型有

```text
a_rel = a_M - a_E
mu_E * a_E + mu_M * a_M = 0

a_M = mu_E / (mu_E + mu_M) * a_rel
a_E = -mu_M / (mu_E + mu_M) * a_rel
```

当前 DE440 的 Moon 份额为 `0.9878494156041707`。过去把完整 `a_rel`
加入 Moon 数组后又减规定 Earth 加速度，会重复加入其 `0.0121505843958293`
份额。这里的 1.215% 是相对潮汐项的份额，不是总轨道加速度或轨道误差比例。

修改 [force_models.py](../lunarops/classes/dynamics/force_models.py) 中
`EarthTideForce.acceleration()`；`EarthTideModel.relative_acceleration()` 和
Cython 内核继续返回论文定义的相对量，未改变负时滞、Love 数或底层公式。
DE430 和 DE440 都使用这一物理边界，因此两种模式均会受该修正影响。

本地 [Julia DE430 RHS](../../PlanetaryEphemeris.jl/src/DE430/dynamicalmodel.jl)
约 2040–2063 行同样把完整潮汐项只加到 Moon，并直接填入各天体的
绝对加速度，没有对应 Earth 分配。这与上述 EMB 约定不一致；所以
“与 Julia 相同”不能否定这次边界修正，也不能把 Julia 当作该项的真值。
本次只检查源码，没有再进行 Julia 传播对比。

新增 Moon-raised／Sun-raised 两个测试：使用真实潮汐核并构造保持 EMB
不变的规定 Earth 加速度，检查实际 LunarDynamics RHS 恢复完整相对潮汐，
同时检查质心加速度为零及 Earth 诊断数组保持规定天体语义。

16 个参考几何上的月球绝对潮汐修正模长为
`3.02e-16…9.48e-16 m/s²`。某些历元的总加速度残差下降，某些上升；
这一修正依据物理边界成立，不能以总残差是否单调下降决定是否采用。

## 已修正的 ABM 导数历史

最后一轮校正先在候选状态评价 RHS，再算出新的接受状态。纯轨道旧路径
直接保存前者的 RHS；联合模式已做最终重算。本次删除模式条件，统一
在最终接受状态重新评价，且不沿用上一候选的 evaluation cache。

独立六状态简谐振子，ABM 2 阶、两次校正：步长 `0.1` 时旧路径的
保存导数与接受状态 RHS 的差模长为 `1.735448097e-5`，步长 `0.5` 时为
`1.074741446e-2`（模型单位）。修正后四组步长／校正次数组合均逐值一致。
这证明历史存储缺陷存在，未证明它是实际 13 阶十年残差的主要原因。

修正前数据：[abm_before.json](../output/chat_review_verification_20261011/abm_before.json)。
新增测试位于 [test_lunar_dynamics.py](../tests/test_lunar_dynamics.py)。

## 负时滞和 C21/S21 的未完成核实

DE440 的 `tauOrbDays[2]=-0.20982243826832672` 确实需要未来状态。
当前 `DelayedHistory` 对未来 Moon 状态和规定天体都进行高阶外推。
已有探针的公里级差来自参考 BSP 样本的外推，不是实际接受节点历史的
积分误差；步长减半时真实长弧轨道差约 14.6 mm。这两种证据要分别解释。

下一次历史改动应让规定天体未来状态直接查询历表，Moon 使用独立验收的
稳定预测；月幔正时滞继续使用已接受历史的因果插值。不能改负时滞符号，
也不能直接用参考 Moon 未来轨道替代待积分状态。本次没有替换该算法。

C21/S21 的 field-only 加常数实验将姿态月面尺度从 2.323216 m 降到
0.039085 m，是强关联证据，尚不构成轴系和机械惯量的一致性证明。
新增核对的 [Pavlov 2016](<../../document/Pavlov 等 - 2016 - Determining parameters of Moon’s orbital and rotational motion from LLR observations using GRAIL and.pdf>)
p.8 式 (16) 明确允许写成 `C21=C21⁽⁰⁾−I13/R²`、
`S21=S21⁽⁰⁾−I32/R²`，但其文字说 DE430 的这些均值为零，并在 p.12
指出非零 GRAIL 值与 PA 轴不对齐的问题，另做了非零 S21 实验。
该论文证明“偏置 + 动态惯量”是有讨论的形式，却没有证明 DE440 注释
中的数值就是应直接加入的永久偏置。

正式修复需要同时说明参考轴、未形变／形变惯量、gravity field 和外部
力矩如何一致。在取得 DE440 原始约定前，保持该实验独立，不把非零
重力偏置无条件塞入机械惯量，也不通过调整初始姿态掩盖问题。

## 地球极轴、J5 与两个历元的参考加速度

真实环境使用 `pyerfa/erfa 2.0.1.5`，无测试替身。16 个历元的 Earth
矩阵正交误差 ≤ `2.22e-16`，行列式约为 1。这只验证几何性质；没有
JPL 动力学极轴的独立数值真值，不能以完整 IERS/EOP 地球取向替代真值。

DE430 本地官方 `de430_1850-2150.bsp` 的原始注释有
`J5E=-2.277345e-7`；Julia 常数及现有 GFC 也一致。
但 Folkner 主论文 p.12 的概述文字确实写 Earth “through fourth degree”。
这里依定版 BSP 常数核实非零 J5E，保留第五阶配置，不能仅凭主论文
那句概述删除 J5E，也不假装本次已读到 review 引用的另一份说明 §4.2。
摘录见 [de430_native_selected_constants.txt](../output/chat_review_verification_20261011/de430_native_selected_constants.txt)。

新增 [verify_de440_chat_review.py](../scripts/verify_de440_chat_review.py) 在
J2000、1969-06-28 两个起点各取 8 个历元，用真实 BSP 的 Moon 状态和
BPC 规定姿态评价全部已实现力项；延迟状态直接查询参考，排除未来外推
和积分历史的影响。输出绝对加速度残差向量、各力项、潮汐边界修正和
近似环效应，结果为 [reference_acceleration_audit.json](../output/chat_review_verification_20261011/reference_acceleration_audit.json)。

J2000 起点的总加速度残差约 `5.35e-15 m/s²`；1969 起点约
`9.86e-12 m/s²`，之后第 1／3 天仍约 `5.70e-13／6.27e-13 m/s²`。
这些数值来自 BSP 拟合位置多项式的二次导数，不是 JPL 原始 RHS。
特别是初始历元附近的大值，不能直接宣布是真实漏力；需要进一步区分
拟合二阶导数误差和物理模型误差。加入近似环后残差也未逐历元统一下降。

因此本次确认了具体缺项和边界错误，尚未确定约 46 cm 轨道残差的主因。
46 cm 与米级姿态残差应继续分开排查，单项瞬时结果不可相加解释长弧。

## 本次验证与后续验收

运行方式：

```bash
.venv/bin/python scripts/verify_de440_chat_review.py
.venv/bin/python -m pytest -q tests/test_lunar_dynamics.py tests/test_lunar_rotation.py tests/test_lunar_deformation.py tests/test_orbit_problem.py
.venv/bin/python -m pytest -q tests/test_pavlov_independent_validation.py
```

四组动力学测试 44 项通过，独立 Pavlov 内核测试 2 项通过，共 46 项。
修改涉及的 Python 文件通过 Ruff，`git diff --check` 通过。
本次没有提交新十年作业，没有改写已有轨道输出，没有重新拟合初值。

优先补齐环与太阳 GM 的可验证约定，再处理未来状态预测和二阶场映射。
每项保持独立对照；最终在两个固定起始历元、相同步长和输出口径下重跑
规定姿态与联合姿态长弧，分别验收力模型正确性和积分收敛。
旧 18 cm／46 cm 数字及此前逐值复跑结果对应本次修复前的源码快照，
不代表修复后代码已经通过长弧验证。
