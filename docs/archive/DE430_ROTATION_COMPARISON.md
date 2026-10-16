# LunarOps 与 PlanetaryEphemeris.jl DE430 自转对比

日期：2026-10-10。实际执行本地 Julia DE430 模型与完整 354 天体传播，
再调用 LunarOps 生产物理模型逐项核对。范围只包含月球平动、自转与流核角速度；
没有接入 LLR、参数估计或变分方程。轨迹中只积分月球的自转，不积分地球或行星自转。

## 参考与可复现性

参考是本地 `PlanetaryEphemeris.jl` 0.12.1 的 `DE430!`，以
`data/de430ic_1969Jun28.txt` 为初值（JD TDB 2440400.5）。Julia 使用 AU/day、
地幔 ZXZ Euler 角，LunarOps 使用米/秒、Hamilton 标量在前的主动 PA→惯性四元数。
比较力矩与惯量时统一为全月 `M R²` 归一化，核角速度均表达在地幔 PA 系。
Julia 还保存核 Euler 角；它们不参与本文比较的地幔/核角速度方程，LunarOps 不积分它们。

本地参考库已存在 Earth orientation 的 `phi_x0/phi_y0` 角秒换算修正，以及默认
推进全部状态的 integration-index hook。本次没有修改参考库。完整已有 diff、Git HEAD
和五个相关源码文件的 SHA256 保存在输出的 `reference/provenance.json`；
这里的参考指本地版本，不声称是未经修改的上游版本。

[Julia 导出脚本](../scripts/export_de430_rotation_comparison.jl) 读取原始 `DE430!`，
只替换副本的最终返回值，导出内部项；没有改写其物理计算。
三天轨迹直接调用原 `DE430!` 的 Taylor 25 阶传播（绝对容差 `1e-20`），
原生分项评价的短回推多项式为 12 阶。LunarOps 900/450 秒、13 阶 ABM 对照
使用同一完整月球初始状态，六个共同外部力矩源，保持各自的外部轨道和延迟算法。

```bash
sbatch run_de430_rotation_comparison.sbatch
# 已导出参考后可单独重跑比较：
.venv/bin/python -m scripts.compare_de430_rotation output/de430_rotation_comparison
```

[Python 比较脚本](../scripts/compare_de430_rotation.py)、
[作业脚本](../run_de430_rotation_comparison.sbatch)；输出目录
`output/de430_rotation_comparison` 包含原始参考项、300 秒轨迹样本、四个 LunarOps
配置剖面的完整联合状态、`comparison.json`、PNG/PDF 对比图。

## 参数和模型差异

11 个大天体 GM、月球半径、未形变惯量、Love k2、延迟、核极惯量比例、核扁率和
黏性参数一致到双精度舍入。Julia 的形变参考角速度为
`2.6653246338670733e-6 rad/s`（由常量 GM 和 `384399.014 km` 的 Kepler 关系计算）；
当前 LunarOps 配置为 `2.6616995e-6 rad/s`。两者差约 0.136%，导致惯量差约
`1.106e-10`。只在专门比较配置中对齐，未修改生产配置。

Earth figure 力矩存在两处源码约定差异：

| 项目 | LunarOps 生产模型 | 本地 Julia DE430 源码 |
|---|---|---|
| 惯量 | 全月 `K_T` | 地幔 `K_m` |
| Eq. 44 的纬度正弦 | `u · p_E`，地月方向与地球极轴的点积 | `sin_ϕ[ea,mo]`，实际为地球方向的 PA z 分量 |

[DE430 论文](<../../document/Folkner 等 - The Planetary and Lunar Ephemerides DE430 and DE431.pdf>)
p.18 Eq. 44 明确给出 `sin φ = u · p_E`；Julia 的数组在所在天体
固定系中计算纬度，其 `[ea,mo]` 与这个定义不符。因此保留 LunarOps 的论文写法。
惯量取全月还是地幔也作为明确的模型约定记录，不为追求源码逐值相同而修改生产实现。
对齐这两项只通过比较脚本中的独立 adapter；它不进入配置 registry 或生产模型。

起始时刻 Julia 的纬度正弦为 `0.1103237609`，论文定义为 `0.4499570919`。
共同参数下，Earth figure 力矩最大分量差 `2.708e-21 s⁻²`，其中惯量选择贡献
约 `1.000e-24 s⁻²`，纬度选择贡献约 `2.707e-21 s⁻²`。

延迟算法也不同：Julia 每次在当前状态短回推刚体 Taylor 弧；LunarOps 只有起始
种子使用刚体回推，随后从接受的非弹性联合历史取状态和同一多项式导数。
分项比较直接喂入 Julia 的同一延迟样本/导数，以检验物理公式；整弧比较保留算法差异。

## 同输入逐项结果

实际评价三个历元：初始、一天、三天。以下为三个历元的最大分量绝对差。
所有共同输入检查通过；Earth figure/地幔角加速度的“Julia 约定”行使用上述
比较 adapter，不能解读成原生产模型已经逐值等于 Julia。

| 计算项 | 最大绝对差 | 单位/约定 |
|---|---:|---|
| PA→惯性姿态矩阵 | 2.220e-16 | 无量纲 |
| 全月惯量 | 5.551e-17 | M R² 归一化 |
| 地幔惯量 | 5.551e-17 | M R² 归一化 |
| 核惯量 | 0.000e+00 | M R² 归一化 |
| 潮汐形变 | 1.588e-22 | M R² 归一化 |
| 旋转形变 | 5.118e-24 | M R² 归一化 |
| 惯量导数 | 3.761e-27 | s⁻¹ |
| 动态二阶系数 | 7.378e-17 | 未归一化系数 |
| 惯量导数 × 角速度 | 1.514e-33 | s⁻² |
| 地幔欧拉交叉项 | 1.431e-31 | s⁻² |
| 核幔耦合力矩 | 1.495e-35 | s⁻² |
| 核幔黏性力矩 | 0.000e+00 | s⁻² |
| 核幔压力力矩 | 1.495e-35 | s⁻² |
| Earth J2 力矩（Julia 约定） | 6.052e-34 | s⁻² |
| 地幔角加速度（Julia 约定） | 6.737e-28 | rad/s² |
| 核角加速度 | 5.049e-29 | rad/s² |
| Euler→四元数运动学 | 1.059e-22 | s⁻¹ |

六个源分别比较完整 degree 2–6 球谐直接加速度和反作用力矩，没有只比较合计。

| 源 | 非球形加速度差（m/s²） | 归一化力矩差（s⁻²） |
|---|---:|---:|
| EARTH | 1.311e-25 | 2.643e-28 |
| SUN | 5.091e-36 | 1.300e-30 |
| MERCURY | 3.703e-36 | 4.546e-37 |
| VENUS | 5.701e-36 | 1.138e-35 |
| MARS BARYCENTER | 4.212e-35 | 6.553e-36 |
| JUPITER BARYCENTER | 1.618e-39 | 1.220e-35 |

## 三天轨迹结果

Julia 同时推进 354 天体；LunarOps 推进月地相对六状态和月球自转，
外部状态来自 DE430 BSP，只包含 11 个大天体质量。下表的轨迹差包含这些物理/外部轨道
和延迟历史差异，不是积分器误差，也不能声称是完整 DE430 复现。

| LunarOps 配置 | 姿态最大差（rad） | 月面位移尺度（m） | 地幔角速度最大分量差（rad/s） | 核角速度最大分量差（rad/s） | 相对位置最大差（m） |
|---|---:|---:|---:|---:|---:|
| production | 2.844e-10 | 4.944e-04 | 2.236e-15 | 7.328e-16 | 2.645e-02 |
| native_mean | 2.440e-10 | 4.241e-04 | 1.924e-15 | 6.290e-16 | 2.645e-02 |
| native_mean_earth_conventions | 3.952e-13 | 6.869e-07 | 4.854e-20 | 2.961e-20 | 2.645e-02 |
| native_mean_half_step | 2.440e-10 | 4.241e-04 | 1.924e-15 | 6.290e-16 | 2.645e-02 |

这些配置分别为：原生产参数；仅对齐形变参考角速度；再对齐 Julia Earth figure 两项约定；
仅对齐参考角速度并减半步长。它们均采用相同月球初始状态。

| LunarOps 内部对照 | 姿态最大差（rad） | 相对位置最大差（m） |
|---|---:|---:|
| production_vs_native_mean | 4.044e-11 | 5.677e-06 |
| native_mean_vs_native_mean_earth_conventions | 2.440e-10 | 1.490e-06 |
| native_mean_vs_native_mean_half_step | 8.320e-15 | 4.074e-06 |

这里的减半步长对照衡量数值误差；改变物理参数/约定的对照衡量模型差异。
月面位移尺度为 `R × 姿态差角`，不是 LLR 测距残差。

额外核对实际 DE430 BSP/BPC：Julia 的三天月地相对位置最大差为 3.195e-04 m，
姿态最大差为 1.143e-09 rad；LunarOps 当前生产参数的对应值为
2.636e-02 m 和
1.128e-09 rad。
因此 Julia 实现也不是 JPL 发布星历的逐值解，匹配它不能替代对 BSP/BPC 的验证。
这些诊断保存在 `bsp_diagnostic.json` 和总报告中。

后续需要明确保留/选择 DE430 的形变参考角速度定义，并单独做共同外部轨道、共同
延迟算法的长期传播对照；不能把目前的源列表和算法差异混入积分器误差预算。
本批不改生产物理参数或参考源码，不进入 LLR 和估计。
