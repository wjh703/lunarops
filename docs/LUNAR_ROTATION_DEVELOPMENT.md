# 月球自转物理与计算约定

本文件保留状态、坐标系、惯量、形变、力矩和核幔耦合的物理约定。
[开发与调试历史](archive/LUNAR_ROTATION_DEVELOPMENT_HISTORY.md) 保存历次作业、
残差排查及阶段验收；当前代码结构见 [动力学架构](LUNAR_DYNAMICS_ARCHITECTURE.md)。

## 当前计算边界

生产传播为规定 Earth 的月地相对平动与可选地幔／流核联合积分。
力、力矩、惯量和自转导数使用单一 Cython fused RHS；Python 负责参数装配、
时间输入、ABMD 循环和接受历史。参考物理实现仅用于 tests 的 oracle。
Earth 引力轴与观测 EOP 分开，初值、核参数和重力场由配置明确指定。
`pressureSpin` 与 `c22Static` 的物理选择仍由配置控制，本轮重构不改变其选择。

正延迟形变采用 method of steps：显式 `rigidBackward` 回推种子、正向 DOP853
启动和固定步长 ABMD。要求步长不大于延迟，种子覆盖延迟及历史模板；零延迟
隐式 spin 方程不在当前实现范围内。`LunarNodeBuffer` 存接受状态与导数，
`DelayedHistory` 的局部 Hermite 多项式同时提供两者；试算节点不进入接受历史。
四元数的输出同号对齐和单位化不修改接受状态／导数历史。

## 物理依据与模型范围

两篇本地论文是方程依据，三个项目是实现参考。下面的页码均为 PDF 页码，
与论文印刷页码一致；源码与论文存在差异时，应记录差异，不能以源码自动覆盖论文。

| 资料 | 本文使用的位置 |
|---|---|
| [Folkner 等，The Planetary and Lunar Ephemerides DE430 and DE431，IPN Progress Report 42-196，2014](<../../document/Folkner 等 - The Planetary and Lunar Ephemerides DE430 and DE431.pdf>) | §II.E，pp.8–10，式 (10)–(20)：PA 姿态、角速度和 MER 变换；§IV，pp.15–18，式 (33)–(45)：地幔／核动力学、惯量、力矩；p.49 表 7 和 p.51 表 11：初值与参数 |
| [Park 等，The JPL Planetary and Lunar Ephemerides DE440 and DE441，AJ 161:105，2021，DOI 10.3847/1538-3881/abd414](<../../document/Park 等 - 2021 - The JPL Planetary and Lunar Ephemerides DE440 and DE441.pdf>) | §2.4，p.4，式 (8)–(19)：姿态与框架；§3.2，pp.5–6，式 (28)–(34)：figure 场与动态二阶系数；§4，pp.7–10，式 (42)–(58)：自转、测地进动、形变和核幔力矩 |

论文中的物理约束不意味着要复制其软件结构。开发期仍直接修改内部接口，
不增加旧配置兼容、备用姿态源、自动猜测核参数或通用多刚体框架。

### DE430／DE440 与长时段版本的区别

| 项目 | DE430／DE431 | DE440／DE441 | 对 LunarOps 的要求 |
|---|---|---|---|
| 地幔和流核 | 非弹性地幔、固定在地幔坐标中的轴对称核边界 | 同一基本结构 | 核角速度用地幔分量；核边界形状不由核 Euler 角决定 |
| 输运交叉项 | 式 (34)、(35) 使用 `omega_m` | 式 (43)、(45) 使用 `omega_m - nu_gp` | DE440 必须包含式 (44) 的测地进动；姿态运动学仍按式 (12)–(14) |
| Moon figure 对点质量的作用对象 | Earth、Sun、Mercury、Venus、Mars、Jupiter | 增加 Saturn | Earth／Sun 二阶模型只是首阶段；模型完整性按实际源列表及 degree/order 记录 |
| PA 的确定 | 文中以 LLR 分析确定，惯性框架为 ICRF2 | 采用 GRAIL 分析的 PA，惯性框架为 ICRF3 | 初值、球谐场、反射器坐标和 PA→MER 常数必须同源 |
| 长时段版本 | DE431 关闭核幔黏性阻尼 | DE441 关闭核幔黏性阻尼 | 仅令黏性参数为零；仍保留流核、核扁率、压力耦合和地幔形变耗散 |

DE430／DE440 保留 LLR 可观测的核幔阻尼；向过去积分时误差会增长，
因此 DE431／DE441 为长时段需求关闭这项阻尼。不能据此在现代 LLR 模型中
默认关闭耗散，也不能用反向长弧中的自由核模态增长直接判定积分器错误。
各版本还使用各自拟合的参数和初值；只把 DE440 的黏性参数清零并不会复现 DE441 文件。

推荐先建立可逐项核对的 DE430 物理基线，再补齐 DE440 差异以服务现有 DE440／LLR
配置。共用运动学、惯量和力矩计算核；用明确的物理参数和模型组成区分实验，
不建立两套传播程序，也不把 `model: de440` 作为“所有物理已经实现”的标签。

## 首先固定状态和物理约定

采用 Hamilton、标量在前的单位四元数 `q=(q0,q1,q2,q3)`，
表示地幔 PA 到所选惯性轴的主动旋转 `R(q)`；地幔角速度 `omega_m`
以地幔 PA 分量表示。时间为 TDB 秒，位置 m，角速度 rad/s。
对照 DE 方程时采用对应 ICRF 轴；当前 LunarOps 的 PA→LCRS 服务将这套
姿态矩阵用于其 LCRS 轴，需明确这是怎样的轴约定。相对论空间坐标变换与轴旋转
是不同操作，不能因为 provider 名字含 LCRS，就认为测地进动已经计算。
必须用一个简单的绕 z 轴正转测试固定乘法顺序和符号：

```text
R_dot = R [omega_m]_cross
q_dot = 0.5 * q ⊗ (0, omega_m)
```

四元数归一化是数值方案的一部分，不能在任意 RHS 阶段偷偷修改输入状态。
先记录范数漂移；若在接受节点做投影，需要同步检查 ABM 导数历史的一致性。
输出插值和延迟插值要保持相邻四元数同号，输出姿态再单位化，避免把 q 与 -q
之间的直线插值当成物理姿态。主传播初版只支持这一种姿态表示。

两篇论文都积分核 Euler 角，但其角速度 RHS 和核幔力矩不依赖这些角，
且核惯量固定在地幔系内。因此在本文的轴对称核模型下，只保存
`omega_c_in_mantle` 就能闭合地幔／核动力学，并提供地幔姿态给 LLR。
核 Euler 角在此模型中属于可后处理的诊断量；以后若改变核惯量／边界模型，
使其真正依赖核取向，再增加核姿态状态。删除的是不参与反馈的状态，不能删除核角速度。

状态布局如下：

| 阶段 | 状态 | 数值分量数 |
|---|---|---:|
| 刚体自转、规定平动 | `[q_m, omega_m]` | 7 |
| 轴对称地核、规定平动 | `[q_m, omega_m, omega_c_in_mantle]` | 10 |
| 平动＋上述自转 | `[r_M/E, v_M/E, q_m, omega_m, omega_c_in_mantle]` | 16 |

四元数有一个单位范数约束；表中是数值数组长度，不是独立物理自由度。
PA、ME、ICRF 轴和 LCRS 坐标变换必须在边界明确，不能用同名“fixed frame”混用。

### 初值、单位和 PA／MER 配准

论文旋转矩阵是被动矩阵。DE430 式 (10)／DE440 式 (8) 给出
`R = Pz(-phi) Px(-theta) Pz(-psi)`；若使用主动基本旋转，则为
`R = Az(phi) Ax(theta) Az(psi)`。先与现有
[CalcephLunarOrientation](../lunarops/classes/ephemerides/lunar_orientation.py)
的矩阵逐时刻比对，再转成四元数，不直接复制 GROOPS 的乘法次序。

若星历查询返回的是 Euler 角及其导数，先按论文的运动学关系转换：

```text
omega_m,x = phi_dot*sin(theta)*sin(psi) + theta_dot*cos(psi)
omega_m,y = phi_dot*sin(theta)*cos(psi) - theta_dot*sin(psi)
omega_m,z = psi_dot + phi_dot*cos(theta)
```

Euler 角导数不是角速度分量。DE430 表 7 的角速度单位为 rad/day，
需除以 86400；角加速度、以 day 为单位的延迟和黏性速率也分别换算。
该表初始 epoch 是 JD(TDB) 2440400.5，不能把其角速度直接放到任意启动日期。
矩阵差分可作独立核对，不作为已经拥有角速度状态后的生产计算路径。

公开 libration 数据不能提供完整流核初值。使用 DE430 表 7，或与目标 DE
对应的原始积分常数和初始 epoch；在别的起点需要先传播这组核状态，或对 LLR
估计核初值并报告不确定性。不能令 `omega_c=omega_m` 作为默认兜底，
也不能只给参考星历的地幔姿态就宣称已经恢复其流核解。

PA→MER 必须写出两个版本：DE430→DE430 MER、DE430→DE421 MER
分别见式 (19)、(20)；DE440→DE421 MER 见式 (19)，三个被动旋转角为
`(-0.2785, -78.6944, -67.8526)` arcsec，依次 Rx、Ry、Rz。
本地 [DE440 frame kernel](../../data/kernels/moon_de440_250416.tf) 还给出相应
SPICE 框架定义。GROOPS 中固定的 ME 转换角不能直接用于 DE440。
先转换反射器／重力场数据到选定 PA；PA→MER 仅在输出或明确的输入边界使用，
不能作为一个小姿态修正叠加进 RHS。当前反射器＋测距偏差估计的坐标框架也需随结果记录。

## 惯量和力矩计算的边界

可以固定采用 `K=I/(M_M R_M²)` 和 `n=N/(M_M R_M²)`，这样 K 无量纲、
n 的单位为 s⁻²，并可复用现有归一化惯量。所有模型都使用同一个 Moon
质量与参考半径归一化；地幔和地核各自的惯量不能分别改用自身质量归一化。
现有全月静态惯量必须按模型参数拆分为地幔与核，不能把全月惯量同时用于两者。

### 全月、地幔和核惯量分别输出

按 DE430 式 (36)–(41)／DE440 式 (46)–(54)，令
`d=2*beta_L-gamma_L+beta_L*gamma_L`：

```text
A_T/(M_M R_M²) = 2*J2_undistorted*(1-beta_L*gamma_L)/d
B_T/(M_M R_M²) = 2*J2_undistorted*(1+gamma_L)/d
C_T/(M_M R_M²) = 2*J2_undistorted*(1+beta_L)/d

K_T_undistorted = diag(A_T, B_T, C_T)/(M_M R_M²)
K_c = alpha_c * K_T_undistorted,zz * diag(1-f_c, 1-f_c, 1)
K_m_undistorted = K_T_undistorted - K_c
K_m(t) = K_m_undistorted + DeltaK_tide(t) + DeltaK_spin(t)
K_T(t) = K_m(t) + K_c
```

未形变惯量／J2 与观测平均值不同，因为形变的平均值不为零。
`LunarDegree2GravityParameters.undistorted_inertia()` 与
`LunarInertiaEvaluation.total` 分别是未形变全月惯量和动态全月惯量，
不能直接作为自转方程中的 `K_m`。耦合形变评价返回 `K_T、K_m_dot` 与形变分项，
地幔方程统一使用 `K_m=K_T-K_c`，重力系数由 `K_T` 得到；常量 `K_c` 由核参数唯一确定。

以 `t_d=t-tau_m` 的位置、姿态、地幔角速度为输入。用 Moon→Earth 的向量
`r=R(t_d).T @ (r_E-r_M)(t_d)`，记 `u=r/|r|、w=omega_m(t_d)`：

```text
DeltaK_tide = -k2*(GM_E/GM_M)*(R_M/|r|)³ * (u uᵀ - Id/3)
DeltaK_spin = b * (w wᵀ - (w·w-n_mean²)*Id/3 - n_mean²*e_z e_zᵀ)
b = k2*R_M³/(3*GM_M)
```

`Id` 是 3×3 单位矩阵，`n_mean` 是固定平均角速度，与归一化力矩 `n` 区分。
潮汐张量对 `r` 的正负不敏感，但其他力矩和速度关系仍使用明确的向量方向。
`r、R、w` 都在同一延迟 epoch 求值；把当前姿态乘延迟位置是另一种模型。
spin 项中的 `n_mean²` 去除参考自转形变，不能简化成未扣参考项的 `w wᵀ`。

可独立验证的矩阵导数为：

```text
Q = r rᵀ - |r|²*Id/3
Q_dot = r_dot rᵀ + r r_dotᵀ - 2*(r·r_dot)*Id/3
c_tide = k2*(GM_E/GM_M)*R_M³

DeltaK_tide_dot = c_tide * (5*(r·r_dot)*Q/|r|⁷ - Q_dot/|r|⁵)
DeltaK_spin_dot = b * (w_dot wᵀ + w w_dotᵀ - 2*(w·w_dot)*Id/3)
r_dot = R(t_d).T @ (v_E-v_M)(t_d) - omega_m(t_d) × r
K_m_dot = DeltaK_tide_dot + DeltaK_spin_dot
```

这里 `r_dot` 是地幔坐标分量的导数，不能直接取旋转后的惯性速度。
正延迟的 `w_dot` 来自同一历史轨迹的导数；历史插值必须提供并验证状态和导数，
不能让 `K_dot` 在当前 RHS 中递归调用同一个 RHS 求当前角加速度。
若明确研究 `tau_m=0`，spin 惯量导数含当前未知角加速度，需收集为线性系统：
`[K_m + b*(|w|²*Id + w wᵀ/3)] * w_dot = 其余已知项`，
不能简单把 `K_dot` 设为零。当前传播按目标 DE 的正延迟实现。

### 地幔与流核角速度方程

DE430 基线的地幔方程应包括变惯量项：

```text
K_m omega_m_dot = n_external + n_cmb
                  - omega_m × (K_m omega_m) - K_m_dot omega_m
```

`K_m_dot` 是在地幔坐标内惯量分量的时间导数。
差分只作为上述解析导数的独立核对；同一次形变计算返回 K 与 K_dot。
求角加速度使用线性求解，不为每次 RHS 显式求逆。

规定的轴对称核惯量在地幔系内为常数时：

```text
K_c omega_c_dot = -n_cmb - omega_m × (K_c omega_c)
```

这里交叉项是 `omega_m`，因为核角速度分量是在旋转的地幔系中求导。
变核惯量模型还必须加入 `-K_c_dot omega_c`，不能继续套常量公式。
核幔内力矩在两套角动量方程中等大反向；压力耦合与黏性耗散分项输出。

DE440 式 (43)、(45) 在上述两式中分别以 `omega_m-nu_gp`
替换输运交叉项的 `omega_m`：

```text
K_m omega_m_dot = n_external + n_cmb - K_m_dot omega_m
                  - (omega_m-nu_gp) × (K_m omega_m)
K_c omega_c_dot = -n_cmb - (omega_m-nu_gp) × (K_c omega_c)

nu_gp = R.T @ [(1+2*gamma_PPN)/(2*c²)
              * v_EM × (GM_E*r_EM/|r_EM|³ + GM_S*r_SM/|r_SM|³)]
```

对应 DE440 式 (44)：`r_EM=r_M-r_E、v_EM=v_M-v_E、r_SM=r_M-r_S`，
均是论文所用惯性轴中的相对状态；`nu_gp` 最后转为 PA 分量，单位 rad/s。
`gamma_PPN` 与惯量参数 `gamma_L` 不是同一个量。`R.T` 替代论文的 L
只在二者惯性轴定义一致时成立，包含固定框架偏置时先明确变换。

姿态运动学仍使用 `omega_m`；论文没有在 Euler 角运动学中减去 `nu_gp`。
不要同时在四元数运动学、provider 和力矩中重复加入进动。
可在角速度 RHS 中单独诊断 `nu_gp × (K_m omega_m)` 和
`nu_gp × (K_c omega_c)`，它们不是普通外部引力力矩。
经典无外力守恒试验须关闭 `nu_gp`；保留它时，应验证对应的进动角动量关系，
不再要求在这套惯性轴下总角动量导数严格为零。

### 核幔压力和黏性耦合

作为与 DE430 论文式 (45) 及本地 RHS 对照的轴对称核起点，固定如下符号：

```text
n_cmb = kappa * (omega_c - omega_m)
        + (K_c,zz - K_c,xx) * omega_c,z * (e_z × omega_c)
```

这是作用在地幔上的力矩；核方程使用负号。`kappa=k_nu/(M_M R_M²)` 的单位
为 s⁻¹。压力项与核扁率及旋转坐标中的输运项配套，不能只摘出一个公式。
在 DE430 经典模型中关闭外力矩和 `nu_gp`、固定惯量时，
核幔力矩应保持惯性总角动量；黏性部分的能量变化应为
`-M_M R_M² * kappa * |omega_c-omega_m|²`，因此 `kappa>=0` 对应耗散。
DE440 印刷式 (58) 则写成 `(K_c,zz-K_c,xx)*omega_m,z*(e_z × omega_c)`；
这个差异不能忽略，见后面的公式核对事项。不能只改模型名字沿用 DE430 力矩。

### 外部 figure 力矩与轨道重力场共用计算

第一项外力矩可以是 point-mass 对三轴 Moon 的重力梯度力矩：

```text
n_gravity = sum_j 3*GM_j/r_j³ * [u_j × (K_gravity u_j)]
```

`u_j` 为 Moon 指向 source 的单位向量，以 PA 表示。
无核刚体阶段 `K_gravity=K_m=K_T`。加入流核后，外部重力场的二阶矩由全月质量
分布决定，使用产生该场的 `K_T`，不能直接用拆分后的地幔惯量替换它。二阶重力系数、
惯量、核幔压力力矩的分配必须与选定的 DE 模型一致；同时明确球谐系数的
归一化约定，避免把 fully normalized 的 C20/C22 当成未归一化值。
这是基本二阶模型，不能据此宣称已经复现 DE430／DE440 的全部高阶 figure torque、
Earth figure 交互或相对论姿态项。

完整 point-mass／Moon-figure 力矩按 DE430 式 (43)／DE440 式 (56)：

```text
n_figure,j = (r_M_to_j × a_M,figure,j)/R_M²
```

这里是 Moon 自身非球形场与源 j 作用产生的 **Moon 反作用加速度**，
不包含单极点质量、Earth figure 或其他力项。若重力场核给出的是 j 的直接加速度
`a_j_due_to_Mfigure`，应先用
`a_M,figure,j = -(GM_j/GM_M)*a_j_due_to_Mfigure`。
向量和加速度都用 PA 分量，或先在同一惯性系叉乘再转 PA；方向／反作用符号须逐项测试。
使用与轨道一致的 degree/order、归一化和动态二阶系数，而不是力矩重新读取另一份重力场。
DE 的 Moon 场到 degree/order 6；先做二阶验证，再补齐选定模型的高阶和源列表。

Moon figure 力与力矩由同一个 compiled RHS 使用相同的重力场和源列表求值，
保留每个源的直接场／反作用约定；不能从最后合计的 Moon 加速度反推出分项力矩。
静态力／力矩 dataclass 仅保存物理参数，候选状态与分项累加位于 fused kernel。

Earth figure／Moon figure 是另一项，按 DE430 式 (44)／DE440 式 (57)，
在 PA 系令 `u=Moon→Earth 单位向量、p=Earth 极轴单位向量、s=u·p`：

```text
n_figM_figE = 15*GM_E*R_E²*J2_E/(2*r_EM⁵) * {
    (1-7*s²) * [u × (K_T u)]
    + 2*s * [u × (K_T p) + p × (K_T u)]
    - (2/5) * [p × (K_T p)] }
```

该式采用与外部 Moon 场一致的全月二阶矩；它不能由 point-mass 二阶力矩替代，
也不能简单把 Earth 的非球形加速度塞进前面的叉乘公式。
Earth 极轴／J2 使用所选动力学模型的长期取向和时变系数，区别于 LLR 地面站的
完整 EOP。现已分别实现 DE430 的 IAU 1976 岁差和 DE440 的 Vondrák 长期岁差，
各自保留论文规定的单项章动及拟合偏置，完成独立极轴和形状耦合力矩核对。
完整 IAU 2006/2000A CIP 是另一种引力极轴假设，不能为压低姿态残差直接替换默认模型。

形变统一产生 `K_m、K_m_dot、K_T、degree2_coefficients`。同一个形变评价供轨道力与自转力矩
消费，避免静态二阶场重复叠加，也避免轨道和自转各维护一套 Love 数与延迟参数。
力与力矩接口可并列为 `acceleration(inputs)`、`torque(inputs)`，不要要求每个点质量力
都返回无意义的零力矩，或给每个力矩添加任意目标天体掩码。

### 参数来源与公式核对事项

DE430 表 11 的 `alpha_c=0.0007、f_c=2.4623904789198150e-4`，
`k_nu/C_T=1.6365616533709530e-8 day⁻¹`，
`k2=0.024059、tau_m=0.09583054727330669 day`，只供 DE430 基线。
表中的 `k_nu/C_T` **不是**本文的 `kappa`；先乘
`C_T/(M_M R_M²)`，再除以 86400 才得到 s⁻¹ 的 `kappa`。
本地 Julia [constants.jl](../../PlanetaryEphemeris.jl/src/constants.jl)
存的是 `G*I、G*N` 及 au/day 单位，不能直接复制到 SI／归一化数组中。

DE440 参数来源核对使用 [本地 DE440 BSP](../../data/kernels/de440.bsp) 注释区，得到：

| 原始常数名 | 本地文件值 | 使用约束 |
|---|---:|---|
| `JDEPOC` | 2440400.5 | 积分初始 JD(TDB)，不是任意作业的启动 epoch |
| `K2M` | 0.02419 | 与现有 DE440 形变参数一致 |
| `TAUM` | 0.14519388258636395 | day，转换为 TDB 秒 |
| `IFAC` | 0.0004 | 核惯量比例；在参数映射时核实其分母与本文 `alpha_c` 定义一致 |
| `COBLAT` | 0.0002 | 核扁率；与选定核惯量定义成套使用 |
| `KVC` | 9.160372536654605e-9 | 保留原始名字；确认其惯量归一化定义后转换，不能直接赋给 `kappa` |
| `OMGCX、OMGCY、OMGCZ` | −0.00676312958983699、−0.001064121181161924、0.22992724827007177 | 核初始角速度；与文件初始 epoch／框架／时间单位一起导入 |

DE440 的核参数和核角速度确实与 DE430 不同。初始化应从明确的物理参数对象读取，
文件解析在装配边界完成；模型 RHS 不扫描 BSP 注释或猜测缺省常数。

公式核对时需区分以下三个差异，并通过解析推导或权威实现确认：

1. **DE440 式 (55) 的潮汐导数 yy 项。** 本地 PDF 印刷为
   `y*y_dot-(r·r_dot)/3`，缺少其他对角项的系数 2。
   由式 (54) 直接求导应为 `2*(y*y_dot-(r·r_dot)/3)`；DE430 式 (42)
   也有这个 2。本文采用上面的矩阵求导式，使用差分验证对称性及形变导数的零迹，
   不复制这个疑似排版错误。尚未据此认定存在已发布的官方勘误。
2. **DE440 式 (58) 与 DE430 式 (45) 的压力项。** 前者用 `omega_m,z`，
   后者及本地 Julia 用 `omega_c,z`。本文的 DE430 守恒／能量核对只证明 DE430
   公式自洽，不能证明 DE440 应改成同式。DE440 实现需明确采用的公式来源，
   查证作者说明／JPL 实现，或把两式作为明确的科学比较实验；不能静默切换，
   也不建立失败后回退的兼容路径。
3. **Earth figure 交互的惯量。** 论文式 (44)／(57) 标记 Moon 惯量 `I_M`，
   本地 Julia 实际把 `I_m_t` 用在该分项中。本文按全月外部重力矩 `K_T` 组织，
   逐项对照必须单独报告这一选择的差异，不能为了匹配 Julia 把动力学地幔惯量与
   外部全月惯量重新混在一起。需要 JPL 原模型证据时继续核实其实际实现。

这些是物理公式核对事项，不能用额外类型检查或兜底模型解决。
