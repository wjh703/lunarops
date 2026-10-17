# 源—源批量辅助量与 KBO ring 性能复核

日期：2026-10-16。配置：`configs/lunarops_orbit_rotation_de440_environment_10a.yml`。

## 已落地的改动

- 新增 `point_mass_symmetric_batch(E,N,3)`，对独立历元使用 OpenMP `prange`；每个历元保留完整的 `i<j` 牛顿加速度和势求和，供 EIH 使用。
- 标量和批量入口共用同一个 `noexcept nogil` 二维内核；没有并行化 ABM 递推或 CALCEPH 句柄。少于 8 个历元或 64 个源天体时串行执行。
- `build_epoch_data_batch()` 先批量读取星历，构造包含 KBO ring 状态及历元太阳 GM 的数组，再调用一次批量内核。单历元启动/种子仍用标量入口。
- Linux 构建可选 `LUNAROPS_ENABLE_OPENMP=1`，默认构建不要求 OpenMP。运行时可用 `lunarops._dynamics_core.openmp_enabled()` 检查。
- DE430、DE440、DE440 environment 及其 5400 s 四份常规十年 Slurm 脚本改为 32 CPU，`OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}`。保留单线程 BLAS，并设置 `OMP_WAIT_POLICY=PASSIVE`，避免 OpenMP 工作线程在串行积分期间忙等。
- 两份 environment 脚本使用包含作业号的新输出目录，保留原有结果。

构建：

```bash
LUNAROPS_ENABLE_OPENMP=1 uv pip install --no-build-isolation --editable .
# 在已有构建目录手工切换编译选项时，强制重编译：
LUNAROPS_ENABLE_OPENMP=1 .venv/bin/python setup.py build_ext --inplace --force
```

## 验证与性能

60 项相关测试通过：批量内核、动力学、太阳环境、传播程序、自转和可选 SciPy 依赖。
包括空批次、无源/单源、不同历元 GM、并行批次任意历元的重合天体报错、缓存牛顿量和月球 EIH 项的逐位对照。
另在隔离构建目录编译无 OpenMP 版本，64 历元 × 418 源天体的加速度和势与 OpenMP 版本逐位相同。
Ruff、`git diff --check` 和修改后的 Slurm 脚本语法检查通过。

Slurm 作业 **44964236**，node15，32 CPU。所有线程档在同一作业内顺序运行，内核预热后取 15 次中位数。
真实配置包含 **420 个天体**（含 36 个 ring 点），缓存源子系统为 418 个天体，每批 64 历元。
“逐历元缓存对照”在同一组装流程中逐个调用标量缓存内核，以隔离批量计算的效果。

| 路径 | 64 历元内核耗时 | 完整 60 天联合积分耗时 |
|---|---:|---:|
| 逐历元缓存对照，1 线程 | 36.35 ms | 28.001 s |
| 批量，1 线程 | 36.29 ms | 28.138 s |
| 批量，4 线程 | 9.33 ms | 27.116 s |
| 批量，8 线程 | 4.63 ms | 27.163 s |
| 批量，16 线程 | 4.91 ms | 26.881 s |
| 批量，32 线程 | 3.05 ms | 27.100 s |

32 线程内核约加速 **12 倍**；本次 60 天积分耗时减少 **3.2%**。
60 天测试含启动和刚性历史种子；这一固定成本不受历元批处理加速。
不同线程数的积分总耗时差距较小，不能据此宣布 32 线程是总耗时的最优配置。
这不是新的十年全链路计时，不能把内核倍数解释成整个积分的倍数。

逐历元对照与 1/4/8/16/32 线程的 301 个等距联合状态采样 SHA256 均为：

```text
2ed6d24b662a613047fcf4ae406cd9bcc04ede6011b882053a0c7e7223c289a7
```

原始结果：`output/source_gravity_batch_44964236/*.json`。
复现：`sbatch run_source_gravity_benchmark.sbatch`；诊断入口为 `scripts/benchmark_source_gravity.py`。

## KBO ring 是否需要数值积分

这里将用户所说的 KRO ring 理解为代码中的 KBO / Kuiper belt ring。
当前是 36 个互不吸引的点，各自受到给定太阳、行星、月球及小天体轨道的时变牛顿引力。
它们只在主积分之前预传播一次；主积分中读取 DOP853 的稠密输出，不会每个 ABM 步重跑 ring 积分。

只保留固定中心、恒定 GM 的太阳两体圆轨道时，现有 `circular_states()` 是解析解。
当前外部源随时间运动，太阳 GM 也可变化，因此这个圆轨道公式不能作为同一受扰模型的精确解。
跟随太阳即时 BCRS 位置/速度的圆环也会把太阳的反射运动直接施加给环中心，与当前独立 BCRS 受力方程不同。
连续轴对称环的解析引力同样对应另一种模型，不能直接替换受扰的离散点及其 EIH 辅助量。
Park et al. (2021) §2.2 说明了 44 au、36 等质量点的设置，并没有要求采用 DOP853 这一特定数值方法。

十年预传播实测（主机单线程，仅作成本量级参考）为正向约 11.88 s、反向种子约 0.30 s，合计 **12.18 s**；
相对之前约 760.5 s 的十年全程序耗时约 **1.6%**，两次计时并非同机同次全链路对照。
在十年内 101 个历元比较受扰数值轨道与圆周解析轨道：

| 量 | 最大差异 |
|---|---:|
| 单点位置 | 3.8533 × 10⁹ m |
| 单点速度 | 23.047 m/s |
| 月球处 ring 牛顿加速度 | 5.8299 × 10⁻¹⁷ m/s² |

对称性抵消了大量单点误差，但不能由此保证厘米级月球轨道精度。
`0.5 × max(Δa) × T² ≈ 2.90 m` 只是一项自由双积分尺度，**不是耦合轨道的误差或上界**；没有以此替代十年联合传播，也未在这个解析环诊断中比较 EIH。
因此本轮保留生产 ring 的受扰 DOP853 模型和精度设置。

## 更稳妥的 ring 优化候选

保持 DOP853、rtol 和 atol 完全相同，仅在隔离诊断中放宽最大步长上限：

| 最大步长上限 | 正反预传播 RHS 次数 | 预传播耗时 | 相对 14 天结果最大单点位置差 | 月球处最大牛顿加速度差 |
|---|---:|---:|---:|---:|
| 14 天（生产默认） | 4144 | 12.18 s | — | — |
| 28 天 | 2194 | 6.71 s | 0.0132 m | 9.77 × 10⁻²⁹ m/s² |
| 56 天 | 1219 | 3.90 s | 0.0176 m | 8.65 × 10⁻²⁹ m/s² |

这表明 14 天上限值得进一步放宽，预计可节省数秒，比换成圆周解析模型稳妥。
这里的差异来自十年 101 点采样；还需密集稠密输出、EIH 和十年联合积分对照后再调整生产默认。
DOP853 并非物理上的必要条件，但它提供自适应误差控制和高阶稠密输出；当前没有证据支持为这 1.6% 的开销重写成另一套积分器。

诊断结果：`output/source_gravity_batch_20261016/ring_step_study.json`。
复现命令：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_WAIT_POLICY=PASSIVE \
  .venv/bin/python -m scripts.benchmark_source_gravity \
  --output output/kbo_ring_step_study.json --ring-step-study --repeats 3
```

原始资料：[Park et al. 2021 PDF](<../../document/Park 等 - 2021 - The JPL Planetary and Lunar Ephemerides DE440 and DE441.pdf>)。
