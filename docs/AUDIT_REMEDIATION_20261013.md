# 2026-10-13 审计整改与验证记录

本说明对应 `LunarOps_LLR_code_review_20261013.md`，区分已实施的软件整改与尚未完成的科研验收。原审计、历史回归和外部科学数据没有覆盖或改写。

> 本文保留第一轮验证历史；接口状态已按 2026-10-14 的 [现行合同](CURRENT_DATA_CONTRACT.md) 更新，历史验证数量不代表本轮验收。

## 当前基线与验证

- 本机环境 Python 3.14.6；修改前全套测试：438 passed。
- 第一轮修改后全套测试：493 passed；新增边界、失败注入和 optional dynamics profile 验证。
- `python -m compileall -q lunarops scripts tests setup.py` 通过。
- 修改文件 Ruff 的 E9/F/I 检查及 `git diff --check` 通过。
- `python -m build --no-isolation` 成功生成 sdist 和 CPython 3.14 wheel；`scripts/verify_distribution.py` 验证两类包内容通过。构建产物位于 `/tmp/lunarops-audit-dist`。
- 基础入口在阻断 SciPy 导入的独立进程中可注册全部程序、物理模型和格式入口；动力学执行报 `install lunarops[dynamics]`。
- 官方 C04 20 文件实际读取 23631 行，本地 C04 文件读取 23578 行；日期/MJD、字段和逐样本来源验证通过。没有替换本地输入文件。
- 四个本地 DE430/DE440 静态场读入成功。地球带谐场使用显式稀疏许可；DE440 月球场缺 `(2,1)`，在相关示例中显式声明稀疏许可并保留缺项清单；DE430 月球场完整。
- 本地 finals2000A 在第 19589 行缺必要字段，严格入口明确拒绝。此次没有把未来行的 LOD/dX/dY 补为零或自动截掉。业务需要先提供完整产品或另行定义经验证的缺测模型。

审计 B-01 的“Python 语法无效”结论不适用于项目声明的 Python 3.14：该版本支持无括号的多异常捕获。第一轮仅增加构建前 compileall 门槛，交付源码仍保留无括号形式；第二轮已将这三个捕获语句实际改为括号形式，以改善工具兼容性。此前“已改为括号形式”的记录不准确；不能将原 3.13 检查结果误认作本仓库 3.14 的构建失败。

## 已实施的合同变更

### 法点入口 B-02～B-06、B-29、B-30

- CRD 仅接受 v2、正常法点 H4 数据类型 1、地面发射 epoch-event 2；其他事件以源行和代码拒绝，绝不改作发射时刻。H2 时标只支持 UTC 实现 3/4/7。头、会话闭合、C0 唯一 ID、波长、11 的 ID 引用均校验。
- H4 必须给出完整起止时间，持续时间小于一天；跨午夜按真实日期匹配。多日和闰秒 seconds-of-day=86400 输入明确不支持，未新增任意事件转换算法。
- 气象必须有限且符合物理界限；坏行进入独立 `meteorologyIssues`，不会混入法点拒绝计数。最近邻以绝对 UTC 差计算，默认最大年龄 1800 s，可显式配置。法点计数满足输入=有效+拒绝。
- `parse_crd_file` 必须传 `uncertainty_policy="binRmsAsSigma"` 或 `"fixed"`；fixed 还必须给出正的两程秒单位 σ。源 bin RMS 与入权 σ 分开记录，没有除以 sqrt(N)。前一策略是调用者有意采用的精度模型，不能视为已证明的 ILRS 法点均值误差。
- `NormalPointsConvert` 新字段：`crdUncertaintyPolicy`、`crdUncertaintyTwoWayS`、`crdMeteorologyMaxAgeSeconds`、`miniTemperatureUnit`。MINI 支持显式 `deciCelsius` / `deciKelvin`，默认保留已使用的 Celsius 处理链；不根据值大小猜测。数据批次的权威单位仍需负责人确认。
- 格式探测只按一次内容结构选择入口；已知扩展名与内容冲突失败；编码严格读取，不再忽略/替换坏字节。导入报告含路径、SHA-256、拒绝行、事件与精度/气象合同。canonical normalPoint 的 metadata 注释保存这些语义，数据列版本保持不变。
- 时间筛选及合并复制记录与索引，源对象不变；显式 `assign_indices` 仍是有意的原位操作。

### EOP、时标 B-10～B-13

- C04 14 的 16 列及 C04 20 的 21 列有独立布局；校验有限数值与日历/MJD 一致性。14 C04 的 LOD 在第 8 字段；20 C04 的 LOD 在第 13 字段。删除 compact/盲扫猜列、缺字段补零和非法字节吞噬。
- finals2000A 只读取 Bulletin A 指定列，要求完整 LOD/dX/dY 与 I/P 标记。任一必要字段缺失报源文件及行号。
- 原生 EOP 写版本 2，保留 source/quality/sourceSha256；仅读取 v2 完整来源表头和行；不为缺失来源补值。C04-over-rapid 合并保留逐样本来源；MPI 数组广播保留来源；`quality_coverage` 可查质量区间。
- `earthRotation.allowPredicted: false` 拒绝未知质量产品，并移除 prediction 样本；覆盖外查询仍拒绝。默认 true 保留既有预报能力，配置产物能见该策略。
- ERFA 日期警告恢复可见；TT→TDB 迭代超限抛异常，包含历元、迭代数与末次改变量。

### 动力学 B-15～B-21、B-35

- 静态 ICGEM 拒绝 gfct/trnd/dot/asin/acos，包括超出截断阶数的时变记录；不实现新时变求值器。
- 检查 norm、tide_system、max_degree、重复/越界/非有限系数。非稀疏域缺项拒绝；中心质心框架的 degree 1 可按已声明的零一次项合同省略。`allowMissingCoefficients: true` 是唯一稀疏许可，记录实际缺项。
- 重力元数据保存文件 GM/半径、实际阶数、覆盖来源。轨道 GM 继续来自质量目录，原文件 GM 同时可查。
- `gravityFields` 统一为 lowercase earth/moon 对象，每项必须有 path；删除 uppercase / string 别名，schema 严格验证。既有正式配置均已检查可解析。
- `bodyForceTerms` 缺键才用既有公布默认；空列表拒绝。figure_moon 配合动态二阶场必须显式包含 lunar_degree2_gravity。示例已迁移，resolved force inventory 与实际活动项写入产物。
- 需要 Earth/Moon figure 定向时缺提供者立即失败；纯质点轨道不受影响，积分自转继续在每次 RHS 用状态四元数。
- 历史和轨迹插值阶数限定 [1,13]，与现有最高阶次一致。这是保守设计范围限制，不代表所有步长/阶次已经获得长弧误差认证。历史缓存、accepted nodes、DOP853 固定子步长启动和延迟机制保留。
- SciPy 在动力学执行边界检查，普通模块注册不强制导入；CI 同时覆盖 core/dynamics profile，轨道元数据保存 SciPy 版本。

### 测距、平差、工程 B-07～B-09、B-23～B-28、B-31～B-34、B-36

- 增加光时双腿与总程事件闭合残差属性及 full diagnostics；不改变原停止条件。最小几何测试通过；不同真实星历/年代/仰角的全面闭合验收尚待科研回归。
- `lowElevationPolicy` 为 rejectClamped（配置默认）或 allowFlagged（诊断）；保留模型数学截断，正式解拒绝含截断法点。原始仰角、使用角度与标志仍保留。直接 Python API 默认 allowFlagged 是现有诊断调用合同，业务程序通过配置传 rejectClamped。
- 满秩、秩亏、内迭代、最终重新线性化、正规方程/协方差算法保留。参数报告新增 `reference_value` / `applied_update` / `final_value` / unit；final 法方程的 remaining correction 仍是未应用量，不再次加到结果。
- `publicationQuality` 区分 passed / diagnostic：不收敛、后验冗余不足、无后验 σ0、协方差非法、光时失败、clamped 或方差组不足冗余均不可正式发布。报告与 state 可以保留；WriteResults 的 solution/covariance/reflector catalog 输出受门槛保护。
- 科学输入清单根据 ArtifactSlot/schema path 遍历；CALCEPH 使用同一个排序/加载函数：全部排序 BSP 然后排序 BPC。包含相对路径、SHA-256、加载序号、声明输入内容、本地 Python/Cython/native 内容摘要及后端版本；fingerprint v2 排除展示/执行和输出槽路径。旧 processing-state fingerprint 不再兼容，须从原输入重新运行。
- 删除缓存 repr 兜底，非字符串键、自定义对象、NaN/Inf 明确拒绝。
- 预报必须声明 meteorologyPolicy=nominal/measured；nominal 的示例数值明确标记，measured 要求四个输入量。产物 metadata 保存气象、执行模式、各条件失败计数、gridDefined、步长与 inclusive 端点说明；不把采样窗口称作连续过境边界。合并检查声明步长和网格对齐，缺采样分割窗口。
- 台站/反射器创建共用一个私有 source validator/executor，公开 ProgramSpec 和坐标合同不变。
- 必需观测模型在装配边界报模型名及 YAML 位置；测距 rangeBias 必須声明，关闭改正采用明确 none。
- 显式 MPI 单 rank 报错；独立串行执行仍可用，execution metadata 标明执行模式。未更改 worker 调度顺序、初始化与错误传播。
- close 错误逐个累积，其他资源照常关闭；成功任务的严格清理失败会使运行失败；已有主失败保留首因。CLI finally 确保即使 context.close 失败也会执行 MPI shutdown。

## 尚待科研验收或后续设计的范围

B-05 的真实 MINI 批次规范与气象/精度来源、B-07/B-14 的真实星历双腿闭合和全参考系/位移独立差分、B-18/B-21/B-22 的长弧轨道/自转阶次与步长实验、B-24/B-25 的全量实测权/WRMS/σ0/协方差、B-26 的 CALCEPH 逐核时间覆盖与完整 run-level resolved manifest 发布，以及真实 MPI 多 rank 物理一致性，均未被本次单元测试替代。

本次不开发变分方程、动力学参数解算、模拟、任意 CRD 事件转换或 ICGEM 时变模型；也不移除数值历史或失败诊断。后续真实数据回归必须固定输入清单和误差预算，不能放宽阈值使错误旧结果“复现”。

## 核对的官方格式来源

- CRD v2.01: https://ilrs.cddis.eosdis.nasa.gov/docs/2022/crd_v2.01e3.pdf
- EDC 字段校验: https://edc.dgfi.tum.de/en/oc/crd/v2/
- 官方 C04 20 样例: https://hpiers.obspm.fr/iers/eop/eopc04/eopc04.1962-now

## 第二轮复审

公开命名、报告版本和文档合同的后续整改与验证见 [AUDIT_NAMING_REMEDIATION_20261013.md](AUDIT_NAMING_REMEDIATION_20261013.md)。第一轮 493 项测试记录保留为历史基线。
