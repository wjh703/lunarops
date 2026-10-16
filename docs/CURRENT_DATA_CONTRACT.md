# LunarOps 现行数据合同

本文件描述唯一现行入口。公开固定配置、报告字段使用 lowerCamelCase；Python 实现属性与科学表格列名保留其命名和单位。字段、类 type、程序名、枚举与 catalog key 逐字匹配；未知键、混合键、裸字符串类配置、历史版本均失败。读取器不改名、不补快照默认值、不重试另一种产品解析器。

## 原生制品与版本

ASCII 首个有效行为 `lunarops <artifactType> version=<currentVersion>`，只允许该类型的当前版本，`.txt` / `.txt.gz` 使用同一内容合同。

| 类型 | 当前版本 | 主要内容 |
|---|---:|---|
| processingReport | 2 | 平差结果、诊断与正式产品资格 |
| lunarOrbitMetadata | 2 | 轨道、自转模型及来源快照 |
| earthOrientationParameter | 2 | EOP 数值、来源、质量、摘要 |
| stationCatalog / reflectorCatalog | 2 | 精确 catalog ID 与坐标 |
| normalEquationInfo / normalEquationMetadata | 2 | 正规方程维度和科学约定 |
| observationPrediction | 5 | 预报字段及失败/质量状态 |
| predictionWindow | 4 | 采样窗口和网格定义 |
| normalPoint | 1 | 原生法点与双程时间/不确定度 |
| processingState | 1 | 当前续跑状态及科学指纹 |
| parameterVector / parameterName / covarianceInfo | 1 | 参数、单位与协方差 |
| massCatalog / matrix / observationResult | 1 | 质量目录、矩阵与观测结果表 |

二进制数值表使用其当前容器合同（版本 1），保留科学列与单位。版本 1 对上述仍以 1 为当前版本的类型合法，不代表历史读取。通用头探查只识别类型；实际数据 reader 必须验证精确版本。通用文本/YAML helper 的调用必须显式提供版本。

## 平差报告

以下顶层字段必需且不可为 null：`converged`、`terminationReason`、`publicationQuality`、`settings`、`summary`、`parameters`、`observations`。固定对象禁止未知字段。完整字段、单位、required 与 nullable 可由 `lunarops describe-artifacts` 查询，源定义为 `lunarops/fileio/report_schema.py`。

- `parameters[]` 至少有非空、唯一 `name` 和非空 `unit`。正式产品必须包含 `finalValue` 与非负、非 null 的 `formalUncertaintyM`；诊断允许明确声明的形式不确定度为 null。
- `observations[]` 至少有唯一非空 `observationId`、精确 `stationId`、合法 UTC `epochUtc`、布尔 `lightTimeConverged` 与正 `aprioriSigmaM`。光时收敛时必须有数值 `currentStateResidualM`、非负 `residualSigmaM`。光时失败时这两项可为空，但结果不能标为正式产品。
- `converged` 当且仅当 `terminationReason == CONVERGED`。summary 中声明的同名状态必须一致。
- `publicationQuality` 的 `formalProductEligible`、`status`、`reasons` 全部必需。正式产品为 `true / passed / []`，且必须收敛、参数满秩、有正的后验冗余和非负 `sigma0Post`。失败诊断为 `false / diagnostic / [原因...]`；未收敛时必须包含终止原因。被夹限的大气延迟、失败光时和无有效冗余不获得正式资格。
- 数值必须有限；标准差、方差、权值和权因子按各自正值/非负/[0,1] 约束验证。

缺字段与显式 null 是不同状态；只有 Schema 明确声明 nullable 的字段可为空。例如未启用的残差筛选阈值、未写出的可选输出路径、失败诊断的部分不确定度可以为 null。动态站 ID、参数名与组 ID 原样保留。已知 stationRangeBias 状态有严格字段合同；外部插件的自由参数状态独立保存。

`scientificInputs.resources` 的每个资源组为数组，每项为 `{path, sha256}` 文件或 `{path, kernels}` 星历目录。kernel 项为 `{relativePath, sha256, loadOrder}`。`codeSha256` 和 `versions` 必需。没有任意固定字段的 opaque 通道。

## 轨道元数据和配置

`model`、`timeScale`、`config` 必需且不可空。配置快照只校验已写入的当前字段，不在读入时补默认值。月球二阶重力、核、形变、Earth tide、J2 时间变化和太阳相对论配置均有共享纯配置 Schema；类声明必须包含精确 `type` 对象。部分来源快照可以只记录一部分配置，不作为续跑输入。

`resolvedForceInventory` 使用 `active` 与 `source`。`resolvedGravityFields` 每个科学天体 ID 对应严格来源记录：`path`、`norm`、`tideSystem`、`declaredDegree`、`loadedDegree`、`allowMissingCoefficients`、`missingCoefficients`、`gmSource`、`radiusSource`、`fileGmM3S2`、`fileRadiusM`。文件 GM 为 m³/s²，半径为 m；缺项为 degree/order 整数对。

运行配置的类声明示例为 `troposphere: {type: mendesPavlis}`。程序装饰器只接受 `ProgramSpec`，注册作用域只使用 `scope`。`figurePartners.earth` 表示 Earth 非球形场的受力伙伴，`figurePartners.moon` 表示 Moon 非球形场的受力伙伴；Earth 的非球形引力仍是月球运动模型的一部分。

## EOP 与独立科学输入

原生 EOP 2 表头固定为：

```text
fields mjdUtc xpArcsec ypArcsec ut1MinusUtcSec lodSec dxArcsec dyArcsec source quality sourceSha256
```

每行恰好 10 列：极移/天极偏移为 arcsec、UT1−UTC/LOD 为 s。来源与质量必须显式给出，不能自动补为 unspecified；显式 unspecified 是可见质量状态，严格无预报质量政策会拒绝未知质量。C04 和 finals2000A 保留各自独立读取入口与质量策略。

CRD/FRD 导入只支持 H1 的 CRD 2，record 11 恰好 14 字段并含 SNR；H4 为法点、epoch-event 为地面发射 2、H2 UTC 实现为 3/4/7。必须明确指定不确定度策略。H2 name/pad 一致性在头读取时检查，矛盾立即失败；H3 使用明确目标映射。坏法点行计数并记录源行，结构错误中止。MINI 是独立声明的外部产品，仍可通过 NormalPointsConvert 导入。历史 CRD v1 无法直接导入；需获得 v2 来源或在项目外独立转换。

公开模型/目录解析不轮询别名。MINI 的五位站代码、CRD 的正式字段和 NAIF 物理数字 ID 在输入边界做确定性身份关联。中心天体与系统质心保持区分；科学内核允许明确 `NAIF:<integer>`。内核生成的 KBO_RING 节点是计算源 ID。原生法点保存精确身份，声明的源站代码不得与其矛盾。

## 坐标与运行方式

公开 `hardisp()` 返回唯一顺序 `(Up, West, South)`，单位 m；BLQ 振幅与相位按同一标准轴排列。海洋潮汐负载结果属性为 `displacement_up_west_south_m`，ENU 转换为 `(-West, -South, Up)`，随后按台站位置转至 ITRF。时间尺度转换、物理轴转换和数值诊断继续保留。

显式 MPI 必须至少两个 rank（主 rank 加至少一个 worker）；单 rank 报错。普通串行执行使用独立串行路径。科学指纹和正规方程的科学约定一致性校验继续保留，不是历史接口兼容机制。

## 持续验收

`python scripts/check_current_contract.py` 在 CI 的两种安装 profile 中检查生产 AST 的历史接口钩子、显式版本和旧报告正向测试。行为测试覆盖版本/字段/身份拒绝、合法诊断、当前制品往返和 MPI rank 1；HARDISP 冻结全振幅/相位、IERS Fortran 与 Orekit 参考阈值保持。该静态检查是额外门槛，不能替代真实科学输入和长弧误差验收。
