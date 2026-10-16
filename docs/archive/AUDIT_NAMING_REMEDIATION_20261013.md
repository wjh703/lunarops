# 2026-10-13 第二轮复审整改记录

> 这是第二轮验证历史记录。现行接口以 [CURRENT_DATA_CONTRACT.md](CURRENT_DATA_CONTRACT.md) 为准；此前允许的历史报告读取已删除。

对应 `LunarOps_second_review_naming_convention_20261013.md`。本轮修改公开配置与序列化合同，保留内部 Python 命名、数值求解器、科学表格列名和科研验收阈值。审计文档原件未改写。

## 已完成与验证边界

| 阶段 | 实施结果 | 验证 |
|---|---|---|
| A 文档与基线 | 修复 INPUTS 的 CRD 事件/精度合同、PROGRAMS 的精度和气象策略；补齐程序示例的输入、输出和 parametrization；修正第一轮 PEP 758 记录 | 冻结上一版 ZIP 摘要；保存 20 份展开后的配置快照并比较；三个异常捕获实际改为括号形式 |
| B 公开配置 | 15 个旧键切换为 camelCase；月球二阶重力补齐全部 13 个字段的类型、默认值、单位、范围与不可空声明，显式构造内部参数；core 显式构造对象 | 旧键、未知键、混合新旧键、非法值、大小写错误均拒绝；13 个内部参数及默认值保持一致；20 份正式 YAML 通过严格 schema 校验 |
| C 报告/元数据 | processingReport、lunarOrbitMetadata 写版本 2；统一 camelCase；只读当前版本 2，版本 1 拒绝；对应分析脚本迁移 | 文本/压缩当前报告往返、旧版拒绝、版本拒绝、类型检查、ID 保留及生产路径集成测试通过 |
| D 科学表头 | 保留既有列名和文件版本 | residual_sigma_m、duration_s、wavelength_nm 的 m/s/nm 声明与读写保持；24 份数值表的列名、形状、浮点值逐项完全一致 |
| E 规范门槛 | ConfigSchema 构造时检查字段名；新增 CI 专项；自动提取用户文档程序 YAML 并验证；开放 ID 映射标记 freeformIdentifiers | 全部注册程序/类及嵌套 schema 检查通过；describe-config、resolve-config、describe-artifacts 可查询规范与单位 |

本机环境为 Python 3.14.6。当前全套测试 **572 passed**，上一轮基线为 493 passed。本轮新增公开合同测试 79 项。源码 compileall、Ruff E9/F/I、git diff --check 通过；sdist 和 CPython 3.14 wheel 构建与包内容检查通过，构建产物在 `/tmp/lunarops-second-review-dist`。

在同一 Python 环境及未改变的本机原生扩展下，从 `lunarops_audit_fixed_20261013.zip` 解压上一版代码，分别执行同一组 11 项合成轨道测试。双向积分、EIH、核/幔自转、figure torque、环状源及时间变化 GM 路径得到的 24 份轨道/加速度/自转表，列名、形状和浮点数据逐项完全一致，不使用放宽容差。6 条合成法点的台站偏差平差同样得到完全一致的参数、标准差与协方差；版本转换后报告除来源指纹、科学输入路径/代码摘要、耗时信息外完全一致。

验证清单及数值数组 SHA-256 见 [SECOND_REVIEW_VALIDATION_20261013.json](SECOND_REVIEW_VALIDATION_20261013.json)。这些是合成短弧与小型平差的等价性证据，不能替代真实 CRD/MINI 批次确认、外部星历完整重算、长弧积分、实测 WRMS/协方差或真实多 rank MPI 验收。前一轮记录中待科研验收事项仍保留。

## 配置迁移

旧键不再作为运行配置别名接受。未知键和新旧键并存均失败，不做自动大小写纠正。所有现存正式 YAML 已迁移，变量/循环展开后的科学参数值、开关、单位与模型 ID 保持一致。

| 配置位置 | 旧键 | 新键 |
|---|---|---|
| stationRangeBias.intervals | end_exclusive | endExclusive |
| lunarRotation.core | polar_moment_fraction | polarMomentFraction |
| lunarRotation.core | viscosity_per_s | viscosityPerS |
| lunarRotation.core | pressure_spin | pressureSpin |
| lunarDegree2Gravity | j2_undistorted | j2Undistorted |
| lunarDegree2Gravity | love_k2 | loveK2 |
| lunarDegree2Gravity | tidal_delay_days | tidalDelayDays |
| lunarDegree2Gravity | radius_m | radiusM |
| lunarDegree2Gravity | mean_motion_rad_s | meanMotionRadS |
| lunarDegree2Gravity | attitude_derivative_step_s | attitudeDerivativeStepS |
| lunarDegree2Gravity | c21_static | c21Static |
| lunarDegree2Gravity | s21_static | s21Static |
| lunarDegree2Gravity | s22_static | s22Static |
| lunarDegree2Gravity | include_tidal_deformation | includeTidalDeformation |
| lunarDegree2Gravity | include_rotational_deformation | includeRotationalDeformation |

`beta`、`gamma`、`flattening` 原名保留。二阶重力要求 0 < gamma < beta < 1、正 J2/半径/平均角速度/姿态差分步长、非负 Love 数与延迟；静态项为有限无量纲数，两个形变开关为 bool。允许零延迟，与既有内部模型合同一致。`lunarRotation.deformation` 和 `lunarDegree2Gravity` 分别控制积分自转与规定姿态模型，未合并开关。

Python 数据类继续使用 `end_exclusive`、`polar_moment_fraction`、`tidal_delay_days` 等属性。`de440_major`、`sb441_asteroid`、`sb441_kbo`、`figure_moon`、参数名及观测 ID 都保留，固定字段检查不对动态映射键或模型标识实施转换。

## 当前报告与续跑合同

本节已由 2026-10-14 唯一现行合同取代此前历史读取方案，详见 [CURRENT_DATA_CONTRACT.md](CURRENT_DATA_CONTRACT.md)。

- 两类报告只读写版本 2；旧版本、旧键、混合键及不完整记录均拒绝，没有迁移入口。
- ReportSchema 只声明当前字段、类型、单位、required 与 nullable。内部求解对象由单向 serializer 输出当前字段；读取器不接受内部诊断字典。
- 配置快照验证共享纯配置 schema，不补默认值；科学输入来源及已知固定对象有明确字段合同。动态 ID 与未知插件自由参数状态保持原样。
- 失败诊断可以写出；publicationQuality、converged 与 terminationReason 必须一致，正式产品还要求合法不确定度与满秩、正冗余。
- processingState 的现行版本仍为 1；代码、输入和配置摘要的科学一致性验证保留。旧指纹不能用于新运行续跑。
- 科学数值表与二进制容器保留各自当前版本和列单位；通用容器读取并不授予嵌入报告任何历史格式读取保证。

## 查询与文档门槛

`describe-config` 的配置目录版本升为 2，已有 GUI schema 输出包含 nested schema 的单位、默认值、范围和 freeformIdentifiers 标记。`resolve-config config.yml [--set name=value]` 在不开启科学输入资源的情况下打印完成变量/循环展开与校验的公开配置；处理步骤保留公开键，不泄露 typed plan 的 Python 属性。`describe-artifacts` 输出两类报告和固定预报/窗口表的版本、字段及单位；观测残差表按输出等级选列，其正式 schema 始终在文件头可查。

自动文档测试检查 `docs/PROGRAMS.md`、`docs/INPUTS.md`、README 中所有含 program 的 YAML 代码块，含反引号及波浪线围栏；历史审计原件和 Python API 签名不视为可运行配置。对显式 CRD/FRD 扩展名的导入配置，schema 已要求 crdUncertaintyPolicy，fixed 策略要求 crdUncertaintyTwoWayS；对没有这种扩展名但内容为 CRD 的来源，严格运行入口仍检查同一精度政策。
