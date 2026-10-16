# LunarOps 第二轮审计与命名规范整改建议

**日期：**2026-10-13  
**对象：**`lunarops_audit_fixed_20261013.zip`  
**比较基线：**`lunarops_code_review_20261013.zip`  
**交付范围：**只审查与提出建议；未修改上传的软件源码。

## 1. 总体评价与证据边界

新版本方向正确，科学输入的严格性、配置与数据的可追溯性、诊断信息和测试保障明显加强。建议在现有基础上稳定科学计算行为，优先清理接口命名和过时文档，不应将本轮变成一次横跨所有内部对象的大型重命名。

文件比较：旧包 339 个文件，新包 344 个；同路径变更 69 个，新增 5 个。Python 文件旧 233 个，新 236 个。新增包括 `tests/test_audit_20261013.py`、`tests/test_optional_dynamics_profile.py`、`lunarops/programs/_catalog_create.py` 和两份审计文档。项目自身整改记录声称全套测试从 438 passed 提升至 493 passed；该结果不是本轮独立复现。

本轮检查环境仅 Python 3.13，且无 `pyerfa`、`calcephpy` 和项目原生扩展，无法按项目声明的 Python 3.14 环境完成全量数值回归。在隔离的临时工作副本中，将三个 `except` 语句改成兼容 3.13 的括号形式后，两个独立配置/资源测试通过；其余一部分测试在导入 ERFA 时阻断。临时副本的语法调整不代表原源码有 Python 3.14 错误，也不应作为完整回归结论。

## 2. 新版本值得保留的整改

1. CRD 限定受支持的 epoch-event，并记录被拒绝的行和原因；双程精度策略显式选择，源 bin RMS 与入权 σ 分离。
2. IERS C04 / finals2000A 入口强化格式与字段校验；EOP 增加来源/质量标记与产品版本。
3. ICGEM 拒绝时变记录；重力系数完整性和缺项策略显式声明。
4. 星历内核的内容摘要与装载顺序进入科学指纹；资源清理显式报告错误。
5. 光时事件闭合、低仰角状态和正式估计结果发布门槛增加诊断。
6. 测试和 CI 增加科学输入失败用例、可选动力学依赖与构建验证。

**仍欠缺：**真实 CRD/MINI 批次规范与精度模型确认、外部星历完整重算、长弧月球轨道/自转积分、正式平差 WRMS/协方差、真实多 rank MPI 回归。单元测试的通过不等于这些科研验收已经完成。

## 3. 已发现的遗留问题

| 优先级 | 精确定位 | 事实及整改意见 |
|---|---|---|
| P1 | `docs/INPUTS.md:19–24` | 仍声称忽略 CRD epoch-event 和直接把 bin RMS 作为 σ，与现行代码和新精度策略冲突。重写为支持事件、显式不确定度政策及原始质量量的真实合同。 |
| P1 | `docs/PROGRAMS.md:32–38,141–158` | 示例分别缺 `crdUncertaintyPolicy` 和 `meteorologyPolicy`。对文档中的程序配置增加自动提取及 schema 验证。 |
| P1（迁移风险） | `lunarops/fileio/result_table.py:20–43,115–160` | `_unit_for_field` 根据 `_m`、`_s` 等 snake_case 后缀判断单位；贸然改成 camelCase 会写出无量纲“1”。更名科学数据列前必须显式化单位 schema，提升文件版本并保留兼容读取。 |
| P2 | `lunarops/classes/parametrization/station_range_bias.py:92–108,136–143` | 同类日期边界在 `varianceComponents` 中为 `endExclusive`，在 `stationRangeBias.intervals` 中却为 `end_exclusive`。优先统一配置键，保留 Python 数据类内部的 `end_exclusive`。 |
| P2 | `lunarops/programs/lunar_orbit.py:574–588` | `lunarRotation.core` 的 `polar_moment_fraction`、`viscosity_per_s` 和 `pressure_spin` 出现在公开 ConfigSchema 中。应改成 camelCase，再显式构建内层 Python 对象。 |
| P2 | `lunarops/programs/lunar_orbit.py:328–338,648` | `lunarDegree2Gravity` 是开放 mapping，直接 `**config` 传给 `LunarDegree2GravityParameters`；造成公开配置必须服从 Python 属性的 snake_case。应设置完备嵌套 schema、字段显式映射，不采用无条件递归改名。 |
| P2 | `lunarops/estimation/adjustment_result_models.py:143–168` | `publicationQuality` 与大量 `termination_reason`、`equation_evaluations` 等 snake_case 混用于一份对外处理报告。建议在报告序列化层统一，而不是更改估计器内部数据类。 |
| P2 | `lunarops/programs/lunar_orbit.py:855–916,978–990` | 轨道元数据同一层混用 `scipyVersion`、`resolvedForceInventory` 和 `time_scale`、`step_seconds` 等。报告 schema 需要一次显式版本化整理。 |
| P2 | `docs/AUDIT_REMEDIATION_20261013.md` 与 `fileio/formats/crd.py:86,94`、`fileio/normal_points.py:38` | 文档声称旧异常语法已改成括号形式，源码仍存在无括号语法。PEP 758 说明其在 Python 3.14 合法；这是整改记录与代码不一致，不是 3.14 P0 故障。 |

### 单位推断的隔离验证

`residual_sigma_m -> m`，`residualSigmaM -> 1`；`duration_s -> s`，`durationS -> 1`；`wavelength_nm -> nm`，`wavelengthNm -> 1`。因此不能以一次全文替换完成科学结果文件的列名重构。

## 4. 命名扫描范围与统计

在 `configs/*.yml` 的 YAML 映射键中，共观察到 **242 个不同字符串键**：167 个 camelCase、18 个含下划线的 snake_case、49 个简单名称、8 个其他形式。三个所谓下划线“键” `de440_major`、`sb441_asteroid`、`sb441_kbo` 实际是质量目录中的 **群组标识**（例如 expectedGroupCounts 的动态 ID），而不是代码字段名，不建议改动。

静态检查 `lunarops/` 的字段构造器 `field/number/integer/boolean/string/mapping/sequence/path/time/class_config/class_list` 的字面量字段名，发现 **154 个不同名称**：110 个 camelCase、40 个简单名称、4 个 snake_case。这四个具体字段是 `end_exclusive`、`polar_moment_fraction`、`pressure_spin`、`viscosity_per_s`。

对于所有可解析 Python 模块，类和函数定义大多自然采用 Python 的 snake_case/PascalCase；这不是配置层的缺陷。字典字面量中另有许多 snake_case 键，但它们同时包含内部临时字典和对外输出，不能将全部字典键统计直接解释为“公开接口违规”。

## 5. 建议采用的正式命名制度

| 场景 | 正式样式 | 处理方式 |
|---|---|---|
| YAML/program/class 配置键 | `lowerCamelCase` | 严格规范；类型名、`type` 取值和用户定义 ID 不纳入自动转换 |
| 对外结构化处理报告/元数据 | `lowerCamelCase` | 在统一 serializer 显式映射，变更时提升 artifact 版本 |
| Python 模块、函数、局部变量、参数、数据类属性 | `snake_case` | 遵循 PEP 8；不做跨项目小驼峰化 |
| Python 类名 | `PascalCase` | 原有类名保持 |
| 科学数据列、固定文件 schema | 暂保留已经发布的列名 | 迁移时建立字段单位注册表、格式升级、双版本读取 |
| 物理量/模型代码/群组 ID | 依领域规范保留 | 如 `J2`、`C20`、`de440_major`、`figure_moon`；不做机械规范化 |
| 配置数值单位 | 统一清晰单位后缀 | 如 `radiusM`、`meanMotionRadS`、`tidalDelayDays`；不要只凭名称推断物理量 |

所有配置字段应采用开头小写、后续单词首字母大写的驼峰。数字标号遵循现存约定，例如 `j2Undistorted`、`c21Static`；缩写按已发布接口保持一致，例如 `utcOffsetHours`、`referenceJdTdb`、`gmM3S2`。布尔值优先使用 `include/allow/enable/compute` 开头。内部 Python 属性中的 `_m`、`_s`、`_rad_s` 等保留，不必为了表面一致而删除单位信息。

### 建议的精确迁移清单

| 现有对外配置键 | 建议改为 | 位置 |
|---|---|---|
| `end_exclusive` | `endExclusive` | `stationRangeBias.intervals` |
| `polar_moment_fraction` | `polarMomentFraction` | `lunarRotation.core` |
| `viscosity_per_s` | `viscosityPerS` | `lunarRotation.core` |
| `pressure_spin` | `pressureSpin` | `lunarRotation.core` |
| `j2_undistorted` | `j2Undistorted` | `lunarDegree2Gravity` |
| `love_k2` | `loveK2` | `lunarDegree2Gravity` |
| `tidal_delay_days` | `tidalDelayDays` | `lunarDegree2Gravity` |
| `radius_m` | `radiusM` | `lunarDegree2Gravity` |
| `mean_motion_rad_s` | `meanMotionRadS` | `lunarDegree2Gravity` |
| `attitude_derivative_step_s` | `attitudeDerivativeStepS` | `lunarDegree2Gravity` |
| `c21_static` | `c21Static` | `lunarDegree2Gravity` |
| `s21_static` | `s21Static` | `lunarDegree2Gravity` |
| `s22_static` | `s22Static` | `lunarDegree2Gravity` |
| `include_tidal_deformation` | `includeTidalDeformation` | `lunarDegree2Gravity` |
| `include_rotational_deformation` | `includeRotationalDeformation` | `lunarDegree2Gravity` |

其中 `lunarRotation.deformation` 已采用 `loveK2`、`meanMotionRadS`、`c21Static` 等小驼峰值，建议据此统一对应的 `lunarDegree2Gravity` 命名。注意两处配置分别控制不同模型，不应仅因名字接近就合并两个物理开关。

**保留不改的例子：**群组名 `de440_major` / `sb441_asteroid` / `sb441_kbo`；模型标识 `iau2006_2000a`、`figure_moon`；Python 字段 `end_exclusive`、`tidal_delay_days`；当前版本的科学结果列 `residual_sigma_m`、`duration_s`。

## 6. 改造边界与做法

**推荐：**外部配置 → 配置 schema 校验 → 显式字段映射 / typed adapter → Python 内部对象。禁止配置层直接把任意 mapping `**dict` 传入科学模型构造器，除非 mapping 的键名被正式定义为 Python 参数名的 API（此项目不建议继续采用）。

`lunarDegree2Gravity` 应获得专门的完整嵌套 schema。需要明确声明全部字段、类型、单位、默认值、允许范围和是否可空。模型构造时只映射已验证字段；禁止对未知字段做静默大小写纠正。`lunarRotation.core` 可以继续保留内部 snake_case 数据类，不必同步更名其属性。

`ConfigSchema` 的字段定义需要统一规范检测；不能仅靠 YAML 搜索。对于自由映射（台站 ID、参数名、群组名、力项清单等），应由专用 schema 标记为 `freeformIdentifiers`，禁止对值和动态键实施固定字段规则。建议提供“打印已解析规范配置”和“列出格式字段及单位”的能力；由统一的 schema 自动形成配置目录和 GUI 输入提示，这与 GROOPS 使用形式化配置 schema 的思路一致。

**不建议：**全仓文本替换 snake_case；对所有字典递归地 `snake_to_camel`；长期同时接受两个别名但不给出优先级；重命名模型代码、参数标识、状态字段后却不增加格式版本；将 Python 内部所有方法改成小驼峰。

## 7. 按风险分阶段实施

1. **阶段 A：冻结基线并修复文档。** 保存一套已通过的科研输入清单和当前数值结果；修复 CRD、预报及 PEP 758 记录；确认旧包测试记录的环境和数据版本。
2. **阶段 B：只改对外配置层。** `endExclusive`、`lunarRotation.core`、`lunarDegree2Gravity` 最优先。迁移所有 configs、用例和 schema 帮助文档；内部 Python 数据类属性保留。若项目尚无正式外部用户，建议一次性切换新键，并严格拒绝旧键，避免永久别名。
3. **阶段 C：对外报告/元数据统一。** 制作 `processingReport` 和 `lunarOrbitMetadata` 的键名对照表，在序列化边界输出 camelCase；修改 artifact 版本，增加旧报告读取/转换策略。内部求解对象不改名。
4. **阶段 D：评估是否迁移科学表头。** 默认暂缓。若确需 camelCase，先将单位从列名推断改为显式列 schema，再同时提升 `observationPrediction`、`predictionWindow`、`observationResult` 等受影响类型的格式版本。旧版本应能按其原 schema 被可靠识别。
5. **阶段 E：上 CI 规范检查。** 全量枚举有效 ConfigSchema 字段，对输出 artifact 显式 schema 逐一检查命名、类型、单位、版本和读写对称性。

### 每阶段必须通过的验收

- 新配置可通过 `describe-program`/schema 严格验证；旧键、未知键和新旧键并存要报精准路径错误；大小写相近但意义不同的字段不发生碰撞。
- 更新全部正式 YAML 示例，变量展开/嵌套引用后仍保留相同模型实例、单位及科学参数值。
- 配置迁移前后在 **相同输入/相同科学设置** 下获得一致的内部解析对象和数值结果；输出文件名和指纹变化不当作物理量变化。
- 标准 CRD/MINI 和官方 EOP 的失败注入用例继续成立。
- EOP、星历、重力场、光时、平差、长弧与正式 MPI 的科研回归必须按已定义阈值通过；不得为了“兼容旧结果”调宽阈值。
- 输出文本保存单位不变；若数据列改名，应验证 m/s/角度的单位声明不丢失，旧文件能正确读取或明确拒绝。
- 续跑 state、处理报告、科学表格的格式版本与来源指纹逻辑一致；旧 state 不得以错误字段悄悄继续运行。

## 8. 最终决策

**推荐方案：保留 GROOPS 风格的外部 camelCase；保留 Python 内部 snake_case；按 artifact 类型分别处理报告与科学数据列。**

本轮最先改的不是全仓数千个下划线变量，而是公开 schema 中的 4 个不合规字段、`lunarDegree2Gravity` 的开放映射以及 `stationRangeBias` 的区间字段；随后修订对外报告/元数据。真正的大规模变更应集中在**序列化边界、schema 和样例配置**，而不是月球动力学和光时求解的数值内核。

## 9. 核对的官方来源

- GROOPS 配置 GUI 与 XSD：https://groops-devs.github.io/groops/html/general.gui.html
- GROOPS 官方仓库：https://github.com/groops-devs/groops
- Astropy Python 编码规范：https://docs.astropy.org/en/latest/development/codeguide.html
- Python 3.14 PEP 758：https://peps.python.org/pep-0758/
