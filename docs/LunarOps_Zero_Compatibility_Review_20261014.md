# LunarOps 零历史兼容层整改建议

**基于 2026-10-14 提交源码的复审修订版**  
审计基线：`lunarops_current_source_20261014.zip`  
文件根目录：`llrops_project_dev/`  
范围：LLR 已实现模块的配置、文件读写、报告输出及其关联的兼容入口。**此文仅提出整改方案，不改动业务源码。**

## 一、执行决策：不留过渡期、不保留兼容入口

LunarOps 今后采用**唯一现行接口合同**：公开配置严格使用 lowerCamelCase；Python 数值内核使用 snake_case；每种原生制品只读写其当前格式版本；旧版本文件、旧键、简写别名与混合命名一律报错。**不提供迁移函数、兼容模式、旧版读取器、降级参数、自动修复、静默别名映射或旧测试正向通过路径。**

需要澄清：本建议的“兼容层”是**为继续接受旧软件接口或历史格式而存在的代码**。版本号与拒绝旧版本的检查必须保留；科学领域所需的坐标轴转换、时间尺度转换、观测质量筛选、收敛与数值有效性诊断，不能因名称中出现 `convert` / `fallback` 就删除。对外部科学标准，建议采取严格的单一版本合同：当前 ILRS CRD 仅接受 v2；MINI、IERS C04、finals2000A 仍可作为**分别声明、分别校验的独立输入产品**，不能靠试错回退互相冒充。

**优先级建议**：P0 = 必须先改，存在已证实历史版本接受路径；P1 = 同批完成，消除双命名或混合 Schema；P2 = 对外入口完全严格化；P3 = 数值/坐标公共 API 收口，需更严格的科学回归。

## 二、已独立核实的证据

1. `lunarops/fileio/structured_reports.py:20–32` 通过 `supported_versions=(1, 2)` 读取报告；`:57–60` 依据 `version == 1` 启用源字段名；`:63–100` 包含 `_migrate_orbit_config`；`:130–135` 对历史轨道配置执行迁移。
2. 隔离调用证实：带 `lunarops processingReport version=1` 头及旧 snake_case 字段的报告，**当前确实能被 `read_processing_report()` 接受并翻译成 v2**；`write_processing_report()` 则写 v2。这是明确的待删除兼容功能，不是推测。
3. `lunarops/fileio/report_schema.py:15–19,29–62,69–73,94–104` 通过 `source`/`name`、`source_names`、`legacyName` 和 `old=new` 的声明同时承担“内部序列化”和“历史命名迁移”两项职责。删除历史分支时要连带改生产方，而非只删除布尔参数。
4. `lunarops/fileio/earth_orientation.py:81–127` 允许 EOP 原生格式 v1/v2，以及“不带来源列”的旧表头，旧数据缺少来源时使用 `EarthOrientationSample` 的默认来源。这是第二条独立的版本兼容路径。
5. `lunarops/fileio/formats/crd.py:174–185,225–252` 允许 ILRS CRD v1/v2，对 record 11 长度与 SNR 字段有基于版本的分支。ILRS 官方说明 CRD v1.01 已在 **2022-08-01** 被 v2.01 取代；若坚持只接收现行外部标准，v1 路径应移除。来源：`https://ilrs.gsfc.nasa.gov/data_and_products/formats/crd.html`。
6. `lunarops/fileio/structured_reports.py` 及 `report_schema.py` 的缺字段问题依然存在：隔离测试中 `parameters: [{}]`、`observations: [{observationId: null}]` 可通过当前 writer。严格单版本必须**同时严格内容合同**，不能把“只读 v2”误认为“v2 数据一定有效”。

## 三、必须删除：原生报告与元数据（P0）

| 编号 | 精确文件位置 | 删除内容 | 唯一替代规则 |
|---|---|---|---|
| ZC-01 | `fileio/structured_reports.py:20–32` | `_read_versioned()` 及逐行猜测/解析 v1/v2 的流程 | 报告 reader 直接复用原生严格读取函数，**仅 `expected_version=2`**；不需返回版本值分支 |
| ZC-02 | `fileio/structured_reports.py:57–60` | `source_names=version == 1` | `read_processing_report()` 只做 v2 严格读取与验证；v1 报错，不翻译 |
| ZC-03 | `fileio/structured_reports.py:63–100` | 整个 `_migrate_orbit_config`、旧键对照字典及历史快照自动改名 | 轨道配置快照只接受当前 `lunarDegree2Gravity`、`lunarRotation.core` 公布的字段 |
| ZC-04 | `fileio/structured_reports.py:130–135` | 旧元数据分支、`source_names`、配置迁移调用 | `read_lunar_orbit_metadata()` 只读严格 v2 |
| ZC-05 | `fileio/structured_reports.py:5–14` | 因 ZC-01 / ZC-03 不再需要的 `StringIO`、`parse_header` 兼容依赖及多余 `yaml.load` 读取逻辑 | 直接使用 `read_structured_text(..., version=2)`；保留必需的 v2 深层 Schema 校验 |

**验收**：`processingReport` / `lunarOrbitMetadata` 的 v1 文件，无论包含旧键还是伪装成新键，均必须因头版本不等于 2 被拒绝；v2 报告混入 `termination_reason`、`time_scale`、`radius_m` 等旧键必须被精确路径拒绝，不能默默删除或改名。v3 同样明确拒绝。

### 三点不可遗漏

- **保留 `REPORT_VERSION = 2` 与 `METADATA_VERSION = 2`**，并在文件头检查“严格等于 2”；不是把版本字段删除。
- 读取 v2 时禁止对未声明字段执行 snake→camel、嵌套配置补默认值、宽松合并；公开 Schema 是唯一来源。
- `structured_reports.py:103–120` 当前为了校验 `lunarRotation.core` 和 `lunarDegree2Gravity`，反向导入 `programs.lunar_orbit`。建议同时把共享配置 Schema 下沉至纯配置模块，不要因为删掉迁移器就保留这一反向依赖。

## 四、必须删除：双命名 ReportSchema（P1）

| 编号 | 精确位置 | 删除 / 改造 | 影响 |
|---|---|---|---|
| ZC-06 | `fileio/report_schema.py:15–19` | `ReportField.source`：报告 Schema 最终只存**一个**对外字段名及类型/单位/必需性 | 所有 `record()` 声明需改为单名字段 |
| ZC-07 | `fileio/report_schema.py:29–62` | 删除 `source_names` 参数及其递归传递、按 `source/name` 切换查找的分支 | `convert` 不再“转换”，建议改名 `validate` / `normalize_canonical`，只接受规范公开键 |
| ZC-08 | `fileio/report_schema.py:69–73` | 删除 `describe()` 的 `legacyName` | `describe-artifacts` 仅展示现行 `name/type/unit/required/nullable` |
| ZC-09 | `fileio/report_schema.py:94–104` 与全部 `source=public` 声明 | 删除旧字段名等号语法；固定字段只声明当前 `terminationReason`、`residualSigmaM` 等 | 不能用“保留映射表但不开放 reader”来规避删除要求 |
| ZC-10 | `estimation/adjustment_result_models.py:143–172` | 删除 `PROCESSING_REPORT_SCHEMA.convert(...source_names=True)` | `to_dict()` 或独立 typed serializer **直接构造当前 camelCase 报告**；内部对象属性仍 snake_case |
| ZC-11 | `programs/llr_processing.py:297–321` | 删除 `array(PROCESSING_STEP).convert(...source_names=True)` | `processing_results` 在构造时即用标准公开字段，直接验证后写入 |
| ZC-12 | `programs/lunar_orbit.py:987–1052` | 删除轨道元数据 `convert(...source_names=True)` | 由单一 canonical metadata builder 明确生成新字段；writer 做**一次**严格校验 |

**关键架构原则：**内部 Python 对象采用 snake_case 是实现语言约定，不是旧接口；仍可有负责“计算对象→当前报告对象”的**单向序列化函数**，但它不接受旧字典、不选择两套字段名、不执行多版本修复。如果要最纯粹的单入口，报告生产方直接填充公开字段，然后由只读 Schema 校验，不再提供含别名的通用转换器。

### 与上轮 P1 问题合并整改：严格内容合同

- 为 `ReportField` 规定 `required`、`nullable`，取消所有字段默认 `nullable: true` 的描述；缺字段和显式 `null` 分别报错。
- `parameters[]` 不得出现空对象或 `name/unit` 缺失；`observations[]` 至少有合法观测身份，需依 `formalProductEligible`、光时状态等定义数值量哪些必需、哪些可空。
- `publicationQuality`、`converged`、`terminationReason` 的组合进行交叉验证：失败诊断可合法写出，但不得装作可发布正式产品。**诊断状态不是旧格式兼容。**
- `opaque` 只用于确属外部 ID/自由参数的数据，不允许掩盖应该有 Schema 的固定字段。动态台站 ID、参数名不受 camelCase 自动改写。

## 五、必须删除：原生 EOP 的 v1 分支（P0）

| 编号 | 精确位置 | 处理建议 |
|---|---|---|
| ZC-13 | `fileio/earth_orientation.py:81–84` | 将 `supported_versions=(1, FORMAT_VERSION)` 改成严格 `expected_version=FORMAT_VERSION`，当前 `FORMAT_VERSION = 2` |
| ZC-14 | `fileio/earth_orientation.py:94–106` | 删除两套 `fields` 表头识别及 `has_provenance` 分支，只接受 `FIELDS + PROVENANCE_FIELDS` 完整表头 |
| ZC-15 | `fileio/earth_orientation.py:106–123` | 每行必须恰好包含当前全部字段；去掉“没有 `source/quality/sourceSha256` 就使用默认值”的老版本恢复流程 |

**注意**：EOP 缺来源时必须拒绝读取而不是伪造来源。当前格式中如果来源字段明确写入字符串 `unspecified`，这是具体的数据质量状态，是否允许进入求解应由科学质量政策控制，不应再因为“兼容 v1”被自动填写。C04、finals2000A 是两个独立的、明确定义的外部产品接口，**保留各自解析器**，不要合成“随便读哪个都试”的回退链。

## 六、删除历史外部 CRD v1：采用 CRD v2 唯一合同（P1）

| 编号 | 精确位置 | 处理建议 |
|---|---|---|
| ZC-16 | `fileio/formats/crd.py:98–132,153–160` | `_CrdSession` 不再需要版本默认值 `crd_version=1`；Parser 仅声明 CRD v2 |
| ZC-17 | `fileio/formats/crd.py:174–185` | H1 只接受 `CRD 2`，明确拒绝 `CRD 1`；不试图按旧格式修补 |
| ZC-18 | `fileio/formats/crd.py:225–252` | record 11 固定按 v2 长度和 SNR 字段解释，删除 `if crd_version == 2 else ...` |
| ZC-19 | `fileio/formats/crd.py:412–421,462` | 修正文档与来源标识，改为唯一 `crd-v2`；移除已无用途的版本传播变量 |

**科学后果要写明：**这样做会使历史 CRD v1 数据**无法直接导入当前 LunarOps**。ILRS 官网标明 v1 于 2022-08-01 被 v2 取代（详见本报告参考）。若研究涉及历史 LLR 数据，必须事先拿到官方 v2 文件或在项目外、由第三方独立规范工具完成转换；**不要在 LunarOps 里保留 v1→v2 迁移程序或开关**。MINI 是另一种明确命名的观测源类型，不能仅因年代较早就误认为“软件旧版兼容层”；若项目正式决定不再研究 MINI 数据，才将 MINI 解析能力作为单独功能删除决策。

## 七、归零其它“宽松接口”与别名入口（P2）

这里采取更严格的政策：**用户公开配置与程序接口只允许一种正式拼写**，但科学输入的规范 ID 到内部标识的唯一映射不是旧版本兼容，必须依据来源文件字段而不是“试多个候选名称”。

| 编号 | 文件位置 | 现状 / 建议改动 | 可能影响 |
|---|---|---|---|
| ZC-20 | `config/registry.py:191–204` | `normalize_class_config` 允许裸字符串和 `{"type": ...}` 两种写法；统一只接受对象并显式 `type`，删字符串分支 | 某些第三方 YAML 配置需改写，不提供自动升级 |
| ZC-21 | `config/registry.py:64–105,136–155` | 注册与解析用 `casefold()` 自动匹配类名；改为逐字精确匹配，拒绝大小写变体，不再自动替用户纠正 `type` | class 名需按 `describe-config` 输出原样书写 |
| ZC-22 | `config/registry.py:64–83,158–182` 与 `classes/observation_factory.py` | `global_scope` 和 `scope` 是两种表达相同作用域的入口；只保留 `scope`，全部内部注册统一 | 需更新观察类注册调用及测试 |
| ZC-23 | `programs/registry.py:182–216` | `program()` 同时支持 `ProgramSpec` 和快捷关键字构造；现有代码均走 `ProgramSpec` 形式，删除快捷分支，统一装饰器签名 | 避免后续重新引入平行声明机制 |
| ZC-24 | `programs/registry.py:209–235` | `get_program` / 注册内部用 `casefold`，改为按正式程序名精确匹配；排序时可继续 `casefold`，那不是兼容行为 | 程序名拼写错误即时失败 |
| ZC-25 | `classes/observation/catalogs.py:106–142` 与 `classes/observation/resolver.py:94–122` | 删除“精确→忽略大小写→压缩标点→alias→多个候选轮询”的解析链，改为**单一显式标识映射**；观测源字段矛盾时失败 | 需在数据导入边界形成规范 station/reflector key |
| ZC-26 | `base/station_identity.py:40–92` / `classes/ephemerides/body_ids.py:65–94` | 用户输入禁止自由别名/随意拼写映射（如可选 `APOL`、`SSB`、`EMB` 简写）；明确选择正式站 ID / NAIF ID 作为标准入口 | CRD H2 / H3 仍需依据官方字段做一次确定性科学身份解析；不要删除必要的 NAIF 物理 ID 定义 |
| ZC-27 | `parallel/mpi.py:459–471` | 若显式选择 MPI 且只有一个 rank，**直接报错**；正常串行运行走独立路径，不再在 MPI 调度器里偷偷回落串行 | 性能/日志模式透明，需覆盖 rank1 负面用例 |

**注意 ZC-25/26 的边界：**删除的是公开接口“尝试多种别名直至成功”；若 ILRS/NAIF 外部文件使用官方数字代码，解析它并关联到唯一台站/天体是领域身份转换，不是软件兼容层。建议由数据导入器或 catalog 构造器完成一次显式、有单位和来源记录的映射；模型/平差层不得再猜测。对观测 H2 的 station name 与 pad ID 应同时验证一致性，不能在一个失败时静默选择另一个。

## 八、删通用多版本钩子，但保留严格版本检测（P1）

| 编号 | 位置 | 建议 |
|---|---|---|
| ZC-28 | `fileio/archive.py:129–161` | 待报告和 EOP 全改为严格版本后，删除无调用者的 `supported_versions` 参数与判断；保留 `expected_version` 及不相等就失败的诊断 |
| ZC-29 | `fileio/archive.py:173–189`、`fileio/yaml_artifact.py:15–46` | 最好移除隐式 `version=1` 默认，要求每个读写调用显式传各制品的 `FORMAT_VERSION`；避免新制品意外写成 v1 |
| ZC-30 | `fileio/normal_equations.py`、`fileio/mass_catalog.py`、`fileio/processing_artifacts.py`、`programs/normal_points_convert.py` | 与 ZC-29 同步补充当前版本显式常量；不要因为其它类型使用 1 就错误地全改成 2 |

**单一版本 ≠ 全部版本都等于 2。** 当前源码中 `processingReport`、`lunarOrbitMetadata`、`earthOrientationParameter` 的现行版本为 2，`observationPrediction` 为 5，`predictionWindow` 为 4，而 `normalPoint`、`parameterVector` 等多种其他原生制品的**现行版本就是 1**。它们可以继续标 1；只要每种类型的读入严格只接受本类型的当前声明版本，就符合零历史兼容要求。

## 九、数值接口中带“legacy”字样的情形：只删除接口旧形式，不删物理转换（P3）

`lunarops/_iers2010_core.pyx:536–555` 的注释写明，BLQ 文件原始排列是 **Up, West, South**，对外 `hardisp()` 当前返回 **Up, South, West**；`classes/displacement/ocean_tidal_loading.py:322–345` 按其返回顺序构造 ENU。

如果坚持连“历史公共 API 顺序”也不要，建议让唯一公开 HARDISP 结果采用带显式轴名的统一对象，或选择一种经过确认的固定顺序，并在 BLQ 文件输入边界做正确坐标排列；删除无必要的旧 API 重排层。**这属于数值 API 变更，绝不可直接删掉 `result[0], result[2], result[1]` 而不改调用方**，否则 Up/South/West 轴会交换并污染台站位移。须用完整相位/振幅、多台站、多历元的 IERS 参考值进行严格数值回归。本项排在报告/EOP/CRD 兼容删除之后。

**不在本次“兼容层删除”中取消的检查**：光时不收敛状态、EOP 日期及质量范围、物理量单位、正规方程秩亏、日月星历内核科学指纹、矩阵尺寸、BLQ 标准轴顺序转换、结果发布资格、必要的串行模式和并行异常传播。`model_compatibility_fingerprint` / 正规方程 `meta["compatibility"]` 表示**科学约定一致性校验**，不是旧版本兼容器，必须保留，可考虑将名字改为 `scientificConventionFingerprint` 减少误会。

## 十、测试必须从“旧版可读”改成“旧版必拒绝”

现有 `tests/test_second_review_contracts.py:200–281,329–353` 含 v1 成功读取及迁移、旧配置快照转换的正向用例；删除相应历史断言，改为**失败即通过**的负面测试；保留 YAML 规范检查、动态 ID 保留和科学表格单位验证。

| 测试 ID | 输入边界 | 新预期 |
|---|---|---|
| T01 | `processingReport` 头版本 1 + 全部旧键 | 拒绝：`expected 2`，不打开数据迁移分支 |
| T02 | 头版本 1 + 看起来是新键 | 仍拒绝：格式版本不是 2 |
| T03 | 头版本 2 + `termination_reason` 等旧键 | 拒绝：精确到路径的未知字段 |
| T04 | 头版本 2 + 新旧键同时存在 | 拒绝，不做“新键优先”选择 |
| T05 | 头版本 3 | 拒绝：不是当前唯一版本 |
| T06 | `lunarOrbitMetadata` 版本 1 | 拒绝，不调用迁移函数 |
| T07 | 版本 2 元数据配置含 `radius_m`/`pressure_spin` | 拒绝，不能在 `config` 快照中隐藏旧键 |
| T08 | EOP 版本 1 + 无来源列 | 拒绝，不能补 `unspecified` 伪装成新来源 |
| T09 | EOP 版本 2 + 无来源列 | 拒绝，必须严格匹配当前表头 |
| T10 | CRD H1 版本 1 | 拒绝，错误定位到 H1 行 |
| T11 | CRD H1 版本 2 + record 11 缺 SNR 列 | 按 v2 结构规则拒绝此行，不误读成 v1 |
| T12 | `parameters: [{}]` 或 `observations: [{}]` | writer/readers 都拒绝不完整正式记录 |
| T13 | `observationId: null` 或必需 σ 非法 | 按记录语义拒绝，而不是仅检查类型 |
| T14 | 程序名/类型名大小写不一致、裸字符串类配置 | 严格 Schema 拒绝；不 auto-normalize |
| T15 | 站点名与 H2 站点代码相互矛盾 | 立即失败；不轮询其它别名候选 |
| T16 | 显式 MPI、但 worker 数量为 0 | 报错；不进入隐式串行调度 |
| T17 | 版本正确且字段正确的 v2 报告/EOP、CRD v2 | 成功，往返后字段、单位和动态科学 ID 不变 |
| T18 | 其它**当前版本是 1**的原生制品 | 继续正常读写，证明“零兼容”没有被误解为“禁用版本 1” |

**测试实施规则**：不要建立转换后的 v2 快照作为旧版读取的回归目标；删除所有 `read v1 → write v2 → roundtrip` 成功链。保留 v1 极小负面 fixture 仅用于断言“拒绝旧版本”，不是兼容实现或可读路径。对数值内核，应保留同一科学设置下的逐字段结果对比，不应为代码删减调整数值阈值。

## 十一、文档、GUI 和 CI 同步清理

**应从现行分发/导航中移除**：

- `docs/REPORT_SCHEMA_MIGRATION_20261013.md`：整份是 v1→v2 兼容字段对照，当前软件不再承诺此路径。需要追溯时从 Git 历史查看，**不保留现行“迁移指南”**。
- `docs/AUDIT_NAMING_REMEDIATION_20261013.md:49–53`：删除 reader 支持 v1/v2、历史轨道配置迁移的承诺，改成唯一合同的状态记录。
- `docs/PROGRAMS.md:191–208`：写成 `processingReport` 与 `lunarOrbitMetadata` **只读 v2**，并移除迁移表链接。
- `docs/AUDIT_REMEDIATION_20261013.md:35,78`：删除 EOP v1 可读与报告历史迁移的当前保证，若作为历史审计记录保留应标识“已被新合同取代”。
- `docs/INPUTS.md:21–24`：更新为仅支持 CRD v2；`docs/LUNAR_ROTATION_DEVELOPMENT.md` 的公开 `pressure_spin` 描述改成 `pressureSpin`，内部 Python 属性可单独注明。
- `describe-artifacts` 的 `legacyName` 必须为零，**只发布当前字段与约束**；GUI 不应生成任何旧式字段或快捷别名。

**建议新增并持续维护**：`docs/CURRENT_DATA_CONTRACT.md`，只列当前版本、文件头、字段、必需/nullable、单位、合法输入产品和失败规则，不附带迁移工具或旧命名表。

CI 可设置**有语义的静态禁用清单**：在生产代码中禁止 `supported_versions`、`_migrate_`、`source_names=`、`legacyName`、`snake_to_camel`、读取历史版本的条件分支；对任何命中进行人工确认，避免把科学语义的 `source_names`（例如太阳系力源数组）误判为兼容。禁止旧代码正向读取测试重新进入主分支。

## 十二、推荐执行顺序（不设兼容过渡期）

1. **冻结当前科学基线**：记录现行 ZIP 摘要、Python/CALCEPH/ERFA 版本、正式 YAML、数值输出、法点与轨道积分容差。旧文件只保存在离线归档，不承诺新程序读取。
2. **一次性取消两种报告的 v1 读入**：删除 `structured_reports` 的所有旧版分支；立即补负面测试，确保输出写 v2。
3. **ReportSchema 单名化**：删除 `source_names` / `legacyName` / `old=new`；修改三处报告生产者为直接构造 camelCase；补 required、nullable 与正式产品/诊断产品交叉约束。
4. **EOP 唯一 v2**：删旧表头与可缺来源分支；补 EOP 行数/列数/来源质量负面测试。
5. **CRD 唯一 v2**：删 v1 的 H1 与 record 11 分支，明确历史输入不可直接读取；确保 MINI、现行 CRD2 和 native 的独立入口不混淆。
6. **公开配置/API 单入口**：移除 `global_scope`、字符串类配置、大小写模糊类/程序查找、旧字段别名；台站/天体身份匹配变成显式、可检查映射。
7. **最后清理通用函数与文件**：移除 `supported_versions`；去掉隐式版本默认；删除过时迁移文档、正向旧版读取测试，更新 GUI/CLI Schema。
8. **完整验收并合并**：先单元和负面合同，再端到端 CRD2/EOP2/LLR 平差与轨道/自转；多 rank MPI、真实星历与大弧段需在正式科学环境重跑。**任何一项失败都不能用旧版兼容分支兜底。**

## 十三、最终的 Do / Do not

| 必须做 | 严禁做 |
|---|---|
| 一个原生类型只认一个当前版本 | “v1 读取成功但在内存中改成 v2” |
| 一个公开字段只有唯一 camelCase 名 | `oldName` 和 `newName` 同时接受 |
| Python 实现继续使用 snake_case | 为消除旧兼容而强制更名整个数值核心 |
| 只读 Schema 严格验证当前版本 | reader 自动转换、修补、填充历史缺失数据 |
| 外部标准的现行版本按字段严格解析 | 解析失败后试 MINI/CRD/其它格式自动兜底 |
| 科学物理转换有唯一明确的坐标/单位公式 | 直接删坐标排列或单位换算造成数值改变 |
| 旧数据与旧结果保留离线记录，当前软件拒绝它们 | 在当前项目保留旧格式转换器或 `--compat` 开关 |

## 参考

- ILRS 官方 CRD 版本说明：<https://ilrs.gsfc.nasa.gov/data_and_products/formats/crd.html>（CRD v1.01 于 2022-08-01 被 v2.01 取代）。
- ILRS 官方历史/现行 LLR 数据格式表：<https://ilrs.gsfc.nasa.gov/data_and_products/formats/index.html>。
- GROOPS 官方项目、配置模式及工具体系：<https://github.com/groops-devs/groops>。
- Python PEP 8：<https://peps.python.org/pep-0008/>。

**结论：**本轮应删除的不是“某一个兼容函数”，而是**历史输入合同、多版本读取器、旧名转换 Schema、模糊别名解析和其正向回归保障整条链**。唯一允许的“转换”是明确的科学输入解析与**当前内部对象→当前公开报告**单向序列化，它们不能再接受或修补旧格式。按上述顺序执行后，LunarOps 的对外行为应做到“旧文件必拒绝、新字段唯一、错误可定位、科学数值不变”。
