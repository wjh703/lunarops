# 零历史兼容层整改与验收（2026-10-14）

依据 `LunarOps_Zero_Compatibility_Review_20261014.md` 完成本轮修改。原始审计文档保留；执行后的唯一合同见 [CURRENT_DATA_CONTRACT.md](CURRENT_DATA_CONTRACT.md)，机器可读验收见 [ZERO_COMPATIBILITY_VALIDATION_20261014.json](ZERO_COMPATIBILITY_VALIDATION_20261014.json)。

## 整改对应

| 审计项 | 最终实现 |
|---|---|
| ZC-01–05 | 两种报告直接复用严格原生 YAML reader，只读本类型版本 2；删除多版本读取、迁移函数、旧配置改名及反向导入 programs。 |
| ZC-06–09 | ReportField 仅有一个公开 name；删除 source/source_names/legacyName/old=new；Schema 仅校验当前字段，分别声明 required 与 nullable。 |
| ZC-10–12 | 平差采用内部对象到当前报告的单向显式序列化；processingSteps 和轨道元数据生产方直接构造公开字段。 |
| ZC-13–15 | 原生 EOP 只认版本 2、完整来源表头和每行 10 列，不恢复历史缺失来源。 |
| ZC-16–19 | CRD H1 仅认 CRD 2；record 11 固定 14 字段且包含 SNR；来源固定 crd-v2，不传播历史版本。 |
| ZC-20–24 | 类声明只接受含精确 type 的对象；类和程序名精确查找；注册只保留 scope；程序装饰器只接受 ProgramSpec。 |
| ZC-25–26 | 删除模型层 alias/模糊目录查找和候选轮询；源边界进行确定性 ILRS/NAIF 身份关联，H2 name/pad 矛盾立即失败；自定义科学 ID 原样保留。 |
| ZC-27 | 显式 MPI 在构造器、CLI 与调度器中要求至少两个 rank；普通串行使用原有独立入口。 |
| ZC-28–30 | 删除 supported_versions；所有通用原生文本/YAML 调用显式提供本类型版本；当前版本为 1 的类型仍正常读写。 |

平差 reader/writer 使用相同的内容及语义验证：必需身份和单位、唯一参数/观测、有限数值和合法 σ、收敛/终止原因/发布资格一致、满秩和真实后验冗余。允许输出合法失败诊断，拒绝把失败光时、夹限大气延迟或虚报冗余标成正式产品。固定科学来源记录和重力文件来源有严格 Schema；外部 ID 与插件自由参数保留其边界。

共享月球模型及嵌套物理配置 Schema 下沉至 `config/lunar_models.py`。读报告时只验证已写入的当前配置快照，不补默认值。科学指纹、时间尺度和物理坐标转换、秩亏诊断继续保留。

HARDISP 唯一公开轴顺序为 Up/West/South；海洋负载结果使用 `displacement_up_west_south_m`，ENU 按 `(-West, -South, Up)` 转换。更新 Cython 实现、facade、类型声明和调用方。保留完整相位/振幅的多台站多历元参考检验，未改变既有物理误差阈值。

删除现行迁移指南，更新文档导航、输入说明和程序说明。CI 两种 profile 增加 AST 合同检查，静态规则同时区分科学 source_names 数组和历史字段选择参数。源码分发包含该检查器和新增 JSON 数值基线。

## 合同测试

`test_second_review_contracts.py` 与 `test_zero_compatibility_contracts.py` 覆盖审计 T01–T18：旧版本、新旧字段混合、深层旧配置、EOP 缺来源、CRD v1/缺 SNR/身份矛盾、不完整正式记录、非法 σ、类/程序拼写与裸字符串、MPI rank 1 拒绝；合法诊断和各类型当前版本成功往返。

全量测试：**683 passed，27.27 s，无失败或跳过**。修改前冻结基线为 572 passed。20 份现行配置与文档程序示例由合同测试校验。AST 合同检查、compileall、Ruff E9/F/I 与 diff 空白检查通过；sdist/wheel 构建及分发内容检查通过。

## 科学数值比较

基线为 `lunarops_current_source_20261014.zip`，SHA-256：

```text
4fe5fa8846bdb896ec58d20c5f995daff47a912c3189530a54a82b8db54652bf
```

基线与新源码使用相同 Python 3.14.6 和依赖环境；真实星历运行时分别从各自源码构建 native 扩展。

- 24 份实际生成的合成轨道/加速度/自转表，以及 2 份数值表 I/O fixture，所有列与浮点数组完全相等。
- 27 份共同矩阵制品（含正规方程和协方差）以及 2 份参数向量，数值完全相等。合成 LLR 平差报告的参数、观测残差、σ、权值、冗余与收敛状态完全相等；比较排除耗时和代码/文件来源指纹等非物理字段。
- HARDISP：2 个台站 × 4 个历元 × 每例 24 个小时采样，Up/West/South 和 ENU 与整改前冻结物理基线逐元素完全相等；再检验 3 处站位置的 ENU/ITRF。既有 IERS Fortran 与 Orekit 参考测试继续通过。
- 真实 DE440：相同九组力模型、de440_major、六小时弧段、5400 s 采样。轨道表 5×31 和加速度表 5×22 与基线完全相等。
- 实际 MPI：两 rank 调度 8 组 native HARDISP 科学任务，结果与串行及冻结基线完全相等；worker 异常能传播到主 rank。单 rank 的显式 MPI 在负面测试中拒绝。

本轮完成的真实星历验收为六小时 prescribed-Earth 轨道；MPI 实测覆盖调度器和 native 科学任务。**未重跑长弧段真实星历耦合自转或完整多 rank LLR 平差**，不能把短弧等价与单元测试计数视为这些完整科学验收。

历史 CRD v1、历史报告/EOP、旧字段、裸字符串类声明和模糊别名现在均拒绝；历史 CRD 需取得 v2 来源或在项目外独立转换。MINI、C04、finals2000A 仍是分别声明的独立科学输入产品。
