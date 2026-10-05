# 仅月球积分的 dynamics 精简审查

审查日期：2026-10-01。依据当前工作区实现，包括尚未提交的修改。
本次按审查结论完成了生产路径的精简，已删除旧联合积分接口。

## 结论

生产程序已经只积分月球，无需再实现一种新的积分模式。
`programs/lunar_orbit.py` 构造 `LunarDynamics`，
状态为 `[r_M/E, v_M/E]`，只有 6 维。Earth、太阳、行星和小天体轨道均由星历规定。
`classes/dynamics/lunar_dynamics.py` 的 RHS 是
`[v_M/E, a_M(model) - a_E(ephemeris)]`。

主要冗余来自旧的 12 维联合积分和为任意目标天体设计的力分配接口。
现在已经收敛为单一月球积分路径。
更大的简化收益在于减少模式、接口和测试组合，而非文件数量。

当前代码量如下，包含空行、注释和 docstring，不含测试和实验脚本：

| 范围 | 行数 |
|---|---:|
| `classes/dynamics/*.py`，含公共导出 | 2353 |
| `_dynamics_core.pyx` | 241 |
| `programs/lunar_orbit.py` | 531 |
| 合计 | 3125 |

## 优先精简项

| 位置 | 当前用途 | 建议 |
|---|---|---|
| `lunar_dynamics.py` 的 12 维联合积分 | 与当前月球任务无关 | 已删除 |
| `state.py` 的 EMB 积分状态 | 与当前 6 维状态无关 | 已删除 |
| `EarthTideForce` 的质心分配分支 | 只服务联合积分 | 已删除 |
| `force_models.py:50` 的 `LunarForceGroup` | 任意天体的 `body_terms` 和目标 mask | 已收敛为 Moon 单目标及 `enabled_force_names` 力项集合 |
| `force_models.py` 的旧动态整场实现 | 生产使用静态场 + 增量 | 已删除 |
| `force_models.py:76` 的 `_all_targets` | 初始化后没有读取引用 | 已删除 |

`BodyState` 保存星历天体的 BCRS 状态，`MoonRelativeState` 直接保存
Earth-relative 的位置和速度；相对状态不再额外包一层通用 `CartesianState`。

## 重复计算和接口问题

### 月球静态场重复求值

生产入口分别装配静态 `FigureForce` 和 `LunarDegree2GravityCorrectionForce`
（`lunar_orbit.py:269`）。旧增量实现会再次计算静态整场和动态整场，
现在 correction 只构造动态减静态的 degree-2 field，静态高阶系数不再重复评估：

```text
static + (dynamic - static)
```

推荐保留 `figure_moon` 和 `lunar_degree2_gravity` 两个诊断名称，
让惯量增量只评估动态与静态二阶系数之差。因为
`inertia.py:249` 只更新 degree-2，其他阶的重复计算可以消除。
变更需验证系数归一化、参考半径、GM、相互作用反作用和浮点误差。
现有 `test_static_figure_plus_lunar_degree2_gravity_increment_matches_dynamic_figure`
可作为分项等价性的基础，不能代替轨道回归。

### 诊断字段沿用旧模式语义

`LunarDynamics` 直接将规定的 Earth 加速度与 modeled Moon 加速度做差。
当前生产诊断只读取各力项的 Moon 行，联合积分投影接口已删除。

### 单目标限制

`bodyForceTerms` 现在直接是 Moon 的力项序列，`LunarForceGroup` 不再接受按天体
分组的目标配置，绑定时始终只生成 Moon target mask。

## 必须保留的计算

- 星历 Earth 的位置、速度和加速度：用于构造 Moon 的 BCRS 状态，并形成相对 RHS。
  不能用模型中的 Earth 牛顿辅助加速度替换星历二阶导数，这会改变受迫系统。
- 太阳、行星和选定小天体的状态与 GM：它们是摄动源，即使不参与积分。
- EIH 的完整牛顿辅助加速度和势：`_dynamics_core.pyx:140` 的月球 EIH 行
  明确读取其他天体的 `newtonian_accelerations[j]` 和 `potentials[j]`。
  不可因为只积分 Moon 就删掉这些量。只有关闭 EIH 的模型才可另走 Moon-only 点质量计算。
- Moon figure 的相互作用反作用：`force_models.py:353` 累加各 partner 对 Moon
  非球形场的反作用。partner 使用规定轨道也不能从这项中删除。
- Earth figure、Earth tide、月球时变二阶场及 PA 姿态：都被当前生产配置启用，
  属于月球平动模型；惯量模型并未积分月球地幔/流核自转。
- 延迟历史插值和有限容量缓存：潮汐及月球惯量读取积分 Moon 的历史。
  改成始终读取参考星历会改变模型，不能当作结构清理。
- ABMD 主积分、DOP853 启动、规则节点查询和批量星历准备：当前均在调用链内。
  Cython 内核也用于实际计算，应保留。

## 文档与测试结果

动力学文档已同步到 6 维 Moon-relative 模型、Moon-only force target 和同步批量缓存实现。

`tests/test_lunar_dynamics.py` 已切换到 6 维系统，并覆盖：

1. 6 维正向/反向传播与输出重建。
2. Earth 及外部天体来自星历，Moon 延迟状态来自接受节点历史。
3. 规定 Earth 系统的状态/加速度批量缓存与标量查询一致。
4. 七组诊断合成结果与传播使用的月球加速度一致。
5. 短弧配置到轨道/诊断/元数据输出的端到端回归。

## 已完成的实现

1. 删除 12 维实现、EMB 状态转换、潮汐质心分配和旧实验脚本。
2. 将生产 `LunarForceGroup` 的目标固定为 Moon，保留 EIH 的完整内部辅助数组。
3. 用独立 degree-2 correction field 消除月球静态高阶场重复求值。
4. 将积分器收敛为唯一的 `AdamsBashforthMoultonIntegrator` 公共入口；DOP853 启动逻辑改为内部函数，删除旧 `Integrator` 基类、通用 `DynamicsSystem` 导出和旧的 observer/history 参数名。

Earth/Moon/EMB 输出列虽然有派生冗余，现有分析脚本使用这些表。
它们不是额外的积分自由度，建议暂时保留文件契约。
不建议同时改质量目录、星历框架、LLR 观测模型或估计模块。

## 本次验证

运行：

```bash
.venv/bin/python -m pytest -q tests/test_lunar_dynamics.py tests/test_ephemeris_programs.py
```

结果：全套测试 340 passed。
本次没有重新运行真实星历长弧传播；短弧动力学与全套单元测试已完成回归。
