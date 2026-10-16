"""Compare native Julia DE430 components and propagated lunar rotation.

The common-input check calls LunarOps production physics. Reference-specific
Earth-J2 conventions are reproduced only in this comparison script, and their
effect is reported separately. No LLR or estimation code is involved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from lunarops.classes.dynamics.context import DynamicsEpochData, DynamicsEvaluationContext
from lunarops.classes.dynamics.deformation import LunarDeformation, LunarDeformationParameters
from lunarops.classes.dynamics.gravity import GravityField, load_gravity_field
from lunarops.classes.dynamics.rotation import (
    LunarCoreParameters,
    LunarRotationDynamics,
    quaternion_derivative,
    quaternion_matrix,
    rotation_vector_from_euler,
)
from lunarops.classes.dynamics.torques import EarthFigureTorque, LunarFigureTorque, LunarTorqueGroup
from lunarops.classes.ephemerides.body_ids import body_name
from lunarops.classes.time import Epoch

ROOT = Path(__file__).resolve().parents[1]
DAY = 86400.0
AU_M = 149597870700.0


def read_terms(path):
    return {row[0]: np.asarray(row[1:], dtype=float)
            for line in path.read_text().splitlines() if (row := line.split("\t"))}


def rotation_state(angles, wm, wc):
    return np.concatenate((rotation_vector_from_euler(*angles, 0, 0, 0)[:4], wm, wc))


class ReferenceDelay:
    """The native backward Taylor polynomial, evaluated at its delay epoch."""

    def __init__(self, d, p):
        q, dq = d["delayed_q_native"], d["delayed_q_dot_native"]
        n = 11
        native_names = tuple(body_name(f"NAIF:{i}") for i in p["body_ids"])
        earth, moon = (native_names.index(name) for name in ("EARTH", "MOON"))
        relative = (q[:3*n].reshape(n, 3)[moon] - q[:3*n].reshape(n, 3)[earth]) * AU_M
        velocity = (q[3*n:6*n].reshape(n, 3)[moon] - q[3*n:6*n].reshape(n, 3)[earth]) * AU_M / DAY
        self.state = np.concatenate((relative, velocity,
            rotation_state(q[6*n:6*n+3], q[6*n+3:6*n+6]/DAY, q[6*n+9:6*n+12]/DAY)))
        self.derivative = np.zeros(16)
        # Use the derivative of the native delayed polynomial, not another RHS.
        self.derivative[:3] = (dq[:3*n].reshape(n,3)[moon] - dq[:3*n].reshape(n,3)[earth]) * AU_M / DAY
        self.derivative[10:13] = dq[6*n+3:6*n+6] / DAY**2
        self.derivative[13:16] = dq[6*n+9:6*n+12] / DAY**2

    def integration_sample(self, epoch):
        return self.state, self.derivative


@dataclass(frozen=True)
class ComparisonEarthTorque:
    value: np.ndarray
    name: str = "figure:EARTH_J2"

    def torque(self, inputs):
        return self.value


def earth_figure_formula(inputs, p, inertia, sine):
    """Independent Eq. 44 evaluation with explicit inertia/latitude choice."""
    R = inputs.moon_fixed2inertial_matrix
    r = R.T @ (inputs.positions_m[0] - inputs.positions_m[1])
    distance = np.linalg.norm(r)
    u = r / distance
    pole = R.T @ inputs.epoch_data.earth_fixed2inertial_matrix[:, 2]
    years = inputs.epoch_data.epoch_tdb.seconds_until(Epoch(2451545, 0, "tdb")) / (-365.25*DAY)
    j2 = p["earth_j2"] + p["earth_j2_rate_per_year"]*years
    terms = ((1-7*sine*sine)*np.cross(u,inertia@u)
             + 2*sine*(np.cross(u,inertia@pole)+np.cross(pole,inertia@u))
             - 0.4*np.cross(pole,inertia@pole))
    return 7.5*inputs.gravitational_parameters_m3_s2[0]*p["earth_radius_m"]**2*j2/distance**5*terms


def components(path, p):
    d = read_terms(path)
    native_names = tuple(body_name(f"NAIF:{i}") for i in p["body_ids"])
    names = ("EARTH", "MOON", *(name for name in native_names if name not in ("EARTH","MOON")))
    indices = [native_names.index(name) for name in names]
    epoch = Epoch(p["jd0"],0,"tdb").shifted(float(d["offset_s"][0]))
    positions, velocities = d["positions_m"].reshape(11,3)[indices], d["velocities_mps"].reshape(11,3)[indices]
    R = quaternion_matrix(rotation_state(d["euler"],d["wm"],d["wc"])[:4])
    data = DynamicsEpochData(epoch,names,positions,velocities,d["earth_matrix"].reshape(3,3),R,np.zeros(3))
    inputs = DynamicsEvaluationContext(names,np.asarray(p["body_mu"])[indices])
    history = ReferenceDelay(d,p)
    inputs.load_evaluation(data,np.concatenate((positions[1]-positions[0],velocities[1]-velocities[0])),history,{},moon_matrix=R)
    field = GravityField(load_gravity_field(ROOT.parent/"data/auxiliary/DE430_MOON_STATIC.gfc",max_degree=6))
    K = d["undistorted"].reshape(3,3)
    parameters = LunarDeformationParameters(p["love_k2"],p["delay_s"],p["mean_motion_rad_s"])
    deformation = LunarDeformation(parameters,K,p["radius_m"],inputs.gravitational_parameters_m3_s2[0],
                                   inputs.gravitational_parameters_m3_s2[1],field)
    inputs.lunar_inertia = deformation.evaluate(inputs)
    inertia = inputs.lunar_inertia
    source_names = [native_names[i-1] for i in p["source_indices"]]
    models = [LunarFigureTorque(name,field) for name in source_names]
    earth_model = EarthFigureTorque(K,p["earth_radius_m"],p["earth_j2"],p["earth_j2_rate_per_year"])
    core = LunarCoreParameters(p["core_fraction"],p["core_flattening"],p["viscosity_per_s"],"core")
    rotation = LunarRotationDynamics(K,p["radius_m"],LunarTorqueGroup([*models,earth_model]),core=core)
    state = rotation_state(d["euler"],d["wm"],d["wc"])
    result = rotation.evaluate(inputs,state)
    checks, differences = {}, {}

    def check(name, actual, expected, atol, rtol=0):
        actual, expected = np.asarray(actual), np.asarray(expected)
        delta = float(np.max(np.abs(actual-expected)))
        scale = float(np.max(abs(expected)))
        checks[name] = {"max_abs":delta,"reference_max_abs":scale,"atol":atol,"rtol":rtol,"pass":delta<=atol+rtol*scale}

    check("attitude_matrix",R,d["moon_matrix"].reshape(3,3),8e-16)
    for name, actual in (("total",inertia.total),("tidal",inertia.tidal),("rotational",inertia.rotational),
                         ("core",rotation.core_inertia),("mantle",inertia.total-rotation.core_inertia)):
        check(f"inertia_{name}",actual,d[name].reshape(3,3),2e-15)
    check("inertia_derivative",inertia.derivative,d["inertia_dot"].reshape(3,3),2e-25)
    check("inertia_rate_product",inertia.derivative@d["wm"],d["inertia_rate_product"],5e-31)
    check("mantle_cross",np.cross(d["wm"],(inertia.total-rotation.core_inertia)@d["wm"]),d["mantle_cross"],5e-29)
    check("cmb",result.viscous_torque+result.pressure_torque,d["cmb"],3e-32)
    reference_viscous = p["viscosity_per_s"]*(d["wc"]-d["wm"])
    check("cmb_viscous",result.viscous_torque,reference_viscous,3e-32)
    check("cmb_pressure",result.pressure_torque,d["cmb"]-reference_viscous,3e-32)
    coefficients = field.coefficients.coeffs
    unnormalized = np.array([coefficients[0,2,0]*np.sqrt(5),coefficients[0,2,1]*np.sqrt(5/3),
                             coefficients[1,2,1]*np.sqrt(5/3),coefficients[0,2,2]*np.sqrt(5/12),
                             coefficients[1,2,2]*np.sqrt(5/12)])
    check("degree2_unnormalized",unnormalized,d["degree2_unnormalized"],2e-16)
    for native_index, model in zip(p["source_indices"],models,strict=True):
        source = inputs.body_indices[model.source]
        _, acceleration = inputs.figure_accelerations(1,(source,),field)
        check(f"figure_acceleration_{model.source}",R@acceleration[0],d[f"figure_acceleration_{native_index}"],1e-37,1e-10)
        check(f"figure_torque_{model.source}",result.torques[model.name],d[f"figure_torque_{native_index}"],1e-38,1e-10)
    u = d["earth_direction_pa"]
    pole = d["earth_pole_pa"]
    native_sine = float(d["reference_latitude_sine"][0])
    native_Km = inertia.total-rotation.core_inertia
    aligned_earth = earth_figure_formula(inputs,p,native_Km,native_sine)
    correct_latitude_mantle = earth_figure_formula(inputs,p,native_Km,float(u@pole))
    check("earth_figure_reference_conventions",aligned_earth,d["earth_figure"],2e-32)
    differences["earth_figure"] = {
        "lunarops_total_inertia_earth_latitude":result.torques[earth_model.name].tolist(),
        "reference_mantle_inertia_pa_latitude":d["earth_figure"].tolist(),
        "max_abs_delta_s2":float(np.max(abs(result.torques[earth_model.name]-d["earth_figure"]))),
        "inertia_choice_delta_s2":float(np.max(abs(result.torques[earth_model.name]-correct_latitude_mantle))),
        "latitude_choice_delta_s2":float(np.max(abs(correct_latitude_mantle-aligned_earth))),
        "reference_latitude_sine":native_sine,"earth_latitude_sine":float(u@pole),"pa_latitude_sine":float(u[2]),
    }
    aligned_rotation = LunarRotationDynamics(K,p["radius_m"],LunarTorqueGroup([*models,ComparisonEarthTorque(aligned_earth)]),core=core)
    aligned = aligned_rotation.evaluate(inputs,state)
    check("mantle_derivative_aligned",aligned.derivative[4:7],d["mantle_dot"],1e-26)
    check("core_derivative",result.derivative[7:10],d["core_dot"],1e-26)
    qdot_euler = sum(
        rotation_vector_from_euler(*(d["euler"].astype(complex)+1e-30j*np.eye(3)[i]),0,0,0)[:4].imag
        /1e-30*d["euler_dot"][i] for i in range(3))
    check("quaternion_kinematics",quaternion_derivative(state[:4],d["wm"]),qdot_euler,1e-21)
    differences["mantle_derivative_original_max_abs_rad_s2"] = float(np.max(abs(result.derivative[4:7]-d["mantle_dot"])))
    original_mean = 2.6616995e-6
    deformation.parameters = replace(parameters,mean_motion_rad_s=original_mean)
    original_inertia = deformation.evaluate(inputs)
    differences["mean_motion_choice_max_inertia_delta"] = float(np.max(abs(original_inertia.total-inertia.total)))
    differences["delayed_position_derivative_vs_state_velocity_mps"] = float(np.max(abs(history.derivative[:3]-history.state[3:6])))
    return {"offset_s":float(d["offset_s"][0]),"checks":checks,"known_model_differences":differences}


@dataclass(frozen=True)
class JuliaEarthFigureConvention:
    """Reference-only adapter; never installed in LunarOps production models."""

    parameters: dict
    core_inertia: np.ndarray
    undistorted_inertia: np.ndarray
    name: str = "figure:EARTH_J2"

    def torque(self, inputs):
        total = self.undistorted_inertia if inputs.lunar_inertia is None else inputs.lunar_inertia.total
        r = inputs.moon_fixed2inertial_matrix.T @ (inputs.positions_m[0]-inputs.positions_m[1])
        return earth_figure_formula(inputs,self.parameters,total-self.core_inertia,float(r[2]/np.linalg.norm(r)))


def trajectories(output, p):
    """Native full-system arc against Moon-only production propagation.

    Equal initial Moon state, six major torque sources, and explicit parameter
    profiles. Reference-only conventions are confined to the adapter above.
    External ephemerides and the delayed-history algorithms still differ.
    """
    from lunarops.classes.observation_factory import ensure_registered
    from lunarops.config.context import RunContext
    from lunarops.config.loader import build_run_plan, load_config_file
    from lunarops.programs.lunar_orbit import build_lunar_propagation_problem
    from lunarops.programs.registry import get_program

    ensure_registered()
    native = np.loadtxt(output/"reference/trajectory.tsv",usecols=range(1,77))
    offsets = native[:,0]
    names = tuple(body_name(f"NAIF:{i}") for i in p["body_ids"])
    earth, moon = (names.index(name) for name in ("EARTH","MOON"))
    positions = native[:,1:34].reshape(-1,3,11).transpose(0,2,1)
    velocities = native[:,34:67].reshape(-1,3,11).transpose(0,2,1)
    angles, wm, wc = native[:,67:70],native[:,70:73],native[:,73:76]
    reference = np.asarray([rotation_state(a,b,c) for a,b,c in zip(angles,wm,wc,strict=True)])
    native_relative = np.concatenate((positions[:,moon]-positions[:,earth],velocities[:,moon]-velocities[:,earth]),axis=1)
    initial = np.concatenate((native_relative[0],reference[0]))
    computed, runs = {}, {}
    profiles = (("production",False,False,900.),("native_mean",True,False,900.),
                ("native_mean_earth_conventions",True,True,900.),
                ("native_mean_half_step",True,False,450.))
    for name, native_mean, native_earth, step in profiles:
        print(f"LunarOps trajectory profile: {name}",flush=True)
        raw = load_config_file(ROOT/"configs/lunarops_orbit_rotation_deformed_de430.yml")
        raw["variables"].update(durationSeconds=float(offsets[-1]),stepSeconds=step,startupStepSeconds=step/8,
                                 outputDir=str(output/name))
        if native_mean:
            raw["programs"][0]["lunarRotation"]["deformation"]["meanMotionRadS"] = p["mean_motion_rad_s"]
        plan = build_run_plan(raw)
        program, unresolved = plan.calls[0]
        config = get_program(program).spec.schema.resolve(unresolved)
        context = RunContext(working_dir=ROOT)
        try:
            problem = build_lunar_propagation_problem(config,context)
            rotation = problem.dynamics.rotation
            if native_earth:
                rotation.torque_group = LunarTorqueGroup([
                    JuliaEarthFigureConvention(p,rotation.core_inertia,rotation.total_inertia)
                    if isinstance(model,EarthFigureTorque) else model for model in rotation.torque_group.models
                ])
            trajectory = problem.integrator.integrate(problem.dynamics,problem.initial_epoch,initial,float(offsets[-1]),
                external_history_provider=problem.external_history,seed_duration_s=problem.seed_duration_s)
            states = np.asarray([trajectory.vector(problem.initial_epoch.shifted(float(t))) for t in offsets])
            if not np.all(np.isfinite(states)):
                raise RuntimeError(f"Nonfinite trajectory in {name}")
            computed[name] = states
            directory = output/name
            directory.mkdir(parents=True,exist_ok=True)
            np.savetxt(directory/"states.tsv",np.column_stack((offsets,states)),delimiter="\t",
                       header="offset_s relative_xyz_m relative_vxyz_mps quaternion_wxyz mantle_omega_xyz core_omega_xyz")
            pose = (Rotation.from_quat(reference[:,[1,2,3,0]]).inv()*Rotation.from_quat(states[:,[7,8,9,6]])).magnitude()
            runs[name] = {
                "step_s":step,"samples":len(states),
                "max_orientation_difference_rad":float(np.max(pose)),
                "final_orientation_difference_rad":float(pose[-1]),
                "max_surface_displacement_scale_m":float(p["radius_m"]*np.max(pose)),
                "max_mantle_omega_difference_rad_s":float(np.max(abs(states[:,10:13]-wm))),
                "max_core_omega_difference_rad_s":float(np.max(abs(states[:,13:16]-wc))),
                "max_relative_position_difference_m":float(np.max(np.linalg.norm(states[:,:3]-native_relative[:,:3],axis=1))),
                "initial_relative_position_difference_m":float(np.linalg.norm(states[0,:3]-native_relative[0,:3])),
                "initial_orientation_difference_rad":float(pose[0]),
                "max_quaternion_norm_error":float(np.max(abs(np.linalg.norm(states[:,6:10],axis=1)-1))),
            }
        finally:
            context.close()
    pairs = {}
    for first,second in (("production","native_mean"),("native_mean","native_mean_earth_conventions"),
                         ("native_mean","native_mean_half_step")):
        a,b = computed[first],computed[second]
        pose = (Rotation.from_quat(a[:,[7,8,9,6]]).inv()*Rotation.from_quat(b[:,[7,8,9,6]])).magnitude()
        pairs[f"{first}_vs_{second}"] = {
            "max_orientation_rad":float(np.max(pose)),
            "max_relative_position_m":float(np.max(np.linalg.norm(a[:,:3]-b[:,:3],axis=1))),
        }
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    for name,states in computed.items():
        pose = (Rotation.from_quat(reference[:,[1,2,3,0]]).inv()*Rotation.from_quat(states[:,[7,8,9,6]])).magnitude()
        axes[0,0].plot(offsets/DAY,pose*206264806.24709636,label=name)
        axes[0,1].plot(offsets/DAY,np.linalg.norm(states[:,10:13]-wm,axis=1))
        axes[1,0].plot(offsets/DAY,np.linalg.norm(states[:,13:16]-wc,axis=1))
        axes[1,1].plot(offsets/DAY,np.linalg.norm(states[:,:3]-native_relative[:,:3],axis=1)*1000)
    for ax,label in zip(axes.flat,("Orientation difference [mas]","Mantle angular velocity difference [rad/s]",
                                   "Core angular velocity difference [rad/s]","Relative position difference [mm]"),strict=True):
        ax.set_xlabel("TDB days from JD 2440400.5")
        ax.set_ylabel(label)
        ax.grid(alpha=.25)
    axes[0,0].legend(fontsize=7)
    fig.suptitle("LunarOps vs native PlanetaryEphemeris.jl DE430 (different external/history models)")
    fig.savefig(output/"comparison.png",dpi=180)
    fig.savefig(output/"comparison.pdf")
    plt.close(fig)
    return {"duration_days":float(offsets[-1]/DAY),"runs":runs,"profile_differences":pairs,
            "external_model_difference":"Julia integrates 354 bodies; LunarOps uses prescribed DE430 external states and 11 massive bodies",
            "history_difference":"Julia recomputes a local rigid backward Taylor arc; LunarOps uses an accepted deformed-state history after rigid seed"}


def write_document(report):
    components = report["components"]
    worst = {name:max(item["checks"][name]["max_abs"] for item in components)
             for name in components[0]["checks"]}
    rows = [("PA→惯性姿态矩阵","attitude_matrix","无量纲"),
            ("全月惯量","inertia_total","M R² 归一化"),
            ("地幔惯量","inertia_mantle","M R² 归一化"),
            ("核惯量","inertia_core","M R² 归一化"),
            ("潮汐形变","inertia_tidal","M R² 归一化"),
            ("旋转形变","inertia_rotational","M R² 归一化"),
            ("惯量导数","inertia_derivative","s⁻¹"),
            ("动态二阶系数","degree2_unnormalized","未归一化系数"),
            ("惯量导数 × 角速度","inertia_rate_product","s⁻²"),
            ("地幔欧拉交叉项","mantle_cross","s⁻²"),
            ("核幔耦合力矩","cmb","s⁻²"),
            ("核幔黏性力矩","cmb_viscous","s⁻²"),
            ("核幔压力力矩","cmb_pressure","s⁻²"),
            ("Earth J2 力矩（Julia 约定）","earth_figure_reference_conventions","s⁻²"),
            ("地幔角加速度（Julia 约定）","mantle_derivative_aligned","rad/s²"),
            ("核角加速度","core_derivative","rad/s²"),
            ("Euler→四元数运动学","quaternion_kinematics","s⁻¹")]
    text = """# LunarOps 与 PlanetaryEphemeris.jl DE430 自转对比

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
"""
    for label,name,unit in rows:
        text += f"| {label} | {worst[name]:.3e} | {unit} |\n"
    text += "\n六个源分别比较完整 degree 2–6 球谐直接加速度和反作用力矩，没有只比较合计。\n\n| 源 | 非球形加速度差（m/s²） | 归一化力矩差（s⁻²） |\n|---|---:|---:|\n"
    for name in ("EARTH","SUN","MERCURY","VENUS","MARS BARYCENTER","JUPITER BARYCENTER"):
        text += f"| {name} | {worst[f'figure_acceleration_{name}']:.3e} | {worst[f'figure_torque_{name}']:.3e} |\n"
    text += "\n## 三天轨迹结果\n\nJulia 同时推进 354 天体；LunarOps 推进月地相对六状态和月球自转，\n外部状态来自 DE430 BSP，只包含 11 个大天体质量。下表的轨迹差包含这些物理/外部轨道\n和延迟历史差异，不是积分器误差，也不能声称是完整 DE430 复现。\n\n| LunarOps 配置 | 姿态最大差（rad） | 月面位移尺度（m） | 地幔角速度最大分量差（rad/s） | 核角速度最大分量差（rad/s） | 相对位置最大差（m） |\n|---|---:|---:|---:|---:|---:|\n"
    for name,result in report["trajectory"]["runs"].items():
        text += f"| {name} | {result['max_orientation_difference_rad']:.3e} | {result['max_surface_displacement_scale_m']:.3e} | {result['max_mantle_omega_difference_rad_s']:.3e} | {result['max_core_omega_difference_rad_s']:.3e} | {result['max_relative_position_difference_m']:.3e} |\n"
    text += "\n这些配置分别为：原生产参数；仅对齐形变参考角速度；再对齐 Julia Earth figure 两项约定；\n仅对齐参考角速度并减半步长。它们均采用相同月球初始状态。\n\n| LunarOps 内部对照 | 姿态最大差（rad） | 相对位置最大差（m） |\n|---|---:|---:|\n"
    for name,result in report["trajectory"]["profile_differences"].items():
        text += f"| {name} | {result['max_orientation_rad']:.3e} | {result['max_relative_position_m']:.3e} |\n"
    text += "\n这里的减半步长对照衡量数值误差；改变物理参数/约定的对照衡量模型差异。\n月面位移尺度为 `R × 姿态差角`，不是 LLR 测距残差。\n"
    bsp = report["bsp_diagnostic"]
    text += f"\n额外核对实际 DE430 BSP/BPC：Julia 的三天月地相对位置最大差为 {bsp['native_relative_position_max_m']:.3e} m，\n姿态最大差为 {bsp['native_orientation_max_rad']:.3e} rad；LunarOps 当前生产参数的对应值为\n{bsp['lunarops_profiles']['production']['relative_position_max_m']:.3e} m 和\n{bsp['lunarops_profiles']['production']['orientation_max_rad']:.3e} rad。\n因此 Julia 实现也不是 JPL 发布星历的逐值解，匹配它不能替代对 BSP/BPC 的验证。\n这些诊断保存在 `bsp_diagnostic.json` 和总报告中。\n"
    text += "\n后续需要明确保留/选择 DE430 的形变参考角速度定义，并单独做共同外部轨道、共同\n延迟算法的长期传播对照；不能把目前的源列表和算法差异混入积分器误差预算。\n本批不改生产物理参数或参考源码，不进入 LLR 和估计。\n"
    (ROOT/"docs/DE430_ROTATION_COMPARISON.md").write_text(text)


def bsp_diagnostic(output,p):
    """Keep the native library comparison distinct from released JPL kernels."""
    from lunarops.classes.ephemerides import body_state_matrices
    from lunarops.classes.ephemerides.calceph import CalcephEphemeris

    native = np.loadtxt(output/"reference/trajectory.tsv",usecols=range(1,77))
    names = tuple(body_name(f"NAIF:{i}") for i in p["body_ids"])
    earth,moon = (names.index(name) for name in ("EARTH","MOON"))
    positions = native[:,1:34].reshape(-1,3,11).transpose(0,2,1)
    epochs = tuple(Epoch(p["jd0"],0,"tdb").shifted(float(t)) for t in native[:,0])
    ephemeris = CalcephEphemeris(ROOT.parent/"data/kernels/de430",
        lunar_relativistic_scale_convention="alreadyScaled",longitude_libration_correction="none")
    try:
        reference = body_state_matrices(ephemeris,names,epochs)
        orientation = Rotation.from_matrix(ephemeris.lunar_orientation.pa_to_lcrs_matrices(epochs))
    finally:
        ephemeris.close()
    quaternions = np.asarray([rotation_state(a,b,c)[:4] for a,b,c in
        zip(native[:,67:70],native[:,70:73],native[:,73:76],strict=True)])
    native_angles = (orientation.inv()*Rotation.from_quat(quaternions[:,[1,2,3,0]])).magnitude()
    relative = reference[:,moon,:3]-reference[:,earth,:3]
    native_relative = positions[:,moon]-positions[:,earth]
    report = {
        "native_orientation_max_rad":float(np.max(native_angles)),
        "native_relative_position_max_m":float(np.max(np.linalg.norm(native_relative-relative,axis=1))),
        "native_relative_position_initial_m":float(np.linalg.norm(native_relative[0]-relative[0])),
        "native_body_position_max_m":{name:float(np.max(np.linalg.norm(positions[:,i]-reference[:,i,:3],axis=1)))
                                      for i,name in enumerate(names)},"lunarops_profiles":{},
    }
    for name in ("production","native_mean","native_mean_earth_conventions","native_mean_half_step"):
        states = np.loadtxt(output/name/"states.tsv")[:,1:]
        angles = (orientation.inv()*Rotation.from_quat(states[:,[7,8,9,6]])).magnitude()
        report["lunarops_profiles"][name] = {
            "orientation_max_rad":float(np.max(angles)),
            "relative_position_max_m":float(np.max(np.linalg.norm(states[:,:3]-relative,axis=1))),
        }
    (output/"bsp_diagnostic.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output",type=Path)
    parser.add_argument("--components-only",action="store_true")
    args = parser.parse_args()
    p = tomllib.loads((args.output/"reference/parameters.toml").read_text())
    from lunarops.config.loader import load_config_file
    from lunarops.fileio.mass_catalog import read_mass_catalog

    cfg = load_config_file(ROOT/"configs/lunarops_orbit_rotation_deformed_de430.yml")["programs"][0]["lunarRotation"]
    catalog = read_mass_catalog(ROOT.parent/"data/auxiliary/de430_mass_catalog.txt")
    native_names = tuple(body_name(f"NAIF:{i}") for i in p["body_ids"])
    parameter_pairs = {
        "radius_m":(cfg["radiusM"],p["radius_m"]),
        "love_k2":(cfg["deformation"]["loveK2"],p["love_k2"]),
        "delay_s":(cfg["deformation"]["delayDays"]*DAY,p["delay_s"]),
        "mean_motion_rad_s":(cfg["deformation"]["meanMotionRadS"],p["mean_motion_rad_s"]),
        "core_fraction":(cfg["core"]["polar_moment_fraction"],p["core_fraction"]),
        "core_flattening":(cfg["core"]["flattening"],p["core_flattening"]),
        "viscosity_per_s":(cfg["core"]["viscosity_per_s"],p["viscosity_per_s"]),
    }
    parameter_pairs.update({f"GM_{name}":(catalog.require(name).gm_m3_s2,mu)
                            for name,mu in zip(native_names,p["body_mu"],strict=True)})
    reference_root = ROOT.parent/"PlanetaryEphemeris.jl"
    paths = ["src/DE430/dynamicalmodel.jl","src/DE430/jetcoeffs.jl","src/DE430/params.jl",
             "src/constants.jl","src/jpl-de-430-431-earth-orientation-model.jl"]
    provenance = {
        "git_head":subprocess.check_output(["git","rev-parse","HEAD"],cwd=reference_root,text=True).strip(),
        "source_sha256":{name:hashlib.sha256((reference_root/name).read_bytes()).hexdigest() for name in paths},
        "existing_local_reference_diff":subprocess.check_output(["git","diff","--",*paths],cwd=reference_root,text=True),
        "reference_instrumentation":"DE430! copy; only final nothing return replaced with diagnostics tuple",
    }
    (args.output/"reference/provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
    report = {"scope":"Moon-only rotation; no LLR, estimation or variational equations",
              "reference":"native PlanetaryEphemeris.jl DE430! and complete 354-body propagation",
              "reference_parameters":p,"reference_provenance":provenance,
              "parameter_comparison":{name:{"lunarops":float(a),"julia":float(b),"difference":float(a-b)}
                                      for name,(a,b) in parameter_pairs.items()},"components":[]}
    for path in sorted((args.output/"reference").glob("terms_*.tsv")):
        report["components"].append(components(path,p))
    report["all_common_input_checks_pass"] = bool(report["components"]) and all(
        check["pass"] for item in report["components"] for check in item["checks"].values())
    (args.output/"comparison.json").write_text(json.dumps(report,indent=2)+"\n")
    if not report["all_common_input_checks_pass"]:
        raise SystemExit("Native DE430 component comparison failed; see comparison.json")
    if not args.components_only:
        report["trajectory"] = trajectories(args.output,p)
        report["bsp_diagnostic"] = bsp_diagnostic(args.output,p)
        step_check = report["trajectory"]["profile_differences"]["native_mean_vs_native_mean_half_step"]
        report["step_convergence_pass"] = step_check["max_orientation_rad"] < 1e-12 and step_check["max_relative_position_m"] < 0.003
        (args.output/"comparison.json").write_text(json.dumps(report,indent=2)+"\n")
        if not report["step_convergence_pass"]:
            raise SystemExit("Step convergence comparison failed; see comparison.json")
        write_document(report)
    print(json.dumps({"all_common_input_checks_pass":report["all_common_input_checks_pass"],"trajectory":report.get("trajectory")},indent=2))


if __name__ == "__main__":
    main()
