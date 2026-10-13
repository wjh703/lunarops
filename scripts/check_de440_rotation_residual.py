"""Isolated diagnostics for the 39 mm DE440 rotational residual.

Adapters live here, never in the production dynamics. No fitted state is applied.
prepare ROOT; run ROOT CASE; analyze ROOT
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from lunarops.classes.dynamics import integrators as integrator_module
from lunarops.classes.dynamics.lunar_dynamics import LunarDynamics
from lunarops.classes.dynamics.rotation import quaternion_derivative, quaternion_matrix
from lunarops.classes.dynamics.torques import LunarTorqueGroup
from lunarops.classes.ephemerides import body_state_matrix
from lunarops.classes.ephemerides.calceph import CalcephEphemeris
from lunarops.classes.observation_factory import ensure_registered
from lunarops.config.context import RunContext
from lunarops.config.loader import build_run_plan, load_config_file
from lunarops.fileio.numeric_table import read_numeric_table
from lunarops.programs import lunar_orbit
from lunarops.programs.registry import get_program
from scripts.compare_de440_iau2006 import csv, save_json, stats
from scripts.compare_rotation_de430 import compare

BASE = Path('output/de440_iau2006_comparison_20261012/A_de440')
CASES = ['control', 'prescribed_orbit', 'effective_quadrupole'] + [
    f'{part}_{axis}_{sign}' for part in ('mantle', 'core') for axis in 'xyz' for sign in ('plus', 'minus')] + ['reference_seed']
PERTURBATION = {'mantle': 1e-15, 'core': 1e-12}  # rad/s, central differences only


def resolved(path):
    ensure_registered()
    return get_program('LunarOrbitPropagation').spec.schema.resolve(
        build_run_plan(load_config_file(path)).calls[0][1])


def reference_state(eph, epoch):
    states = body_state_matrix(eph, ('EARTH', 'MOON'), epoch)
    return states[1] - states[0]


class PrescribedHistory:
    """Replace all current/delayed translation; preserve accepted rotation history."""
    def __init__(self, original, ephemeris):
        self.original, self.ephemeris = original, ephemeris

    def integration_sample(self, epoch):
        state, derivative = self.original.integration_sample(epoch)
        state, derivative = state.copy(), derivative.copy()
        state[:6] = reference_state(self.ephemeris, epoch)
        derivative[:3] = state[3:6]
        derivative[3:6] = (self.ephemeris.body_acceleration_bcrs('MOON', epoch)
                            - self.ephemeris.body_acceleration_bcrs('EARTH', epoch))
        return state, derivative

    def __call__(self, names, epoch):
        return body_state_matrix(self.ephemeris, names, epoch)


class PrescribedOrbitDynamics(LunarDynamics):
    """Six dormant translation slots keep the unmodified joint ABM history layout."""
    def _load_evaluation(self, data, vector, history, cache, *, rigid=False):
        physical = vector.copy()
        physical[:6] = reference_state(self.ephemeris, data.epoch_tdb)
        exact_history = None if history is None else PrescribedHistory(history, self.ephemeris)
        return super()._load_evaluation(data, physical, exact_history, cache, rigid=rigid)

    def derivatives(self, epoch, vector, epoch_data=None, *, history=None, evaluation_cache=None):
        data = self.build_epoch_data(epoch) if epoch_data is None else epoch_data
        inputs = self._load_evaluation(data, vector, history, evaluation_cache)
        return np.concatenate((np.zeros(6), self.rotation.evaluate(inputs, vector[6:]).derivative))

    def rigid_seed_derivatives(self, epoch, vector, *, history=None):
        inputs = self._load_evaluation(self.build_epoch_data(epoch), vector, history, None, rigid=True)
        return np.concatenate((np.zeros(6), self.rotation.evaluate(inputs, vector[6:]).derivative))



def bpc_mantle(eph, epoch):
    h = eph._require_handle()
    a = np.asarray(h.orient_unit(epoch.jd1, epoch.jd2, eph._orientation_target, eph._angle_units))
    w = omega_from_euler(np.r_[a, np.zeros(3)])[0]
    qxyzw = Rotation.from_matrix(eph.lunar_orientation.pa_to_lcrs_matrix(epoch)).as_quat()
    return qxyzw[[3,0,1,2]], w


class ReferenceSeedDynamics(LunarDynamics):
    """Only negative-time seed uses BSP/BPC; core remains a model propagation."""
    def seed_state(self, epoch, vector):
        physical = vector.copy()
        physical[:6] = reference_state(self.ephemeris, epoch)
        physical[6:10], physical[10:13] = bpc_mantle(self.ephemeris, epoch)
        if physical[6:10] @ vector[6:10] < 0:
            physical[6:10] *= -1
        return physical

    def rigid_seed_derivatives(self, epoch, vector, *, history=None):
        physical = self.seed_state(epoch, vector)
        inputs = self._load_evaluation(self.build_epoch_data(epoch), physical, None, None, rigid=True)
        result = self.rotation.evaluate(inputs, physical[6:]).derivative
        # BPC second derivatives require a separate audit; differentiate first
        # derivatives over 50 s for this history experiment, not torque closure.
        wm_dot = (bpc_mantle(self.ephemeris, epoch.shifted(50.))[1]
                  - bpc_mantle(self.ephemeris, epoch.shifted(-50.))[1])/100.
        return np.concatenate((physical[3:6],
            self.ephemeris.body_acceleration_bcrs('MOON',epoch)-self.ephemeris.body_acceleration_bcrs('EARTH',epoch),
            quaternion_derivative(physical[6:10], physical[10:13]), wm_dot, result[7:]))


class PrescribedTrajectory:
    def __init__(self, trajectory, ephemeris):
        self.trajectory, self.ephemeris = trajectory, ephemeris
        self.history = trajectory.history

    def vector(self, epoch):
        result = self.trajectory.vector(epoch)
        result[:6] = reference_state(self.ephemeris, epoch)
        return result


class PrescribedIntegrator:
    def __init__(self, integrator):
        self.integrator = integrator

    def integrate(self, system, *args, **kwargs):
        return PrescribedTrajectory(self.integrator.integrate(system, *args, **kwargs), system.ephemeris)


class EffectiveQuadrupoleTorque:
    """Diagnostic gravitational quadrupole only; mechanical tensors remain intact."""
    def __init__(self, original, parameters):
        self.original, self.name = original, original.name
        self.offset = np.zeros((3, 3))
        self.offset[0, 2] = self.offset[2, 0] = -parameters.c21_static
        self.offset[1, 2] = self.offset[2, 1] = -parameters.s21_static
        self.offset[0, 1] = self.offset[1, 0] = -2 * parameters.s22_static

    def torque(self, inputs):
        original = inputs.lunar_inertia
        # Deformation is always enabled in this experiment.
        inputs.lunar_inertia = replace(original, total=original.total + self.offset)
        try:
            return self.original.torque(inputs)
        finally:
            inputs.lunar_inertia = original


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    base = yaml.safe_load((BASE/'config.yml').read_text())
    for case in CASES:
        folder = root/case
        folder.mkdir()
        document = yaml.safe_load(yaml.safe_dump(base))
        document['variables']['outputDir'] = str(folder)
        if case.startswith(('mantle_', 'core_')):
            part, axis, sign = case.split('_')
            key = 'initialAngularVelocityRadS' if part == 'mantle' else 'initialCoreAngularVelocityRadS'
            document['programs'][0]['lunarRotation'][key]['xyz'.index(axis)] += (
                PERTURBATION[part] * (1 if sign == 'plus' else -1))
        (folder/'config.yml').write_text(yaml.safe_dump(document, sort_keys=False))
    files = sorted(Path('lunarops').rglob('*.py')) + sorted(Path('lunarops').rglob('*.pyx'))
    save_json(root/'source_manifest.json', {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    save_json(root/'experiment.json', {'cases': CASES, 'perturbation_rad_s': PERTURBATION,
        'baseline': str(BASE), 'scope': 'diagnostic only; no physical default or fitted initial state changed',
        'prescribed_orbit': '6 dormant slots, exact BSP current and delayed translation; only 10 rotational states evolve',
        'effective_quadrupole': 'static offsets only in Earth J2 figure-figure torque, never Euler inertia'})


def run(root, case):
    config = resolved(root/case/'config.yml')
    original_build = lunar_orbit.build_lunar_propagation_problem

    def build(cfg, context):
        problem = original_build(cfg, context)
        dynamics = problem.dynamics
        if case in ('prescribed_orbit', 'reference_seed'):
            # Explicitly reconstruct the same environment, changing only its RHS.
            adapter = PrescribedOrbitDynamics if case == 'prescribed_orbit' else ReferenceSeedDynamics
            prescribed = adapter(
                dynamics.earth_body, dynamics.moon_body, dynamics.perturbing_bodies,
                dynamics.ephemeris, dynamics.force_group,
                earth_fixed2inertial_matrix_provider=dynamics._earth_fixed2inertial_matrix_provider,
                rotation=dynamics.rotation, kuiper_belt_ring=dynamics.kuiper_belt_ring,
                solar_gm_variation=dynamics.solar_gm_variation)
            problem = replace(problem, dynamics=prescribed,
                integrator=PrescribedIntegrator(problem.integrator) if case == 'prescribed_orbit' else problem.integrator)
            if case == 'reference_seed':
                original_startup = integrator_module.dop853_startup

                def startup(rhs, initial, duration_s, *, step_s, sample_step_s):
                    states = original_startup(rhs, initial, duration_s,
                        step_s=step_s, sample_step_s=sample_step_s)
                    if rhs.__name__ == 'seed_rhs':
                        offsets = integrator_module.regular_times(duration_s, sample_step_s)
                        for i, offset in enumerate(offsets):
                            e = problem.initial_epoch.shifted(float(offset))
                            states[:,i] = prescribed.seed_state(e, states[:,i])
                    return states

                integrator_module.dop853_startup = startup
        elif case == 'effective_quadrupole':
            rotation = dynamics.rotation
            rotation.torque_group = LunarTorqueGroup([
                EffectiveQuadrupoleTorque(t, rotation.deformation.parameters)
                if t.name == 'figure:EARTH_J2' else t for t in rotation.torque_group.models])
        return problem

    lunar_orbit.build_lunar_propagation_problem = build
    with RunContext(working_dir=Path.cwd()) as context:
        lunar_orbit.lunar_orbit_propagation(config, context)
    compare(root/case, Path('../data/kernels/de440'))
    case_scope = {
        'prescribed_orbit': 'BSP current/delayed translation; six dormant slots; only mantle/core rotation integrated',
        'reference_seed': 'BSP/BPC negative-time seed, model liquid core; original coupled forward propagation',
        'effective_quadrupole': 'gravitational cross terms only in Earth figure-figure torque; Euler tensors unchanged',
    }
    scope = case_scope.get(case, 'production coupled model; initial omega perturbation only' if case.startswith(('mantle_','core_')) else 'production coupled baseline')
    save_json(root/case/'diagnostic_mode.json', {'case':case, 'scope':scope,'physical_defaults_changed':False})
    if case == 'prescribed_orbit':
        path = root/case/'de440_reference_comparison.json'
        report = json.loads(path.read_text())
        report['scope'] = scope
        save_json(path,report)
    (root/case/'completed').write_text('completed\n')


def omega_from_euler(a):
    _, t, p, fd, td, pd = a[:6]
    fdd, tdd, pdd = a[6:9]
    w = np.array([fd*np.sin(t)*np.sin(p)+td*np.cos(p),
                  fd*np.sin(t)*np.cos(p)-td*np.sin(p), pd+fd*np.cos(t)])
    wd = np.array([fdd*np.sin(t)*np.sin(p)+fd*td*np.cos(t)*np.sin(p)+fd*pd*np.sin(t)*np.cos(p)
                   +tdd*np.cos(p)-td*pd*np.sin(p),
                   fdd*np.sin(t)*np.cos(p)+fd*td*np.cos(t)*np.cos(p)-fd*pd*np.sin(t)*np.sin(p)
                   -tdd*np.sin(p)-td*pd*np.cos(p), pdd+fdd*np.cos(t)-fd*td*np.sin(t)])
    return w, wd


def initial_audit(root):
    cfg = resolved(BASE/'config.yml')
    epoch = lunar_orbit.Epoch(cfg['initialTdbJd1'], cfg['initialTdbJd2'], 'tdb')
    r = cfg['lunarRotation']
    with CalcephEphemeris('../data/kernels/de440', lunar_relativistic_scale_convention='tdbCompatibleLunarSurface') as eph:
        h = eph._require_handle()
        a = np.asarray(h.orient_order(epoch.jd1, epoch.jd2, eph._orientation_target, eph._angle_units, 2))
        w, wd = omega_from_euler(a)
        R = eph.lunar_orientation.pa_to_lcrs_matrix(epoch)
        q = np.array(r['initialQuaternion'])
        qerr = (Rotation.from_matrix(R).inv()*Rotation.from_matrix(quaternion_matrix(q))).as_rotvec()
        finite_differences = {}
        for delta in (10., 100., 1000.):
            samples = [np.asarray(h.orient_order(e.jd1, e.jd2, eph._orientation_target, eph._angle_units, 2))
                       for e in (epoch.shifted(-delta), epoch.shifted(delta))]
            finite_differences[str(delta)] = ((omega_from_euler(samples[1])[0]-omega_from_euler(samples[0])[0])/(2*delta)-wd).tolist()
        derivative_probe = {}
        def bpc_wd(offset):
            e = epoch.shifted(offset)
            return omega_from_euler(np.asarray(h.orient_order(e.jd1,e.jd2,eph._orientation_target,eph._angle_units,2)))
        for offset in (0., -1000., 1000., -86400., 86400.):
            w0, wd0 = bpc_wd(offset)
            central = (bpc_wd(offset+10)[0]-bpc_wd(offset-10)[0])/20
            forward = (-3*w0+4*bpc_wd(offset+10)[0]-bpc_wd(offset+20)[0])/20
            backward = (3*w0-4*bpc_wd(offset-10)[0]+bpc_wd(offset-20)[0])/20
            derivative_probe[str(offset)] = {'central_error': (central-wd0).tolist(),
                'forward_error': (forward-wd0).tolist(), 'backward_error': (backward-wd0).tolist()}
    bpc_path = Path('../data/kernels/de440/moon_pa_de440_200625.bpc')
    with bpc_path.open('rb') as stream:
        header = stream.read(1024)
        record = struct.unpack_from('<i',header,76)[0]
        stream.seek((record-1)*1024)
        summary = stream.read(1024)
        identifiers = struct.unpack_from('<iiiii',summary,40)
        assert identifiers[2] == 2  # PCK type 2, Chebyshev angle polynomials
        stream.seek((identifiers[4]-4)*8)
        first_et, interval, record_size, count = struct.unpack('<dddd',stream.read(32))
    segment_position = (((epoch.jd1-2451545.)+epoch.jd2)*86400-first_et)/interval
    Kz = r['inertiaMoments'][2] * r['core']['polar_moment_fraction']
    kappa = r['core']['viscosity_per_s']
    report = {'reference_mantle_omega_pa_rad_s': w.tolist(), 'configured_mantle_omega_pa_rad_s': r['initialAngularVelocityRadS'],
              'initial_omega_difference_rad_s': (np.asarray(r['initialAngularVelocityRadS'])-w).tolist(),
              'initial_omega_difference_norm_rad_s': float(np.linalg.norm(np.asarray(r['initialAngularVelocityRadS'])-w)),
              'initial_orientation_difference_rad': qerr.tolist(),
              'reference_mantle_omega_dot_pa_rad_s2': wd.tolist(), 'omega_dot_finite_difference_errors': finite_differences,
              'core_viscous_relaxation_years': Kz/kappa/(365.25*86400),
              'equivalent_constant_drift_rad_s': .0047/1738000/(365.25*86400),
              'warning': 'BPC provides no liquid-core truth; Kc/kappa is viscous timescale, not complete coupled mode damping'}
    native_path = Path('output/rotation_de440_10a_full_20261011/de440_comment_constants.txt')
    native = {}
    for line in native_path.read_text().splitlines():
        cells = line.split()
        if len(cells) == 2 and cells[0] in ('OMGCX','OMGCY','OMGCZ','KVC','IFAC','COBLAT'):
            native[cells[0]] = float(cells[1].replace('D','E'))
    reference_core = np.array([native['OMGC'+a] for a in 'XYZ'])/86400.
    report['native_core_initial_rad_s'] = reference_core.tolist()
    report['native_core_initial_difference_rad_s'] = (np.array(r['initialCoreAngularVelocityRadS'])-reference_core).tolist()
    report['native_kappa_per_s'] = native['KVC']*r['inertiaMoments'][2]/86400.
    report['configured_kappa_difference_per_s'] = kappa-report['native_kappa_per_s']
    report['native_parameters'] = native
    report['native_constants_source'] = str(native_path)
    report['BPC_derivative_boundary_audit'] = {'polynomial_interval_s': interval,
        'initial_record_coordinate': segment_position,'initial_is_exact_boundary': segment_position == round(segment_position),
        'record_size': record_size, 'record_count': count, 'probes_rad_s2': derivative_probe}
    report['BPC_second_derivative_caution'] = 'Initial epoch is an exact 8-day BPC polynomial boundary. Central differencing spans second-derivative discontinuity. Right-sided initial derivative and interior derivatives agree with orient_order(2); no CALCEPH derivative error found. Torque closure should use same-segment derivatives.'
    save_json(root/'initial_audit.json', report)
    print(json.dumps(report, indent=2))


def analyze(root):
    data, rotations, columns = {}, {}, {}
    with CalcephEphemeris('../data/kernels/de440', lunar_relativistic_scale_convention='tdbCompatibleLunarSurface') as eph:
        for case in CASES:
            assert (root/case/'completed').exists(), case
            _, _, values, _ = read_numeric_table(root/case/'de440_reference_comparison.dat', 'lunarDynamicsReferenceComparison')
            _, rc, rot, _ = read_numeric_table(root/case/'lunar_rotation.dat', 'lunarRotation')
            data[case], rotations[case], columns[case] = values, rot, rc
        days = data['control'][:,0]/86400
        cfg = resolved(root/'control/config.yml')
        epoch = lunar_orbit.Epoch(cfg['initialTdbJd1'], cfg['initialTdbJd2'], 'tdb')
        epochs = [epoch.shifted(t) for t in data['control'][:,0]]
        reference = Rotation.from_matrix(eph.lunar_orientation.pa_to_lcrs_matrices(epochs))
    radius = cfg['lunarRotation']['radiusM']
    errors, calculated = {}, {}
    report = {'cases': {}, 'initial_audit': json.loads((root/'initial_audit.json').read_text())}
    for case in CASES:
        assert np.array_equal(data[case][:,0], data['control'][:,0])
        calculated[case] = Rotation.from_quat(rotations[case][:,[columns[case].index('q_'+a) for a in 'xyzw']])
        errors[case] = radius*(calculated[case]*reference.inv()).as_rotvec()  # ICRF
        metrics, residual, _ = stats(days, errors[case])
        delta = radius*(calculated[case]*calculated['control'].inv()).as_rotvec()
        delta_metrics, _, _ = stats(days, delta)
        orbit_metrics, _, _ = stats(days, data[case][:,1:4])
        report['cases'][case] = {'surface_error_icrf_m': metrics, 'minus_control_surface_icrf_m': delta_metrics,
                                'orbit_error_m': orbit_metrics}
        csv(root/case/'residual_icrf.csv', ['offset_days']+[f'{prefix}_{a}_m' for prefix in ('error','detrended','minus_control') for a in 'xyz'],
            np.column_stack((days, errors[case], residual, delta)))
    # Fresh control must reproduce all three previously archived production tables.
    report['control_reproduces_archive'] = {}
    for file, typ in [('lunar_orbit.dat','lunarOrbit'), ('lunar_rotation.dat','lunarRotation'),
                      ('lunar_acceleration_diagnostics.dat','lunarAccelerationDiagnostics')]:
        a = read_numeric_table(BASE/file,typ)[2]
        b = read_numeric_table(root/'control'/file,typ)[2]
        report['control_reproduces_archive'][file] = bool(np.array_equal(a,b))
        np.testing.assert_array_equal(a,b)
    sensitivity = []
    for part in ('mantle','core'):
        for axis in 'xyz':
            plus, minus = f'{part}_{axis}_plus', f'{part}_{axis}_minus'
            differential = radius*(calculated[plus]*calculated[minus].inv()).as_rotvec()/(2*PERTURBATION[part])
            nonlinearity = (errors[plus]+errors[minus])/2-errors['control']
            metrics, _, _ = stats(days, differential*1e-16)
            nonlinear, _, _ = stats(days, nonlinearity)
            sensitivity.append({'part': part, 'axis': axis, 'applied_delta_rad_s': PERTURBATION[part],
                                'response_for_1e_minus16_rad_s_m': metrics, 'central_even_component_m': nonlinear})
            csv(root/f'sensitivity_{part}_{axis}.csv',['offset_days']+[f'd_surface_{a}_m_per_rad_s' for a in 'xyz'],np.column_stack((days,differential)))
    base_drift = np.array(report['cases']['control']['surface_error_icrf_m']['linear_drift_components_per_julian_year'])
    for entry in sensitivity:
        drift = np.array(entry['response_for_1e_minus16_rad_s_m']['linear_drift_components_per_julian_year'])
        entry['cosine_to_control_icrf_drift'] = float(drift@base_drift/(np.linalg.norm(drift)*np.linalg.norm(base_drift)))
    report['initial_state_sensitivities'] = sensitivity
    # Export every torque and the often overlooked I_dot * omega term.
    rot, rc = rotations['control'], columns['control']
    torque_columns = [c for c in rc if c.endswith('_per_s2') or c.startswith('inertia_dot_per_s_')]
    torque_values = rot[:,[rc.index(c) for c in torque_columns]]
    Kdot = np.zeros((len(rot),3,3))
    for i,j,axes in ((0,0,'xx'),(1,1,'yy'),(2,2,'zz'),(0,1,'xy'),(0,2,'xz'),(1,2,'yz')):
        Kdot[:,i,j] = Kdot[:,j,i] = rot[:,rc.index('inertia_dot_per_s_'+axes)]
    wm = rot[:,[rc.index('mantle_omega_pa_'+a+'_rad_s') for a in 'xyz']]
    inertia_term = np.einsum('nij,nj->ni',Kdot,wm)
    csv(root/'control_torques_and_inertia_term.csv', ['offset_tdb_s',*torque_columns,*['I_dot_omega_'+a+'_per_s2' for a in 'xyz']],
        np.column_stack((data['control'][:,0],torque_values,inertia_term)))
    report['I_dot_omega_norm_per_s2'] = stats(days,inertia_term)[0]
    earth_cols = [rc.index('figure:EARTH_J2_'+a+'_per_s2') for a in 'xyz']
    report['effective_quadrupole_direct_torque_difference_per_s2'] = stats(days, rotations['effective_quadrupole'][:,earth_cols]-rot[:,earth_cols])[0]
    report['kernel_audits'] = {name: json.loads((root/(name+'.json')).read_text())
        for name in ('degree3_6_independent_torque','accepted_history_tensor_derivative')}

    save_json(root/'summary.json', report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,1,figsize=(11,8),constrained_layout=True)
    for case in (*CASES[:3], 'reference_seed'):
        axes[0].plot(days/365.25,1000*np.linalg.norm(errors[case],axis=1),label=case)
    for case in (*CASES[1:3], 'reference_seed'):
        delta=radius*(calculated[case]*calculated['control'].inv()).as_rotvec()
        axes[1].plot(days/365.25,1000*np.linalg.norm(delta,axis=1),label=case+' minus control')
    for ax in axes:
        ax.grid(alpha=.25); ax.legend(); ax.set_xlabel('Julian years')
    axes[0].set_ylabel('BPC attitude residual [mm]'); axes[1].set_ylabel('Diagnostic model effect [mm]')
    for suffix in ('png','pdf'):
        fig.savefig(root/f'controlled_comparison.{suffix}',dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2,1,figsize=(11,8),constrained_layout=True)
    for row, part in enumerate(('mantle','core')):
        for axis in 'xyz':
            plus, minus = f'{part}_{axis}_plus', f'{part}_{axis}_minus'
            response = radius*(calculated[plus]*calculated[minus].inv()).as_rotvec()/2
            axes[row].plot(days/365.25, np.linalg.norm(response,axis=1)*1000,label=axis)
        axes[row].set_ylabel(f'{part} central response [mm]')
        axes[row].set_title(f'Applied initial perturbation: +/- {PERTURBATION[part]:.0e} rad/s')
        axes[row].set_xlabel('Julian years'); axes[row].grid(alpha=.25); axes[row].legend()
    for suffix in ('png','pdf'):
        fig.savefig(root/f'initial_omega_sensitivities.{suffix}',dpi=180)
    plt.close(fig)
    print(json.dumps({case: report['cases'][case]['surface_error_icrf_m']['max_norm'] for case in CASES},indent=2))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('action', choices=['prepare','run','analyze','initial'])
    parser.add_argument('root', type=Path)
    parser.add_argument('case', nargs='?', choices=CASES)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.root); initial_audit(args.root)
    elif args.action == 'initial': initial_audit(args.root)
    elif args.action == 'run': run(args.root,args.case)
    else: analyze(args.root)


if __name__ == '__main__':
    main()
