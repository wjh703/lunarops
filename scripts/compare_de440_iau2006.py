"""Reproducible DE440 versus IAU 2006/2000A pole and dynamics experiments.

python -m scripts.compare_de440_iau2006 independent OUTPUT
python -m scripts.compare_de440_iau2006 integration OUTPUT
No fitted parameters, initial conditions or default DE440 model are changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import erfa
import numpy as np
from numpy.polynomial.legendre import Legendre
from scipy.signal import find_peaks
from scipy.spatial.transform import Rotation

from lunarops.classes.dynamics.context import DynamicsEpochData, DynamicsEvaluationContext
from lunarops.classes.dynamics.deformation import deformation_tensors
from lunarops.classes.dynamics.earth_gravity_orientation import de440_earth_fixed2inertial_matrix
from lunarops.classes.dynamics.earth_iau2006_orientation import iau2006_earth_pole_vector_inertial
from lunarops.classes.dynamics.forces import EarthTideModel, EarthTideParameters
from lunarops.classes.dynamics.gravity import GravityCoefficients, GravityField
from lunarops.classes.dynamics.orientation import inertial2fixed_matrix_from_pole
from lunarops.classes.dynamics.solar_environment import SolarGmVariation
from lunarops.classes.ephemerides import body_state_matrices
from lunarops.classes.ephemerides.calceph import CalcephEphemeris
from lunarops.classes.observation_factory import ensure_registered
from lunarops.classes.time import Epoch
from lunarops.config.context import RunContext
from lunarops.config.loader import build_run_plan, load_config_file
from lunarops.fileio.mass_catalog import read_mass_catalog
from lunarops.fileio.numeric_table import read_numeric_table
from lunarops.fileio.structured_reports import read_lunar_orbit_metadata
from lunarops.programs.lunar_orbit import _build_rotation, _load_configured_gravity_fields
from lunarops.programs.registry import get_program

MAS_PER_RAD = 1 / erfa.DAS2R * 1000
PERIODS = {'half_year': 365.25/2, 'year': 365.25, 'node_18_6_year': 6798.3835,
           'node_9_3_year': 6798.3835/2,
           'draconic_month': 27.21222, 'anomalistic_month': 27.55455, 'fortnight': 13.66079}


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def csv(path, columns, data):
    assert np.isfinite(data).all()
    np.savetxt(path, data, delimiter=',', header=','.join(columns), comments='', fmt='%.17g')


def separation(a, b):
    return np.arctan2(np.linalg.norm(np.cross(a, b), axis=1), np.sum(a*b, axis=1))


def stats(days, values):
    """Vector metrics; joint trend/harmonic fit and detrended FFT on regular grid."""
    values = np.asarray(values)
    if values.ndim == 1:
        values = values[:, None]
    years = (days-days[0])/365.25
    design = np.column_stack((np.ones(len(days)), years-years.mean()))
    trend = np.linalg.lstsq(design, values, rcond=None)[0]
    residual = values-design@trend
    norms = np.linalg.norm(values, axis=1)
    harmonics = [design]
    for period in PERIODS.values():
        phase = 2*np.pi*(days-days[0])/period
        harmonics.extend([np.sin(phase)[:, None], np.cos(phase)[:, None]])
    fit = np.linalg.lstsq(np.column_stack(harmonics), values, rcond=None)[0]
    periodic = {}
    for i, (name, period) in enumerate(PERIODS.items()):
        coeff = fit[2+2*i:4+2*i]
        periodic[name] = {'period_days': period, 'amplitude_components': np.sqrt(np.sum(coeff**2, axis=0)).tolist(),
                          'rss_component_amplitude': float(np.linalg.norm(coeff))}
    # Ten-year outputs have one terminal half-day; keep the daily prefix for FFT.
    count = len(days)
    if not np.isclose(days[-1]-days[-2], days[1]-days[0], rtol=0, atol=1e-7):
        count -= 1
    window = np.hanning(count)
    spectrum = 2*np.fft.rfft(residual[:count]*window[:, None], axis=0)/window.sum()
    amplitude = np.linalg.norm(spectrum, axis=1)
    frequencies = np.fft.rfftfreq(count, days[1]-days[0])
    peaks = find_peaks(amplitude)[0]
    peaks = sorted((i for i in peaks if i > 0), key=lambda i: amplitude[i], reverse=True)[:10]
    report = {'max_norm': float(norms.max()), 'rms_norm': float(np.sqrt(np.mean(norms**2))),
              'mean_norm': float(norms.mean()), 'final_norm': float(norms[-1]),
              'mean_components': values.mean(axis=0).tolist(), 'final_components': values[-1].tolist(),
              'max_abs_components': abs(values).max(axis=0).tolist(),
              'linear_drift_components_per_julian_year': trend[1].tolist(),
              'detrended_rms_norm': float(np.sqrt(np.mean(np.sum(residual**2, axis=1)))),
              'joint_trend_harmonic_fit': periodic,
              'harmonic_fit_caution': 'Descriptive fixed-period fit, not a physical parameter adjustment; periods longer than the data arc cannot be resolved reliably.',
              'fft_peaks': [{'period_days': float(1/frequencies[i]), 'amplitude_rss': float(amplitude[i])} for i in peaks]}
    return report, residual, (frequencies[1:], amplitude[1:])


def plot_series(path, days, series, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(series), 1, figsize=(12, 3*len(series)), constrained_layout=True, squeeze=False)
    years = (days-days[0])/365.25
    for ax, (label, values) in zip(axes[:, 0], series.items(), strict=True):
        values = np.asarray(values)
        if values.ndim == 1:
            values = values[:, None]
        for i in range(values.shape[1]):
            ax.plot(years, values[:, i], lw=.65, label='xyz'[i] if values.shape[1] == 3 else str(i))
        ax.set_ylabel(label); ax.set_xlabel('Julian years from first sample'); ax.grid(alpha=.25)
        if values.shape[1] > 1:
            ax.legend()
    fig.suptitle(title)
    fig.savefig(path.with_suffix('.png'), dpi=180); fig.savefig(path.with_suffix('.pdf'))
    plt.close(fig)


def plot_isolation_spectra(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, names, unit in [(axes[0], [('J2_J5_sum','ICRF'),('J2_J5_sum_pa','reference PA')], 'm/s2'),
                            (axes[1], [('figure_torque_icrf_per_s2','ICRF'),('figure_torque_pa_per_s2','reference PA')], 's^-2')]:
        for name, label in names:
            data = np.loadtxt(root/f'{name}_spectrum.csv', delimiter=',', skiprows=1)
            ax.loglog(data[:,1], data[:,2], label=label)
        for period in (182.625,365.25,6798.3835):
            ax.axvline(period, ls='--', alpha=.25)
        ax.set(xlim=(2,10000),xlabel='Period [days]',ylabel=f'Hann FFT amplitude RSS [{unit}]')
        ax.legend(); ax.grid(alpha=.25)
    fig.suptitle('Same DE440 states; IAU minus DE440 Earth-pole acceleration / torque')
    fig.savefig(root/'force_torque_spectra.png',dpi=180); fig.savefig(root/'force_torque_spectra.pdf'); plt.close(fig)


def pole_comparison(root, days):
    epochs = tuple(Epoch(2451545., float(day-2451545.), 'tdb') for day in days)
    de_matrices = np.array([de440_earth_fixed2inertial_matrix(e) for e in epochs])
    pole_a = de_matrices[:, :, 2]
    delta_s = erfa.dtdb(2451545., days-2451545., 0, 0, 0, 0)
    tt1, tt2 = erfa.tdbtt(2451545., days-2451545., delta_s)
    rbpn = erfa.pnm06a(tt1, tt2)
    pole_b = rbpn[:, 2, :]
    # Cross-check actual scalar provider, not just the vectorized ERFA reference.
    for i in np.linspace(0, len(days)-1, 25, dtype=int):
        np.testing.assert_allclose(pole_b[i], iau2006_earth_pole_vector_inertial(epochs[i]), rtol=0, atol=4e-16)
    x, y, _ = erfa.xys06a(tt1, tt2)
    np.testing.assert_allclose(pole_b[:, :2], np.column_stack((x, y)), rtol=0, atol=1e-18)
    # Independent Park Eq. 23 and Eq. 26 construction (no production helpers).
    t = (days-2451545.)/36525
    node = (450160.280-6962890.539*t+7.455*t*t+.008*t**3)*erfa.DAS2R
    dp = -17.1996*erfa.DAS2R*np.sin(node); de = 9.2025*erfa.DAS2R*np.cos(node)
    eps = (84381.448-46.815*t-.00059*t*t+.001813*t**3)*erfa.DAS2R
    date_pole = np.column_stack((np.sin(dp)*np.sin(eps+de),
        np.cos(dp)*np.sin(eps+de)*np.cos(eps)-np.cos(eps+de)*np.sin(eps),
        np.cos(dp)*np.sin(eps+de)*np.sin(eps)+np.cos(eps+de)*np.cos(eps)))
    fx, fy = .005343991604414805*erfa.DAS2R, -.017128824840317532*erfa.DAS2R
    corrected = Rotation.from_rotvec([fx, 0, 0]).apply(Rotation.from_rotvec([0, fy, 0]).apply(date_pole))
    epj = 2000+(days-2451545.)/365.25
    n, k = erfa.ltpequ(epj), erfa.ltpecl(epj)
    equinox = np.cross(n, k); equinox /= np.linalg.norm(equinox, axis=1)[:, None]
    pv = np.stack((equinox, np.cross(n, equinox), n), axis=2)
    independent = np.einsum('nij,nj->ni', pv, corrected)
    independent_error = separation(pole_a, independent)
    assert independent_error.max() < 2e-15
    difference = (pole_b-pole_a)*MAS_PER_RAD
    angle = separation(pole_a, pole_b)*MAS_PER_RAD
    report, residual, spectrum = stats(days, difference)
    report['pole_angle_mas'] = {'max': float(angle.max()), 'rms': float(np.sqrt(np.mean(angle**2))),
                              'mean': float(angle.mean()), 'max_at_tdb_jd': float(days[angle.argmax()])}
    report['published_de440_max_angle_error_rad'] = float(independent_error.max())
    # Exact telescoping diagnostic hybrids. Attribution depends on this order;
    # these intermediates are not physical alternative production models.
    D = Rotation.from_rotvec([fx, 0, 0]).as_matrix() @ Rotation.from_rotvec([0, fy, 0]).as_matrix()
    def old_frame(p):
        return np.einsum('nij,nj->ni', pv, p @ D.T)
    eps80_tdb = erfa.obl80(2451545., days-2451545.)
    p_full80_tdb = old_frame(erfa.numat(eps80_tdb, *erfa.nut80(2451545., days-2451545.))[:, 2])
    p_full80_tt = old_frame(erfa.numat(erfa.obl80(tt1, tt2), *erfa.nut80(tt1, tt2))[:, 2])
    _, _, _, bias, precession, _, nutation, full = erfa.pn06a(tt1, tt2)
    np.testing.assert_allclose(full, rbpn, rtol=0, atol=4e-16)
    modern_date_pole = nutation[:, 2]
    p_modern_old_frame = old_frame(modern_date_pole)
    p_without_jpl_bias = np.einsum('nij,nj->ni', pv, modern_date_pole)
    p_without_iau_bias = np.einsum('nji,nj->ni', precession, modern_date_pole)
    p_modern = np.einsum('nji,nj->ni', bias, p_without_iau_bias)
    np.testing.assert_allclose(p_modern, pole_b, rtol=0, atol=1e-15)
    terms = {'nutation_truncation': p_full80_tdb-pole_a,
             'time_argument_tdb_to_tt': p_full80_tt-p_full80_tdb,
             'modern_nutation_and_obliquity': p_modern_old_frame-p_full80_tt,
             'precession': p_without_iau_bias-p_without_jpl_bias,
             'frame_bias_net': (p_without_jpl_bias-p_modern_old_frame)+(p_modern-p_without_iau_bias)}
    closure = sum(terms.values())-(pole_b-pole_a)
    assert abs(closure).max() < 1e-15
    report['attribution'] = {name: stats(days, value*MAS_PER_RAD)[0] for name, value in terms.items()}
    report['attribution_closure_max_vector_component'] = float(abs(closure).max())
    report['tdb_minus_tt_max_abs_s'] = float(abs(delta_s).max())
    columns = ['tdb_jd', 'tdb_minus_tt_s', *(f'de440_pole_{a}' for a in 'xyz'), *(f'iau_pole_{a}' for a in 'xyz'),
               *(f'delta_pole_{a}_mas' for a in 'xyz'), 'angle_mas', *(f'detrended_delta_{a}_mas' for a in 'xyz')]
    csv(root/'poles.csv', columns, np.column_stack((days, delta_s, pole_a, pole_b, difference, angle, residual)))
    csv(root/'pole_attribution.csv', ['tdb_jd', *(f'{name}_{a}_mas' for name in terms for a in 'xyz')],
        np.column_stack((days, *(v*MAS_PER_RAD for v in terms.values()))))
    csv(root/'pole_spectrum.csv', ['frequency_per_day','period_days','rss_amplitude_mas'],
        np.column_stack((spectrum[0], 1/spectrum[0], spectrum[1])))
    plot_series(root/'pole_comparison', days, {'DE440 pole': pole_a, 'IAU pole': pole_b,
        'IAU - DE440 [mas]': difference, 'Pole angle [mas]': angle, 'Detrended difference [mas]': residual},
        'IAU 2006/2000A CIP versus DE440 gravity pole; no EOP')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    ax.loglog(1/spectrum[0], spectrum[1]); ax.set(xlim=(2,30000), xlabel='Period [days]', ylabel='Hann FFT amplitude RSS [mas]')
    for name, period in PERIODS.items():
        ax.axvline(period, ls='--', alpha=.35, label=name)
    ax.legend(); ax.grid(alpha=.25)
    fig.savefig(root/'pole_spectrum.png', dpi=180); fig.savefig(root/'pole_spectrum.pdf'); plt.close(fig)
    return report, epochs, de_matrices, pole_b


def zonal_acceleration(r, pole, mu, radius, degree, jn):
    """Independent analytic gradient of -mu*Jn*R^n*P_n(p.u)/r^(n+1)."""
    distance = np.linalg.norm(r); u = r/distance; s = pole@u
    polynomial = Legendre.basis(degree)
    P, dP = polynomial(s), polynomial.deriv()(s)
    return mu*jn*radius**degree/distance**(degree+2)*(((degree+1)*P+s*dP)*u-dP*pole)


def quadrupole_hessian_torque(r, pole, K, mu, radius, j2):
    """Independent tidal-gradient tensor contraction with second moment Q."""
    distance = np.linalg.norm(r); u = r/distance; s = pole@u
    H = mu*j2*radius**2/distance**5*(1.5*(5*s*s-1)*np.eye(3)
        +7.5*(1-7*s*s)*np.outer(u,u)+15*s*(np.outer(pole,u)+np.outer(u,pole))-3*np.outer(pole,pole))
    Q = .5*np.trace(K)*np.eye(3)-K
    QH = Q@H
    return np.array([QH[1,2]-QH[2,1], QH[2,0]-QH[0,2], QH[0,1]-QH[1,0]])


def isolation(root, days, epochs, matrices_a, poles_b):
    ensure_registered()
    cfg = get_program('LunarOrbitPropagation').spec.schema.resolve(build_run_plan(load_config_file(root/'A_de440/config.yml')).calls[0][1])
    context = RunContext(working_dir=Path.cwd())
    catalog = read_mass_catalog(context.resolve_path(cfg['inputFileMassCatalog']))
    mu = {name: catalog.require(name).gm_m3_s2 for name in ('EARTH','MOON')}
    fields = _load_configured_gravity_fields(cfg['gravityFields'], context, mu)
    earth = fields['EARTH']; rotation, _ = _build_rotation(cfg, fields, mu['MOON'], mu['EARTH'])
    torque_model = next(t for t in rotation.torque_group.models if t.name == 'figure:EARTH_J2')
    with CalcephEphemeris(context.resolve_path(cfg['ephemerides']['directory']),
                         lunar_relativistic_scale_convention=cfg['ephemerides']['lunarRelativisticScaleConvention'],
                         longitude_libration_correction='none') as ephemeris:
        states = body_state_matrices(ephemeris, ('EARTH','MOON'), epochs)
        moon_matrices = ephemeris.lunar_orientation.pa_to_lcrs_matrices(epochs)
        delayed = tuple(e.shifted(-rotation.deformation.parameters.delay_s) for e in epochs)
        delayed_states = body_state_matrices(ephemeris, ('EARTH','MOON'), delayed)
        delayed_matrices = ephemeris.lunar_orientation.pa_to_lcrs_matrices(delayed)
        angles = np.asarray(ephemeris._require_handle().orient_unit(np.array([e.jd1 for e in delayed]),
            np.array([e.jd2 for e in delayed]), ephemeris._orientation_target, ephemeris._angle_units)).T
    _, theta, psi, phidot, thetadot, psidot = angles.T
    omega = np.column_stack((phidot*np.sin(theta)*np.sin(psi)+thetadot*np.cos(psi),
                            phidot*np.sin(theta)*np.cos(psi)-thetadot*np.sin(psi), psidot+phidot*np.cos(theta)))
    coefficients = earth.coefficients.coeffs
    assert np.count_nonzero(coefficients[:, 2:, 1:]) == 0
    degree_fields = {}
    for n in range(2,6):
        c = np.zeros_like(coefficients); c[0,0,0] = 1; c[0,n,0] = coefficients[0,n,0]
        degree_fields[n] = GravityField(GravityCoefficients(c, earth.gm_m3_s2, earth.radius_m))
    output = {f'J{n}': np.empty((len(days), 2, 3)) for n in range(2,6)}
    output['figure_torque_pa_per_s2'] = np.empty((len(days),2,3))
    output['figure_torque_icrf_per_s2'] = np.empty((len(days),2,3))
    max_accel_error = max_torque_error = max_diff_accel_error = max_diff_torque_error = 0.
    years = (days-torque_model.reference_jd_tdb)/365.25
    delta_j2 = torque_model.j2_rate_per_year*years+torque_model.j2_quadratic_per_year2*years**2
    inputs = DynamicsEvaluationContext(('EARTH','MOON'), [mu['EARTH'],mu['MOON']])
    for i, epoch in enumerate(epochs):
        R = moon_matrices[i]
        dR = delayed_matrices[i]; relative = delayed_states[i,1]-delayed_states[i,0]
        r_pa = -dR.T@relative[:3]; r_dot = -dR.T@relative[3:]-np.cross(omega[i],r_pa)
        tidal, spin, _ = deformation_tensors(r_pa,r_dot,omega[i],np.zeros(3),
            parameters=rotation.deformation.parameters, radius_m=rotation.radius_m, earth_mu=mu['EARTH'],moon_mu=mu['MOON'])
        K = rotation.total_inertia+tidal+spin
        r = states[i,1,:3]-states[i,0,:3]
        independent_accels, independent_torques = [], []
        for model, E in enumerate((matrices_a[i], inertial2fixed_matrix_from_pole(poles_b[i]).T)):
            ed = DynamicsEpochData(epoch, ('EARTH','MOON'), states[i,:,:3], states[i,:,3:], E, R, np.zeros(3))
            inputs.load_evaluation(ed, states[i,1]-states[i,0], None, None)
            # Only the pole changes; the BPC-derived deformation is identical.
            inputs.lunar_inertia = SimpleNamespace(total=K)
            independent_terms = []
            for n in range(2,6):
                field = degree_fields[n]
                if n == 2:
                    c = field.coefficients.coeffs.copy(); c[0,2,0] = coefficients[0,2,0]-delta_j2[i]/np.sqrt(5)
                    field = GravityField(GravityCoefficients(c, earth.gm_m3_s2, earth.radius_m))
                _, direct = inputs.figure_accelerations(0, (1,), field)
                output[f'J{n}'][i,model] = E@direct[0]
                jn = -np.sqrt(2*n+1)*field.coefficients.coeffs[0,n,0]
                check = zonal_acceleration(r, E[:,2], mu['EARTH'], earth.radius_m, n, jn)
                max_accel_error = max(max_accel_error, float(np.linalg.norm(check-output[f'J{n}'][i,model])))
                independent_terms.append(check)
            independent_accels.append(sum(independent_terms))
            torque = torque_model.torque(inputs)
            output['figure_torque_pa_per_s2'][i,model] = torque
            output['figure_torque_icrf_per_s2'][i,model] = R@torque
            check = quadrupole_hessian_torque(R.T@(-r), R.T@E[:,2], K, mu['EARTH'], earth.radius_m, torque_model.earth_j2+delta_j2[i])
            max_torque_error = max(max_torque_error, float(np.linalg.norm(check-torque)))
            independent_torques.append(check)
        da = sum(output[f'J{n}'][i,1]-output[f'J{n}'][i,0] for n in range(2,6))
        max_diff_accel_error = max(max_diff_accel_error, float(np.linalg.norm(da-(independent_accels[1]-independent_accels[0]))))
        dt = output['figure_torque_pa_per_s2'][i,1]-output['figure_torque_pa_per_s2'][i,0]
        max_diff_torque_error = max(max_diff_torque_error, float(np.linalg.norm(dt-(independent_torques[1]-independent_torques[0]))))
    output['J2_J5_sum'] = sum(output[f'J{n}'] for n in range(2,6))
    # Report every acceleration in both ICRF and the SAME reference BPC PA axes.
    for name in [f'J{n}' for n in range(2,6)]+['J2_J5_sum']:
        output[name+'_pa'] = np.einsum('nji,nkj->nki', moon_matrices, output[name])
    report = {}; columns = ['tdb_jd']; rows = [days]
    plots = {}
    for name, values in output.items():
        difference = values[:,1]-values[:,0]
        unit = 'per_s2' if 'torque' in name else 'mps2'
        summary, residual, spectrum = stats(days, difference)
        summary['baseline'] = stats(days, values[:,0])[0]
        summary['iau'] = stats(days, values[:,1])[0]
        report[name] = summary
        columns.extend(f'{name}_{model}_{a}_{unit}' for model in ('de440','iau','delta','detrended_delta') for a in 'xyz')
        rows.extend((values[:,0],values[:,1],difference,residual))
        csv(root/f'{name}_spectrum.csv', ['frequency_per_day','period_days',f'rss_amplitude_{unit}'], np.column_stack((spectrum[0],1/spectrum[0],spectrum[1])))
        if name in ('J2_J5_sum','J2_J5_sum_pa','figure_torque_pa_per_s2','figure_torque_icrf_per_s2'):
            plots[name+' delta ['+unit+']'] = difference
    csv(root/'force_torque_isolation.csv', columns, np.column_stack(rows))
    report['independent_checks'] = {'analytic_legendre_max_acceleration_error_mps2': max_accel_error,
        'hessian_contraction_max_torque_error_per_s2': max_torque_error,
        'analytic_acceleration_difference_max_error_mps2': max_diff_accel_error,
        'hessian_torque_difference_max_error_per_s2': max_diff_torque_error}
    assert max_accel_error < 1e-20 and max_torque_error < 1e-28
    plot_series(root/'force_torque_isolation', days, plots, 'Same DE440 BSP/BPC states and inertia; change only Earth gravity pole')
    plot_isolation_spectra(root)
    return report


def independent(root):
    start = sum(erfa.cal2jd(1969,1,1)); end = sum(erfa.cal2jd(2031,1,1))
    days = np.arange(start,end,1.)
    report, epochs, E, poles = pole_comparison(root,days)
    print('Pole comparison complete', flush=True)
    forces = isolation(root,days,epochs,E,poles)
    save_json(root/'independent_results.json', {'tdb_jd_start': float(days[0]),'tdb_jd_end': float(days[-1]),
        'step_days': 1, 'samples':len(days), 'poles':report,'isolation':forces,
        'torque_unit':'N/(M_moon R_moon^2), s^-2; whole-Moon mass/radius normalization',
        'inertia':'same BPC-derived delayed tidal and rotational total inertia for A/B',
        'time_conversion':'geocentric ERFA dtdb then tdbtt; no EOP or observer terms',
        'axis_convention':'ICRF-aligned GCRS axes; CIP is a deliberate proxy for gravity symmetry axis, not instantaneous spin',
        'erfa_version':erfa.__version__})
    print('Isolation comparison complete', flush=True)
    tide_isolation(root, days, epochs, E, poles)


def tide_isolation(root, days, epochs, matrices_a, poles_b):
    """Supplement: the same selected axis also enters Earth's tidal response."""
    cfg = get_program('LunarOrbitPropagation').spec.schema.resolve(build_run_plan(load_config_file(root/'A_de440/config.yml')).calls[0][1])
    catalog = read_mass_catalog(cfg['inputFileMassCatalog'])
    mu = {name:catalog.require(name).gm_m3_s2 for name in ('EARTH','MOON','SUN')}
    t = cfg['earthTide']
    parameters = EarthTideParameters(t['k20'], t['k21'], t['k22'], tuple(t['tauOrbDays']), tuple(t['tauRotDays']), t['earthRotationRateRadS'])
    gm = cfg['solarGmVariation']
    solar_gm = SolarGmVariation(gm['fractionalRatePerYear'], gm['referenceJdTdb'])
    names = ('EARTH','MOON','SUN')
    delayed_epochs = tuple(epoch.shifted(-lag*86400) for epoch in epochs for lag in parameters.tau_orb_days)
    with CalcephEphemeris(cfg['ephemerides']['directory'], lunar_relativistic_scale_convention='tdbCompatibleLunarSurface',
                         longitude_libration_correction='none') as ephemeris:
        states = body_state_matrices(ephemeris, ('EARTH','MOON'), epochs)
        delayed = body_state_matrices(ephemeris, names, delayed_epochs)
        R = ephemeris.lunar_orientation.pa_to_lcrs_matrices(epochs)
    lookup = {(e.jd1,e.jd2):v for e,v in zip(delayed_epochs,delayed,strict=True)}
    model = EarthTideModel(mu['EARTH'],mu['MOON'],lambda names,e:lookup[e.jd1,e.jd2],
        earth_radius_m=t['earthRadiusM'], tide_parameters=parameters, tide_raiser_gm={name:mu[name] for name in t['tideRaisers']},
        tide_raisers=tuple(t['tideRaisers']),solar_gm_variation=solar_gm)
    values = np.empty((len(days),2,3))
    for i,epoch in enumerate(epochs):
        for case,E in enumerate((matrices_a[i],inertial2fixed_matrix_from_pole(poles_b[i]).T)):
            values[i,case] = mu['EARTH']/(mu['EARTH']+mu['MOON'])*model.relative_acceleration(
                epoch,states[i,0,:3],states[i,1,:3],E.T)
    pa = np.einsum('nji,nkj->nki',R,values)
    columns=['tdb_jd']; rows=[days]; summary={}; plot={}
    for name,v in [('icrf',values),('pa',pa)]:
        diff=v[:,1]-v[:,0]; s,residual,spec=stats(days,diff); summary[name]=s
        columns.extend(f'{name}_{model}_{a}_mps2' for model in ('de440','iau','delta','detrended_delta') for a in 'xyz')
        rows.extend((v[:,0],v[:,1],diff,residual));plot[name+' tide delta [m/s2]']=diff
        csv(root/f'tide_{name}_spectrum.csv',['frequency_per_day','period_days','rss_amplitude_mps2'],np.column_stack((spec[0],1/spec[0],spec[1])))
    csv(root/'earth_tide_isolation.csv',columns,np.column_stack(rows))
    save_json(root/'earth_tide_isolation.json',summary)
    plot_series(root/'earth_tide_isolation',days,plot,'Same reference states; pole effect on Earth tides, same lags and Love numbers')
    print('Earth tide isolation complete',flush=True)


def integration(root):
    meta = {c: read_lunar_orbit_metadata(root/c/'lunar_orbit_metadata.txt') for c in ('A_de440','B_iau2006')}
    a_cfg = dict(meta['A_de440']['config']); b_cfg = dict(meta['B_iau2006']['config'])
    for cfg in (a_cfg,b_cfg):
        for key in ('earthPoleModel','outputFileOrbit','outputFileAccelerationDiagnostics','outputFileMetadata','outputFileRotation'):
            cfg.pop(key)
    assert a_cfg == b_cfg
    np.testing.assert_array_equal(meta['A_de440']['initialStateSi'],meta['B_iau2006']['initialStateSi'])
    hashes = json.loads((root/'source_manifest.json').read_text())
    mismatches = [p for p,h in hashes.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
    assert not mismatches
    rotations, orbits = {}, {}
    for case in meta:
        _, cols, a, _ = read_numeric_table(root/case/'lunar_orbit.dat','lunarOrbit')
        orbits[case] = a[:,[cols.index(f'relative_{axis}_m') for axis in 'xyz']]
        offset = a[:,cols.index('offset_tdb_s')]
        _, cols, q, _ = read_numeric_table(root/case/'lunar_rotation.dat','lunarRotation')
        np.testing.assert_array_equal(offset, q[:,cols.index('offset_tdb_s')])
        rotations[case] = Rotation.from_quat(q[:,[cols.index(f'q_{axis}') for axis in 'xyzw']])
    epoch = Epoch(a_cfg['initialTdbJd1'], a_cfg['initialTdbJd2'], 'tdb')
    epochs = tuple(epoch.shifted(float(t)) for t in offset)
    days = epoch.jd1+epoch.jd2+offset/86400
    with CalcephEphemeris(a_cfg['ephemerides']['directory'], lunar_relativistic_scale_convention='tdbCompatibleLunarSurface',
                         longitude_libration_correction='none') as eph:
        states = body_state_matrices(eph,('EARTH','MOON'),epochs)
        reference = states[:,1,:3]-states[:,0,:3]
        refR = Rotation.from_matrix(eph.lunar_orientation.pa_to_lcrs_matrices(epochs))
    report = {'configuration_equal_except_pole_and_output_paths': True, 'initial_state_exactly_equal': True,
              'source_manifest_mismatches': mismatches, 'samples':len(days), 'duration_julian_years':float(offset[-1]/86400/365.25),
              'cases':{}, 'B_minus_A':{}}
    columns = ['tdb_jd','offset_tdb_s']; rows = [days,offset]; plots = {}; residual_plots = {}
    for case in meta:
        pa = (refR.inv()*rotations[case]).as_rotvec(); inertial = refR.apply(pa)
        errors = {'orbit_position_m':orbits[case]-reference, 'attitude_ref_pa_rad':pa, 'attitude_icrf_rad':inertial,
                  'attitude_surface_m':1738000*inertial}
        summary = {}
        for name, value in errors.items():
            s, residual, spec = stats(days,value); summary[name] = s
            columns.extend(f'{case}_{name}_{kind}_{axis}' for kind in ('raw','detrended') for axis in 'xyz')
            rows.extend((value,residual))
            csv(root/f'{case}_{name}_spectrum.csv', ['frequency_per_day','period_days','rss_amplitude'],np.column_stack((spec[0],1/spec[0],spec[1])))
            if name in ('attitude_icrf_rad','attitude_ref_pa_rad'):
                plots[case+' '+name+' [mas]'] = value*MAS_PER_RAD
                residual_plots[case+' detrended '+name+' [mas]'] = residual*MAS_PER_RAD
        summary['reference_report'] = json.loads((root/case/'de440_reference_comparison.json').read_text())
        report['cases'][case] = summary
    delta_pa = (rotations['A_de440'].inv()*rotations['B_iau2006']).as_rotvec()
    differences = {'orbit_position_m':orbits['B_iau2006']-orbits['A_de440'],
                   'attitude_A_pa_rad':delta_pa, 'attitude_icrf_rad':rotations['A_de440'].apply(delta_pa),
                   'attitude_surface_m':1738000*rotations['A_de440'].apply(delta_pa)}
    for name,value in differences.items():
        s,residual,spec = stats(days,value); report['B_minus_A'][name]=s
        columns.extend(f'B_minus_A_{name}_{kind}_{axis}' for kind in ('raw','detrended') for axis in 'xyz')
        rows.extend((value,residual))
        csv(root/f'B_minus_A_{name}_spectrum.csv',['frequency_per_day','period_days','rss_amplitude'],np.column_stack((spec[0],1/spec[0],spec[1])))
    csv(root/'ten_year_comparison.csv',columns,np.column_stack(rows))
    plot_series(root/'ten_year_attitude_vectors',days,plots,'Attitude errors versus DE440 BPC; ICRF axes')
    plot_series(root/'ten_year_attitude_residuals',days,residual_plots,'Linear trend removed; no physical parameter fitting')
    angle_a=(refR.inv()*rotations['A_de440']).magnitude()*1738000
    angle_b=(refR.inv()*rotations['B_iau2006']).magnitude()*1738000
    plot_series(root/'ten_year_comparison',days,{'A orbit error [m]':np.linalg.norm(orbits['A_de440']-reference,axis=1),
        'B orbit error [m]':np.linalg.norm(orbits['B_iau2006']-reference,axis=1), 'B - A orbit [m]':differences['orbit_position_m'],
        'A attitude surface error [m]':angle_a,'B attitude surface error [m]':angle_b,
        'B - A attitude surface [m]':differences['attitude_surface_m']},'DE440 / IAU Earth-pole ten-year controlled joint integration')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    years = offset/(86400*365.25)
    for case, label in [('A_de440', 'A: DE440'), ('B_iau2006', 'B: IAU 2006/2000A')]:
        axes[0,0].plot(years, np.linalg.norm(orbits[case]-reference,axis=1)*1000, lw=.7, label=label)
    axes[0,1].plot(years, angle_a*1000, label='A: DE440')
    axes[0,1].plot(years, angle_b*1000, ls='--', label='B: IAU 2006/2000A')
    axes[1,0].plot(years, np.linalg.norm(differences['orbit_position_m'],axis=1)*1000, lw=.7)
    axes[1,1].plot(years, np.linalg.norm(differences['attitude_surface_m'],axis=1)*1e6)
    for ax, label in zip(axes.flat, ['Orbit vs BSP [mm]', 'Attitude vs BPC: surface scale [mm]',
                                    'B minus A position norm [mm]', 'B minus A attitude: surface scale [um]'],strict=True):
        ax.set(xlabel='Julian years from JD2440400.5', ylabel=label); ax.grid(alpha=.25)
    axes[0,0].legend(); axes[0,1].legend()
    fig.suptitle('Change only Earth pole; identical DE440 parameters and initial states')
    fig.savefig(root/'ten_year_summary.png',dpi=180); fig.savefig(root/'ten_year_summary.pdf'); plt.close(fig)
    # Ensure A reproduces the archived baseline under the unchanged DE440 pole.
    archived = root.parent/'earth_orientation_erfa_validation_20261012/de440_after_10a'
    report['archived_A_control'] = {}
    for filename in ('lunar_orbit.dat','lunar_rotation.dat','lunar_acceleration_diagnostics.dat'):
        old = read_numeric_table(archived/filename)[2]; now = read_numeric_table(root/'A_de440'/filename)[2]
        np.testing.assert_array_equal(old,now)
        report['archived_A_control'][filename] = 'exactly equal'
    save_json(root/'ten_year_results.json',report)
    print(json.dumps({k:{n:{m:v for m,v in s.items() if m in ('max_norm','rms_norm','final_norm')}
                     for n,s in case.items() if n!='reference_report'} for k,case in report['cases'].items()},indent=2))
    print('B minus A',json.dumps({k:{n:v for n,v in s.items() if n in ('max_norm','rms_norm','final_norm')}
                                 for k,s in report['B_minus_A'].items()},indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('mode',choices=['independent','integration']); parser.add_argument('root',type=Path)
    args=parser.parse_args()
    {'independent':independent,'integration':integration}[args.mode](args.root)
