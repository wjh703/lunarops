"""Check isolation of diagnostic adapters, not fitted DE440 physics."""
from types import SimpleNamespace

import numpy as np

from lunarops.classes.dynamics.deformation import LunarInertiaEvaluation
from lunarops.classes.dynamics.torques import EarthFigureTorque
from scripts.check_de440_rotation_residual import EffectiveQuadrupoleTorque, PrescribedHistory, omega_from_euler


def test_effective_quadrupole_changes_torque_without_mutating_mechanics():
    K = np.diag([.39276,.39285,.39301])
    original = LunarInertiaEvaluation(K, np.zeros((3,3)), np.zeros((3,3)), np.zeros((3,3)))
    model = EarthFigureTorque(K,6378136.6,.0010826)
    inputs = SimpleNamespace(lunar_inertia=original, moon_fixed2inertial_matrix=np.eye(3),
        positions_m=np.array([[3.6e8, 8e7, -5e7],[0,0,0]]), gravitational_parameters_m3_s2=[3.986e14,4.9e12],
        epoch_data=SimpleNamespace(earth_fixed2inertial_matrix=np.eye(3),epoch_tdb=SimpleNamespace(jd1=2451545.,jd2=0.)))
    params=SimpleNamespace(c21_static=2.3e-10,s21_static=1.3e-9,s22_static=0.)
    adapter=EffectiveQuadrupoleTorque(model,params)
    before=model.torque(inputs)
    after=adapter.torque(inputs)
    assert np.linalg.norm(after-before)>0
    assert inputs.lunar_inertia is original
    np.testing.assert_array_equal(K, original.total)
    np.testing.assert_array_equal(adapter.torque(inputs),after)


def test_prescribed_history_replaces_all_translation_and_keeps_delayed_rotation(monkeypatch):
    import scripts.check_de440_rotation_residual as experiment
    state=np.arange(16,dtype=float); derivative=-state
    history=SimpleNamespace(integration_sample=lambda epoch:(state,derivative))
    monkeypatch.setattr(experiment,'reference_state',lambda eph,epoch:np.arange(6)+100.)
    monkeypatch.setattr(experiment,'body_state_matrix',lambda eph,names,epoch:np.full((len(names),6),37.))
    eph=SimpleNamespace(body_acceleration_bcrs=lambda body,epoch:np.full(3,2. if body=='MOON' else 1.))
    adapter=PrescribedHistory(history,eph)
    s,d=adapter.integration_sample(None)
    np.testing.assert_array_equal(s[:6],np.arange(6)+100.)
    np.testing.assert_array_equal(d[:3],s[3:6]);np.testing.assert_array_equal(d[3:6],1.)
    np.testing.assert_array_equal(s[6:],state[6:]);np.testing.assert_array_equal(d[6:],derivative[6:])
    np.testing.assert_array_equal(state,np.arange(16))
    np.testing.assert_array_equal(adapter(('MOON','EARTH'),None),np.full((2,6),37.))


def test_euler_angular_acceleration_against_independent_finite_difference():
    a=np.array([.3,.4,.5,2e-6,3e-7,5e-6,2e-12,-3e-12,4e-12])
    w,wd=omega_from_euler(a)
    h=.01
    def sample(t):
        b=a.copy(); b[:3]=a[:3]+t*a[3:6]+t*t/2*a[6:9]; b[3:6]=a[3:6]+t*a[6:9]
        return omega_from_euler(b)[0]
    np.testing.assert_allclose((sample(h)-sample(-h))/(2*h),wd,rtol=1e-8,atol=3e-20)
    assert np.isfinite(w).all()


def test_prescribed_rhs_is_independent_of_current_and_delayed_translation_slots(monkeypatch):
    from test_lunar_deformation import delayed_system
    from test_lunar_rotation import EPOCH, RELATIVE

    import scripts.check_de440_rotation_residual as experiment
    system,state=delayed_system()
    prescribed=experiment.PrescribedOrbitDynamics(system.earth_body,system.moon_body,system.perturbing_bodies,
        system.ephemeris,system.force_group,rotation=system.rotation)
    monkeypatch.setattr(experiment,'reference_state',lambda eph,epoch:RELATIVE.copy())
    history=SimpleNamespace(integration_sample=lambda epoch:(state.copy(),np.zeros(16)))
    baseline=prescribed.derivatives(EPOCH,state,history=history)
    changed=state.copy();changed[:6]*=2
    bad_history=SimpleNamespace(integration_sample=lambda epoch:(changed.copy(),np.zeros(16)))
    np.testing.assert_array_equal(prescribed.derivatives(EPOCH,changed,history=bad_history),baseline)
    np.testing.assert_array_equal(baseline[:6],0.)
    assert np.linalg.norm(baseline[6:])>0


def test_independent_potential_rotation_gradient_respects_all_orders_and_phases():
    from lunarops.classes.dynamics.context import DynamicsEpochData, DynamicsEvaluationContext
    from lunarops.classes.dynamics.gravity import GravityCoefficients, GravityField
    from lunarops.classes.dynamics.torques import LunarFigureTorque
    from lunarops.classes.time import Epoch
    from scripts.audit_de440_rotation_kernels import independent_torque
    radius,moon_mu,earth_mu=1738000.,4.9e12,3.986e14
    r=np.array([3.4e8,-8e7,5e7])
    inputs=DynamicsEvaluationContext(('EARTH','MOON'),[earth_mu,moon_mu])
    for degree in range(3,7):
        c=np.zeros((2,7,7));c[0,0,0]=1.
        c[:,degree,:degree+1]=np.arange(2*(degree+1)).reshape(2,-1)*1e-7-3e-7
        field=GravityField(GravityCoefficients(c,moon_mu,radius))
        data=DynamicsEpochData(Epoch(2451545.,0,'tdb'),('EARTH','MOON'),np.array([r,np.zeros(3)]),
            np.zeros((2,3)),np.eye(3),np.eye(3),np.zeros(3))
        inputs.load_evaluation(data,-np.r_[r,np.zeros(3)],None,None)
        torque=LunarFigureTorque('EARTH',field).torque(inputs)
        check=independent_torque(r,earth_mu,radius,c,degree,3e-4)
        np.testing.assert_allclose(check,torque,rtol=2e-10,atol=1e-35)
