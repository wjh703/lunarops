"""Independent potential-gradient and accepted-history checks for DE440."""
from __future__ import annotations

import argparse
from math import factorial
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.special import lpmv

from lunarops.classes.dynamics.context import DynamicsEpochData, DynamicsEvaluationContext
from lunarops.classes.dynamics.gravity import GravityCoefficients, GravityField
from lunarops.classes.dynamics.torques import LunarFigureTorque
from lunarops.classes.ephemerides import body_state_matrices
from lunarops.classes.ephemerides.calceph import CalcephEphemeris
from lunarops.classes.time import Epoch
from lunarops.config.context import RunContext
from lunarops.fileio.mass_catalog import read_mass_catalog
from lunarops.programs.lunar_orbit import (
    _build_rotation,
    _load_configured_gravity_fields,
    build_lunar_propagation_problem,
)
from scripts.check_de440_rotation_residual import BASE, resolved
from scripts.compare_de440_iau2006 import csv, save_json


def potential_per_mu(r, radius, coeffs, degree):
    """4pi-normalized, no Condon-Shortley phase; independent SciPy Legendre."""
    distance=np.linalg.norm(r); longitude=np.arctan2(r[1],r[0]); z=r[2]/distance
    total=0.
    for m in range(degree+1):
        normal=np.sqrt((2 if m else 1)*(2*degree+1)*factorial(degree-m)/factorial(degree+m))
        P=(-1)**m*normal*lpmv(m,degree,z)
        total+=P*(coeffs[0,degree,m]*np.cos(m*longitude)+coeffs[1,degree,m]*np.sin(m*longitude))
    return (radius/distance)**degree/distance*total


def independent_torque(r, source_mu, radius, coeffs, degree, step):
    # Rotate the Moon about its PA axes while keeping the source fixed inertially.
    result=[]
    for axis in np.eye(3):
        points=[Rotation.from_rotvec(-axis*d).apply(r) for d in (-2*step,-step,step,2*step)]
        values=[potential_per_mu(p,radius,coeffs,degree) for p in points]
        derivative=(values[0]-8*values[1]+8*values[2]-values[3])/(12*step)
        result.append(source_mu/radius**2*derivative)
    return np.array(result)


def gravity_audit(root, cfg):
    with RunContext(working_dir=Path.cwd()) as context:
        catalog=read_mass_catalog(context.resolve_path(cfg['inputFileMassCatalog']))
        mu={name:catalog.require(name).gm_m3_s2 for name in ('EARTH','MOON')}
        fields=_load_configured_gravity_fields(cfg['gravityFields'],context,mu)
        model,_=_build_rotation(cfg,fields,mu['MOON'],mu['EARTH'])
        initial=Epoch(cfg['initialTdbJd1'],cfg['initialTdbJd2'],'tdb')
        offsets=np.linspace(0,315576000.,81)
        epochs=[initial.shifted(float(t)) for t in offsets]
        with CalcephEphemeris(context.resolve_path(cfg['ephemerides']['directory']),lunar_relativistic_scale_convention='tdbCompatibleLunarSurface') as eph:
            states=body_state_matrices(eph,('EARTH','MOON'),epochs)
            R=eph.lunar_orientation.pa_to_lcrs_matrices(epochs)
        coefficients=fields['MOON'].coefficients.coeffs
        inputs=DynamicsEvaluationContext(('EARTH','MOON'),[mu['EARTH'],mu['MOON']])
        rows=[];report={}
        for degree in range(3,7):
            c=np.zeros_like(coefficients);c[0,0,0]=1.;c[:,degree,:degree+1]=coefficients[:,degree,:degree+1]
            field=GravityField(GravityCoefficients(c,mu['MOON'],model.radius_m))
            torque=LunarFigureTorque('EARTH',field)
            exact=[];differences=[]
            for i,epoch in enumerate(epochs):
                ed=DynamicsEpochData(epoch,('EARTH','MOON'),states[i,:,:3],states[i,:,3:],np.eye(3),R[i],np.zeros(3))
                inputs.load_evaluation(ed,states[i,1]-states[i,0],None,None)
                actual=torque.torque(inputs)
                r=R[i].T@(states[i,0,:3]-states[i,1,:3])
                checks=[independent_torque(r,mu['EARTH'],model.radius_m,c,degree,h) for h in (1e-3,3e-4,1e-4)]
                errors=[np.linalg.norm(a-actual) for a in checks]
                exact.append(actual);differences.append(errors)
                rows.append([offsets[i],degree,*actual,*checks[1],*errors])
            exact=np.array(exact);differences=np.array(differences)
            report[str(degree)]={'torque_max_per_s2':float(np.linalg.norm(exact,axis=1).max()),
                'max_absolute_error_per_s2_by_step':differences.max(axis=0).tolist(),
                'relative_rms_error_by_step':(np.sqrt(np.mean(differences**2,axis=0))/np.sqrt(np.mean(np.sum(exact**2,axis=1)))).tolist()}
    csv(root/'degree3_6_independent_torque.csv',['offset_tdb_s','degree']+[f'{prefix}_{a}_per_s2' for prefix in ('actual','potential_fd') for a in 'xyz']+['error_h_1e_3','error_h_3e_4','error_h_1e_4'],np.array(rows))
    save_json(root/'degree3_6_independent_torque.json',report)
    return report


def tensor_history_audit(root, cfg):
    cfg=dict(cfg);cfg['durationSeconds']=2*86400.
    with RunContext(working_dir=Path.cwd()) as context:
        problem=build_lunar_propagation_problem(cfg,context)
        trajectory=problem.integrator.integrate(problem.dynamics,problem.initial_epoch,problem.initial_vector,problem.duration_s,
            external_history_provider=problem.external_history,seed_duration_s=problem.seed_duration_s)
        dynamics=problem.dynamics;rotation=dynamics.rotation;history=trajectory.history
        # The accepted-history derivative is independently checked with nearby
        # *same-stencil* samples; also compare the resulting physical K(t).
        rows=[]
        for t in (100.,5000.,10000.,20000.,50000.,90000.,130000.,170000.):
            epoch=problem.initial_epoch.shifted(t)
            delayed=epoch.shifted(-rotation.deformation.parameters.delay_s)
            accepted,derivative=history.integration_sample(delayed)
            inputs=dynamics._load_evaluation(dynamics.build_epoch_data(epoch),trajectory.vector(epoch),history,None)
            evaluation=inputs.lunar_inertia
            Kdot=evaluation.derivative.copy()
            for h in (.1,1.,10.):
                nearby=[];nearby_full=[];states=[]
                for delta in (-h,h):
                    e=epoch.shifted(delta)
                    inp=dynamics._load_evaluation(dynamics.build_epoch_data(e),trajectory.vector(e),history,None)
                    nearby.append(inp.lunar_inertia.tidal + inp.lunar_inertia.rotational)
                    nearby_full.append(inp.lunar_inertia.total.copy())
                    states.append(history.integration_sample(delayed.shifted(delta))[0])
                fd=(nearby[1]-nearby[0])/(2*h)
                state_fd=(states[1]-states[0])/(2*h)
                full_fd=(nearby_full[1]-nearby_full[0])/(2*h)
                rows.append([t,h,np.linalg.norm(fd-Kdot),np.linalg.norm(Kdot),np.linalg.norm(state_fd[10:13]-derivative[10:13]),
                    np.linalg.norm(derivative[10:13]),np.linalg.norm(rotation.core_inertia-rotation.core_inertia.T),
                    np.linalg.norm(Kdot@accepted[10:13]),np.linalg.norm(full_fd-Kdot)])
    a=np.array(rows)
    csv(root/'accepted_history_tensor_derivative.csv',['offset_tdb_s','fd_step_s','Kdot_error_per_s','Kdot_norm_per_s',
        'delayed_omega_dot_error_per_s2','delayed_omega_dot_norm_per_s2','core_inertia_symmetry_error','Kdot_omega_norm_per_s2','full_K_subtraction_error_per_s'],a)
    report={'samples':len(a),'max_Kdot_absolute_error_per_s':float(a[:,2].max()),'max_Kdot_relative_error':float((a[:,2]/a[:,3]).max()),
        'max_delayed_omega_dot_absolute_error_per_s2':float(a[:,4].max()),
        'max_delayed_omega_dot_relative_error':float((a[:,4]/a[:,5]).max()),
        'full_K_subtraction_max_error_per_s':float(a[:,8].max()),
        'scope':'fd uses deformation tensor to avoid subtracting 0.39 inertias; actual 2-day accepted ABM/Hermite history, including negative rigid seed; not just polynomial fixture'}
    save_json(root/'accepted_history_tensor_derivative.json',report)
    return report


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('root',type=Path);a=p.parse_args()
    cfg=resolved(BASE/'config.yml')
    print('independent gravity',gravity_audit(a.root,cfg),flush=True)
    print('accepted-history tensor derivative',tensor_history_audit(a.root,cfg),flush=True)


if __name__=='__main__':main()
