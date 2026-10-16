"""Compare a joint orbit/rotation output against its configured BSP and BPC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from lunarops.classes.ephemerides import body_state_matrices
from lunarops.classes.ephemerides.calceph import CalcephEphemeris
from lunarops.classes.time import Epoch
from lunarops.fileio.numeric_table import read_numeric_table, write_numeric_table
from lunarops.fileio.structured_reports import read_lunar_orbit_metadata


def compare(directory, kernel_directory):
    run_metadata = read_lunar_orbit_metadata(directory/"lunar_orbit_metadata.txt")
    config = run_metadata["config"]
    reference_name = config["earthDynamicsFrame"].upper()
    prefix = reference_name.lower()
    radius_m = config["lunarRotation"]["radiusM"]
    _, orbit_columns, orbit, metadata = read_numeric_table(directory/"lunar_orbit.dat", "lunarOrbit")
    _, rotation_columns, rotation, _ = read_numeric_table(directory/"lunar_rotation.dat", "lunarRotation")
    offsets = orbit[:, orbit_columns.index("offset_tdb_s")]
    np.testing.assert_array_equal(offsets, rotation[:, rotation_columns.index("offset_tdb_s")])
    initial = Epoch(metadata["initialTdbJd1"], metadata["initialTdbJd2"], "tdb")
    epochs = tuple(initial.shifted(float(t)) for t in offsets)
    ephemeris = CalcephEphemeris(kernel_directory,
        lunar_relativistic_scale_convention=config["ephemerides"]["lunarRelativisticScaleConvention"],
        longitude_libration_correction=config["ephemerides"]["longitudeLibrationCorrection"])
    try:
        states = body_state_matrices(ephemeris, ("EARTH", "MOON"), epochs)
        reference_matrices = ephemeris.lunar_orientation.pa_to_lcrs_matrices(epochs)
        # CALCEPH returns Euler angles and their analytic TDB-second rates.
        # Read the BPC rates directly; no finite differences of attitudes.
        angles = np.asarray(ephemeris._require_handle().orient_unit(
            np.asarray([t.jd1 for t in epochs]), np.asarray([t.jd2 for t in epochs]),
            ephemeris._orientation_target, ephemeris._angle_units)).T
    finally:
        ephemeris.close()
    reference = states[:,1]-states[:,0]
    labels = [f"relative_{axis}" for axis in ("x_m","y_m","z_m","vx_mps","vy_mps","vz_mps")]
    difference = orbit[:,[orbit_columns.index(label) for label in labels]]-reference
    radial = reference[:,:3]/np.linalg.norm(reference[:,:3],axis=1)[:,None]
    normal = np.cross(reference[:,:3],reference[:,3:])
    normal /= np.linalg.norm(normal,axis=1)[:,None]
    transverse = np.cross(normal,radial)
    rtn = np.column_stack([np.einsum("ij,ij->i",difference[:,:3],axis) for axis in (radial,transverse,normal)])
    calculated_rotation = Rotation.from_quat(rotation[:,[rotation_columns.index(f"q_{axis}") for axis in "xyzw"]])
    reference_rotation = Rotation.from_matrix(reference_matrices)
    rotation_error = (reference_rotation.inv()*calculated_rotation).as_rotvec()
    angle_error = np.linalg.norm(rotation_error,axis=1)
    _,theta,psi,phi_dot,theta_dot,psi_dot = angles.T
    reference_omega = np.column_stack((
        phi_dot*np.sin(theta)*np.sin(psi)+theta_dot*np.cos(psi),
        phi_dot*np.sin(theta)*np.cos(psi)-theta_dot*np.sin(psi),
        psi_dot+phi_dot*np.cos(theta)))
    wm = rotation[:,[rotation_columns.index(f"mantle_omega_pa_{axis}_rad_s") for axis in "xyz"]]
    # Compare physical angular velocities in the SAME reference PA axes.
    omega_error = reference_rotation.inv().apply(calculated_rotation.apply(wm))-reference_omega
    norm_error = rotation[:,rotation_columns.index("quaternion_norm_error")]
    columns = ("offset_tdb_s",*(f"relative_position_error_{axis}_m" for axis in "xyz"),
        *(f"relative_velocity_error_{axis}_mps" for axis in "xyz"),*(f"position_error_rtn_{axis}_m" for axis in "rtn"),
        *(f"orientation_error_ref_pa_{axis}_rad" for axis in "xyz"),"orientation_error_angle_rad",
        "orientation_surface_displacement_scale_m",*(f"mantle_omega_error_ref_pa_{axis}_rad_s" for axis in "xyz"),
        "quaternion_norm_error")
    values = np.column_stack((offsets,difference,rtn,rotation_error,angle_error,radius_m*angle_error,omega_error,norm_error))
    write_numeric_table(directory/f"{prefix}_reference_comparison.dat","lunarDynamicsReferenceComparison",columns,values,
        metadata={"reference":f"{reference_name} BSP/BPC","initialTdbJd1":initial.jd1,"initialTdbJd2":initial.jd2,
                  "angular_velocity_error_frame":"reference PA; analytic BPC Euler rates, TDB rad/s"})

    def metrics(values):
        norms = np.linalg.norm(values,axis=1)
        return {"max_norm":float(np.max(norms)),"rms_norm":float(np.sqrt(np.mean(norms**2))),
                "final_norm":float(norms[-1]),"max_abs_components":np.max(abs(values),axis=0).tolist()}

    report = {"reference":f"{reference_name} BSP/BPC only","initial_tdb_jd":initial.jd1+initial.jd2,
        "duration_julian_years":float(offsets[-1]/(365.25*86400)),"samples":len(offsets),
        "relative_position_error_m":metrics(difference[:,:3]),"relative_velocity_error_mps":metrics(difference[:,3:]),
        "relative_position_error_rtn_m":metrics(rtn),"orientation_error_rad":metrics(rotation_error),
        "orientation_surface_displacement_scale_m":metrics(radius_m*rotation_error),
        "mantle_omega_error_ref_pa_rad_s":metrics(omega_error),
        "max_quaternion_norm_error":float(np.max(abs(norm_error))),
        "reference_core_angular_velocity":"not available in BSP/BPC; core state remains in propagation output",
        "external_body_groups":run_metadata["config"]["externalBodyGroups"],
        "external_body_ids":run_metadata["config"]["externalBodyIds"],
        "scope":"Moon-only coupled translation, mantle and fluid core; prescribed external trajectories, explicit rigid seed"}
    (directory/f"{prefix}_reference_comparison.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times = offsets/(365.25*86400)
    fig,axes = plt.subplots(3,2,figsize=(12,10),constrained_layout=True)
    for i,axis in enumerate("RTN"):
        axes[0,0].plot(times,rtn[:,i],label=axis)
    for i,axis in enumerate("xyz"):
        axes[0,1].plot(times,difference[:,3+i],label=axis)
        axes[1,0].plot(times,rotation_error[:,i]*206264806.24709636,label=axis)
        axes[2,0].plot(times,omega_error[:,i],label=axis)
    axes[1,1].plot(times,angle_error*radius_m*1000)
    axes[2,1].plot(times,norm_error)
    for ax,label in zip(axes.flat,("Relative position error RTN [m]","Relative velocity error [m/s]",
        "Attitude error in reference PA [mas]","Surface displacement scale [mm]",
        "Mantle angular velocity error in reference PA [rad/s]","Quaternion norm error"),strict=True):
        ax.set_xlabel("Julian years from initial epoch")
        ax.set_ylabel(label)
        ax.grid(alpha=.25)
    for ax in (axes[0,0],axes[0,1],axes[1,0],axes[2,0]):
        ax.legend()
    fig.suptitle(f"LunarOps coupled orbit / rotation vs {reference_name} BSP and BPC")
    fig.savefig(directory/f"{prefix}_reference_comparison.png",dpi=180)
    fig.savefig(directory/f"{prefix}_reference_comparison.pdf")
    plt.close(fig)
    print(json.dumps(report,indent=2,allow_nan=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory",type=Path)
    parser.add_argument("kernel_directory",type=Path)
    args = parser.parse_args()
    compare(args.directory,args.kernel_directory)


if __name__ == "__main__":
    main()
