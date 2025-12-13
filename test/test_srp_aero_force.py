import config_imports
from scipy.spatial.transform import Rotation as R
from simulation import Simulation, rk4_step
from satellite import Spacecraft, replace_orientation_matrices
from kinematics import eci_to_geodedic, orc_to_eci, orc_to_sbc, quaternion_kinematics, euler_ocr_to_sbc
import disturbances as dis
import environment as env
from tqdm import tqdm
import os
import json
from astropy import units as u
from astropy.coordinates import SkyCoord, CartesianRepresentation, TEME
from astropy.time import Time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime


def to_datetime(eos_data: pd.DataFrame):
    eos_data["Datetime"] = eos_data[["Date", "Time", "Time since start (s)"]].apply(lambda x: datetime.datetime.fromisoformat(
        x.iloc[0] + "T" + x.iloc[1] + "Z") + datetime.timedelta(seconds=x.iloc[2] % 1.0), axis=1)
    eos_data = eos_data.drop(columns=["Date", "Time", "Time since start (s)"])
    return eos_data.set_index("Datetime", inplace=False)


eos_data_raw = pd.read_csv(os.path.join(config_imports.PROJECT_DIR, "test", "EOS Sim Data", "Sim2", "sat_kinematic_state.CSV"))


eos_data = eos_data_raw.drop(columns=["M11", "M12", "M13", "M21", "M22", "M23", "M31", "M32", "M33", 'M11.1', 'M12.1', 'M13.1', 'M21.1',
                                      'M22.1', 'M23.1', 'M31.1', 'M32.1', 'M33.1', 'X (deg/s).1', 'Y (deg/s).1', 'Z (deg/s).1'])

eos_data = to_datetime(eos_data)
time = eos_data.index
time_passed_hours = (time - time[0]).total_seconds()/(3600)


quat_OB = eos_data[['Q1', 'Q2', 'Q3', 'Q4']]
quat_OB /= np.linalg.norm(quat_OB, axis=1, keepdims=True)


pos = eos_data[['X (km)', 'Y (km)', 'Z (km)']]
vel = eos_data[['X (km/s)', 'Y (km/s)', 'Z (km/s)']]


# convert from TEME to GCRS inertial frame
def teme_to_gcrs(t: datetime.datetime, x_TEME):
    teme = TEME(obstime=Time(t, format="datetime", scale="utc"))
    x_ECI = SkyCoord(CartesianRepresentation(x_TEME[:, 0], x_TEME[:, 1], x_TEME[:, 2], unit=u.km), frame=teme,
                     representation_type='cartesian').transform_to("gcrs").cartesian.xyz.to(u.m).value  # type: ignore
    return x_ECI.T


pos_arr = teme_to_gcrs(time, np.array(pos))
vel_arr = teme_to_gcrs(time, np.array(vel))

aero_force = to_datetime(pd.read_csv(os.path.join(config_imports.PROJECT_DIR, "test", "EOS Sim Data", "Sim2", "aero_force.CSV")))
srp_force = to_datetime(pd.read_csv(os.path.join(config_imports.PROJECT_DIR, "test", "EOS Sim Data", "Sim2", "srp_force.CSV")))


with open(os.path.join(config_imports.PROJECT_DIR, "tudsat-trace_eos.json"), "r") as f:
    eos_file = json.load(f)

sim_init_data: dict = replace_orientation_matrices(eos_file)

sim = Simulation.from_json(os.path.join(config_imports.PROJECT_DIR, "tudsat-trace_eos.json"))
sat = sim.sat

q_BI_true = np.empty((len(time), 4))

F_grav = np.empty((len(time), 3))
F_aero = np.empty_like(F_grav)
tau_aero = np.empty_like(F_grav)

rho = np.empty_like(F_grav)
sun_pos = np.empty_like(F_grav)
in_shadow = np.empty(len(time), dtype=bool)
F_SRP = np.empty_like(F_grav)
tau_SRP = np.empty_like(F_grav)

with tqdm(total=len(time), desc="Running") as pbar:
    for i, t in enumerate(time):

        q_BO_true = R.from_quat(quat_OB.iloc[i], scalar_first=False).inv()
        q_BI_true[i] = (q_BO_true * orc_to_eci(pos_arr[i], vel_arr[i]).inv()).as_quat(scalar_first=False)

        lat, lon, alt = eci_to_geodedic(pos_arr[i])

        rho[i] = env.atmosphere_density_msis(t, lat, lon, alt)
        sun_pos[i] = env.sun_position(t)
        in_shadow[i] = env.is_in_shadow(pos_arr[i], sun_pos[i])

        F_aero[i], tau_aero[i] = dis.aerodynamic_drag(pos_arr[i], vel_arr[i], q_BI_true[i], sat.surfaces, rho[i])
        F_SRP[i], tau_SRP[i] = dis.solar_radiation_pressure(pos_arr[i], sun_pos[i], in_shadow[i], q_BI_true[i], sat.surfaces)

        pbar.update()


fix, ax = plt.subplots(1, 3)
dir = ["X Direction", "Y Direction", "Z Direction"]
for i in range(3):
    ax[i].set_title(dir[i])
    ax[i].plot(time_passed_hours[1:], F_aero[1:, i])
    ax[i].plot(time_passed_hours[1:], aero_force.iloc[:, i], color="tab:orange")
    ax[i].grid()

fix.supxlabel("Time in hours")
fix.supylabel("Aerodynamic drag force in N")
fix.tight_layout()

fig, ax = plt.subplots(1, 3)
dir = ["X Direction", "Y Direction", "Z Direction"]
for i in range(3):
    ax[i].set_title(dir[i])
    ax[i].plot(time_passed_hours[1:], F_SRP[1:, i])
    ax[i].plot(time_passed_hours[1:], srp_force.iloc[:, i], color="tab:orange")
    ax[i].grid()

fig.supxlabel("Time in hours")
fig.supylabel("Solar radiation pressure in N")
fig.tight_layout()
plt.show(block=True)

