from typing import Iterable
from scipy.spatial.transform import Rotation
from disturbances import MU
from scipy.spatial.transform import Rotation as R
from simulation import Simulation, rk4_step
from satellite import Spacecraft, replace_orientation_matrices
from kinematics import eci_to_geodedic, orc_to_eci, orc_to_sbc, quaternion_kinematics, euler_ocr_to_sbc
import disturbances as dis
import environment as env
from tqdm import tqdm
import os
import json
import config_imports
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


eos_data_raw = pd.read_csv(r"EOS Sim Data\Sim2\sat_kinematic_state.CSV")


eos_data = eos_data_raw.drop(columns=["M11", "M12", "M13", "M21", "M22", "M23", "M31", "M32", "M33", 'M11.1', 'M12.1', 'M13.1', 'M21.1',
                                      'M22.1', 'M23.1', 'M31.1', 'M32.1', 'M33.1', 'X (deg/s).1', 'Y (deg/s).1', 'Z (deg/s).1'])

eos_data = to_datetime(eos_data).iloc[1:, :]
time = eos_data.index
time_passed_hours = (time - time[0]).total_seconds()/(3600)

eos_data.columns

quat_OB = eos_data[['Q1', 'Q2', 'Q3', 'Q4']]
quat_OB /= np.linalg.norm(quat_OB, axis=1, keepdims=True)

omega_BO_deg = eos_data[['X (deg/s)', 'Y (deg/s)', 'Z (deg/s)']]

euler_OB = eos_data[['Roll (deg)', 'Pitch (deg)', 'Yaw (deg)']]

pos = eos_data[['X (km)', 'Y (km)', 'Z (km)']]
vel = eos_data[['X (km/s)', 'Y (km/s)', 'Z (km/s)']]


def teme_to_gcrs(t: Iterable[datetime.datetime], x_TEME):
    teme = TEME(obstime=Time(t, format="datetime", scale="utc"))
    x_ECI = SkyCoord(CartesianRepresentation(x_TEME[:, 0], x_TEME[:, 1], x_TEME[:, 2], unit=u.km), frame=teme,
                     representation_type='cartesian').transform_to("gcrs").cartesian.xyz.to(u.m).value  # type: ignore
    return x_ECI.T

pos_arr = teme_to_gcrs(time, np.array(pos))
vel_arr = teme_to_gcrs(time, np.array(vel))

qq_torque = to_datetime(pd.read_csv(r"EOS Sim Data\Sim2\gg_torque.CSV"))

with open(os.path.join(config_imports.PROJECT_DIR, "test", "tudsat-trace_eos.json"), "r") as f:
    eos_file = json.load(f)

sim_init_data: dict = replace_orientation_matrices(eos_file)

sim = Simulation.from_json(os.path.join(config_imports.PROJECT_DIR, "test", "tudsat-trace_eos.json"))
sat = sim.sat

q_BI_true = np.empty((len(time), 4))

for i, t in enumerate(time):

    q_BO_true = R.from_quat(quat_OB.iloc[i], scalar_first=False).inv()
    q_BI_true[i] = (q_BO_true * orc_to_eci(pos_arr[i], vel_arr[i]).inv()).as_quat(scalar_first=False)



 # Important when using as_euler() is that outcome will be y x z (yaw, roll, pitch)
# yrp = orc_to_sbc(q_BI[i], pos_arr[i], vel_arr[i]).as_euler(seq="yxz", degrees=True)
# ryp[i+1] = (yrp[1], yrp[0], yrp[2])


nadir_body_axis = Rotation.from_quat(quat_OB.to_numpy(), scalar_first=False).inv().apply([0, 0, 1])

tau_gg_new = (3 * MU) / np.linalg.norm(np.atleast_2d(pos.to_numpy())*1000, axis=1, keepdims=True)**3 * np.cross(nadir_body_axis, np.matvec(sat.J_B, nadir_body_axis))

plt.plot(time_passed_hours, 100*(tau_gg_new - qq_torque)/qq_torque, label=["X dir", "Y dir", "Z dir"])
plt.title("Relative error Gravity Gradient troque")
plt.ylabel("Rel. Error in %")
plt.xlabel("Time in Hours")
plt.grid()
plt.legend()
