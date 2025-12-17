import config_imports
import os
from dynamics import SGP4
from utils import replace_orientation_matrices
import json
from astropy import units as u
from astropy.coordinates import SkyCoord, CartesianRepresentation, TEME
from astropy.time import Time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime

from simulation import Simulation



def teme_to_gcrs(t, x_TEME):
    teme = TEME(obstime=Time(t, format="datetime", scale="utc"))
    x_ECI = SkyCoord(CartesianRepresentation(x_TEME[:, 0], x_TEME[:, 1], x_TEME[:, 2], unit=u.km), frame=teme, #type: ignore
                     representation_type='cartesian').transform_to("gcrs").cartesian.xyz.to(u.m).value  # type: ignore
    return x_ECI.T.squeeze()


def to_datetime(eos_data: pd.DataFrame):
    eos_data["Datetime"] = eos_data[["Date", "Time", "Time since start (s)"]].apply(lambda x: datetime.datetime.fromisoformat(
        x.iloc[0] + "T" + x.iloc[1] + "Z") + datetime.timedelta(seconds=x.iloc[2] % 1.0), axis=1)
    eos_data = eos_data.drop(columns=["Date", "Time", "Time since start (s)"])
    return eos_data.set_index("Datetime", inplace=False)


eos_data_raw = pd.read_csv(os.path.join(config_imports.PROJECT_DIR, r"test\EOS Sim Data\Sim2\sat_kinematic_state.CSV"))


eos_data = eos_data_raw.drop(columns=["M11", "M12", "M13", "M21", "M22", "M23", "M31", "M32", "M33", 'M11.1', 'M12.1', 'M13.1', 'M21.1',
                                      'M22.1', 'M23.1', 'M31.1', 'M32.1', 'M33.1', 'X (deg/s).1', 'Y (deg/s).1', 'Z (deg/s).1'])

eos_data = to_datetime(eos_data)
time = eos_data.index

pos = eos_data[['X (km)', 'Y (km)', 'Z (km)']]
vel = eos_data[['X (km/s)', 'Y (km/s)', 'Z (km/s)']]

pos_arr  = teme_to_gcrs(time, pos.to_numpy())
vel_arr = teme_to_gcrs(time, vel.to_numpy())


with open(os.path.join(config_imports.PROJECT_DIR,  "tudsat-trace_eos.json"), "r") as f:
    eos_file = json.load(f)

sim_init_data: dict = replace_orientation_matrices(eos_file) #type: ignore

# tle1 = sim_init_data["ModelObjects"]["Orbit Model"]["Tle1"] # type: ignore
# tle2 = sim_init_data["ModelObjects"]["Orbit Model"]["Tle2"] # type: ignore
# orbit_model = SGP4.from_tle(tle1, tle2)
orbit_model = SGP4.from_elements(0.0001, 97.6, 10., 0.0, 0.0, 15.25, datetime.datetime.fromisoformat("2024-01-01T12:00:00Z"), 0.0)

asc_pos, asc_vel = orbit_model.propagate(time) #type: ignore

log_folder = os.path.join(config_imports.PROJECT_DIR, "Simulation_2025-12-16_16-50-48")
if not os.path.exists(log_folder):
    sim = Simulation.from_json(os.path.join(config_imports.PROJECT_DIR, "tudsat-trace_eos.json"))
    sim.tf = time[-1]
    sim.run()

state = pd.read_csv(os.path.join(log_folder, "state.csv"))
state.set_index("t", inplace=True)

print(time)
print(state.index)

t_hours = (time - time[0]).total_seconds() / 3600
state_pos = state[["r_eci_x", "r_eci_y", "r_eci_z"]].to_numpy()
diff = asc_pos - state_pos

fig, axs = plt.subplots(2, 2, figsize=(10, 8))
axs[0, 0].plot(t_hours, diff[:, 0])
axs[0, 0].set_title("Error X")
axs[0, 0].set_ylabel("Error [m]")
axs[0, 0].grid()
axs[0, 1].plot(t_hours, diff[:, 1])
axs[0, 1].set_title("Error Y")
axs[0, 1].grid()
axs[1, 0].plot(t_hours, diff[:, 2])
axs[1, 0].set_title("Error Z")
axs[1, 0].set_xlabel("Time [h]")
axs[1, 0].set_ylabel("Error [m]")
axs[1, 0].grid()
axs[1, 1].plot(t_hours, np.linalg.norm(asc_pos, axis=1) - np.linalg.norm(state_pos, axis=1))
axs[1, 1].set_title("Norm of Error")
axs[1, 1].set_xlabel("Time [h]")
axs[1, 1].grid()
plt.suptitle("Error of the coordinates and norm over a 3 hour simulation period")
plt.tight_layout()
plt.show()

plt.plot((time - time[0]).total_seconds() / (60 * 60), asc_pos[:, 0] - pos_arr[:, 0])  # , label=["X", "Y", "Z"])
# plt.plot((time - time[0]).total_seconds() / (60 * 60), pos_arr[:, 1])
# plt.vlines([24/15.25], [-1500], [1500], "k", "--")
plt.grid()
plt.legend()
plt.title("Error of the X coordinate over a 3 hour simulation period")
plt.xlabel("Time [h]")
plt.ylabel("Error [m]")
plt.show()
print("Max absolute distance: ", np.max(np.abs(asc_pos - pos_arr)), "m")
