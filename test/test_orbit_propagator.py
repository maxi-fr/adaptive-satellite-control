import config_imports
import os
from dynamics import SGP4
from satellite import replace_orientation_matrices
import json
from astropy import units as u
from astropy.coordinates import SkyCoord, CartesianRepresentation, TEME
from astropy.time import Time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime



def teme_to_gcrs(t: datetime.datetime, x_TEME):
    teme = TEME(obstime=Time(t, format="datetime", scale="utc"))
    x_ECI = SkyCoord(CartesianRepresentation(x_TEME[:, 0], x_TEME[:, 1], x_TEME[:, 2], unit=u.km), frame=teme,
                     representation_type='cartesian').transform_to("gcrs").cartesian.xyz.to(u.m).value  # type: ignore
    return x_ECI.T.squeeze()


def to_datetime(eos_data: pd.DataFrame):
    eos_data["Datetime"] = eos_data[["Date", "Time", "Time since start (s)"]].apply(lambda x: datetime.datetime.fromisoformat(
        x.iloc[0] + "T" + x.iloc[1] + "Z") + datetime.timedelta(seconds=x.iloc[2] % 1.0), axis=1)
    eos_data = eos_data.drop(columns=["Date", "Time", "Time since start (s)"])
    return eos_data.set_index("Datetime", inplace=False)


eos_data_raw = pd.read_csv(os.path.join(os.path.dirname(config_imports.HERE), r"test\EOS Sim Data\Sim1\sat_kinematic_state.CSV"))


eos_data = eos_data_raw.drop(columns=["M11", "M12", "M13", "M21", "M22", "M23", "M31", "M32", "M33", 'M11.1', 'M12.1', 'M13.1', 'M21.1',
                                      'M22.1', 'M23.1', 'M31.1', 'M32.1', 'M33.1', 'X (deg/s).1', 'Y (deg/s).1', 'Z (deg/s).1'])

eos_data = to_datetime(eos_data)
time = eos_data.index

pos = eos_data[['X (km)', 'Y (km)', 'Z (km)']]
vel = eos_data[['X (km/s)', 'Y (km/s)', 'Z (km/s)']]

pos_arr  = teme_to_gcrs(time, pos.to_numpy())
vel_arr = teme_to_gcrs(time, vel.to_numpy())


with open(os.path.join(os.path.dirname(config_imports.HERE), "tudsat-trace_eos.json"), "r") as f:
    eos_file = json.load(f)

sim_init_data: dict = replace_orientation_matrices(eos_file)

# tle1 = sim_init_data["ModelObjects"]["Orbit Model"]["Tle1"] # type: ignore
# tle2 = sim_init_data["ModelObjects"]["Orbit Model"]["Tle2"] # type: ignore
# orbit_model = SGP4.from_tle(tle1, tle2)
orbit_model = SGP4.from_elements(0.0001, 97.6, 10., 0.0, 0.0, 15.25, datetime.datetime.fromisoformat("2024-01-01T12:00:00Z"), 0.0)

asc_pos, asc_vel = orbit_model.propagate(time)


plt.plot((time - time[0]).total_seconds() / (60 * 60), asc_pos[:, 0] - pos_arr[:, 0])  # , label=["X", "Y", "Z"])
# plt.plot((time - time[0]).total_seconds() / (60 * 60), pos_arr[:, 1])
# plt.vlines([24/15.25], [-1500], [1500], "k", "--")
plt.grid()
plt.legend()
plt.title("Error of the X coordinate over a 3 hour simulation period")
plt.show()
print("Max absolute distance: ", np.max(np.abs(asc_pos - pos_arr)), "m")
