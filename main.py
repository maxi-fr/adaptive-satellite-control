import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from dynamics import CubeSat, KeplarElements

dt = 1.0          # [s]
tf = 10*3600.0       # simulate 10 hours
time = np.arange(0, tf+dt, dt)

a = 6771e3           # semi-major axis [m]
e = 0.001            # small eccentricity
i = np.deg2rad(51.6) # inclination [rad]
raan = 0.0
arg_pe = 0.0
M0 = 0.0

kep = KeplarElements(a, e, i, raan, arg_pe, M0)
initial_att = R.from_euler('yxz', [0, 0, 0], degrees=True)  # identity attitude
sat = CubeSat(kep, initial_att)


r_sat_log = np.zeros((time.size, 13))
r_kep_log = np.zeros((time.size, 3))

r_sat_log[0] = sat.state
state_kep = kep.to_eci()  # [r, v]
r_kep_log[0] = state_kep[0:3]

for k in range(1, time.size):
    sat.update(dt)             
    state_kep = kep.to_eci(time[k])

    r_sat_log[k] = sat.state[0:3]
    r_kep_log[k] = state_kep[0:3]

pos_error = np.linalg.norm(r_sat_log - r_kep_log, axis=1)

plt.figure()
plt.plot(time / 60, pos_error)
plt.xlabel('Time [min]')
plt.ylabel('‖r_sat - r_kep‖ [m]')
plt.title('Position Error: CubeSat vs Keplerian Propagation')
plt.grid(True)
plt.show()

plt.figure()
plt.plot(r_kep_log[:,0]/1e3, r_kep_log[:,1]/1e3, label='Keplerian')
plt.plot(r_sat_log[:,0]/1e3, r_sat_log[:,1]/1e3, '--', label='Full Dynamics')
plt.xlabel('x [km]')
plt.ylabel('y [km]')
plt.axis('equal')
plt.legend()
plt.title('Orbit Comparison in ECI Frame')
plt.grid(True)
plt.show()
