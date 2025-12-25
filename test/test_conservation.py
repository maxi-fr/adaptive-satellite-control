import config_imports
import os
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from IPython.display import display
from simulation import Simulation

# -------------------------------------------------
# INPUTS (replace with your loaded logs)
# -------------------------------------------------
# q_BI : (N, 4) array, scalar-last quaternion
# omega_B : (N, 3) array, body angular velocity [rad/s]

states = pd.read_csv(os.path.join(config_imports.PROJECT_DIR, "Simulation_2025-12-25_12-29-44", "state.csv"))


q_BI = states[["q_BI_x", "q_BI_y","q_BI_z","q_BI_w"]].to_numpy()
omega_B = states[["omega_x", "omega_y", "omega_z"]].to_numpy()

display(states[["q_BI_x", "q_BI_y","q_BI_z","q_BI_w", "omega_x", "omega_y", "omega_z"]])

sim = Simulation.from_json(os.path.join(config_imports.PROJECT_DIR, "tudsat-trace_eos.json"), enable_viz=False, enable_log=False)
sat = sim.sat

J = sat.J_B

H_I = []
H_norm = []
T_rot = []

for q, omega in zip(q_BI, omega_B):
    R_BI = R.from_quat(q, scalar_first=False)
    H_B = J @ omega
    
    H_I_k = R_BI.inv().apply(H_B)

    H_I.append(H_I_k)
    H_norm.append(np.linalg.norm(H_I_k))
    T_rot.append(0.5 * omega @ J @ omega)

# -------------------------------------------------
# CONSERVATION METRICS
# -------------------------------------------------
H0 = H_I[0]
H0_unit = H0 / np.linalg.norm(H0)

# Direction error (angle between H(t) and H(0))
angle_error = np.array([np.arccos(np.clip(np.dot(H0_unit, H / np.linalg.norm(H)),-1.0, 1.)) for H in H_I])

# Relative magnitude error
rel_H_error = (np.array(H_norm) - H_norm[0]) / H_norm[0]

# Relative energy error
rel_T_error = (np.array(T_rot) - T_rot[0]) / T_rot[0]

# -------------------------------------------------
# PRINT SUMMARY
# -------------------------------------------------
print("Angular momentum magnitude:")
print(f"  max rel error: {np.max(np.abs(rel_H_error)):.3e}")

print("Angular momentum direction:")
print(f"  max angle error [rad]: {np.max(angle_error):.3e}")

print("Rotational kinetic energy:")
print(f"  max rel error: {np.max(np.abs(rel_T_error)):.3e}")

# -------------------------------------------------
# PLOTS (optional but recommended)
# -------------------------------------------------
plt.figure()
plt.plot(rel_H_error)
plt.title("Relative Angular Momentum Magnitude Error")
plt.xlabel("Sample")
plt.ylabel("Relative error")
plt.grid()

plt.figure()
plt.plot(angle_error)
plt.title("Angular Momentum Direction Error [rad]")
plt.xlabel("Sample")
plt.ylabel("Angle [rad]")
plt.grid()

plt.figure()
plt.plot(rel_T_error)
plt.title("Relative Rotational Energy Error")
plt.xlabel("Sample")
plt.ylabel("Relative error")
plt.grid()

plt.show()
