import numpy as np


def uncontrolled(state: np.ndarray):

    return np.zeros(3)



import casadi as ca
import numpy as np

def integrator(f: ca.Function, x: ca.SX, u: ca.SX, dt: ca.SX):
    k1 = f(x, u)
    k2 = f(x + 0.5 * dt * k1, u)
    k3 = f(x + 0.5 * dt * k2, u)
    k4 = f(x + dt * k3, u)
    return x + (dt/ 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

def quaternion_product():
    qa = ca.SX.sym('qa', 4)
    qb = ca.SX.sym('qb', 4)
    q_ret = ca.SX.sym('q_ret', 4)

    q_ret[:3] = qb[3]*qa[:3] + qa[3]*qb[:3] + ca.cross(qa[:3], qb[:3])
    q_ret[3] = qa[3]*qb[3] - ca.dot(qa[:3], qb[:3])

    return ca.Function("quaternion_product", [qa, qb], [q_ret])
quat_prod = quaternion_product()

def quaternion_rotation():
    q = ca.SX.sym('q', 4)
    x_ = ca.SX.sym('vec', 4)
    x = x_[:3]
    x_[3] = 0
    x_ret = quat_prod(quat_prod(q, x_), q)[:3] #type: ignore

    return ca.Function("quaternion_rotation", [q, x], [x_ret])
quat_rot = quaternion_rotation()



dipole_magnitude = 7.77e22 # Am^2
M_dipole = dipole_magnitude * np.array([np.sin(169.7)*np.cos(108.2), np.sin(169.7)*np.sin(108.2), np.cos(169.7)])

def build_system_dynamics(J_hat: np.ndarray, J_w: np.ndarray, A_hat: np.ndarray, K_e_rw: np.ndarray, K_t_dash: np.ndarray, K_mag: np.ndarray):
    """ 
    K_mag = N*A/R, L = 0 (approx)
    K_t_dash = K_t / R, L = 0 (approx) 
    no disturbance torques
    satellite position as a paramter

    """

    # symbolic dynamics model for estimation and control
    q_BI = ca.SX.sym('q_BI', 4)
    omega = ca.SX.sym('omega', 3)
    omega_w = ca.SX.sym("omega_w", 3)
    x = ca.vertcat(q_BI, omega, omega_w)
    u = ca.SX.sym('inputs', 6)
    u_rw = u[:3]
    u_mag = u[3:]

    r_ECI = ca.SX.sym('r_ECI', 3)                        

    h_int = A_hat @ (J_w * (omega_w + A_hat @ omega)) 
    tau_rw = A_hat @ K_t_dash * (u_rw - K_e_rw * omega_w) 


    r_norm = ca.norm_2(r_ECI)
    r_hat = r_ECI / r_norm

    B_I = (3 * r_hat * (ca.dot(M_dipole, r_hat)) - M_dipole) / r_norm**3

    tau_mag = ca.cross(K_mag * u_mag, quat_rot(q_BI, B_I)) 


    qx, qy, qz, qw = q_BI[0], q_BI[1], q_BI[2], q_BI[3]
    wx, wy, wz = omega[0], omega[1], omega[2]
    q_dot = 0.5 * ca.vertcat(
        -qx*wx - qy*wy - qz*wz,
        qw*wx + qy*wz - qz*wy,
        qw*wy - qx*wz + qz*wx,
        qw*wz + qx*wy - qy*wx
    )

    cross_term = ca.cross(omega, J_hat @ omega + h_int)
    total_torque = tau_mag - tau_rw - cross_term
    omega_dot = ca.solve(J_hat, total_torque)
    omega_w_dot = u_rw / J_w - A_hat @ omega_dot

    dx = ca.vertcat(q_dot, omega_dot, omega_w_dot)

    f = ca.Function("attitude_dynamics", [x, u, r_ECI], [dx])

    return f