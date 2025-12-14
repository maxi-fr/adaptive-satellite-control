import casadi as ca
import numpy as np
from typing import List
from kinematics import quaternion_product as quat_prod_np

def integrator(f: ca.Function, x: ca.SX, u: ca.SX, dt: ca.SX) -> ca.SX:
    """
    Performs a single RK4 integration step for a system without parameters.

    Parameters
    ----------
    f : ca.Function
        The system dynamics function with the signature f(x, u).
    x : ca.SX
        The current state vector.
    u : ca.SX
        The current control input vector.
    dt : ca.SX
        The integration time step.

    Returns
    -------
    ca.SX
        The state vector at the next time step.
    """
    k1 = f(x, u)
    k2 = f(x + 0.5 * dt * k1, u)
    k3 = f(x + 0.5 * dt * k2, u)
    k4 = f(x + dt * k3, u)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

def quaternion_product() -> ca.Function:
    """
    Builds a CasADi function for quaternion multiplication.

    The function implements the product of two quaternions, qa and qb,
    assuming scalar-last format [qx, qy, qz, qw].

    Returns
    -------
    ca.Function
        A CasADi function with signature `f(qa, qb) -> q_ret`.
    """
    qa = ca.SX.sym('qa', 4)
    qb = ca.SX.sym('qb', 4)
    q_ret = ca.SX.sym('q_ret', 4)

    q_ret[:3] = qb[3]*qa[:3] + qa[3]*qb[:3] + ca.cross(qa[:3], qb[:3])
    q_ret[3] = qa[3]*qb[3] - ca.dot(qa[:3], qb[:3])

    return ca.Function("quaternion_product", [qa, qb], [q_ret])
quat_prod: ca.Function = quaternion_product()

def quaternion_rotation() -> ca.Function:
    """
    Builds a CasADi function for rotating a 3D vector by a quaternion.

    Returns
    -------
    ca.Function
        A CasADi function with signature `f(q, x) -> x_ret`.
    """
    q = ca.SX.sym('q', 4)
    x_vec_4 = ca.SX.sym('vec', 4)
    x = x_vec_4[:3]
    x_vec_4[3] = 0

    q_conj = q.copy()
    q_conj[3] = -q[3]

    x_ret = quat_prod(quat_prod(q, x_vec_4), q_conj)[:3]

    return ca.Function("quaternion_rotation", [q, x], [x_ret])
quat_rot: ca.Function = quaternion_rotation()

def _attitude_error_vec(q_ref: np.ndarray, q_est: np.ndarray) -> np.ndarray:
    q_ref = np.asarray(q_ref, dtype=float).reshape(4)
    q_est = np.asarray(q_est, dtype=float).reshape(4)

    q_est_conj = q_est.copy()
    q_est_conj[:3] *= -1.0

    q_err = quat_prod_np(q_ref, q_est_conj)
    if q_err[3] < 0:
        q_err = -q_err
    e_q = 2.0 * q_err[:3]

    return e_q


def build_kinematics() -> ca.Function:
    """
    Builds the symbolic quaternion kinematics function.

    The function describes the time derivative of a quaternion based on angular velocity.
    q_dot = 0.5 * Omega(w) * q

    Returns
    -------
    ca.Function
        A CasADi function `f(q, w) -> q_dot`.
    """
    q = ca.SX.sym('q', 4)
    w = ca.SX.sym(r'\omega', 3)

    qv = q[:3]
    qw = q[3]

    qv_dot = 0.5 * (qw * w + ca.cross(w, qv))
    qw_dot = -0.5 * ca.dot(w, qv)

    q_dot = ca.vertcat(qv_dot, qw_dot)

    return ca.Function("f_kin", [q, w], [q_dot], ["q", "w"], ["q_dot"])


def build_rotational_dynamics(J_hat: np.ndarray, J_w: np.ndarray, A_hat: np.ndarray, K_e_rw: np.ndarray, K_t_dash: np.ndarray, K_mag: np.ndarray) -> ca.Function:
    """
    Builds the symbolic rotational dynamics function (Euler's equation).

    Parameters
    ----------
    J_hat : np.ndarray
        Estimated inertia tensor of the satellite.
    J_w : np.ndarray
        Inertia of reaction wheels.
    A_hat : np.ndarray
        Estimated reaction wheel alignment matrix.
    K_e_rw : np.ndarray
        Reaction wheel back-EMF constant.
    K_t_dash : np.ndarray
        Reaction wheel torque constant.
    K_mag : np.ndarray
        Magnetorquer dipole moment constant.

    Returns
    -------
    ca.Function
        A CasADi function `f(q_BI, omega, u_rw, u_mag, omega_w, B_B) -> omega_dot`.
    """
    q_BI = ca.SX.sym('q_BI', 4)
    omega = ca.SX.sym('omega', 3)
    omega_w = ca.SX.sym("omega_w", 3)

    u_rw = ca.SX.sym('u_rw', 3)
    u_mag = ca.SX.sym('u_mag', 3)
    
    B_B = ca.SX.sym('B_B', 3)


    h_int = A_hat @ (J_w * (omega_w + A_hat @ omega)) 
    tau_rw = A_hat @ K_t_dash * (u_rw - K_e_rw * omega_w) 
    tau_mag = ca.cross(K_mag * u_mag, B_B) 

    cross_term = ca.cross(omega, J_hat @ omega + h_int)
    total_torque = tau_mag - tau_rw - cross_term
    omega_dot = ca.solve(J_hat, total_torque)

    return ca.Function(
        "f_rot",
        [q_BI, omega, u_rw, u_mag, omega_w, B_B],
        [omega_dot],
        ["q_BI", "omega", "u_rw", "u_mag", "omega_w", "B_B"],
        ["omega_dot"],
    )

def build_wheel_dynamics(J_w: np.ndarray, A_hat: np.ndarray) -> ca.Function:
    """
    Builds the symbolic dynamics for the reaction wheels.

    Parameters
    ----------
    J_w : np.ndarray
        Inertia of reaction wheels.
    A_hat : np.ndarray
        Estimated reaction wheel alignment matrix.

    Returns
    -------
    ca.Function
        A CasADi function `f(u_rw, omega_dot) -> omega_w_dot`.
    """
    u_rw = ca.SX.sym('u_rw', 3)
    omega_dot = ca.SX.sym('omega_dot', 3)

    omega_w_dot = u_rw / J_w - A_hat @ omega_dot

    return ca.Function("f_wheel", [u_rw, omega_dot], [omega_w_dot], ["u_rw", "omega_dot"], ["omega_w_dot"])


def build_system_dynamics(J_hat: np.ndarray, J_w: np.ndarray, A_hat: np.ndarray, K_e_rw: np.ndarray, K_t_dash: np.ndarray, K_mag: np.ndarray) -> ca.Function:
    """
    Builds the complete symbolic dynamics model for the satellite attitude.

    This combines kinematics, rotational dynamics, and wheel dynamics into a single
    state-space function dx/dt = f(x, u, p).

    Parameters
    ----------
    J_hat, J_w, A_hat, K_e_rw, K_t_dash, K_mag : np.ndarray
        Physical parameters of the satellite and actuators.

    Returns
    -------
    ca.Function
        A CasADi function `f(x, u, B_B) -> dx`.
    """
    f_kin: ca.Function = build_kinematics()
    f_rot: ca.Function = build_rotational_dynamics(J_hat, J_w, A_hat, K_e_rw, K_t_dash, K_mag)
    f_wheel: ca.Function = build_wheel_dynamics(J_w, A_hat)

    q_BI = ca.SX.sym('q_BI', 4)
    omega = ca.SX.sym('omega', 3)
    omega_w = ca.SX.sym("omega_w", 3)
    x = ca.vertcat(q_BI, omega, omega_w)

    u_rw = ca.SX.sym('u_rw', 3)
    u_mag = ca.SX.sym('u_mag', 3)
    u = ca.vertcat(u_rw, u_mag)

    B_B = ca.SX.sym('B_B', 3)

    d_q_BI = f_kin(q_BI, omega)
    d_omega = f_rot(q_BI, omega, u_rw, u_mag, omega_w, B_B)
    d_omega_w = f_wheel(u_rw, d_omega)

    dx = ca.vertcat(d_q_BI, d_omega, d_omega_w)

    return ca.Function(
        "system_dynamics",
        [x, u, B_B],
        [dx],
        ["x", "u", "B_B"],
        ["dx"],
    )

def build_ekf_process_model():
    """
    Builds the symbolic dynamics model for the Extended Kalman Filter (EKF).

    This function defines the state-space model dx/dt = f(x, u, p) for the EKF,
    where the state `x` includes the quaternion, angular velocity, and reaction wheel speeds,
    and the parameters `p` include the magnetic field vector.

    Returns
    -------
    ca.Function
        A CasADi function `f(x, p) -> dx`.
    """
    # Define symbolic variables for the EKF state and inputs
    q_BI = ca.SX.sym('q_BI', 4)
    omega = ca.SX.sym('omega', 3)
    
    bias = ca.SX.sym('bias', 3)

    f_kin = build_kinematics()

    dq = f_kin(q_BI, omega - bias)

    dbias = ca.SX.zeros(3)

    x = ca.vertcat(q_BI, bias)

    dx = ca.vertcat(dq, dbias)

    return ca.Function("ekf_dynamics", [x], [dx], ["x"], ["dx"])

def simple_rw_mag_controller(
    q_est: np.ndarray,
    omega_est: np.ndarray,
    q_ref: np.ndarray,
    omega_ref: np.ndarray,
    Kp_q: float,
    Kd_w: float,
    K_t_rw: float,
) -> tuple[np.ndarray, np.ndarray]:
    q_est = np.asarray(q_est, dtype=float).reshape(4)
    q_ref = np.asarray(q_ref, dtype=float).reshape(4)
    omega_est = np.asarray(omega_est, dtype=float).reshape(3)
    omega_ref = np.asarray(omega_ref, dtype=float).reshape(3)

    # error of attitude and angular velocity
    e_q = _attitude_error_vec(q_ref, q_est)
    e_w = omega_est - omega_ref
    # body moment that we want
    tau_cmd = -Kp_q * e_q - Kd_w * e_w
    # map
    K_t_rw = float(K_t_rw)
    u_rw = tau_cmd / K_t_rw   # current to 3 wheels
    # magnetorquers turned off for now
    u_mag = np.zeros(3)

    return u_rw, u_mag