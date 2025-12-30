import casadi as ca
import numpy as np
from typing import Callable, List
import control as ct

def integrator(f: ca.Function, x: ca.SX, u: ca.SX, p: ca.SX, dt: ca.SX) -> ca.SX:
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
    k1 = f(x, u, p)
    k2 = f(x + 0.5 * dt * k1, u, p)
    k3 = f(x + 0.5 * dt * k2, u, p)
    k4 = f(x + dt * k3, u, p)
    x_next = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4) #type: ignore

    x_next[:4] /= ca.norm_2(x_next[:4])

    return x_next


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
    qa = ca.SX.sym('qa', 4) #type: ignore
    qb = ca.SX.sym('qb', 4) #type: ignore
    q_ret = ca.SX.sym('q_ret', 4) #type: ignore

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
    q = ca.SX.sym('q', 4) #type: ignore
    q_conj = ca.SX.sym('q_conj', 4) #type: ignore
    x_vec_4 = ca.SX.sym('vec', 4) #type: ignore
    x = x_vec_4[:3]
    x_vec_4[3] = 0

    q_conj[:3] = q[:3]
    q_conj[3] = -q[3]

    x_ret = quat_prod(quat_prod(q, x_vec_4), q_conj)[:3] #type: ignore

    return ca.Function("quaternion_rotation", [q, x], [x_ret])
quat_rot: ca.Function = quaternion_rotation()


def build_attitude_jacobian() -> ca.Function:
    """
    Builds the Xi matrix function.

    Xi(q) = [ q_4 * I_3 + [q_{1:3} x] ]
            [ -q_{1:3}^T              ]

    Returns
    -------
    ca.Function
        A CasADi function `f(q) -> Xi`.
    """
    q = ca.SX.sym('q', 4) #type: ignore

    q_vec = q[:3]
    q_w = q[3]

    top = ca.vertcat(
        ca.horzcat(q_w, -q_vec[2], q_vec[1]),
        ca.horzcat(q_vec[2], q_w, -q_vec[0]),
        ca.horzcat(-q_vec[1], q_vec[0], q_w)
    )

    bottom = -q_vec.T

    Xi = ca.vertcat(top, bottom)

    return ca.Function("Xi", [q], [Xi], ["q"], ["Xi"])


def build_kinematics() -> ca.Function:
    """
    Builds the symbolic quaternion kinematics function.

    The function describes the time derivative of a quaternion based on angular velocity.
    q_dot = 0.5 * w (x) q

    Returns
    -------
    ca.Function
        A CasADi function `f(q, w) -> q_dot`.
    """
    q = ca.SX.sym('q', 4) #type: ignore
    w = ca.SX.sym(r'\omega', 3) #type: ignore

    qv = q[:3]
    qw = q[3]

    qv_dot = 0.5 * (qw * w + ca.cross(w, qv))
    qw_dot = -0.5 * ca.dot(w, qv)

    q_dot = ca.vertcat(qv_dot, qw_dot)

    return ca.Function("f_kin", [q, w], [q_dot], ["q", "w"], ["q_dot"])    


def build_rotational_dynamics(J_hat: np.ndarray, A_hat: np.ndarray, K_rw: np.ndarray, K_mag: np.ndarray) -> ca.Function:
    """
    Builds the symbolic rotational dynamics function (Euler's equation).

    Parameters
    ----------
    J_hat : np.ndarray
        Estimated inertia tensor of the satellite.
    A_hat : np.ndarray
        Reaction wheel alignment matrix.
    K_t_dash : np.ndarray
        Reaction wheel torque constant.
    K_mag : np.ndarray
        Magnetorquer dipole moment constant.

    Returns
    -------
    ca.Function
        A CasADi function `f(q_BI, omega, u_rw, u_mag, omega_w, B_B) -> omega_dot`.
    """
    omega = ca.SX.sym('omega', 3) #type: ignore
    h_w = ca.SX.sym("h_w", 3) #type: ignore

    u_rw = ca.SX.sym('u_rw', 3) #type: ignore
    u_mag = ca.SX.sym('u_mag', 3) #type: ignore
    
    B_B = ca.SX.sym('B_B', 3) #type: ignore


    # TODO: maybe implement build_rw and build_mag functions to deal with saturation and so on
    tau_rw = A_hat @ (K_rw * u_rw) 
    tau_mag = ca.cross(K_mag * u_mag, B_B) 

    cross_term = ca.cross(omega, J_hat @ omega + h_w)
    total_torque = tau_mag - tau_rw - cross_term
    omega_dot = ca.solve(J_hat, total_torque)

    return ca.Function(
        "f_rot",
        [omega, u_rw, u_mag, h_w, B_B],
        [omega_dot],
        ["omega", "u_rw", "u_mag", "h_w", "B_B"],
        ["omega_dot"]
    )


def build_system_dynamics(J_hat: np.ndarray, A_hat: np.ndarray, K_rw: np.ndarray, K_mag: np.ndarray):
    """
    Builds the complete symbolic dynamics model for the satellite attitude.

    This combines kinematics, rotational dynamics, and wheel dynamics into a single
    state-space function dx/dt = f(x, u, p).

    Parameters
    ----------
    J_hat, A_hat, K_e_rw, K_t_dash, K_mag : np.ndarray
        Physical parameters of the satellite and actuators.

    Returns
    -------
    ca.Function
        A CasADi function `f(x, u, B_B) -> dx`.
    """
    f_kin: ca.Function = build_kinematics()
    f_rot: ca.Function = build_rotational_dynamics(J_hat, A_hat, K_rw, K_mag)

    q_BI = ca.SX.sym('q_BI', 4) #type: ignore 
    omega = ca.SX.sym('omega', 3) #type: ignore
    h_w = ca.SX.sym("h_w", 3) #type: ignore
    x = ca.vertcat(q_BI, omega, h_w)

    u_rw = ca.SX.sym('u_rw', 3) #type: ignore
    u_mag = ca.SX.sym('u_mag', 3) #type: ignore
    u = ca.vertcat(u_rw, u_mag)

    B_field = ca.SX.sym('B_field', 3) #type: ignore

    d_q_BI = f_kin(q_BI, omega)
    d_omega = f_rot(omega, u_rw, u_mag, h_w, quat_rot(q_BI, B_field))
    d_h_w = A_hat @ (K_rw * u_rw)

    dx = ca.vertcat(d_q_BI, d_omega, d_h_w)

    return dx, x, u, B_field


def build_reduced_system_dynamics(J_hat: np.ndarray, A_hat: np.ndarray, K_rw: np.ndarray, K_mag: np.ndarray):

    dx, x, u, B_field = build_system_dynamics(J_hat, A_hat, K_rw, K_mag)

    dt = ca.SX.sym("dt") #type: ignore
    f = ca.Function("system_dynamics", [x, u, B_field], [dx], ["x", "u", "B_fiel"], ["dx"])
    x_next = integrator(f, x, u, B_field, dt)

    A = ca.jacobian(x_next, x)

    B = ca.jacobian(x_next, u)

    Xi = build_attitude_jacobian()
    xi = Xi(x[:4])

    XI = ca.SX.zeros(10, 9)
    XI[:4, :3] = xi
    XI[4:, 3:] += ca.SX.eye(6)

    A_tilde = XI.T @ A @ XI

    B_tilde = XI.T @ B

    f_lin = ca.Function("get_discrete_linearized_dynamics", [x, u, B_field, dt], [A_tilde, B_tilde], ["x_star", "u_star", "B_field_star", "dt"], ["A_tilde", "B_tilde"])
    F = ca.Function("discrete_dynamics", [x, u, B_field, dt], [x_next], ["x", "u", "B_field", "dt"], ["x_next"])


    return F, f_lin


if __name__ == "__main__":
    J_hat = np.eye(3)
    A_hat = np.eye(3)
    K_rw = np.ones(3)
    K_mag = np.ones(3)

    F, f_lin = build_reduced_system_dynamics(J_hat, A_hat, K_rw, K_mag)
    print(F)

    print(f_lin)
    