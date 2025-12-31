import casadi as ca
import numpy as np
from typing import Callable, List, Tuple
import control as ct

def integrator(f: Callable[[ca.SX, ca.SX, ca.SX], ca.SX], x: ca.SX, u: ca.SX, p: ca.SX, dt: ca.SX|float) -> ca.SX:
    """
    Performs a single RK4 integration step using a Python callable for dynamics.

    This function also normalizes the quaternion part of the state vector (first 4 elements)
    after the integration step to ensure unit norm.

    Parameters
    ----------
    f : Callable[[ca.SX, ca.SX, ca.SX], ca.SX]
        The system dynamics function with the signature f(x, u, p) -> dx.
    x : ca.SX
        The current state vector.
    u : ca.SX
        The current control input vector.
    p : ca.SX
        Time-varying parameters (e.g., magnetic field).
    dt : ca.SX | float
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

    x_next[:4] = x_next[:4]/ca.norm_2(x_next[:4])

    return x_next


def quaternion_product(qa: ca.SX, qb: ca.SX) -> ca.SX:
    """
    Symbolic quaternion multiplication.

    Computes the product of two quaternions qa and qb.
    Assumes scalar-last format: q = [qx, qy, qz, qw].

    Parameters
    ----------
    qa : ca.SX
        First quaternion (4x1).
    qb : ca.SX
        Second quaternion (4x1).

    Returns
    -------
    ca.SX
        The product quaternion (4x1).
    """
    q_ret_vec = qb[3]*qa[:3] + qa[3]*qb[:3] + ca.cross(qa[:3], qb[:3])
    q_ret_w = qa[3]*qb[3] - ca.dot(qa[:3], qb[:3])

    return ca.vertcat(q_ret_vec, q_ret_w)


def quaternion_rotation(q: ca.SX, x: ca.SX) -> ca.SX:
    """
    Rotates a 3D vector x by a quaternion q.
    
    Implements the operation v' = q * v * q_conj.

    Parameters
    ----------
    q : ca.SX
        Rotation quaternion (4x1), scalar-last.
    x : ca.SX
        3D vector to rotate (3x1).

    Returns
    -------
    ca.SX
        Rotated 3D vector (3x1).
    """
    q_conj = ca.vertcat(q[:3], -q[3])
    x_vec_4 = ca.vertcat(x, 0)

    # q * x * q_conj
    temp = quaternion_product(q, x_vec_4)
    x_ret = quaternion_product(temp, q_conj)[:3]

    return x_ret


def attitude_jacobian(q: ca.SX) -> ca.SX:
    """
    Builds the Xi matrix symbolically.

    The Xi matrix maps angular velocity to quaternion derivative:
    dq/dt = 0.5 * Xi(q) * omega.
    Also 

    Xi(q) = | q_4 * I_3 + [q_{1:3} x] |
            | -q_{1:3}^T              |

    Parameters
    ----------
    q : ca.SX
        Quaternion (4x1), scalar-last.

    Returns
    -------
    ca.SX
        The Xi matrix (4x3).
    """
    q_vec = q[:3]
    q_w = q[3]

    Xi = ca.vertcat(
        ca.horzcat(q_w, -q_vec[2], q_vec[1]),
        ca.horzcat(q_vec[2], q_w, -q_vec[0]),
        ca.horzcat(-q_vec[1], q_vec[0], q_w),
        -q_vec.T
    )

    return Xi


def kinematics(q: ca.SX, w: ca.SX) -> ca.SX:
    """
    Symbolic quaternion kinematics.

    Computes the time derivative of a quaternion based on angular velocity.
    q_dot = 0.5 * q (x) w  (quaternion multiplication)

    Parameters
    ----------
    q : ca.SX
        Current quaternion (4x1).
    w : ca.SX
        Angular velocity in body frame (3x1).

    Returns
    -------
    ca.SX
        Quaternion derivative (4x1).
    """
    qv = q[:3]
    qw = q[3]

    qv_dot = 0.5 * (qw * w + ca.cross(w, qv))
    qw_dot = -0.5 * ca.dot(w, qv)

    # TODO: could also just be 0.5 * (attitude_jacobian(q) @ w)

    return ca.vertcat(qv_dot, qw_dot)


def rotational_dynamics(omega: ca.SX, u_rw: ca.SX, u_mag: ca.SX, h_w: ca.SX, B_B: ca.SX, J_hat: np.ndarray) -> ca.SX:
    """
    Symbolic rotational dynamics (Euler's equation).

    Computes angular acceleration considering control torques and gyroscopic terms.

    Parameters
    ----------
    omega : ca.SX
        Angular velocity (3x1).
    u_rw : ca.SX
        Reaction wheel control torque (3x1).
    u_mag : ca.SX
        Commanded magnetic torque vector (3x1).
    h_w : ca.SX
        Reaction wheel angular momentum (3x1).
    B_B : ca.SX
        Magnetic field vector in body frame (3x1).
    J_hat : np.ndarray
        Estimated inertia tensor of the satellite (3x3).

    Returns
    -------
    ca.SX
        Angular acceleration (3x1).
    """
    # TODO: maybe implement build_rw and build_mag functions to deal with saturation and so on
    tau_rw = -u_rw 

    b = B_B/ca.norm_2(B_B)

    tau_mag = (ca.SX.eye(3) - b @ b.T) @ u_mag 

    cross_term = ca.cross(omega, J_hat @ omega + h_w)
    total_torque = tau_mag - tau_rw - cross_term
    
    return ca.solve(J_hat, total_torque)


def satellite_dynamics(x: ca.SX, u: ca.SX, B_eci: ca.SX, J_hat: np.ndarray) -> ca.SX:
    """
    Computes the state derivative dx/dt for the satellite system.

    State vector x: [q_BI (4), omega (3), h_w (3)] (size 10).
    Input vector u: [u_rw (3), u_mag (3)] (size 6).

    Parameters
    ----------
    x : ca.SX
        State vector.
    u : ca.SX
        Control input vector.
    B_eci : ca.SX
        Magnetic field vector in inertial frame (3x1).
    J_hat : np.ndarray
        Inertia matrix.

    Returns
    -------
    ca.SX
        State derivative dx/dt (10x1).
    """
    q_BI = x[:4]
    omega = x[4:7]
    h_w = x[7:10]

    u_rw = u[:3]
    u_mag = u[3:]

    d_q_BI = kinematics(q_BI, omega)

    d_omega = rotational_dynamics(omega, u_rw, u_mag, h_w, quaternion_rotation(q_BI, B_eci), J_hat)
    d_h_w = u_rw

    return ca.vertcat(d_q_BI, d_omega, d_h_w)


def build_system_dynamics(J_hat: np.ndarray) -> ca.Function:
    """
    Builds the complete symbolic dynamics model for the satellite attitude.

    This combines kinematics, rotational dynamics, and wheel dynamics into a single
    state-space function dx/dt = f(x, u, p).

    Parameters
    ----------
    J_hat : np.ndarray
        Inertia tensor of the satellite.

    Returns
    -------
    ca.Function
        A CasADi function `f(x, u, B_B) -> dx`.
    """
    q_BI = ca.SX.sym('q_BI', 4) #type: ignore 
    omega = ca.SX.sym('omega', 3) #type: ignore
    h_w = ca.SX.sym("h_w", 3) #type: ignore
    x = ca.vertcat(q_BI, omega, h_w)

    u_rw = ca.SX.sym('u_rw', 3) #type: ignore
    u_mag = ca.SX.sym('u_mag', 3) #type: ignore
    u = ca.vertcat(u_rw, u_mag)

    B_eci = ca.SX.sym('B_eci', 3) #type: ignore

    dx = satellite_dynamics(x, u, B_eci, J_hat)

    return ca.Function("continuous_dynamics", [x, u, B_eci], [dx], ["x", "u", "B_eci"], ["dx"])


def build_reduced_system_dynamics(dt: float, J_hat: np.ndarray) -> Tuple[ca.Function, ca.Function, ca.Function]:
    """
    Builds the discrete-time linearized dynamics for the reduced attitude system.

    This function performs RK4 integration symbolically to obtain discrete dynamics,
    linearizes the system around a symbolic operating point, and applies a coordinate
    change using the attitude Jacobian (Xi) to handle the quaternion constraint.

    Parameters
    ----------
    dt : float
        Sampling time step.
    J_hat : np.ndarray
        Inertia tensor.

    Returns
    -------
    Tuple[ca.Function, ca.Function, ca.Function]
        - F: Discrete dynamics function `x_next = F(x, u, B_eci)`.
        - A_func: Linearized state matrix function `A_tilde = A_func(x, u, B_eci)`.
        - B_func: Linearized input matrix function `B_tilde = B_func(x, u, B_eci)`.
    """

    q_BI = ca.SX.sym('q', 4) #type: ignore 
    omega = ca.SX.sym('omega', 3) #type: ignore
    h_w = ca.SX.sym("h_w", 3) #type: ignore
    x = ca.vertcat(q_BI, omega, h_w)

    u_rw = ca.SX.sym('u_rw', 3) #type: ignore
    u_mag = ca.SX.sym('u_mag', 3) #type: ignore
    u = ca.vertcat(u_rw, u_mag)

    B_eci = ca.SX.sym('B_eci', 3) #type: ignore
    
    # Define a lambda for the dynamics to pass to the integrator
    dyn_fn = lambda x_s, u_s, p_s: satellite_dynamics(x_s, u_s, p_s, J_hat)
    
    x_next = integrator(dyn_fn, x, u, B_eci, dt)

    A = ca.jacobian(x_next, x)

    # print("Check the rank of A: should be 9")
    # ax = np.random.randn(10)
    # ax[:4] = ax[:4]/np.linalg.norm(ax[:4])
    # Ax = ca.Function("A", [x, u, B_eci], [A], ["x", "u", "B_eci"], ["A"])
    # print("A: ", np.array(Ax(ax, np.zeros(6), np.zeros(3))).shape)
    # print("rank(A) = ", np.linalg.matrix_rank(Ax(ax, np.zeros(6), np.zeros(3))))

    B = ca.jacobian(x_next, u)

    xi = attitude_jacobian(x[:4])

    E = ca.SX.zeros(10, 9)
    E[:4, :3] = xi
    E[4:, 3:] = ca.SX.eye(6)

    A_tilde = E.T @ A @ E

    B_tilde = E.T @ B

    A_func = ca.Function("get_discrete_linearized_dynamics", [x, u, B_eci], [A_tilde], ["x_star", "u_star", "B_eci_star"], ["A_tilde"])
    B_func = ca.Function("get_discrete_linearized_dynamics", [x, u, B_eci], [B_tilde], ["x_star", "u_star", "B_eci_star"], ["B_tilde"])
    F = ca.Function("discrete_dynamics", [x, u, B_eci], [x_next], ["x", "u", "B_eci"], ["x_next"])

    return F, A_func, B_func



def error_dynamics(x: ca.SX, u: ca.SX, B_orc: ca.SX, omega_c: np.ndarray, J_hat: np.ndarray) -> ca.SX:
    """
    Computes the state derivative dx/dt for the attitude error dynamics.

    State vector x: [q_err (4), omega_err (3), h_w (3)] (size 10).
    Input vector u: [u_rw (3), u_mag (3)] (size 6).

    Parameters
    ----------
    x : ca.SX
        State vector (error state).
    u : ca.SX
        Control input vector.
    B_orc : ca.SX
        Magnetic field vector in the Orbit Reference Frame (ORC) (3x1).
    omega_c : np.ndarray
        Orbital angular velocity vector (3x1).
    J_hat : np.ndarray
        Estimated inertia tensor of the satellite (3x3).

    Returns
    -------
    ca.SX
        State derivative dx/dt (10x1).
    """

    q_err = x[:4]
    omega_err = x[4:7]
    h_w = x[7:10]

    u_rw = u[:3]
    u_mag = u[3:]
    
    omega_c_b = quaternion_rotation(q_err, omega_c)
    omega = omega_err + omega_c_b
    
    q_err[3] = 1 - ca.dot(q_err[:3], q_err[:3])

    d_omega = rotational_dynamics(omega, u_rw, u_mag, h_w, quaternion_rotation(q_err, B_orc), J_hat)

    d_q_err = kinematics(q_err, omega_err)
    d_omega_err = d_omega + ca.cross(omega_err, omega_c_b)
    d_h_w = -u_rw

    return ca.vertcat(d_q_err, d_omega_err, d_h_w)


def build_reduced_error_dynamics(dt: float, omega_c: np.ndarray, J_hat: np.ndarray) -> Tuple[ca.Function, ca.Function, ca.Function]:
    """
    Builds the discrete-time linearized dynamics for the reduced attitude error system.

    This function performs RK4 integration symbolically to obtain discrete dynamics,
    linearizes the system around a symbolic operating point, and applies a coordinate
    change using the attitude Jacobian (Xi) to handle the quaternion constraint.

    Parameters
    ----------
    dt : float
        Sampling time step.
    omega_c : np.ndarray
        Orbital angular velocity vector (3x1).
    J_hat : np.ndarray
        Inertia tensor.

    Returns
    -------
    Tuple[ca.Function, ca.Function, ca.Function]
        - F: Discrete dynamics function `x_next = F(x, u, B_orc)`.
        - A_func: Linearized state matrix function `A_tilde = A_func(x, u, B_orc)`.
        - B_func: Linearized input matrix function `B_tilde = B_func(x, u, B_orc)`.
    """

    q_BI = ca.SX.sym('q', 4) #type: ignore 
    omega = ca.SX.sym('omega', 3) #type: ignore
    h_w = ca.SX.sym("h_w", 3) #type: ignore
    x = ca.vertcat(q_BI, omega, h_w)

    u_rw = ca.SX.sym('u_rw', 3) #type: ignore
    u_mag = ca.SX.sym('u_mag', 3) #type: ignore
    u = ca.vertcat(u_rw, u_mag)

    B_orc = ca.SX.sym('B_orc', 3) #type: ignore
    
    # Define a lambda for the dynamics to pass to the integrator
    dyn_fn = lambda x_s, u_s, p_s: error_dynamics(x_s, u_s, p_s, omega_c, J_hat)
    
    x_next = integrator(dyn_fn, x, u, B_orc, dt)

    A = ca.jacobian(x_next, x)

    B = ca.jacobian(x_next, u)

    xi = attitude_jacobian(x[:4])

    E = ca.SX.zeros(10, 9)
    E[:4, :3] = xi
    E[4:, 3:] = ca.SX.eye(6)

    A_tilde = E.T @ A @ E

    B_tilde = E.T @ B

    A_func = ca.Function("get_discrete_linearized_dynamics", [x, u, B_orc], [A_tilde], ["x_star", "u_star", "B_orc_star"], ["A_tilde"])
    B_func = ca.Function("get_discrete_linearized_dynamics", [x, u, B_orc], [B_tilde], ["x_star", "u_star", "B_orc_star"], ["B_tilde"])
    F = ca.Function("discrete_dynamics", [x, u, B_orc], [x_next], ["x", "u", "B_orc"], ["x_next"])

    return F, A_func, B_func