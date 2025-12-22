import casadi as ca
import numpy as np
from typing import Callable, List
import control as ct

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
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4) #type: ignore

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


def build_rotational_dynamics(J_hat: np.ndarray, A_hat: np.ndarray, K_t_dash: np.ndarray, K_mag: np.ndarray) -> ca.Function:
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
    tau_rw = A_hat @ (K_t_dash * u_rw) 
    tau_mag = ca.cross(K_mag * u_mag, B_B) 

    cross_term = ca.cross(omega, J_hat @ omega + h_w)
    total_torque = tau_mag - tau_rw - cross_term
    omega_dot = ca.solve(J_hat, total_torque)

    return ca.Function(
        "f_rot",
        [omega, u_rw, u_mag, h_w, B_B],
        [omega_dot],
        ["q_BI", "omega", "u_rw", "u_mag", "h_w", "B_B"],
        ["omega_dot"]
    )


def build_system_dynamics(J_hat: np.ndarray, A_hat: np.ndarray, K_t_dash: np.ndarray, K_mag: np.ndarray) -> ca.Function:
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
    f_rot: ca.Function = build_rotational_dynamics(J_hat, A_hat, K_t_dash, K_mag)

    q_BI = ca.SX.sym('q_BI', 4) #type: ignore 
    omega = ca.SX.sym('omega', 3) #type: ignore
    h_w = ca.SX.sym("h_w", 3) #type: ignore
    x = ca.vertcat(q_BI, omega, h_w)

    u_rw = ca.SX.sym('u_rw', 3) #type: ignore
    u_mag = ca.SX.sym('u_mag', 3) #type: ignore
    u = ca.vertcat(u_rw, u_mag)

    B_B = ca.SX.sym('B_B', 3) #type: ignore

    d_q_BI = f_kin(q_BI, omega)
    d_omega = f_rot(omega, u_rw, u_mag, h_w, B_B)
    d_h_w = A_hat @ (K_t_dash * u_rw)

    dx = ca.vertcat(d_q_BI, d_omega, d_h_w)

    return ca.Function("system_dynamics", [x, u, B_B], [dx], ["x", "u", "B_B"], ["dx"])


def build_error_dynamics(omega_c: np.ndarray, J_hat: np.ndarray, A_hat: np.ndarray, K_t_dash: np.ndarray, K_mag: np.ndarray) -> tuple[ca.Function, ca.Function, ca.Function]:
    """
    Builds the complete symbolic dynamics model for the attitude error.

    This combines kinematics, rotational dynamics, and wheel dynamics into a single
    state-space function dx/dt = f(x, u, *params).

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
    f_rot: ca.Function = build_rotational_dynamics(J_hat, A_hat, K_t_dash, K_mag)

    B_B = ca.SX.sym('B_B', 3) #type: ignore

    q_err = ca.SX.sym("q_err", 3) #type: ignore
    omega_err = ca.SX.sym("omega_err", 3) #type: ignore
    h_w = ca.SX.sym("h_w", 3) #type: ignore
    x = ca.vertcat(q_err, omega_err, h_w)

    u_rw = ca.SX.sym('u_rw', 3) #type: ignore
    u_mag = ca.SX.sym('u_mag', 3) #type: ignore
    u = ca.vertcat(u_rw, u_mag)
    
    omega = omega_err + quat_rot(q_err) @ omega_c
    
    q_err[3] = 1 - ca.dot(q_err[:3], q_err[:3])

    d_omega = f_rot(omega, u_rw, u_mag, h_w, B_B)

    d_q_err = f_kin(q_err, omega_err)
    d_omega_err = d_omega + ca.cross(omega_err, quat_rot(q_err) @ omega_c)

    # TODO: maybe implement build_rw and build_mag functions to deal with saturation and so on
    d_h_w = A_hat @ (K_t_dash * u_rw)

    dx  = ca.vertcat(d_q_err, d_omega_err, d_h_w)

    f_tot = ca.Function("error_dynamics", [x, u, B_B], [dx], ["x", "u", "B_B"], ["dx"])
    f_jac_x = ca.Function("f_jac_x", [x, u, B_B], [ca.jacobian(dx, x)], ["x", "u", "B_B"], ["jac_x"])
    f_jac_u = ca.Function("f_jac_u", [x, u, B_B], [ca.jacobian(dx, u)], ["x", "u", "B_B"], ["jac_u"])

    return f_tot, f_jac_x, f_jac_u


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


def jacobian_no_seeds(f):
    """
    Wrap CasADi's Jacobian function so that seed inputs are hidden.
    Returns a callable that only requires the original inputs.
    """
    jac_f = f.jacobian()

    n_in = f.n_in()
    n_out = f.n_out()

    def wrapped(*args):
        assert len(args) == n_in

        # Evaluate outputs to infer correct seed shapes
        outs = f(*args)

        # Ensure tuple output
        if n_out == 1:
            outs = (outs,)

        # Zero seeds for each output
        seeds = [ca.DM.zeros(o.size()) for o in outs]

        return jac_f(*args, *seeds)

    return wrapped

def split_jacobian_x_u(f):
    """
    Given f(x, u), return two functions:
      - f_jac_x(x, u): Jacobian wrt x
      - f_jac_u(x, u): Jacobian wrt u
    """
    assert f.n_in() == 2, "f must have exactly two inputs (x, u)"

    jac_f = f.jacobian()

    def _zero_seeds(x_val, u_val):
        outs = f(x_val, u_val)
        if f.n_out() == 1:
            outs = (outs,)
        return [ca.DM.zeros(o.size()) for o in outs]

    def f_jac_x(x_val, u_val):
        seeds = _zero_seeds(x_val, u_val)
        jac_blocks = jac_f(x_val, u_val, *seeds)

        # Blocks are ordered: (out1_x, out1_u, out2_x, out2_u, ...)
        return tuple(jac_blocks[::2])

    def f_jac_u(x_val, u_val):
        seeds = _zero_seeds(x_val, u_val)
        jac_blocks = jac_f(x_val, u_val, *seeds)

        return tuple(jac_blocks[1::2])

    return f_jac_x, f_jac_u

class GainScheduling:

    def __init__(self, f: ca.Function, f_jac_x: ca.Function, f_jac_u: ca.Function, 
                 x_rho: Callable[[np.ndarray], np.ndarray], w_rho: Callable[[np.ndarray], np.ndarray], 
                 u_rho: Callable[[np.ndarray], np.ndarray], 
                 rho: List[np.ndarray], calc_scheduling_param: Callable[[np.ndarray], np.ndarray],
                 Q: List[np.ndarray]|np.ndarray, R: List[np.ndarray]|np.ndarray):
        
        self.operating_points = [(x_rho(p), u_rho(p), w_rho(p)) for p in rho]
        self.rho = np.stack([np.atleast_1d(p) for p in rho], axis=0)
        self.calc_scheduling_param = calc_scheduling_param

        self.f = f

        self.Q = Q if isinstance(Q, list) else [Q] * len(self.operating_points)
        self.R = R if isinstance(R, list) else [R] * len(self.operating_points)

        # for x, u, w in self.operating_points:
        #     print(f"Operating Point: x={x}, u={u}, w={w}")

        self.linear_models = [(f_jac_x(x, u), f_jac_u(x, u)) for x, u, w in self.operating_points]

        #self.place_gains = [ct.place(np.array(A).squeeze(), np.array(B).squeeze(), P[i]) for i, (A, B) in enumerate(self.linear_models)]
        self.lqr_gains = [ct.lqr(np.squeeze(A), np.squeeze(B), self.Q[i], self.R[i])[0] for i, (A, B) in enumerate(self.linear_models)] # type: ignore

    def closest_operating_points(self, beta):
        i, j = np.argsort(np.linalg.norm(self.rho - beta, axis=1))[:2]

        return i, j


    def calc_input_cmds(self, w, x, *args):
        beta = self.calc_scheduling_param(x, *args)

        i, j = self.closest_operating_points(beta)

        delta_u_i = np.dot(self.lqr_gains[i], ((w - self.operating_points[i][2]) - (x - self.operating_points[i][0])))
        delta_u_j = np.dot(self.lqr_gains[j], ((w - self.operating_points[j][2]) - (x - self.operating_points[j][0])))

        alpha = np.dot(self.rho[i] - self.rho[j], beta - self.rho[j]) / np.linalg.norm(self.rho[i] - self.rho[j])

        delta_u = delta_u_i * alpha + (1-alpha) * delta_u_j

        u_i = self.operating_points[i][1]
        
        u = delta_u + u_i

        return u

if __name__ == "__main__":

    x = ca.SX.sym("x", 2)
    dx = ca.SX.sym("dx", 2)
    u = ca.SX.sym("u")
    p = ca.SX.sym("p")

    dx = - x**3 + np.ones(2)*u
    f = ca.Function("f", [x, u], [dx], ["x", "u"], ["dx"])

    f_jac_x = ca.Function("f_jac_x", [x, u], [ca.jacobian(dx, x)], ["x", "u"], ["jac_x"])
    f_jac_u = ca.Function("f_jac_u", [x, u], [ca.jacobian(dx, u)], ["x", "u"], ["jac_u"])


    rho = np.arange(-2, 3)

    u_rho = lambda p: p # ca.Function("u_rho", [p], [p], ["rho"], ["u"])
    x_rho = lambda p: np.array((np.sign(p)*np.abs(p)**(1/3), np.sign(p)*np.abs(p)**(1/3))) # ca.Function("x_rho", [p], [ca.sign(p)*ca.fabs(p)**(1/3), ca.sign(p)*ca.fabs(p)**(1/3)], ["rho"], ["x1", "x2"])
    w_rho = lambda p : ca.sign(p)*ca.fabs(p)**(1/3) # ca.Function("w_rho", [p], [ca.sign(p)*ca.fabs(p)**(1/3)], ["rho"], ["w"])

    c = lambda x: x[0]**3
    
    gs = GainScheduling(f, f_jac_x, f_jac_u, x_rho, w_rho, u_rho, rho, c, Q, R)

    # print(gs.linear_models)
    # print(gs.lqr_gains)

    t = np.arange(0, 2.5, 0.01)
    w = 1.3 * np.ones_like(t)
    x = np.zeros((t.size, 2))
    u = np.zeros_like(t)
    for i in range(t.size-1):
        u[i] = gs.calc_input_cmds(w[i], x[i])
        
        x[i+1] = x[i] + 0.01 * np.array(f(x[i], u[i])).item(0)

    u[-1] = u[-2]
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.plot(t, w, linestyle="--")
    ax1.plot(t, x)
    ax2.plot(t, u)
    ax1.grid()
    ax2.grid()
    plt.show()