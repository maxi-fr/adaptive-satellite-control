import numpy as np
from scipy.spatial.transform import Rotation as R
from controllers import uncontrolled
from typing import Callable

G = 6.67430e-11 # universal gravitational constant
M = 5.972e24 # mass of earth
MU = G*M # gravitational parameter

class CubeSat:
    def __init__(self, keplar_elements: KeplarElements, initial_attitude: R, initial_ang_vel_B: np.ndarray|None=None):
        """
        Initialize the Satellites dynamics parameters and state.

        Parameters
        ----------
        keplar_elements : KeplarElements
            Describing the initial orbital parameters
        initial_attitude : scipy.spatial.transform.Rotation
            Initial attitude as a Rotation object. The attitude represent the transformation from ORC to Body Frame.
        initial_ang_vel_B : np.ndarray, shape (3,), optional
            Initial angular velocity in the body frame with respect to the geocentric inertial frame (ECI) in [rad/s]. 
            If None, it is
            initialized to zeros.

        """
        self.m = 0.1
        self.J_body = np.diag([1., 1., 1.]) # moment of inertia in body frame

        self.body_shape = ... # TODO: define shape

        self.state: np.ndarray = np.zeros(13)
        self.state[:6] = keplar_elements.to_eci()

        # BI = BO * (IO)-1
        self.state[6:10] = (initial_attitude * orc_to_eci(self.state[0:3], self.state[3:6]).inv()).as_quat(scalar_first=True) # quaternion representing rotation from ECI to Body Frame
        if initial_ang_vel_B is not None:
            self.state[10:13] = initial_ang_vel_B


    def dynamics(self, state: np.ndarray, ctrl_torque: np.ndarray) -> np.ndarray:
        """
        Compute the dynamics of the satellite for use in an ODE solver.

        This function defines the differential equations for the satellite's state,
        which includes orbital and attitude motion.

        Parameters
        ----------
        u : np.ndarray, shape (3,)
            The current attitude control torques in body frame.

        Returns
        -------
        np.ndarray, shape (13,)
            The derivative of the state vector (d_state/dt).
        """
        r = state[0:3]
        v = state[3:6]
        q = state[6:10]
        omega = state[10:13]

        r_dot = v
        q_dot = quaternion_kinematics(q, omega)

        dist_torque = np.zeros(3) # TODO: implement disturbance models

        ctrl_force = np.zeros(3)
        dist_force = np.zeros(3) # TODO: implement disturbance models

        d_state = np.zeros(13)
        d_state[0:3] = r_dot
        d_state[3:6] = orbit_dynamics(self.m, r, ctrl_force, dist_force)
        d_state[6:10] = q_dot
        d_state[10:13] = attitude_dynamics(omega, self.J_body, ctrl_torque, dist_torque)

        return d_state
    
    def update(self, dt: float) -> np.ndarray:
        u = uncontrolled(self.state)

        next_state = rk4_step(self.dynamics, self.state, u, dt=dt)

        q = next_state[6:10]
        q = q / np.linalg.norm(q)
        next_state[6:10] = q

        self.state = next_state 

        return next_state


    def sun_facing_area(self) -> None:
        pass

    def  air_facing_area(self) -> None:
        pass

def orc_to_eci(r: np.ndarray, v: np.ndarray) -> R:
    """
    Calculates the rotation from the Orbital Reference Frame (ORC) to the Earth-Centered Inertial (ECI) frame.

    Parameters
    ----------
    r : np.ndarray, shape (3,)
        Position vector in the ECI frame [m].
    v : np.ndarray, shape (3,)
        Velocity vector in the ECI frame [m/s].

    Returns
    -------
    R_IO : scipy.spatial.transform.Rotation
        Rotation object representing the transformation from ORC to ECI.
   
    """
    o_3I = - r / np.linalg.norm(r)
    o_2I = np.cross(v, o_3I) / np.linalg.norm(v)
    o_1I = np.cross(o_2I, o_3I)
    R_IO = R.from_matrix(np.array([o_1I, o_2I, o_3I]))
    return R_IO
    

def quaternion_kinematics(q: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """
    Compute the derivative of the quaternion.

    Parameters
    ----------
    q : np.ndarray, shape (4,)
        Current attitude quaternion [qx, qy, qz, qw].
    omega : np.ndarray, shape (3,)
        Angular velocity in the body frame [wx, wy, wz] [rad/s].

    Returns
    -------
    np.ndarray, shape (4,)
        The time derivative of the quaternion (dq/dt).
    """
    qx, qy, qz, qw = q
    wx, wy, wz = omega
    return 0.5 * np.array([
        -qx*wx - qy*wy - qz*wz,
         qw*wx + qy*wz - qz*wy,
         qw*wy - qx*wz + qz*wx,
         qw*wz + qx*wy - qy*wx
    ]) 

def orbit_dynamics(m: float, r: np.ndarray, ctrl_force: np.ndarray, dist_force: np.ndarray) -> np.ndarray:
    """
    Compute orbital acceleration of the center of mass of the satellite according to Newtons laws of motion.


    Parameters
    ----------
    m : float
        Mass of the satellite [kg].
    r : np.ndarray, shape (3,)
        Position vector in the ECI frame [m].
    ctrl_force : np.ndarray, shape (3,)
        Control force vector in the ECI frame [N].
    dist_force : np.ndarray, shape (3,)
        Disturbance force vector in the ECI frame [N].

    Returns
    -------
    np.ndarray, shape (3,)
        Acceleration vector (d^2r/dt^2) in the ECI frame [m/s^2].
    """
    r_norm = np.linalg.norm(r)
    d_v = - (MU/r_norm**3) * r + (ctrl_force + dist_force)/m
    return d_v

def attitude_dynamics(omega: np.ndarray, J: np.ndarray, ctrl_torque: np.ndarray, dist_torque: np.ndarray) -> np.ndarray:
    """
    Compute angular acceleration (omega_dot) from Euler's rotational dynamics.
    
    Parameters
    ----------  
    omega : ndarray, shape (3,)
        Angular velocity in body frame [wx, wy, wz].
    J : ndarray, shape (3,3)
        Inertia matrix in body frame.
    ctrl_torque : ndarray, shape (3,)
        Control torque vector in body frame.
    dist_torque : ndarray, shape (3,)
        Disturbance torque vector in body frame.

    Returns
    -------
    omega_dot : ndarray, shape (3,)
        Angular acceleration in body frame.
    """
    cross_term = np.cross(omega, J @ omega)
    total_torque = ctrl_torque + dist_torque - cross_term
    omega_dot = np.linalg.solve(J, total_torque) # TODO: faster solving by precomputing stuff because J is constant
    return omega_dot 

def rk4_step(f: Callable[[np.ndarray, np.ndarray], np.ndarray], x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
    """
    Classic 4th-order Runge-Kutta integrator.

    Parameters
    ----------
    f : Callable[[np.ndarray, np.ndarray], np.ndarray]
        Function f(x, u) -> dx/dt that computes the state derivative.
    x : np.ndarray
        Current state.
    u : np.ndarray
        Current input
    dt : float
        Time step.
    """
    k1 = f(x, u)
    k2 = f(x + 0.5 * dt * k1, u)
    k3 = f(x + 0.5 * dt * k2, u)
    k4 = f(x + dt * k3, u)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


class KeplarElements:
    def __init__(self, a: float, e: float, i: float, raan: float, arg_pe: float, M0: float, t0: float=0.0):
        """
        Classical Keplerian orbital elements.

        Parameters
        ----------
        a : float
            Semi-major axis [m]
        e : float
            Eccentricity [-]
        i : float
            Inclination [rad]
        raan : float
            Right ascension of ascending node Ω [rad]
        arg_pe : float
            Argument of perigee ω [rad]
        M0 : float
            Mean anomaly at epoch t0 [rad]
        t0 : float, optional
            Reference time [s]
        """
        self.a = a
        self.e = e
        self.i = i
        self.raan = raan
        self.arg_pe = arg_pe
        self.M0 = M0
        self.t0 = t0

    def mean_motion(self) -> float:
        """
        Calculate the mean motion of the orbit.

        Returns
        -------
        float
            Mean motion (n) [rad/s].
        """
        return np.sqrt(MU / self.a**3)

    def _solve_kepler(self, M: float) -> float:
        """Solve Kepler’s equation M = E - e*sin(E) for E using Newton-Raphson."""
        E = M  # initial guess
        for _ in range(20):
            f = E - self.e * np.sin(E) - M
            if np.abs(f) < 1e-10:
                break
            f_prime = 1 - self.e * np.cos(E)
            E -= f / f_prime
        return E
    
    def _rotation_matrix(self) -> np.ndarray:
        """Return rotation matrix from perifocal to ECI using explicit formulas."""
        
        c_raan = np.cos(self.raan)
        s_raan = np.sin(self.raan)

        c_w = np.cos(self.arg_pe)
        s_w = np.sin(self.arg_pe)

        c_i = np.cos(self.i)
        s_i = np.sin(self.i)

        R = np.array([
            [ c_raan*c_w - s_raan*s_w*c_i,
              s_raan*c_w + c_raan*s_w*c_i,
              s_w*s_i ],
            [-c_raan*s_w - s_raan*c_w*c_i,
             -s_raan*s_w + c_raan*c_w*c_i,
              c_w*s_i ],
            [ s_raan*s_i,
             -c_raan*s_i,
              c_i ]
        ])
        return R

    def to_eci(self, t: float=0.0) -> np.ndarray:
        """
        Compute position and velocity in ECI frame from Keplerian elements.

        Parameters
        ----------
        t : float
            Current time [s]

        Returns
        -------
        np.ndarray, shape (6,)
            State vector [r_eci, v_eci] in ECI frame.
        """
        n = self.mean_motion()
        M = self.M0 + n * (t - self.t0)
        E = self._solve_kepler(M)

        # Orbital plane coordinates (perifocal frame)
        r = self.a * (1 - self.e * np.cos(E))
        x = self.a * (np.cos(E) - self.e)
        y = self.a * np.sqrt(1 - self.e**2) * np.sin(E)

        x_dot = - (n * self.a**2 / r) * np.sin(E)
        y_dot = (n * self.a**2 / r) * np.sqrt(1 - self.e**2) * np.cos(E)

        r_pf = np.array([x, y, 0.0])
        v_pf = np.array([x_dot, y_dot, 0.0])

        # Convert to ECI frame
        R = self._rotation_matrix()
        r_I = R @ r_pf
        v_I = R @ v_pf

        return np.concatenate((r_I, v_I))
