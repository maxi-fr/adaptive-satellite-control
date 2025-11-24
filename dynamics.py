import numpy as np
from scipy.spatial.transform import Rotation as R

G = 6.67430e-11  # universal gravitational constant
M = 5.972e24  # mass of earth
MU = G*M  # gravitational parameter

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


def attitude_dynamics(omega: np.ndarray, J_B: np.ndarray, ctrl_torque: np.ndarray,
                      dist_torque: np.ndarray, h_int: np.ndarray | None = None) -> np.ndarray:
    """
    Compute the spacecrafts angular acceleration (omega_dot) 
    from Euler's rotational dynamics inclduing the effects of internal angular momentum from e.g. reaction wheels.

    Parameters
    ----------  
    omega : ndarray, shape (3,)
        Angular velocity in body frame [wx, wy, wz].
    J_B : ndarray, shape (3, 3)
        Total inertia tensor of the satellite minus the contribution of the reaction wheels spin axis inertia in the body frame [kg*m^2].
    ctrl_torque : ndarray, shape (3,)
        Control torque vector in body frame.
    dist_torque : ndarray, shape (3,)
        Disturbance torque vector in body frame.
    h_int: ndarray|None, shape (3,)
        Internal angular momentum vector from reaction wheels. Default is None.

    Returns
    -------
    omega_dot : ndarray, shape (3,)
        Angular acceleration in body frame.
    """

    if h_int is None:
        h_int = np.zeros(3)

    cross_term = np.cross(omega, J_B @ omega + h_int)
    total_torque = ctrl_torque + dist_torque - cross_term
    omega_dot = np.linalg.solve(J_B, total_torque)  # TODO: faster solving by precomputing stuff because J is constant
    return omega_dot


class KeplarElements:
    def __init__(self, a: float, e: float, i: float, raan: float, arg_pe: float, M0: float, t0: float = 0.0):
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

        # TODO: take a look at TLE and to what inertial frame these elements actually transform

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
            [c_raan*c_w - s_raan*s_w*c_i,
             s_raan*c_w + c_raan*s_w*c_i,
             s_w*s_i],
            [-c_raan*s_w - s_raan*c_w*c_i,
             -s_raan*s_w + c_raan*c_w*c_i,
             c_w*s_i],
            [s_raan*s_i,
             -c_raan*s_i,
             c_i]
        ])
        return R

    def to_eci(self, t: float = 0.0) -> np.ndarray:
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
