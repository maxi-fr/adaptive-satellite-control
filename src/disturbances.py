
import numpy as np
from environment import solar_radiation_pressure_constant
from kinematics import orc_to_sbc, eci_to_sbc
from typing import List

from satellite import Surface

# Earth constants
R_EARTH = 6.378137e6  # Earth's equatorial radius in meters
G = 6.67430e-11  # universal gravitational constant
M = 5.972e24  # mass of earth
MU = G*M  # gravitational parameter

J2 = 1.08262668e-3   # J2 zonal harmonic
J3 = -2.5327e-6      # J3 zonal harmonic
J4 = -1.6196e-6      # J4 zonal harmonic

def non_spherical_gravity_forces(r_eci: np.ndarray, m: float) -> np.ndarray:
    """
    Calculates the disturbance forces due to Earth's non-spherical gravity.

    This function computes the perturbing acceleration from the J2, J3, and J4
    zonal harmonic coefficients and returns the corresponding force.

    Parameters
    ----------
    r_eci : np.ndarray, shape (3,)
        Position vector of the satellite in the ECI frame [m].
    m : float
        Mass of the satellite [kg].

    Returns
    -------
    np.ndarray, shape (3,)
        The total disturbance force vector in the ECI frame [N].
    """
    r = np.linalg.norm(r_eci)
    x, y, z = r_eci
    x_r = x / r
    y_r = y /r

    z_r = z / r
    z_r2 = z_r * z_r
    z_r3 = z_r2 * z_r
    z_r4 = z_r3 * z_r

    MU_r = MU / r**2
    RE_r = R_EARTH / r

    # J2 acceleration
    factor_J2 = -1.5 * J2 * MU_r * RE_r**2
    term_J2 = 1 - 5 * z_r2
    a_J2 = factor_J2 * np.array([term_J2 * x_r,
                                 term_J2 * y_r,
                                 (3 - 5 * z_r2) * z_r])

    # J3 acceleration
    factor_J3 = -0.5 * J3 * MU_r * RE_r**3
    term_J3_xy = 5 * (7 * z_r3 - 3 * z_r)
    a_J3 = factor_J3 * np.array([term_J3_xy * x_r,
                                 term_J3_xy * y_r,
                                 3 * (10 * z_r2 - (35/3) * z_r4 - 1)])

    # J4 acceleration
    factor_J4 = -0.625 * J4 * MU_r * RE_r**4
    term_J4_xy = 3 - 42 * z_r2 + 63 * z_r4
    a_J4 = factor_J4 * np.array([term_J4_xy * x_r,
                                 term_J4_xy * y_r,
                                 -(15 - 70 * z_r2 + 63 * z_r4) * z_r])

    a_total = a_J2 + a_J3 + a_J4
    return a_total * m


# Gravitational parameters (m^3 / s^2)
MU_SUN = 1.32712440018e20    # standard GM of Sun
MU_MOON = 4.9048695e12       # standard GM of Moon
def third_body_forces(r_eci: np.ndarray, m: float, sun_pos_eci: np.ndarray, moon_pos_eci: np.ndarray):

    a = 0
    for mu, r_third in zip([MU_SUN, MU_MOON], [sun_pos_eci, moon_pos_eci]):

        dist = r_third - r_eci
        a += mu * (dist / np.linalg.norm(dist)**3 - r_third / np.linalg.norm(r_third)**3)

    return a * m


def gravity_gradient(r_eci: np.ndarray, v_eci: np.ndarray, q_BI: np.ndarray, J_B: np.ndarray) -> np.ndarray:
    """
    Calculates the gravity gradient torque on a satellite.

    Parameters
    ----------
    r_eci : np.ndarray, shape (3,)
        Position vector in the ECI frame [m].
    v_eci : np.ndarray, shape (3,)
        Velocity vector in the ECI frame [m/s].
    q_BI : np.ndarray, shape (4,)
        Attitude quaternion [w, x, y, z] from ECI to the body frame.
    J_B : np.ndarray, shape (3, 3)
        Inertia tensor of the satellite in the body frame [kg*m^2].

    Returns
    -------
    np.ndarray, shape (3,)
        The gravity gradient torque vector in the body frame [N*m].
    """

    nadir_body_axis = orc_to_sbc(q_BI, r_eci, v_eci).apply(np.array([0, 0, 1]))

    gg_torque = (3 * MU) / np.linalg.norm(np.atleast_2d(r_eci), axis=1, keepdims=True)**3 * np.cross(nadir_body_axis, np.matvec(J_B, nadir_body_axis))

    return gg_torque.squeeze()


# rad/s earth rotates about the z axis of the eci frame with angular velocity OMEGA_E
OMEGA_E = np.array([0, 0, 0.000_072_921_158_553])

def aerodynamic_drag(r_eci: np.ndarray, v_eci: np.ndarray, q_BI: np.ndarray, surfaces: List[Surface], rho: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates the aerodynamic drag force and torque on the satellite.

    This function iterates through the satellite's surfaces to compute the total
    aerodynamic force and torque based on a simplified impact model.

    Parameters
    ----------
    v_eci : np.ndarray, shape (3,)
        Velocity of the satellite in the eci frame [m/s].
    q_BI : np.ndarray, shape (4,)
        Attitude quaternion [w, x, y, z] from ECI to the body frame.
    surfaces : List[Surface]
        A list of Surface objects representing the satellite's geometry.
    rho : float
        Atmospheric density at the satellite's position [kg/m^3].

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing the total aerodynamic force [N] and torque [N*m] vectors in the body frame.
    """

    v_atm_I = np.cross(OMEGA_E, r_eci)  # = np.array([OMEGA_E[2] * r_eci[1], OMEGA_E[2] * r_eci[0], 0])

    #TODO: rotations get recomputed many times. Speed up by handling them better
    v_rel_B = eci_to_sbc(q_BI).apply(v_atm_I + v_eci)

    v_rel_B_norm = np.linalg.norm(v_rel_B)
    v_rel_B_unit = v_rel_B / v_rel_B_norm

    F = np.zeros(3)
    tau = np.zeros(3)

    for s in surfaces:
        cos_theta_i = np.dot(v_rel_B_unit, s.normal)

        if cos_theta_i < 0:
            continue

        F -= rho * v_rel_B_norm**2 * s.area * cos_theta_i * (s.sigma_t * v_rel_B_unit +
                                                            (s.sigma_n * s.S + (2 - s.sigma_n - s.sigma_t) * cos_theta_i) * s.normal)
        tau += np.cross(s.center, F)

    return F, tau


def solar_radiation_pressure(r_eci: np.ndarray, sun_pos_eci: np.ndarray, in_shadow: bool, q_BI: np.ndarray, surfaces: List["Surface"]) -> tuple[np.ndarray, np.ndarray]:  
    if in_shadow:
        return np.zeros(3), np.zeros(3)

    dist = sun_pos_eci - r_eci # spacecraft to sun vector
    P = solar_radiation_pressure_constant(dist)

    sun_dir = eci_to_sbc(q_BI).apply(dist / np.linalg.norm(dist))

    F = np.zeros(3)
    tau = np.zeros(3)

    for s in surfaces:
        cos_theta_i = np.dot(sun_dir, s.normal)
        
        if cos_theta_i < 0:
            continue

        F -= P * s.area * cos_theta_i * ((1 - s.rho_s - s.rho_t) * sun_dir + (2 * s.rho_s * cos_theta_i + 2/3 * s.rho_d) * s.normal)
        
        tau += np.cross(s.center, F)

    return F, tau
