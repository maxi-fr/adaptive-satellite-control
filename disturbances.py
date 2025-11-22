
import numpy as np
from dynamics import MU
from environment import solar_radiation_pressure_constant
from kinematics import orc_to_sbc, eci_to_sbc
from typing import List

from satellite import Surface, center_of_pressure


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

    nadir_body_axis = orc_to_sbc(q_BI, r_eci, v_eci) * np.array([0, 0, 1])

    gg_torque = (3 * MU) / np.linalg.norm(r_eci)**3 * np.cross(nadir_body_axis, J_B @ nadir_body_axis)

    return gg_torque


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

    cop = center_of_pressure(surfaces)

    v_atm_I = np.cross(OMEGA_E, r_eci)  # = np.array([OMEGA_E[2] * r_eci[1], OMEGA_E[2] * r_eci[0], 0])
    v_rel_B = eci_to_sbc(q_BI).apply(v_atm_I - v_eci)

    v_rel_B_norm = np.linalg.norm(v_rel_B)
    v_rel_B_unit = v_rel_B / v_rel_B_norm

    F = np.zeros(3)
    tau = np.zeros(3)

    for s in surfaces:

        if s.self_occlusion(v_rel_B, surfaces):
            continue

        cos_theta_i = np.dot(v_rel_B_unit, s.normal)

        F += rho * v_rel_B_norm**2 * s.area * cos_theta_i * (s.sigma_t * v_rel_B_norm +
                                                             (s.sigma_n * s.S + (2 - s.sigma_n - s.sigma_t) * cos_theta_i) * s.normal)
        tau += np.cross(cop - s.center, F)

    return F, tau


def solar_radiation_pressure(r_eci: np.ndarray, sun_pos_eci: np.ndarray, in_shadow: bool, q_BI: np.ndarray, surfaces: List["Surface"]) -> tuple[np.ndarray, np.ndarray]:
    
    if in_shadow:
        return np.zeros(3), np.zeros(3)

    cop = center_of_pressure(surfaces)
    P = solar_radiation_pressure_constant(r_eci, sun_pos_eci)

    sun_dir = eci_to_sbc(q_BI).apply(r_eci - sun_pos_eci)
    sun_dir /= np.linalg.norm(sun_dir)

    F = np.zeros(3)
    tau = np.zeros(3)

    for s in surfaces:

        if s.self_occlusion(sun_dir, surfaces):
            continue

        sn = np.dot(sun_dir, s.normal)

        F += P * s.area * sn((1 - s.rho_s - s.rho_t) * sun_dir + (2 * s.rho_s * sn + 2/3 * s.rho_d) * s.normal)
        tau += np.cross(cop - s.center, F)

    return F, tau
