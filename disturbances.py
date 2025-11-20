
import numpy as np
from dynamics import MU
from environment import solar_radiation_pressure_constant
from kinematics import orc_to_sbc, eci_to_sbc
from typing import List


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
    nadir_body = orc_to_sbc(q_BI, r_eci, v_eci) * np.array([0, 0, 1])

    gg_torque = (3 * MU) / np.linalg.norm(r_eci)**3 * np.cross(nadir_body, J_B @ nadir_body)

    return gg_torque


# rad/s earth rotates about the z axis of the eci frame with angular velocity OMEGA_E
OMEGA_E = np.array([0, 0, 0.000_072_921_158_553])

def aerodynamic_drag(r_eci: np.ndarray, v_eci: np.ndarray, q_BI: np.ndarray, surfaces: List["Surface"], rho: float) -> tuple[np.ndarray, np.ndarray]:
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


class Surface:
    """
    Represents a flat rectangular surface of a satellite for aerodynamic calculations.

    Attributes
    ----------
    center : np.ndarray
        Center of the surface in the body frame [m].
    normal : np.ndarray
        Unit normal vector of the surface.
    width : float
        Width of the surface [m].
    height : float
        Height of the surface [m].
    u : np.ndarray
        Unit vector defining the width direction in the surface plane.
    v : np.ndarray
        Unit vector defining the height direction in the surface plane.
    sigma_t : float
        Tangential momentum accommodation coefficient.
    sigma_n : float
        Normal momentum accommodation coefficient.
    S : float
        A parameter for the aerodynamic model.
    area : float
        Area of the surface (width * height) [m^2].
    rho_s : float
        Specular reflectivity coefficient for solar radiation pressure.
    rho_d : float
        Diffuse reflectivity coefficient for solar radiation pressure.
    rho_t : float
        Transmissivity coefficient for solar radiation pressure.
    rho_a : float
        Absorptivity coefficient for solar radiation pressure.

    """

    def __init__(self, center: np.ndarray, normal: np.ndarray, width: float, height: float, u: np.ndarray, v: np.ndarray,
                 sigma_t: float = 0.8, sigma_n: float = 0.8, S: float = 0.05,
                 rho_s: float = 0.83, rho_d: float = 0.0, rho_t: float = 0.0, rho_a: float = 0.17):
        """
        Initializes a Surface object.
        """
        self.center = center
        self.normal = normal / np.linalg.norm(normal)
        self.width = width
        self.height = height
        self.area = width * height
        self.u = u / np.linalg.norm(u)
        self.v = v / np.linalg.norm(v)
        self.sigma_t = sigma_t
        self.sigma_n = sigma_n
        self.S = S
        self.rho_s = rho_s
        self.rho_d = rho_d
        self.rho_t = rho_t
        self.rho_a = rho_a

    def self_occlusion(self, direction: np.ndarray, surfaces: List["Surface"]) -> bool:
        """
        Checks if this surface is occluded by any other surface from a given direction.
        Occlusion is determined/approximated by if a ray coming from a specific direction passing through 
        the geometric center of the surface has passed through other surfaces on the way.

        Parameters
        ----------
        direction : np.ndarray, shape (3,)
            The direction vector of incoming atmospheric flow.
        surfaces : List[Surface]
            A list of all surfaces on the satellite.

        Returns
        -------
        bool
            True if the surface is occluded, False otherwise.
        """
        for s in surfaces:
            if s == self:
                continue

            if s.passed_through(self.center, direction):
                return True

        return False

    def passed_through(self, point: np.ndarray, direction: np.ndarray) -> bool:
        """
        Checks if a ray defined by a point and direction has intersected this surface before passing through the point.

        Parameters
        ----------
        point : np.ndarray, shape (3,)
            The origin point of the ray.
        direction : np.ndarray, shape (3,)
            The direction vector of the ray.

        Returns
        -------
        bool
            True if the ray passes through the surface, False otherwise.
        """
        dn = np.dot(direction, self.normal)

        # ray is parallel
        if dn == 0:
            return False

        t = np.dot(self.center - point, self.normal) / dn

        if t >= 0:
            return False

        p = point + t * direction

        p_xy = np.vstack([self.u, self.v]) @ (p - self.center)

        return bool(np.all(np.abs(p_xy) <= np.array([self.width, self.height]) / 2))


def center_of_pressure(surfaces: List["Surface"]) -> np.ndarray:
    """
    Calculates the geometric center of pressure for a collection of surfaces.

    This is a simplified model that computes the area-weighted average of the
    centers of the provided surfaces.

    Parameters
    ----------
    surfaces : List[Surface]
        A list of Surface objects.

    Returns
    -------
    np.ndarray, shape (3,)
        The calculated center of pressure vector [m].
    """

    # TODO: idk need a better center of pressure formula

    A_tot = sum([s.area for s in surfaces])

    cop = sum([s.area * s.center for s in surfaces]) / A_tot

    return np.asarray(cop)
