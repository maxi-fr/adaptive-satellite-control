

from typing import List
import numpy as np
from actuators import Magnetorquer, ReactionWheel
from dynamics import orbit_dynamics, attitude_dynamics


"""
The idea of these classes is that they hold the parameters and provide wrappers for the dynamics functions 
but they do not contain the state. The state is managed externally.
(TODO: maybe doesnt have to be a wrapper, they can just implement the dynamics) 
"""

class Spacecraft:

    def __init__(self, m, J_B, surfaces: list["Surface"], sensors, rws: list[ReactionWheel], magnetorquers: list[Magnetorquer]) -> None:
        self.m = m
        self.J_B = J_B
        self.sensors = sensors
        self.rws = rws
        self.mag = magnetorquers

        J_rw = sum([rws[i].inertia * rws[i].spin_axis @ rws[i].spin_axis.T 
                       for i in range(len(rws))])
        
        self.J_tilde = self.J_B - J_rw
        self.surfaces = surfaces

    @classmethod
    def from_json(cls, file_path: str):
        # TODO: implement 
        pass


    def orbit_dynamics(self, r_eci: np.ndarray, ctrl_force: np.ndarray, dist_force: np.ndarray):

        return orbit_dynamics(self.m, r_eci, ctrl_force, dist_force)

    def attitude_dynamics(self, omega: np.ndarray, h_int: np.ndarray, ctrl_torque: np.ndarray, dist_torque: np.ndarray):

        return attitude_dynamics(omega, self.J_tilde, ctrl_torque, dist_torque, h_int)


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
