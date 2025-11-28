

import json
from typing import List
import numpy as np
from actuators import Magnetorquer, ReactionWheel
from dynamics import orbit_dynamics, attitude_dynamics
from sensors import SunSensor, Magnetometer, GPS, Accelerometer, Gyroscope, RW_tachometer
from scipy.spatial.transform import Rotation as R
import re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


"""
The idea of these classes is that they hold the parameters and provide wrappers for the dynamics functions 
but they do not contain the state. The state is managed externally.
(TODO: maybe doesnt have to be a wrapper, they can just implement the dynamics) 
"""

class Spacecraft:

    def __init__(self, m, J_B, surfaces: list["Surface"], rws: list[ReactionWheel], magnetorquers: list[Magnetorquer],
                 sun_sensor: SunSensor, magnetometer: Magnetometer, gps: GPS, accelerometer: Accelerometer, gyro: Gyroscope,
                 rw_speed_sensors: list[RW_tachometer]) -> None:
        self.m = m
        self.J_B = J_B

        self.sun_sensor = sun_sensor
        self.magnetometer = magnetometer
        self.gps = gps
        self.accelerometer = accelerometer
        self.gyro = gyro
        self.rw_speed_sensors = rw_speed_sensors

        self.rws = rws
        self.mag = magnetorquers

        J_rw = sum([rws[i].inertia * rws[i].spin_axis @ rws[i].spin_axis.T 
                       for i in range(len(rws))])
        
        self.J_tilde = self.J_B - J_rw
        self.surfaces = surfaces

    @classmethod
    def from_eos_file(cls, data: dict):
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

    def __init__(self, position: np.ndarray, x_len: float, y_len: float, R_BS: np.ndarray,
                 sigma_t: float = 0.8, sigma_n: float = 0.8, S: float = 0.05,
                 rho_s: float = 0.83, rho_d: float = 0.0, rho_t: float = 0.0, rho_a: float = 0.17):
        
        self.pos = position
        self.x_len = x_len
        self.y_len = y_len
        self.x_half = x_len / 2
        self.y_half = y_len / 2

        self.R_BS = R_BS

        self.normal = self.R_BS[:, 2]
        self.x_axis = x_len * self.R_BS[:, 0]
        self.y_axis = y_len * self.R_BS[:, 1]

        self.center = self.pos + self.x_axis /2 + self.y_axis / 2

        self.area = self.x_len * self.y_len

        self.sigma_t = sigma_t
        self.sigma_n = sigma_n
        self.S = S
        self.rho_s = rho_s
        self.rho_d = rho_d
        self.rho_t = rho_t
        self.rho_a = rho_a

    @classmethod
    def from_eos_panel(cls, dict: dict):
        """
        "Panel -X": {
        "$type": "Eos.Models.Satellite.CubeSatLibrary.CoverPanel, CubeSatLibrary",
        "DimX": 0.14,
        "DimY": 0.1,
        "NumSurfModelSegsX": 1,
        "NumSurfModelSegsY": 1,
        "HasHole": false,
        "HoleX": 0.0,
        "HoleY": 0.0,
        "HoleRadius": 0.0,
        "Position": "{X:-0.05, Y:0.05, Z:0.02}",
        "Orientation":  [0.00000000, 0.00000000, -1.00000000], 
                        [0.00000000, 1.00000000, 0.00000000], 
                        [-1.00000000, 0.00000000, 0.00000000]]"
        }
        """
        R_BS = np.asarray(dict["Orientation"]).T   # idk why the transpose is needed 
        

        return cls(string_to_matrix(dict["Position"]), dict.get("DimX", 0.1), dict.get("DimY", 0.1), R_BS)

    def plot(self, ax, color="cyan", alpha=0.4, normal_scale=0.05):
        """
        Plot the rectangular surface and its normal vector in a 3D matplotlib axis.

        Example:
                surface = Surface(...)
                fig = plt.figure(figsize=(8, 8))
                ax = fig.add_subplot(111, projection='3d')

                surface.plot(ax)

                ax.set_ylabel("Y axis")
                ax.set_xlabel("X axis")
                ax.set_zlabel("Z axis")

                ax.set_aspect("equal")
                ax.legend()
                fig.tight_layout()
        """

        
        corners = np.array([
            self.pos,
            self.pos + self.x_axis,
            self.pos + self.x_axis + self.y_axis,
            self.pos + self.y_axis
        ])

        corners_closed = np.vstack([corners, corners[0]])

        # Draw rectangle edges
        ax.plot(corners_closed[:, 0], corners_closed[:, 1], corners_closed[:, 2],
                color=color)

        # Draw filled surface (optional)
        ax.add_collection3d(
            Poly3DCollection([corners], alpha=alpha, facecolor=color)
        )

        # Draw normal vector
        ax.quiver(
            *self.center,
            *self.normal,
            length=normal_scale,
            color="red"
        )

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
        dn = abs(np.dot(direction, self.normal))

        # ray is parallel
        if dn == 0:
            return False

        t = np.dot(self.center - point, self.normal) / dn

        if t >= 0:
            return False

        p = point + t * direction

        p_xy = self.R_BS[:, :2] @ (p - self.center)

        return bool(np.all(np.abs(p_xy) <= np.array([self.x_half, self.x_half])))


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


def string_to_matrix(matrix_str: str) -> np.ndarray:
    """
    Parses a matrix string like '{{M11:1.0,...}}' and converts it
    into a NumPy array.
    """
    clean_str = re.sub(r"\w+:", "", matrix_str).replace("{", "[").replace("}", "]")
    
    return np.array(json.loads(clean_str))

    

def replace_orientation_matrices(data_structure):
    """
    Recursively traverses a nested dictionary or list and replaces the
    string value for all keys named "Orientation" with its matrix representation.
    """
    if isinstance(data_structure, dict):
        for key, value in data_structure.items():
            if (key == "Orientation" or key == "Position") and isinstance(value, str):
                data_structure[key] = string_to_matrix(value)
            else:
                replace_orientation_matrices(value)

    elif isinstance(data_structure, list):
        for item in data_structure:
            replace_orientation_matrices(item)
            
    return data_structure

