import csv
import datetime
import json
import os
import re
from typing import List

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation as R


class Logger:
    def __init__(self, log_file: str, header: list[str]):
        base = log_file.rsplit(".", 1)[0]
        ext = ".csv"
        candidate = base + ext

        if os.path.exists(candidate):
            for i in range(1000):
                candidate = f"{base}_{i}{ext}"
                if not os.path.exists(candidate):
                    break

        self.log_file = open(candidate, "a", buffering=8192, newline="")
        self.csv_writer = csv.writer(self.log_file)
        self.csv_writer.writerow(header)
        self.row_len = len(header)

    def log(self, row: list):
        # ochrana: nech ti to nehádže tiché bordely do CSV
        if len(row) != self.row_len:
            raise ValueError(f"Logger: expected {self.row_len} columns, got {len(row)}")

        self.csv_writer.writerow(row)

        # aby si po páde nestratil celý buffer (8192B) – voliteľné, ale praktické
        self.log_file.flush()

    def close(self):
        if not self.log_file.closed:
            self.log_file.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

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
                 rho_s: float = 0.83, rho_d: float = 0.0, rho_t: float = 0.0, rho_a: float = 0.17, name: str = "-"):
        
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
        self.name = name


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
        R_BS = np.asarray(
            dict["Orientation"]).T  # EOS often gives the orientation as rows; we want a matrix with axes in columns

        if not np.allclose(R_BS.T @ R_BS, np.eye(3), atol=1e-6):
            raise ValueError("Surface.from_eos_panel: Orientation matrix is not orthonormal after transpose.")
        if np.linalg.det(R_BS) < 0.0:
            raise ValueError(
                "Surface.from_eos_panel: Orientation matrix has det < 0 (reflection), expected proper rotation.")

        return cls(dict["Position"], dict.get("DimX", 0.1), dict.get("DimY", 0.1), R_BS)
    
    @classmethod
    def from_dict(cls, name, dict: dict):
        R_BS = np.array(dict["Rotation (Surface frame to Body)"])

        if not np.allclose(R_BS.T @ R_BS, np.eye(3), atol=1e-6):
            raise ValueError("Surface.from_eos_panel: Orientation matrix is not orthonormal after transpose.")
        if np.linalg.det(R_BS) < 0.0:
            raise ValueError(
                "Surface.from_eos_panel: Orientation matrix has det < 0 (reflection), expected proper rotation.")

        return cls(np.array(dict["Origin"]), dict.get("DimX", 0.1), dict.get("DimY", 0.1), R_BS, name=name)
    
    def to_dict(self):
        return {
            "Origin": self.pos.tolist(),
            "DimX": self.x_len,
            "DimY": self.y_len,
            "Rotation (Surface frame to Body)": self.R_BS.tolist(),
            "sigma_t": self.sigma_t,
            "sigma_n": self.sigma_n,
            "S": self.S,
            "rho_s": self.rho_s,
            "rho_d": self.rho_d,
            "rho_t": self.rho_t,
            "rho_a": self.rho_a,
        }
        

    def plot(self, ax, R_FB: R|None=None, color="cyan", alpha=0.4, normal_scale=0.05):
        """
        Plot the rectangular surface and its normal vector in a 3D matplotlib axis.

        R_FB: a rotation from the body frame to another frame

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
        if R_FB is None:
            pos = self.pos
            x_axis = self.x_axis
            y_axis = self.y_axis
            center = self.center
            normal = self.normal
        else:
            pos = R_FB.apply(self.pos)
            x_axis = R_FB.apply(self.x_axis)
            y_axis = R_FB.apply(self.y_axis)
            center = R_FB.apply(self.center)
            normal = R_FB.apply(self.normal)

        
        corners = np.array([
            pos,
            pos + x_axis,
            pos + x_axis + y_axis,
            pos + y_axis
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
            *center,
            *normal,
            length=normal_scale,
            color="red"
        )

    def corners(self) -> np.ndarray:
        p = self.pos
        return np.array([
            p,
            p + self.x_axis,
            p + self.x_axis + self.y_axis,
            p + self.y_axis
        ])


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

        p_xy = self.R_BS[:, :2].T @ (p - self.center)

        return bool(np.all(np.abs(p_xy) <= np.array([self.x_half, self.x_half])))


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

def string_to_timedelta(total_time: str) -> datetime.timedelta:
    match tuple(map(float, total_time.split(':'))):
        case (hours, minutes, seconds):
            return datetime.timedelta(hours=hours, minutes=minutes, seconds=seconds)
        case (minutes, seconds):
            return datetime.timedelta(minutes=minutes, seconds=seconds)
        case (seconds,):
            return datetime.timedelta(seconds=seconds)
    raise ValueError(f"String '{total_time}' not in format h:m:s")


class PiecewiseConstant:
    def __init__(self, fn, time_bucket_fn):
        self.fn = fn
        self.time_bucket_fn = time_bucket_fn
        self._last_bucket = None
        self._value = None

    def __call__(self, t, *args, **kwargs):
        bucket = self.time_bucket_fn(t)

        if bucket != self._last_bucket:
            self._value = self.fn(bucket, *args, **kwargs)
            self._last_bucket = bucket

        return self._value

    def reset(self):
        self._last_bucket = None
        self._value = None

def floor_time_to_minute(t: datetime.datetime) -> datetime.datetime:
    return t.replace(second=0, microsecond=0)
def floor_time_to_second(t: datetime.datetime) -> datetime.datetime:
    return t.replace(microsecond=0)


def weighted_pinv(M, W):
    Winv = np.linalg.inv(W)
    return Winv @ M.T @ np.linalg.pinv(M @ Winv @ M.T)


def cgi_allocation(M: np.ndarray, tau_cmd: np.ndarray, u_min: np.ndarray, u_max: np.ndarray, W: np.ndarray|None=None, tol=1e-9):
    """
    Control allocation using the constrained-gradient inverse (CGI) method.
    Optimizes the control input `u` to minimize `u^T W u`
    subject to `M * u = tau_cmd` and `u_min <= u <= u_max`.
    

    Parameters
    ----------
    M : np.ndarray
        Effectiveness matrix (m x n), where m is the number of controlled axes
        and n is the number of actuators.
    tau_cmd : np.ndarray
        Desired control torque vector (m x 1).
    u_min : np.ndarray
        Minimum control input for each actuator (n x 1).
    u_max : np.ndarray
        Maximum control input for each actuator (n x 1).
    W : np.ndarray, optional
        Weighting matrix for the actuators (n x n), by default identity.
    tol : float, optional
        Tolerance for constraint violation, by default 1e-9.

    Returns
    -------
    np.ndarray
        Allocated control input vector (n x 1).
    """
    
    m, n = M.shape
    u = np.zeros(n)
    free = np.ones(n, dtype=bool)
    tau_res = tau_cmd.copy()

    if W is None:
        W = np.eye(n)

    while True:
        if not np.any(free):
            break

        M_avail = M[:, free]
        W_avail = W[np.ix_(free, free)]

        P = weighted_pinv(M_avail, W_avail)
        u_hat = P @ tau_res

        violated_low = u_hat < u_min[free] - tol
        violated_high = u_hat > u_max[free] + tol
        violated = violated_low | violated_high

        if not np.any(violated):
            u[free] = u_hat
            break

        idx_free = np.where(free)[0]
        idx_sat = idx_free[violated]

        u[idx_sat] = np.clip(u_hat[violated], u_min[idx_sat], u_max[idx_sat])
        tau_res -= M[:, idx_sat] @ u[idx_sat]
        free[idx_sat] = False

    return u
