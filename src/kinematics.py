from scipy.spatial.transform import Rotation as R
import numpy as np

from astropy.time import Time
from astropy.coordinates import EarthLocation, SkyCoord
import astropy.units as u
import numpy as np
import casadi as ca


def orc_to_eci(r: np.ndarray, v: np.ndarray) -> R:
    """
    Calculates the rotation from the Orbital Reference Frame (ORC) to the Earth-Centered Inertial (ECI) frame.

    Parameters
    ----------
    r : np.ndarray, shape (3,)
        Position vector in the ECI frame.
    v : np.ndarray, shape (3,)
        Velocity vector in the ECI frame.

    Returns
    -------
    R_IO : scipy.spatial.transform.Rotation
        Rotation object representing the transformation from ORC to ECI.

    """
    o_3I = (- r / np.linalg.norm(r))
    o_2I = (np.cross(v, -o_3I) / np.linalg.norm(v))
    o_1I = np.cross(o_2I, o_3I)
    
    R_IO = R.from_matrix(np.vstack([o_1I, o_2I, o_3I]).T)
    return R_IO

def euler_ocr_to_sbc(roll_deg: float, pitch_deg: float, yaw_deg: float):
    """
    TODO
    """

    R_BO = R.from_euler('yxz', [pitch_deg, roll_deg, yaw_deg], degrees=True)

    return R_BO


def orc_to_sbc(q_BI: np.ndarray, r_eci: np.ndarray, v_eci: np.ndarray) -> R:
    """
    Calculates rotation from ORC to SBC using the attitude quaternion as well as position and velocity vectors.

    This is achieved by composing the rotation from ECI to the body frame (from the quaternion)
    with the rotation from the ORC to the ECI frame.

    Parameters
    ----------
    q_BI : np.ndarray, shape (4,)
        Attitude quaternion [qx, qy, qz, qw] for the rotation from ECI (I) to the body frame (B).
    r_eci : np.ndarray, shape (3,)
        Position vector in the ECI frame.
    v_eci : np.ndarray, shape (3,)
        Velocity vector in the ECI frame.

    Returns
    -------
    R_BO :  scipy.spatial.transform.Rotation
            Rotation object representing the transformation from ORC to SBC.
    """

    R_BO = eci_to_sbc(q_BI) * orc_to_eci(r_eci, v_eci)

    return R_BO


def eci_to_sbc(q_BI: np.ndarray) -> R:

    return R.from_quat(q_BI, scalar_first=False)

def eci_to_geodedic(pos_eci: np.ndarray) -> tuple[float, float, float]:
    """
    pos_eci [x, y, z] in meters, 
    return deg, deg, m
    
    """
    
    loc = EarthLocation.from_geocentric(*(pos_eci*u.m)).to_geodetic("WGS84") # type: ignore

    lat = loc.lat.value
    lon = loc.lon.value
    alt = loc.height.to(u.m).value # type: ignore

    return lat, lon, alt


def quaternion_product(qa: np.ndarray, qb: np.ndarray):
    q_ret = np.empty(4)

    q_ret[:3] = qb[3]*qa[:3] + qa[3]*qb[:3] + np.cross(qa[:3], qb[:3])
    q_ret[3] = qa[3]*qb[3] - np.dot(qa[:3], qb[:3])

    return q_ret

def quaternion_rotation(q, x):

    return quaternion_product(q1, quaternion_product(q2, q3))

def quaternion_kinematics(q_BI: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """
    Compute the derivative of the quaternion. Using the scalar last convention: q_BI = [qx, qy, qz, qw]
    
    Parameters
    ----------
    q_BI : np.ndarray, shape (4,)
        Current attitude quaternion [qx, qy, qz, qw].
    omega : np.ndarray, shape (3,)
        Angular velocity of the body frame with respect to the inertial frame 
        represented in the body frame [wx, wy, wz] [rad/s].

    Returns
    -------
    np.ndarray, shape (4,)
        The time derivative of the quaternion (dq/dt).
    """
    q_ret = np.empty(4)

    q_ret[:3] = 0.5 * (omega * q_BI[3] + np.cross(omega, q_BI[:3]))
    q_ret[3] = -0.5 * np.dot(omega, q_BI[:3])

    return q_ret


