from scipy.spatial.transform import Rotation as R
import numpy as np

from astropy.time import Time
from astropy.coordinates import GCRS, ITRS, CartesianRepresentation, EarthLocation
import astropy.units as u
import numpy as np



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
    o_3I = - r / np.linalg.norm(r)
    o_2I = np.cross(v, o_3I) / np.linalg.norm(v)
    o_1I = np.cross(o_2I, o_3I)
    R_IO = R.from_matrix(np.array([o_1I, o_2I, o_3I]))
    return R_IO

# def ocr_to_sbc(roll_deg, pitch_deg, yaw_deg):

#     R_BO = R.from_euler('yxz', [pitch_deg, roll_deg, yaw_deg], degrees=True)

#     return R_BO

def orc_to_sbc(q_BI: np.ndarray, r_eci: np.ndarray, v_eci: np.ndarray) -> R:
    """
    Calculates rotation from ORC to SBC using the attitude quaternion as well as position and velocity vectors.

    This is achieved by composing the rotation from ECI to the body frame (from the quaternion)
    with the rotation from the ORC to the ECI frame.

    Parameters
    ----------
    q_BI : np.ndarray, shape (4,)
        Attitude quaternion [w, x, y, z] for the rotation from ECI (I) to the body frame (B).
    r_eci : np.ndarray, shape (3,)
        Position vector in the ECI frame.
    v_eci : np.ndarray, shape (3,)
        Velocity vector in the ECI frame.

    Returns
    -------
    R_BO :  scipy.spatial.transform.Rotation
            Rotation object representing the transformation from ORC to SBC.
    """

    R_BO = R.from_quat(q_BI, scalar_first=True) * orc_to_eci(r_eci, v_eci)

    return R_BO



def eci_to_sbc(q_BI: np.ndarray) -> R:

    return R.from_quat(q_BI, scalar_first=True)








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