import numpy as np
from scipy.spatial.transform import Rotation as R
from astropy.coordinates import SkyCoord, CartesianRepresentation, TEME
from astropy import units as u
from sgp4.api import Satrec, WGS84
from sgp4.conveniences import jday_datetime
from astropy.time import Time
import datetime

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


class SGP4(Satrec):
    """ 
    Wrapper for Satrec SGP4 implementation to handle coordinate conversions
    """

    def __init__(self, e: float, i: float, raan: float, arg_pe: float, M0: float, MM: float, t0: datetime.datetime, B_star=0.0) -> None:
        """
        Classical Keplerian orbital elements.

        Parameters
        ----------
        e : float
            Eccentricity [-]
        i : float
            Inclination [deg]
        raan : float
            Right ascension of ascending node Ω [deg]
        arg_pe : float
            Argument of perigee ω [deg]
        M0 : float
            Mean anomaly at epoch t0 [deg]
        MM : float
            Mean motion in [rev/day]
        t0 : float, optional
            Reference time [s]
        """
        super().__init__()
        epoch = t0 - datetime.datetime.fromisoformat("1949-12-31T00:00:00Z")

        epoch = epoch.days + epoch.seconds / (3600. * 24.)

        no_kozai = 2*np.pi / (24 * 60) * MM

        self.sgp4init(WGS84, "i", "69", epoch, B_star, 0.0, 0.0, e, np.deg2rad(arg_pe), np.deg2rad(i), np.deg2rad(M0), no_kozai, np.deg2rad(raan))
    

    @classmethod
    def twoline2rv(cls, tle1, tle2, earth_grav= WGS84):
        """
        initializes orbit from tle elements
        """
        return super().twoline2rv(tle1, tle2, earth_grav)
        
    
    def propagate(self, t: datetime.datetime):

        jd, fr = jday_datetime(t)
        error_code, r_TEME, v_TEME = self.sgp4(jd, fr)
        assert error_code == 0

        teme = TEME(obstime=Time(t, format="datetime", scale="utc"))
        r_ECI = SkyCoord(CartesianRepresentation(*r_TEME, unit=u.km), frame=teme, representation_type='cartesian').transform_to("gcrs").cartesian.xyz.to(u.m).value # type: ignore
        v_ECI = SkyCoord(CartesianRepresentation(*v_TEME, unit=u.km), frame=teme, representation_type='cartesian').transform_to("gcrs").cartesian.xyz.to(u.m).value # type: ignore

        return r_ECI, v_ECI  
