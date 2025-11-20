import numpy as np
import pymsis
import pyIGRF
import datetime
import pymap3d
import astropy.coordinates as coord
from astropy import units as u
from astropy.time import Time



def atmosphere_density_static(altitude):
    """
    Implementation from TRACE gitlab 
    this very simple static model is a first approximation. altitude in km, density in kg/m^3
    a more precise model should be implemented in a final version
    TODO: upgrade model
    """
    # the values are taken from Fundamentals of Spacecraft Attitude Determination and Control
    # by f. Markley and John Crassidis, table D.1
    const = {"p_0":[2.418e-11, 9.158e-12, 3.725e-12, 1.585e-12, 6.967e-13, 1.454e-13, 3.614e-14],
                "h_0":[300, 350, 400, 450, 500, 600, 700],
                "H":[52.5, 56.4, 59.4, 62.2, 65.8, 79, 109]}
    if altitude < 300 or altitude > 800:
        raise Exception("the altitude is outside the range for the atmospheric model.\n" + \
                        "it is:{}, but it should be between 300 and 800".format(altitude))
    else:
        # depending on the altitude a different set of constants needs to be used
        # the altitude should be bigger than h_0 but smaller than the next value of h_0
        i = [j > altitude for j in const["h_0"]].index(True) - 1
    
    return const["p_0"][i] * np.exp(-(altitude - const["h_0"][i])/const["H"][i])


def atmosphere_density_msis(dt_utc: datetime.datetime, lat_deg: float, lon_deg: float, alt_m: float, 
                       f107: float = 150, f107a: float = 150, ap: int = 4) -> float:
    """
    Calculate atmospheric density using the pymsis library.

    This function calls the MSIS model to get atmospheric density for a
    specific time and location.

    Parameters
    ----------
    dt_utc : datetime.datetime
        The UTC datetime for the density calculation.
    lat_deg : float
        Latitude in degrees.
    lon_deg : float
        Longitude in degrees.
    alt_m : float
        Altitude in meters.
    f107 : float, optional
        Daily F10.7 solar flux, by default 150.
    f107a : float, optional
        81-day average of F10.7 solar flux, by default 150.
    ap : int, optional
        The Ap geomagnetic index, by default 4.

    Returns
    -------
    float
        The calculated total mass density in kg/m^3.

    """

    datetimes = np.array([dt_utc], dtype="datetime64[s]")
    alt_km = alt_m / 1000.0

    result = pymsis.calculate(datetimes, lat_deg, lon_deg, alt_km, f107, f107a, ap)

    rho_kg_m3 = result[:, 0].item(0)

    return rho_kg_m3


def magnetic_field_vector(datetime: datetime.datetime, lat_deg: float, lon_deg: float, alt_m: float):

    D, I, H, Bn, Be, Bv, B_tot = pyIGRF.igrf_value(lat_deg, lon_deg, alt_m/1000)

    B_ecef = pymap3d.ned2ecef(Bn, Be, Bv, lat_deg, lon_deg, alt_m, pymap3d.Ellipsoid.from_name("wgs84"))

    B_eci = np.asarray(pymap3d.ecef2eci(*B_ecef, time=datetime))

    return (B_eci / np.linalg.norm(B_eci)) * B_tot

def sun_position(datetime: datetime.datetime):
    time = Time(datetime.strftime("%Y-%m-%d %H:%M:%S"), scale="utc", format="iso")
    sun = coord.get_sun(time)

    return sun.cartesian

def moon_position(datetime: datetime.datetime):
    time = Time(datetime.strftime("%Y-%m-%d %H:%M:%S"), scale="utc", format="iso")
    moon = coord.get_body("moon", time)

    return moon.cartesian
    
def solar_radiation_pressure_constant(r_eci, sun_pos_eci) -> float:

    P = 4.543142976598831e-06 * 6.68459e12 # constant at 1 AU * 6,68459e12 m/AU

    dist = np.asarray(sun_pos_eci) - np.asarray(r_eci)

    return P / float(np.linalg.norm(dist))**2


E_RADIUS = 6371 #km
def is_in_shadow(r_eci, sun_pos_eci):
    """
    returns a boolean describing wether the satellite is in the shadow 
    of the earth. uses the cylindrical shadow projection.
    from: Fundamentals of Spacecraft Attitude Determination and Control
    by f. Markley and John Crassidis Appendix D.3
    """
    sun_position_unit = sun_pos_eci / np.linalg.norm(sun_pos_eci)
    return np.dot(r_eci, sun_position_unit) < - np.sqrt(np.linalg.norm(r_eci)**2 - E_RADIUS**2)




if __name__ == "__main__":
    print(atmosphere_density_static(500))
    print(atmosphere_density_msis(datetime.datetime.now(), 40, -73, 500_000))

    magnetic_field_vector(datetime.datetime.now(), 40, -73, 500_000)

    sun_position(datetime.datetime.now())